#!/usr/bin/env python3
"""
统计套利与配对交易系统
支持协整检验、价差分析、配对交易信号生成、半衰期估计
"""
import argparse
import json
import sys
import os
import time
from datetime import datetime

_agent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

try:
    import akshare as ak
    import pandas as pd
    import numpy as np
    from scipy import stats
except ImportError:
    print("请先安装依赖: pip install akshare pandas numpy scipy")
    sys.exit(1)

from data_utils import get_stock_kline


def get_pair_kline(symbol_a, symbol_b, days=500):
    """获取配对股票K线数据"""
    df_a = get_stock_kline(symbol_a, days=days + 50)
    df_b = get_stock_kline(symbol_b, days=days + 50)
    time.sleep(0.3)

    if df_a is None or df_b is None:
        return None, None

    close_a = df_a['收盘'] if '收盘' in df_a.columns else df_a['close']
    close_b = df_b['收盘'] if '收盘' in df_b.columns else df_b['close']

    pair_df = pd.DataFrame({
        'A': close_a.values,
        'B': close_b.values,
    }, index=close_a.index[-min(len(close_a), len(close_b)):])

    return pair_df.dropna(), (df_a, df_b)


# ==================== 协整检验 ====================

def cointegration_test(symbol_a, symbol_b, days=500):
    """
    协整检验（Engle-Granger两步法）

    参数:
        symbol_a: 股票A代码
        symbol_b: 股票B代码
        days: 分析天数

    返回: {
        "是否协整": bool,
        "协整关系": {...},
        "对冲比率": float,
        "价差统计": {...},
        "交易建议": str,
    }
    """
    pair_df, raw_dfs = get_pair_kline(symbol_a, symbol_b, days=days)

    if pair_df is None or len(pair_df) < 60:
        return {"error": "数据不足，至少需要60个交易日"}

    price_a = pair_df['A'].values
    price_b = pair_df['B'].values

    # 对数价格
    log_a = np.log(price_a)
    log_b = np.log(price_b)

    # 第一步：OLS回归 log(A) = alpha + beta * log(B)
    X = np.column_stack([np.ones(len(log_b)), log_b])
    beta_hat = np.linalg.lstsq(X, log_a, rcond=None)[0]
    alpha, hedge_ratio = beta_hat[0], beta_hat[1]

    # 计算残差（价差）
    residuals = log_a - (alpha + hedge_ratio * log_b)

    # 第二步：对残差进行ADF检验
    adf_result = _adf_test(residuals)

    # 价差统计
    spread_mean = float(np.mean(residuals))
    spread_std = float(np.std(residuals))
    current_spread = float(residuals[-1])
    z_score = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0

    # 半衰期估计
    half_life = _estimate_half_life(residuals)

    # 协整判断
    is_cointegrated = adf_result["p_value"] < 0.05

    # 交易建议
    if is_cointegrated:
        if z_score > 2:
            suggestion = f"价差Z-score={z_score:.2f}，显著偏高，建议做空价差（卖出A买入B）"
            direction = "short_spread"
        elif z_score > 1:
            suggestion = f"价差Z-score={z_score:.2f}，偏高，可考虑做空价差"
            direction = "short_spread"
        elif z_score < -2:
            suggestion = f"价差Z-score={z_score:.2f}，显著偏低，建议做多价差（买入A卖出B）"
            direction = "long_spread"
        elif z_score < -1:
            suggestion = f"价差Z-score={z_score:.2f}，偏低，可考虑做多价差"
            direction = "long_spread"
        else:
            suggestion = f"价差Z-score={z_score:.2f}，处于正常范围，观望"
            direction = "neutral"
    else:
        suggestion = "两只股票不存在协整关系，不适合配对交易"
        direction = "none"

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "配对": f"{symbol_a} vs {symbol_b}",
        "协整检验": {
            "是否协整": is_cointegrated,
            "ADF统计量": adf_result["adf_stat"],
            "P值": adf_result["p_value"],
            "1%临界值": adf_result["critical_1pct"],
            "5%临界值": adf_result["critical_5pct"],
            "10%临界值": adf_result["critical_10pct"],
        },
        "对冲比率": round(hedge_ratio, 4),
        "对冲比率含义": f"1份{symbol_a}对应{hedge_ratio:.2f}份{symbol_b}",
        "价差统计": {
            "均值": round(spread_mean, 6),
            "标准差": round(spread_std, 6),
            "当前价差": round(current_spread, 6),
            "Z-score": round(z_score, 4),
            "半衰期(天)": round(half_life, 1) if half_life else "N/A",
        },
        "交易建议": suggestion,
        "交易方向": direction,
        "价差历史": {
            "日期": pair_df.index[-100:].strftime('%Y-%m-%d').tolist() if hasattr(pair_df.index, 'strftime') else list(range(100)),
            "价差": [round(float(r), 6) for r in residuals[-100:]],
            "上轨(+2σ)": [round(spread_mean + 2 * spread_std, 6)] * min(100, len(residuals)),
            "下轨(-2σ)": [round(spread_mean - 2 * spread_std, 6)] * min(100, len(residuals)),
            "均值": [round(spread_mean, 6)] * min(100, len(residuals)),
        },
    }


def _adf_test(series, max_lag=None):
    """ADF单位根检验"""
    series = np.asarray(series)
    n = len(series)

    if max_lag is None:
        max_lag = int(np.floor(12 * (n / 100) ** 0.25))

    # ADF临界值（近似）
    critical_values = {
        "1%": -3.43,
        "5%": -2.86,
        "10%": -2.57,
    }

    # 差分
    dy = np.diff(series)
    y_lag = series[:-1]

    # 构建回归矩阵
    n_obs = len(dy) - max_lag
    if n_obs <= 0:
        return {"adf_stat": 0, "p_value": 1.0, "critical_1pct": -3.43, "critical_5pct": -2.86, "critical_10pct": -2.57}

    y_lag_reg = y_lag[max_lag:]
    X = np.column_stack([y_lag_reg] + [dy[max_lag - i - 1:-i - 1] if i > 0 else dy[max_lag:] for i in range(max_lag)])
    y = dy[max_lag:]

    # OLS
    try:
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]
        residuals = y - (beta[0] + X @ beta[1:])
        se = np.sqrt(np.sum(residuals ** 2) / (len(y) - len(beta)))
        X_with_const = np.column_stack([np.ones(len(X)), X])
        cov_matrix = np.linalg.inv(X_with_const.T @ X_with_const) * se ** 2
        adf_stat = beta[1] / np.sqrt(cov_matrix[1, 1])
    except Exception:
        return {"adf_stat": 0, "p_value": 1.0, "critical_1pct": -3.43, "critical_5pct": -2.86, "critical_10pct": -2.57}

    # P值近似（基于MacKinnon响应面）
    if adf_stat < critical_values["1%"]:
        p_value = 0.001
    elif adf_stat < critical_values["5%"]:
        p_value = 0.01 + (adf_stat - critical_values["1%"]) / (critical_values["5%"] - critical_values["1%"]) * 0.04
    elif adf_stat < critical_values["10%"]:
        p_value = 0.05 + (adf_stat - critical_values["5%"]) / (critical_values["10%"] - critical_values["5%"]) * 0.05
    else:
        p_value = 0.10 + (adf_stat - critical_values["10%"]) / (0 - critical_values["10%"]) * 0.90
        p_value = min(p_value, 0.99)

    return {
        "adf_stat": round(float(adf_stat), 4),
        "p_value": round(float(p_value), 4),
        "critical_1pct": critical_values["1%"],
        "critical_5pct": critical_values["5%"],
        "critical_10pct": critical_values["10%"],
    }


def _estimate_half_life(spread):
    """估计价差均值回归半衰期"""
    spread = np.asarray(spread)
    y = np.diff(spread)
    x = spread[:-1]

    mask = ~(np.isnan(y) | np.isnan(x))
    y, x = y[mask], x[mask]

    if len(y) < 10:
        return None

    X = np.column_stack([np.ones(len(x)), x])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        theta = -beta[1]
        if theta <= 0:
            return None
        half_life = np.log(2) / theta
        return float(half_life)
    except Exception:
        return None


# ==================== 配对筛选 ====================

def find_pairs(symbols, days=250, min_correlation=0.8):
    """
    从候选股票池中筛选配对

    参数:
        symbols: 候选股票代码列表
        days: 分析天数
        min_correlation: 最小相关系数

    返回: 配对候选列表
    """
    if len(symbols) < 2:
        return {"error": "至少需要2只股票"}

    # 获取所有股票收盘价
    closes = {}
    for sym in symbols:
        df = get_stock_kline(sym, days=days + 20)
        if df is not None:
            close = df['收盘'] if '收盘' in df.columns else df['close']
            closes[sym] = close.values[-days:]
        time.sleep(0.3)

    if len(closes) < 2:
        return {"error": "有效数据不足"}

    # 对齐数据
    min_len = min(len(v) for v in closes.values())
    aligned = {}
    for sym, vals in closes.items():
        aligned[sym] = vals[-min_len:]

    df_prices = pd.DataFrame(aligned)

    # 计算相关系数矩阵
    corr_matrix = df_prices.corr()

    # 筛选高相关性配对
    pairs = []
    syms = list(aligned.keys())
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            corr = corr_matrix.iloc[i, j]
            if corr >= min_correlation:
                # 协整检验
                log_a = np.log(aligned[syms[i]])
                log_b = np.log(aligned[syms[j]])
                X = np.column_stack([np.ones(len(log_b)), log_b])
                beta = np.linalg.lstsq(X, log_a, rcond=None)[0]
                residuals = log_a - (beta[0] + beta[1] * log_b)
                adf = _adf_test(residuals)

                pairs.append({
                    "股票A": syms[i],
                    "股票B": syms[j],
                    "相关系数": round(float(corr), 4),
                    "对冲比率": round(float(beta[1]), 4),
                    "ADF统计量": adf["adf_stat"],
                    "P值": adf["p_value"],
                    "是否协整": adf["p_value"] < 0.05,
                    "价差标准差": round(float(np.std(residuals)), 6),
                })

    # 按协整质量排序
    pairs.sort(key=lambda x: (x["是否协整"], -abs(x["ADF统计量"])), reverse=True)

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "候选股票": symbols,
        "分析天数": days,
        "最小相关系数": min_correlation,
        "配对数量": len(pairs),
        "协整配对": [p for p in pairs if p["是否协整"]],
        "所有配对": pairs,
        "最佳配对": pairs[0] if pairs else None,
    }


# ==================== 配对交易信号 ====================

def pair_trading_signals(symbol_a, symbol_b, days=500, entry_z=2.0, exit_z=0.5):
    """
    配对交易信号生成

    参数:
        symbol_a: 股票A
        symbol_b: 股票B
        days: 分析天数
        entry_z: 入场Z-score阈值
        exit_z: 出场Z-score阈值

    返回: 交易信号列表
    """
    pair_df, _ = get_pair_kline(symbol_a, symbol_b, days=days)

    if pair_df is None or len(pair_df) < 60:
        return {"error": "数据不足"}

    price_a = pair_df['A'].values
    price_b = pair_df['B'].values

    # 滚动窗口协整
    window = min(120, len(price_a) // 3)
    signals = []
    position = 0

    for t in range(window, len(price_a)):
        train_a = np.log(price_a[t - window:t])
        train_b = np.log(price_b[t - window:t])

        X = np.column_stack([np.ones(len(train_b)), train_b])
        beta = np.linalg.lstsq(X, train_a, rcond=None)[0]
        hedge_ratio = beta[1]

        # 当前价差
        current_spread = np.log(price_a[t]) - (beta[0] + hedge_ratio * np.log(price_b[t]))

        # 滚动价差统计
        spread_history = train_a - (beta[0] + hedge_ratio * train_b)
        spread_mean = float(np.mean(spread_history))
        spread_std = float(np.std(spread_history))

        if spread_std < 0.0001:
            continue

        z_score = (current_spread - spread_mean) / spread_std

        signal = None
        if position == 0:
            if z_score > entry_z:
                signal = "做空价差(卖出A买入B)"
                position = -1
            elif z_score < -entry_z:
                signal = "做多价差(买入A卖出B)"
                position = 1
        elif position == 1 and z_score > -exit_z:
            signal = "平仓(卖出A买入B)"
            position = 0
        elif position == -1 and z_score < exit_z:
            signal = "平仓(买入A卖出B)"
            position = 0

        if signal:
            signals.append({
                "日期": str(pair_df.index[t])[:10],
                "信号": signal,
                "Z-score": round(z_score, 4),
                "价差": round(current_spread, 6),
                "A价格": round(float(price_a[t]), 2),
                "B价格": round(float(price_b[t]), 2),
            })

    return {
        "配对": f"{symbol_a} vs {symbol_b}",
        "参数": {"入场阈值": entry_z, "出场阈值": exit_z},
        "信号总数": len(signals),
        "交易信号": signals[-30:],
        "最近信号": signals[-1] if signals else None,
    }


# ==================== 价差分析 ====================

def spread_analysis(symbol_a, symbol_b, days=500):
    """
    价差深度分析
    包括分布特征、均值回归强度、滚动统计
    """
    pair_df, _ = get_pair_kline(symbol_a, symbol_b, days=days)

    if pair_df is None or len(pair_df) < 60:
        return {"error": "数据不足"}

    price_a = pair_df['A'].values
    price_b = pair_df['B'].values

    log_a = np.log(price_a)
    log_b = np.log(price_b)

    X = np.column_stack([np.ones(len(log_b)), log_b])
    beta = np.linalg.lstsq(X, log_a, rcond=None)[0]
    hedge_ratio = beta[1]
    spread = log_a - (beta[0] + hedge_ratio * log_b)

    # 基本统计
    spread_mean = float(np.mean(spread))
    spread_std = float(np.std(spread))
    spread_min = float(np.min(spread))
    spread_max = float(np.max(spread))
    current = float(spread[-1])
    z_score = (current - spread_mean) / spread_std if spread_std > 0 else 0

    # 分布特征
    skewness = float(pd.Series(spread).skew())
    kurtosis = float(pd.Series(spread).kurtosis())

    # 均值回归强度
    half_life = _estimate_half_life(spread)

    # 滚动统计
    window = 60
    rolling_mean = pd.Series(spread).rolling(window).mean()
    rolling_std = pd.Series(spread).rolling(window).std()

    # 极端值分析
    extreme_high = sum(1 for s in spread if s > spread_mean + 2.5 * spread_std)
    extreme_low = sum(1 for s in spread if s < spread_mean - 2.5 * spread_std)

    # 回归概率
    if abs(z_score) > 2:
        # 模拟回归时间
        if half_life:
            expected_return_days = half_life * abs(z_score) / 2
        else:
            expected_return_days = "无法估计"
    else:
        expected_return_days = "当前无需回归"

    return {
        "配对": f"{symbol_a} vs {symbol_b}",
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "对冲比率": round(hedge_ratio, 4),
        "价差统计": {
            "均值": round(spread_mean, 6),
            "标准差": round(spread_std, 6),
            "最小值": round(spread_min, 6),
            "最大值": round(spread_max, 6),
            "当前值": round(current, 6),
            "Z-score": round(z_score, 4),
        },
        "分布特征": {
            "偏度": round(skewness, 4),
            "超额峰度": round(kurtosis, 4),
            "分布评价": "价差分布近似正态" if abs(skewness) < 0.5 and abs(kurtosis) < 1 else "价差分布偏离正态",
        },
        "均值回归": {
            "半衰期(天)": round(half_life, 1) if half_life else "无法估计",
            "回归强度": "强" if half_life and half_life < 10 else ("中等" if half_life and half_life < 30 else "弱"),
        },
        "极端值": {
            "极端高值次数": int(extreme_high),
            "极端低值次数": int(extreme_low),
            "极端值比例": f"{(extreme_high + extreme_low) / len(spread) * 100:.2f}%",
        },
        "滚动统计": {
            "最近均值": round(float(rolling_mean.iloc[-1]), 6) if not pd.isna(rolling_mean.iloc[-1]) else None,
            "最近标准差": round(float(rolling_std.iloc[-1]), 6) if not pd.isna(rolling_std.iloc[-1]) else None,
        },
        "预期回归时间": str(expected_return_days),
    }


def main():
    parser = argparse.ArgumentParser(description='统计套利与配对交易系统')
    subparsers = parser.add_subparsers(dest='command')

    # 协整检验
    coint_parser = subparsers.add_parser('coint', help='协整检验')
    coint_parser.add_argument('--a', required=True, help='股票A代码')
    coint_parser.add_argument('--b', required=True, help='股票B代码')
    coint_parser.add_argument('--days', type=int, default=500, help='分析天数')

    # 配对筛选
    find_parser = subparsers.add_parser('find', help='筛选配对')
    find_parser.add_argument('--symbols', required=True, help='候选股票,逗号分隔')
    find_parser.add_argument('--days', type=int, default=250, help='分析天数')
    find_parser.add_argument('--min-corr', type=float, default=0.8, help='最小相关系数')

    # 交易信号
    signal_parser = subparsers.add_parser('signals', help='配对交易信号')
    signal_parser.add_argument('--a', required=True, help='股票A代码')
    signal_parser.add_argument('--b', required=True, help='股票B代码')
    signal_parser.add_argument('--days', type=int, default=500, help='分析天数')
    signal_parser.add_argument('--entry', type=float, default=2.0, help='入场Z-score阈值')
    signal_parser.add_argument('--exit', type=float, default=0.5, help='出场Z-score阈值')

    # 价差分析
    spread_parser = subparsers.add_parser('spread', help='价差深度分析')
    spread_parser.add_argument('--a', required=True, help='股票A代码')
    spread_parser.add_argument('--b', required=True, help='股票B代码')
    spread_parser.add_argument('--days', type=int, default=500, help='分析天数')

    args = parser.parse_args()

    if args.command == 'coint':
        result = cointegration_test(args.a, args.b, days=args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'find':
        symbols = [s.strip() for s in args.symbols.split(',')]
        result = find_pairs(symbols, days=args.days, min_correlation=args.min_corr)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'signals':
        result = pair_trading_signals(args.a, args.b, days=args.days,
                                       entry_z=args.entry, exit_z=args.exit)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'spread':
        result = spread_analysis(args.a, args.b, days=args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
