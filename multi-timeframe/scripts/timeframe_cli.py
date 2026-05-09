#!/usr/bin/env python3
"""
多周期/多级别分析模块 - 支持60分钟、30分钟、周线、月线等多周期共振分析
"""
import argparse
import json
import sys
import os
from datetime import datetime, timedelta

_agent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("请先安装依赖: pip install pandas numpy")
    sys.exit(1)

from data_utils import get_stock_kline


# 周期配置
PERIOD_CONFIG = {
    "60min": {"name": "60分钟", "freq": "60min", "days_per_bar": 1/4},
    "30min": {"name": "30分钟", "freq": "30min", "days_per_bar": 1/8},
    "daily": {"name": "日线", "freq": "daily", "days_per_bar": 1},
    "weekly": {"name": "周线", "freq": "weekly", "days_per_bar": 5},
    "monthly": {"name": "月线", "freq": "monthly", "days_per_bar": 21},
}


def resample_to_period(df, period):
    """将日线数据重采样为指定周期，适配中文列名"""
    if df is None or df.empty:
        return None

    df = df.copy()

    if '日期' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if '日期' not in df.columns and df.columns[0] != '日期':
                col_name = df.columns[0]
                if col_name.lower() in ('date', 'index', ''):
                    df = df.rename(columns={col_name: '日期'})
        elif df.index.name == '日期':
            df = df.reset_index()
        else:
            return None

    if '日期' not in df.columns:
        return None

    df['日期'] = pd.to_datetime(df['日期'])
    df = df.set_index('日期')

    if period == "daily":
        return df.reset_index()

    # 检查可用列
    agg_dict = {}
    col_map = {'open': '开盘', 'close': '收盘', 'high': '最高', 'low': '最低', 'volume': '成交量'}
    for eng, chn in col_map.items():
        if chn in df.columns:
            agg_dict[chn] = 'ohlcv'
        elif eng in df.columns:
            agg_dict[eng] = 'ohlcv'

    if not agg_dict:
        return df.reset_index()

    # 确定实际使用的列名
    open_col = '开盘' if '开盘' in df.columns else ('open' if 'open' in df.columns else None)
    high_col = '最高' if '最高' in df.columns else ('high' if 'high' in df.columns else None)
    low_col = '最低' if '最低' in df.columns else ('low' if 'low' in df.columns else None)
    close_col = '收盘' if '收盘' in df.columns else ('close' if 'close' in df.columns else None)
    volume_col = '成交量' if '成交量' in df.columns else ('volume' if 'volume' in df.columns else None)

    if not all([open_col, high_col, low_col, close_col]):
        return df.reset_index()

    if period == "weekly":
        agg_spec = {
            open_col: 'first',
            high_col: 'max',
            low_col: 'min',
            close_col: 'last',
        }
        if volume_col:
            agg_spec[volume_col] = 'sum'
        resampled = df.resample('W').agg(agg_spec).dropna()
    elif period == "monthly":
        agg_spec = {
            open_col: 'first',
            high_col: 'max',
            low_col: 'min',
            close_col: 'last',
        }
        if volume_col:
            agg_spec[volume_col] = 'sum'
        resampled = df.resample('ME').agg(agg_spec).dropna()
    elif period == "60min":
        resampled = df.copy()
        resampled = resampled.reset_index()
        return resampled
    elif period == "30min":
        resampled = df.copy()
        resampled = resampled.reset_index()
        return resampled
    else:
        return df.reset_index()

    return resampled.reset_index()


def calc_indicators(df):
    """计算常用技术指标，适配中文列名"""
    if df is None or df.empty:
        return df

    df = df.copy()

    close_col = '收盘' if '收盘' in df.columns else ('close' if 'close' in df.columns else None)
    if close_col is None:
        return df

    close = df[close_col].astype(float)

    for p in [5, 10, 20, 60]:
        df[f'ma_{p}'] = close.rolling(p).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd_dif'] = ema12 - ema26
    df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
    df['macd_bar'] = 2 * (df['macd_dif'] - df['macd_dea'])

    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    df['bb_mid'] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std

    df['trend'] = 0
    df.loc[close > df['ma_20'], 'trend'] = 1
    df.loc[close < df['ma_20'], 'trend'] = -1

    df['macd_signal'] = 0
    df.loc[df['macd_dif'] > df['macd_dea'], 'macd_signal'] = 1
    df.loc[df['macd_dif'] < df['macd_dea'], 'macd_signal'] = -1

    return df


def analyze_multi_timeframe(symbol, periods=None):
    """
    多周期共振分析
    分析多个时间周期的技术指标，判断是否形成共振
    """
    if periods is None:
        periods = ["daily", "weekly", "monthly"]

    # 获取日线数据
    try:
        df_daily = get_stock_kline(symbol, days=500)
    except Exception as e:
        return {"error": f"获取K线数据异常: {str(e)}"}

    if df_daily is None or df_daily.empty:
        return {"error": f"无法获取 {symbol} 的K线数据，可能今日休市或网络异常"}

    # 检查必要列
    required_cols = ['开盘', '最高', '最低', '收盘']
    missing_cols = [c for c in required_cols if c not in df_daily.columns]
    if missing_cols:
        return {"error": f"K线数据缺少必要列: {', '.join(missing_cols)}"}

    results = {
        "股票代码": symbol,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "数据日期": str(df_daily.index[-1]) if isinstance(df_daily.index, pd.DatetimeIndex) else "",
        "周期分析": {},
        "共振信号": {},
        "综合建议": ""
    }

    period_scores = []

    for period in periods:
        config = PERIOD_CONFIG.get(period, {})
        period_name = config.get("name", period)

        df_period = resample_to_period(df_daily, period)
        if df_period is None or df_period.empty:
            continue

        df_period = calc_indicators(df_period)
        latest = df_period.iloc[-1] if len(df_period) > 0 else None

        if latest is None:
            continue

        trend = "上涨" if latest.get('trend', 0) == 1 else ("下跌" if latest.get('trend', 0) == -1 else "震荡")
        ma_5 = latest.get('ma_5', 0)
        ma_20 = latest.get('ma_20', 0)
        close_price = latest.get('收盘', latest.get('close', 0))

        # MACD分析
        macd_dif = latest.get('macd_dif', 0)
        macd_dea = latest.get('macd_dea', 0)
        macd_bar = latest.get('macd_bar', 0)
        macd_status = "金叉" if macd_dif > macd_dea else "死叉"

        # RSI分析
        rsi = latest.get('rsi_14', 50)
        if rsi > 70:
            rsi_status = "超买"
        elif rsi < 30:
            rsi_status = "超卖"
        else:
            rsi_status = "中性"

        # 布林带位置
        bb_upper = latest.get('bb_upper', 0)
        bb_lower = latest.get('bb_lower', 0)
        if close_price > bb_upper:
            bb_status = "突破上轨"
        elif close_price < bb_lower:
            bb_status = "跌破下轨"
        else:
            bb_status = "轨道内"

        # 综合评分 (-3 到 +3)
        score = 0
        if trend == "上涨":
            score += 1
        elif trend == "下跌":
            score -= 1

        if macd_status == "金叉":
            score += 1
        else:
            score -= 1

        if rsi_status == "超卖":
            score += 1
        elif rsi_status == "超买":
            score -= 1

        period_analysis = {
            "周期名称": period_name,
            "最新收盘价": round(float(close_price), 2),
            "趋势": trend,
            "MA5": round(float(ma_5), 2) if pd.notna(ma_5) else None,
            "MA20": round(float(ma_20), 2) if pd.notna(ma_20) else None,
            "MACD状态": macd_status,
            "MACD_DIF": round(float(macd_dif), 4) if pd.notna(macd_dif) else None,
            "MACD_BAR": round(float(macd_bar), 4) if pd.notna(macd_bar) else None,
            "RSI(14)": round(float(rsi), 1) if pd.notna(rsi) else None,
            "RSI状态": rsi_status,
            "布林带位置": bb_status,
            "周期评分": score,
            "数据条数": len(df_period)
        }

        results["周期分析"][period] = period_analysis
        period_scores.append(score)

    # 共振分析
    if period_scores:
        total_score = sum(period_scores)
        avg_score = total_score / len(period_scores)

        bullish_count = sum(1 for s in period_scores if s > 0)
        bearish_count = sum(1 for s in period_scores if s < 0)

        if bullish_count == len(period_scores):
            resonance = "多头共振 - 所有周期一致看多"
            suggestion = "强烈看多，可考虑积极做多"
        elif bearish_count == len(period_scores):
            resonance = "空头共振 - 所有周期一致看空"
            suggestion = "强烈看空，建议观望或做空"
        elif bullish_count > bearish_count:
            resonance = f"偏多 - {bullish_count}个周期看多, {bearish_count}个周期看空"
            suggestion = "偏多格局，可谨慎做多，注意短周期回调风险"
        elif bearish_count > bullish_count:
            resonance = f"偏空 - {bearish_count}个周期看空, {bullish_count}个周期看多"
            suggestion = "偏空格局，建议减仓或观望"
        else:
            resonance = "分歧 - 多空周期数量相当"
            suggestion = "周期分歧较大，建议观望等待方向明确"

        results["共振信号"] = {
            "共振类型": resonance,
            "总评分": total_score,
            "平均评分": round(avg_score, 2),
            "看多周期数": bullish_count,
            "看空周期数": bearish_count,
            "分析周期数": len(period_scores)
        }
        results["综合建议"] = suggestion

    return results


def compare_periods(symbol, periods=None):
    """
    对比不同周期的表现
    """
    if periods is None:
        periods = ["daily", "weekly", "monthly"]

    try:
        df_daily = get_stock_kline(symbol, days=500)
    except Exception as e:
        return {"error": f"获取K线数据异常: {str(e)}"}

    if df_daily is None or df_daily.empty:
        return {"error": f"无法获取 {symbol} 的K线数据，可能今日休市或网络异常"}

    required_cols = ['开盘', '最高', '最低', '收盘']
    missing_cols = [c for c in required_cols if c not in df_daily.columns]
    if missing_cols:
        return {"error": f"K线数据缺少必要列: {', '.join(missing_cols)}"}

    comparison = {
        "股票代码": symbol,
        "数据日期": str(df_daily.index[-1]) if isinstance(df_daily.index, pd.DatetimeIndex) else "",
        "周期对比": []
    }

    for period in periods:
        config = PERIOD_CONFIG.get(period, {})
        period_name = config.get("name", period)

        df_period = resample_to_period(df_daily, period)
        if df_period is None or df_period.empty:
            continue

        df_period = calc_indicators(df_period)

        recent = df_period.tail(20)
        if len(recent) < 2:
            continue

        close = recent['收盘'].astype(float)
        returns = close.pct_change().dropna()

        if len(returns) > 0:
            total_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
            volatility = returns.std() * np.sqrt(len(returns)) * 100
            win_rate = (returns > 0).sum() / len(returns) * 100

            latest = df_period.iloc[-1]
            signal = "买入" if latest.get('macd_signal', 0) == 1 else "卖出"

            comparison["周期对比"].append({
                "周期": period_name,
                "数据条数": len(df_period),
                "近20周期收益率": round(total_return, 2),
                "年化波动率": round(volatility, 2),
                "胜率": round(win_rate, 1),
                "最新MACD信号": signal,
                "最新收盘价": round(float(latest['收盘']), 2),
                "趋势": "上涨" if latest.get('trend', 0) == 1 else ("下跌" if latest.get('trend', 0) == -1 else "震荡")
            })

    return comparison


def get_resonance_score(symbol):
    """
    获取多周期共振评分，供评分系统调用
    返回 -100 到 +100 的共振分数
    """
    result = analyze_multi_timeframe(symbol, periods=["daily", "weekly", "monthly"])
    if "error" in result:
        return {"共振评分": 0, "共振类型": "未知", "置信度": 0}

    resonance = result.get("共振信号", {})
    total_score = resonance.get("总评分", 0)
    period_count = resonance.get("分析周期数", 3)

    # 将原始评分映射到 -100 到 +100
    max_possible = period_count * 3
    normalized = (total_score / max_possible) * 100 if max_possible > 0 else 0

    resonance_type = resonance.get("共振类型", "未知")
    if "多头共振" in resonance_type:
        confidence = 90
    elif "空头共振" in resonance_type:
        confidence = 90
    elif "偏多" in resonance_type:
        confidence = 70
    elif "偏空" in resonance_type:
        confidence = 70
    else:
        confidence = 50

    return {
        "共振评分": round(normalized, 1),
        "共振类型": resonance_type,
        "置信度": confidence,
        "周期详情": result.get("周期分析", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="多周期/多级别分析")
    subparsers = parser.add_subparsers(dest="action", help="操作")

    # 共振分析
    resonance_parser = subparsers.add_parser("resonance", help="多周期共振分析")
    resonance_parser.add_argument("--symbol", required=True, help="股票代码")
    resonance_parser.add_argument("--periods", default="daily,weekly,monthly",
                                  help="分析周期，逗号分隔 (daily/weekly/monthly)")

    # 周期对比
    compare_parser = subparsers.add_parser("compare", help="周期对比分析")
    compare_parser.add_argument("--symbol", required=True, help="股票代码")
    compare_parser.add_argument("--periods", default="daily,weekly,monthly",
                                help="对比周期，逗号分隔")

    args = parser.parse_args()

    if args.action == "resonance":
        periods = [p.strip() for p in args.periods.split(",") if p.strip() in PERIOD_CONFIG]
        if not periods:
            periods = ["daily", "weekly", "monthly"]
        result = analyze_multi_timeframe(args.symbol, periods)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.action == "compare":
        periods = [p.strip() for p in args.periods.split(",") if p.strip() in PERIOD_CONFIG]
        if not periods:
            periods = ["daily", "weekly", "monthly"]
        result = compare_periods(args.symbol, periods)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
