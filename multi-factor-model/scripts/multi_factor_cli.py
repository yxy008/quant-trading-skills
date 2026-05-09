#!/usr/bin/env python3
"""
多因子选股模型 - 因子标准化 / 因子合成 / 因子择时 / 股票排名
量化选股的核心引擎，支持多种因子加权方式
"""
import argparse
import json
import sys
import os
import time
from datetime import datetime, timedelta

# 添加agent目录到路径以导入data_utils
_agent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

try:
    import akshare as ak
    import pandas as pd
    import numpy as np
except ImportError:
    print("请先安装依赖: pip install akshare pandas numpy")
    sys.exit(1)

from data_utils import get_stock_kline


def calc_single_factor(df, factor_name):
    """计算单个因子值"""
    close = df['收盘'] if '收盘' in df.columns else df['close']
    high = df['最高'] if '最高' in df.columns else df['high']
    low = df['最低'] if '最低' in df.columns else df['low']
    if '成交量' in df.columns:
        volume = df['成交量']
    elif 'volume' in df.columns:
        volume = df['volume']
    else:
        volume = df.get('成交额', df.get('amount', 0)) / close

    factor_map = {
        "5日动量": lambda: (close.iloc[-1] / close.iloc[-min(5, len(close))] - 1) * 100,
        "10日动量": lambda: (close.iloc[-1] / close.iloc[-min(10, len(close))] - 1) * 100,
        "20日动量": lambda: (close.iloc[-1] / close.iloc[-min(20, len(close))] - 1) * 100,
        "60日动量": lambda: (close.iloc[-1] / close.iloc[-min(60, len(close))] - 1) * 100,
        "14日RSI": lambda: _calc_rsi(close, 14),
        "20日波动率": lambda: float(close.pct_change().tail(20).std() * np.sqrt(252) * 100),
        "60日波动率": lambda: float(close.pct_change().tail(60).std() * np.sqrt(252) * 100) if len(close) >= 60 else None,
        "5日量比": lambda: float(volume.iloc[-1] / volume.tail(5).mean()) if len(volume) >= 5 else None,
        "偏离20日均线": lambda: float((close.iloc[-1] / close.rolling(20).mean().iloc[-1] - 1) * 100),
        "20日价格位置": lambda: float((close.iloc[-1] - low.rolling(20).min().iloc[-1]) / max(high.rolling(20).max().iloc[-1] - low.rolling(20).min().iloc[-1], 0.01) * 100),
        "60日价格位置": lambda: float((close.iloc[-1] - low.rolling(60).min().iloc[-1]) / max(high.rolling(60).max().iloc[-1] - low.rolling(60).min().iloc[-1], 0.01) * 100) if len(close) >= 60 else None,
        "布林带位置": lambda: _calc_bollinger_pos(close),
        "20日振幅": lambda: float(((high.rolling(20).max() - low.rolling(20).min()) / close.rolling(20).mean() * 100).iloc[-1]),
        "换手率": lambda: float(volume.iloc[-1] / volume.tail(5).mean()) if len(volume) >= 5 else None,
    }

    if factor_name in factor_map:
        try:
            return factor_map[factor_name]()
        except Exception:
            return None
    return None


def _calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None


def _calc_bollinger_pos(close):
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    if upper.iloc[-1] - lower.iloc[-1] > 0:
        return float((close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) * 100)
    return None


# ==================== 因子标准化 ====================

def normalize_zscore(factor_values):
    """Z-Score标准化 - 均值0标准差1"""
    arr = np.array([v for v in factor_values if v is not None])
    if len(arr) < 2:
        return {i: 0.0 for i in range(len(factor_values))}
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std == 0:
        return {i: 0.0 for i in range(len(factor_values))}
    result = {}
    for i, v in enumerate(factor_values):
        if v is not None:
            result[i] = float((v - mean) / std)
        else:
            result[i] = 0.0
    return result


def normalize_rank(factor_values):
    """Rank标准化 - 百分位排名(0-100)"""
    valid = [(i, v) for i, v in enumerate(factor_values) if v is not None]
    if not valid:
        return {i: 50.0 for i in range(len(factor_values))}
    sorted_valid = sorted(valid, key=lambda x: x[1])
    n = len(sorted_valid)
    result = {}
    for rank, (idx, _) in enumerate(sorted_valid):
        result[idx] = float(rank / (n - 1) * 100) if n > 1 else 50.0
    for i in range(len(factor_values)):
        if i not in result:
            result[i] = 50.0
    return result


def normalize_minmax(factor_values):
    """Min-Max标准化 - 缩放到0-100"""
    valid = [v for v in factor_values if v is not None]
    if not valid:
        return {i: 50.0 for i in range(len(factor_values))}
    min_val = min(valid)
    max_val = max(valid)
    if max_val == min_val:
        return {i: 50.0 for i in range(len(factor_values))}
    result = {}
    for i, v in enumerate(factor_values):
        if v is not None:
            result[i] = float((v - min_val) / (max_val - min_val) * 100)
        else:
            result[i] = 50.0
    return result


def winsorize(values, limits=(0.01, 0.99)):
    """去极值处理"""
    arr = np.array([v for v in values if v is not None])
    if len(arr) < 3:
        return values
    lower = np.percentile(arr, limits[0] * 100)
    upper = np.percentile(arr, limits[1] * 100)
    result = []
    for v in values:
        if v is not None:
            result.append(float(np.clip(v, lower, upper)))
        else:
            result.append(None)
    return result


# ==================== 因子合成 ====================

def synthesize_equal_weight(factor_matrix, factor_names, directions=None):
    """
    等权合成
    factor_matrix: {symbol: {factor_name: value}}
    directions: {factor_name: 1(正向) or -1(反向)}
    """
    if directions is None:
        directions = {f: 1 for f in factor_names}

    symbols = list(factor_matrix.keys())
    scores = {}

    for sym in symbols:
        total = 0.0
        count = 0
        for fname in factor_names:
            val = factor_matrix[sym].get(fname)
            if val is not None:
                total += val * directions.get(fname, 1)
                count += 1
        scores[sym] = round(total / count, 2) if count > 0 else 0.0

    return scores


def synthesize_ic_weighted(factor_matrix, factor_names, ic_values, directions=None):
    """
    IC加权合成 - 按因子IC绝对值分配权重
    ic_values: {factor_name: ic_value}
    """
    if directions is None:
        directions = {f: 1 for f in factor_names}

    abs_ic = {f: abs(ic_values.get(f, 0)) for f in factor_names}
    total_ic = sum(abs_ic.values())
    if total_ic == 0:
        return synthesize_equal_weight(factor_matrix, factor_names, directions)

    weights = {f: abs_ic[f] / total_ic for f in factor_names}
    symbols = list(factor_matrix.keys())
    scores = {}

    for sym in symbols:
        total = 0.0
        for fname in factor_names:
            val = factor_matrix[sym].get(fname)
            if val is not None:
                total += val * directions.get(fname, 1) * weights[fname]
        scores[sym] = round(total, 2)

    return scores


def synthesize_icir_weighted(factor_matrix, factor_names, ic_values, ir_values, directions=None):
    """
    IC-IR加权合成 - 综合考虑IC和IR
    权重 = |IC| * IR_normalized
    """
    if directions is None:
        directions = {f: 1 for f in factor_names}

    abs_ic = {f: abs(ic_values.get(f, 0)) for f in factor_names}
    ir_vals = {f: max(ir_values.get(f, 0), 0.01) for f in factor_names}

    combined = {f: abs_ic[f] * ir_vals[f] for f in factor_names}
    total = sum(combined.values())
    if total == 0:
        return synthesize_equal_weight(factor_matrix, factor_names, directions)

    weights = {f: combined[f] / total for f in factor_names}
    symbols = list(factor_matrix.keys())
    scores = {}

    for sym in symbols:
        total_score = 0.0
        for fname in factor_names:
            val = factor_matrix[sym].get(fname)
            if val is not None:
                total_score += val * directions.get(fname, 1) * weights[fname]
        scores[sym] = round(total_score, 2)

    return scores


# ==================== 因子择时 ====================

def factor_timing_weights(factor_names, ic_history, lookback=60):
    """
    因子择时 - 基于近期IC动态调整因子权重
    ic_history: {factor_name: [ic_sequence]}
    返回: {factor_name: dynamic_weight}
    """
    weights = {}
    for fname in factor_names:
        ic_seq = ic_history.get(fname, [])
        if len(ic_seq) < lookback:
            weights[fname] = 1.0 / len(factor_names)
            continue

        recent_ic = ic_seq[-lookback:]
        mean_ic = np.mean(recent_ic)
        std_ic = np.std(recent_ic, ddof=1)

        if std_ic > 0:
            ir_recent = mean_ic / std_ic
        else:
            ir_recent = 0

        weights[fname] = max(abs(mean_ic) * max(ir_recent, 0), 0.01)

    total = sum(weights.values())
    if total > 0:
        weights = {f: w / total for f, w in weights.items()}

    return weights


# ==================== 多因子选股 ====================

def multi_factor_select(symbols, factor_names=None, method="equal_weight",
                         ic_values=None, ir_values=None, ic_history=None,
                         normalize_method="rank", days=250, top_n=20,
                         directions=None):
    """
    多因子选股主函数
    参数:
        symbols: 股票代码列表
        factor_names: 使用的因子列表
        method: 合成方法 (equal_weight/ic_weighted/icir_weighted/timing)
        ic_values: 各因子IC值 {factor: ic}
        ir_values: 各因子IR值 {factor: ir}
        ic_history: 各因子IC历史序列 {factor: [ic_seq]}
        normalize_method: 标准化方法 (zscore/rank/minmax)
        days: 数据天数
        top_n: 选取前N只股票
        directions: 因子方向 {factor: 1(正向) or -1(反向)}
    返回:
        dict: 选股结果
    """
    if factor_names is None:
        factor_names = ["20日动量", "14日RSI", "偏离20日均线",
                         "20日价格位置", "5日量比", "20日波动率"]

    if directions is None:
        directions = {
            "20日动量": 1, "14日RSI": 1, "偏离20日均线": 1,
            "20日价格位置": 1, "5日量比": 1, "20日波动率": -1
        }

    # 计算每只股票的因子值
    raw_factors = {}
    valid_symbols = []

    for symbol in symbols:
        df = get_stock_kline(symbol, days=days)
        if df is None or len(df) < 30:
            continue

        symbol_factors = {}
        all_valid = True
        for fname in factor_names:
            val = calc_single_factor(df, fname)
            if val is None:
                all_valid = False
            symbol_factors[fname] = val

        if all_valid:
            raw_factors[symbol] = symbol_factors
            valid_symbols.append(symbol)

        time.sleep(0.05)

    if len(valid_symbols) < 5:
        return {"error": f"有效股票不足，仅{len(valid_symbols)}只，至少需要5只"}

    # 去极值处理
    for fname in factor_names:
        values = [raw_factors[s].get(fname) for s in valid_symbols]
        winsorized = winsorize(values)
        for i, sym in enumerate(valid_symbols):
            raw_factors[sym][fname] = winsorized[i]

    # 因子标准化
    normalize_funcs = {
        "zscore": normalize_zscore,
        "rank": normalize_rank,
        "minmax": normalize_minmax
    }
    norm_func = normalize_funcs.get(normalize_method, normalize_rank)

    normalized_factors = {sym: {} for sym in valid_symbols}
    for fname in factor_names:
        values = [raw_factors[sym].get(fname) for sym in valid_symbols]
        normed = norm_func(values)
        for i, sym in enumerate(valid_symbols):
            normalized_factors[sym][fname] = normed[i]

    # 因子合成
    if method == "ic_weighted" and ic_values:
        scores = synthesize_ic_weighted(normalized_factors, factor_names, ic_values, directions)
    elif method == "icir_weighted" and ic_values and ir_values:
        scores = synthesize_icir_weighted(normalized_factors, factor_names, ic_values, ir_values, directions)
    elif method == "timing" and ic_history:
        timing_w = factor_timing_weights(factor_names, ic_history)
        scores = {}
        for sym in valid_symbols:
            total = 0.0
            for fname in factor_names:
                val = normalized_factors[sym].get(fname, 0)
                total += val * directions.get(fname, 1) * timing_w.get(fname, 0)
            scores[sym] = round(total, 2)
    else:
        scores = synthesize_equal_weight(normalized_factors, factor_names, directions)

    # 排名
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_stocks = ranked[:top_n]

    # 构建结果
    result_stocks = []
    for rank_idx, (sym, score) in enumerate(top_stocks):
        stock_info = {
            "排名": rank_idx + 1,
            "代码": sym,
            "综合得分": score,
            "因子明细": {f: round(normalized_factors[sym].get(f, 0), 2) for f in factor_names}
        }
        result_stocks.append(stock_info)

    # 因子权重信息
    factor_weights = {}
    if method == "ic_weighted" and ic_values:
        abs_ic = {f: abs(ic_values.get(f, 0)) for f in factor_names}
        total_ic = sum(abs_ic.values())
        factor_weights = {f: round(abs_ic[f] / total_ic * 100, 1) if total_ic > 0 else 0 for f in factor_names}
    elif method == "timing" and ic_history:
        timing_w = factor_timing_weights(factor_names, ic_history)
        factor_weights = {f: round(w * 100, 1) for f, w in timing_w.items()}
    else:
        factor_weights = {f: round(100.0 / len(factor_names), 1) for f in factor_names}

    return {
        "选股时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "候选股票数": len(valid_symbols),
        "使用因子": factor_names,
        "合成方法": method,
        "标准化方法": normalize_method,
        "因子权重": factor_weights,
        "因子方向": directions,
        "选股结果": result_stocks,
        "全部排名": [{"代码": sym, "得分": score} for sym, score in ranked]
    }


# ==================== 因子暴露分析 ====================

def factor_exposure_analysis(symbols, factor_names=None, days=250):
    """
    因子暴露分析 - 分析组合在各因子上的暴露情况
    """
    if factor_names is None:
        factor_names = ["20日动量", "14日RSI", "偏离20日均线",
                         "20日价格位置", "5日量比", "20日波动率"]

    exposures = {}
    for symbol in symbols:
        df = get_stock_kline(symbol, days=days)
        if df is None or len(df) < 30:
            continue

        symbol_exposures = {}
        for fname in factor_names:
            val = calc_single_factor(df, fname)
            symbol_exposures[fname] = round(val, 2) if val is not None else None

        exposures[symbol] = symbol_exposures
        time.sleep(0.2)

    if not exposures:
        return {"error": "无法获取任何股票的因子暴露数据"}

    # 计算平均暴露
    avg_exposures = {}
    for fname in factor_names:
        valid_vals = [exposures[s].get(fname) for s in exposures if exposures[s].get(fname) is not None]
        avg_exposures[fname] = round(float(np.mean(valid_vals)), 2) if valid_vals else None

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "分析股票数": len(exposures),
        "个股暴露": exposures,
        "平均暴露": avg_exposures
    }


def main():
    parser = argparse.ArgumentParser(description='多因子选股模型')
    parser.add_argument('action', choices=['select', 'exposure'],
                        help='操作: select(多因子选股), exposure(因子暴露分析)')
    parser.add_argument('--symbols', required=True, help='股票代码列表,逗号分隔')
    parser.add_argument('--factors', default=None, help='因子列表,逗号分隔')
    parser.add_argument('--method', default='equal_weight',
                        choices=['equal_weight', 'ic_weighted', 'icir_weighted', 'timing'],
                        help='合成方法')
    parser.add_argument('--normalize', default='rank',
                        choices=['zscore', 'rank', 'minmax'],
                        help='标准化方法')
    parser.add_argument('--days', type=int, default=250, help='数据天数')
    parser.add_argument('--top', type=int, default=20, help='选取前N只')

    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    factor_names = [f.strip() for f in args.factors.split(",")] if args.factors else None

    try:
        if args.action == 'select':
            data = multi_factor_select(
                symbols, factor_names=factor_names,
                method=args.method, normalize_method=args.normalize,
                days=args.days, top_n=args.top
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'exposure':
            data = factor_exposure_analysis(symbols, factor_names=factor_names, days=args.days)
            print(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
