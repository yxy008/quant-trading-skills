#!/usr/bin/env python3
"""
策略信号历史胜率统计系统
对每个策略信号回溯历史，统计买入/卖出信号的准确率、平均收益、信号可靠性评分
"""
import argparse
import json
import sys
import os
import time
from datetime import datetime, timedelta

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


def calculate_ma(close, period):
    return close.rolling(window=period).mean()


def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd = 2 * (dif - dea)
    return dif, dea, macd


def calculate_bollinger(close, period=20, std_dev=2.0):
    ma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = ma + std_dev * std
    lower = ma - std_dev * std
    return ma, upper, lower


def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def generate_signals(df, strategy_id, params=None):
    """根据策略ID和参数生成交易信号"""
    if params is None:
        params = {}
    df = df.copy()
    close = df['close']

    if strategy_id == "ma_cross":
        fast = int(params.get("fast_period", 5))
        slow = int(params.get("slow_period", 20))
        ma_fast = calculate_ma(close, fast)
        ma_slow = calculate_ma(close, slow)
        df['signal'] = 0
        cross_up = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
        cross_down = (ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))
        df.loc[cross_up, 'signal'] = 1
        df.loc[cross_down, 'signal'] = -1

    elif strategy_id == "macd":
        fast = int(params.get("fast", 12))
        slow = int(params.get("slow", 26))
        signal_p = int(params.get("signal_period", 9))
        dif, dea, _ = calculate_macd(close, fast, slow, signal_p)
        df['signal'] = 0
        golden = (dif > dea) & (dif.shift(1) <= dea.shift(1))
        dead = (dif < dea) & (dif.shift(1) >= dea.shift(1))
        df.loc[golden, 'signal'] = 1
        df.loc[dead, 'signal'] = -1

    elif strategy_id == "rsi":
        period = int(params.get("period", 14))
        oversold = int(params.get("oversold", 30))
        overbought = int(params.get("overbought", 70))
        rsi = calculate_rsi(close, period)
        df['signal'] = 0
        df.loc[rsi < oversold, 'signal'] = 1
        df.loc[rsi > overbought, 'signal'] = -1

    elif strategy_id == "bollinger":
        period = int(params.get("period", 20))
        std_dev = float(params.get("std_dev", 2.0))
        _, upper, lower = calculate_bollinger(close, period, std_dev)
        df['signal'] = 0
        df.loc[close < lower, 'signal'] = 1
        df.loc[close > upper, 'signal'] = -1

    elif strategy_id == "turtle":
        entry_period = int(params.get("entry_period", 20))
        exit_period = int(params.get("exit_period", 10))
        high_n = df['high'].rolling(window=entry_period).max()
        low_n = df['low'].rolling(window=exit_period).min()
        df['signal'] = 0
        df.loc[df['high'] > high_n.shift(1), 'signal'] = 1
        df.loc[df['low'] < low_n.shift(1), 'signal'] = -1

    elif strategy_id == "volume_breakout":
        lookback = int(params.get("lookback", 20))
        vol_mult = float(params.get("volume_multiple", 1.5))
        price_th = float(params.get("price_threshold", 0.03))
        amount = df['amount']
        avg_amount = amount.rolling(window=lookback).mean()
        high_n = close.rolling(window=lookback).max()
        df['signal'] = 0
        buy_cond = (amount > avg_amount * vol_mult) & (close > high_n.shift(1) * (1 + price_th))
        df.loc[buy_cond, 'signal'] = 1
        sell_cond = close < calculate_ma(close, 10)
        df.loc[sell_cond, 'signal'] = -1

    return df


def analyze_signal_winrate(symbol, strategy_id, days=1000, hold_periods=None,
                            params=None):
    """
    分析策略信号的历史胜率

    参数:
        symbol: 股票代码
        strategy_id: 策略ID
        days: 历史数据天数
        hold_periods: 持仓周期列表 [1, 3, 5, 10, 20]
        params: 策略参数

    返回: 信号胜率分析报告
    """
    if hold_periods is None:
        hold_periods = [1, 3, 5, 10, 20]
    if params is None:
        params = {}

    df = get_stock_kline(symbol, days=days)
    if df is None or len(df) < 100:
        return {"error": f"无法获取 {symbol} 的足够历史数据"}

    signals_df = generate_signals(df, strategy_id, params)
    close = df['close']

    buy_signals = signals_df[signals_df['signal'] == 1].index
    sell_signals = signals_df[signals_df['signal'] == -1].index

    buy_analysis = _analyze_signal_performance(close, buy_signals, hold_periods, "买入")
    sell_analysis = _analyze_signal_performance(close, sell_signals, hold_periods, "卖出")

    signal_frequency = _calculate_signal_frequency(signals_df, df)

    reliability = _calculate_reliability_score(buy_analysis, sell_analysis, signal_frequency)

    return {
        "股票代码": symbol,
        "策略ID": strategy_id,
        "策略参数": params,
        "数据范围": f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}",
        "数据天数": len(df),
        "买入信号分析": buy_analysis,
        "卖出信号分析": sell_analysis,
        "信号频率": signal_frequency,
        "可靠性评分": reliability,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def _analyze_signal_performance(close, signal_indices, hold_periods, signal_type):
    """分析信号表现"""
    close_values = close.values
    date_index = close.index

    if len(signal_indices) == 0:
        return {
            "信号类型": signal_type,
            "信号总数": 0,
            "说明": "无信号产生",
        }

    result = {
        "信号类型": signal_type,
        "信号总数": len(signal_indices),
        "持仓周期分析": {},
    }

    for period in hold_periods:
        wins = 0
        total = 0
        returns_list = []
        max_gains = []
        max_losses = []

        for sig_idx in signal_indices:
            sig_pos = close.index.get_loc(sig_idx)
            if sig_pos + period >= len(close_values):
                continue

            entry_price = close_values[sig_pos]
            exit_price = close_values[sig_pos + period]
            ret = (exit_price / entry_price - 1) * 100

            if signal_type == "买入":
                is_win = ret > 0
            else:
                is_win = ret < 0

            if is_win:
                wins += 1
            total += 1
            returns_list.append(ret)

            segment = close_values[sig_pos:sig_pos + period + 1]
            peak = np.max(segment)
            trough = np.min(segment)
            max_gains.append((peak / entry_price - 1) * 100)
            max_losses.append((trough / entry_price - 1) * 100)

        if total == 0:
            continue

        win_rate = wins / total * 100
        avg_return = np.mean(returns_list)
        median_return = np.median(returns_list)
        std_return = np.std(returns_list, ddof=1) if len(returns_list) > 1 else 0
        avg_max_gain = np.mean(max_gains)
        avg_max_loss = np.mean(max_losses)

        profit_returns = [r for r in returns_list if (signal_type == "买入" and r > 0) or (signal_type == "卖出" and r < 0)]
        loss_returns = [r for r in returns_list if (signal_type == "买入" and r <= 0) or (signal_type == "卖出" and r >= 0)]

        avg_profit = np.mean(profit_returns) if profit_returns else 0
        avg_loss = np.mean(loss_returns) if loss_returns else 0
        profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0

        result["持仓周期分析"][f"{period}日"] = {
            "样本数": total,
            "胜率": round(win_rate, 1),
            "平均收益": round(float(avg_return), 2),
            "中位数收益": round(float(median_return), 2),
            "收益标准差": round(float(std_return), 2),
            "平均最大盈利": round(float(avg_max_gain), 2),
            "平均最大亏损": round(float(avg_max_loss), 2),
            "平均盈利": round(float(avg_profit), 2),
            "平均亏损": round(float(avg_loss), 2),
            "盈亏比": round(float(profit_loss_ratio), 2),
        }

    return result


def _calculate_signal_frequency(signals_df, df):
    """计算信号频率"""
    total_days = len(df)
    buy_count = (signals_df['signal'] == 1).sum()
    sell_count = (signals_df['signal'] == -1).sum()

    buy_freq_days = total_days / buy_count if buy_count > 0 else 0
    sell_freq_days = total_days / sell_count if sell_count > 0 else 0

    return {
        "总交易日": total_days,
        "买入信号数": int(buy_count),
        "卖出信号数": int(sell_count),
        "买入信号频率": f"每{round(buy_freq_days, 1)}天一次" if buy_freq_days > 0 else "无信号",
        "卖出信号频率": f"每{round(sell_freq_days, 1)}天一次" if sell_freq_days > 0 else "无信号",
        "信号密度": round((buy_count + sell_count) / total_days * 100, 2),
    }


def _calculate_reliability_score(buy_analysis, sell_analysis, signal_frequency):
    """计算信号可靠性综合评分"""
    score = 0
    details = []

    buy_5d = buy_analysis.get("持仓周期分析", {}).get("5日", {})
    if buy_5d:
        buy_winrate = buy_5d.get("胜率", 0)
        if buy_winrate >= 60:
            score += 30
            details.append(f"买入5日胜率{buy_winrate}% >= 60%，+30分")
        elif buy_winrate >= 50:
            score += 20
            details.append(f"买入5日胜率{buy_winrate}% >= 50%，+20分")
        elif buy_winrate >= 40:
            score += 10
            details.append(f"买入5日胜率{buy_winrate}% >= 40%，+10分")
        else:
            details.append(f"买入5日胜率{buy_winrate}% < 40%，+0分")

        profit_loss = buy_5d.get("盈亏比", 0)
        if profit_loss >= 2.0:
            score += 25
            details.append(f"买入盈亏比{profit_loss} >= 2.0，+25分")
        elif profit_loss >= 1.5:
            score += 15
            details.append(f"买入盈亏比{profit_loss} >= 1.5，+15分")
        elif profit_loss >= 1.0:
            score += 10
            details.append(f"买入盈亏比{profit_loss} >= 1.0，+10分")
        else:
            details.append(f"买入盈亏比{profit_loss} < 1.0，+0分")

    sell_5d = sell_analysis.get("持仓周期分析", {}).get("5日", {})
    if sell_5d:
        sell_winrate = sell_5d.get("胜率", 0)
        if sell_winrate >= 60:
            score += 20
            details.append(f"卖出5日胜率{sell_winrate}% >= 60%，+20分")
        elif sell_winrate >= 50:
            score += 12
            details.append(f"卖出5日胜率{sell_winrate}% >= 50%，+12分")
        else:
            details.append(f"卖出5日胜率{sell_winrate}% < 50%，+0分")

    buy_count = signal_frequency.get("买入信号数", 0)
    if buy_count >= 30:
        score += 15
        details.append(f"买入信号数{buy_count} >= 30，样本充足，+15分")
    elif buy_count >= 15:
        score += 10
        details.append(f"买入信号数{buy_count} >= 15，样本一般，+10分")
    elif buy_count >= 5:
        score += 5
        details.append(f"买入信号数{buy_count} >= 5，样本偏少，+5分")
    else:
        details.append(f"买入信号数{buy_count} < 5，样本不足，+0分")

    buy_10d = buy_analysis.get("持仓周期分析", {}).get("10日", {})
    buy_1d = buy_analysis.get("持仓周期分析", {}).get("1日", {})
    if buy_10d and buy_1d:
        trend_consistency = buy_10d.get("胜率", 0) - buy_1d.get("胜率", 0)
        if trend_consistency > 5:
            score += 10
            details.append(f"趋势一致性良好（10日胜率-1日胜率={trend_consistency:.1f}%），+10分")
        elif trend_consistency > 0:
            score += 5
            details.append(f"趋势一致性一般（10日胜率-1日胜率={trend_consistency:.1f}%），+5分")
        else:
            details.append(f"趋势一致性差（10日胜率-1日胜率={trend_consistency:.1f}%），+0分")

    if score >= 80:
        level = "优秀 - 信号可靠性高，可放心使用"
    elif score >= 60:
        level = "良好 - 信号较可靠，建议结合其他指标"
    elif score >= 40:
        level = "一般 - 信号可靠性中等，需谨慎使用"
    elif score >= 20:
        level = "较差 - 信号可靠性低，不建议单独使用"
    else:
        level = "极差 - 信号不可靠，建议放弃该策略"

    return {
        "综合评分": score,
        "满分": 100,
        "评级": level,
        "评分明细": details,
    }


def batch_analyze(symbol, strategy_ids=None, days=1000, hold_periods=None):
    """
    批量分析多个策略的信号胜率

    参数:
        symbol: 股票代码
        strategy_ids: 策略ID列表
        days: 历史数据天数
        hold_periods: 持仓周期列表

    返回: 批量分析报告
    """
    if strategy_ids is None:
        strategy_ids = ["ma_cross", "macd", "rsi", "bollinger", "turtle", "volume_breakout"]

    results = []
    for sid in strategy_ids:
        analysis = analyze_signal_winrate(symbol, sid, days, hold_periods)
        if "error" in analysis:
            results.append({"策略ID": sid, "状态": "分析失败", "错误": analysis["error"]})
            continue

        buy_5d = analysis.get("买入信号分析", {}).get("持仓周期分析", {}).get("5日", {})
        sell_5d = analysis.get("卖出信号分析", {}).get("持仓周期分析", {}).get("5日", {})

        results.append({
            "策略ID": sid,
            "可靠性评分": analysis["可靠性评分"]["综合评分"],
            "可靠性评级": analysis["可靠性评分"]["评级"],
            "买入5日胜率": buy_5d.get("胜率", 0),
            "买入5日盈亏比": buy_5d.get("盈亏比", 0),
            "卖出5日胜率": sell_5d.get("胜率", 0),
            "买入信号数": analysis["信号频率"]["买入信号数"],
        })

    results.sort(key=lambda x: x["可靠性评分"], reverse=True)

    return {
        "股票代码": symbol,
        "分析策略数": len(strategy_ids),
        "策略排名": results,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def main():
    parser = argparse.ArgumentParser(description="策略信号历史胜率统计系统")
    subparsers = parser.add_subparsers(dest='command')

    single_parser = subparsers.add_parser('single', help='单策略信号胜率分析')
    single_parser.add_argument('--symbol', required=True, help='股票代码')
    single_parser.add_argument('--strategy', default='ma_cross', help='策略ID')
    single_parser.add_argument('--days', type=int, default=1000, help='历史数据天数')
    single_parser.add_argument('--periods', default='1,3,5,10,20', help='持仓周期，逗号分隔')

    batch_parser = subparsers.add_parser('batch', help='批量策略信号胜率分析')
    batch_parser.add_argument('--symbol', required=True, help='股票代码')
    batch_parser.add_argument('--strategies', default='ma_cross,macd,rsi,bollinger,turtle,volume_breakout',
                              help='策略ID列表，逗号分隔')
    batch_parser.add_argument('--days', type=int, default=1000, help='历史数据天数')
    batch_parser.add_argument('--periods', default='1,3,5,10,20', help='持仓周期，逗号分隔')

    args = parser.parse_args()

    hold_periods = [int(p.strip()) for p in args.periods.split(',')] if hasattr(args, 'periods') else None

    if args.command == 'single':
        result = analyze_signal_winrate(args.symbol, args.strategy, args.days, hold_periods)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.command == 'batch':
        strategy_ids = [s.strip() for s in args.strategies.split(',')]
        result = batch_analyze(args.symbol, strategy_ids, args.days, hold_periods)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
