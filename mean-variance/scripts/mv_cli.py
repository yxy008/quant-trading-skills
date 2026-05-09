#!/usr/bin/env python3
"""
策略组合均值-方差优化系统
马科维茨有效前沿、最优风险收益比权重分配、多目标优化
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
    from scipy.optimize import minimize
except ImportError:
    print("请先安装依赖: pip install akshare pandas numpy scipy")
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


def generate_signals(df, strategy_id, params=None):
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

    elif strategy_id == "buy_hold":
        df['signal'] = 0
        df.iloc[0, df.columns.get_loc('signal')] = 1

    return df


def get_strategy_returns(symbol, strategy_id, days=500, initial_capital=100000,
                          commission_rate=0.0003, slippage=0.001):
    """获取策略的日收益率序列"""
    df = get_stock_kline(symbol, days=days)
    if df is None or len(df) < 60:
        return None

    signals_df = generate_signals(df, strategy_id)
    close = df['close']
    signals = signals_df['signal'].reindex(df.index).fillna(0)

    cash = initial_capital
    shares = 0
    position = 0
    equity_curve = []

    for i in range(len(df)):
        price = close.iloc[i]
        signal = int(signals.iloc[i])

        if signal == 1 and position == 0 and cash > 0:
            buy_price = price * (1 + slippage)
            max_shares = int(cash / buy_price / 100) * 100
            if max_shares >= 100:
                trade_amount = max_shares * buy_price
                fee = max(trade_amount * commission_rate, 5)
                total_cost = trade_amount + fee
                if total_cost <= cash:
                    cash -= total_cost
                    shares = max_shares
                    position = 1

        elif signal == -1 and position == 1 and shares > 0:
            sell_price = price * (1 - slippage)
            trade_amount = shares * sell_price
            fee = max(trade_amount * commission_rate, 5) + trade_amount * 0.001
            cash += trade_amount - fee
            shares = 0
            position = 0

        current_equity = cash + shares * price
        equity_curve.append(float(current_equity))

    if position == 1 and shares > 0:
        final_price = close.iloc[-1]
        trade_amount = shares * final_price
        fee = max(trade_amount * commission_rate, 5) + trade_amount * 0.001
        cash += trade_amount - fee

    if len(equity_curve) < 2:
        return None

    equities = np.array(equity_curve)
    daily_returns = np.diff(equities) / equities[:-1]
    return daily_returns


def mean_variance_optimization(symbol, strategy_ids=None, days=500,
                                initial_capital=100000, risk_free_rate=0.02,
                                n_portfolios=10000):
    """
    均值-方差优化

    参数:
        symbol: 股票代码
        strategy_ids: 策略ID列表
        days: 回测天数
        initial_capital: 初始资金
        risk_free_rate: 无风险利率
        n_portfolios: 随机组合数量

    返回: 优化结果
    """
    if strategy_ids is None:
        strategy_ids = ["ma_cross", "macd", "rsi", "bollinger", "buy_hold"]

    strategy_names = {
        "ma_cross": "双均线",
        "macd": "MACD",
        "rsi": "RSI",
        "bollinger": "布林带",
        "buy_hold": "买入持有",
    }

    returns_dict = {}
    valid_strategies = []

    for sid in strategy_ids:
        rets = get_strategy_returns(symbol, sid, days, initial_capital)
        if rets is not None and len(rets) > 20:
            returns_dict[sid] = rets
            valid_strategies.append(sid)

    if len(valid_strategies) < 2:
        return {"error": "有效策略数量不足，至少需要2个策略"}

    min_len = min(len(v) for v in returns_dict.values())
    aligned_returns = {}
    for sid in valid_strategies:
        aligned_returns[sid] = returns_dict[sid][-min_len:]

    returns_df = pd.DataFrame(aligned_returns)
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252

    n_assets = len(valid_strategies)

    results = np.zeros((3 + n_assets, n_portfolios))
    weights_record = []

    for i in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)

        port_return = np.sum(weights * mean_returns.values) * 100
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights))) * 100

        sharpe = (port_return - risk_free_rate * 100) / port_std if port_std > 0 else 0

        results[0, i] = port_return
        results[1, i] = port_std
        results[2, i] = sharpe
        for j in range(n_assets):
            results[3 + j, i] = weights[j]

        weights_record.append({
            "权重": {valid_strategies[j]: round(float(weights[j]), 4) for j in range(n_assets)},
            "年化收益率": round(float(port_return), 2),
            "年化波动率": round(float(port_std), 2),
            "夏普比率": round(float(sharpe), 2),
        })

    max_sharpe_idx = np.argmax(results[2])
    min_vol_idx = np.argmin(results[1])
    max_return_idx = np.argmax(results[0])

    max_sharpe_weights = {valid_strategies[j]: round(float(results[3 + j, max_sharpe_idx]), 4) for j in range(n_assets)}
    min_vol_weights = {valid_strategies[j]: round(float(results[3 + j, min_vol_idx]), 4) for j in range(n_assets)}
    max_return_weights = {valid_strategies[j]: round(float(results[3 + j, max_return_idx]), 4) for j in range(n_assets)}

    efficient_frontier = _extract_efficient_frontier(results, n_assets)

    individual_stats = {}
    for sid in valid_strategies:
        rets = aligned_returns[sid]
        ann_ret = np.mean(rets) * 252 * 100
        ann_vol = np.std(rets, ddof=1) * np.sqrt(252) * 100
        sharpe = (ann_ret - risk_free_rate * 100) / ann_vol if ann_vol > 0 else 0
        individual_stats[strategy_names.get(sid, sid)] = {
            "年化收益率": round(float(ann_ret), 2),
            "年化波动率": round(float(ann_vol), 2),
            "夏普比率": round(float(sharpe), 2),
        }

    correlation_matrix = []
    for s1 in valid_strategies:
        row = {"策略": strategy_names.get(s1, s1)}
        for s2 in valid_strategies:
            corr = returns_df[s1].corr(returns_df[s2])
            row[strategy_names.get(s2, s2)] = round(float(corr), 4)
        correlation_matrix.append(row)

    return {
        "股票代码": symbol,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "优化参数": {
            "策略数量": n_assets,
            "随机组合数": n_portfolios,
            "无风险利率": f"{risk_free_rate*100}%",
            "数据天数": days,
        },
        "各策略独立表现": individual_stats,
        "策略相关性矩阵": correlation_matrix,
        "最优组合": {
            "最大夏普比率": {
                "权重": max_sharpe_weights,
                "年化收益率": round(float(results[0, max_sharpe_idx]), 2),
                "年化波动率": round(float(results[1, max_sharpe_idx]), 2),
                "夏普比率": round(float(results[2, max_sharpe_idx]), 2),
            },
            "最小波动率": {
                "权重": min_vol_weights,
                "年化收益率": round(float(results[0, min_vol_idx]), 2),
                "年化波动率": round(float(results[1, min_vol_idx]), 2),
                "夏普比率": round(float(results[2, min_vol_idx]), 2),
            },
            "最大收益率": {
                "权重": max_return_weights,
                "年化收益率": round(float(results[0, max_return_idx]), 2),
                "年化波动率": round(float(results[1, max_return_idx]), 2),
                "夏普比率": round(float(results[2, max_return_idx]), 2),
            },
        },
        "有效前沿": efficient_frontier,
        "建议": _generate_optimization_advice(
            max_sharpe_weights, min_vol_weights, individual_stats,
            strategy_names, valid_strategies
        ),
    }


def _extract_efficient_frontier(results, n_assets):
    """提取有效前沿"""
    returns = results[0]
    vols = results[1]

    sorted_idx = np.argsort(vols)
    sorted_vols = vols[sorted_idx]
    sorted_rets = returns[sorted_idx]

    frontier_points = []
    max_ret_so_far = -float('inf')

    for i in range(len(sorted_vols)):
        if sorted_rets[i] > max_ret_so_far:
            max_ret_so_far = sorted_rets[i]
            frontier_points.append({
                "波动率": round(float(sorted_vols[i]), 2),
                "收益率": round(float(sorted_rets[i]), 2),
            })

    return frontier_points


def _generate_optimization_advice(max_sharpe_w, min_vol_w, individual_stats,
                                   strategy_names, valid_strategies):
    """生成优化建议"""
    advice = []

    max_weight_item = max(max_sharpe_w.items(), key=lambda x: x[1])
    max_weight_name = strategy_names.get(max_weight_item[0], max_weight_item[0])
    advice.append(f"最大夏普组合中，{max_weight_name}权重最高({max_weight_item[1]*100:.1f}%)，建议作为核心策略")

    low_weight_items = [(k, v) for k, v in max_sharpe_w.items() if v < 0.1]
    if low_weight_items:
        names = [strategy_names.get(k, k) for k, v in low_weight_items]
        advice.append(f"{'、'.join(names)}在最优组合中权重较低，可考虑降低配置")

    best_individual = max(individual_stats.items(), key=lambda x: x[1].get("夏普比率", 0))
    advice.append(f"独立表现最佳策略: {best_individual[0]}（夏普{best_individual[1]['夏普比率']}）")

    advice.append("建议定期（每季度/半年）重新优化权重，适应市场变化")
    advice.append("实际配置时应考虑交易成本、流动性等约束条件")

    return advice


def optimize_with_constraints(symbol, strategy_ids=None, days=500,
                               initial_capital=100000, risk_free_rate=0.02,
                               min_weight=0.05, max_weight=0.5):
    """
    带约束的均值-方差优化（使用scipy优化器）

    参数:
        symbol: 股票代码
        strategy_ids: 策略ID列表
        days: 回测天数
        initial_capital: 初始资金
        risk_free_rate: 无风险利率
        min_weight: 最小权重
        max_weight: 最大权重

    返回: 优化结果
    """
    if strategy_ids is None:
        strategy_ids = ["ma_cross", "macd", "rsi", "bollinger"]

    strategy_names = {
        "ma_cross": "双均线",
        "macd": "MACD",
        "rsi": "RSI",
        "bollinger": "布林带",
    }

    returns_dict = {}
    valid_strategies = []

    for sid in strategy_ids:
        rets = get_strategy_returns(symbol, sid, days, initial_capital)
        if rets is not None and len(rets) > 20:
            returns_dict[sid] = rets
            valid_strategies.append(sid)

    if len(valid_strategies) < 2:
        return {"error": "有效策略数量不足"}

    min_len = min(len(v) for v in returns_dict.values())
    aligned_returns = {}
    for sid in valid_strategies:
        aligned_returns[sid] = returns_dict[sid][-min_len:]

    returns_df = pd.DataFrame(aligned_returns)
    mean_returns = returns_df.mean().values * 252
    cov_matrix = returns_df.cov().values * 252
    n = len(valid_strategies)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def neg_sharpe(weights):
        port_ret = np.sum(weights * mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_ret - risk_free_rate) / port_vol if port_vol > 0 else 1e10

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((min_weight, max_weight) for _ in range(n))
    init_guess = np.array([1.0 / n] * n)

    try:
        opt_result = minimize(neg_sharpe, init_guess, method='SLSQP',
                              bounds=bounds, constraints=constraints,
                              options={'maxiter': 1000, 'ftol': 1e-12})
    except Exception as e:
        return {"error": f"优化失败: {str(e)}"}

    if not opt_result.success:
        return {"error": f"优化未收敛: {opt_result.message}"}

    optimal_weights = opt_result.x
    opt_ret = np.sum(optimal_weights * mean_returns) * 100
    opt_vol = portfolio_volatility(optimal_weights) * 100
    opt_sharpe = (opt_ret - risk_free_rate * 100) / opt_vol if opt_vol > 0 else 0

    weights_detail = {}
    for i, sid in enumerate(valid_strategies):
        weights_detail[strategy_names.get(sid, sid)] = {
            "策略ID": sid,
            "权重": round(float(optimal_weights[i]) * 100, 1),
            "贡献收益率": round(float(optimal_weights[i] * mean_returns[i] * 100), 2),
        }

    return {
        "股票代码": symbol,
        "优化方法": "带约束的均值-方差优化（SLSQP）",
        "约束条件": {
            "最小权重": f"{min_weight*100}%",
            "最大权重": f"{max_weight*100}%",
            "权重之和": "100%",
        },
        "最优权重": weights_detail,
        "组合指标": {
            "年化收益率": round(float(opt_ret), 2),
            "年化波动率": round(float(opt_vol), 2),
            "夏普比率": round(float(opt_sharpe), 2),
        },
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def main():
    parser = argparse.ArgumentParser(description="策略组合均值-方差优化系统")
    subparsers = parser.add_subparsers(dest='command')

    mv_parser = subparsers.add_parser('optimize', help='均值-方差优化（随机模拟）')
    mv_parser.add_argument('--symbol', required=True, help='股票代码')
    mv_parser.add_argument('--strategies', default='ma_cross,macd,rsi,bollinger,buy_hold',
                           help='策略ID列表，逗号分隔')
    mv_parser.add_argument('--days', type=int, default=500, help='回测天数')
    mv_parser.add_argument('--capital', type=float, default=100000, help='初始资金')
    mv_parser.add_argument('--risk-free', type=float, default=0.02, help='无风险利率')
    mv_parser.add_argument('--portfolios', type=int, default=10000, help='随机组合数')

    constrained_parser = subparsers.add_parser('constrained', help='带约束的均值-方差优化')
    constrained_parser.add_argument('--symbol', required=True, help='股票代码')
    constrained_parser.add_argument('--strategies', default='ma_cross,macd,rsi,bollinger',
                                    help='策略ID列表，逗号分隔')
    constrained_parser.add_argument('--days', type=int, default=500, help='回测天数')
    constrained_parser.add_argument('--capital', type=float, default=100000, help='初始资金')
    constrained_parser.add_argument('--risk-free', type=float, default=0.02, help='无风险利率')
    constrained_parser.add_argument('--min-weight', type=float, default=0.05, help='最小权重')
    constrained_parser.add_argument('--max-weight', type=float, default=0.5, help='最大权重')

    args = parser.parse_args()

    if args.command == 'optimize':
        strategy_ids = [s.strip() for s in args.strategies.split(',')]
        result = mean_variance_optimization(
            args.symbol, strategy_ids, args.days, args.capital,
            args.risk_free, args.portfolios
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.command == 'constrained':
        strategy_ids = [s.strip() for s in args.strategies.split(',')]
        result = optimize_with_constraints(
            args.symbol, strategy_ids, args.days, args.capital,
            args.risk_free, args.min_weight, args.max_weight
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
