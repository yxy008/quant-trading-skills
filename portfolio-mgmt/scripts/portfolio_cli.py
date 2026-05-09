#!/usr/bin/env python3
"""
策略组合管理模块 - 多策略资金分配、组合优化、组合回测
"""
import argparse
import json
import sys
import os
from datetime import datetime

_agent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

from data_utils import get_stock_kline


def calc_returns(df):
    """计算日收益率序列"""
    df = df.copy()
    df['return'] = df['close'].pct_change()
    return df.dropna()


def create_portfolio(name, strategies, initial_capital=1000000):
    """
    创建策略组合
    strategies: [{"name": "策略1", "symbol": "600519", "weight": 0.3, "params": {...}}, ...]
    """
    total_weight = sum(s.get("weight", 0) for s in strategies)
    if abs(total_weight - 1.0) > 0.01:
        return {"error": f"策略权重之和必须为1.0，当前为{total_weight:.2f}"}

    portfolio = {
        "组合名称": name,
        "初始资金": initial_capital,
        "创建时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "策略列表": [],
        "权重分配": {},
    }

    for s in strategies:
        strategy_entry = {
            "策略名称": s.get("name", ""),
            "股票代码": s.get("symbol", ""),
            "权重": s.get("weight", 0),
            "分配资金": initial_capital * s.get("weight", 0),
            "参数": s.get("params", {}),
        }
        portfolio["策略列表"].append(strategy_entry)
        portfolio["权重分配"][s.get("name", "")] = s.get("weight", 0)

    return portfolio


def allocate_capital(method, strategies_data, total_capital=1000000, risk_free_rate=0.03):
    """
    资金分配方法
    method: "equal" | "risk_parity" | "kelly" | "volatility_weighted"
    strategies_data: [{"name": "策略1", "returns": [...], "sharpe": 1.5, "win_rate": 0.6}, ...]
    """
    if not strategies_data:
        return {"error": "请提供策略数据"}

    n = len(strategies_data)
    weights = []

    if method == "equal":
        weights = [1.0 / n] * n

    elif method == "volatility_weighted":
        vols = []
        for s in strategies_data:
            returns = s.get("returns", [])
            if len(returns) > 1:
                vol = np.std(returns) * np.sqrt(252)
            else:
                vol = 0.2
            vols.append(max(vol, 0.01))
        inv_vols = [1.0 / v for v in vols]
        total_inv = sum(inv_vols)
        weights = [iv / total_inv for iv in inv_vols]

    elif method == "risk_parity":
        vols = []
        for s in strategies_data:
            returns = s.get("returns", [])
            if len(returns) > 1:
                vol = np.std(returns) * np.sqrt(252)
            else:
                vol = 0.2
            vols.append(max(vol, 0.01))
        inv_vols = [1.0 / v for v in vols]
        total_inv = sum(inv_vols)
        weights = [iv / total_inv for iv in inv_vols]

    elif method == "kelly":
        kelly_weights = []
        for s in strategies_data:
            win_rate = s.get("win_rate", 0.5)
            avg_win = s.get("avg_win", 0.02)
            avg_loss = abs(s.get("avg_loss", 0.01))
            if avg_loss > 0:
                kelly_f = win_rate - (1 - win_rate) / (avg_win / avg_loss)
                kelly_f = max(0, min(kelly_f, 0.5))
            else:
                kelly_f = 0
            kelly_weights.append(kelly_f)

        total_kelly = sum(kelly_weights)
        if total_kelly > 0:
            weights = [k / total_kelly for k in kelly_weights]
        else:
            weights = [1.0 / n] * n

    else:
        weights = [1.0 / n] * n

    allocation = []
    for i, s in enumerate(strategies_data):
        allocation.append({
            "策略名称": s.get("name", f"策略{i+1}"),
            "权重": round(weights[i], 4),
            "分配资金": round(total_capital * weights[i], 2),
            "权重百分比": round(weights[i] * 100, 1),
        })

    return {
        "分配方法": method,
        "总资金": total_capital,
        "分配结果": allocation,
        "权重之和": round(sum(weights), 4),
    }


def backtest_portfolio(strategies_data, initial_capital=1000000):
    """
    组合回测 - 将多个策略的收益率按权重合并
    strategies_data: [{"name": "策略1", "returns": [0.01, -0.005, ...], "weight": 0.3}, ...]
    """
    if not strategies_data:
        return {"error": "请提供策略数据"}

    # 找到最长的收益率序列
    max_len = max(len(s.get("returns", [])) for s in strategies_data)

    # 对齐收益率序列
    aligned_returns = []
    for s in strategies_data:
        returns = s.get("returns", [])
        if len(returns) < max_len:
            returns = [0] * (max_len - len(returns)) + returns
        aligned_returns.append(returns)

    # 组合日收益率
    weights = [s.get("weight", 1.0 / len(strategies_data)) for s in strategies_data]
    portfolio_returns = []
    for i in range(max_len):
        daily_ret = sum(aligned_returns[j][i] * weights[j] for j in range(len(strategies_data)))
        portfolio_returns.append(daily_ret)

    # 计算组合绩效
    cumulative = [1.0]
    for r in portfolio_returns:
        cumulative.append(cumulative[-1] * (1 + r))

    total_return = (cumulative[-1] - 1) * 100
    annual_return = ((1 + total_return / 100) ** (252 / max_len) - 1) * 100 if max_len > 0 else 0
    daily_vol = np.std(portfolio_returns) if len(portfolio_returns) > 1 else 0
    annual_vol = daily_vol * np.sqrt(252) * 100

    # 最大回撤
    peak = cumulative[0]
    max_dd = 0
    for v in cumulative:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    # 夏普比率
    sharpe = (annual_return / 100 - 0.03) / (annual_vol / 100) if annual_vol > 0 else 0

    # 各策略贡献
    strategy_contributions = []
    for i, s in enumerate(strategies_data):
        s_returns = s.get("returns", [])
        if len(s_returns) > 1:
            s_total = (np.prod([1 + r for r in s_returns]) - 1) * 100
            s_vol = np.std(s_returns) * np.sqrt(252) * 100
        else:
            s_total = 0
            s_vol = 0

        strategy_contributions.append({
            "策略名称": s.get("name", f"策略{i+1}"),
            "权重": round(weights[i] * 100, 1),
            "收益率": round(s_total, 2),
            "波动率": round(s_vol, 2),
            "贡献收益": round(s_total * weights[i], 2),
        })

    return {
        "初始资金": initial_capital,
        "最终价值": round(initial_capital * cumulative[-1], 2),
        "总收益率": round(total_return, 2),
        "年化收益率": round(annual_return, 2),
        "年化波动率": round(annual_vol, 2),
        "夏普比率": round(sharpe, 2),
        "最大回撤": round(max_dd * 100, 2),
        "回测天数": max_len,
        "策略贡献": strategy_contributions,
        "组合权益": [round(v * initial_capital, 2) for v in cumulative],
    }


def optimize_portfolio(strategies_data, objective="sharpe"):
    """
    组合优化 - 寻找最优权重
    objective: "sharpe" | "min_vol" | "max_return"
    """
    if not strategies_data or len(strategies_data) < 2:
        return {"error": "至少需要2个策略进行优化"}

    n = len(strategies_data)

    # 提取收益率矩阵
    returns_list = [s.get("returns", []) for s in strategies_data]
    max_len = max(len(r) for r in returns_list)

    aligned = []
    for r in returns_list:
        if len(r) < max_len:
            r = [0] * (max_len - len(r)) + r
        aligned.append(r)

    returns_matrix = np.array(aligned).T

    # 计算协方差矩阵
    cov_matrix = np.cov(returns_matrix.T) * 252
    mean_returns = np.mean(returns_matrix, axis=0) * 252

    best_weights = None
    best_score = -np.inf if objective != "min_vol" else np.inf

    # 网格搜索
    for i in range(2000):
        if n == 2:
            w1 = np.random.random()
            w = np.array([w1, 1 - w1])
        else:
            w = np.random.random(n)
            w = w / w.sum()

        port_return = np.dot(w, mean_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

        if objective == "sharpe":
            score = (port_return - 0.03) / port_vol if port_vol > 0 else -np.inf
        elif objective == "min_vol":
            score = -port_vol
        elif objective == "max_return":
            score = port_return
        else:
            score = (port_return - 0.03) / port_vol if port_vol > 0 else -np.inf

        if (objective != "min_vol" and score > best_score) or (objective == "min_vol" and score > best_score):
            best_score = score
            best_weights = w.copy()

    if best_weights is None:
        best_weights = np.ones(n) / n

    optimization_result = []
    for i, s in enumerate(strategies_data):
        optimization_result.append({
            "策略名称": s.get("name", f"策略{i+1}"),
            "最优权重": round(float(best_weights[i]) * 100, 1),
            "分配资金": round(1000000 * float(best_weights[i]), 2),
        })

    port_return = float(np.dot(best_weights, mean_returns)) * 100
    port_vol = float(np.sqrt(np.dot(best_weights.T, np.dot(cov_matrix, best_weights)))) * 100
    port_sharpe = (port_return / 100 - 0.03) / (port_vol / 100) if port_vol > 0 else 0

    return {
        "优化目标": objective,
        "优化结果": optimization_result,
        "预期年化收益": round(port_return, 2),
        "预期年化波动": round(port_vol, 2),
        "预期夏普比率": round(port_sharpe, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="策略组合管理")
    subparsers = parser.add_subparsers(dest="action", help="操作")

    # 创建组合
    create_parser = subparsers.add_parser("create", help="创建策略组合")
    create_parser.add_argument("--name", required=True, help="组合名称")
    create_parser.add_argument("--strategies", required=True, help="策略列表JSON")
    create_parser.add_argument("--capital", type=float, default=1000000, help="初始资金")

    # 资金分配
    alloc_parser = subparsers.add_parser("allocate", help="资金分配")
    alloc_parser.add_argument("--method", default="equal",
                               choices=["equal", "risk_parity", "kelly", "volatility_weighted"])
    alloc_parser.add_argument("--data", required=True, help="策略数据JSON")
    alloc_parser.add_argument("--capital", type=float, default=1000000, help="总资金")

    # 组合回测
    bt_parser = subparsers.add_parser("backtest", help="组合回测")
    bt_parser.add_argument("--data", required=True, help="策略数据JSON")
    bt_parser.add_argument("--capital", type=float, default=1000000, help="初始资金")

    # 组合优化
    opt_parser = subparsers.add_parser("optimize", help="组合优化")
    opt_parser.add_argument("--data", required=True, help="策略数据JSON")
    opt_parser.add_argument("--objective", default="sharpe",
                             choices=["sharpe", "min_vol", "max_return"])

    args = parser.parse_args()

    try:
        if args.action == "create":
            strategies = json.loads(args.strategies)
            result = create_portfolio(args.name, strategies, args.capital)
        elif args.action == "allocate":
            data = json.loads(args.data)
            result = allocate_capital(args.method, data, args.capital)
        elif args.action == "backtest":
            data = json.loads(args.data)
            result = backtest_portfolio(data, args.capital)
        elif args.action == "optimize":
            data = json.loads(args.data)
            result = optimize_portfolio(data, args.objective)
        else:
            parser.print_help()
            return
    except json.JSONDecodeError as e:
        result = {"error": f"JSON解析失败: {str(e)}"}
    except Exception as e:
        result = {"error": str(e)}

    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
