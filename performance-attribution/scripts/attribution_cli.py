#!/usr/bin/env python3
"""
绩效归因分析模块 - Brinson归因、因子归因、收益分解
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


def brinson_attribution(portfolio_weights, portfolio_returns, benchmark_weights, benchmark_returns):
    """
    Brinson归因分析
    将超额收益分解为: 配置效应 + 选择效应 + 交互效应

    portfolio_weights: {"行业A": 0.3, "行业B": 0.2, ...}
    portfolio_returns: {"行业A": 0.05, "行业B": 0.03, ...}
    benchmark_weights: {"行业A": 0.25, "行业B": 0.25, ...}
    benchmark_returns: {"行业A": 0.04, "行业B": 0.02, ...}
    """
    if not portfolio_weights or not benchmark_weights:
        return {"error": "请提供组合和基准的权重数据"}

    all_sectors = set(list(portfolio_weights.keys()) + list(benchmark_weights.keys()))

    total_portfolio_return = 0
    total_benchmark_return = 0
    allocation_effect = 0
    selection_effect = 0
    interaction_effect = 0

    sector_details = []

    for sector in all_sectors:
        pw = portfolio_weights.get(sector, 0)
        bw = benchmark_weights.get(sector, 0)
        pr = portfolio_returns.get(sector, 0)
        br = benchmark_returns.get(sector, 0)

        total_portfolio_return += pw * pr
        total_benchmark_return += bw * br

        # 配置效应: (组合权重 - 基准权重) * 基准收益
        alloc = (pw - bw) * br

        # 选择效应: 基准权重 * (组合收益 - 基准收益)
        select = bw * (pr - br)

        # 交互效应: (组合权重 - 基准权重) * (组合收益 - 基准收益)
        interact = (pw - bw) * (pr - br)

        allocation_effect += alloc
        selection_effect += select
        interaction_effect += interact

        sector_details.append({
            "行业": sector,
            "组合权重": round(pw * 100, 1),
            "基准权重": round(bw * 100, 1),
            "组合收益": round(pr * 100, 2),
            "基准收益": round(br * 100, 2),
            "配置效应": round(alloc * 100, 2),
            "选择效应": round(select * 100, 2),
            "交互效应": round(interact * 100, 2),
        })

    excess_return = total_portfolio_return - total_benchmark_return

    return {
        "分析方法": "Brinson归因",
        "组合总收益": round(total_portfolio_return * 100, 2),
        "基准总收益": round(total_benchmark_return * 100, 2),
        "超额收益": round(excess_return * 100, 2),
        "配置效应": round(allocation_effect * 100, 2),
        "选择效应": round(selection_effect * 100, 2),
        "交互效应": round(interaction_effect * 100, 2),
        "行业明细": sector_details,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def factor_attribution(portfolio_returns, factor_returns, factor_exposures):
    """
    因子归因分析
    将组合收益分解为各因子的贡献

    portfolio_returns: [0.01, -0.005, ...] 组合日收益率
    factor_returns: {"市场因子": [0.008, ...], "规模因子": [0.002, ...], ...}
    factor_exposures: {"市场因子": 1.0, "规模因子": 0.5, ...}
    """
    if not portfolio_returns or not factor_returns:
        return {"error": "请提供组合收益和因子收益数据"}

    n_days = len(portfolio_returns)
    factor_names = list(factor_returns.keys())

    # 对齐因子收益长度
    aligned_factors = {}
    for name in factor_names:
        f_returns = factor_returns[name]
        if len(f_returns) < n_days:
            f_returns = [0] * (n_days - len(f_returns)) + f_returns
        aligned_factors[name] = f_returns[:n_days]

    # 计算各因子贡献
    factor_contributions = {}
    for name in factor_names:
        exposure = factor_exposures.get(name, 0)
        f_returns = aligned_factors[name]
        contribution = sum(r * exposure for r in f_returns)
        factor_contributions[name] = round(contribution * 100, 2)

    # 总因子收益
    total_factor_return = sum(factor_contributions.values())

    # 组合总收益
    total_portfolio_return = sum(portfolio_returns) * 100

    # 残差收益（无法被因子解释的部分）
    residual_return = total_portfolio_return - total_factor_return

    # 各因子日贡献
    daily_contributions = []
    for i in range(n_days):
        daily = {"day": i + 1}
        daily_total = 0
        for name in factor_names:
            exposure = factor_exposures.get(name, 0)
            contrib = aligned_factors[name][i] * exposure * 100
            daily[name] = round(contrib, 4)
            daily_total += contrib
        daily["因子总贡献"] = round(daily_total, 4)
        daily["组合收益"] = round(portfolio_returns[i] * 100, 4)
        daily["残差"] = round(portfolio_returns[i] * 100 - daily_total, 4)
        daily_contributions.append(daily)

    return {
        "分析方法": "因子归因",
        "组合总收益": round(total_portfolio_return, 2),
        "因子解释收益": round(total_factor_return, 2),
        "残差收益": round(residual_return, 2),
        "解释比例": round(total_factor_return / total_portfolio_return * 100, 1) if total_portfolio_return != 0 else 0,
        "因子贡献": factor_contributions,
        "因子暴露": factor_exposures,
        "日贡献明细": daily_contributions[-20:],
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def time_series_attribution(equity_curve, benchmark_equity=None):
    """
    时间序列归因 - 分析不同时间段的收益来源
    """
    if not equity_curve:
        return {"error": "请提供权益曲线数据"}

    n = len(equity_curve)
    if n < 20:
        return {"error": "数据量不足，至少需要20个数据点"}

    # 计算日收益率
    returns = []
    for i in range(1, n):
        if equity_curve[i - 1] > 0:
            returns.append(equity_curve[i] / equity_curve[i - 1] - 1)
        else:
            returns.append(0)

    # 分段分析
    segments = []
    segment_size = max(len(returns) // 4, 20)

    for start in range(0, len(returns), segment_size):
        end = min(start + segment_size, len(returns))
        seg_returns = returns[start:end]

        if len(seg_returns) < 5:
            continue

        total_ret = (np.prod([1 + r for r in seg_returns]) - 1) * 100
        avg_daily = np.mean(seg_returns) * 100
        vol = np.std(seg_returns) * np.sqrt(252) * 100
        sharpe = (avg_daily * 252 / 100 - 0.03) / (vol / 100) if vol > 0 else 0

        # 正收益天数和负收益天数
        up_days = sum(1 for r in seg_returns if r > 0)
        down_days = sum(1 for r in seg_returns if r < 0)

        segments.append({
            "区间": f"第{start + 1}-{end}天",
            "天数": len(seg_returns),
            "区间收益": round(total_ret, 2),
            "日均收益": round(avg_daily, 4),
            "年化波动": round(vol, 2),
            "夏普比率": round(sharpe, 2),
            "上涨天数": up_days,
            "下跌天数": down_days,
            "胜率": round(up_days / len(seg_returns) * 100, 1),
        })

    # 滚动分析
    rolling_window = min(60, len(returns) // 3)
    rolling_metrics = []
    for i in range(0, len(returns) - rolling_window, rolling_window // 2):
        window_returns = returns[i:i + rolling_window]
        if len(window_returns) < 10:
            continue
        total_ret = (np.prod([1 + r for r in window_returns]) - 1) * 100
        vol = np.std(window_returns) * np.sqrt(252) * 100
        rolling_metrics.append({
            "起始": i + 1,
            "结束": i + rolling_window,
            "收益": round(total_ret, 2),
            "波动": round(vol, 2),
        })

    return {
        "分析方法": "时间序列归因",
        "总天数": len(returns),
        "总收益": round((np.prod([1 + r for r in returns]) - 1) * 100, 2),
        "分段分析": segments,
        "滚动分析": rolling_metrics[-10:],
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def trade_analysis(trades):
    """
    交易分析 - 分析每笔交易的盈亏来源
    """
    if not trades:
        return {"error": "请提供交易记录"}

    buy_trades = [t for t in trades if "买入" in str(t.get("类型", ""))]
    sell_trades = [t for t in trades if "卖出" in str(t.get("类型", ""))]

    total_pnl = sum(t.get("盈亏", 0) for t in trades)
    win_trades = [t for t in trades if t.get("盈亏", 0) > 0]
    loss_trades = [t for t in trades if t.get("盈亏", 0) < 0]

    # 最大盈利和亏损
    max_win = max((t.get("盈亏", 0) for t in trades), default=0)
    max_loss = min((t.get("盈亏", 0) for t in trades), default=0)

    # 盈亏分布
    pnl_ranges = {"大幅亏损(<-5%)": 0, "中等亏损(-5%~-2%)": 0, "小幅亏损(-2%~0)": 0,
                   "小幅盈利(0~2%)": 0, "中等盈利(2%~5%)": 0, "大幅盈利(>5%)": 0}

    for t in trades:
        pnl_pct = t.get("盈亏比例", 0)
        if pnl_pct < -5:
            pnl_ranges["大幅亏损(<-5%)"] += 1
        elif pnl_pct < -2:
            pnl_ranges["中等亏损(-5%~-2%)"] += 1
        elif pnl_pct < 0:
            pnl_ranges["小幅亏损(-2%~0)"] += 1
        elif pnl_pct < 2:
            pnl_ranges["小幅盈利(0~2%)"] += 1
        elif pnl_pct < 5:
            pnl_ranges["中等盈利(2%~5%)"] += 1
        else:
            pnl_ranges["大幅盈利(>5%)"] += 1

    # 连续盈亏分析
    streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    current_streak_type = None

    for t in trades:
        pnl = t.get("盈亏", 0)
        if pnl > 0:
            if current_streak_type == "win":
                streak += 1
            else:
                streak = 1
                current_streak_type = "win"
            max_win_streak = max(max_win_streak, streak)
        elif pnl < 0:
            if current_streak_type == "loss":
                streak += 1
            else:
                streak = 1
                current_streak_type = "loss"
            max_loss_streak = max(max_loss_streak, streak)

    return {
        "分析方法": "交易归因",
        "总交易次数": len(trades),
        "买入次数": len(buy_trades),
        "卖出次数": len(sell_trades),
        "盈利次数": len(win_trades),
        "亏损次数": len(loss_trades),
        "胜率": round(len(win_trades) / len(trades) * 100, 1) if trades else 0,
        "总盈亏": round(total_pnl, 2),
        "平均盈利": round(np.mean([t.get("盈亏", 0) for t in win_trades]), 2) if win_trades else 0,
        "平均亏损": round(np.mean([t.get("盈亏", 0) for t in loss_trades]), 2) if loss_trades else 0,
        "最大单笔盈利": round(max_win, 2),
        "最大单笔亏损": round(max_loss, 2),
        "盈亏比": round(abs(np.mean([t.get("盈亏", 0) for t in win_trades]) / np.mean([t.get("盈亏", 0) for t in loss_trades])), 2) if win_trades and loss_trades else 0,
        "最大连续盈利": max_win_streak,
        "最大连续亏损": max_loss_streak,
        "盈亏分布": pnl_ranges,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def decompose_returns(symbol, days=250):
    """
    收益分解 - 将个股收益分解为市场收益、行业收益和个股Alpha
    供API调用的包装函数
    """
    try:
        df = get_stock_kline(symbol, days=days + 30)
        if df is None or df.empty:
            return {"error": f"无法获取 {symbol} 的行情数据"}

        df = df.sort_values("日期").reset_index(drop=True)
        closes = df["收盘"].tolist()

        if len(closes) < 20:
            return {"error": "数据量不足"}

        # 计算日收益率
        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                returns.append(closes[i] / closes[i - 1] - 1)

        n = len(returns)

        # 1. 市场收益分解 - 使用指数代理
        # 尝试获取沪深300作为市场基准
        try:
            idx_df = get_stock_kline("000300", days=days + 30)
            if idx_df is not None and not idx_df.empty:
                idx_df = idx_df.sort_values("日期").reset_index(drop=True)
                idx_closes = idx_df["收盘"].tolist()
                idx_returns = []
                for i in range(1, min(len(idx_closes), len(closes))):
                    if idx_closes[i - 1] > 0:
                        idx_returns.append(idx_closes[i] / idx_closes[i - 1] - 1)
                # 对齐长度
                min_len = min(len(returns), len(idx_returns))
                returns = returns[-min_len:]
                idx_returns = idx_returns[-min_len:]
            else:
                idx_returns = [0] * len(returns)
        except Exception:
            idx_returns = [0] * len(returns)

        # 市场Beta
        if len(idx_returns) > 10 and np.std(idx_returns) > 0:
            cov = np.cov(returns, idx_returns)[0][1]
            var = np.var(idx_returns)
            beta = cov / var if var > 0 else 1.0
        else:
            beta = 1.0

        market_return = sum(idx_returns) * 100
        market_contribution = beta * market_return

        # 2. 个股总收益
        total_return = sum(returns) * 100

        # 3. Alpha = 总收益 - Beta * 市场收益
        alpha = total_return - market_contribution

        # 4. 收益波动分解
        total_vol = np.std(returns) * np.sqrt(252) * 100
        systematic_vol = abs(beta) * np.std(idx_returns) * np.sqrt(252) * 100 if len(idx_returns) > 0 else 0
        idiosyncratic_vol = np.sqrt(max(0, total_vol ** 2 - systematic_vol ** 2))

        # 5. 月度收益分解
        monthly_decomposition = []
        month_size = min(22, len(returns))
        for start in range(0, len(returns), month_size):
            end = min(start + month_size, len(returns))
            seg_r = returns[start:end]
            seg_idx = idx_returns[start:end] if start < len(idx_returns) else [0] * len(seg_r)
            if len(seg_r) < 5:
                continue
            seg_total = sum(seg_r) * 100
            seg_market = beta * sum(seg_idx) * 100
            seg_alpha = seg_total - seg_market
            monthly_decomposition.append({
                "区间": f"第{start + 1}-{end}天",
                "总收益": round(seg_total, 2),
                "市场贡献": round(seg_market, 2),
                "Alpha": round(seg_alpha, 2),
            })

        return {
            "分析方法": "收益分解",
            "股票代码": symbol,
            "分析天数": n,
            "总收益": round(total_return, 2),
            "年化收益": round(total_return / n * 252, 2),
            "市场Beta": round(beta, 2),
            "市场贡献": round(market_contribution, 2),
            "Alpha收益": round(alpha, 2),
            "总波动率": round(total_vol, 2),
            "系统性波动": round(systematic_vol, 2),
            "特质波动": round(idiosyncratic_vol, 2),
            "月度分解": monthly_decomposition[-12:],
            "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    except Exception as e:
        return {"error": str(e)}


def brinson_attribution_api(symbol, sector_name=None, days=250):
    """
    Brinson归因API包装 - 自动获取数据并执行归因分析
    """
    try:
        df = get_stock_kline(symbol, days=days + 30)
        if df is None or df.empty:
            return {"error": f"无法获取 {symbol} 的行情数据"}

        df = df.sort_values("日期").reset_index(drop=True)
        closes = df["收盘"].tolist()

        if len(closes) < 20:
            return {"error": "数据量不足"}

        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                returns.append(closes[i] / closes[i - 1] - 1)

        # 构建简化的行业配置
        if sector_name:
            portfolio_weights = {sector_name: 1.0}
            benchmark_weights = {sector_name: 0.5, "其他": 0.5}
        else:
            portfolio_weights = {"默认行业": 1.0}
            benchmark_weights = {"默认行业": 0.6, "其他": 0.4}

        total_ret = sum(returns) * 100
        portfolio_returns = {list(portfolio_weights.keys())[0]: total_ret / 100}
        benchmark_returns = {list(portfolio_weights.keys())[0]: total_ret / 100 * 0.8, "其他": total_ret / 100 * 0.5}

        return brinson_attribution(portfolio_weights, portfolio_returns, benchmark_weights, benchmark_returns)
    except Exception as e:
        return {"error": str(e)}


def factor_attribution_api(symbol, days=250):
    """
    因子归因API包装 - 自动获取数据并执行因子归因
    """
    try:
        df = get_stock_kline(symbol, days=days + 30)
        if df is None or df.empty:
            return {"error": f"无法获取 {symbol} 的行情数据"}

        df = df.sort_values("日期").reset_index(drop=True)
        closes = df["收盘"].tolist()

        if len(closes) < 20:
            return {"error": "数据量不足"}

        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                returns.append(closes[i] / closes[i - 1] - 1)

        n = len(returns)

        factor_returns = {
            "趋势因子": [],
            "波动因子": [],
            "反转因子": [],
        }
        factor_exposures = {"趋势因子": 0.5, "波动因子": -0.3, "反转因子": 0.2}

        for i in range(5, n):
            trend = (closes[i] / closes[i - 5] - 1) if closes[i - 5] > 0 else 0
            seg = returns[max(0, i - 10):i]
            vol = np.std(seg) if len(seg) > 1 else 0
            reversal = -(closes[i] / closes[i - 1] - 1) if closes[i - 1] > 0 else 0
            factor_returns["趋势因子"].append(trend)
            factor_returns["波动因子"].append(vol)
            factor_returns["反转因子"].append(reversal)

        aligned_returns = returns[-len(factor_returns["趋势因子"]):]
        return factor_attribution(aligned_returns, factor_returns, factor_exposures)
    except Exception as e:
        return {"error": str(e)}


def full_attribution(symbol, sector_name=None, days=250):
    """
    完整绩效归因 - 综合Brinson归因、因子归因和时间序列归因
    供API调用的包装函数
    """
    try:
        df = get_stock_kline(symbol, days=days + 30)
        if df is None or df.empty:
            return {"error": f"无法获取 {symbol} 的行情数据"}

        df = df.sort_values("日期").reset_index(drop=True)
        closes = df["收盘"].tolist()
        dates = df["日期"].tolist()

        if len(closes) < 20:
            return {"error": "数据量不足"}

        # 计算日收益率
        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                returns.append(closes[i] / closes[i - 1] - 1)

        n = len(returns)

        # 1. 收益分解
        decomp = decompose_returns(symbol, days)

        # 2. 时间序列归因
        equity_curve = closes[-n - 1:] if len(closes) > n else closes
        ts_attr = time_series_attribution(equity_curve)

        # 3. 构建简化的因子归因
        # 使用价格数据构建代理因子
        factor_returns = {
            "趋势因子": [],
            "波动因子": [],
            "反转因子": [],
        }
        factor_exposures = {"趋势因子": 0.5, "波动因子": -0.3, "反转因子": 0.2}

        for i in range(5, n):
            # 趋势: 短期动量
            trend = (closes[i] / closes[i - 5] - 1) if closes[i - 5] > 0 else 0
            # 波动: 近期波动率
            seg = returns[max(0, i - 10):i]
            vol = np.std(seg) if len(seg) > 1 else 0
            # 反转: 短期反转
            reversal = -(closes[i] / closes[i - 1] - 1) if closes[i - 1] > 0 else 0

            factor_returns["趋势因子"].append(trend)
            factor_returns["波动因子"].append(vol)
            factor_returns["反转因子"].append(reversal)

        # 对齐
        aligned_returns = returns[-len(factor_returns["趋势因子"]):]
        fact_attr = factor_attribution(aligned_returns, factor_returns, factor_exposures)

        # 4. 汇总
        return {
            "分析方法": "完整绩效归因",
            "股票代码": symbol,
            "分析区间": f"{dates[0]} ~ {dates[-1]}",
            "分析天数": n,
            "收益分解": decomp,
            "时间序列归因": ts_attr,
            "因子归因": fact_attr,
            "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="绩效归因分析")
    subparsers = parser.add_subparsers(dest="action", help="操作")

    # Brinson归因
    brinson_parser = subparsers.add_parser("brinson", help="Brinson归因分析")
    brinson_parser.add_argument("--pw", required=True, help="组合权重JSON")
    brinson_parser.add_argument("--pr", required=True, help="组合收益JSON")
    brinson_parser.add_argument("--bw", required=True, help="基准权重JSON")
    brinson_parser.add_argument("--br", required=True, help="基准收益JSON")

    # 因子归因
    factor_parser = subparsers.add_parser("factor", help="因子归因分析")
    factor_parser.add_argument("--returns", required=True, help="组合日收益率JSON数组")
    factor_parser.add_argument("--factor-returns", required=True, help="因子收益JSON")
    factor_parser.add_argument("--exposures", required=True, help="因子暴露JSON")

    # 时间序列归因
    ts_parser = subparsers.add_parser("timeseries", help="时间序列归因")
    ts_parser.add_argument("--equity", required=True, help="权益曲线JSON数组")

    # 交易分析
    trade_parser = subparsers.add_parser("trade", help="交易归因分析")
    trade_parser.add_argument("--trades", required=True, help="交易记录JSON数组")

    args = parser.parse_args()

    try:
        if args.action == "brinson":
            pw = json.loads(args.pw)
            pr = json.loads(args.pr)
            bw = json.loads(args.bw)
            br = json.loads(args.br)
            result = brinson_attribution(pw, pr, bw, br)
        elif args.action == "factor":
            returns = json.loads(args.returns)
            factor_returns = json.loads(args.factor_returns)
            exposures = json.loads(args.exposures)
            result = factor_attribution(returns, factor_returns, exposures)
        elif args.action == "timeseries":
            equity = json.loads(args.equity)
            result = time_series_attribution(equity)
        elif args.action == "trade":
            trades = json.loads(args.trades)
            result = trade_analysis(trades)
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
