#!/usr/bin/env python3
"""
网格交易策略系统
支持等距网格、等比网格、网格回测、网格参数优化
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


# ==================== 网格设计 ====================

def design_grid(current_price, grid_type="equal", grid_count=10, grid_range_pct=20,
                price_precision=0.01):
    """
    设计网格参数

    参数:
        current_price: 当前价格
        grid_type: 网格类型（equal=等距, ratio=等比）
        grid_count: 网格层数
        grid_range_pct: 价格范围（%）
        price_precision: 价格精度

    返回: 网格设计结果
    """
    price_low = current_price * (1 - grid_range_pct / 100)
    price_high = current_price * (1 + grid_range_pct / 100)

    grid_levels = []

    if grid_type == "equal":
        step = (price_high - price_low) / grid_count
        for i in range(grid_count + 1):
            price = price_low + i * step
            grid_levels.append({
                "层级": i,
                "价格": round(price / price_precision) * price_precision,
                "距当前": f"{(price / current_price - 1) * 100:+.1f}%",
            })
    elif grid_type == "ratio":
        ratio = (price_high / price_low) ** (1 / grid_count)
        for i in range(grid_count + 1):
            price = price_low * (ratio ** i)
            grid_levels.append({
                "层级": i,
                "价格": round(price / price_precision) * price_precision,
                "距当前": f"{(price / current_price - 1) * 100:+.1f}%",
            })

    # 每格资金
    per_grid_capital = None  # 需要总资金才能计算

    return {
        "设计时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "当前价格": current_price,
        "网格类型": "等距网格" if grid_type == "equal" else "等比网格",
        "价格区间": {
            "下限": round(price_low, 2),
            "上限": round(price_high, 2),
            "范围": f"±{grid_range_pct}%",
        },
        "网格层数": grid_count,
        "网格层级": grid_levels,
        "操作说明": [
            "价格触及网格线时执行买入（下方）或卖出（上方）",
            "每格买卖数量相同，实现低买高卖",
            "震荡市中网格交易收益稳定，单边市中可能被套",
        ],
    }


# ==================== 网格回测 ====================

def grid_backtest(symbol, grid_count=10, grid_range_pct=20, initial_capital=100000,
                  days=500, grid_type="equal"):
    """
    网格交易回测

    参数:
        symbol: 股票代码
        grid_count: 网格层数
        grid_range_pct: 价格范围（%）
        initial_capital: 初始资金
        days: 回测天数
        grid_type: 网格类型

    返回: 回测结果
    """
    df = get_stock_kline(symbol, days=days)
    if df is None or len(df) < 50:
        return {"error": f"{symbol}数据不足"}

    close_col = '收盘' if '收盘' in df.columns else 'close'
    close = pd.to_numeric(df[close_col], errors='coerce').dropna()

    if len(close) < 50:
        return {"error": "有效数据不足"}

    # 以起始价格为基准设计网格
    start_price = float(close.iloc[0])
    price_low = start_price * (1 - grid_range_pct / 100)
    price_high = start_price * (1 + grid_range_pct / 100)

    if grid_type == "equal":
        step = (price_high - price_low) / grid_count
        grid_prices = [price_low + i * step for i in range(grid_count + 1)]
    else:
        ratio = (price_high / price_low) ** (1 / grid_count)
        grid_prices = [price_low * (ratio ** i) for i in range(grid_count + 1)]

    # 每格资金
    per_grid_capital = initial_capital / (grid_count + 1)

    # 初始化
    cash = initial_capital
    shares = 0
    base_shares = int(per_grid_capital / start_price / 100) * 100  # 每格100股整数倍
    if base_shares < 100:
        base_shares = 100

    # 记录每格状态
    grid_bought = [False] * (grid_count + 1)
    grid_sold = [False] * (grid_count + 1)

    equity_curve = []
    trades = []
    total_buy = 0
    total_sell = 0

    for i in range(len(close)):
        price = float(close.iloc[i])
        date_str = str(close.index[i])[:10]

        # 检查网格触发
        for j, grid_price in enumerate(grid_prices):
            # 价格从上往下穿越网格线 -> 买入
            if i > 0 and float(close.iloc[i - 1]) > grid_price and price <= grid_price:
                if not grid_bought[j] and cash >= base_shares * price:
                    buy_shares = base_shares
                    cost = buy_shares * price
                    cash -= cost
                    shares += buy_shares
                    grid_bought[j] = True
                    grid_sold[j] = False
                    total_buy += 1
                    trades.append({
                        "日期": date_str,
                        "方向": "买入",
                        "价格": round(price, 2),
                        "数量": buy_shares,
                        "金额": round(cost, 2),
                        "网格层级": j,
                    })

            # 价格从下往上穿越网格线 -> 卖出
            if i > 0 and float(close.iloc[i - 1]) < grid_price and price >= grid_price:
                if grid_bought[j] and not grid_sold[j] and shares >= base_shares:
                    sell_shares = base_shares
                    revenue = sell_shares * price
                    cash += revenue
                    shares -= sell_shares
                    grid_sold[j] = True
                    grid_bought[j] = False
                    total_sell += 1
                    trades.append({
                        "日期": date_str,
                        "方向": "卖出",
                        "价格": round(price, 2),
                        "数量": sell_shares,
                        "金额": round(revenue, 2),
                        "网格层级": j,
                    })

        equity = cash + shares * price
        equity_curve.append(float(equity))

    # 计算指标
    equity_arr = np.array(equity_curve)
    total_return = (equity_arr[-1] / initial_capital - 1) * 100
    annual_return = ((equity_arr[-1] / initial_capital) ** (252 / len(equity_arr)) - 1) * 100

    returns = np.diff(equity_arr) / equity_arr[:-1]
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
    else:
        sharpe = 0

    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - peak) / peak * 100
    max_dd = float(np.min(drawdown))

    # 网格收益
    grid_profit = 0
    buy_trades = [t for t in trades if t["方向"] == "买入"]
    sell_trades = [t for t in trades if t["方向"] == "卖出"]
    for i in range(min(len(buy_trades), len(sell_trades))):
        grid_profit += sell_trades[i]["金额"] - buy_trades[i]["金额"]

    return {
        "股票代码": symbol,
        "回测时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "网格参数": {
            "类型": "等距网格" if grid_type == "equal" else "等比网格",
            "层数": grid_count,
            "价格范围": f"±{grid_range_pct}%",
            "每格资金": f"{per_grid_capital:,.0f}元",
            "每格数量": f"{base_shares}股",
        },
        "回测指标": {
            "初始资金": f"{initial_capital:,}元",
            "最终权益": f"{equity_arr[-1]:,.2f}元",
            "总收益率": f"{total_return:.2f}%",
            "年化收益率": f"{annual_return:.2f}%",
            "夏普比率": round(sharpe, 2),
            "最大回撤": f"{max_dd:.2f}%",
        },
        "交易统计": {
            "总交易次数": len(trades),
            "买入次数": total_buy,
            "卖出次数": total_sell,
            "网格套利收益": f"{grid_profit:,.2f}元",
        },
        "最近交易": trades[-10:],
        "风险提示": [
            "单边上涨市中网格会过早卖飞，收益不如买入持有",
            "单边下跌市中网格会持续买入被套，需设置止损",
            "网格交易最适合震荡市，波动越大收益越高",
        ],
    }


# ==================== 网格参数优化 ====================

def grid_optimization(symbol, days=500):
    """
    网格参数优化
    遍历不同网格层数和范围，找最优参数

    参数:
        symbol: 股票代码
        days: 数据天数

    返回: 优化结果
    """
    grid_counts = [5, 10, 15, 20]
    range_pcts = [10, 15, 20, 25, 30]

    results = []

    for count in grid_counts:
        for rng in range_pcts:
            bt = grid_backtest(symbol, grid_count=count, grid_range_pct=rng,
                              initial_capital=100000, days=days)
            if "error" in bt:
                continue

            total_return = float(bt["回测指标"]["总收益率"].replace("%", ""))
            max_dd = float(bt["回测指标"]["最大回撤"].replace("%", ""))
            trades = bt["交易统计"]["总交易次数"]

            results.append({
                "网格层数": count,
                "价格范围": f"±{rng}%",
                "总收益率": f"{total_return:.2f}%",
                "最大回撤": f"{max_dd:.2f}%",
                "交易次数": trades,
                "收益回撤比": round(total_return / abs(max_dd), 2) if max_dd != 0 else 0,
            })
            time.sleep(0.3)

    if not results:
        return {"error": "未生成有效的参数组合"}

    # 排序
    results.sort(key=lambda x: x["收益回撤比"], reverse=True)

    return {
        "股票代码": symbol,
        "优化时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "最优参数": results[0] if results else None,
        "全部结果": results,
        "建议": [
            "网格层数过多会导致单格收益低，交易频繁增加成本",
            "网格层数过少会错过震荡机会",
            "价格范围应根据股票历史波动率设定",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description='网格交易策略系统')
    subparsers = parser.add_subparsers(dest='command')

    # 网格设计
    design_parser = subparsers.add_parser('design', help='设计网格参数')
    design_parser.add_argument('--price', type=float, required=True, help='当前价格')
    design_parser.add_argument('--type', default='equal', choices=['equal', 'ratio'], help='网格类型')
    design_parser.add_argument('--count', type=int, default=10, help='网格层数')
    design_parser.add_argument('--range', type=float, default=20, help='价格范围(%)')

    # 网格回测
    backtest_parser = subparsers.add_parser('backtest', help='网格交易回测')
    backtest_parser.add_argument('--symbol', required=True, help='股票代码')
    backtest_parser.add_argument('--count', type=int, default=10, help='网格层数')
    backtest_parser.add_argument('--range', type=float, default=20, help='价格范围(%)')
    backtest_parser.add_argument('--capital', type=float, default=100000, help='初始资金')
    backtest_parser.add_argument('--days', type=int, default=500, help='回测天数')
    backtest_parser.add_argument('--type', default='equal', choices=['equal', 'ratio'], help='网格类型')

    # 参数优化
    opt_parser = subparsers.add_parser('optimize', help='网格参数优化')
    opt_parser.add_argument('--symbol', required=True, help='股票代码')
    opt_parser.add_argument('--days', type=int, default=500, help='数据天数')

    args = parser.parse_args()

    if args.command == 'design':
        result = design_grid(args.price, args.type, args.count, args.range)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'backtest':
        result = grid_backtest(args.symbol, args.count, args.range,
                              args.capital, args.days, args.type)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'optimize':
        result = grid_optimization(args.symbol, args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
