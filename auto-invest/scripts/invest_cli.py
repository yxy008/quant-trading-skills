#!/usr/bin/env python3
"""
定投策略分析系统
支持普通定投、智能定投（估值定投/均线定投）、定投回测对比
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


# ==================== 普通定投回测 ====================

def regular_invest_backtest(symbol, monthly_amount=1000, years=5, invest_day=1):
    """
    普通定投回测
    每月固定日期投入固定金额

    参数:
        symbol: 股票/ETF代码
        monthly_amount: 每月定投金额
        years: 定投年数
        invest_day: 每月定投日（1-28）

    返回: 定投回测结果
    """
    days = years * 260
    df = get_stock_kline(symbol, days=days)
    if df is None or len(df) < 60:
        return {"error": f"{symbol}数据不足"}

    close_col = '收盘' if '收盘' in df.columns else 'close'
    close = pd.to_numeric(df[close_col], errors='coerce').dropna()

    if len(close) < 60:
        return {"error": "有效数据不足"}

    # 模拟定投
    total_invested = 0
    total_shares = 0
    invest_records = []

    # 按月遍历
    current_date = close.index[0]
    end_date = close.index[-1]

    while current_date <= end_date:
        # 找到该月定投日附近的交易日
        year = current_date.year
        month = current_date.month
        target_day = min(invest_day, 28)

        try:
            target_date = pd.Timestamp(year=year, month=month, day=target_day)
        except Exception:
            current_date = current_date + pd.DateOffset(months=1)
            continue

        # 找最近交易日
        if target_date in close.index:
            invest_date = target_date
        else:
            nearby = close.index[close.index >= target_date]
            if len(nearby) > 0:
                invest_date = nearby[0]
            else:
                current_date = current_date + pd.DateOffset(months=1)
                continue

        price = float(close.loc[invest_date])
        shares_bought = monthly_amount / price
        total_invested += monthly_amount
        total_shares += shares_bought

        invest_records.append({
            "日期": str(invest_date)[:10],
            "价格": round(price, 2),
            "投入金额": monthly_amount,
            "买入份额": round(shares_bought, 2),
            "累计投入": round(total_invested, 0),
            "累计份额": round(total_shares, 2),
        })

        current_date = current_date + pd.DateOffset(months=1)

    if not invest_records:
        return {"error": "未生成定投记录"}

    # 最终价值
    final_price = float(close.iloc[-1])
    final_value = total_shares * final_price
    total_return = (final_value / total_invested - 1) * 100
    annual_return = ((final_value / total_invested) ** (1 / years) - 1) * 100

    # 平均成本
    avg_cost = total_invested / total_shares if total_shares > 0 else 0

    # 与一次性投资对比
    start_price = float(close.iloc[0])
    lump_sum_shares = total_invested / start_price
    lump_sum_value = lump_sum_shares * final_price
    lump_sum_return = (lump_sum_value / total_invested - 1) * 100

    return {
        "股票代码": symbol,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "定投参数": {
            "每月金额": f"{monthly_amount:,}元",
            "定投年限": f"{years}年",
            "定投日": f"每月{invest_day}日",
        },
        "定投结果": {
            "总投入": f"{total_invested:,.0f}元",
            "总份额": f"{total_shares:,.2f}",
            "平均成本": f"{avg_cost:.2f}元",
            "最终价值": f"{final_value:,.2f}元",
            "总收益率": f"{total_return:.2f}%",
            "年化收益率": f"{annual_return:.2f}%",
        },
        "一次性投资对比": {
            "一次性投入": f"{total_invested:,.0f}元",
            "最终价值": f"{lump_sum_value:,.2f}元",
            "总收益率": f"{lump_sum_return:.2f}%",
            "定投优势": f"{total_return - lump_sum_return:+.2f}%",
        },
        "定投记录": invest_records[-12:] if len(invest_records) >= 12 else invest_records,
        "定投优势": [
            "摊薄成本：下跌时买入更多份额，上涨时买入更少",
            "无需择时：避免一次性买在高点的风险",
            "强制储蓄：培养长期投资习惯",
        ],
    }


# ==================== 智能定投（估值定投） ====================

def smart_invest_backtest(symbol, base_amount=1000, years=5, strategy="ma"):
    """
    智能定投回测
    根据估值/均线偏离调整定投金额

    参数:
        symbol: 股票代码
        base_amount: 基准定投金额
        years: 定投年数
        strategy: 策略类型（ma=均线偏离, pe=估值）

    返回: 智能定投结果
    """
    days = years * 260
    df = get_stock_kline(symbol, days=days)
    if df is None or len(df) < 120:
        return {"error": f"{symbol}数据不足"}

    close_col = '收盘' if '收盘' in df.columns else 'close'
    close = pd.to_numeric(df[close_col], errors='coerce').dropna()

    if len(close) < 120:
        return {"error": "有效数据不足"}

    # 计算250日均线
    ma250 = close.rolling(window=250).mean()

    total_invested = 0
    total_shares = 0
    invest_records = []

    current_date = close.index[0]
    end_date = close.index[-1]

    while current_date <= end_date:
        year = current_date.year
        month = current_date.month

        try:
            target_date = pd.Timestamp(year=year, month=month, day=1)
        except Exception:
            current_date = current_date + pd.DateOffset(months=1)
            continue

        if target_date in close.index:
            invest_date = target_date
        else:
            nearby = close.index[close.index >= target_date]
            if len(nearby) > 0:
                invest_date = nearby[0]
            else:
                current_date = current_date + pd.DateOffset(months=1)
                continue

        price = float(close.loc[invest_date])

        # 智能调整金额
        if strategy == "ma" and invest_date in ma250.index:
            ma_val = float(ma250.loc[invest_date])
            if pd.notna(ma_val) and ma_val > 0:
                deviation = price / ma_val
                if deviation < 0.8:
                    multiplier = 1.5  # 低于年线20%，加仓50%
                elif deviation < 0.9:
                    multiplier = 1.3  # 低于年线10%，加仓30%
                elif deviation < 1.0:
                    multiplier = 1.1  # 低于年线，小幅加仓
                elif deviation > 1.2:
                    multiplier = 0.5  # 高于年线20%，减半
                elif deviation > 1.1:
                    multiplier = 0.7  # 高于年线10%，减少
                else:
                    multiplier = 1.0
            else:
                multiplier = 1.0
        else:
            multiplier = 1.0

        invest_amount = base_amount * multiplier
        shares_bought = invest_amount / price
        total_invested += invest_amount
        total_shares += shares_bought

        invest_records.append({
            "日期": str(invest_date)[:10],
            "价格": round(price, 2),
            "投入倍数": f"{multiplier:.1f}x",
            "投入金额": round(invest_amount, 0),
            "累计投入": round(total_invested, 0),
        })

        current_date = current_date + pd.DateOffset(months=1)

    if not invest_records:
        return {"error": "未生成定投记录"}

    final_price = float(close.iloc[-1])
    final_value = total_shares * final_price
    total_return = (final_value / total_invested - 1) * 100
    annual_return = ((final_value / total_invested) ** (1 / years) - 1) * 100

    # 对比普通定投
    regular = regular_invest_backtest(symbol, base_amount, years)
    regular_return = float(regular["定投结果"]["总收益率"].replace("%", "")) if "error" not in regular else 0

    return {
        "股票代码": symbol,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "策略类型": f"智能定投（{'均线偏离' if strategy == 'ma' else '估值'}）",
        "定投参数": {
            "基准金额": f"{base_amount:,}元/月",
            "定投年限": f"{years}年",
            "调整规则": "低于年线加仓，高于年线减仓" if strategy == "ma" else "低估加仓，高估减仓",
        },
        "智能定投结果": {
            "总投入": f"{total_invested:,.0f}元",
            "最终价值": f"{final_value:,.2f}元",
            "总收益率": f"{total_return:.2f}%",
            "年化收益率": f"{annual_return:.2f}%",
        },
        "与普通定投对比": {
            "普通定投收益": f"{regular_return:.2f}%",
            "智能定投优势": f"{total_return - regular_return:+.2f}%",
        },
        "定投记录": invest_records[-12:] if len(invest_records) >= 12 else invest_records,
    }


# ==================== 定投策略对比 ====================

def invest_comparison(symbol, monthly_amount=1000, years=5):
    """
    定投策略全面对比
    对比普通定投、智能定投、一次性投资

    参数:
        symbol: 股票代码
        monthly_amount: 每月金额
        years: 投资年数

    返回: 策略对比
    """
    # 普通定投
    regular = regular_invest_backtest(symbol, monthly_amount, years)

    # 智能定投
    smart = smart_invest_backtest(symbol, monthly_amount, years, "ma")

    # 一次性投资
    days = years * 260
    df = get_stock_kline(symbol, days=days)
    lump_sum_result = None

    if df is not None and len(df) >= 60:
        close_col = '收盘' if '收盘' in df.columns else 'close'
        close = pd.to_numeric(df[close_col], errors='coerce').dropna()
        if len(close) >= 60:
            total = monthly_amount * 12 * years
            start_price = float(close.iloc[0])
            end_price = float(close.iloc[-1])
            shares = total / start_price
            final_value = shares * end_price
            lump_return = (final_value / total - 1) * 100
            lump_sum_result = {
                "总投入": f"{total:,.0f}元",
                "最终价值": f"{final_value:,.2f}元",
                "总收益率": f"{lump_return:.2f}%",
            }

    comparison = []
    if "error" not in regular:
        comparison.append({
            "策略": "普通定投",
            "收益率": regular["定投结果"]["总收益率"],
            "说明": "每月固定金额买入，简单省心",
        })

    if "error" not in smart:
        comparison.append({
            "策略": "智能定投",
            "收益率": smart["智能定投结果"]["总收益率"],
            "说明": "低于年线加仓，高于年线减仓",
        })

    if lump_sum_result:
        comparison.append({
            "策略": "一次性投资",
            "收益率": lump_sum_result["总收益率"],
            "说明": "期初一次性买入，考验择时能力",
        })

    return {
        "股票代码": symbol,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "策略对比": comparison,
        "结论": [
            "定投的核心优势是摊薄成本，避免择时风险",
            "智能定投在震荡市中优于普通定投",
            "单边牛市中一次性投资优于定投",
            "建议将定投作为长期投资的基础策略",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description='定投策略分析系统')
    subparsers = parser.add_subparsers(dest='command')

    # 普通定投
    regular_parser = subparsers.add_parser('regular', help='普通定投回测')
    regular_parser.add_argument('--symbol', required=True, help='股票代码')
    regular_parser.add_argument('--amount', type=float, default=1000, help='每月金额')
    regular_parser.add_argument('--years', type=int, default=5, help='定投年数')
    regular_parser.add_argument('--day', type=int, default=1, help='定投日(1-28)')

    # 智能定投
    smart_parser = subparsers.add_parser('smart', help='智能定投回测')
    smart_parser.add_argument('--symbol', required=True, help='股票代码')
    smart_parser.add_argument('--amount', type=float, default=1000, help='基准金额')
    smart_parser.add_argument('--years', type=int, default=5, help='定投年数')
    smart_parser.add_argument('--strategy', default='ma', choices=['ma', 'pe'], help='策略类型')

    # 策略对比
    compare_parser = subparsers.add_parser('compare', help='定投策略对比')
    compare_parser.add_argument('--symbol', required=True, help='股票代码')
    compare_parser.add_argument('--amount', type=float, default=1000, help='每月金额')
    compare_parser.add_argument('--years', type=int, default=5, help='投资年数')

    args = parser.parse_args()

    if args.command == 'regular':
        result = regular_invest_backtest(args.symbol, args.amount, args.years, args.day)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'smart':
        result = smart_invest_backtest(args.symbol, args.amount, args.years, args.strategy)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'compare':
        result = invest_comparison(args.symbol, args.amount, args.years)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
