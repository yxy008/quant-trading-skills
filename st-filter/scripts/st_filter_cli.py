#!/usr/bin/env python3
"""
ST/*ST股票特殊处理系统
识别ST股票、应用5%涨跌幅限制、风险评估、退市预警
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


ST_RISK_LEVELS = {
    "ST": {
        "涨跌幅限制": 5.0,
        "风险等级": "高",
        "说明": "ST股票（Special Treatment），公司经营连续两年亏损，特别处理",
        "交易限制": ["涨跌幅限制5%", "需签署风险揭示书", "部分券商限制买入"],
        "退市风险": "中",
    },
    "*ST": {
        "涨跌幅限制": 5.0,
        "风险等级": "极高",
        "说明": "*ST股票，公司经营连续三年亏损，有退市风险",
        "交易限制": ["涨跌幅限制5%", "需签署风险揭示书", "退市风险警示", "部分券商禁止买入"],
        "退市风险": "高",
    },
    "科创板ST": {
        "涨跌幅限制": 20.0,
        "风险等级": "高",
        "说明": "科创板ST股票，涨跌幅限制仍为20%",
        "交易限制": ["涨跌幅限制20%", "需开通科创板权限", "需签署风险揭示书"],
        "退市风险": "中",
    },
    "创业板ST": {
        "涨跌幅限制": 20.0,
        "风险等级": "高",
        "说明": "创业板ST股票，涨跌幅限制仍为20%",
        "交易限制": ["涨跌幅限制20%", "需开通创业板权限", "需签署风险揭示书"],
        "退市风险": "中",
    },
    "正常": {
        "涨跌幅限制": 10.0,
        "风险等级": "正常",
        "说明": "正常交易股票",
        "交易限制": [],
        "退市风险": "无",
    },
}


def is_st_stock(symbol):
    """
    判断是否为ST股票

    参数:
        symbol: 股票代码

    返回: ST类型 ("ST", "*ST", "正常")
    """
    try:
        stock_info = ak.stock_individual_info_em(symbol=symbol)
        if stock_info is None or stock_info.empty:
            return "未知"

        name_row = stock_info[stock_info['item'] == '股票简称']
        if name_row.empty:
            return "未知"

        name = str(name_row['value'].values[0])

        if name.startswith('*ST'):
            return "*ST"
        elif name.startswith('ST'):
            return "ST"
        else:
            return "正常"
    except Exception:
        return "未知"


def get_st_risk_info(symbol):
    """
    获取ST股票风险信息

    参数:
        symbol: 股票代码

    返回: 风险信息字典
    """
    st_type = is_st_stock(symbol)

    if st_type == "未知":
        return {
            "股票代码": symbol,
            "ST类型": "未知",
            "风险等级": "未知",
            "说明": "无法获取股票信息，请检查股票代码",
            "建议": "请确认股票代码是否正确",
        }

    base_info = ST_RISK_LEVELS.get(st_type, ST_RISK_LEVELS["正常"]).copy()

    market = "主板"
    if symbol.startswith("688"):
        market = "科创板"
        if st_type in ("ST", "*ST"):
            base_info = ST_RISK_LEVELS["科创板ST"].copy()
    elif symbol.startswith("300") or symbol.startswith("301"):
        market = "创业板"
        if st_type in ("ST", "*ST"):
            base_info = ST_RISK_LEVELS["创业板ST"].copy()

    result = {
        "股票代码": symbol,
        "ST类型": st_type,
        "所属市场": market,
        **base_info,
    }

    if st_type in ("ST", "*ST"):
        result["风险提示"] = _generate_risk_warning(st_type, market)
        result["交易建议"] = _generate_trading_advice(st_type)

    return result


def _generate_risk_warning(st_type, market):
    """生成风险提示"""
    warnings = []

    if st_type == "*ST":
        warnings.append("该股票为*ST，存在退市风险，可能被终止上市")
        warnings.append("公司连续三年亏损，基本面严重恶化")
    elif st_type == "ST":
        warnings.append("该股票为ST，公司连续两年亏损")
        warnings.append("公司经营状况不佳，存在持续亏损风险")

    if market == "科创板":
        warnings.append("科创板股票波动较大，ST后涨跌幅限制仍为20%")
    elif market == "创业板":
        warnings.append("创业板股票波动较大，ST后涨跌幅限制仍为20%")

    warnings.append("ST股票流动性通常较差，买卖价差较大")
    warnings.append("部分机构投资者被限制买入ST股票")

    return warnings


def _generate_trading_advice(st_type):
    """生成交易建议"""
    if st_type == "*ST":
        return [
            "强烈建议回避*ST股票，退市风险极高",
            "如已持有，建议尽快评估是否止损离场",
            "不建议使用杠杆或重仓持有",
            "关注公司是否发布重组或摘帽公告",
        ]
    elif st_type == "ST":
        return [
            "建议谨慎参与ST股票交易",
            "如参与，仓位控制在总资产的5%以内",
            "设置严格的止损线（建议-5%）",
            "关注公司基本面改善和摘帽进展",
        ]
    return []


def filter_st_stocks(symbols):
    """
    过滤ST股票，返回正常股票列表和ST股票列表

    参数:
        symbols: 股票代码列表

    返回: {"正常股票": [...], "ST股票": [...], "*ST股票": [...]}
    """
    normal = []
    st_list = []
    xst_list = []
    unknown = []

    for symbol in symbols:
        st_type = is_st_stock(symbol)
        if st_type == "正常":
            normal.append(symbol)
        elif st_type == "ST":
            st_list.append(symbol)
        elif st_type == "*ST":
            xst_list.append(symbol)
        else:
            unknown.append(symbol)

    return {
        "正常股票": normal,
        "ST股票": st_list,
        "*ST股票": xst_list,
        "未知": unknown,
        "总计": len(symbols),
        "ST占比": f"{(len(st_list) + len(xst_list)) / len(symbols) * 100:.1f}%" if symbols else "0%",
    }


def get_price_limit(symbol):
    """
    获取股票涨跌幅限制

    参数:
        symbol: 股票代码

    返回: 涨跌幅限制百分比
    """
    st_type = is_st_stock(symbol)

    if symbol.startswith("688"):
        return 20.0
    elif symbol.startswith("300") or symbol.startswith("301"):
        return 20.0
    elif st_type in ("ST", "*ST"):
        return 5.0
    else:
        return 10.0


def calculate_st_risk_score(symbol, days=250):
    """
    计算ST股票风险评分

    参数:
        symbol: 股票代码
        days: 分析天数

    返回: 风险评分报告
    """
    st_type = is_st_stock(symbol)
    if st_type == "正常":
        return {
            "股票代码": symbol,
            "ST类型": "正常",
            "风险评分": 0,
            "风险等级": "正常",
            "说明": "该股票为正常交易股票",
        }

    df = get_stock_kline(symbol, days=days)
    if df is None or len(df) < 20:
        return {
            "股票代码": symbol,
            "ST类型": st_type,
            "风险评分": 80,
            "风险等级": "高",
            "说明": "数据不足，默认高风险",
        }

    close = df['close']
    amount = df['amount']

    score = 0
    details = []

    if st_type == "*ST":
        score += 30
        details.append("*ST退市风险: +30分")
    elif st_type == "ST":
        score += 15
        details.append("ST特别处理: +15分")

    recent_return = (close.iloc[-1] / close.iloc[-20] - 1) * 100 if len(close) >= 20 else 0
    if recent_return < -20:
        score += 20
        details.append(f"近20日跌幅{recent_return:.1f}%: +20分")
    elif recent_return < -10:
        score += 10
        details.append(f"近20日跌幅{recent_return:.1f}%: +10分")

    avg_amount = amount.tail(20).mean()
    if avg_amount < 1e7:
        score += 15
        details.append(f"日均成交额{avg_amount/1e4:.0f}万 < 1000万: +15分")
    elif avg_amount < 5e7:
        score += 8
        details.append(f"日均成交额{avg_amount/1e4:.0f}万 < 5000万: +8分")

    returns = close.pct_change().dropna()
    if len(returns) > 20:
        volatility = returns.tail(60).std() * np.sqrt(252) * 100
        if volatility > 60:
            score += 15
            details.append(f"年化波动率{volatility:.1f}% > 60%: +15分")
        elif volatility > 40:
            score += 8
            details.append(f"年化波动率{volatility:.1f}% > 40%: +8分")

    price = close.iloc[-1]
    if price < 2:
        score += 20
        details.append(f"股价{price:.2f} < 2元（面值退市风险）: +20分")
    elif price < 5:
        score += 10
        details.append(f"股价{price:.2f} < 5元: +10分")

    score = min(score, 100)

    if score >= 70:
        level = "极高风险"
    elif score >= 50:
        level = "高风险"
    elif score >= 30:
        level = "中风险"
    else:
        level = "低风险"

    return {
        "股票代码": symbol,
        "ST类型": st_type,
        "风险评分": score,
        "风险等级": level,
        "评分明细": details,
        "当前价格": round(float(close.iloc[-1]), 2),
        "近20日涨跌幅": round(recent_return, 1),
        "日均成交额(万)": round(float(avg_amount) / 1e4, 0),
    }


def batch_st_risk_assessment(symbols, days=250):
    """
    批量ST风险评估

    参数:
        symbols: 股票代码列表
        days: 分析天数

    返回: 批量评估报告
    """
    results = []
    for symbol in symbols:
        risk = calculate_st_risk_score(symbol, days)
        results.append(risk)

    results.sort(key=lambda x: x.get("风险评分", 0), reverse=True)

    high_risk = [r for r in results if r.get("风险等级") in ("极高风险", "高风险")]
    st_count = len([r for r in results if r.get("ST类型") in ("ST", "*ST")])

    return {
        "评估股票数": len(symbols),
        "ST股票数": st_count,
        "高风险股票数": len(high_risk),
        "高风险股票": high_risk,
        "全部结果": results,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def main():
    parser = argparse.ArgumentParser(description="ST/*ST股票特殊处理系统")
    subparsers = parser.add_subparsers(dest='command')

    check_parser = subparsers.add_parser('check', help='检查单只股票ST状态')
    check_parser.add_argument('--symbol', required=True, help='股票代码')

    filter_parser = subparsers.add_parser('filter', help='过滤ST股票')
    filter_parser.add_argument('--symbols', required=True, help='股票代码列表，逗号分隔')

    risk_parser = subparsers.add_parser('risk', help='ST股票风险评估')
    risk_parser.add_argument('--symbol', required=True, help='股票代码')
    risk_parser.add_argument('--days', type=int, default=250, help='分析天数')

    batch_parser = subparsers.add_parser('batch', help='批量ST风险评估')
    batch_parser.add_argument('--symbols', required=True, help='股票代码列表，逗号分隔')
    batch_parser.add_argument('--days', type=int, default=250, help='分析天数')

    limit_parser = subparsers.add_parser('limit', help='查询涨跌幅限制')
    limit_parser.add_argument('--symbol', required=True, help='股票代码')

    args = parser.parse_args()

    if args.command == 'check':
        result = get_st_risk_info(args.symbol)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'filter':
        symbols = [s.strip() for s in args.symbols.split(',')]
        result = filter_st_stocks(symbols)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'risk':
        result = calculate_st_risk_score(args.symbol, args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'batch':
        symbols = [s.strip() for s in args.symbols.split(',')]
        result = batch_st_risk_assessment(symbols, args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'limit':
        limit = get_price_limit(args.symbol)
        st_type = is_st_stock(args.symbol)
        print(json.dumps({
            "股票代码": args.symbol,
            "ST类型": st_type,
            "涨跌幅限制": f"{limit}%",
        }, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
