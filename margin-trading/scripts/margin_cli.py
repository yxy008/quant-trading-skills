#!/usr/bin/env python3
"""
融资融券分析系统
支持两融余额分析、融资买入情绪、融券做空力量、个股两融数据
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


# ==================== 两融余额分析 ====================

def margin_balance_analysis():
    """
    融资融券余额分析
    获取市场整体两融数据

    返回: 两融余额分析
    """
    try:
        df = ak.stock_margin_sz_sh_daily()
    except Exception as e:
        return {"error": f"获取两融数据失败: {str(e)}"}

    if df is None or len(df) == 0:
        return {"error": "未获取到两融数据"}

    # 取最近数据
    recent = df.tail(20)

    # 识别列
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if '日期' in col or 'date' in col_lower:
            col_map['日期'] = col
        elif '融资' in col and '余额' in col:
            col_map['融资余额'] = col
        elif '融券' in col and '余额' in col:
            col_map['融券余额'] = col
        elif '融资' in col and '买入' in col:
            col_map['融资买入额'] = col
        elif '融券' in col and '卖出' in col:
            col_map['融券卖出量'] = col

    recent_data = []
    for i in range(len(recent)):
        row_data = {}
        for meaning, col in col_map.items():
            if col in recent.columns:
                val = recent.iloc[i][col]
                if isinstance(val, (np.floating,)):
                    val = round(float(val), 4)
                elif isinstance(val, float):
                    val = round(val, 4)
                row_data[meaning] = val
        recent_data.append(row_data)

    # 计算关键指标
    if len(recent_data) >= 2:
        latest = recent_data[-1]
        prev = recent_data[-2]

        # 融资余额变化
        if '融资余额' in latest and '融资余额' in prev:
            try:
                margin_change = float(latest['融资余额']) - float(prev['融资余额'])
                margin_change_pct = margin_change / float(prev['融资余额']) * 100
            except (ValueError, TypeError):
                margin_change = 0
                margin_change_pct = 0
        else:
            margin_change = 0
            margin_change_pct = 0
    else:
        margin_change = 0
        margin_change_pct = 0

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "数据列": list(df.columns),
        "识别字段": col_map,
        "最近数据": recent_data,
        "融资余额变化": {
            "变化金额": f"{margin_change:,.0f}元" if abs(margin_change) > 0 else "N/A",
            "变化比例": f"{margin_change_pct:+.2f}%" if margin_change_pct != 0 else "N/A",
        },
        "两融分析框架": {
            "融资余额": {
                "含义": "投资者借钱买股票的总额",
                "上升": "市场做多情绪高涨，杠杆资金入场",
                "下降": "市场谨慎，杠杆资金离场",
                "极值信号": "融资余额创历史新高时警惕过热",
            },
            "融券余额": {
                "含义": "投资者借股票卖出的总额",
                "上升": "做空力量增强，看跌情绪上升",
                "下降": "做空力量减弱",
                "极值信号": "融券余额骤增可能是利空出尽",
            },
            "融资买入额": {
                "含义": "当日融资买入的金额",
                "占成交额比": "融资买入/总成交额，反映杠杆参与度",
                "高比例": ">10%说明杠杆资金活跃",
            },
        },
        "两融情绪指标": [
            "融资余额持续增加+指数上涨：健康上涨",
            "融资余额持续增加+指数滞涨：警惕见顶",
            "融资余额快速下降+指数下跌：恐慌出逃",
            "融资余额下降+指数企稳：可能是底部",
            "融券余额骤增：市场分歧加大",
        ],
    }


# ==================== 个股两融分析 ====================

def stock_margin_analysis(symbol):
    """
    个股融资融券分析

    参数:
        symbol: 股票代码

    返回: 个股两融分析
    """
    try:
        df = ak.stock_margin_detail_sse(date=datetime.now().strftime('%Y%m%d'))
    except Exception:
        try:
            df = ak.stock_margin_underlying_info_szse(date=datetime.now().strftime('%Y%m%d'))
        except Exception as e:
            return {"error": f"获取个股两融数据失败: {str(e)}"}

    if df is None or len(df) == 0:
        return {"error": "未获取到个股两融数据"}

    # 查找目标股票
    target_row = None
    for i in range(len(df)):
        code = str(df.iloc[i].get('标的代码', df.iloc[i].get('证券代码', '')))
        if symbol in code:
            target_row = df.iloc[i]
            break

    if target_row is None:
        return {"error": f"未找到{symbol}的两融数据"}

    return {
        "股票代码": symbol,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "数据列": list(df.columns),
        "个股数据": {str(k): str(v) for k, v in target_row.items()},
        "个股两融分析要点": [
            "融资余额占流通市值比：反映杠杆程度",
            "融资买入额占成交额比：反映短线资金热度",
            "融券余量变化：反映做空力量",
            "融资净买入连续增加：资金持续看好",
            "融券余量骤增：可能有负面信息",
        ],
    }


# ==================== 两融标的分析 ====================

def margin_eligible_stocks():
    """
    两融标的分析
    分析可融资融券的股票范围

    返回: 两融标的分析
    """
    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "两融标的规则": {
            "融资标的": "沪深两市约2000只股票可融资买入",
            "融券标的": "约1000只股票可融券卖出",
            "科创板": "上市首日即可融资融券",
            "创业板": "注册制后上市首日即可融资融券",
        },
        "两融策略": {
            "融资做多": {
                "策略": "看好后市时融资买入",
                "风险": "下跌时亏损放大，需支付利息",
                "利率": "约6%-8%年化",
            },
            "融券做空": {
                "策略": "看空个股时融券卖出",
                "风险": "上涨时亏损无限，融券成本高",
                "成本": "融券费率+利息",
            },
            "配对交易": {
                "策略": "融资买入强势股+融券卖出弱势股",
                "优势": "对冲市场风险，赚取相对收益",
                "风险": "配对错误可能导致双边亏损",
            },
        },
        "两融风险提示": [
            "维持担保比例低于130%会被强制平仓",
            "融资利息按日计算，长期持有成本高",
            "融券标的可能被暂停融券",
            "单只股票融资余额过高可能是风险信号",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description='融资融券分析系统')
    subparsers = parser.add_subparsers(dest='command')

    # 两融余额
    balance_parser = subparsers.add_parser('balance', help='两融余额分析')

    # 个股两融
    stock_parser = subparsers.add_parser('stock', help='个股两融分析')
    stock_parser.add_argument('--symbol', required=True, help='股票代码')

    # 两融标的
    eligible_parser = subparsers.add_parser('eligible', help='两融标的分析')

    args = parser.parse_args()

    if args.command == 'balance':
        result = margin_balance_analysis()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'stock':
        result = stock_margin_analysis(args.symbol)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'eligible':
        result = margin_eligible_stocks()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
