#!/usr/bin/env python3
"""
龙虎榜分析系统
支持龙虎榜数据获取、游资席位识别、机构动向分析、资金博弈分析
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


# ==================== 龙虎榜数据获取 ====================

def get_dragon_tiger_list(date=None):
    """
    获取龙虎榜数据

    参数:
        date: 日期（YYYYMMDD），默认最新

    返回: 龙虎榜数据
    """
    if date is None:
        date = datetime.now().strftime('%Y%m%d')

    try:
        df = ak.stock_lhb_detail_em(date=date)
    except Exception as e:
        return {"error": f"获取龙虎榜数据失败: {str(e)}"}

    if df is None or len(df) == 0:
        return {"error": f"{date}无龙虎榜数据（可能非交易日）"}

    # 识别列
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if '代码' in col or 'code' in col_lower:
            col_map['代码'] = col
        elif '名称' in col or 'name' in col_lower:
            col_map['名称'] = col
        elif '收盘' in col or 'close' in col_lower:
            col_map['收盘价'] = col
        elif '涨跌' in col or 'change' in col_lower:
            col_map['涨跌幅'] = col
        elif '成交' in col and '额' in col:
            col_map['成交额'] = col
        elif '净买' in col or '净买' in col:
            col_map['净买入'] = col
        elif '买入' in col and '额' in col:
            col_map['买入额'] = col
        elif '卖出' in col and '额' in col:
            col_map['卖出额'] = col
        elif '原因' in col or 'reason' in col_lower:
            col_map['上榜原因'] = col

    preview = []
    for i in range(min(30, len(df))):
        row_data = {}
        for meaning, col in col_map.items():
            if col in df.columns:
                val = df.iloc[i][col]
                if isinstance(val, (np.floating,)):
                    val = round(float(val), 4)
                elif isinstance(val, float):
                    val = round(val, 4)
                row_data[meaning] = val
        preview.append(row_data)

    return {
        "日期": date,
        "获取时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "上榜股票数": len(df),
        "数据列": list(df.columns),
        "识别字段": col_map,
        "龙虎榜列表": preview,
    }


# ==================== 游资席位分析 ====================

def hot_money_analysis():
    """
    游资席位分析
    识别知名游资席位及其操作风格

    返回: 游资席位分析
    """
    famous_seats = [
        {
            "席位名称": "中信证券上海分公司",
            "别名": "中信上海分",
            "风格": "趋势龙头，偏好大市值科技股",
            "资金量级": "10亿+",
            "操作特点": "锁仓为主，持股周期长",
        },
        {
            "席位名称": "华泰证券总部",
            "别名": "华泰总部",
            "风格": "量化交易，高频操作",
            "资金量级": "5亿+",
            "操作特点": "快进快出，不恋战",
        },
        {
            "席位名称": "国泰君安证券上海分公司",
            "别名": "国君上海分",
            "风格": "题材挖掘，偏好政策利好",
            "资金量级": "5亿+",
            "操作特点": "提前布局，利好兑现出货",
        },
        {
            "席位名称": "东方财富证券拉萨团结路",
            "别名": "拉萨天团",
            "风格": "散户集中营，跟风操作",
            "资金量级": "分散",
            "操作特点": "追涨杀跌，反向指标",
        },
        {
            "席位名称": "深股通专用",
            "别名": "北向资金",
            "风格": "价值投资，偏好白马蓝筹",
            "资金量级": "百亿+",
            "操作特点": "长期持有，逆向布局",
        },
        {
            "席位名称": "机构专用",
            "别名": "机构席位",
            "风格": "基本面驱动，偏好业绩增长",
            "资金量级": "不确定",
            "操作特点": "调研后买入，持股周期3-12个月",
        },
    ]

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "知名游资席位": famous_seats,
        "龙虎榜分析要点": [
            "买一至买五：买入金额最大的5个席位",
            "卖一至卖五：卖出金额最大的5个席位",
            "净买入=买入总额-卖出总额",
            "机构席位净买入：机构看好信号",
            "游资席位净买入：短期炒作信号",
            "拉萨天团净买入：散户跟风，谨慎追高",
        ],
        "龙虎榜选股策略": [
            "机构席位净买入>5000万且无游资卖出：中线关注",
            "知名游资净买入+机构净买入共振：短线爆发力强",
            "买一金额远超买二：主力控盘度高",
            "卖出席位全是拉萨天团：散户恐慌出逃，可能是洗盘",
            "同一席位买卖金额接近：对倒出货嫌疑",
        ],
    }


# ==================== 龙虎榜统计 ====================

def dragon_tiger_statistics(days=5):
    """
    龙虎榜统计分析
    统计近期龙虎榜上榜股票的规律

    参数:
        days: 统计天数

    返回: 统计分析
    """
    all_stocks = {}
    total_records = 0

    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
        try:
            df = ak.stock_lhb_detail_em(date=date)
            if df is not None and len(df) > 0:
                total_records += len(df)
                for j in range(len(df)):
                    code = str(df.iloc[j].get('代码', ''))
                    name = str(df.iloc[j].get('名称', ''))
                    if code not in all_stocks:
                        all_stocks[code] = {"名称": name, "上榜次数": 0}
                    all_stocks[code]["上榜次数"] += 1
            time.sleep(0.5)
        except Exception:
            continue

    # 排序
    sorted_stocks = sorted(all_stocks.items(), key=lambda x: x[1]["上榜次数"], reverse=True)

    frequent_stocks = []
    for code, info in sorted_stocks[:20]:
        if info["上榜次数"] >= 2:
            frequent_stocks.append({
                "代码": code,
                "名称": info["名称"],
                "上榜次数": info["上榜次数"],
                "频率": f"{info['上榜次数']}/{days}天",
            })

    return {
        "统计时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "统计天数": days,
        "总上榜记录": total_records,
        "频繁上榜股票": frequent_stocks,
        "分析": [
            "连续上榜的股票通常是市场热点",
            "上榜次数多但涨幅不大的可能是主力吸筹",
            "上榜次数多且涨幅大的需警惕高位出货",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description='龙虎榜分析系统')
    subparsers = parser.add_subparsers(dest='command')

    # 龙虎榜列表
    list_parser = subparsers.add_parser('list', help='龙虎榜数据')
    list_parser.add_argument('--date', help='日期(YYYYMMDD)')

    # 游资分析
    hot_parser = subparsers.add_parser('hot-money', help='游资席位分析')

    # 统计分析
    stat_parser = subparsers.add_parser('stats', help='龙虎榜统计')
    stat_parser.add_argument('--days', type=int, default=5, help='统计天数')

    args = parser.parse_args()

    if args.command == 'list':
        result = get_dragon_tiger_list(args.date)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'hot-money':
        result = hot_money_analysis()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'stats':
        result = dragon_tiger_statistics(args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
