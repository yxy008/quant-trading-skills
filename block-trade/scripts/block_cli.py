#!/usr/bin/env python3
"""
大宗交易分析系统
支持大宗交易数据查询、折溢价分析、机构接盘判断、大宗交易策略
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


# ==================== 大宗交易数据 ====================

def get_block_trade_data(date=None):
    """
    获取大宗交易数据

    参数:
        date: 日期（YYYYMMDD），默认最新

    返回: 大宗交易数据
    """
    if date is None:
        date = datetime.now().strftime('%Y%m%d')

    try:
        df = ak.stock_dzjy_mrmx(symbol='沪深A股', start_date=date, end_date=date)
    except Exception:
        try:
            df = ak.stock_dzjy_mrtj(start_date=date, end_date=date)
        except Exception as e:
            return {"error": f"获取大宗交易数据失败: {str(e)}"}

    if df is None or len(df) == 0:
        return {"error": f"{date}无大宗交易数据（可能非交易日）"}

    # 识别列
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if '代码' in col or 'code' in col_lower:
            col_map['代码'] = col
        elif '名称' in col or 'name' in col_lower:
            col_map['名称'] = col
        elif '成交' in col and '价' in col:
            col_map['成交价'] = col
        elif '成交' in col and ('量' in col or '额' in col):
            col_map['成交额'] = col
        elif '收盘' in col or 'close' in col_lower:
            col_map['收盘价'] = col
        elif '折' in col or '溢价' in col:
            col_map['折溢价'] = col
        elif '买方' in col or 'buyer' in col_lower:
            col_map['买方'] = col
        elif '卖方' in col or 'seller' in col_lower:
            col_map['卖方'] = col

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
        "大宗交易笔数": len(df),
        "数据列": list(df.columns),
        "识别字段": col_map,
        "大宗交易列表": preview,
    }


# ==================== 折溢价分析 ====================

def discount_premium_analysis():
    """
    大宗交易折溢价分析
    分析折溢价背后的信号含义

    返回: 折溢价分析
    """
    analysis = {
        "折价交易（成交价<收盘价）": {
            "常见折价范围": "5%-10%",
            "含义": [
                "卖方急于套现，愿意折价出让",
                "大宗交易流动性补偿",
                "大股东减持常用方式",
                "折价越大，卖方减持意愿越强",
            ],
            "对股价影响": [
                "短期：利空，接盘方可能次日卖出套利",
                "中期：如果买方是机构且锁仓6个月，影响有限",
                "长期：取决于公司基本面",
            ],
        },
        "溢价交易（成交价>收盘价）": {
            "常见溢价范围": "1%-5%",
            "含义": [
                "买方非常看好，愿意溢价接盘",
                "可能是利益输送或市值管理",
                "机构抢筹信号",
                "溢价越大，买方信心越强",
            ],
            "对股价影响": [
                "短期：利好，显示有资金看好",
                "中期：溢价接盘方通常不会短期卖出",
                "长期：溢价买入往往伴随后续利好",
            ],
        },
        "平价交易（成交价=收盘价）": {
            "含义": "正常的大宗交易，买卖双方按市价成交",
            "对股价影响": "中性，需结合买卖双方身份判断",
        },
    }

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "折溢价分析": analysis,
        "大宗交易选股信号": [
            "溢价>3%且买方为机构：强烈看多信号",
            "折价>10%且卖方为大股东：减持信号，回避",
            "连续多日大宗交易折价买入：可能是机构建仓",
            "大宗交易金额>1亿且溢价：重大利好",
            "同一营业部频繁大宗交易：可能是过桥或利益输送",
        ],
        "大宗交易策略": {
            "跟随策略": "溢价大宗交易次日买入，持有1-3个月",
            "回避策略": "大额折价大宗交易后回避该股",
            "套利策略": "折价>8%时，次日开盘买入（接盘方可能砸盘）",
        },
    }


# ==================== 大宗交易统计 ====================

def block_trade_statistics(days=10):
    """
    大宗交易统计分析
    统计近期大宗交易的规律

    参数:
        days: 统计天数

    返回: 统计分析
    """
    all_stocks = {}
    total_trades = 0

    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
        try:
            df = ak.stock_dzjy_mrmx(symbol='沪深A股', start_date=date, end_date=date)
            if df is not None and len(df) > 0:
                total_trades += len(df)
                for j in range(len(df)):
                    code = str(df.iloc[j].get('证券代码', df.iloc[j].get('代码', '')))
                    name = str(df.iloc[j].get('证券简称', df.iloc[j].get('名称', '')))
                    if code not in all_stocks:
                        all_stocks[code] = {"名称": name, "交易次数": 0}
                    all_stocks[code]["交易次数"] += 1
            time.sleep(0.5)
        except Exception:
            continue

    sorted_stocks = sorted(all_stocks.items(), key=lambda x: x[1]["交易次数"], reverse=True)

    frequent = []
    for code, info in sorted_stocks[:20]:
        if info["交易次数"] >= 2:
            frequent.append({
                "代码": code,
                "名称": info["名称"],
                "交易次数": info["交易次数"],
            })

    return {
        "统计时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "统计天数": days,
        "总交易笔数": total_trades,
        "频繁大宗交易": frequent,
        "分析": [
            "频繁大宗交易的股票可能有重大事项",
            "连续折价大宗交易：大股东持续减持",
            "连续溢价大宗交易：机构持续建仓",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description='大宗交易分析系统')
    subparsers = parser.add_subparsers(dest='command')

    # 大宗交易数据
    data_parser = subparsers.add_parser('data', help='大宗交易数据')
    data_parser.add_argument('--date', help='日期(YYYYMMDD)')

    # 折溢价分析
    dp_parser = subparsers.add_parser('discount', help='折溢价分析')

    # 统计分析
    stat_parser = subparsers.add_parser('stats', help='大宗交易统计')
    stat_parser.add_argument('--days', type=int, default=10, help='统计天数')

    args = parser.parse_args()

    if args.command == 'data':
        result = get_block_trade_data(args.date)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'discount':
        result = discount_premium_analysis()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'stats':
        result = block_trade_statistics(args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
