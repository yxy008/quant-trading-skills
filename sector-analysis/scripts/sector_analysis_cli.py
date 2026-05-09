#!/usr/bin/env python3
"""
板块分析工具 - sector-analysis (全真实数据版)
所有数据均来自真实接口，不包含任何虚拟模拟数据
"""
import argparse
import json
import sys
import os
from datetime import datetime, timedelta

# 添加agent目录到路径以导入data_utils
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


# 板块成分股结构（市场公认分类，不包含虚拟数据）
SECTOR_STOCKS = {
    "银行": [
        {"代码": "000001", "名称": "平安银行"},
        {"代码": "600036", "名称": "招商银行"},
        {"代码": "601318", "名称": "中国平安"},
        {"代码": "601939", "名称": "建设银行"},
        {"代码": "601398", "名称": "工商银行"}
    ],
    "白酒": [
        {"代码": "600519", "名称": "贵州茅台"},
        {"代码": "000858", "名称": "五粮液"},
        {"代码": "000568", "名称": "泸州老窖"},
        {"代码": "600809", "名称": "山西汾酒"}
    ],
    "新能源": [
        {"代码": "300750", "名称": "宁德时代"},
        {"代码": "002594", "名称": "比亚迪"},
        {"代码": "002466", "名称": "天齐锂业"},
        {"代码": "300014", "名称": "亿纬锂能"}
    ],
    "半导体": [
        {"代码": "300782", "名称": "卓胜微"},
        {"代码": "688981", "名称": "中芯国际"},
        {"代码": "603986", "名称": "兆易创新"}
    ],
    "消费": [
        {"代码": "600887", "名称": "伊利股份"},
        {"代码": "000333", "名称": "美的集团"},
        {"代码": "000651", "名称": "格力电器"}
    ]
}


def get_stock_data(symbol):
    """
    获取单只股票真实数据（K线）
    :param symbol: 6位股票代码
    :return: dict 或 None
    """
    df = get_stock_kline(symbol, days=30)

    if df is not None and not df.empty and len(df) >= 2:
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        change_pct = (float(latest['收盘']) - float(prev['收盘'])) / float(prev['收盘']) * 100

        recent_closes = df['收盘'].tail(20)
        returns = recent_closes.pct_change().dropna()
        volatility = returns.std() * 100 if len(returns) > 5 else 0

        return {
            "代码": symbol,
            "最新价": round(float(latest['收盘']), 2),
            "涨跌幅": round(change_pct, 2),
            "成交量": int(latest.get('成交量', 0)),
            "成交额": round(float(latest.get('成交额', 0)), 2),
            "波动率": round(volatility, 2),
            "数据源": "真实K线数据"
        }

    return None


def get_sector_list():
    """
    获取板块列表及真实涨跌幅
    :return: dict
    """
    sector_list = []
    
    for sector_name, stocks in SECTOR_STOCKS.items():
        # 计算板块平均涨跌幅（真实数据）
        changes = []
        stock_datas = []
        
        for stock in stocks:
            data = get_stock_data(stock['代码'])
            if data:
                data['名称'] = stock['名称']
                stock_datas.append(data)
                changes.append(data['涨跌幅'])
        
        avg_change = round(sum(changes) / len(changes), 2) if changes else 0
        
        sector_list.append({
            "名称": sector_name,
            "平均涨跌幅": avg_change,
            "成分股数量": len(stock_datas),
            "成分股示例": stock_datas[:3]
        })
    
    # 按涨跌幅排序
    sector_list_sorted = sorted(sector_list, key=lambda x: -x["平均涨跌幅"])
    
    return {
        "日期": datetime.now().strftime('%Y-%m-%d'),
        "板块数量": len(sector_list_sorted),
        "板块列表": sector_list_sorted
    }


def get_sector_detail(sector_name):
    """
    获取板块详情（全真实数据）
    :param sector_name: 板块名称
    :return: dict
    """
    if sector_name not in SECTOR_STOCKS:
        return {
            "error": f"未找到板块: {sector_name}",
            "支持板块": list(SECTOR_STOCKS.keys())
        }
        
    # 获取所有成分股真实数据
    stock_list = []
    changes = []
    volatilities = []
    
    for stock in SECTOR_STOCKS[sector_name]:
        data = get_stock_data(stock['代码'])
        if data:
            data['名称'] = stock['名称']
            stock_list.append(data)
            changes.append(data['涨跌幅'])
            if data['波动率'] > 0:
                volatilities.append(data['波动率'])
    
    # 统计真实数据
    avg_change = round(sum(changes) / len(changes), 2) if changes else 0
    avg_volatility = round(sum(volatilities) / len(volatilities), 2) if volatilities else 0
    
    # 找到领涨领跌
    leaders = sorted(stock_list, key=lambda x: -x['涨跌幅'])
    
    return {
        "板块名称": sector_name,
        "板块平均涨跌幅": avg_change,
        "板块平均波动率": avg_volatility,
        "领涨股": leaders[0] if leaders else None,
        "领跌股": leaders[-1] if leaders else None,
        "成分股数量": len(stock_list),
        "成分股数据": stock_list
    }


def compare_sectors(sectors_str):
    """
    对比多个板块（真实数据）
    :param sectors_str: 板块名称，逗号分隔
    :return: dict
    """
    sector_names = [s.strip() for s in sectors_str.split(',')]
    compare_list = []
    
    for name in sector_names:
        if name in SECTOR_STOCKS:
            detail = get_sector_detail(name)
            if "error" not in detail:
                compare_list.append({
                    "名称": detail["板块名称"],
                    "平均涨跌幅": detail["板块平均涨跌幅"],
                    "平均波动率": detail["板块平均波动率"],
                    "成分股数量": detail["成分股数量"],
                    "领涨股": detail["领涨股"]
                })
    
    # 按涨跌幅排序
    compare_list_sorted = sorted(compare_list, key=lambda x: -x["平均涨跌幅"])
    
    return {
        "日期": datetime.now().strftime('%Y-%m-%d'),
        "对比板块数量": len(compare_list_sorted),
        "对比结果": compare_list_sorted
    }


def main():
    parser = argparse.ArgumentParser(description='行业板块分析工具（全真实数据）')
    parser.add_argument('action', choices=['list', 'detail', 'compare'],
                        help='操作类型: list（板块列表）, detail（板块详情）, compare（板块对比）')
    parser.add_argument('--sector', help='板块名称（仅 detail 需要）')
    parser.add_argument('--sectors', help='板块名称，逗号分隔（仅 compare 需要）')
    
    args = parser.parse_args()
    
    try:
        if args.action == 'list':
            data = get_sector_list()
        elif args.action == 'detail':
            if not args.sector:
                print(json.dumps({"error": "需要 --sector 参数"}, ensure_ascii=False, indent=2))
                sys.exit(1)
            data = get_sector_detail(args.sector)
        elif args.action == 'compare':
            if not args.sectors:
                print(json.dumps({"error": "需要 --sectors 参数"}, ensure_ascii=False, indent=2))
                sys.exit(1)
            data = compare_sectors(args.sectors)
        
        print(json.dumps(data, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == '__main__':
    main()
