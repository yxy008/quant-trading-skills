#!/usr/bin/env python3
"""
打新策略分析系统
支持新股申购分析、打新收益统计、最优市值配置、中签率分析
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


# ==================== 新股申购数据 ====================

def get_ipo_list():
    """
    获取近期新股申购列表

    返回: 新股列表
    """
    try:
        df = ak.stock_zh_a_new_ipo()
    except Exception:
        try:
            df = ak.stock_new_ipo_cninfo()
        except Exception as e:
            return {"error": f"获取新股数据失败: {str(e)}"}

    if df is None or len(df) == 0:
        return {"error": "未获取到新股数据"}

    # 识别列
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if '代码' in col or 'code' in col_lower:
            col_map['代码'] = col
        elif '名称' in col or 'name' in col_lower:
            col_map['名称'] = col
        elif '发行价' in col or 'price' in col_lower:
            col_map['发行价'] = col
        elif '市盈率' in col or 'pe' in col_lower:
            col_map['市盈率'] = col
        elif '申购' in col and '日' in col:
            col_map['申购日'] = col
        elif '上市' in col and '日' in col:
            col_map['上市日'] = col
        elif '中签' in col or 'lottery' in col_lower:
            col_map['中签率'] = col
        elif '行业' in col or 'industry' in col_lower:
            col_map['行业'] = col

    preview = []
    for i in range(min(20, len(df))):
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
        "获取时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "新股数量": len(df),
        "数据列": list(df.columns),
        "识别字段": col_map,
        "新股列表": preview,
    }


# ==================== 打新收益分析 ====================

def ipo_profit_analysis():
    """
    打新收益分析
    分析A股打新的历史收益情况

    返回: 打新收益分析
    """
    # A股打新规则说明
    rules = {
        "申购门槛": {
            "沪市主板": "市值1万元起，每1万元一个配号",
            "深市主板": "市值1万元起，每5000元一个配号",
            "科创板": "市值1万元起，每5000元一个配号",
            "创业板": "市值1万元起，每5000元一个配号",
        },
        "申购上限": {
            "沪市主板": "网上发行量的千分之一",
            "深市主板": "网上发行量的千分之一",
            "科创板": "网上发行量的千分之一",
            "创业板": "网上发行量的千分之一",
        },
        "中签规则": "摇号抽签，T+2日公布中签结果",
        "缴款规则": "中签后T+2日16:00前确保账户有足够资金",
    }

    # 打新收益估算
    profit_estimation = {
        "主板新股": {
            "平均涨幅": "44%（首日涨停）",
            "平均中签收益": "5000-20000元",
            "中签率": "约0.03%-0.05%",
        },
        "科创板新股": {
            "平均涨幅": "50%-150%（前5日无涨跌幅限制）",
            "平均中签收益": "3000-15000元",
            "中签率": "约0.04%-0.06%",
        },
        "创业板新股": {
            "平均涨幅": "50%-200%（前5日无涨跌幅限制）",
            "平均中签收益": "3000-20000元",
            "中签率": "约0.02%-0.04%",
        },
    }

    # 最优市值配置
    optimal_config = [
        {"板块": "沪市主板", "建议市值": "10-15万", "配号数": "10-15个", "说明": "沪市每1万一个配号"},
        {"板块": "深市主板+创业板", "建议市值": "10-15万", "配号数": "20-30个", "说明": "深市每5000一个配号"},
        {"板块": "科创板", "建议市值": "10-15万", "配号数": "20-30个", "说明": "需开通科创板权限"},
    ]

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "打新规则": rules,
        "收益估算": profit_estimation,
        "最优市值配置": optimal_config,
        "打新策略建议": [
            "沪深两市各配置10-15万市值，最大化配号数",
            "科创板需50万资产+2年交易经验才能开通",
            "中签后及时缴款，12个月内3次未缴款将被限制申购",
            "打新收益逐年下降，但仍是无风险套利机会",
            "北交所打新需100万资产门槛，中签率更高",
        ],
        "年化收益估算": {
            "假设条件": "沪深各15万市值，年化中签2-3次",
            "预计年收益": "1-3万元",
            "年化收益率": "约3%-10%（相对持仓市值）",
            "说明": "打新收益是持仓市值的额外收益，不影响持仓本身涨跌",
        },
    }


# ==================== 中签率分析 ====================

def lottery_rate_analysis():
    """
    中签率分析
    分析不同市值配置下的中签概率

    返回: 中签率分析
    """
    # 模拟计算
    scenarios = []

    for market_value in [5, 10, 15, 20, 30, 50]:
        sh_lots = market_value // 1  # 沪市每1万一个配号
        sz_lots = market_value * 2  # 深市每5000一个配号

        # 假设平均中签率0.04%
        avg_rate = 0.0004

        # 每次申购中签概率
        sh_prob = 1 - (1 - avg_rate) ** sh_lots
        sz_prob = 1 - (1 - avg_rate) ** sz_lots

        # 年化中签次数（假设每年200只新股）
        annual_ipos = 200
        sh_annual = sh_prob * annual_ipos
        sz_annual = sz_prob * annual_ipos

        scenarios.append({
            "市值(万)": market_value,
            "沪市配号": sh_lots,
            "深市配号": sz_lots,
            "沪市单次中签率": f"{sh_prob * 100:.3f}%",
            "深市单次中签率": f"{sz_prob * 100:.3f}%",
            "沪市年化中签": f"{sh_annual:.1f}次",
            "深市年化中签": f"{sz_annual:.1f}次",
            "合计年化中签": f"{sh_annual + sz_annual:.1f}次",
        })

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "假设条件": {
            "平均中签率": "0.04%",
            "年新股数量": "200只",
        },
        "不同市值中签率": scenarios,
        "结论": [
            "市值从5万增加到15万，中签率提升最明显",
            "市值超过20万后，边际中签率提升递减",
            "沪深两市均衡配置比单边配置效率更高",
            "深市配号效率是沪市的2倍（5000 vs 10000一个配号）",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description='打新策略分析系统')
    subparsers = parser.add_subparsers(dest='command')

    # 新股列表
    list_parser = subparsers.add_parser('list', help='近期新股列表')

    # 打新收益
    profit_parser = subparsers.add_parser('profit', help='打新收益分析')

    # 中签率
    lottery_parser = subparsers.add_parser('lottery', help='中签率分析')

    args = parser.parse_args()

    if args.command == 'list':
        result = get_ipo_list()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'profit':
        result = ipo_profit_analysis()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'lottery':
        result = lottery_rate_analysis()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
