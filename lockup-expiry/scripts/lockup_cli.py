#!/usr/bin/env python3
"""
限售股解禁分析系统
支持解禁数据查询、解禁影响评估、解禁前后策略、大额解禁预警
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


# ==================== 解禁数据查询 ====================

def get_lockup_expiry_data(month=None):
    """
    获取限售股解禁数据

    参数:
        month: 月份（YYYY-MM），默认当前月

    返回: 解禁数据
    """
    if month is None:
        month = datetime.now().strftime('%Y-%m')

    try:
        df = ak.stock_restricted_release_queue_sse()
    except Exception:
        try:
            df = ak.stock_restricted_release_detail_em(date=datetime.now().strftime('%Y%m%d'))
        except Exception as e:
            return {"error": f"获取解禁数据失败: {str(e)}"}

    if df is None or len(df) == 0:
        return {"error": "未获取到解禁数据"}

    # 识别列
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if '代码' in col or 'code' in col_lower:
            col_map['代码'] = col
        elif '名称' in col or 'name' in col_lower:
            col_map['名称'] = col
        elif '解禁' in col and '日' in col:
            col_map['解禁日期'] = col
        elif '解禁' in col and ('数量' in col or '股' in col):
            col_map['解禁数量'] = col
        elif '解禁' in col and ('市值' in col or '金额' in col):
            col_map['解禁市值'] = col
        elif '占总股' in col or '比例' in col:
            col_map['占总股本'] = col
        elif '类型' in col or 'type' in col_lower:
            col_map['解禁类型'] = col

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
        "查询月份": month,
        "获取时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "解禁股票数": len(df),
        "数据列": list(df.columns),
        "识别字段": col_map,
        "解禁列表": preview,
    }


# ==================== 解禁影响评估 ====================

def lockup_impact_analysis(symbol=None):
    """
    解禁影响评估
    分析解禁对股价的影响

    参数:
        symbol: 股票代码（可选）

    返回: 解禁影响分析
    """
    # 解禁类型与影响
    impact_factors = {
        "首发原股东限售股份": {
            "减持意愿": "中高",
            "影响程度": "中高",
            "说明": "原始股东成本极低，解禁后减持动力强",
        },
        "定向增发机构配售股份": {
            "减持意愿": "高",
            "影响程度": "高",
            "说明": "定增机构有退出需求，解禁后大概率减持",
        },
        "股权激励限售股份": {
            "减持意愿": "中",
            "影响程度": "中低",
            "说明": "高管和员工持股，减持相对有序",
        },
        "追加承诺限售股份": {
            "减持意愿": "低",
            "影响程度": "低",
            "说明": "大股东自愿锁定，减持意愿低",
        },
        "首发战略配售股份": {
            "减持意愿": "中",
            "影响程度": "中",
            "说明": "战略投资者，可能长期持有",
        },
    }

    # 解禁比例影响
    ratio_impact = [
        {"解禁比例": "<1%", "影响": "轻微，市场可消化"},
        {"解禁比例": "1%-5%", "影响": "中等，短期承压"},
        {"解禁比例": "5%-10%", "影响": "较大，可能持续下跌"},
        {"解禁比例": "10%-30%", "影响": "重大，需高度警惕"},
        {"解禁比例": ">30%", "影响": "极大，可能改变供需格局"},
    ]

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "解禁类型影响": impact_factors,
        "解禁比例影响": ratio_impact,
        "解禁前后股价规律": [
            "解禁前1-2周：股价通常承压，市场提前消化利空",
            "解禁当日：可能出现大幅波动",
            "解禁后1-2周：实际减持压力释放，股价可能企稳",
            "解禁后1-3月：如果减持力度大，股价持续走弱",
        ],
        "解禁应对策略": {
            "持仓者": [
                "解禁前1-2周减仓或清仓",
                "关注公司是否发布不减持承诺",
                "解禁后观察实际减持情况再决定",
            ],
            "空仓者": [
                "解禁前不急于买入，等待利空释放",
                "解禁后大跌可能是抄底机会（需确认基本面）",
                "关注大宗交易承接情况",
            ],
        },
        "解禁预警规则": [
            "解禁比例>10%且为定增解禁：红色预警",
            "解禁比例>5%且股价处于高位：橙色预警",
            "解禁比例>1%且近期有减持公告：黄色预警",
            "大股东承诺不减持：可降低预警级别",
        ],
    }


# ==================== 大额解禁预警 ====================

def large_lockup_alert(days=30):
    """
    大额解禁预警
    预警未来N天内的大额解禁

    参数:
        days: 预警天数

    返回: 大额解禁预警
    """
    try:
        df = ak.stock_restricted_release_detail_em(date=datetime.now().strftime('%Y%m%d'))
    except Exception as e:
        return {"error": f"获取解禁数据失败: {str(e)}"}

    if df is None or len(df) == 0:
        return {"error": "未获取到解禁数据"}

    alerts = []

    for i in range(len(df)):
        row = df.iloc[i]

        # 尝试获取解禁比例
        ratio_col = None
        for col in df.columns:
            if '比例' in str(col) or '占比' in str(col):
                ratio_col = col
                break

        if ratio_col:
            try:
                ratio = float(row[ratio_col])
            except (ValueError, TypeError):
                ratio = 0
        else:
            ratio = 0

        if ratio >= 5:
            code = str(row.get('代码', row.get('股票代码', '')))
            name = str(row.get('名称', row.get('股票名称', '')))
            alerts.append({
                "代码": code,
                "名称": name,
                "解禁比例": f"{ratio:.1f}%",
                "预警级别": "红色" if ratio >= 10 else "橙色",
            })

    alerts.sort(key=lambda x: float(x["解禁比例"].replace("%", "")), reverse=True)

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "预警天数": f"未来{days}天",
        "预警数量": len(alerts),
        "大额解禁预警": alerts[:20],
        "操作建议": [
            "红色预警股票建议在解禁前减仓或清仓",
            "橙色预警股票建议控制仓位，设置止损",
            "关注解禁后是否有大宗交易承接",
            "基本面优秀的公司解禁可能是买入机会",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description='限售股解禁分析系统')
    subparsers = parser.add_subparsers(dest='command')

    # 解禁数据
    data_parser = subparsers.add_parser('data', help='解禁数据查询')
    data_parser.add_argument('--month', help='月份(YYYY-MM)')

    # 影响评估
    impact_parser = subparsers.add_parser('impact', help='解禁影响评估')

    # 大额预警
    alert_parser = subparsers.add_parser('alert', help='大额解禁预警')
    alert_parser.add_argument('--days', type=int, default=30, help='预警天数')

    args = parser.parse_args()

    if args.command == 'data':
        result = get_lockup_expiry_data(args.month)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'impact':
        result = lockup_impact_analysis()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'alert':
        result = large_lockup_alert(args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
