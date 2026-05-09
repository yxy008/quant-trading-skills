#!/usr/bin/env python3
"""
股东增减持分析系统
支持大股东增减持数据、高管增减持、增减持信号解读、增减持策略
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


# ==================== 股东增减持数据 ====================

def get_shareholder_trade_data():
    """
    获取股东增减持数据

    返回: 增减持数据
    """
    try:
        df = ak.stock_hold_management_detail_em()
    except Exception:
        try:
            df = ak.stock_share_hold_change_em()
        except Exception as e:
            return {"error": f"获取增减持数据失败: {str(e)}"}

    if df is None or len(df) == 0:
        return {"error": "未获取到增减持数据"}

    # 识别列
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if '代码' in col or 'code' in col_lower:
            col_map['代码'] = col
        elif '名称' in col or 'name' in col_lower:
            col_map['名称'] = col
        elif '股东' in col:
            col_map['股东名称'] = col
        elif '变动' in col and '方向' in col:
            col_map['变动方向'] = col
        elif '变动' in col and ('数量' in col or '股' in col):
            col_map['变动数量'] = col
        elif '变动' in col and '比例' in col:
            col_map['变动比例'] = col
        elif '变动' in col and '金额' in col:
            col_map['变动金额'] = col
        elif '日期' in col or 'date' in col_lower:
            col_map['变动日期'] = col
        elif '均价' in col or '价格' in col:
            col_map['成交均价'] = col

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
        "获取时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "增减持记录数": len(df),
        "数据列": list(df.columns),
        "识别字段": col_map,
        "增减持列表": preview,
    }


# ==================== 增减持信号解读 ====================

def shareholder_signal_analysis():
    """
    增减持信号解读
    分析不同增减持行为的信号含义

    返回: 信号解读
    """
    signal_analysis = {
        "增持信号": {
            "大股东大额增持": {
                "信号强度": "极强",
                "含义": "最了解公司的人用真金白银投票，强烈看多",
                "操作建议": "跟随买入，中线持有",
                "注意事项": "需区分真实增持和作秀式增持（金额小/比例低）",
            },
            "高管集体增持": {
                "信号强度": "强",
                "含义": "管理层对公司未来有信心",
                "操作建议": "关注增持金额和人数，金额越大信号越强",
                "注意事项": "个别高管小额增持可能是作秀",
            },
            "公司回购": {
                "信号强度": "强",
                "含义": "公司认为股价被低估",
                "操作建议": "回购金额大且持续时跟随",
                "注意事项": "回购用于股权激励的利好程度低于注销",
            },
        },
        "减持信号": {
            "大股东大额减持": {
                "信号强度": "极强（利空）",
                "含义": "最了解公司的人在卖出，强烈看空",
                "操作建议": "跟随减仓或清仓",
                "注意事项": "需区分是资金需求还是看空公司",
            },
            "高管集体减持": {
                "信号强度": "强（利空）",
                "含义": "管理层对公司前景不乐观",
                "操作建议": "减仓观望",
                "注意事项": "高管减持需要提前公告，有时间窗口",
            },
            "减持比例小": {
                "信号强度": "弱",
                "含义": "可能是个人资金需求，不一定看空",
                "操作建议": "关注后续是否继续减持",
                "注意事项": "单次减持<0.1%影响有限",
            },
        },
    }

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "增减持信号": signal_analysis,
        "增减持判断框架": [
            "增持金额>1000万且占总股本>0.1%：有效增持信号",
            "减持金额>5000万且占总股本>0.5%：有效减持信号",
            "大股东增持>高管增持>员工持股计划",
            "减持原因中'资金需求'比'优化资产配置'更需警惕",
            "连续多次增持/减持比单次更有参考价值",
        ],
        "增减持策略": {
            "增持跟随策略": "大股东增持公告后次日买入，持有1-3个月",
            "减持回避策略": "大股东减持公告后回避该股至少1个月",
            "逆向策略": "大股东增持但股价继续下跌，可能是黄金坑",
        },
    }


# ==================== 高管增减持分析 ====================

def executive_trade_analysis(symbol=None):
    """
    高管增减持分析
    分析高管买卖股票的行为

    参数:
        symbol: 股票代码（可选）

    返回: 高管增减持分析
    """
    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "高管增减持规则": {
            "窗口期限制": "定期报告发布前30日内不得买卖",
            "短线交易限制": "买入后6个月内卖出收益归公司",
            "减持预披露": "大股东减持需提前15个交易日公告",
            "减持额度": "集中竞价连续90日减持不超过1%",
        },
        "高管增减持信号": [
            "高管增持：通常比大股东增持信号更强（更了解经营）",
            "董事长/总经理增持：最强看多信号",
            "财务总监增持：对公司财务真实性有信心",
            "技术总监增持：对公司技术/产品有信心",
            "多名高管同时增持：集体看好信号",
            "高管在股价低位增持：底部信号",
            "高管在股价高位减持：顶部信号",
        ],
        "增减持与股价关系": [
            "增持后1个月：平均超额收益2-5%",
            "减持后1个月：平均超额收益-3-8%",
            "增持金额越大，后续涨幅越大",
            "减持比例越大，后续跌幅越大",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description='股东增减持分析系统')
    subparsers = parser.add_subparsers(dest='command')

    # 增减持数据
    data_parser = subparsers.add_parser('data', help='增减持数据')

    # 信号解读
    signal_parser = subparsers.add_parser('signal', help='增减持信号解读')

    # 高管分析
    exec_parser = subparsers.add_parser('executive', help='高管增减持分析')

    args = parser.parse_args()

    if args.command == 'data':
        result = get_shareholder_trade_data()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'signal':
        result = shareholder_signal_analysis()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'executive':
        result = executive_trade_analysis()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
