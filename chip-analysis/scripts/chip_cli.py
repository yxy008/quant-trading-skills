#!/usr/bin/env python3
"""
筹码集中度分析系统
支持股东人数变化、筹码集中度计算、主力吸筹/出货判断、筹码分布分析
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


# ==================== 股东人数变化 ====================

def shareholder_count_analysis(symbol):
    """
    股东人数变化分析
    股东人数减少=筹码集中，增加=筹码分散

    参数:
        symbol: 股票代码

    返回: 股东人数分析
    """
    try:
        df = ak.stock_hold_num_cninfo(symbol=symbol)
    except Exception:
        try:
            df = ak.stock_hold_num_em(symbol=symbol)
        except Exception as e:
            return {"error": f"获取股东人数数据失败: {str(e)}"}

    if df is None or len(df) < 2:
        return {"error": f"{symbol}股东人数数据不足"}

    # 识别列
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if '日期' in col or 'date' in col_lower or '时间' in col:
            col_map['日期'] = col
        elif '股东' in col and ('人数' in col or '户数' in col):
            col_map['股东人数'] = col
        elif '人均' in col and '持股' in col:
            col_map['人均持股'] = col

    # 取最近两期
    if len(df) >= 2:
        latest = df.iloc[0]
        prev = df.iloc[1]

        latest_count = None
        prev_count = None

        for meaning, col in col_map.items():
            if meaning == '股东人数':
                try:
                    latest_count = float(latest[col])
                    prev_count = float(prev[col])
                except (ValueError, TypeError):
                    pass

        if latest_count and prev_count and prev_count > 0:
            change = (latest_count - prev_count) / prev_count * 100
            if change < -5:
                trend = "筹码快速集中（主力吸筹）"
                signal = "强烈看多"
            elif change < -2:
                trend = "筹码温和集中"
                signal = "看多"
            elif change > 5:
                trend = "筹码快速分散（主力出货）"
                signal = "强烈看空"
            elif change > 2:
                trend = "筹码温和分散"
                signal = "看空"
            else:
                trend = "筹码基本稳定"
                signal = "中性"
        else:
            change = 0
            trend = "无法判断"
            signal = "N/A"
    else:
        change = 0
        trend = "数据不足"
        signal = "N/A"

    # 历史数据
    history = []
    for i in range(min(10, len(df))):
        row_data = {}
        for meaning, col in col_map.items():
            if col in df.columns:
                val = df.iloc[i][col]
                if isinstance(val, (np.floating,)):
                    val = round(float(val), 4)
                elif isinstance(val, float):
                    val = round(val, 4)
                row_data[meaning] = val
        history.append(row_data)

    return {
        "股票代码": symbol,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "数据列": list(df.columns),
        "识别字段": col_map,
        "最新股东人数": latest_count,
        "上期股东人数": prev_count,
        "变化率": f"{change:+.1f}%" if change else "N/A",
        "筹码趋势": trend,
        "信号": signal,
        "历史数据": history,
        "分析说明": [
            "股东人数减少=筹码从散户向主力集中=看多",
            "股东人数增加=筹码从主力向散户分散=看空",
            "连续3期以上减少：主力持续吸筹",
            "股东人数减少+股价横盘：主力压价吸筹",
            "股东人数增加+股价上涨：主力边拉边出",
        ],
    }


# ==================== 筹码集中度评估 ====================

def chip_concentration_assessment(symbol):
    """
    筹码集中度综合评估
    结合多维度判断筹码状态

    参数:
        symbol: 股票代码

    返回: 筹码集中度评估
    """
    # 先获取股东人数
    holder_result = shareholder_count_analysis(symbol)

    # 获取K线数据辅助判断
    try:
        from data_utils import get_stock_kline
        df = get_stock_kline(symbol, days=250)
        if df is not None and len(df) >= 60:
            close_col = '收盘' if '收盘' in df.columns else 'close'
            close = pd.to_numeric(df[close_col], errors='coerce').dropna()
            volume_col = '成交量' if '成交量' in df.columns else 'volume'
            volume = pd.to_numeric(df[volume_col], errors='coerce').dropna()

            # 量价分析
            recent_close = close.tail(60)
            recent_volume = volume.tail(60)

            # 缩量横盘=筹码锁定
            vol_ratio = float(recent_volume.tail(20).mean() / recent_volume.head(20).mean()) if len(recent_volume) >= 40 else 1
            price_range = float((recent_close.max() - recent_close.min()) / recent_close.mean() * 100)

            if vol_ratio < 0.7 and price_range < 15:
                volume_signal = "缩量横盘，筹码锁定良好"
            elif vol_ratio > 1.5 and price_range > 25:
                volume_signal = "放量宽幅震荡，筹码不稳定"
            else:
                volume_signal = "量价正常"
        else:
            volume_signal = "数据不足"
    except Exception:
        volume_signal = "数据不足"

    # 综合评分
    score = 50
    signals = []

    if "error" not in holder_result:
        change_str = holder_result.get("变化率", "N/A")
        if change_str != "N/A":
            try:
                change_val = float(change_str.replace("%", "").replace("+", ""))
                if change_val < -5:
                    score += 25
                    signals.append("股东人数大幅减少+25")
                elif change_val < -2:
                    score += 15
                    signals.append("股东人数温和减少+15")
                elif change_val > 5:
                    score -= 25
                    signals.append("股东人数大幅增加-25")
                elif change_val > 2:
                    score -= 15
                    signals.append("股东人数温和增加-15")
            except ValueError:
                pass

    if "缩量横盘" in volume_signal:
        score += 10
        signals.append("缩量横盘+10")
    elif "不稳定" in volume_signal:
        score -= 10
        signals.append("放量震荡-10")

    if score >= 70:
        grade = "高度集中"
        advice = "主力控盘度高，适合中线持有"
    elif score >= 55:
        grade = "相对集中"
        advice = "筹码在集中过程中，可关注"
    elif score >= 40:
        grade = "一般"
        advice = "筹码分布中性，需结合其他指标"
    elif score >= 25:
        grade = "相对分散"
        advice = "筹码在分散过程中，谨慎"
    else:
        grade = "高度分散"
        advice = "散户主导，回避"

    return {
        "股票代码": symbol,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "股东人数分析": holder_result if "error" not in holder_result else {"error": holder_result["error"]},
        "量价信号": volume_signal,
        "综合评分": f"{score}/100",
        "筹码集中度": grade,
        "操作建议": advice,
        "评分明细": signals,
    }


# ==================== 筹码分布理论 ====================

def chip_distribution_theory():
    """
    筹码分布理论知识
    介绍筹码分析的核心概念和方法

    返回: 筹码分布理论
    """
    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "筹码分析核心概念": {
            "筹码集中": {
                "定义": "少数人持有大部分流通股",
                "特征": "股东人数减少，人均持股增加",
                "含义": "主力控盘，股价易涨难跌",
            },
            "筹码分散": {
                "定义": "多数人持有流通股",
                "特征": "股东人数增加，人均持股减少",
                "含义": "散户主导，股价易跌难涨",
            },
            "筹码锁定": {
                "定义": "持有者不愿卖出",
                "特征": "缩量横盘或缩量上涨",
                "含义": "主力锁仓，后市看涨",
            },
            "筹码松动": {
                "定义": "持有者开始卖出",
                "特征": "放量滞涨或放量下跌",
                "含义": "主力出货，后市看跌",
            },
        },
        "筹码分析指标": [
            "股东人数变化率：最直接的筹码指标",
            "人均持股金额：人均持股*股价，越高越集中",
            "前十大股东持股比例：>60%为高度集中",
            "户均持股比例变化：增加=集中，减少=分散",
            "换手率：低位低换手=筹码锁定，高位高换手=筹码松动",
        ],
        "筹码选股策略": [
            "股东人数连续3期减少>5%：中线关注",
            "股东人数减少+股价在年线附近：最佳买点",
            "股东人数减少+成交量萎缩：主力锁仓",
            "股东人数增加+成交量放大：主力出货",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description='筹码集中度分析系统')
    subparsers = parser.add_subparsers(dest='command')

    # 股东人数
    holder_parser = subparsers.add_parser('holder', help='股东人数分析')
    holder_parser.add_argument('--symbol', required=True, help='股票代码')

    # 筹码评估
    assess_parser = subparsers.add_parser('assess', help='筹码集中度评估')
    assess_parser.add_argument('--symbol', required=True, help='股票代码')

    # 筹码理论
    theory_parser = subparsers.add_parser('theory', help='筹码分布理论')

    args = parser.parse_args()

    if args.command == 'holder':
        result = shareholder_count_analysis(args.symbol)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'assess':
        result = chip_concentration_assessment(args.symbol)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'theory':
        result = chip_distribution_theory()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
