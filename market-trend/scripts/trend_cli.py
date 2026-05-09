#!/usr/bin/env python3
"""
大盘趋势判断工具 - market-trend
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

from data_utils import get_index_kline


def analyze_trend(df, index_name):
    """分析单指数趋势，适配中文列名"""
    if df is None or len(df) < 60:
        return {"指数": index_name, "趋势": "未知", "说明": "数据不足"}

    close = df['收盘']

    # 计算均线
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    ma120 = close.rolling(120).mean().iloc[-1] if len(close) >= 120 else ma60
    ma250 = close.rolling(250).mean().iloc[-1] if len(close) >= 250 else ma120
    
    latest = close.iloc[-1]
    
    # 计算20日涨跌幅
    change_20 = (latest / close.iloc[-20] - 1) * 100 if len(close)>=20 else 0
    change_60 = (latest / close.iloc[-60] - 1) * 100 if len(close)>=60 else 0
    
    # 趋势判断规则
    bull_score = 0
    bear_score = 0
    
    if latest > ma20:
        bull_score += 1
    elif latest < ma20:
        bear_score += 1
        
    if latest > ma60:
        bull_score += 1
    elif latest < ma60:
        bear_score += 1
        
    if ma20 > ma60:
        bull_score += 1
    elif ma20 < ma60:
        bear_score += 1
        
    if change_20 > 5:
        bull_score += 1
    elif change_20 < -5:
        bear_score += 1
        
    if change_60 > 10:
        bull_score += 2
    elif change_60 < -10:
        bear_score += 2
        
    # 最终判断
    if bull_score >= 4:
        trend = "牛市"
    elif bear_score >= 4:
        trend = "熊市"
    else:
        trend = "震荡市"
        
    return {
        "指数": index_name,
        "最新点位": round(latest, 2),
        "20日均线": round(ma20, 2),
        "60日均线": round(ma60, 2),
        "20日涨跌幅": round(change_20, 2),
        "60日涨跌幅": round(change_60, 2),
        "趋势": trend,
        "牛熊评分": f"{bull_score}/{bear_score}"
    }


def get_overall_trend():
    """获取整体趋势"""
    # 获取上证指数和深证成指
    sh_df = get_index_kline("sh000001")
    sz_df = get_index_kline("sz399001")
    
    sh_res = analyze_trend(sh_df, "上证指数")
    sz_res = analyze_trend(sz_df, "深证成指")
    
    # 综合判断
    trends = [sh_res['趋势'], sz_res['趋势']]

    # 如果两个指数都数据不足，则整体也无法判断
    if trends.count("未知") == 2:
        overall = "未知"
        risk_level = "medium"
        pos_suggestion = "数据不足，无法给出仓位建议，请稍后重试"
    elif "牛市" in trends and "熊市" not in trends:
        overall = "牛市"
        risk_level = "high"
        pos_suggestion = "建议积极配置，仓位可较高"
    elif "熊市" in trends and "牛市" not in trends:
        overall = "熊市"
        risk_level = "low"
        pos_suggestion = "建议保守配置，保持高现金比例"
    else:
        overall = "震荡市"
        risk_level = "medium"
        pos_suggestion = "建议平衡配置，保持适度仓位"
        
    return {
        "日期": datetime.now().strftime('%Y-%m-%d'),
        "整体趋势": overall,
        "建议风险等级": risk_level,
        "仓位建议": pos_suggestion,
        "上证指数": sh_res,
        "深证成指": sz_res
    }


def main():
    parser = argparse.ArgumentParser(description='大盘趋势判断工具')
    parser.add_argument('action', choices=['analyze'], help='操作类型: analyze（分析趋势）')
    
    args = parser.parse_args()
    
    try:
        if args.action == 'analyze':
            data = get_overall_trend()
            print(json.dumps(data, ensure_ascii=False, indent=2))
            
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
