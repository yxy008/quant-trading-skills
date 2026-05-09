#!/usr/bin/env python3
"""
实时监控 - 盈亏实时计算、异常波动告警、风险预警
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

from data_utils import get_stock_kline, get_realtime_quote, get_market_overview


def calc_position_pnl(positions):
    """计算持仓盈亏"""
    if not positions:
        return {"error": "无持仓数据"}

    result = {
        "计算时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "持仓明细": [],
        "汇总": {
            "总成本": 0,
            "总市值": 0,
            "总盈亏": 0,
            "总盈亏比例": 0,
            "盈利股票数": 0,
            "亏损股票数": 0
        }
    }

    total_cost = 0
    total_value = 0
    profit_count = 0
    loss_count = 0

    for pos in positions:
        symbol = pos.get('symbol', '')
        cost_price = float(pos.get('cost_price', 0))
        quantity = int(pos.get('quantity', 0))
        buy_date = pos.get('buy_date', '')

        quote = get_realtime_quote(symbol)
        if quote is None:
            continue

        current_price = quote['最新价']
        cost_total = cost_price * quantity
        current_value = current_price * quantity
        pnl = current_value - cost_total
        pnl_pct = (current_price / cost_price - 1) * 100 if cost_price > 0 else 0

        total_cost += cost_total
        total_value += current_value

        if pnl > 0:
            profit_count += 1
        elif pnl < 0:
            loss_count += 1

        result["持仓明细"].append({
            "代码": symbol,
            "名称": quote.get('名称', ''),
            "成本价": cost_price,
            "现价": current_price,
            "数量": quantity,
            "成本总额": round(cost_total, 2),
            "当前市值": round(current_value, 2),
            "盈亏金额": round(pnl, 2),
            "盈亏比例": round(pnl_pct, 2),
            "今日涨跌": quote.get('涨跌幅', 0),
            "持有天数": (datetime.now() - datetime.strptime(buy_date, '%Y-%m-%d')).days if buy_date else None
        })

    total_pnl = total_value - total_cost
    total_pnl_pct = (total_value / total_cost - 1) * 100 if total_cost > 0 else 0

    result["汇总"] = {
        "总成本": round(total_cost, 2),
        "总市值": round(total_value, 2),
        "总盈亏": round(total_pnl, 2),
        "总盈亏比例": round(total_pnl_pct, 2),
        "盈利股票数": profit_count,
        "亏损股票数": loss_count
    }

    return result


def detect_anomalies(symbol):
    """检测异常波动"""
    quote = get_realtime_quote(symbol)
    df = get_stock_kline(symbol, days=60)

    if quote is None:
        return {"error": f"无法获取 {symbol} 实时行情"}

    alerts = []
    risk_level = "正常"

    # 1. 涨跌幅异常
    change_pct = quote['涨跌幅']
    if change_pct >= 9.5:
        alerts.append({"类型": "涨停", "级别": "提示", "描述": f"涨停 +{change_pct}%", "建议": "关注封板力度和次日走势"})
    elif change_pct <= -9.5:
        alerts.append({"类型": "跌停", "级别": "警告", "描述": f"跌停 {change_pct}%", "建议": "评估是否止损或减仓"})
    elif change_pct >= 7:
        alerts.append({"类型": "大涨", "级别": "提示", "描述": f"大涨 +{change_pct}%", "建议": "关注是否放量，考虑部分止盈"})
    elif change_pct <= -7:
        alerts.append({"类型": "大跌", "级别": "警告", "描述": f"大跌 {change_pct}%", "建议": "检查基本面是否有变化，评估止损"})
    elif change_pct >= 5:
        alerts.append({"类型": "明显上涨", "级别": "提示", "描述": f"上涨 +{change_pct}%"})
    elif change_pct <= -5:
        alerts.append({"类型": "明显下跌", "级别": "关注", "描述": f"下跌 {change_pct}%"})

    # 2. 振幅异常
    amplitude = quote['振幅']
    if amplitude > 10:
        alerts.append({"类型": "剧烈波动", "级别": "警告", "描述": f"振幅 {amplitude}%", "建议": "市场分歧大，谨慎操作"})
    elif amplitude > 7:
        alerts.append({"类型": "大幅波动", "级别": "关注", "描述": f"振幅 {amplitude}%"})

    # 3. 量比异常
    volume_ratio = quote.get('量比', 1)
    if volume_ratio > 3:
        alerts.append({"类型": "巨量", "级别": "关注", "描述": f"量比 {volume_ratio}", "建议": "关注是否有利好消息或主力出货"})
    elif volume_ratio > 2:
        alerts.append({"类型": "放量", "级别": "提示", "描述": f"量比 {volume_ratio}"})

    # 4. 历史对比异常
    if df is not None and len(df) >= 20:
        current = quote['最新价']

        high_20 = df.tail(20)['high'].max() if 'high' in df.columns else df['high'].tail(20).max()
        low_20 = df.tail(20)['low'].min() if 'low' in df.columns else df['low'].tail(20).min()
        n60 = min(60, len(df))
        high_60 = df.tail(n60)['high'].max() if 'high' in df.columns else df['high'].tail(n60).max()
        low_60 = df.tail(n60)['low'].min() if 'low' in df.columns else df['low'].tail(n60).min()

        if current >= high_60:
            alerts.append({"类型": "创60日新高", "级别": "提示", "描述": f"突破60日高点 {high_60}", "建议": "趋势向好，可考虑加仓"})
        elif current >= high_20:
            alerts.append({"类型": "创20日新高", "级别": "提示", "描述": f"突破20日高点 {high_20}"})

        if current <= low_60:
            alerts.append({"类型": "创60日新低", "级别": "警告", "描述": f"跌破60日低点 {low_60}", "建议": "趋势走坏，建议减仓或止损"})
        elif current <= low_20:
            alerts.append({"类型": "创20日新低", "级别": "关注", "描述": f"跌破20日低点 {low_20}"})

        # 连续涨跌
        close_col = df['收盘'] if '收盘' in df.columns else (df['close'] if 'close' in df.columns else None)
        if close_col is not None:
            returns = close_col.pct_change().dropna().tail(5)
            if len(returns) >= 5:
                consecutive_up = (returns > 0).sum()
                consecutive_down = (returns < 0).sum()
                if consecutive_up >= 5:
                    alerts.append({"类型": "连涨5日", "级别": "提示", "描述": "连续5日上涨", "建议": "短期可能超买，注意回调风险"})
                if consecutive_down >= 5:
                    alerts.append({"类型": "连跌5日", "级别": "警告", "描述": "连续5日下跌", "建议": "短期超卖，关注反弹机会或止损"})

    # 5. 换手率异常
    turnover = quote.get('换手率', 0)
    if turnover > 20:
        alerts.append({"类型": "超高换手", "级别": "警告", "描述": f"换手率 {turnover}%", "建议": "筹码松动，主力可能出货"})
    elif turnover > 10:
        alerts.append({"类型": "高换手", "级别": "关注", "描述": f"换手率 {turnover}%"})

    # 确定风险等级
    warning_count = sum(1 for a in alerts if a['级别'] == '警告')
    attention_count = sum(1 for a in alerts if a['级别'] == '关注')

    if warning_count >= 2:
        risk_level = "高风险"
    elif warning_count >= 1:
        risk_level = "较高风险"
    elif attention_count >= 2:
        risk_level = "关注"
    elif attention_count >= 1:
        risk_level = "留意"

    return {
        "股票代码": symbol,
        "股票名称": quote.get('名称', ''),
        "当前价": quote['最新价'],
        "涨跌幅": change_pct,
        "风险等级": risk_level,
        "告警数量": len(alerts),
        "告警列表": alerts,
        "实时数据": {
            "今开": quote['今开'],
            "最高": quote['最高'],
            "最低": quote['最低'],
            "昨收": quote['昨收'],
            "振幅": amplitude,
            "换手率": turnover,
            "量比": volume_ratio,
            "成交额(亿)": round(quote['成交额'] / 100000000, 2)
        }
    }


def monitor_portfolio(positions):
    """监控整个持仓组合"""
    if not positions:
        return {"error": "无持仓数据"}

    result = {
        "监控时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "持仓数量": len(positions),
        "个股监控": [],
        "组合风险等级": "正常",
        "需要关注的股票": []
    }

    max_risk = 0
    risk_scores = {"正常": 0, "留意": 1, "关注": 2, "较高风险": 3, "高风险": 4}

    for pos in positions:
        symbol = pos.get('symbol', '')
        anomaly = detect_anomalies(symbol)
        result["个股监控"].append(anomaly)

        risk = anomaly.get('风险等级', '正常')
        risk_score = risk_scores.get(risk, 0)
        if risk_score > max_risk:
            max_risk = risk_score

        if risk_score >= 2:
            result["需要关注的股票"].append({
                "代码": symbol,
                "名称": anomaly.get('股票名称', ''),
                "风险等级": risk,
                "告警数": anomaly.get('告警数量', 0)
            })

    # 组合整体风险
    for level, score in risk_scores.items():
        if score == max_risk:
            result["组合风险等级"] = level
            break

    # 盈亏计算
    pnl = calc_position_pnl(positions)
    if 'error' not in pnl:
        result["盈亏汇总"] = pnl.get("汇总", {})

    return result


def main():
    parser = argparse.ArgumentParser(description='实时监控工具')
    parser.add_argument('action', choices=['quote', 'anomaly', 'pnl', 'monitor', 'market'],
                        help='操作类型')
    parser.add_argument('--symbol', default='600519', help='股票代码')
    parser.add_argument('--positions', default='[]', help='持仓JSON')

    args = parser.parse_args()

    try:
        if args.action == 'quote':
            data = get_realtime_quote(args.symbol)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'anomaly':
            data = detect_anomalies(args.symbol)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'pnl':
            positions = json.loads(args.positions)
            data = calc_position_pnl(positions)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'monitor':
            positions = json.loads(args.positions)
            data = monitor_portfolio(positions)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'market':
            data = get_market_overview()
            print(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
