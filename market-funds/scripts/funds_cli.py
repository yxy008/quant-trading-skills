#!/usr/bin/env python3
"""
市场资金流向查询 - 基于 AkShare 真实数据
"""
import argparse
import json
import sys
import os
import time
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

from data_utils import get_northbound_funds, get_industry_funds, get_stock_kline, get_data_date_note, _get_spot_df


def get_stock_funds(symbol):
    """获取个股资金流向 - 真实数据"""
    if len(symbol) == 6:
        if symbol.startswith('6'):
            full_symbol = f"sh{symbol}"
        else:
            full_symbol = f"sz{symbol}"
    else:
        full_symbol = symbol

    for attempt in range(3):
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')

            df = ak.stock_zh_a_hist_tx(
                symbol=full_symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )

            if df is not None and not df.empty:
                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else latest

                volume_ratio = 1.0
                if len(df) >= 6:
                    avg_vol = df['amount'].iloc[-6:-1].mean()
                    if avg_vol > 0:
                        volume_ratio = float(latest['amount']) / avg_vol

                return {
                    "代码": symbol,
                    "最新日期": str(latest['date']),
                    "收盘价": float(latest['close']),
                    "涨跌幅": round((float(latest['close']) / float(prev['close']) - 1) * 100, 2),
                    "成交量": int(latest['amount']),
                    "成交额": round(float(latest['amount']) * float(latest['close']), 2),
                    "量比": round(volume_ratio, 2),
                    "换手率": "数据暂缺"
                }
        except Exception:
            pass

        time.sleep(1)

    return {"error": f"无法获取股票 {symbol} 的资金数据"}


def get_market_overview(date=None):
    """获取市场总览 - 支持指定日期查询历史数据"""
    if date:
        return _get_market_overview_by_date(date)

    for attempt in range(3):
        try:
            df = _get_spot_df()
            if df is not None and not df.empty:
                up_count = len(df[df['涨跌幅'] > 0])
                down_count = len(df[df['涨跌幅'] < 0])
                flat_count = len(df[df['涨跌幅'] == 0])
                total = len(df)

                total_volume = df['成交量'].sum() if '成交量' in df.columns else 0
                total_amount = df['成交额'].sum() if '成交额' in df.columns else 0

                return {
                    "统计时间": datetime.now().strftime('%Y-%m-%d %H:%M'),
                    "数据日期说明": get_data_date_note(),
                    "上涨家数": int(up_count),
                    "下跌家数": int(down_count),
                    "平盘家数": int(flat_count),
                    "总家数": int(total),
                    "上涨比例": round(up_count / total * 100, 2) if total > 0 else 0,
                    "总成交量": int(total_volume),
                    "总成交额": round(float(total_amount), 2)
                }
        except Exception:
            pass

        time.sleep(1)

    return {}


def comprehensive_capital_flow_analysis():
    """
    综合资金流向分析
    整合北向资金、板块资金、市场总览，给出综合判断
    """
    result = {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "数据日期说明": get_data_date_note(),
    }

    # 1. 北向资金
    try:
        northbound = get_northbound_funds()
        if northbound and "error" not in northbound:
            result["北向资金"] = northbound
    except Exception:
        result["北向资金"] = {"状态": "获取失败"}

    # 2. 板块资金流向
    try:
        industry = get_industry_funds()
        if industry and "error" not in industry:
            result["板块资金"] = industry
    except Exception:
        result["板块资金"] = {"状态": "获取失败"}

    # 3. 市场总览
    try:
        overview = get_market_overview()
        if overview:
            result["市场总览"] = overview
    except Exception:
        result["市场总览"] = {"状态": "获取失败"}

    # 4. 综合判断
    signals = []
    score = 0

    # 北向资金信号
    nb = result.get("北向资金", {})
    if nb.get("今日净流入", 0) > 50:
        signals.append("北向资金大幅净流入，外资看多")
        score += 2
    elif nb.get("今日净流入", 0) > 0:
        signals.append("北向资金小幅净流入")
        score += 1
    elif nb.get("今日净流入", 0) < -50:
        signals.append("北向资金大幅净流出，外资看空")
        score -= 2
    elif nb.get("今日净流入", 0) < 0:
        signals.append("北向资金小幅净流出")
        score -= 1

    # 市场宽度信号
    overview = result.get("市场总览", {})
    up_ratio = overview.get("上涨比例", 50)
    if up_ratio > 70:
        signals.append("市场普涨，赚钱效应强")
        score += 2
    elif up_ratio > 50:
        signals.append("涨多跌少，市场偏强")
        score += 1
    elif up_ratio < 30:
        signals.append("市场普跌，亏钱效应强")
        score -= 2
    elif up_ratio < 50:
        signals.append("跌多涨少，市场偏弱")
        score -= 1

    # 成交额信号
    total_amount = overview.get("总成交额", 0)
    if total_amount > 15000:
        signals.append("成交额超1.5万亿，市场活跃度高")
        score += 1
    elif total_amount < 5000:
        signals.append("成交额不足5000亿，市场低迷")
        score -= 1

    # 综合评分
    if score >= 4:
        overall = "资金面强势，多方占优"
        suggestion = "可积极做多，关注北向资金持续流入的板块"
    elif score >= 1:
        overall = "资金面偏多，谨慎乐观"
        suggestion = "可适度参与，注意板块轮动节奏"
    elif score >= -1:
        overall = "资金面中性，方向不明"
        suggestion = "观望为主，等待资金面明确信号"
    elif score >= -4:
        overall = "资金面偏空，谨慎防御"
        suggestion = "降低仓位，关注北向资金流出压力"
    else:
        overall = "资金面弱势，空方占优"
        suggestion = "防御为主，现金为王"

    result["综合判断"] = {
        "评分": score,
        "信号": signals,
        "总体评价": overall,
        "操作建议": suggestion,
    }

    return result


def _get_market_overview_by_date(date_str):
    """根据指定日期获取历史市场总览数据"""
    try:
        target_date = pd.to_datetime(date_str)

        # 检查是否为周末
        if target_date.weekday() >= 5:
            weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            return {
                "统计时间": target_date.strftime('%Y-%m-%d'),
                "数据日期说明": f"{target_date.strftime('%Y-%m-%d')} {weekday_names[target_date.weekday()]} 休市",
                "休市": True,
                "上涨家数": 0, "下跌家数": 0, "平盘家数": 0,
                "总家数": 0, "上涨比例": 0, "总成交量": 0, "总成交额": 0
            }

        result = {
            "统计时间": target_date.strftime('%Y-%m-%d'),
            "数据日期说明": f"历史数据({target_date.strftime('%Y-%m-%d')})",
            "上涨家数": 0, "下跌家数": 0, "平盘家数": 0,
            "总家数": 0, "上涨比例": 0, "总成交量": 0, "总成交额": 0
        }

        # 获取涨跌家数（主要数据源）
        has_data = False
        try:
            detail_df = ak.stock_market_detail_up_down_em(date=target_date.strftime('%Y%m%d'))
            if detail_df is not None and not detail_df.empty:
                up_count = len(detail_df[detail_df['涨跌幅'] > 0])
                down_count = len(detail_df[detail_df['涨跌幅'] < 0])
                flat_count = len(detail_df[detail_df['涨跌幅'] == 0])
                total = len(detail_df)
                result["上涨家数"] = int(up_count)
                result["下跌家数"] = int(down_count)
                result["平盘家数"] = int(flat_count)
                result["总家数"] = int(total)
                result["上涨比例"] = round(up_count / total * 100, 2) if total > 0 else 0
                if '成交额' in detail_df.columns:
                    result["总成交额"] = round(float(detail_df['成交额'].sum()), 2)
                has_data = True
        except Exception as e:
            print(f"[市场总览] stock_market_detail_up_down_em 失败: {e}")

        # 获取上证指数数据
        try:
            sh_df = ak.stock_zh_index_daily_em(symbol="sh000001")
            if sh_df is not None and not sh_df.empty:
                sh_df['date'] = pd.to_datetime(sh_df['date'])
                match = sh_df[sh_df['date'] == target_date]
                if not match.empty:
                    row = match.iloc[0]
                    result["上证指数"] = {
                        "开盘": float(row.get('open', 0)),
                        "收盘": float(row.get('close', 0)),
                        "最高": float(row.get('high', 0)),
                        "最低": float(row.get('low', 0)),
                        "涨跌幅": float(row.get('pct_chg', 0)) if 'pct_chg' in row else 0,
                        "成交量": int(row.get('volume', 0)),
                        "成交额": float(row.get('amount', 0))
                    }
                    has_data = True
        except Exception as e:
            print(f"[市场总览] 上证指数获取失败: {e}")

        # 获取深证成指数据
        try:
            sz_df = ak.stock_zh_index_daily_em(symbol="sz399001")
            if sz_df is not None and not sz_df.empty:
                sz_df['date'] = pd.to_datetime(sz_df['date'])
                match = sz_df[sz_df['date'] == target_date]
                if not match.empty:
                    row = match.iloc[0]
                    result["深证成指"] = {
                        "收盘": float(row.get('close', 0)),
                        "涨跌幅": float(row.get('pct_chg', 0)) if 'pct_chg' in row else 0,
                        "成交额": float(row.get('amount', 0))
                    }
                    has_data = True
        except Exception as e:
            print(f"[市场总览] 深证成指获取失败: {e}")

        # 汇总成交额
        if result.get("总成交额", 0) == 0:
            total_amount = 0
            if result.get("上证指数", {}).get("成交额", 0):
                total_amount += result["上证指数"]["成交额"]
            if result.get("深证成指", {}).get("成交额", 0):
                total_amount += result["深证成指"]["成交额"]
            if total_amount > 0:
                result["总成交额"] = round(total_amount, 2)

        if not has_data:
            result["数据日期说明"] = f"{target_date.strftime('%Y-%m-%d')} 可能为节假日休市，无交易数据"
            result["休市"] = True

        return result
    except Exception as e:
        print(f"[市场总览] _get_market_overview_by_date 异常: {e}")
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description='市场资金流向查询')
    parser.add_argument('action', choices=['northbound', 'industry', 'stock', 'overview'],
                        help='操作类型: northbound（北向）, industry（板块）, stock（个股）, overview（市场总览）')
    parser.add_argument('--symbol', help='股票代码（仅 stock 操作需要）')

    args = parser.parse_args()

    try:
        if args.action == 'northbound':
            data = get_northbound_funds()
            print(json.dumps(data, ensure_ascii=False, indent=2))

        elif args.action == 'industry':
            data = get_industry_funds()
            print(json.dumps(data, ensure_ascii=False, indent=2))

        elif args.action == 'stock':
            if not args.symbol:
                print(json.dumps({"error": "需要 --symbol 参数"}, ensure_ascii=False, indent=2))
                sys.exit(1)
            data = get_stock_funds(args.symbol)
            print(json.dumps(data, ensure_ascii=False, indent=2))

        elif args.action == 'overview':
            data = get_market_overview()
            print(json.dumps(data, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == '__main__':
    main()
