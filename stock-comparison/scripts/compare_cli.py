#!/usr/bin/env python3
"""
股票对比分析 - 多股横向对比
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

from data_utils import get_stock_kline, _get_spot_df


def get_stock_basic_info(symbol):
    """获取单只股票基本信息"""
    for attempt in range(3):
        try:
            df = _get_spot_df()
            filtered = df[df['代码'] == symbol]
            if not filtered.empty:
                row = filtered.iloc[0]
                return {
                    "代码": symbol,
                    "名称": str(row.get('名称', '')),
                    "最新价": float(row.get('最新价', 0)) if pd.notna(row.get('最新价')) else 0,
                    "涨跌幅": float(row.get('涨跌幅', 0)) if pd.notna(row.get('涨跌幅')) else 0,
                    "市盈率-动态": float(row.get('市盈率-动态', 0)) if pd.notna(row.get('市盈率-动态')) else None,
                    "市净率": float(row.get('市净率', 0)) if pd.notna(row.get('市净率')) else None,
                    "总市值": float(row.get('总市值', 0)) if pd.notna(row.get('总市值')) else 0,
                    "流通市值": float(row.get('流通市值', 0)) if pd.notna(row.get('流通市值')) else 0,
                    "成交额": float(row.get('成交额', 0)) if pd.notna(row.get('成交额')) else 0,
                    "换手率": float(row.get('换手率', 0)) if pd.notna(row.get('换手率')) else None,
                    "60日涨跌幅": float(row.get('60日涨跌幅', 0)) if pd.notna(row.get('60日涨跌幅')) else None
                }
        except Exception:
            pass

        time.sleep(1)

    return None


def get_stock_kline_returns(symbol, days=60):
    """获取股票近期收益率"""
    df = get_stock_kline(symbol, days=days + 10)

    if df is not None and not df.empty:
        close = df['收盘']
        returns = close.pct_change().dropna()

        ret_5d = (float(close.iloc[-1]) / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 else None
        ret_20d = (float(close.iloc[-1]) / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else None
        ret_60d = (float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100

        volatility = returns.std() * np.sqrt(252) * 100

        return {
            "5日涨跌幅": round(ret_5d, 2) if ret_5d is not None else None,
            "20日涨跌幅": round(ret_20d, 2) if ret_20d is not None else None,
            "60日涨跌幅": round(ret_60d, 2),
                    "年化波动率": round(volatility, 2)
                }
        # except Exception:  # pyright: ignore[reportUnreachable]
        #     pass

        time.sleep(1)

    return None


def compare_stocks(symbols_str):
    """对比多只股票"""
    symbols = [s.strip() for s in symbols_str.split(',') if s.strip()]
    if len(symbols) < 2:
        return {"error": "至少需要2只股票进行对比"}

    results = []
    for symbol in symbols:
        info = get_stock_basic_info(symbol)
        returns_data = get_stock_kline_returns(symbol, days=60)

        if info is None:
            results.append({"代码": symbol, "error": "无法获取数据"})
            continue

        if returns_data:
            info.update(returns_data)

        results.append(info)

    # 生成对比排名
    if len(results) >= 2:
        valid = [r for r in results if 'error' not in r]

        # 按涨跌幅排名
        by_change = sorted(valid, key=lambda x: x.get('涨跌幅', -999), reverse=True)
        # 按市盈率排名（越低越好）
        by_pe = sorted(valid, key=lambda x: x.get('市盈率-动态') if x.get('市盈率-动态') and x['市盈率-动态'] > 0 else 9999)
        # 按总市值排名
        by_mcap = sorted(valid, key=lambda x: x.get('总市值', 0), reverse=True)

        rankings = {
            "今日涨幅排名": [{"代码": r['代码'], "名称": r['名称'], "涨跌幅": r['涨跌幅']} for r in by_change],
            "市盈率排名": [{"代码": r['代码'], "名称": r['名称'], "市盈率": r.get('市盈率-动态')} for r in by_pe],
            "市值排名": [{"代码": r['代码'], "名称": r['名称'], "总市值": r.get('总市值')} for r in by_mcap]
        }
    else:
        rankings = {}

    return {
        "对比股票": results,
        "排名分析": rankings,
        "对比数量": len(results)
    }


def main():
    parser = argparse.ArgumentParser(description='股票对比分析')
    parser.add_argument('action', choices=['compare'], help='操作类型')
    parser.add_argument('--symbols', required=True, help='股票代码列表，逗号分隔，如 600519,000858,000568')

    args = parser.parse_args()

    try:
        if args.action == 'compare':
            data = compare_stocks(args.symbols)
            print(json.dumps(data, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == '__main__':
    main()
