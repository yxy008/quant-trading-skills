#!/usr/bin/env python3
"""
市场宽度分析 - 涨跌统计/涨停跌停/市场情绪
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

from data_utils import get_market_breadth


def get_sector_breadth(date=None):
    """获取行业板块宽度，支持指定日期查询历史数据"""
    if date:
        return _get_sector_breadth_by_date(date)

    for attempt in range(3):
        try:
            df = ak.stock_board_industry_name_em()
            if df is None or df.empty:
                continue

            total = len(df)
            up_count = int((df['涨跌幅'] > 0).sum())
            down_count = int((df['涨跌幅'] < 0).sum())

            df_sorted = df.sort_values('涨跌幅', ascending=False)
            top_sectors = []
            for _, row in df_sorted.head(5).iterrows():
                top_sectors.append({
                    "板块": str(row.get('板块名称', '')),
                    "涨跌幅": float(row.get('涨跌幅', 0))
                })

            bottom_sectors = []
            for _, row in df_sorted.tail(5).iterrows():
                bottom_sectors.append({
                    "板块": str(row.get('板块名称', '')),
                    "涨跌幅": float(row.get('涨跌幅', 0))
                })

            return {
                "板块总数": total,
                "上涨板块": up_count,
                "下跌板块": down_count,
                "板块上涨比例": round(up_count / total * 100, 2) if total > 0 else 0,
                "领涨板块": top_sectors,
                "领跌板块": bottom_sectors
            }

        except Exception:
            pass

        time.sleep(1)

    return {"error": "无法获取板块宽度数据"}


def _get_sector_breadth_by_date(date_str):
    """根据指定日期获取历史行业板块宽度"""
    try:
        target_date = pd.to_datetime(date_str)

        if target_date.weekday() >= 5:
            weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            return {
                "数据日期说明": f"{target_date.strftime('%Y-%m-%d')} {weekday_names[target_date.weekday()]} 休市",
                "休市": True,
                "板块总数": 0, "上涨板块": 0, "下跌板块": 0,
                "板块上涨比例": 0, "领涨板块": [], "领跌板块": []
            }

        has_data = False
        result = {
            "数据日期说明": f"历史数据({target_date.strftime('%Y-%m-%d')})",
            "板块总数": 0, "上涨板块": 0, "下跌板块": 0,
            "板块上涨比例": 0, "领涨板块": [], "领跌板块": []
        }

        try:
            df = ak.stock_board_industry_hist_em(
                symbol="沪深两市",
                start_date=target_date.strftime('%Y%m%d'),
                end_date=target_date.strftime('%Y%m%d')
            )
            if df is not None and not df.empty:
                total = len(df)
                up_count = int((df['涨跌幅'] > 0).sum())
                down_count = int((df['涨跌幅'] < 0).sum())

                df_sorted = df.sort_values('涨跌幅', ascending=False)
                top_sectors = []
                for _, row in df_sorted.head(5).iterrows():
                    top_sectors.append({
                        "板块": str(row.get('板块名称', row.get('板块', ''))),
                        "涨跌幅": float(row.get('涨跌幅', 0))
                    })

                bottom_sectors = []
                for _, row in df_sorted.tail(5).iterrows():
                    bottom_sectors.append({
                        "板块": str(row.get('板块名称', row.get('板块', ''))),
                        "涨跌幅": float(row.get('涨跌幅', 0))
                    })

                result.update({
                    "板块总数": total, "上涨板块": up_count, "下跌板块": down_count,
                    "板块上涨比例": round(up_count / total * 100, 2) if total > 0 else 0,
                    "领涨板块": top_sectors, "领跌板块": bottom_sectors
                })
                has_data = True
        except Exception as e:
            print(f"[板块宽度] stock_board_industry_hist_em 失败: {e}")

        if not has_data:
            result["数据日期说明"] = f"{target_date.strftime('%Y-%m-%d')} 可能为节假日休市，无交易数据"
            result["休市"] = True

        return result
    except Exception as e:
        print(f"[板块宽度] _get_sector_breadth_by_date 异常: {e}")
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description='市场宽度分析')
    parser.add_argument('action', choices=['breadth', 'sector'],
                        help='操作类型: breadth（市场宽度）, sector（板块宽度）')

    args = parser.parse_args()

    try:
        if args.action == 'breadth':
            data = get_market_breadth()
            print(json.dumps(data, ensure_ascii=False, indent=2))

        elif args.action == 'sector':
            data = get_sector_breadth()
            print(json.dumps(data, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == '__main__':
    main()
