#!/usr/bin/env python3
"""
交易日历模块 - 基于 AkShare 真实交易日历数据
提供交易日判断、交易日区间查询、前后交易日推算等功能
"""
import argparse
import json
import sys
import os
from datetime import datetime, date, timedelta

_agent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

try:
    import akshare as ak
    import pandas as pd
except ImportError:
    print("请先安装依赖: pip install akshare pandas")
    sys.exit(1)


_trade_date_cache = None
_cache_timestamp = None
_CACHE_TTL_SECONDS = 3600


def _get_trade_date_df():
    """获取交易日历DataFrame（带缓存）"""
    global _trade_date_cache, _cache_timestamp
    now = datetime.now()
    if _trade_date_cache is not None and _cache_timestamp is not None:
        if (now - _cache_timestamp).total_seconds() < _CACHE_TTL_SECONDS:
            return _trade_date_cache

    try:
        df = ak.tool_trade_date_hist_sina()
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
        _trade_date_cache = df
        _cache_timestamp = now
        return df
    except Exception:
        if _trade_date_cache is not None:
            return _trade_date_cache
        raise


def _get_trade_date_set():
    """获取交易日日期集合"""
    df = _get_trade_date_df()
    return set(df['trade_date'].tolist())


def _get_trade_date_list():
    """获取交易日日期列表（排序）"""
    df = _get_trade_date_df()
    return sorted(df['trade_date'].tolist())


def is_trading_day(check_date=None):
    """
    判断是否为交易日
    参数:
        check_date: 日期，支持 date 对象或 'YYYY-MM-DD' / 'YYYYMMDD' 字符串，默认今天
    返回:
        dict: {"日期": str, "是否交易日": bool, "星期": str, "说明": str}
    """
    if check_date is None:
        check_date = date.today()
    elif isinstance(check_date, str):
        check_date = check_date.replace('-', '').replace('/', '')
        if len(check_date) == 8:
            check_date = date(int(check_date[:4]), int(check_date[4:6]), int(check_date[6:8]))
        else:
            return {"error": f"日期格式错误: {check_date}，支持 YYYY-MM-DD 或 YYYYMMDD"}

    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday = check_date.weekday()

    trade_dates = _get_trade_date_set()
    is_trade = check_date in trade_dates

    if is_trade:
        note = f"{weekday_names[weekday]}，是交易日"
    elif weekday >= 5:
        note = f"{weekday_names[weekday]}，周末休市"
    else:
        note = f"{weekday_names[weekday]}，节假日休市"

    return {
        "日期": check_date.strftime('%Y-%m-%d'),
        "是否交易日": is_trade,
        "星期": weekday_names[weekday],
        "说明": note,
    }


def get_trading_days(start_date, end_date):
    """
    获取指定区间内的交易日列表
    参数:
        start_date: 开始日期 'YYYY-MM-DD' 或 'YYYYMMDD'
        end_date: 结束日期 'YYYY-MM-DD' 或 'YYYYMMDD'
    返回:
        dict: {"区间": str, "交易日数量": int, "交易日列表": list, "非交易日数量": int}
    """
    def _parse_date(d):
        if isinstance(d, date):
            return d
        d = d.replace('-', '').replace('/', '')
        return date(int(d[:4]), int(d[4:6]), int(d[6:8]))

    start = _parse_date(start_date)
    end = _parse_date(end_date)

    if start > end:
        start, end = end, start

    trade_dates = _get_trade_date_set()
    all_dates = _get_trade_date_list()

    trading_list = [d for d in all_dates if start <= d <= end]
    trading_list_str = [d.strftime('%Y-%m-%d') for d in trading_list]

    total_days = (end - start).days + 1
    non_trading = total_days - len(trading_list)

    return {
        "区间": f"{start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}",
        "自然日数": total_days,
        "交易日数量": len(trading_list),
        "非交易日数量": non_trading,
        "交易日列表": trading_list_str,
    }


def get_previous_trading_day(target_date=None, n=1):
    """
    获取前N个交易日
    参数:
        target_date: 目标日期，默认今天
        n: 向前推N个交易日
    返回:
        dict: {"目标日期": str, "前N个交易日": str, "间隔自然日": int}
    """
    if target_date is None:
        target_date = date.today()
    elif isinstance(target_date, str):
        target_date = target_date.replace('-', '').replace('/', '')
        target_date = date(int(target_date[:4]), int(target_date[4:6]), int(target_date[6:8]))

    all_dates = _get_trade_date_list()

    prev_dates = [d for d in all_dates if d < target_date]
    if len(prev_dates) < n:
        return {"error": f"目标日期 {target_date} 之前不足 {n} 个交易日"}

    prev_date = prev_dates[-n]
    gap_days = (target_date - prev_date).days

    return {
        "目标日期": target_date.strftime('%Y-%m-%d'),
        "前N个交易日": prev_date.strftime('%Y-%m-%d'),
        "N": n,
        "间隔自然日": gap_days,
    }


def get_next_trading_day(target_date=None, n=1):
    """
    获取后N个交易日
    参数:
        target_date: 目标日期，默认今天
        n: 向后推N个交易日
    返回:
        dict: {"目标日期": str, "后N个交易日": str, "间隔自然日": int}
    """
    if target_date is None:
        target_date = date.today()
    elif isinstance(target_date, str):
        target_date = target_date.replace('-', '').replace('/', '')
        target_date = date(int(target_date[:4]), int(target_date[4:6]), int(target_date[6:8]))

    all_dates = _get_trade_date_list()

    next_dates = [d for d in all_dates if d > target_date]
    if len(next_dates) < n:
        return {"error": f"目标日期 {target_date} 之后不足 {n} 个交易日"}

    next_date = next_dates[n - 1]
    gap_days = (next_date - target_date).days

    return {
        "目标日期": target_date.strftime('%Y-%m-%d'),
        "后N个交易日": next_date.strftime('%Y-%m-%d'),
        "N": n,
        "间隔自然日": gap_days,
    }


def get_latest_trading_day():
    """
    获取最近一个交易日（含今天）
    返回:
        dict: {"最近交易日": str, "星期": str, "是否今天": bool}
    """
    today = date.today()
    all_dates = _get_trade_date_list()

    valid_dates = [d for d in all_dates if d <= today]
    if not valid_dates:
        return {"error": "无法获取最近交易日"}

    latest = valid_dates[-1]
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

    return {
        "最近交易日": latest.strftime('%Y-%m-%d'),
        "星期": weekday_names[latest.weekday()],
        "是否今天": latest == today,
    }


def get_trading_calendar(year=None):
    """
    获取指定年份的交易日历概览
    参数:
        year: 年份，默认当前年份
    返回:
        dict: 包含该年各月交易日统计
    """
    if year is None:
        year = date.today().year

    all_dates = _get_trade_date_list()
    year_dates = [d for d in all_dates if d.year == year]

    monthly = {}
    for d in year_dates:
        month_key = d.strftime('%Y-%m')
        if month_key not in monthly:
            monthly[month_key] = 0
        monthly[month_key] += 1

    monthly_stats = [
        {"月份": k, "交易日数": v}
        for k, v in sorted(monthly.items())
    ]

    return {
        "年份": year,
        "全年交易日数": len(year_dates),
        "月均交易日": round(len(year_dates) / 12, 1),
        "月度统计": monthly_stats,
    }


def get_trading_days_count(start_date, end_date):
    """
    获取区间内交易日数量
    参数:
        start_date: 开始日期
        end_date: 结束日期
    返回:
        dict: {"区间": str, "交易日数量": int}
    """
    result = get_trading_days(start_date, end_date)
    return {
        "区间": result["区间"],
        "交易日数量": result["交易日数量"],
    }


def main():
    parser = argparse.ArgumentParser(description='交易日历工具')
    parser.add_argument('action', choices=[
        'check', 'range', 'prev', 'next', 'latest', 'calendar', 'count'
    ], help='操作类型')
    parser.add_argument('--date', help='日期 (YYYY-MM-DD 或 YYYYMMDD)，默认今天')
    parser.add_argument('--start', help='开始日期')
    parser.add_argument('--end', help='结束日期')
    parser.add_argument('--n', type=int, default=1, help='前后N个交易日')
    parser.add_argument('--year', type=int, help='年份')

    args = parser.parse_args()

    try:
        if args.action == 'check':
            data = is_trading_day(args.date)
        elif args.action == 'range':
            if not args.start or not args.end:
                print(json.dumps({"error": "需要 --start 和 --end 参数"}, ensure_ascii=False, indent=2))
                sys.exit(1)
            data = get_trading_days(args.start, args.end)
        elif args.action == 'prev':
            data = get_previous_trading_day(args.date, args.n)
        elif args.action == 'next':
            data = get_next_trading_day(args.date, args.n)
        elif args.action == 'latest':
            data = get_latest_trading_day()
        elif args.action == 'calendar':
            data = get_trading_calendar(args.year)
        elif args.action == 'count':
            if not args.start or not args.end:
                print(json.dumps({"error": "需要 --start 和 --end 参数"}, ensure_ascii=False, indent=2))
                sys.exit(1)
            data = get_trading_days_count(args.start, args.end)
        else:
            parser.print_help()
            return

        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))

    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == '__main__':
    main()
