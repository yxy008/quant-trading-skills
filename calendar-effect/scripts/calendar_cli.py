#!/usr/bin/env python3
"""
日历效应分析
分析A股市场的日历效应，包括：
- 周内效应（星期几表现差异）
- 月初/月末效应
- 节前/节后效应
- 季度效应
- 两会效应
"""
import argparse
import json
import sys
import os
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

from data_utils import get_stock_kline, get_index_kline


def weekday_effect(symbol="000300", years=3):
    """
    周内效应分析
    分析一周中每天的平均涨跌幅和上涨概率
    """
    df = get_index_kline(symbol, days=years * 252)
    if df is None or len(df) < 100:
        return {"error": "数据不足"}

    date_col = '日期' if '日期' in df.columns else 'date'
    close_col = '收盘' if '收盘' in df.columns else 'close'

    df[date_col] = pd.to_datetime(df[date_col])
    df['weekday'] = df[date_col].dt.dayofweek
    df['return'] = df[close_col].pct_change()

    weekday_names = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五"}

    result = {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "分析标的": symbol,
        "分析周期": f"{years}年",
        "周内效应": {},
    }

    for day_num, day_name in weekday_names.items():
        day_data = df[df['weekday'] == day_num]['return'].dropna()
        if len(day_data) < 10:
            continue

        avg_return = float(day_data.mean()) * 100
        win_rate = float((day_data > 0).mean()) * 100
        std_return = float(day_data.std()) * 100

        result["周内效应"][day_name] = {
            "样本数": len(day_data),
            "平均涨跌幅": f"{avg_return:+.2f}%",
            "上涨概率": f"{win_rate:.1f}%",
            "波动率": f"{std_return:.2f}%",
        }

    # 找出最佳和最差交易日
    if result["周内效应"]:
        best_day = max(result["周内效应"].items(),
                       key=lambda x: float(x[1]["平均涨跌幅"].replace('%', '').replace('+', '')))
        worst_day = min(result["周内效应"].items(),
                        key=lambda x: float(x[1]["平均涨跌幅"].replace('%', '').replace('+', '')))

        result["周内总结"] = {
            "最佳交易日": f"{best_day[0]}（平均{best_day[1]['平均涨跌幅']}）",
            "最差交易日": f"{worst_day[0]}（平均{worst_day[1]['平均涨跌幅']}）",
            "操作建议": _weekday_advice(best_day[0], worst_day[0]),
        }

    return result


def _weekday_advice(best_day, worst_day):
    """根据周内效应给出建议"""
    advice = []
    if "周四" == worst_day:
        advice.append("周四通常表现较差，避免在周四追高买入")
    if "周五" == best_day:
        advice.append("周五表现较好，可考虑在周五收盘前布局周末利好预期")
    if "周一" == best_day:
        advice.append("周一通常有周末消息面催化，注意开盘方向")
    if not advice:
        advice.append("周内效应不显著，以其他分析为主")
    return advice


def month_effect(symbol="000300", years=5):
    """
    月份效应分析
    分析每个月的平均涨跌幅和上涨概率
    """
    df = get_index_kline(symbol, days=years * 252)
    if df is None or len(df) < 200:
        return {"error": "数据不足"}

    date_col = '日期' if '日期' in df.columns else 'date'
    close_col = '收盘' if '收盘' in df.columns else 'close'

    df[date_col] = pd.to_datetime(df[date_col])
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year

    # 按月计算涨跌幅
    monthly = df.groupby(['year', 'month']).agg(
        月初=('close', 'first'),
        月末=('close', 'last'),
    ).reset_index()
    monthly['月涨跌幅'] = (monthly['月末'] / monthly['月初'] - 1) * 100

    month_names = {
        1: "一月", 2: "二月", 3: "三月", 4: "四月",
        5: "五月", 6: "六月", 7: "七月", 8: "八月",
        9: "九月", 10: "十月", 11: "十一月", 12: "十二月"
    }

    result = {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "分析标的": symbol,
        "分析周期": f"{years}年",
        "月份效应": {},
    }

    for month_num, month_name in month_names.items():
        month_data = monthly[monthly['month'] == month_num]['月涨跌幅']
        if len(month_data) < 2:
            continue

        avg_return = float(month_data.mean())
        win_rate = float((month_data > 0).mean()) * 100
        max_return = float(month_data.max())
        min_return = float(month_data.min())

        result["月份效应"][month_name] = {
            "样本年数": len(month_data),
            "平均涨跌幅": f"{avg_return:+.2f}%",
            "上涨概率": f"{win_rate:.1f}%",
            "最大涨幅": f"{max_return:+.2f}%",
            "最大跌幅": f"{min_return:+.2f}%",
        }

    if result["月份效应"]:
        best_month = max(result["月份效应"].items(),
                         key=lambda x: float(x[1]["平均涨跌幅"].replace('%', '').replace('+', '')))
        worst_month = min(result["月份效应"].items(),
                          key=lambda x: float(x[1]["平均涨跌幅"].replace('%', '').replace('+', '')))

        result["月份总结"] = {
            "最佳月份": f"{best_month[0]}（平均{best_month[1]['平均涨跌幅']}）",
            "最差月份": f"{worst_month[0]}（平均{worst_month[1]['平均涨跌幅']}）",
            "操作建议": _month_advice(best_month[0], worst_month[0]),
        }

    return result


def _month_advice(best_month, worst_month):
    """根据月份效应给出建议"""
    advice = []
    if "二月" == best_month:
        advice.append("二月通常有春季行情，可适当提高仓位")
    if "四月" == worst_month:
        advice.append("四月年报密集披露，注意业绩雷风险")
    if "五月" == worst_month:
        advice.append("注意'五穷六绝'效应，五月可适当降低仓位")
    if "十月" == best_month:
        advice.append("十月国庆后常有秋季行情，关注布局机会")
    if not advice:
        advice.append("月份效应不显著，以其他分析为主")
    return advice


def month_begin_end_effect(symbol="000300", years=5):
    """
    月初/月末效应分析
    分析每月前N个交易日和最后N个交易日的表现
    """
    df = get_index_kline(symbol, days=years * 252)
    if df is None or len(df) < 200:
        return {"error": "数据不足"}

    date_col = '日期' if '日期' in df.columns else 'date'
    close_col = '收盘' if '收盘' in df.columns else 'close'

    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['return'] = df[close_col].pct_change()

    # 标记每月前5个和后5个交易日
    df['day_in_month'] = df.groupby(['year', 'month']).cumcount() + 1
    df['days_in_month'] = df.groupby(['year', 'month'])['day_in_month'].transform('max')
    df['days_from_end'] = df['days_in_month'] - df['day_in_month']

    result = {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "分析标的": symbol,
        "分析周期": f"{years}年",
    }

    # 月初效应（前5个交易日）
    month_begin = df[df['day_in_month'] <= 5]['return'].dropna()
    if len(month_begin) > 20:
        result["月初效应(前5日)"] = {
            "样本数": len(month_begin),
            "平均涨跌幅": f"{float(month_begin.mean())*100:+.2f}%",
            "上涨概率": f"{float((month_begin > 0).mean())*100:.1f}%",
        }

    # 月末效应（最后5个交易日）
    month_end = df[df['days_from_end'] < 5]['return'].dropna()
    if len(month_end) > 20:
        result["月末效应(后5日)"] = {
            "样本数": len(month_end),
            "平均涨跌幅": f"{float(month_end.mean())*100:+.2f}%",
            "上涨概率": f"{float((month_end > 0).mean())*100:.1f}%",
        }

    # 月中效应（第10-15个交易日）
    month_mid = df[(df['day_in_month'] >= 10) & (df['day_in_month'] <= 15)]['return'].dropna()
    if len(month_mid) > 20:
        result["月中效应(第10-15日)"] = {
            "样本数": len(month_mid),
            "平均涨跌幅": f"{float(month_mid.mean())*100:+.2f}%",
            "上涨概率": f"{float((month_mid > 0).mean())*100:.1f}%",
        }

    return result


def holiday_effect(symbol="000300", years=5):
    """
    节假日效应分析
    分析春节、国庆等长假前后的市场表现
    """
    df = get_index_kline(symbol, days=years * 252)
    if df is None or len(df) < 200:
        return {"error": "数据不足"}

    date_col = '日期' if '日期' in df.columns else 'date'
    close_col = '收盘' if '收盘' in df.columns else 'close'

    df[date_col] = pd.to_datetime(df[date_col])
    df['return'] = df[close_col].pct_change()

    # 中国主要节假日（简化版，按月份估算）
    holidays = {
        "春节(2月)": (2, "前5后5"),
        "清明节(4月)": (4, "前3后3"),
        "劳动节(5月)": (5, "前3后3"),
        "端午节(6月)": (6, "前3后3"),
        "中秋节(9月)": (9, "前3后3"),
        "国庆节(10月)": (10, "前5后5"),
    }

    result = {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "分析标的": symbol,
        "分析周期": f"{years}年",
        "节假日效应": {},
        "说明": "节假日日期按月份估算，实际日期可能有偏差",
    }

    for holiday_name, (month, _) in holidays.items():
        holiday_data = df[df[date_col].dt.month == month]['return'].dropna()
        if len(holiday_data) < 10:
            continue

        avg_return = float(holiday_data.mean()) * 100
        win_rate = float((holiday_data > 0).mean()) * 100

        result["节假日效应"][holiday_name] = {
            "样本数": len(holiday_data),
            "当月平均日涨跌幅": f"{avg_return:+.2f}%",
            "当月上涨概率": f"{win_rate:.1f}%",
        }

    # 特殊分析：春节月份和国庆月份
    if "春节(2月)" in result["节假日效应"]:
        feb = result["节假日效应"]["春节(2月)"]
        feb_avg = float(feb["当月平均日涨跌幅"].replace('%', '').replace('+', ''))
        if feb_avg > 0.05:
            result["春节效应提示"] = "春节月份历史表现较好，可关注节前布局机会"
        else:
            result["春节效应提示"] = "春节月份历史表现一般，注意节前资金面收紧"

    if "国庆节(10月)" in result["节假日效应"]:
        oct_data = result["节假日效应"]["国庆节(10月)"]
        oct_avg = float(oct_data["当月平均日涨跌幅"].replace('%', '').replace('+', ''))
        if oct_avg > 0.05:
            result["国庆效应提示"] = "国庆后历史表现较好，可关注节后布局机会"

    return result


def calendar_effect_summary(symbol="000300", years=5):
    """
    日历效应综合分析
    汇总所有日历效应，给出综合判断
    """
    weekday = weekday_effect(symbol, years=min(years, 3))
    month = month_effect(symbol, years=years)
    begin_end = month_begin_end_effect(symbol, years=years)
    holiday = holiday_effect(symbol, years=years)

    now = datetime.now()
    current_month = now.month
    current_weekday = now.weekday()
    weekday_names = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五"}
    month_names = {
        1: "一月", 2: "二月", 3: "三月", 4: "四月",
        5: "五月", 6: "六月", 7: "七月", 8: "八月",
        9: "九月", 10: "十月", 11: "十一月", 12: "十二月"
    }

    current_weekday_name = weekday_names.get(current_weekday, "未知")
    current_month_name = month_names.get(current_month, "未知")

    # 当前日历效应评估
    current_effects = []

    if "周内效应" in weekday and current_weekday_name in weekday["周内效应"]:
        wd = weekday["周内效应"][current_weekday_name]
        wd_avg = float(wd["平均涨跌幅"].replace('%', '').replace('+', ''))
        if wd_avg > 0.05:
            current_effects.append(f"今天是{current_weekday_name}，历史平均涨幅{wd['平均涨跌幅']}，偏利好")
        elif wd_avg < -0.05:
            current_effects.append(f"今天是{current_weekday_name}，历史平均跌幅{wd['平均涨跌幅']}，偏利空")

    if "月份效应" in month and current_month_name in month["月份效应"]:
        md = month["月份效应"][current_month_name]
        md_avg = float(md["平均涨跌幅"].replace('%', '').replace('+', ''))
        if md_avg > 1:
            current_effects.append(f"当前是{current_month_name}，历史平均涨幅{md['平均涨跌幅']}，偏利好")
        elif md_avg < -1:
            current_effects.append(f"当前是{current_month_name}，历史平均跌幅{md['平均涨跌幅']}，偏利空")

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "分析标的": symbol,
        "当前日期": now.strftime('%Y-%m-%d'),
        "当前星期": current_weekday_name,
        "当前月份": current_month_name,
        "当前日历效应": current_effects if current_effects else ["当前无明显日历效应"],
        "周内效应": weekday,
        "月份效应": month,
        "月初月末效应": begin_end,
        "节假日效应": holiday,
    }


def main():
    parser = argparse.ArgumentParser(description="日历效应分析")
    subparsers = parser.add_subparsers(dest="command")

    weekday_parser = subparsers.add_parser("weekday", help="周内效应分析")
    weekday_parser.add_argument("--symbol", default="000300", help="指数代码")
    weekday_parser.add_argument("--years", type=int, default=3, help="分析年数")

    month_parser = subparsers.add_parser("month", help="月份效应分析")
    month_parser.add_argument("--symbol", default="000300", help="指数代码")
    month_parser.add_argument("--years", type=int, default=5, help="分析年数")

    begin_end_parser = subparsers.add_parser("begin-end", help="月初月末效应分析")
    begin_end_parser.add_argument("--symbol", default="000300", help="指数代码")
    begin_end_parser.add_argument("--years", type=int, default=5, help="分析年数")

    holiday_parser = subparsers.add_parser("holiday", help="节假日效应分析")
    holiday_parser.add_argument("--symbol", default="000300", help="指数代码")
    holiday_parser.add_argument("--years", type=int, default=5, help="分析年数")

    summary_parser = subparsers.add_parser("summary", help="日历效应综合分析")
    summary_parser.add_argument("--symbol", default="000300", help="指数代码")
    summary_parser.add_argument("--years", type=int, default=5, help="分析年数")

    args = parser.parse_args()

    try:
        if args.command == "weekday":
            result = weekday_effect(args.symbol, args.years)
        elif args.command == "month":
            result = month_effect(args.symbol, args.years)
        elif args.command == "begin-end":
            result = month_begin_end_effect(args.symbol, args.years)
        elif args.command == "holiday":
            result = holiday_effect(args.symbol, args.years)
        elif args.command == "summary":
            result = calendar_effect_summary(args.symbol, args.years)
        else:
            parser.print_help()
            sys.exit(0)

        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
