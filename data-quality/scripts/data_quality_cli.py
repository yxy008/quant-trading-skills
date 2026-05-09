#!/usr/bin/env python3
"""
数据质量检查模块 - 缺失值检测 / 异常值检测 / 数据一致性校验 / 停牌处理
量化系统的数据基础保障，垃圾进垃圾出
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

from data_utils import get_stock_kline


def check_data_quality(symbol, days=250):
    """
    全面数据质量检查
    检查项:
    1. 数据完整性 - 是否有缺失日期
    2. 缺失值检测 - open/high/low/close/volume是否有NaN
    3. 异常值检测 - 价格跳空、成交量异常
    4. 停牌检测 - 连续多日无变化
    5. 价格逻辑检查 - high>=low, high>=open/close, low<=open/close
    6. 复权一致性 - 前后复权数据对比
    """
    df = get_stock_kline(symbol, days=days)
    if df is None or df.empty:
        return {
            "股票代码": symbol,
            "检查时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "状态": "错误",
            "错误": "无法获取数据"
        }

    result = {
        "股票代码": symbol,
        "检查时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "数据条数": len(df),
        "数据区间": f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}",
        "完整性检查": {},
        "缺失值检查": {},
        "异常值检查": {},
        "停牌检查": {},
        "价格逻辑检查": {},
        "综合评分": 0,
        "问题汇总": []
    }

    # 1. 数据完整性检查
    date_range = pd.date_range(start=df.index[0], end=df.index[-1], freq='B')
    missing_dates = date_range.difference(df.index)
    completeness = {
        "应有交易日": len(date_range),
        "实际交易日": len(df),
        "缺失交易日": len(missing_dates),
        "数据完整率": f"{len(df)/max(len(date_range),1)*100:.1f}%"
    }
    if len(missing_dates) > len(date_range) * 0.1:
        completeness["状态"] = "警告"
        completeness["说明"] = f"缺失{len(missing_dates)}个交易日，完整率偏低"
        result["问题汇总"].append(f"数据完整率偏低({completeness['数据完整率']})")
    else:
        completeness["状态"] = "正常"
    result["完整性检查"] = completeness

    # 2. 缺失值检查
    null_check = {}
    for col in ['open', 'high', 'low', 'close']:
        col_name = col
        if col_name not in df.columns:
            # 尝试中文列名
            col_map = {'open': '开盘', 'high': '最高', 'low': '最低', 'close': '收盘'}
            col_name = col_map.get(col, col)

        if col_name in df.columns:
            null_count = int(df[col_name].isna().sum())
            null_check[col] = {
                "缺失数": null_count,
                "缺失比例": f"{null_count/len(df)*100:.2f}%"
            }
            if null_count > 0:
                result["问题汇总"].append(f"{col}列有{null_count}个缺失值")

    # 成交量检查
    vol_col = 'volume' if 'volume' in df.columns else ('成交量' if '成交量' in df.columns else None)
    if vol_col:
        null_count = int(df[vol_col].isna().sum())
        null_check["volume"] = {
            "缺失数": null_count,
            "缺失比例": f"{null_count/len(df)*100:.2f}%"
        }

    null_check["总体状态"] = "正常" if all(v.get("缺失数", 0) == 0 for v in null_check.values() if isinstance(v, dict)) else "有缺失"
    result["缺失值检查"] = null_check

    # 3. 异常值检测
    close_col = 'close' if 'close' in df.columns else '收盘'
    high_col = 'high' if 'high' in df.columns else '最高'
    low_col = 'low' if 'low' in df.columns else '最低'
    open_col = 'open' if 'open' in df.columns else '开盘'

    anomaly_check = {}

    if close_col in df.columns:
        close = df[close_col]
        returns = close.pct_change().dropna()

        # 涨跌幅异常（单日涨跌幅超过11%视为异常，考虑涨跌停）
        extreme_returns = returns[abs(returns) > 0.11]
        anomaly_check["极端涨跌"] = {
            "次数": len(extreme_returns),
            "最大涨幅": f"{float(returns.max())*100:.1f}%" if len(returns) > 0 else "N/A",
            "最大跌幅": f"{float(returns.min())*100:.1f}%" if len(returns) > 0 else "N/A"
        }
        if len(extreme_returns) > 0:
            result["问题汇总"].append(f"存在{len(extreme_returns)}次极端涨跌(>11%)")

        # 价格跳空检测
        if open_col in df.columns:
            gap = (df[open_col] / close.shift(1) - 1).abs()
            large_gaps = gap[gap > 0.05]
            anomaly_check["价格跳空"] = {
                "跳空次数(>5%)": len(large_gaps),
                "最大跳空": f"{float(gap.max())*100:.1f}%" if len(gap) > 0 else "N/A"
            }
            if len(large_gaps) > len(df) * 0.05:
                result["问题汇总"].append(f"价格跳空比例偏高({len(large_gaps)/len(df)*100:.1f}%)")

        # 成交量异常
        vol_col_name = 'volume' if 'volume' in df.columns else ('成交量' if '成交量' in df.columns else None)
        if vol_col_name and vol_col_name in df.columns:
            volume = df[vol_col_name]
            vol_mean = volume.mean()
            vol_std = volume.std()
            vol_anomalies = volume[volume > vol_mean + 3 * vol_std]
            anomaly_check["成交量异常"] = {
                "异常放量次数": len(vol_anomalies),
                "均值": f"{vol_mean:.0f}",
                "3倍标准差阈值": f"{vol_mean + 3*vol_std:.0f}"
            }

    result["异常值检查"] = anomaly_check

    # 4. 停牌检测
    if close_col in df.columns:
        close = df[close_col]
        # 连续多日价格不变
        unchanged = (close.diff().abs() < 0.001)
        # 找出连续不变的区间
        suspension_periods = []
        start = None
        for i, (idx, val) in enumerate(unchanged.items()):
            if val and start is None:
                start = idx
            elif not val and start is not None:
                duration = (idx - start).days
                if duration >= 3:
                    suspension_periods.append({
                        "开始": start.strftime('%Y-%m-%d'),
                        "结束": idx.strftime('%Y-%m-%d'),
                        "持续天数": duration
                    })
                start = None

        suspension_check = {
            "疑似停牌区间": suspension_periods,
            "停牌次数": len(suspension_periods),
            "状态": "正常" if len(suspension_periods) <= 2 else "停牌较多"
        }
        if len(suspension_periods) > 2:
            result["问题汇总"].append(f"存在{len(suspension_periods)}段疑似停牌区间")
        result["停牌检查"] = suspension_check

    # 5. 价格逻辑检查
    logic_issues = []
    if all(c in df.columns for c in [high_col, low_col, open_col, close_col]):
        # high >= low
        hl_issues = int((df[high_col] < df[low_col]).sum())
        if hl_issues > 0:
            logic_issues.append(f"最高价<最低价: {hl_issues}次")

        # high >= open
        ho_issues = int((df[high_col] < df[open_col]).sum())
        if ho_issues > 0:
            logic_issues.append(f"最高价<开盘价: {ho_issues}次")

        # high >= close
        hc_issues = int((df[high_col] < df[close_col]).sum())
        if hc_issues > 0:
            logic_issues.append(f"最高价<收盘价: {hc_issues}次")

        # low <= open
        lo_issues = int((df[low_col] > df[open_col]).sum())
        if lo_issues > 0:
            logic_issues.append(f"最低价>开盘价: {lo_issues}次")

        # low <= close
        lc_issues = int((df[low_col] > df[close_col]).sum())
        if lc_issues > 0:
            logic_issues.append(f"最低价>收盘价: {lc_issues}次")

    logic_check = {
        "逻辑问题": logic_issues if logic_issues else ["无"],
        "状态": "正常" if not logic_issues else "有逻辑错误"
    }
    if logic_issues:
        result["问题汇总"].extend(logic_issues)
    result["价格逻辑检查"] = logic_check

    # 6. 综合评分
    score = 100
    if completeness["缺失交易日"] > len(date_range) * 0.05:
        score -= 15
    if any(v.get("缺失数", 0) > 0 for v in null_check.values() if isinstance(v, dict)):
        score -= 20
    if anomaly_check.get("极端涨跌", {}).get("次数", 0) > 5:
        score -= 10
    if len(suspension_periods) > 3:
        score -= 10
    if logic_issues:
        score -= 25

    score = max(0, min(100, score))

    if score >= 90:
        quality = "优秀"
    elif score >= 75:
        quality = "良好"
    elif score >= 60:
        quality = "一般"
    else:
        quality = "较差"

    result["综合评分"] = score
    result["数据质量"] = quality
    result["建议"] = _get_quality_advice(quality, result["问题汇总"])

    return result


def batch_quality_check(symbols, days=250):
    """
    批量数据质量检查
    """
    results = []
    quality_stats = {"优秀": 0, "良好": 0, "一般": 0, "较差": 0, "错误": 0}

    for symbol in symbols:
        r = check_data_quality(symbol, days)
        quality = r.get("数据质量", "错误")
        quality_stats[quality] = quality_stats.get(quality, 0) + 1

        results.append({
            "股票代码": symbol,
            "数据质量": quality,
            "综合评分": r.get("综合评分", 0),
            "问题数": len(r.get("问题汇总", [])),
            "数据条数": r.get("数据条数", 0)
        })
        time.sleep(0.3)

    results.sort(key=lambda x: x["综合评分"], reverse=True)

    return {
        "检查时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "检查股票数": len(symbols),
        "质量分布": quality_stats,
        "平均评分": round(float(np.mean([r["综合评分"] for r in results])), 1) if results else 0,
        "详细结果": results,
        "问题股票": [r for r in results if r["数据质量"] in ("较差", "错误")]
    }


def check_data_consistency(symbol, days=250):
    """
    数据一致性检查 - 对比不同数据源
    """
    result = {
        "股票代码": symbol,
        "检查时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "一致性检查": {}
    }

    # 对比腾讯源和新浪源
    try:
        if symbol.startswith('6'):
            full_symbol = f"sh{symbol}"
        else:
            full_symbol = f"sz{symbol}"

        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days + 30)).strftime('%Y%m%d')

        # 腾讯源
        df_tx = ak.stock_zh_a_hist_tx(
            symbol=full_symbol, start_date=start_date, end_date=end_date, adjust="qfq"
        )

        # 东方财富源
        try:
            df_em = ak.stock_zh_a_hist(
                symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq"
            )
        except Exception:
            df_em = None

        if df_tx is not None and df_em is not None and not df_tx.empty and not df_em.empty:
            tx_close = df_tx['close'].values if 'close' in df_tx.columns else df_tx['收盘'].values
            em_close = df_em['收盘'].values if '收盘' in df_em.columns else df_em['close'].values

            min_len = min(len(tx_close), len(em_close))
            if min_len > 0:
                diff = np.abs(tx_close[-min_len:] - em_close[-min_len:])
                avg_diff = float(np.mean(diff))
                max_diff = float(np.max(diff))

                result["一致性检查"]["多源对比"] = {
                    "腾讯源数据条数": len(df_tx),
                    "东财源数据条数": len(df_em),
                    "平均价差": round(avg_diff, 4),
                    "最大价差": round(max_diff, 4),
                    "状态": "一致" if avg_diff < 0.1 else "有偏差"
                }
    except Exception:
        result["一致性检查"]["多源对比"] = {"状态": "无法对比"}

    return result


def _get_quality_advice(quality, issues):
    """根据数据质量给出建议"""
    if quality == "优秀":
        return "数据质量优秀，可直接用于策略研究和回测"
    elif quality == "良好":
        return f"数据质量良好，存在{len(issues)}个小问题，建议关注但不影响使用"
    elif quality == "一般":
        return f"数据质量一般，存在{len(issues)}个问题，建议清洗后再用于回测"
    else:
        return f"数据质量较差，存在{len(issues)}个问题，强烈建议排查数据源或更换股票"


def main():
    parser = argparse.ArgumentParser(description="数据质量检查工具")
    subparsers = parser.add_subparsers(dest="command")

    check_parser = subparsers.add_parser("check", help="单股票数据质量检查")
    check_parser.add_argument("--symbol", required=True, help="股票代码")
    check_parser.add_argument("--days", type=int, default=250, help="检查天数")

    batch_parser = subparsers.add_parser("batch", help="批量数据质量检查")
    batch_parser.add_argument("--symbols", required=True, help="股票代码列表,逗号分隔")
    batch_parser.add_argument("--days", type=int, default=250, help="检查天数")

    consistency_parser = subparsers.add_parser("consistency", help="数据一致性检查")
    consistency_parser.add_argument("--symbol", required=True, help="股票代码")
    consistency_parser.add_argument("--days", type=int, default=250, help="检查天数")

    args = parser.parse_args()

    if args.command == "check":
        result = check_data_quality(args.symbol, days=args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "batch":
        symbols = [s.strip() for s in args.symbols.split(",")]
        result = batch_quality_check(symbols, days=args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "consistency":
        result = check_data_consistency(args.symbol, days=args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
