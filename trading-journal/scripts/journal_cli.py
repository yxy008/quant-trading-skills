#!/usr/bin/env python3
"""
交易日志与复盘系统
记录每笔交易，支持复盘分析、错误分类、绩效统计
帮助交易者从历史交易中学习和改进
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
    import pandas as pd
    import numpy as np
except ImportError:
    print("请先安装依赖: pip install pandas numpy")
    sys.exit(1)

from data_utils import get_stock_kline

TRADE_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_log.json")


def _load_trades():
    """加载交易记录"""
    if os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def _save_trades(trades):
    """保存交易记录"""
    with open(TRADE_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(trades, f, ensure_ascii=False, indent=2)


def add_trade(symbol, direction, entry_date, entry_price, quantity,
              exit_date=None, exit_price=None, strategy="手动",
              tags=None, notes=""):
    """
    添加一笔交易记录

    参数:
        symbol: 股票代码
        direction: 方向 buy/sell
        entry_date: 入场日期
        entry_price: 入场价格
        quantity: 数量（股）
        exit_date: 离场日期（可选）
        exit_price: 离场价格（可选）
        strategy: 使用的策略名称
        tags: 标签列表
        notes: 备注
    """
    trades = _load_trades()

    trade = {
        "id": len(trades) + 1,
        "股票代码": symbol,
        "方向": direction,
        "入场日期": entry_date,
        "入场价格": round(float(entry_price), 2),
        "数量": int(quantity),
        "入场金额": round(float(entry_price) * int(quantity), 2),
        "离场日期": exit_date,
        "离场价格": round(float(exit_price), 2) if exit_price else None,
        "策略": strategy,
        "标签": tags or [],
        "备注": notes,
        "创建时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # 如果已离场，计算盈亏
    if exit_date and exit_price:
        exit_price = float(exit_price)
        entry_price = float(entry_price)
        qty = int(quantity)

        if direction == "buy":
            pnl = (exit_price - entry_price) * qty
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:
            pnl = (entry_price - exit_price) * qty
            pnl_pct = (entry_price / exit_price - 1) * 100

        trade["盈亏金额"] = round(pnl, 2)
        trade["盈亏比例"] = round(pnl_pct, 2)
        trade["状态"] = "已平仓"
    else:
        trade["状态"] = "持仓中"

    trades.append(trade)
    _save_trades(trades)

    return {"success": True, "交易ID": trade["id"], "交易": trade}


def update_trade_exit(trade_id, exit_date, exit_price, tags=None, notes=""):
    """更新交易离场信息"""
    trades = _load_trades()

    for trade in trades:
        if trade["id"] == trade_id:
            trade["离场日期"] = exit_date
            trade["离场价格"] = round(float(exit_price), 2)

            entry_price = float(trade["入场价格"])
            exit_price_f = float(exit_price)
            qty = int(trade["数量"])
            direction = trade["方向"]

            if direction == "buy":
                pnl = (exit_price_f - entry_price) * qty
                pnl_pct = (exit_price_f / entry_price - 1) * 100
            else:
                pnl = (entry_price - exit_price_f) * qty
                pnl_pct = (entry_price / exit_price_f - 1) * 100

            trade["盈亏金额"] = round(pnl, 2)
            trade["盈亏比例"] = round(pnl_pct, 2)
            trade["状态"] = "已平仓"

            if tags:
                trade["标签"] = tags
            if notes:
                trade["备注"] = notes

            _save_trades(trades)
            return {"success": True, "交易": trade}

    return {"error": f"未找到交易ID {trade_id}"}


def list_trades(status=None, symbol=None, limit=50):
    """列出交易记录"""
    trades = _load_trades()

    if status:
        trades = [t for t in trades if t.get("状态") == status]
    if symbol:
        trades = [t for t in trades if t["股票代码"] == symbol]

    trades = trades[-limit:]

    return {
        "总交易数": len(_load_trades()),
        "筛选后": len(trades),
        "交易列表": trades,
    }


def trade_review():
    """
    交易复盘分析
    全面分析历史交易，找出规律和问题
    """
    trades = _load_trades()
    closed = [t for t in trades if t.get("状态") == "已平仓"]
    open_trades = [t for t in trades if t.get("状态") == "持仓中"]

    if not closed:
        return {"error": "没有已平仓的交易可供复盘"}

    result = {
        "复盘时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "总交易数": len(trades),
        "已平仓": len(closed),
        "持仓中": len(open_trades),
    }

    # 盈亏统计
    pnl_list = [t["盈亏金额"] for t in closed]
    pnl_pct_list = [t["盈亏比例"] for t in closed]
    win_trades = [t for t in closed if t["盈亏金额"] > 0]
    lose_trades = [t for t in closed if t["盈亏金额"] < 0]
    flat_trades = [t for t in closed if t["盈亏金额"] == 0]

    result["盈亏统计"] = {
        "总盈亏": round(sum(pnl_list), 2),
        "平均盈亏": round(np.mean(pnl_list), 2),
        "平均盈亏比例": f"{np.mean(pnl_pct_list):.2f}%",
        "最大单笔盈利": round(max(pnl_list), 2),
        "最大单笔亏损": round(min(pnl_list), 2),
        "盈利次数": len(win_trades),
        "亏损次数": len(lose_trades),
        "持平次数": len(flat_trades),
        "胜率": f"{len(win_trades)/len(closed)*100:.1f}%",
    }

    # 盈亏比
    if lose_trades:
        avg_win = np.mean([t["盈亏金额"] for t in win_trades]) if win_trades else 0
        avg_loss = abs(np.mean([t["盈亏金额"] for t in lose_trades]))
        result["盈亏统计"]["平均盈利"] = round(avg_win, 2)
        result["盈亏统计"]["平均亏损"] = round(avg_loss, 2)
        result["盈亏统计"]["盈亏比"] = round(avg_win / avg_loss, 2) if avg_loss > 0 else "N/A"

    # 按策略统计
    strategy_stats = {}
    for t in closed:
        s = t.get("策略", "未知")
        if s not in strategy_stats:
            strategy_stats[s] = {"次数": 0, "盈利次数": 0, "总盈亏": 0}
        strategy_stats[s]["次数"] += 1
        strategy_stats[s]["总盈亏"] += t["盈亏金额"]
        if t["盈亏金额"] > 0:
            strategy_stats[s]["盈利次数"] += 1

    for s, stats in strategy_stats.items():
        stats["胜率"] = f"{stats['盈利次数']/stats['次数']*100:.1f}%"
        stats["总盈亏"] = round(stats["总盈亏"], 2)

    result["按策略统计"] = strategy_stats

    # 按标签统计
    tag_stats = {}
    for t in closed:
        for tag in t.get("标签", []):
            if tag not in tag_stats:
                tag_stats[tag] = {"次数": 0, "盈利次数": 0, "总盈亏": 0}
            tag_stats[tag]["次数"] += 1
            tag_stats[tag]["总盈亏"] += t["盈亏金额"]
            if t["盈亏金额"] > 0:
                tag_stats[tag]["盈利次数"] += 1

    for tag, stats in tag_stats.items():
        stats["胜率"] = f"{stats['盈利次数']/stats['次数']*100:.1f}%" if stats["次数"] > 0 else "0%"
        stats["总盈亏"] = round(stats["总盈亏"], 2)

    result["按标签统计"] = tag_stats

    # 错误分类分析
    result["错误分析"] = _analyze_mistakes(lose_trades)

    # 改进建议
    result["改进建议"] = _generate_improvement_advice(result)

    return result


def _analyze_mistakes(lose_trades):
    """分析亏损交易中的常见错误"""
    if not lose_trades:
        return {"说明": "没有亏损交易，表现优秀"}

    mistakes = {
        "不止损": [],
        "追高买入": [],
        "过早止盈后追回": [],
        "仓位过重": [],
        "逆势交易": [],
        "频繁交易": [],
        "其他": [],
    }

    for t in lose_trades:
        tags = t.get("标签", [])
        notes = t.get("备注", "").lower()

        categorized = False
        if "不止损" in tags or "不止损" in notes:
            mistakes["不止损"].append(t)
            categorized = True
        if "追高" in tags or "追高" in notes:
            mistakes["追高买入"].append(t)
            categorized = True
        if "过早止盈" in tags or "追回" in notes:
            mistakes["过早止盈后追回"].append(t)
            categorized = True
        if "重仓" in tags or "仓位过重" in notes:
            mistakes["仓位过重"].append(t)
            categorized = True
        if "逆势" in tags or "逆势" in notes:
            mistakes["逆势交易"].append(t)
            categorized = True
        if "频繁" in tags or "频繁" in notes:
            mistakes["频繁交易"].append(t)
            categorized = True

        if not categorized:
            mistakes["其他"].append(t)

    result = {}
    for category, trades_list in mistakes.items():
        if trades_list:
            total_loss = sum(t["盈亏金额"] for t in trades_list)
            result[category] = {
                "次数": len(trades_list),
                "总亏损": round(total_loss, 2),
                "占比": f"{len(trades_list)/len(lose_trades)*100:.1f}%",
            }

    return result


def _generate_improvement_advice(review_result):
    """根据复盘结果生成改进建议"""
    advice = []

    stats = review_result.get("盈亏统计", {})
    mistakes = review_result.get("错误分析", {})

    # 胜率建议
    win_rate_str = stats.get("胜率", "0%")
    win_rate = float(win_rate_str.replace('%', ''))

    if win_rate < 40:
        advice.append("胜率偏低（<40%），建议减少交易频率，提高入场标准，只做高确定性机会")
    elif win_rate < 50:
        advice.append("胜率一般（<50%），建议优化入场时机，增加确认信号")
    elif win_rate >= 60:
        advice.append("胜率良好（>=60%），保持当前交易纪律")

    # 盈亏比建议
    profit_ratio = stats.get("盈亏比", 0)
    if isinstance(profit_ratio, str):
        profit_ratio = 0
    if profit_ratio < 1.5:
        advice.append("盈亏比偏低（<1.5），建议扩大止盈目标或收紧止损，确保每笔交易风险回报合理")
    elif profit_ratio >= 2:
        advice.append("盈亏比优秀（>=2），继续保持良好的风险回报管理")

    # 错误类型建议
    if "不止损" in mistakes:
        advice.append("不止损是最大亏损来源，严格执行止损纪律，每笔交易入场前必须设好止损位")

    if "追高买入" in mistakes:
        advice.append("追高买入导致亏损，建议等待回调确认后再入场，或使用分批建仓策略")

    if "仓位过重" in mistakes:
        advice.append("仓位过重放大了亏损，建议单票仓位不超过20%，总仓位根据市场状态动态调整")

    if "逆势交易" in mistakes:
        advice.append("逆势交易风险大，建议顺势而为，在上升趋势中只做多，下降趋势中只做空或观望")

    if "频繁交易" in mistakes:
        advice.append("频繁交易增加成本和错误概率，建议降低交易频率，提高每笔交易的质量")

    if not advice:
        advice.append("当前交易表现良好，继续保持纪律，定期复盘优化")

    return advice


def trade_performance_over_time():
    """
    交易绩效时间序列分析
    按月份统计交易表现
    """
    trades = _load_trades()
    closed = [t for t in trades if t.get("状态") == "已平仓"]

    if not closed:
        return {"error": "没有已平仓的交易"}

    monthly = {}
    for t in closed:
        exit_date = t.get("离场日期", "")
        if len(exit_date) >= 7:
            month_key = exit_date[:7]
        else:
            continue

        if month_key not in monthly:
            monthly[month_key] = {"交易次数": 0, "盈利次数": 0, "总盈亏": 0, "总手续费": 0}

        monthly[month_key]["交易次数"] += 1
        monthly[month_key]["总盈亏"] += t.get("盈亏金额", 0)
        if t.get("盈亏金额", 0) > 0:
            monthly[month_key]["盈利次数"] += 1

    result = []
    for month_key in sorted(monthly.keys()):
        m = monthly[month_key]
        result.append({
            "月份": month_key,
            "交易次数": m["交易次数"],
            "盈利次数": m["盈利次数"],
            "胜率": f"{m['盈利次数']/m['交易次数']*100:.1f}%",
            "总盈亏": round(m["总盈亏"], 2),
        })

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "月度统计": result,
    }


def compare_with_benchmark():
    """
    与基准对比
    将自己的交易表现与沪深300对比
    """
    trades = _load_trades()
    closed = [t for t in trades if t.get("状态") == "已平仓"]

    if not closed:
        return {"error": "没有已平仓的交易"}

    from data_utils import get_index_kline

    # 获取交易时间范围
    dates = []
    for t in closed:
        if t.get("入场日期"):
            dates.append(t["入场日期"])
        if t.get("离场日期"):
            dates.append(t["离场日期"])

    if not dates:
        return {"error": "无法确定交易时间范围"}

    start_date = min(dates)
    end_date = max(dates)

    # 获取基准数据
    try:
        bench_df = get_index_kline("000300", days=1500)
        if bench_df is not None and not bench_df.empty:
            close_col = '收盘' if '收盘' in bench_df.columns else 'close'
            date_col = '日期' if '日期' in bench_df.columns else 'date'

            bench_df[date_col] = pd.to_datetime(bench_df[date_col])
            bench_df = bench_df.set_index(date_col).sort_index()

            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            if start_dt in bench_df.index and end_dt in bench_df.index:
                bench_start = float(bench_df.loc[start_dt, close_col])
                bench_end = float(bench_df.loc[end_dt, close_col])
                bench_return = (bench_end / bench_start - 1) * 100
            else:
                bench_df_filtered = bench_df[(bench_df.index >= start_dt) & (bench_df.index <= end_dt)]
                if len(bench_df_filtered) >= 2:
                    bench_start = float(bench_df_filtered[close_col].iloc[0])
                    bench_end = float(bench_df_filtered[close_col].iloc[-1])
                    bench_return = (bench_end / bench_start - 1) * 100
                else:
                    bench_return = None
        else:
            bench_return = None
    except Exception:
        bench_return = None

    total_pnl = sum(t.get("盈亏金额", 0) for t in closed)
    total_invested = sum(t.get("入场金额", 0) for t in closed)
    my_return = (total_pnl / total_invested * 100) if total_invested > 0 else 0

    result = {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "交易时间范围": f"{start_date} 至 {end_date}",
        "我的总盈亏": round(total_pnl, 2),
        "我的总投入": round(total_invested, 2),
        "我的收益率": f"{my_return:.2f}%",
    }

    if bench_return is not None:
        result["基准(沪深300)收益率"] = f"{bench_return:.2f}%"
        alpha = my_return - bench_return
        result["超额收益(Alpha)"] = f"{alpha:.2f}%"
        if alpha > 0:
            result["评价"] = f"跑赢基准{abs(alpha):.2f}%，表现优秀"
        else:
            result["评价"] = f"跑输基准{abs(alpha):.2f}%，需要改进策略"
    else:
        result["基准对比"] = "无法获取基准数据"

    return result


def main():
    parser = argparse.ArgumentParser(description="交易日志与复盘系统")
    subparsers = parser.add_subparsers(dest="command")

    add_parser = subparsers.add_parser("add", help="添加交易记录")
    add_parser.add_argument("--symbol", required=True, help="股票代码")
    add_parser.add_argument("--direction", required=True, choices=["buy", "sell"], help="方向")
    add_parser.add_argument("--entry-date", required=True, help="入场日期 YYYY-MM-DD")
    add_parser.add_argument("--entry-price", type=float, required=True, help="入场价格")
    add_parser.add_argument("--quantity", type=int, required=True, help="数量")
    add_parser.add_argument("--exit-date", help="离场日期")
    add_parser.add_argument("--exit-price", type=float, help="离场价格")
    add_parser.add_argument("--strategy", default="手动", help="策略名称")
    add_parser.add_argument("--tags", help="标签，逗号分隔")
    add_parser.add_argument("--notes", default="", help="备注")

    exit_parser = subparsers.add_parser("exit", help="更新离场信息")
    exit_parser.add_argument("--trade-id", type=int, required=True, help="交易ID")
    exit_parser.add_argument("--exit-date", required=True, help="离场日期")
    exit_parser.add_argument("--exit-price", type=float, required=True, help="离场价格")
    exit_parser.add_argument("--tags", help="标签")
    exit_parser.add_argument("--notes", default="", help="备注")

    subparsers.add_parser("list", help="列出交易记录")
    subparsers.add_parser("review", help="交易复盘分析")
    subparsers.add_parser("monthly", help="月度绩效统计")
    subparsers.add_parser("benchmark", help="与基准对比")

    args = parser.parse_args()

    try:
        if args.command == "add":
            tags = [t.strip() for t in args.tags.split(",")] if args.tags else []
            result = add_trade(
                args.symbol, args.direction, args.entry_date,
                args.entry_price, args.quantity,
                args.exit_date, args.exit_price,
                args.strategy, tags, args.notes
            )
        elif args.command == "exit":
            tags = [t.strip() for t in args.tags.split(",")] if args.tags else []
            result = update_trade_exit(
                args.trade_id, args.exit_date, args.exit_price, tags, args.notes
            )
        elif args.command == "list":
            result = list_trades()
        elif args.command == "review":
            result = trade_review()
        elif args.command == "monthly":
            result = trade_performance_over_time()
        elif args.command == "benchmark":
            result = compare_with_benchmark()
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
