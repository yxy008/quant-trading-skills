#!/usr/bin/env python3
"""
多策略融合决策系统
整合多个策略的信号，通过加权投票、置信度评估、冲突检测等机制
生成综合交易决策
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

from data_utils import get_stock_kline


def _get_strategy_signals(symbol, strategy_ids, days=250):
    """
    获取多个策略的信号
    返回每个策略的信号序列
    """
    from strategy_cli import get_strategy

    df = get_stock_kline(symbol, days=days)
    if df is None or len(df) < 60:
        return None, None

    all_signals = {}
    for sid in strategy_ids:
        try:
            strategy = get_strategy(sid)
            if strategy is None:
                continue
            result = strategy.generate_signals(df.copy())
            all_signals[sid] = result['signal'].values
        except Exception:
            continue

    return df, all_signals


def _signal_consistency(signals_list):
    """
    计算信号一致性
    返回一致比例和冲突程度
    """
    if not signals_list:
        return 0, 0

    n = len(signals_list)
    if n < 2:
        return 1.0, 0

    buy_count = sum(1 for s in signals_list if s == 1)
    sell_count = sum(1 for s in signals_list if s == -1)
    hold_count = sum(1 for s in signals_list if s == 0)

    max_count = max(buy_count, sell_count, hold_count)
    consistency = max_count / n

    conflict = 0
    if buy_count > 0 and sell_count > 0:
        conflict = min(buy_count, sell_count) / n

    return round(consistency, 3), round(conflict, 3)


def _strategy_weight(strategy_id, market_state=None):
    """
    根据市场状态动态调整策略权重
    """
    base_weights = {
        "ma_cross": 1.0,
        "macd": 1.0,
        "rsi": 0.8,
        "bollinger": 0.8,
        "volume_breakout": 0.9,
        "multi_factor": 1.2,
        "turtle": 1.0,
        "dual_thrust": 0.9,
        "momentum_reversal": 0.8,
        "mean_reversion": 0.7,
    }

    weight = base_weights.get(strategy_id, 1.0)

    if market_state:
        state_weights = {
            "牛市": {"ma_cross": 1.3, "macd": 1.2, "turtle": 1.3, "volume_breakout": 1.2,
                     "mean_reversion": 0.5, "rsi": 0.6},
            "熊市": {"mean_reversion": 1.3, "rsi": 1.2, "bollinger": 1.1,
                     "ma_cross": 0.6, "turtle": 0.5, "volume_breakout": 0.5},
            "横盘震荡": {"bollinger": 1.3, "mean_reversion": 1.3, "rsi": 1.2,
                         "ma_cross": 0.6, "turtle": 0.6},
            "高波动": {"bollinger": 1.2, "dual_thrust": 1.2,
                       "ma_cross": 0.7, "mean_reversion": 0.6},
        }
        if market_state in state_weights:
            weight *= state_weights[market_state].get(strategy_id, 1.0)

    return weight


def strategy_fusion_decision(symbol, strategy_ids=None, days=250,
                              market_state=None, min_consensus=0.5):
    """
    多策略融合决策
    整合多个策略信号，生成综合交易决策

    参数:
        symbol: 股票代码
        strategy_ids: 策略ID列表，默认使用全部策略
        days: 数据天数
        market_state: 市场状态（可选，用于动态调整权重）
        min_consensus: 最低共识阈值
    """
    if strategy_ids is None:
        strategy_ids = ["ma_cross", "macd", "rsi", "bollinger",
                        "volume_breakout", "multi_factor"]

    df, all_signals = _get_strategy_signals(symbol, strategy_ids, days)
    if df is None or all_signals is None:
        return {"error": f"无法获取 {symbol} 的数据"}

    if len(all_signals) < 2:
        return {"error": "至少需要2个策略才能进行融合决策"}

    close_col = '收盘' if '收盘' in df.columns else 'close'
    current_price = float(df[close_col].iloc[-1])

    # 获取最新信号
    latest_signals = {}
    for sid, signals in all_signals.items():
        for i in range(len(signals) - 1, -1, -1):
            if signals[i] != 0:
                latest_signals[sid] = int(signals[i])
                break
        if sid not in latest_signals:
            latest_signals[sid] = 0

    # 计算加权得分
    weighted_score = 0
    total_weight = 0
    strategy_details = []

    for sid, signal in latest_signals.items():
        weight = _strategy_weight(sid, market_state)
        weighted_score += signal * weight
        total_weight += weight
        strategy_details.append({
            "策略ID": sid,
            "信号": "买入" if signal == 1 else ("卖出" if signal == -1 else "持有"),
            "权重": round(weight, 2),
            "加权贡献": round(signal * weight, 2),
        })

    # 归一化得分
    if total_weight > 0:
        normalized_score = weighted_score / total_weight
    else:
        normalized_score = 0

    # 信号一致性
    signal_values = list(latest_signals.values())
    consistency, conflict = _signal_consistency(signal_values)

    # 投票统计
    buy_votes = sum(1 for s in signal_values if s == 1)
    sell_votes = sum(1 for s in signal_values if s == -1)
    hold_votes = sum(1 for s in signal_values if s == 0)
    total_votes = len(signal_values)

    # 决策逻辑
    if consistency >= min_consensus and conflict < 0.3:
        if normalized_score > 0.3:
            decision = "买入"
            confidence = min(95, int(consistency * 100))
            reason = f"多策略一致看多（一致性{consistency*100:.0f}%），加权得分{normalized_score:.2f}"
        elif normalized_score < -0.3:
            decision = "卖出"
            confidence = min(95, int(consistency * 100))
            reason = f"多策略一致看空（一致性{consistency*100:.0f}%），加权得分{normalized_score:.2f}"
        else:
            decision = "观望"
            confidence = int(consistency * 80)
            reason = f"策略信号中性，加权得分{normalized_score:.2f}，建议观望"
    elif conflict > 0.3:
        decision = "观望"
        confidence = max(30, int((1 - conflict) * 70))
        reason = f"策略信号存在冲突（冲突度{conflict*100:.0f}%），建议观望等待信号一致"
    else:
        if normalized_score > 0.15:
            decision = "轻仓买入"
            confidence = int(consistency * 70)
            reason = f"策略偏多但一致性不足，加权得分{normalized_score:.2f}"
        elif normalized_score < -0.15:
            decision = "减仓"
            confidence = int(consistency * 70)
            reason = f"策略偏空但一致性不足，加权得分{normalized_score:.2f}"
        else:
            decision = "观望"
            confidence = 50
            reason = "策略信号不明确，建议观望"

    # 历史信号回溯
    signal_history = _backtest_fusion_signals(all_signals, df, min_consensus)

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "股票代码": symbol,
        "当前价格": round(current_price, 2),
        "融合策略数": len(strategy_ids),
        "有效策略数": len(all_signals),
        "投票统计": {
            "买入": buy_votes,
            "卖出": sell_votes,
            "持有": hold_votes,
            "总票数": total_votes,
        },
        "信号一致性": f"{consistency*100:.0f}%",
        "信号冲突度": f"{conflict*100:.0f}%",
        "加权得分": round(normalized_score, 3),
        "各策略详情": strategy_details,
        "融合决策": {
            "决策": decision,
            "置信度": confidence,
            "原因": reason,
        },
        "历史信号统计": signal_history,
        "操作建议": _fusion_advice(decision, confidence, consistency, conflict),
    }


def _backtest_fusion_signals(all_signals, df, min_consensus):
    """
    回溯融合信号的历史表现
    """
    if not all_signals:
        return {}

    min_len = min(len(s) for s in all_signals.values())
    if min_len < 60:
        return {"说明": "数据不足，无法回溯"}

    buy_signals = 0
    sell_signals = 0
    correct_buy = 0
    correct_sell = 0

    for i in range(60, min_len - 5):
        current_signals = []
        for sid, signals in all_signals.items():
            current_signals.append(int(signals[i]))

        consistency, conflict = _signal_consistency(current_signals)

        if consistency < min_consensus or conflict > 0.3:
            continue

        buy_count = sum(1 for s in current_signals if s == 1)
        sell_count = sum(1 for s in current_signals if s == -1)

        if buy_count > sell_count and buy_count > len(current_signals) * min_consensus:
            buy_signals += 1
            future_return = (df['close'].iloc[i + 5] / df['close'].iloc[i] - 1)
            if future_return > 0:
                correct_buy += 1
        elif sell_count > buy_count and sell_count > len(current_signals) * min_consensus:
            sell_signals += 1
            future_return = (df['close'].iloc[i + 5] / df['close'].iloc[i] - 1)
            if future_return < 0:
                correct_sell += 1

    return {
        "回溯周期": f"{min_len}个交易日",
        "融合买入信号数": buy_signals,
        "融合卖出信号数": sell_signals,
        "买入准确率": f"{correct_buy/buy_signals*100:.1f}%" if buy_signals > 0 else "N/A",
        "卖出准确率": f"{correct_sell/sell_signals*100:.1f}%" if sell_signals > 0 else "N/A",
    }


def _fusion_advice(decision, confidence, consistency, conflict):
    """根据融合决策给出操作建议"""
    if decision == "买入" and confidence >= 80:
        return "多策略共振买入信号，可积极建仓，建议分2-3批入场"
    elif decision == "买入":
        return "策略偏多，可轻仓试探，等待更多确认信号"
    elif decision == "卖出" and confidence >= 80:
        return "多策略共振卖出信号，建议及时减仓或清仓"
    elif decision == "卖出":
        return "策略偏空，建议逐步减仓，控制风险"
    elif decision == "轻仓买入":
        return "信号偏多但不够强，可用小仓位试探，严格止损"
    elif decision == "减仓":
        return "信号偏空但不够强，可适当降低仓位，观察后续走势"
    else:
        if conflict > 0.3:
            return "策略信号冲突，建议观望，等待信号趋于一致"
        return "信号不明确，建议观望，等待更清晰的交易机会"


def multi_symbol_fusion(symbols, strategy_ids=None, market_state=None):
    """
    多股票融合决策
    对多个股票分别进行融合决策，给出排序
    """
    if not symbols:
        return {"error": "请提供股票列表"}

    results = []
    for symbol in symbols[:20]:
        result = strategy_fusion_decision(
            symbol, strategy_ids=strategy_ids,
            market_state=market_state
        )
        if "error" not in result:
            fusion = result.get("融合决策", {})
            results.append({
                "股票代码": symbol,
                "当前价格": result.get("当前价格", 0),
                "决策": fusion.get("决策", "未知"),
                "置信度": fusion.get("置信度", 0),
                "加权得分": result.get("加权得分", 0),
                "一致性": result.get("信号一致性", "0%"),
            })

    # 按加权得分排序
    results.sort(key=lambda x: x["加权得分"], reverse=True)

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "分析股票数": len(results),
        "融合排名": results,
        "买入候选": [r for r in results if r["决策"] in ("买入", "轻仓买入")],
        "卖出候选": [r for r in results if r["决策"] in ("卖出", "减仓")],
    }


def main():
    parser = argparse.ArgumentParser(description="多策略融合决策系统")
    subparsers = parser.add_subparsers(dest="command")

    single_parser = subparsers.add_parser("single", help="单股票融合决策")
    single_parser.add_argument("--symbol", required=True, help="股票代码")
    single_parser.add_argument("--strategies", default="ma_cross,macd,rsi,bollinger,volume_breakout,multi_factor",
                               help="策略ID列表，逗号分隔")
    single_parser.add_argument("--market-state", help="市场状态（牛市/熊市/震荡等）")
    single_parser.add_argument("--min-consensus", type=float, default=0.5,
                               help="最低共识阈值")

    multi_parser = subparsers.add_parser("multi", help="多股票融合决策")
    multi_parser.add_argument("--symbols", required=True, help="股票代码列表，逗号分隔")
    multi_parser.add_argument("--strategies", default="ma_cross,macd,rsi,bollinger,volume_breakout,multi_factor",
                              help="策略ID列表，逗号分隔")
    multi_parser.add_argument("--market-state", help="市场状态")

    args = parser.parse_args()

    try:
        if args.command == "single":
            strategy_ids = [s.strip() for s in args.strategies.split(",")]
            result = strategy_fusion_decision(
                args.symbol,
                strategy_ids=strategy_ids,
                market_state=args.market_state,
                min_consensus=args.min_consensus,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))

        elif args.command == "multi":
            symbols = [s.strip() for s in args.symbols.split(",")]
            strategy_ids = [s.strip() for s in args.strategies.split(",")]
            result = multi_symbol_fusion(
                symbols,
                strategy_ids=strategy_ids,
                market_state=args.market_state,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))

        else:
            parser.print_help()

    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
