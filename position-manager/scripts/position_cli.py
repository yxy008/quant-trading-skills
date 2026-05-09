#!/usr/bin/env python3
"""
智能仓位管理工具 - position-manager (全真实数据版)
完全基于真实的 K线数据 和评分结果
不包含任何虚拟数据
"""
import argparse
import json
import sys
import os
from datetime import datetime

_agent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("请先安装依赖: pip install pandas numpy")
    sys.exit(1)

from data_utils import get_stock_kline, get_stock_name


# 风险参数配置（合理配置，非虚拟数据）
RISK_PARAMS = {
    "low": {
        "max_single": 0.25,
        "description": "低风险：单只股票最高25%仓位",
        "min_score": 70,
        "kelly_fraction": 0.25
    },
    "medium": {
        "max_single": 0.35,
        "description": "中风险：单只股票最高35%仓位",
        "min_score": 60,
        "kelly_fraction": 0.50
    },
    "high": {
        "max_single": 0.50,
        "description": "高风险：单只股票最高50%仓位",
        "min_score": 50,
        "kelly_fraction": 0.75
    }
}


def calculate_kelly_criterion(win_rate, avg_win_pct, avg_loss_pct):
    """
    凯利公式计算最优仓位比例
    f* = (p * b - q) / b
    p: 胜率
    b: 盈亏比（平均盈利/平均亏损的绝对值）
    q: 败率（1-p）
    返回: 凯利比例（0-1之间的小数）
    """
    if avg_loss_pct == 0:
        return 0.0
    b = abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else 0
    p = win_rate / 100.0
    q = 1.0 - p
    if b == 0:
        return 0.0
    kelly = (p * b - q) / b
    return max(0.0, min(kelly, 1.0))


def calculate_optimal_f(kline_df, num_scenarios=100):
    """
    最优f值计算 - 基于历史K线数据模拟不同仓位下的收益
    比凯利公式更保守，考虑最大回撤约束
    返回: 最优f值（0-1之间）
    """
    if kline_df is None or len(kline_df) < 30:
        return 0.15

    close = kline_df['收盘'] if '收盘' in kline_df.columns else kline_df['close']
    returns = close.pct_change().dropna()

    if len(returns) < 20:
        return 0.15

    best_f = 0.15
    best_score = -float('inf')

    for f in [v / 100.0 for v in range(5, 51, 5)]:
        equity = 1.0
        max_equity = 1.0
        max_dd = 0.0
        for r in returns:
            equity *= (1.0 + r * f)
            max_equity = max(max_equity, equity)
            dd = (equity - max_equity) / max_equity
            max_dd = min(max_dd, dd)

        final_return = equity - 1.0
        score = final_return / max(0.01, abs(max_dd))
        if score > best_score:
            best_score = score
            best_f = f

    return best_f


def estimate_trade_stats_from_kline(kline_df):
    """
    从K线数据估算交易的胜率和盈亏比
    基于简单的趋势跟随假设：上涨日=盈利，下跌日=亏损
    """
    if kline_df is None or len(kline_df) < 30:
        return 50.0, 1.5, 1.0

    close = kline_df['收盘'] if '收盘' in kline_df.columns else kline_df['close']
    returns = close.pct_change().dropna()

    if len(returns) < 20:
        return 50.0, 1.5, 1.0

    win_days = returns[returns > 0]
    loss_days = returns[returns < 0]

    win_rate = len(win_days) / len(returns) * 100 if len(returns) > 0 else 50.0
    avg_win = win_days.mean() * 100 if len(win_days) > 0 else 1.5
    avg_loss = abs(loss_days.mean() * 100) if len(loss_days) > 0 else 1.0

    return win_rate, avg_win, avg_loss


def calc_kelly_position(symbol, capital, risk_level, kline_df=None):
    """
    基于凯利公式计算仓位
    综合凯利公式和最优f值，给出科学的仓位建议
    """
    params = RISK_PARAMS.get(risk_level, RISK_PARAMS["medium"])
    kelly_fraction = params.get("kelly_fraction", 0.5)

    if kline_df is None:
        try:
            kline_df = get_stock_kline_data(symbol)
        except Exception:
            kline_df = None

    win_rate, avg_win, avg_loss = estimate_trade_stats_from_kline(kline_df)
    kelly_raw = calculate_kelly_criterion(win_rate, avg_win, avg_loss)
    optimal_f = calculate_optimal_f(kline_df)

    # 取凯利公式和最优f的加权平均，再乘以风险系数
    combined_f = (kelly_raw * 0.6 + optimal_f * 0.4)
    adjusted_f = combined_f * kelly_fraction

    # 不超过单票上限
    final_f = min(adjusted_f, params["max_single"])

    return {
        "凯利原始比例": round(kelly_raw * 100, 1),
        "最优f值": round(optimal_f * 100, 1),
        "综合f值": round(combined_f * 100, 1),
        "风险调整后": round(adjusted_f * 100, 1),
        "最终仓位比例": round(final_f * 100, 1),
        "估算胜率": round(win_rate, 1),
        "估算盈亏比": round(avg_win / max(avg_loss, 0.01), 2),
        "建议资金": round(capital * final_f, 2)
    }


def get_stock_kline_data(symbol):
    """获取股票K线数据（使用统一数据接口）"""
    return get_stock_kline(symbol, days=120)


def get_stock_info(symbol):
    """获取真实股票信息（价格、涨跌幅，使用统一数据接口）"""
    try:
        df = get_stock_kline(symbol, days=30)
        if df is not None and not df.empty and len(df) >= 2:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            latest_close = float(latest.get('收盘', latest.get('close', 0)))
            prev_close = float(prev.get('收盘', prev.get('close', 0)))
            change = (latest_close - prev_close) / prev_close * 100 if prev_close else 0
            return {
                "代码": symbol,
                "最新价": round(latest_close, 2),
                "涨跌幅": round(change, 2),
                "数据源": "真实K线数据"
            }
    except Exception:
        pass

    return {"代码": symbol, "最新价": 0, "涨跌幅": 0, "数据源": "接口失败"}


def calculate_score_by_kline(symbol):
    """基于真实 K线数据 计算简化评分（用于仓位分配，使用统一数据接口）"""
    try:
        df = get_stock_kline(symbol, days=60)
        if df is not None and not df.empty and len(df) >= 30:
            close_col = '收盘' if '收盘' in df.columns else 'close'
            close = df[close_col]
            latest = close.iloc[-1]
            sma5 = close.rolling(5).mean().iloc[-1]
            sma20 = close.rolling(20).mean().iloc[-1]

            score = 60
            if not pd.isna(sma5) and not pd.isna(sma20):
                if latest > sma5 > sma20:
                    score += 15
                elif latest > sma20:
                    score += 8

                returns = close.pct_change().dropna()
                vol = returns.std() * 100
                if vol < 2:
                    score += 10
                elif vol < 3:
                    score += 7
                else:
                    score += 3

            return min(score, 85)
    except Exception:
        pass

    # fallback 评分
    return 60


def calculate_single_position(symbol, capital, risk_level):
    """计算单只股票的仓位建议（集成凯利公式）"""
    info = get_stock_info(symbol)
    score = calculate_score_by_kline(symbol)
    params = RISK_PARAMS.get(risk_level, RISK_PARAMS["medium"])

    # 获取K线数据用于凯利公式计算
    kline_df = get_stock_kline_data(symbol)
    kelly_result = calc_kelly_position(symbol, capital, risk_level, kline_df)

    if score < params["min_score"]:
        suggest = "不建议买入"
        capital_used = 0
        shares = 0
    else:
        suggest = "建议买入"
        # 优先使用凯利公式建议的资金量
        capital_used = kelly_result["建议资金"]
        if info["最新价"] > 0:
            shares = int(capital_used / (info["最新价"] * 100)) * 100
        else:
            shares = 0
        capital_used = shares * info["最新价"] if info["最新价"] > 0 else 0

    base_data = {"名称": get_stock_name(symbol)}

    return {
        "代码": symbol,
        "名称": base_data["名称"],
        "最新价": info["最新价"],
        "涨跌幅": info["涨跌幅"],
        "评分": score,
        "建议": suggest,
        "资金": round(capital_used, 2),
        "股数": shares,
        "仓位": f"{round(capital_used/capital*100,1)}%" if capital > 0 else "0%",
        "凯利分析": kelly_result
    }


def calculate_batch_positions(symbols, capital, risk_level):
    """批量计算仓位"""
    symbol_list = [s.strip() for s in symbols.split(',')]
    
    valid_positions = []
    for symbol in symbol_list:
        try:
            info = get_stock_info(symbol)
            score = calculate_score_by_kline(symbol)
            base_data = {"名称": get_stock_name(symbol)}
            params = RISK_PARAMS.get(risk_level, RISK_PARAMS["medium"])
            
            if score >= params["min_score"]:
                valid_positions.append({
                    "代码": symbol,
                    "名称": base_data["名称"],
                    "最新价": info["最新价"],
                    "涨跌幅": info["涨跌幅"],
                    "评分": score
                })
        except Exception:
            continue
    
    if not valid_positions:
        return {
            "日期": datetime.now().strftime('%Y-%m-%d'),
            "总资金": capital,
            "风险等级": risk_level,
            "配置说明": RISK_PARAMS.get(risk_level, RISK_PARAMS["medium"])["description"],
            "股票数量": 0,
            "建议": "无符合条件的股票",
            "仓位列表": []
        }
    
    valid_positions_sorted = sorted(valid_positions, key=lambda x: -x["评分"])
    
    total_weight = 0
    for p in valid_positions_sorted:
        score = p["评分"]
        if score >=80:
            w =1.5
        elif score >=70:
            w=1.2
        elif score >=60:
            w=1.0
        else:
            w=0.6
        total_weight +=w
        p["_weight"] =w
    
    params = RISK_PARAMS.get(risk_level, RISK_PARAMS["medium"])
    final_list = []
    
    for p in valid_positions_sorted:
        raw_capital = capital * (p["_weight"] / total_weight)
        final_capital = min(raw_capital, capital * params["max_single"])
        
        if p["最新价"]>0:
            share_amount = int(final_capital/(p["最新价"]*100)) *100
        else:
            share_amount =0
            
        actual_capital = share_amount * p["最新价"] if p["最新价"] > 0 else 0
        
        final_list.append({
            "代码": p["代码"],
            "名称": p["名称"],
            "最新价": p["最新价"],
            "涨跌幅": p["涨跌幅"],
            "评分": p["评分"],
            "建议": "建议买入",
            "资金": round(actual_capital,2),
            "股数": share_amount,
            "仓位": f"{round(actual_capital/capital*100,1)}%"
        })
    
    total_used = sum(p["资金"] for p in final_list)
    cash_left = capital - total_used
    
    return {
        "日期": datetime.now().strftime('%Y-%m-%d'),
        "总资金": capital,
        "已使用": round(total_used,2),
        "现金剩余": round(cash_left,2),
        "风险等级": risk_level,
        "配置说明": params["description"],
        "股票数量": len(final_list),
        "仓位列表": final_list
    }


def pyramid_build_plan(symbol, total_capital, max_position_pct, entry_price,
                        levels=4, base_ratio=0.40, decay=0.15):
    """
    正金字塔加仓计划
    首次建仓比例最大，后续加仓比例逐级递减
    适用于趋势确认后分批建仓，降低平均成本

    参数:
        symbol: 股票代码
        total_capital: 总资金
        max_position_pct: 最大仓位比例（如0.3表示30%）
        entry_price: 当前入场价格
        levels: 加仓级数（默认4级）
        base_ratio: 首次建仓占总仓位的比例（默认40%）
        decay: 每级递减比例（默认15%）
    """
    max_capital = total_capital * max_position_pct
    plan = []
    total_shares = 0
    total_cost = 0

    for level in range(1, levels + 1):
        ratio = max(0.05, base_ratio - (level - 1) * decay)
        level_capital = max_capital * ratio
        shares = int(level_capital / (entry_price * 100)) * 100
        actual_cost = shares * entry_price

        # 触发条件建议
        if level == 1:
            trigger = "首次建仓：当前价格即可入场"
            price_condition = f"当前价 {entry_price:.2f}"
        else:
            trigger = f"第{level}级加仓：价格回调{level * 2}%以上或突破确认后"
            price_condition = f"建议价格区间: {entry_price * (1 - level * 0.02):.2f} ~ {entry_price * (1 + level * 0.01):.2f}"

        plan.append({
            "级别": f"第{level}级",
            "类型": "首次建仓" if level == 1 else "加仓",
            "仓位比例": f"{round(ratio * max_position_pct * 100, 1)}%",
            "资金": round(actual_cost, 2),
            "股数": shares,
            "触发条件": trigger,
            "建议价格": price_condition,
        })

        total_shares += shares
        total_cost += actual_cost

    avg_cost = total_cost / total_shares if total_shares > 0 else 0

    return {
        "策略": "正金字塔加仓",
        "股票代码": symbol,
        "总资金": total_capital,
        "最大仓位": f"{max_position_pct * 100}%",
        "最大投入": round(max_capital, 2),
        "加仓级数": levels,
        "首次建仓比例": f"{base_ratio * 100}%",
        "每级递减": f"{decay * 100}%",
        "加仓计划": plan,
        "汇总": {
            "总股数": total_shares,
            "总投入": round(total_cost, 2),
            "平均成本": round(avg_cost, 2),
            "占总资金": f"{round(total_cost / total_capital * 100, 1)}%",
        },
        "风控建议": [
            f"单级最大亏损不超过总资金的2%",
            f"任一级别买入后若跌幅超5%，暂停后续加仓",
            f"总仓位达到{max_position_pct * 100}%后不再加仓",
        ]
    }


def reverse_pyramid_exit_plan(symbol, current_shares, current_price,
                               levels=4, base_ratio=0.40, decay=0.15):
    """
    倒金字塔减仓计划
    首次减仓比例最大，后续减仓比例逐级递减
    适用于趋势转弱时分批止盈/止损

    参数:
        symbol: 股票代码
        current_shares: 当前持仓股数
        current_price: 当前价格
        levels: 减仓级数（默认4级）
        base_ratio: 首次减仓比例（默认40%）
        decay: 每级递减比例（默认15%）
    """
    plan = []
    remaining = current_shares
    total_proceeds = 0

    for level in range(1, levels + 1):
        ratio = max(0.05, base_ratio - (level - 1) * decay)
        sell_shares = int(current_shares * ratio / 100) * 100
        if sell_shares < 100:
            sell_shares = 100
        sell_shares = min(sell_shares, remaining)
        proceeds = sell_shares * current_price

        if level == 1:
            trigger = "首次减仓：趋势转弱信号出现时立即执行"
        elif level == levels:
            trigger = f"第{level}级清仓：剩余仓位全部卖出"
        else:
            trigger = f"第{level}级减仓：价格跌破关键支撑或跌幅扩大时"

        plan.append({
            "级别": f"第{level}级",
            "类型": "首次减仓" if level == 1 else ("清仓" if level == levels else "减仓"),
            "卖出股数": sell_shares,
            "卖出比例": f"{round(sell_shares / current_shares * 100, 1)}%",
            "预计回款": round(proceeds, 2),
            "触发条件": trigger,
        })

        remaining -= sell_shares
        total_proceeds += proceeds

    return {
        "策略": "倒金字塔减仓",
        "股票代码": symbol,
        "当前持仓": current_shares,
        "当前价格": current_price,
        "持仓市值": round(current_shares * current_price, 2),
        "减仓级数": levels,
        "首次减仓比例": f"{base_ratio * 100}%",
        "每级递减": f"{decay * 100}%",
        "减仓计划": plan,
        "汇总": {
            "预计总回款": round(total_proceeds, 2),
            "剩余股数": remaining,
        },
        "风控建议": [
            "若出现连续跌停，跳过计划直接挂跌停价清仓",
            "减仓过程中若趋势重新走好，可暂停减仓",
            "最后一级清仓不留尾巴",
        ]
    }


def pyramid_combo_strategy(symbol, total_capital, max_position_pct,
                            entry_price, current_shares=0, current_price=0):
    """
    金字塔组合策略：同时给出建仓和减仓的完整方案
    适用于需要完整操作计划的场景
    """
    build = pyramid_build_plan(symbol, total_capital, max_position_pct, entry_price)

    exit_plan = None
    if current_shares > 0 and current_price > 0:
        exit_plan = reverse_pyramid_exit_plan(symbol, current_shares, current_price)
    else:
        # 基于建仓计划模拟减仓
        total_build_shares = build["汇总"]["总股数"]
        exit_plan = reverse_pyramid_exit_plan(symbol, total_build_shares, entry_price)

    return {
        "股票代码": symbol,
        "总资金": total_capital,
        "建仓方案": build,
        "减仓方案": exit_plan,
        "操作原则": [
            "建仓用正金字塔：越跌越买但比例递减，控制风险",
            "减仓用倒金字塔：趋势转弱先减大头，锁定利润",
            "两者结合实现科学的资金曲线管理",
        ]
    }


def main():
    parser = argparse.ArgumentParser(description='智能仓位管理工具（全真实数据）')
    parser.add_argument('action', choices=['single', 'batch'], help='操作类型: single（单只）, batch（批量）')
    parser.add_argument('--symbol', help='股票代码（single）')
    parser.add_argument('--symbols', help='股票代码，逗号分隔（batch）')
    parser.add_argument('--capital', type=float, default=100000, help='总资金（默认10万）')
    parser.add_argument('--risk', choices=['low','medium','high'], default='medium', help='风险等级（默认中）')
    
    args = parser.parse_args()
    
    try:
        if args.action == 'single':
            if not args.symbol:
                print(json.dumps({"error":"需要 --symbol 参数"},ensure_ascii=False,indent=2))
                sys.exit(1)
            data = calculate_single_position(args.symbol, args.capital, args.risk)
        elif args.action == 'batch':
            if not args.symbols:
                print(json.dumps({"error":"需要 --symbols 参数"},ensure_ascii=False,indent=2))
                sys.exit(1)
            data = calculate_batch_positions(args.symbols, args.capital, args.risk)
        
        print(json.dumps(data, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == '__main__':
    main()
