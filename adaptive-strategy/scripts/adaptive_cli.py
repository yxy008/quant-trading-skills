#!/usr/bin/env python3
"""
自适应策略切换引擎 - 根据市场状态自动选择最优策略
市场状态识别 + 策略-状态映射 + 平滑切换机制
"""
import argparse
import json
import sys
import os
import time
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


# ==================== 市场状态定义 ====================

MARKET_STATES = {
    "strong_bull": {
        "名称": "强势牛市",
        "描述": "指数在多头排列，持续上涨，波动率适中",
        "特征": {"均线": "多头排列", "动量": "强正向", "波动率": "适中"},
        "推荐策略": ["趋势跟踪", "突破策略", "动量策略"],
        "仓位建议": 0.8,
    },
    "weak_bull": {
        "名称": "弱势牛市",
        "描述": "指数偏多但上涨力度减弱，波动率上升",
        "特征": {"均线": "偏多", "动量": "弱正向", "波动率": "偏高"},
        "推荐策略": ["均线策略", "波段操作", "回调买入"],
        "仓位建议": 0.6,
    },
    "sideways_high": {
        "名称": "高波动震荡",
        "描述": "指数横盘但波动率较高，方向不明",
        "特征": {"均线": "缠绕", "动量": "中性", "波动率": "高"},
        "推荐策略": ["网格交易", "布林带策略", "RSI策略"],
        "仓位建议": 0.4,
    },
    "sideways_low": {
        "名称": "低波动震荡",
        "描述": "指数窄幅震荡，波动率低，等待方向选择",
        "特征": {"均线": "缠绕", "动量": "中性", "波动率": "低"},
        "推荐策略": ["观望为主", "轻仓试探", "期权策略"],
        "仓位建议": 0.2,
    },
    "weak_bear": {
        "名称": "弱势熊市",
        "描述": "指数偏空但下跌力度减弱，可能出现反弹",
        "特征": {"均线": "偏空", "动量": "弱负向", "波动率": "偏高"},
        "推荐策略": ["超跌反弹", "RSI超卖", "对冲策略"],
        "仓位建议": 0.3,
    },
    "strong_bear": {
        "名称": "强势熊市",
        "描述": "指数空头排列，持续下跌，波动率放大",
        "特征": {"均线": "空头排列", "动量": "强负向", "波动率": "高"},
        "推荐策略": ["空仓观望", "做空策略", "防御性持仓"],
        "仓位建议": 0.1,
    },
}

# 策略-状态适配评分
STRATEGY_STATE_FITNESS = {
    "趋势跟踪": {
        "strong_bull": 95, "weak_bull": 80, "sideways_high": 30,
        "sideways_low": 20, "weak_bear": 10, "strong_bear": 5,
    },
    "突破策略": {
        "strong_bull": 90, "weak_bull": 75, "sideways_high": 50,
        "sideways_low": 30, "weak_bear": 20, "strong_bear": 10,
    },
    "动量策略": {
        "strong_bull": 90, "weak_bull": 70, "sideways_high": 40,
        "sideways_low": 25, "weak_bear": 15, "strong_bear": 5,
    },
    "均线策略": {
        "strong_bull": 85, "weak_bull": 80, "sideways_high": 40,
        "sideways_low": 35, "weak_bear": 25, "strong_bear": 10,
    },
    "波段操作": {
        "strong_bull": 70, "weak_bull": 85, "sideways_high": 75,
        "sideways_low": 50, "weak_bear": 60, "strong_bear": 20,
    },
    "回调买入": {
        "strong_bull": 80, "weak_bull": 85, "sideways_high": 50,
        "sideways_low": 30, "weak_bear": 40, "strong_bear": 10,
    },
    "网格交易": {
        "strong_bull": 30, "weak_bull": 50, "sideways_high": 90,
        "sideways_low": 85, "weak_bear": 50, "strong_bear": 20,
    },
    "布林带策略": {
        "strong_bull": 50, "weak_bull": 60, "sideways_high": 85,
        "sideways_low": 75, "weak_bear": 55, "strong_bear": 30,
    },
    "RSI策略": {
        "strong_bull": 40, "weak_bull": 55, "sideways_high": 80,
        "sideways_low": 70, "weak_bear": 65, "strong_bear": 40,
    },
    "超跌反弹": {
        "strong_bull": 20, "weak_bull": 30, "sideways_high": 50,
        "sideways_low": 40, "weak_bear": 85, "strong_bear": 60,
    },
    "对冲策略": {
        "strong_bull": 30, "weak_bull": 40, "sideways_high": 60,
        "sideways_low": 50, "weak_bear": 80, "strong_bear": 85,
    },
    "防御性持仓": {
        "strong_bull": 20, "weak_bull": 30, "sideways_high": 50,
        "sideways_low": 60, "weak_bear": 75, "strong_bear": 90,
    },
}


# ==================== 市场状态识别 ====================

def _identify_market_state(df_index):
    """
    识别当前市场状态
    基于均线排列、动量、波动率三维度判断

    返回: {
        "状态代码": "strong_bull",
        "状态名称": "强势牛市",
        "置信度": 85,
        "指标详情": {...}
    }
    """
    if df_index is None or len(df_index) < 60:
        return {
            "状态代码": "sideways_low",
            "状态名称": "低波动震荡",
            "置信度": 20,
            "指标详情": {"说明": "数据不足，默认低波动震荡"},
        }

    close = df_index['收盘'] if '收盘' in df_index.columns else df_index['close']

    # 计算均线
    ma5 = close.rolling(5).mean().iloc[-1]
    ma10 = close.rolling(10).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    latest = close.iloc[-1]

    # 均线排列评分
    ma_alignment_score = 0
    if latest > ma5:
        ma_alignment_score += 1
    if ma5 > ma10:
        ma_alignment_score += 1
    if ma10 > ma20:
        ma_alignment_score += 1
    if ma20 > ma60:
        ma_alignment_score += 1

    # 动量评分
    change_5 = (latest / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
    change_10 = (latest / close.iloc[-10] - 1) * 100 if len(close) >= 10 else 0
    change_20 = (latest / close.iloc[-20] - 1) * 100 if len(close) >= 20 else 0
    change_60 = (latest / close.iloc[-60] - 1) * 100 if len(close) >= 60 else 0

    momentum_score = 0
    if change_5 > 2:
        momentum_score += 2
    elif change_5 > 0:
        momentum_score += 1
    elif change_5 < -2:
        momentum_score -= 2
    elif change_5 < 0:
        momentum_score -= 1

    if change_20 > 5:
        momentum_score += 2
    elif change_20 > 0:
        momentum_score += 1
    elif change_20 < -5:
        momentum_score -= 2
    elif change_20 < 0:
        momentum_score -= 1

    # 波动率评分
    returns = close.pct_change().dropna()
    vol_20 = returns.tail(20).std() * np.sqrt(252) * 100 if len(returns) >= 20 else 20
    vol_60 = returns.tail(60).std() * np.sqrt(252) * 100 if len(returns) >= 60 else vol_20

    # 波动率相对历史水平
    vol_percentile = 50
    if len(returns) >= 120:
        rolling_vol = returns.rolling(60).std() * np.sqrt(252) * 100
        vol_percentile = (rolling_vol.iloc[-1] > rolling_vol).sum() / len(rolling_vol) * 100

    # 状态判定
    if ma_alignment_score >= 3 and momentum_score >= 3:
        state_code = "strong_bull"
        confidence = 85
    elif ma_alignment_score >= 2 and momentum_score >= 1:
        state_code = "weak_bull"
        confidence = 75
    elif ma_alignment_score <= 1 and momentum_score <= -3:
        state_code = "strong_bear"
        confidence = 85
    elif ma_alignment_score <= 1 and momentum_score <= -1:
        state_code = "weak_bear"
        confidence = 75
    elif abs(momentum_score) <= 1:
        if vol_percentile > 70:
            state_code = "sideways_high"
            confidence = 70
        else:
            state_code = "sideways_low"
            confidence = 70
    else:
        # 默认
        if momentum_score > 0:
            state_code = "weak_bull"
            confidence = 55
        elif momentum_score < 0:
            state_code = "weak_bear"
            confidence = 55
        else:
            state_code = "sideways_low"
            confidence = 50

    state_info = MARKET_STATES.get(state_code, MARKET_STATES["sideways_low"])

    return {
        "状态代码": state_code,
        "状态名称": state_info["名称"],
        "置信度": confidence,
        "仓位建议": state_info["仓位建议"],
        "指标详情": {
            "均线排列得分": f"{ma_alignment_score}/4",
            "动量得分": momentum_score,
            "20日年化波动率": f"{vol_20:.1f}%",
            "波动率分位": f"{vol_percentile:.0f}%",
            "5日涨跌": f"{change_5:+.2f}%",
            "20日涨跌": f"{change_20:+.2f}%",
            "60日涨跌": f"{change_60:+.2f}%",
            "最新价": round(float(latest), 2),
            "MA5": round(float(ma5), 2),
            "MA20": round(float(ma20), 2),
            "MA60": round(float(ma60), 2),
        },
    }


# ==================== 策略推荐 ====================

def recommend_strategies(market_state_code, top_n=3):
    """
    根据市场状态推荐最优策略

    返回: [
        {"策略名称": "趋势跟踪", "适配度": 95, "说明": "..."},
        ...
    ]
    """
    fitness = STRATEGY_STATE_FITNESS
    state_info = MARKET_STATES.get(market_state_code, MARKET_STATES["sideways_low"])

    # 计算每个策略的适配度
    strategy_scores = []
    for strategy, state_scores in fitness.items():
        score = state_scores.get(market_state_code, 30)
        strategy_scores.append({
            "策略名称": strategy,
            "适配度": score,
            "说明": _get_strategy_explanation(strategy, market_state_code),
        })

    strategy_scores.sort(key=lambda x: x["适配度"], reverse=True)
    return strategy_scores[:top_n]


def _get_strategy_explanation(strategy, state_code):
    """生成策略推荐说明"""
    explanations = {
        "趋势跟踪": {
            "strong_bull": "强势牛市中趋势跟踪策略表现最佳，顺势而为",
            "weak_bull": "弱势牛市中趋势跟踪仍有效，但需注意回调风险",
            "sideways_high": "震荡市中趋势策略容易反复止损，不推荐",
            "strong_bear": "熊市中趋势跟踪做空方向可用",
        },
        "网格交易": {
            "sideways_high": "高波动震荡是网格交易的最佳环境",
            "sideways_low": "低波动震荡中网格交易收益有限",
            "strong_bull": "牛市中网格容易卖飞，不推荐",
        },
        "RSI策略": {
            "sideways_high": "震荡市中RSI超买超卖信号更可靠",
            "weak_bear": "弱势熊市中RSI超卖反弹策略有效",
        },
        "防御性持仓": {
            "strong_bear": "强势熊市中以防御为主，保住本金",
            "weak_bear": "弱势熊市中防御性持仓降低回撤",
        },
    }

    if strategy in explanations and state_code in explanations[strategy]:
        return explanations[strategy][state_code]

    # 通用说明
    state_name = MARKET_STATES.get(state_code, {}).get("名称", "当前市场")
    return f"在{state_name}环境下，{strategy}的适配度为参考值"


# ==================== 策略切换信号 ====================

def detect_regime_change(df_index, prev_state=None):
    """
    检测市场状态是否发生变化
    返回切换信号和平滑过渡建议
    """
    current = _identify_market_state(df_index)
    current_code = current["状态代码"]

    if prev_state is None:
        return {
            "当前状态": current,
            "是否切换": False,
            "切换类型": "初始状态",
            "过渡建议": "首次运行，建立初始仓位",
            "推荐策略": recommend_strategies(current_code),
        }

    if prev_state == current_code:
        return {
            "当前状态": current,
            "前一状态": MARKET_STATES.get(prev_state, {}).get("名称", prev_state),
            "是否切换": False,
            "切换类型": "维持不变",
            "过渡建议": "市场状态未变，维持当前策略",
            "推荐策略": recommend_strategies(current_code),
        }

    # 状态发生变化
    prev_info = MARKET_STATES.get(prev_state, {})
    curr_info = MARKET_STATES.get(current_code, {})

    # 判断切换类型
    prev_bull = prev_state in ["strong_bull", "weak_bull"]
    curr_bull = current_code in ["strong_bull", "weak_bull"]
    prev_bear = prev_state in ["strong_bear", "weak_bear"]
    curr_bear = current_code in ["strong_bear", "weak_bear"]

    if prev_bull and curr_bear:
        change_type = "牛转熊"
        transition = "市场由牛转熊，建议逐步减仓，切换至防御策略"
    elif prev_bear and curr_bull:
        change_type = "熊转牛"
        transition = "市场由熊转牛，建议逐步加仓，切换至进攻策略"
    elif prev_bull and not curr_bull and not curr_bear:
        change_type = "牛市转震荡"
        transition = "牛市进入震荡，建议降低仓位，切换至震荡策略"
    elif prev_bear and not curr_bull and not curr_bear:
        change_type = "熊市转震荡"
        transition = "熊市进入震荡，可适当参与反弹，控制仓位"
    elif not prev_bull and not prev_bear and curr_bull:
        change_type = "震荡转牛市"
        transition = "震荡转牛市，建议逐步加仓，切换至趋势策略"
    elif not prev_bull and not prev_bear and curr_bear:
        change_type = "震荡转熊市"
        transition = "震荡转熊市，建议减仓防守，控制风险"
    else:
        change_type = "强度变化"
        transition = f"市场状态从{prev_info.get('名称', '')}变为{curr_info.get('名称', '')}"

    return {
        "当前状态": current,
        "前一状态": prev_info.get("名称", prev_state),
        "是否切换": True,
        "切换类型": change_type,
        "过渡建议": transition,
        "仓位变化": f"{prev_info.get('仓位建议', 0.5)*100:.0f}% -> {curr_info.get('仓位建议', 0.5)*100:.0f}%",
        "推荐策略": recommend_strategies(current_code),
    }


# ==================== 综合市场状态分析 ====================

def analyze_market_regime():
    """
    综合分析当前市场状态
    同时分析上证指数和深证成指，综合判断
    """
    sh_df = get_index_kline("sh000001")
    sz_df = get_index_kline("sz399001")

    sh_state = _identify_market_state(sh_df)
    sz_state = _identify_market_state(sz_df)

    # 综合判断
    sh_code = sh_state["状态代码"]
    sz_code = sz_state["状态代码"]

    # 如果两个指数状态一致，取置信度高的
    if sh_code == sz_code:
        combined_code = sh_code
        combined_confidence = max(sh_state["置信度"], sz_state["置信度"])
    else:
        # 取更保守的状态
        state_priority = ["strong_bear", "weak_bear", "sideways_high",
                          "sideways_low", "weak_bull", "strong_bull"]
        sh_rank = state_priority.index(sh_code) if sh_code in state_priority else 3
        sz_rank = state_priority.index(sz_code) if sz_code in state_priority else 3
        combined_code = state_priority[min(sh_rank, sz_rank)]
        combined_confidence = min(sh_state["置信度"], sz_state["置信度"])

    combined_info = MARKET_STATES.get(combined_code, MARKET_STATES["sideways_low"])

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "综合状态": {
            "状态代码": combined_code,
            "状态名称": combined_info["名称"],
            "置信度": combined_confidence,
            "仓位建议": f"{combined_info['仓位建议']*100:.0f}%",
        },
        "上证指数": sh_state,
        "深证成指": sz_state,
        "推荐策略": recommend_strategies(combined_code, top_n=5),
        "策略说明": {
            s["策略名称"]: s["说明"]
            for s in recommend_strategies(combined_code, top_n=5)
        },
    }


def main():
    parser = argparse.ArgumentParser(description='自适应策略切换引擎')
    parser.add_argument('action', choices=[
        'analyze', 'recommend', 'detect_change'
    ], help='操作类型: analyze(综合市场状态分析), recommend(策略推荐), detect_change(检测状态切换)')
    parser.add_argument('--prev_state', type=str, help='前一市场状态代码（用于detect_change）')

    args = parser.parse_args()

    try:
        if args.action == 'analyze':
            data = analyze_market_regime()
            print(json.dumps(data, ensure_ascii=False, indent=2))

        elif args.action == 'recommend':
            sh_df = get_index_kline("sh000001")
            state = _identify_market_state(sh_df)
            data = {
                "当前市场状态": state,
                "推荐策略": recommend_strategies(state["状态代码"], top_n=5),
            }
            print(json.dumps(data, ensure_ascii=False, indent=2))

        elif args.action == 'detect_change':
            sh_df = get_index_kline("sh000001")
            data = detect_regime_change(sh_df, args.prev_state)
            print(json.dumps(data, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
