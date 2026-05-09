#!/usr/bin/env python3
"""
市场微观结构分析 - 盘口深度、买卖价差、流动性评估、市场冲击成本
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


# ==================== 买卖价差分析 ====================

def bid_ask_spread_analysis(symbol):
    """
    买卖价差分析
    基于实时行情数据计算买卖价差及相关指标

    返回: {
        "买卖价差": float,
        "相对价差": float,
        "价差分位": float,
        "流动性评级": str,
    }
    """
    try:
        df_spot = ak.stock_zh_a_spot_em()
        if df_spot is None or df_spot.empty:
            return {"error": "无法获取实时行情数据"}

        row = df_spot[df_spot['代码'] == symbol]
        if row.empty:
            return {"error": f"未找到股票 {symbol}"}

        row = row.iloc[0]
        bid = float(row.get('买入', 0)) if pd.notna(row.get('买入')) else 0
        ask = float(row.get('卖出', 0)) if pd.notna(row.get('卖出')) else 0
        latest = float(row.get('最新价', 0)) if pd.notna(row.get('最新价')) else 0
        volume = float(row.get('成交量', 0)) if pd.notna(row.get('成交量')) else 0
        amount = float(row.get('成交额', 0)) if pd.notna(row.get('成交额')) else 0
        turnover = float(row.get('换手率', 0)) if pd.notna(row.get('换手率')) else 0

        if bid <= 0 or ask <= 0:
            return {"error": "买卖盘数据不完整"}

        spread = ask - bid
        relative_spread = spread / ((bid + ask) / 2) * 100 if (bid + ask) > 0 else 0

        # 流动性评级
        if relative_spread < 0.05:
            liquidity = "极高流动性"
            description = "买卖价差极小，适合大额交易"
        elif relative_spread < 0.1:
            liquidity = "高流动性"
            description = "买卖价差小，交易成本低"
        elif relative_spread < 0.2:
            liquidity = "中等流动性"
            description = "买卖价差适中，注意大额交易冲击"
        elif relative_spread < 0.5:
            liquidity = "低流动性"
            description = "买卖价差较大，大额交易需谨慎"
        else:
            liquidity = "极低流动性"
            description = "买卖价差很大，不适合大额交易"

        return {
            "股票代码": symbol,
            "股票名称": str(row.get('名称', '')),
            "买入价": round(bid, 2),
            "卖出价": round(ask, 2),
            "最新价": round(latest, 2),
            "绝对价差": round(spread, 2),
            "相对价差": f"{relative_spread:.3f}%",
            "流动性评级": liquidity,
            "说明": description,
            "成交量": round(volume, 0),
            "成交额": round(amount, 0),
            "换手率": f"{turnover:.2f}%" if turnover else "N/A",
            "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    except Exception as e:
        return {"error": f"价差分析失败: {str(e)}"}


# ==================== 成交量分布分析 ====================

def volume_profile_analysis(symbol, days=60, bins=20):
    """
    成交量分布分析（Volume Profile）
    分析价格在不同区间的成交量分布，识别支撑/阻力位

    参数:
        symbol: 股票代码
        days: 分析天数
        bins: 价格区间数量

    返回: {
        "价值区域": {...},
        "POC": float,
        "支撑位": [...],
        "阻力位": [...],
    }
    """
    df = get_stock_kline(symbol, days=days + 20)
    if df is None or len(df) < 20:
        return {"error": f"无法获取股票 {symbol} 的足够数据"}

    df = df.tail(days)
    close = df['close']
    amount = df['amount']
    high = df['high']
    low = df['low']

    # 价格区间划分
    price_min = float(low.min())
    price_max = float(high.max())
    price_range = price_max - price_min
    bin_size = price_range / bins

    # 计算每个价格区间的成交量
    volume_profile = []
    for i in range(bins):
        bin_low = price_min + i * bin_size
        bin_high = bin_low + bin_size
        bin_center = (bin_low + bin_high) / 2

        # 统计该区间内的成交量
        bin_volume = 0
        for j in range(len(df)):
            row_high = float(high.iloc[j])
            row_low = float(low.iloc[j])
            row_amount = float(amount.iloc[j])

            # 估算该K线在该区间的成交量占比
            if row_high >= bin_low and row_low <= bin_high:
                overlap_low = max(row_low, bin_low)
                overlap_high = min(row_high, bin_high)
                overlap_ratio = (overlap_high - overlap_low) / (row_high - row_low) if row_high > row_low else 1
                bin_volume += row_amount * overlap_ratio

        volume_profile.append({
            "价格区间": f"{bin_low:.2f}-{bin_high:.2f}",
            "中心价": round(bin_center, 2),
            "成交量": round(bin_volume, 0),
            "占比": 0,  # 稍后计算
        })

    # 计算占比
    total_volume = sum(v["成交量"] for v in volume_profile)
    if total_volume > 0:
        for v in volume_profile:
            v["占比"] = f"{v['成交量'] / total_volume * 100:.1f}%"

    # 找到POC（Point of Control - 最大成交量价格）
    poc = max(volume_profile, key=lambda x: x["成交量"])

    # 价值区域（Value Area - 70%成交量区间）
    sorted_profile = sorted(volume_profile, key=lambda x: x["成交量"], reverse=True)
    cumulative_vol = 0
    value_area_bins = []
    for v in sorted_profile:
        cumulative_vol += v["成交量"]
        value_area_bins.append(v)
        if cumulative_vol / total_volume >= 0.7:
            break

    value_area_low = min(v["中心价"] for v in value_area_bins)
    value_area_high = max(v["中心价"] for v in value_area_bins)

    # 识别支撑/阻力位（成交量密集区边界）
    high_volume_bins = [v for v in volume_profile if v["成交量"] > total_volume / bins * 1.5]
    support_levels = []
    resistance_levels = []
    current_price = float(close.iloc[-1])

    for v in high_volume_bins:
        if v["中心价"] < current_price:
            support_levels.append({"价格": v["中心价"], "区间": v["价格区间"], "成交量占比": v["占比"]})
        else:
            resistance_levels.append({"价格": v["中心价"], "区间": v["价格区间"], "成交量占比": v["占比"]})

    support_levels.sort(key=lambda x: x["价格"], reverse=True)
    resistance_levels.sort(key=lambda x: x["价格"])

    return {
        "股票代码": symbol,
        "分析天数": days,
        "价格区间": f"{price_min:.2f} - {price_max:.2f}",
        "当前价格": round(current_price, 2),
        "POC": {
            "价格": poc["中心价"],
            "区间": poc["价格区间"],
            "成交量占比": poc["占比"],
            "说明": "POC是成交量最大的价格水平，是重要的支撑/阻力参考",
        },
        "价值区域": {
            "上沿": round(value_area_high, 2),
            "下沿": round(value_area_low, 2),
            "说明": "价值区域包含70%的成交量，价格在此区间内运行较为正常",
        },
        "支撑位": support_levels[:5],
        "阻力位": resistance_levels[:5],
        "成交量分布": volume_profile,
        "当前价格位置": _price_position_analysis(current_price, value_area_low, value_area_high, poc["中心价"]),
    }


def _price_position_analysis(current_price, va_low, va_high, poc):
    """分析当前价格在价值区域中的位置"""
    if current_price > va_high:
        return "价格高于价值区域上沿，短期偏强但可能面临回调压力"
    elif current_price < va_low:
        return "价格低于价值区域下沿，短期偏弱但可能存在反弹机会"
    elif current_price > poc:
        return "价格在价值区域内偏上方运行，走势偏强"
    elif current_price < poc:
        return "价格在价值区域内偏下方运行，走势偏弱"
    else:
        return "价格在POC附近，处于平衡状态"


# ==================== 流动性评估 ====================

def liquidity_assessment(symbol, days=60):
    """
    综合流动性评估
    基于换手率、成交量、价差、市值等多维度评估

    返回: {
        "流动性评分": 0-100,
        "流动性等级": str,
        "各维度评分": {...},
    }
    """
    df = get_stock_kline(symbol, days=days + 20)
    if df is None or len(df) < 20:
        return {"error": f"无法获取股票 {symbol} 的足够数据"}

    df = df.tail(days)
    amount = df['amount']
    close = df['close']

    # 获取实时数据
    try:
        df_spot = ak.stock_zh_a_spot_em()
        spot_row = df_spot[df_spot['代码'] == symbol]
        if not spot_row.empty:
            spot_row = spot_row.iloc[0]
            market_cap = float(spot_row.get('总市值', 0)) if pd.notna(spot_row.get('总市值')) else 0
            turnover = float(spot_row.get('换手率', 0)) if pd.notna(spot_row.get('换手率')) else 0
            bid = float(spot_row.get('买入', 0)) if pd.notna(spot_row.get('买入')) else 0
            ask = float(spot_row.get('卖出', 0)) if pd.notna(spot_row.get('卖出')) else 0
        else:
            market_cap = 0
            turnover = 0
            bid = 0
            ask = 0
    except Exception:
        market_cap = 0
        turnover = 0
        bid = 0
        ask = 0

    # 日均成交额
    avg_daily_amount = float(amount.mean())

    # 成交额稳定性（变异系数）
    amount_cv = float(amount.std() / amount.mean()) if amount.mean() > 0 else 1

    # 各维度评分（0-100）
    scores = {}

    # 1. 成交额维度
    if avg_daily_amount > 1e9:  # 日均成交额 > 10亿
        scores["成交额"] = 95
    elif avg_daily_amount > 5e8:
        scores["成交额"] = 85
    elif avg_daily_amount > 1e8:
        scores["成交额"] = 70
    elif avg_daily_amount > 5e7:
        scores["成交额"] = 50
    elif avg_daily_amount > 1e7:
        scores["成交额"] = 30
    else:
        scores["成交额"] = 10

    # 2. 换手率维度
    if turnover > 5:
        scores["换手率"] = 90
    elif turnover > 2:
        scores["换手率"] = 75
    elif turnover > 1:
        scores["换手率"] = 60
    elif turnover > 0.5:
        scores["换手率"] = 40
    else:
        scores["换手率"] = 20

    # 3. 市值维度
    if market_cap > 1e11:  # > 1000亿
        scores["市值"] = 95
    elif market_cap > 5e10:
        scores["市值"] = 85
    elif market_cap > 1e10:
        scores["市值"] = 70
    elif market_cap > 5e9:
        scores["市值"] = 50
    else:
        scores["市值"] = 30

    # 4. 价差维度
    if bid > 0 and ask > 0:
        relative_spread = (ask - bid) / ((bid + ask) / 2) * 100
        if relative_spread < 0.05:
            scores["价差"] = 95
        elif relative_spread < 0.1:
            scores["价差"] = 80
        elif relative_spread < 0.2:
            scores["价差"] = 60
        elif relative_spread < 0.5:
            scores["价差"] = 35
        else:
            scores["价差"] = 10
    else:
        scores["价差"] = 50

    # 5. 成交稳定性维度
    if amount_cv < 0.3:
        scores["成交稳定性"] = 90
    elif amount_cv < 0.5:
        scores["成交稳定性"] = 75
    elif amount_cv < 0.8:
        scores["成交稳定性"] = 55
    elif amount_cv < 1.2:
        scores["成交稳定性"] = 35
    else:
        scores["成交稳定性"] = 15

    # 综合评分（加权平均）
    weights = {"成交额": 0.3, "换手率": 0.2, "市值": 0.2, "价差": 0.15, "成交稳定性": 0.15}
    total_score = sum(scores[k] * weights[k] for k in weights)

    # 流动性等级
    if total_score >= 85:
        level = "极佳"
        advice = "流动性极佳，适合大资金操作，市场冲击成本低"
    elif total_score >= 70:
        level = "良好"
        advice = "流动性良好，大资金操作需适当注意时机"
    elif total_score >= 50:
        level = "一般"
        advice = "流动性一般，大额交易建议分批操作"
    elif total_score >= 30:
        level = "较差"
        advice = "流动性较差，不适合大额交易，注意冲击成本"
    else:
        level = "差"
        advice = "流动性很差，建议回避或极小仓位参与"

    return {
        "股票代码": symbol,
        "流动性评分": round(total_score, 1),
        "流动性等级": level,
        "投资建议": advice,
        "各维度评分": scores,
        "关键指标": {
            "日均成交额": f"{avg_daily_amount / 1e8:.2f}亿",
            "换手率": f"{turnover:.2f}%",
            "总市值": f"{market_cap / 1e8:.2f}亿" if market_cap > 0 else "N/A",
            "成交额变异系数": round(amount_cv, 2),
        },
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


# ==================== 市场冲击成本估算 ====================

def market_impact_estimation(symbol, trade_amount, days=60):
    """
    市场冲击成本估算
    估算大额交易对市场价格的影响

    参数:
        symbol: 股票代码
        trade_amount: 计划交易金额（元）
        days: 分析天数

    返回: {
        "预计冲击成本": float,
        "建议交易策略": str,
    }
    """
    df = get_stock_kline(symbol, days=days + 20)
    if df is None or len(df) < 20:
        return {"error": f"无法获取股票 {symbol} 的足够数据"}

    df = df.tail(days)
    amount = df['amount']
    close = df['close']

    avg_daily_amount = float(amount.mean())
    participation_rate = trade_amount / avg_daily_amount if avg_daily_amount > 0 else 1

    # 获取实时价差
    try:
        df_spot = ak.stock_zh_a_spot_em()
        spot_row = df_spot[df_spot['代码'] == symbol]
        if not spot_row.empty:
            spot_row = spot_row.iloc[0]
            bid = float(spot_row.get('买入', 0)) if pd.notna(spot_row.get('买入')) else 0
            ask = float(spot_row.get('卖出', 0)) if pd.notna(spot_row.get('卖出')) else 0
            half_spread = (ask - bid) / 2 if bid > 0 and ask > 0 else 0.01
        else:
            half_spread = 0.01
    except Exception:
        half_spread = 0.01

    # 波动率
    returns = close.pct_change().dropna()
    daily_vol = float(returns.std())

    # 市场冲击模型（Almgren-Chriss简化版）
    # 永久冲击 = eta * sigma * (Q/V)^beta
    # 临时冲击 = epsilon * sigma * (Q/V)^gamma
    eta = 0.1  # 永久冲击系数
    epsilon = 0.05  # 临时冲击系数
    beta = 0.5  # 永久冲击指数
    gamma = 0.6  # 临时冲击指数

    permanent_impact = eta * daily_vol * (participation_rate ** beta)
    temporary_impact = epsilon * daily_vol * (participation_rate ** gamma)
    spread_cost = half_spread * participation_rate

    total_impact_bps = (permanent_impact + temporary_impact + spread_cost) * 10000

    # 交易策略建议
    if participation_rate < 0.01:
        strategy = "参与率极低，可一次性完成交易，冲击成本可忽略"
        split_suggestion = "无需拆分"
    elif participation_rate < 0.05:
        strategy = "参与率较低，可在当日完成交易，冲击成本较小"
        split_suggestion = "建议分1-2笔完成"
    elif participation_rate < 0.1:
        strategy = "参与率适中，建议分批交易以降低冲击"
        split_suggestion = "建议分2-3笔，在1-2天内完成"
    elif participation_rate < 0.2:
        strategy = "参与率较高，冲击成本显著，必须分批交易"
        split_suggestion = "建议分3-5笔，在2-3天内完成"
    elif participation_rate < 0.5:
        strategy = "参与率很高，冲击成本较大，建议使用算法交易"
        split_suggestion = "建议分5-10笔，在3-5天内完成，考虑VWAP/TWAP算法"
    else:
        strategy = "参与率极高，冲击成本很大，强烈建议延长交易周期"
        split_suggestion = "建议分10+笔，在5天以上完成，使用冰山订单"

    return {
        "股票代码": symbol,
        "计划交易金额": f"{trade_amount / 1e4:.2f}万",
        "日均成交额": f"{avg_daily_amount / 1e8:.2f}亿",
        "参与率": f"{participation_rate * 100:.2f}%",
        "预计冲击成本": {
            "永久冲击": f"{permanent_impact * 10000:.2f}bp",
            "临时冲击": f"{temporary_impact * 10000:.2f}bp",
            "价差成本": f"{spread_cost * 10000:.2f}bp",
            "总冲击成本": f"{total_impact_bps:.2f}bp",
            "预计金额损失": f"{trade_amount * total_impact_bps / 10000:.2f}元",
        },
        "交易策略": strategy,
        "拆分建议": split_suggestion,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


# ==================== 价格效率分析 ====================

def price_efficiency_analysis(symbol, days=60):
    """
    价格效率分析
    分析价格走势的随机性/趋势性，判断市场效率

    返回: {
        "效率比率": float,
        "市场效率": str,
    }
    """
    df = get_stock_kline(symbol, days=days + 20)
    if df is None or len(df) < 20:
        return {"error": f"无法获取股票 {symbol} 的足够数据"}

    df = df.tail(days)
    close = df['close']

    # 计算Kaufman效率比率
    # ER = |价格变化| / 价格路径总长度
    price_change = abs(float(close.iloc[-1]) - float(close.iloc[0]))
    path_length = float(sum(abs(close.diff().dropna())))

    efficiency_ratio = price_change / path_length if path_length > 0 else 0

    # 趋势强度
    if efficiency_ratio > 0.5:
        trend_strength = "强趋势"
        description = "价格走势具有强趋势性，趋势跟踪策略效果较好"
    elif efficiency_ratio > 0.3:
        trend_strength = "中等趋势"
        description = "价格有一定趋势性，但噪音较多"
    elif efficiency_ratio > 0.15:
        trend_strength = "弱趋势"
        description = "价格走势偏随机，趋势策略效果有限"
    else:
        trend_strength = "随机游走"
        description = "价格走势接近随机游走，趋势策略难以获利"

    # 自相关分析
    returns = close.pct_change().dropna()
    autocorr_1 = float(returns.autocorr(lag=1)) if len(returns) > 1 else 0
    autocorr_5 = float(returns.autocorr(lag=5)) if len(returns) > 5 else 0

    # 方差比检验（简化版）
    var_1 = float(returns.var())
    var_5 = float(returns.rolling(5).sum().dropna().var()) / 5 if len(returns) >= 5 else var_1
    variance_ratio = var_5 / var_1 if var_1 > 0 else 1

    if variance_ratio > 1.2:
        vr_interpretation = "方差比>1，存在正自相关，价格有趋势性"
    elif variance_ratio < 0.8:
        vr_interpretation = "方差比<1，存在均值回归特征"
    else:
        vr_interpretation = "方差比接近1，价格接近随机游走"

    return {
        "股票代码": symbol,
        "分析天数": days,
        "效率比率": round(efficiency_ratio, 3),
        "趋势强度": trend_strength,
        "说明": description,
        "自相关分析": {
            "1阶自相关": round(autocorr_1, 3),
            "5阶自相关": round(autocorr_5, 3),
            "1阶解读": "正自相关=趋势延续" if autocorr_1 > 0.1 else ("负自相关=均值回归" if autocorr_1 < -0.1 else "无明显自相关"),
        },
        "方差比检验": {
            "方差比": round(variance_ratio, 3),
            "解读": vr_interpretation,
        },
        "策略建议": _get_efficiency_strategy_advice(efficiency_ratio, autocorr_1),
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def _get_efficiency_strategy_advice(efficiency_ratio, autocorr_1):
    """根据价格效率给出策略建议"""
    if efficiency_ratio > 0.4:
        return "价格趋势性强，推荐使用趋势跟踪策略（均线交叉、MACD、海龟交易）"
    elif efficiency_ratio > 0.2:
        return "价格有一定趋势，可结合趋势和震荡策略"
    elif autocorr_1 < -0.1:
        return "价格有均值回归特征，推荐使用均值回归策略、RSI策略"
    else:
        return "价格接近随机游走，建议观望或使用高频统计套利策略"


def main():
    parser = argparse.ArgumentParser(description='市场微观结构分析')
    parser.add_argument('action', choices=[
        'spread', 'volume_profile', 'liquidity', 'impact', 'efficiency'
    ], help='操作类型')
    parser.add_argument('--symbol', required=True, help='股票代码')
    parser.add_argument('--days', type=int, default=60, help='分析天数')
    parser.add_argument('--amount', type=float, default=100000, help='交易金额（用于冲击成本估算）')
    parser.add_argument('--bins', type=int, default=20, help='价格区间数（用于成交量分布）')

    args = parser.parse_args()

    try:
        if args.action == 'spread':
            data = bid_ask_spread_analysis(args.symbol)
        elif args.action == 'volume_profile':
            data = volume_profile_analysis(args.symbol, args.days, args.bins)
        elif args.action == 'liquidity':
            data = liquidity_assessment(args.symbol, args.days)
        elif args.action == 'impact':
            data = market_impact_estimation(args.symbol, args.amount, args.days)
        elif args.action == 'efficiency':
            data = price_efficiency_analysis(args.symbol, args.days)
        else:
            parser.print_help()
            return

        print(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
