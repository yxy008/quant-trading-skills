#!/usr/bin/env python3
"""
期权与衍生品分析系统
支持期权Greeks计算、隐含波动率分析、期权策略盈亏分析、50ETF/300ETF期权数据
"""
import argparse
import json
import sys
import os
from datetime import datetime
from math import log, sqrt, exp, pi
from statistics import NormalDist

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


def normal_cdf(x):
    """标准正态分布累积分布函数"""
    return NormalDist().cdf(x)


def normal_pdf(x):
    """标准正态分布概率密度函数"""
    return NormalDist().pdf(x)


# ==================== 期权Greeks计算 ====================

def calculate_option_greeks(option_type, S, K, T, r, sigma, q=0):
    """
    Black-Scholes期权定价与Greeks计算

    参数:
        option_type: 'call' 或 'put'
        S: 标的资产当前价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        q: 股息率

    返回: {
        "理论价格": float,
        "Delta": float,
        "Gamma": float,
        "Theta": float,
        "Vega": float,
        "Rho": float,
    }
    """
    if T <= 0 or sigma <= 0:
        return {"error": "到期时间和波动率必须大于0"}

    d1 = (log(S / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == 'call':
        price = S * exp(-q * T) * normal_cdf(d1) - K * exp(-r * T) * normal_cdf(d2)
        delta = exp(-q * T) * normal_cdf(d1)
        theta = (-S * exp(-q * T) * normal_pdf(d1) * sigma / (2 * sqrt(T))
                 - r * K * exp(-r * T) * normal_cdf(d2)
                 + q * S * exp(-q * T) * normal_cdf(d1))
    else:
        price = K * exp(-r * T) * normal_cdf(-d2) - S * exp(-q * T) * normal_cdf(-d1)
        delta = -exp(-q * T) * normal_cdf(-d1)
        theta = (-S * exp(-q * T) * normal_pdf(d1) * sigma / (2 * sqrt(T))
                 + r * K * exp(-r * T) * normal_cdf(-d2)
                 - q * S * exp(-q * T) * normal_cdf(-d1))

    gamma = exp(-q * T) * normal_pdf(d1) / (S * sigma * sqrt(T))
    vega = S * exp(-q * T) * normal_pdf(d1) * sqrt(T) / 100
    rho = K * T * exp(-r * T) * normal_cdf(d2) / 100 if option_type == 'call' else -K * T * exp(-r * T) * normal_cdf(-d2) / 100

    return {
        "理论价格": round(price, 4),
        "Delta": round(delta, 4),
        "Gamma": round(gamma, 4),
        "Theta": round(theta / 365, 4),
        "Vega": round(vega, 4),
        "Rho": round(rho, 4),
        "d1": round(d1, 4),
        "d2": round(d2, 4),
    }


def implied_volatility(option_type, market_price, S, K, T, r, q=0, precision=0.0001, max_iter=100):
    """
    计算隐含波动率（Newton-Raphson迭代法）

    参数:
        option_type: 'call' 或 'put'
        market_price: 期权市场价格
        S: 标的资产价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        q: 股息率
        precision: 精度
        max_iter: 最大迭代次数

    返回: 隐含波动率
    """
    sigma = 0.3
    for i in range(max_iter):
        greeks = calculate_option_greeks(option_type, S, K, T, r, sigma, q)
        if "error" in greeks:
            return None
        price = greeks["理论价格"]
        vega = greeks["Vega"] * 100
        diff = price - market_price
        if abs(diff) < precision:
            return round(sigma, 4)
        if vega < 0.0001:
            sigma = sigma * 0.5 if diff > 0 else sigma * 1.5
        else:
            sigma = sigma - diff / vega
        sigma = max(0.01, min(sigma, 2.0))
    return round(sigma, 4)


# ==================== 期权策略分析 ====================

def option_strategy_analysis(strategy_type, S, K_list, T, r, sigma, q=0):
    """
    期权策略盈亏分析

    支持的策略:
        - covered_call: 备兑看涨（持有标的+卖出看涨）
        - protective_put: 保护性看跌（持有标的+买入看跌）
        - bull_spread: 牛市价差（买入低行权价看涨+卖出高行权价看涨）
        - bear_spread: 熊市价差（买入高行权价看跌+卖出低行权价看跌）
        - straddle: 跨式（同时买入看涨和看跌）
        - strangle: 宽跨式（买入不同行权价的看涨和看跌）
        - butterfly: 蝶式价差
        - iron_condor: 铁鹰式

    参数:
        strategy_type: 策略类型
        S: 标的资产价格
        K_list: 行权价列表
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        q: 股息率

    返回: 策略分析结果
    """
    strategies = {
        "covered_call": _analyze_covered_call,
        "protective_put": _analyze_protective_put,
        "bull_spread": _analyze_bull_spread,
        "bear_spread": _analyze_bear_spread,
        "straddle": _analyze_straddle,
        "strangle": _analyze_strangle,
        "butterfly": _analyze_butterfly,
        "iron_condor": _analyze_iron_condor,
    }

    if strategy_type not in strategies:
        return {"error": f"不支持的策略类型: {strategy_type}，支持: {list(strategies.keys())}"}

    return strategies[strategy_type](S, K_list, T, r, sigma, q)


def _analyze_covered_call(S, K_list, T, r, sigma, q=0):
    """备兑看涨策略"""
    if len(K_list) < 1:
        return {"error": "备兑看涨需要1个行权价"}
    K = K_list[0]
    call = calculate_option_greeks('call', S, K, T, r, sigma, q)
    if "error" in call:
        return call

    call_price = call["理论价格"]
    net_cost = S - call_price
    max_profit = K - net_cost
    max_loss = net_cost
    breakeven = net_cost

    price_range = np.linspace(S * 0.7, S * 1.3, 50)
    pnl = []
    for p in price_range:
        stock_pnl = p - S
        if p <= K:
            option_pnl = call_price
        else:
            option_pnl = call_price - (p - K)
        pnl.append(float(stock_pnl + option_pnl))

    return {
        "策略名称": "备兑看涨(Covered Call)",
        "构建方式": f"持有标的(成本{S}) + 卖出看涨期权(K={K}, 权利金{call_price})",
        "净成本": round(net_cost, 4),
        "最大收益": round(max_profit, 4),
        "最大亏损": round(max_loss, 4),
        "盈亏平衡点": round(breakeven, 4),
        "收益率": f"{max_profit / net_cost * 100:.2f}%" if net_cost > 0 else "N/A",
        "适用场景": "温和看涨或横盘，通过卖出期权获取额外收益",
        "风险提示": "如果标的大幅上涨，收益被锁定在行权价，错失超额收益",
        "盈亏曲线": {
            "价格区间": [round(p, 2) for p in price_range.tolist()],
            "盈亏": [round(v, 4) for v in pnl],
        },
    }


def _analyze_protective_put(S, K_list, T, r, sigma, q=0):
    """保护性看跌策略"""
    if len(K_list) < 1:
        return {"error": "保护性看跌需要1个行权价"}
    K = K_list[0]
    put = calculate_option_greeks('put', S, K, T, r, sigma, q)
    if "error" in put:
        return put

    put_price = put["理论价格"]
    net_cost = S + put_price
    max_profit = float('inf')
    max_loss = net_cost - K
    breakeven = net_cost

    price_range = np.linspace(S * 0.7, S * 1.3, 50)
    pnl = []
    for p in price_range:
        stock_pnl = p - S
        if p >= K:
            option_pnl = -put_price
        else:
            option_pnl = (K - p) - put_price
        pnl.append(float(stock_pnl + option_pnl))

    return {
        "策略名称": "保护性看跌(Protective Put)",
        "构建方式": f"持有标的(成本{S}) + 买入看跌期权(K={K}, 权利金{put_price})",
        "净成本": round(net_cost, 4),
        "最大收益": "无限",
        "最大亏损": round(max_loss, 4),
        "盈亏平衡点": round(breakeven, 4),
        "保险成本": f"{put_price / S * 100:.2f}%",
        "适用场景": "看好标的但担心短期下跌风险，为持仓买保险",
        "盈亏曲线": {
            "价格区间": [round(p, 2) for p in price_range.tolist()],
            "盈亏": [round(v, 4) for v in pnl],
        },
    }


def _analyze_bull_spread(S, K_list, T, r, sigma, q=0):
    """牛市看涨价差"""
    if len(K_list) < 2:
        return {"error": "牛市价差需要2个行权价(K1<K2)"}
    K1, K2 = sorted(K_list[:2])
    call1 = calculate_option_greeks('call', S, K1, T, r, sigma, q)
    call2 = calculate_option_greeks('call', S, K2, T, r, sigma, q)
    if "error" in call1 or "error" in call2:
        return call1 if "error" in call1 else call2

    net_debit = call1["理论价格"] - call2["理论价格"]
    max_profit = (K2 - K1) - net_debit
    max_loss = net_debit
    breakeven = K1 + net_debit

    price_range = np.linspace(S * 0.7, S * 1.3, 50)
    pnl = []
    for p in price_range:
        if p <= K1:
            val = -net_debit
        elif p >= K2:
            val = (K2 - K1) - net_debit
        else:
            val = (p - K1) - net_debit
        pnl.append(float(val))

    return {
        "策略名称": "牛市看涨价差(Bull Call Spread)",
        "构建方式": f"买入看涨(K1={K1}, 权利金{call1['理论价格']}) + 卖出看涨(K2={K2}, 权利金{call2['理论价格']})",
        "净支出": round(net_debit, 4),
        "最大收益": round(max_profit, 4),
        "最大亏损": round(max_loss, 4),
        "盈亏平衡点": round(breakeven, 4),
        "收益风险比": f"{max_profit / max_loss:.2f}" if max_loss > 0 else "N/A",
        "适用场景": "温和看涨，降低权利金成本，但限制上行空间",
        "盈亏曲线": {
            "价格区间": [round(p, 2) for p in price_range.tolist()],
            "盈亏": [round(v, 4) for v in pnl],
        },
    }


def _analyze_bear_spread(S, K_list, T, r, sigma, q=0):
    """熊市看跌价差"""
    if len(K_list) < 2:
        return {"error": "熊市价差需要2个行权价(K1<K2)"}
    K1, K2 = sorted(K_list[:2])
    put1 = calculate_option_greeks('put', S, K1, T, r, sigma, q)
    put2 = calculate_option_greeks('put', S, K2, T, r, sigma, q)
    if "error" in put1 or "error" in put2:
        return put1 if "error" in put1 else put2

    net_debit = put2["理论价格"] - put1["理论价格"]
    max_profit = (K2 - K1) - net_debit
    max_loss = net_debit
    breakeven = K2 - net_debit

    price_range = np.linspace(S * 0.7, S * 1.3, 50)
    pnl = []
    for p in price_range:
        if p >= K2:
            val = -net_debit
        elif p <= K1:
            val = (K2 - K1) - net_debit
        else:
            val = (K2 - p) - net_debit
        pnl.append(float(val))

    return {
        "策略名称": "熊市看跌价差(Bear Put Spread)",
        "构建方式": f"买入看跌(K2={K2}, 权利金{put2['理论价格']}) + 卖出看跌(K1={K1}, 权利金{put1['理论价格']})",
        "净支出": round(net_debit, 4),
        "最大收益": round(max_profit, 4),
        "最大亏损": round(max_loss, 4),
        "盈亏平衡点": round(breakeven, 4),
        "收益风险比": f"{max_profit / max_loss:.2f}" if max_loss > 0 else "N/A",
        "适用场景": "温和看跌，降低权利金成本",
        "盈亏曲线": {
            "价格区间": [round(p, 2) for p in price_range.tolist()],
            "盈亏": [round(v, 4) for v in pnl],
        },
    }


def _analyze_straddle(S, K_list, T, r, sigma, q=0):
    """跨式策略"""
    if len(K_list) < 1:
        return {"error": "跨式策略需要1个行权价"}
    K = K_list[0]
    call = calculate_option_greeks('call', S, K, T, r, sigma, q)
    put = calculate_option_greeks('put', S, K, T, r, sigma, q)
    if "error" in call or "error" in put:
        return call if "error" in call else put

    total_cost = call["理论价格"] + put["理论价格"]
    breakeven_up = K + total_cost
    breakeven_down = K - total_cost

    price_range = np.linspace(S * 0.5, S * 1.5, 50)
    pnl = []
    for p in price_range:
        call_pnl = max(p - K, 0) - call["理论价格"]
        put_pnl = max(K - p, 0) - put["理论价格"]
        pnl.append(float(call_pnl + put_pnl))

    return {
        "策略名称": "跨式策略(Long Straddle)",
        "构建方式": f"买入看涨(K={K}, 权利金{call['理论价格']}) + 买入看跌(K={K}, 权利金{put['理论价格']})",
        "总成本": round(total_cost, 4),
        "最大收益": "无限（大幅波动时）",
        "最大亏损": round(total_cost, 4),
        "上盈亏平衡点": round(breakeven_up, 4),
        "下盈亏平衡点": round(breakeven_down, 4),
        "所需波动": f"{(breakeven_up / S - 1) * 100:.1f}%",
        "适用场景": "预期标的将大幅波动但方向不确定（如财报发布前）",
        "风险提示": "如果标的横盘不动，将损失全部权利金",
        "盈亏曲线": {
            "价格区间": [round(p, 2) for p in price_range.tolist()],
            "盈亏": [round(v, 4) for v in pnl],
        },
    }


def _analyze_strangle(S, K_list, T, r, sigma, q=0):
    """宽跨式策略"""
    if len(K_list) < 2:
        return {"error": "宽跨式策略需要2个行权价(K1<K2)"}
    K1, K2 = sorted(K_list[:2])
    call = calculate_option_greeks('call', S, K2, T, r, sigma, q)
    put = calculate_option_greeks('put', S, K1, T, r, sigma, q)
    if "error" in call or "error" in put:
        return call if "error" in call else put

    total_cost = call["理论价格"] + put["理论价格"]
    breakeven_up = K2 + total_cost
    breakeven_down = K1 - total_cost

    price_range = np.linspace(S * 0.5, S * 1.5, 50)
    pnl = []
    for p in price_range:
        call_pnl = max(p - K2, 0) - call["理论价格"]
        put_pnl = max(K1 - p, 0) - put["理论价格"]
        pnl.append(float(call_pnl + put_pnl))

    return {
        "策略名称": "宽跨式策略(Long Strangle)",
        "构建方式": f"买入看涨(K2={K2}, 权利金{call['理论价格']}) + 买入看跌(K1={K1}, 权利金{put['理论价格']})",
        "总成本": round(total_cost, 4),
        "最大收益": "无限（大幅波动时）",
        "最大亏损": round(total_cost, 4),
        "上盈亏平衡点": round(breakeven_up, 4),
        "下盈亏平衡点": round(breakeven_down, 4),
        "适用场景": "预期大幅波动但想降低成本（比跨式便宜但需要更大波动）",
        "盈亏曲线": {
            "价格区间": [round(p, 2) for p in price_range.tolist()],
            "盈亏": [round(v, 4) for v in pnl],
        },
    }


def _analyze_butterfly(S, K_list, T, r, sigma, q=0):
    """蝶式价差"""
    if len(K_list) < 3:
        return {"error": "蝶式价差需要3个行权价(K1<K2<K3)，K2=(K1+K3)/2"}
    K1, K2, K3 = sorted(K_list[:3])
    call1 = calculate_option_greeks('call', S, K1, T, r, sigma, q)
    call2 = calculate_option_greeks('call', S, K2, T, r, sigma, q)
    call3 = calculate_option_greeks('call', S, K3, T, r, sigma, q)
    if "error" in call1 or "error" in call2 or "error" in call3:
        return call1 if "error" in call1 else (call2 if "error" in call2 else call3)

    net_debit = call1["理论价格"] - 2 * call2["理论价格"] + call3["理论价格"]
    max_profit = (K2 - K1) - net_debit
    max_loss = net_debit
    breakeven_low = K1 + net_debit
    breakeven_high = K3 - net_debit

    price_range = np.linspace(S * 0.7, S * 1.3, 50)
    pnl = []
    for p in price_range:
        val1 = max(p - K1, 0)
        val2 = max(p - K2, 0)
        val3 = max(p - K3, 0)
        val = val1 - 2 * val2 + val3 - net_debit
        pnl.append(float(val))

    return {
        "策略名称": "蝶式价差(Butterfly Spread)",
        "构建方式": f"买K1={K1}看涨 + 卖2份K2={K2}看涨 + 买K3={K3}看涨",
        "净支出": round(net_debit, 4),
        "最大收益": round(max_profit, 4),
        "最大亏损": round(max_loss, 4),
        "下盈亏平衡点": round(breakeven_low, 4),
        "上盈亏平衡点": round(breakeven_high, 4),
        "适用场景": "预期标的在K2附近小幅波动，低风险低收益",
        "盈亏曲线": {
            "价格区间": [round(p, 2) for p in price_range.tolist()],
            "盈亏": [round(v, 4) for v in pnl],
        },
    }


def _analyze_iron_condor(S, K_list, T, r, sigma, q=0):
    """铁鹰式策略"""
    if len(K_list) < 4:
        return {"error": "铁鹰式需要4个行权价(K1<K2<K3<K4)"}
    K1, K2, K3, K4 = sorted(K_list[:4])
    put1 = calculate_option_greeks('put', S, K1, T, r, sigma, q)
    put2 = calculate_option_greeks('put', S, K2, T, r, sigma, q)
    call3 = calculate_option_greeks('call', S, K3, T, r, sigma, q)
    call4 = calculate_option_greeks('call', S, K4, T, r, sigma, q)
    if any("error" in g for g in [put1, put2, call3, call4]):
        return {"error": "Greeks计算失败"}

    net_credit = put2["理论价格"] - put1["理论价格"] + call3["理论价格"] - call4["理论价格"]
    max_profit = net_credit
    max_loss = max(K2 - K1, K4 - K3) - net_credit
    breakeven_low = K2 - net_credit
    breakeven_high = K3 + net_credit

    price_range = np.linspace(S * 0.7, S * 1.3, 50)
    pnl = []
    for p in price_range:
        put1_val = max(K1 - p, 0)
        put2_val = max(K2 - p, 0)
        call3_val = max(p - K3, 0)
        call4_val = max(p - K4, 0)
        val = -put1_val + put2_val + call3_val - call4_val + net_credit
        pnl.append(float(val))

    return {
        "策略名称": "铁鹰式(Iron Condor)",
        "构建方式": f"卖K2={K2}看跌+买K1={K1}看跌 + 卖K3={K3}看涨+买K4={K4}看涨",
        "净收入": round(net_credit, 4),
        "最大收益": round(max_profit, 4),
        "最大亏损": round(max_loss, 4),
        "下盈亏平衡点": round(breakeven_low, 4),
        "上盈亏平衡点": round(breakeven_high, 4),
        "收益风险比": f"{max_profit / max_loss:.2f}" if max_loss > 0 else "N/A",
        "适用场景": "预期标的在K2-K3区间内窄幅震荡，赚取时间价值",
        "盈亏曲线": {
            "价格区间": [round(p, 2) for p in price_range.tolist()],
            "盈亏": [round(v, 4) for v in pnl],
        },
    }


# ==================== 隐含波动率分析 ====================

def implied_volatility_surface(symbol="510050", days=60):
    """
    隐含波动率曲面分析
    获取50ETF期权的隐含波动率数据

    参数:
        symbol: 标的代码（510050=50ETF, 510300=300ETF）
        days: 数据天数

    返回: 隐含波动率分析
    """
    try:
        df_opt = ak.option_50etf_spot_sina(symbol=symbol)
    except Exception:
        try:
            df_opt = ak.option_300etf_spot_sina(symbol=symbol)
        except Exception:
            return {"error": f"无法获取{symbol}期权数据，请检查网络或symbol参数"}

    if df_opt is None or len(df_opt) == 0:
        return {"error": "未获取到期权数据"}

    # 获取标的现价
    try:
        df_spot = ak.stock_zh_a_spot_em()
        spot_row = df_spot[df_spot['代码'] == symbol]
        if not spot_row.empty:
            spot_price = float(spot_row.iloc[0]['最新价'])
        else:
            spot_price = 0
    except Exception:
        spot_price = 0

    # 按行权价和到期月份分组
    result = {
        "标的代码": symbol,
        "标的价格": spot_price,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "期权概况": {
            "总合约数": len(df_opt),
            "列名": list(df_opt.columns),
        },
    }

    # 尝试提取关键字段
    key_cols = []
    for col in df_opt.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in ['行权', 'strike', '执行']):
            key_cols.append(('行权价', col))
        elif any(kw in col_lower for kw in ['买', 'bid', '申买']):
            key_cols.append(('买价', col))
        elif any(kw in col_lower for kw in ['卖', 'ask', '申卖']):
            key_cols.append(('卖价', col))
        elif any(kw in col_lower for kw in ['最新', 'last', '现价']):
            key_cols.append(('最新价', col))
        elif any(kw in col_lower for kw in ['涨跌', 'change']):
            key_cols.append(('涨跌幅', col))
        elif any(kw in col_lower for kw in ['量', 'vol', '成交']):
            key_cols.append(('成交量', col))
        elif any(kw in col_lower for kw in ['仓', 'open', '持仓']):
            key_cols.append(('持仓量', col))
        elif any(kw in col_lower for kw in ['隐', 'iv', 'implied']):
            key_cols.append(('隐含波动率', col))
        elif any(kw in col_lower for kw in ['名称', 'name', '简称']):
            key_cols.append(('合约名称', col))
        elif any(kw in col_lower for kw in ['代码', 'code', 'symbol']):
            key_cols.append(('合约代码', col))

    result["识别字段"] = [{"含义": k, "列名": v} for k, v in key_cols]

    # 提取前20条数据预览
    preview = []
    for i in range(min(20, len(df_opt))):
        row = {}
        for meaning, col in key_cols:
            if col in df_opt.columns:
                val = df_opt.iloc[i][col]
                if isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = round(float(val), 4)
                row[meaning] = val
        preview.append(row)

    result["数据预览"] = preview

    # 波动率微笑分析（如果有行权价和价格数据）
    strike_col = next((c for m, c in key_cols if m == '行权价'), None)
    price_col = next((c for m, c in key_cols if m == '最新价'), None)

    if strike_col and price_col and spot_price > 0:
        df_valid = df_opt[[strike_col, price_col]].dropna()
        df_valid = df_valid[df_valid[price_col] > 0]

        if len(df_valid) >= 5:
            strikes = df_valid[strike_col].values
            prices = df_valid[price_col].values

            # 计算ATM附近的隐含波动率
            atm_idx = np.argmin(np.abs(strikes - spot_price))
            atm_strike = float(strikes[atm_idx])

            # 简化IV计算（使用30天到期）
            T_est = 30 / 365
            r_est = 0.02

            iv_data = []
            for i in range(len(df_valid)):
                s_val = float(strikes[i])
                p_val = float(prices[i])
                opt_type = 'call' if s_val >= spot_price * 0.95 else 'put'
                iv = implied_volatility(opt_type, p_val, spot_price, s_val, T_est, r_est)
                if iv:
                    iv_data.append({
                        "行权价": s_val,
                        "价格": p_val,
                        "隐含波动率": f"{iv * 100:.2f}%",
                        "虚实度": f"{(s_val / spot_price - 1) * 100:+.1f}%",
                    })

            result["波动率微笑"] = iv_data[:30]

            # 偏度分析
            if len(iv_data) >= 5:
                otm_put_ivs = [d for d in iv_data if d["行权价"] < spot_price * 0.95]
                otm_call_ivs = [d for d in iv_data if d["行权价"] > spot_price * 1.05]

                if otm_put_ivs and otm_call_ivs:
                    avg_put_iv = float(otm_put_ivs[0]["隐含波动率"].replace('%', ''))
                    avg_call_iv = float(otm_call_ivs[-1]["隐含波动率"].replace('%', ''))
                    skew = avg_put_iv - avg_call_iv

                    result["波动率偏度"] = {
                        "虚值看跌IV": f"{avg_put_iv:.2f}%",
                        "虚值看涨IV": f"{avg_call_iv:.2f}%",
                        "偏度": f"{skew:+.2f}%",
                        "解读": "正偏度表示市场对下跌风险的担忧大于上涨（正常现象）" if skew > 0 else "负偏度表示市场对上涨的预期更强",
                    }

    return result


# ==================== 期权数据获取 ====================

def get_option_chain(symbol="510050", expire_month=None):
    """
    获取期权链数据

    参数:
        symbol: 标的代码
        expire_month: 到期月份（如'2406'），不传则获取所有

    返回: 期权链数据
    """
    try:
        if symbol == "510050":
            df = ak.option_50etf_spot_sina(symbol=symbol)
        elif symbol == "510300":
            df = ak.option_300etf_spot_sina(symbol=symbol)
        elif symbol == "510500":
            df = ak.option_500etf_spot_sina(symbol=symbol)
        elif symbol == "159919":
            df = ak.option_300etf_spot_sina(symbol=symbol)
        else:
            df = ak.option_50etf_spot_sina(symbol=symbol)
    except Exception as e:
        return {"error": f"获取期权数据失败: {str(e)}"}

    if df is None or len(df) == 0:
        return {"error": "未获取到期权数据"}

    result = {
        "标的": symbol,
        "合约总数": len(df),
        "数据列": list(df.columns),
        "数据时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # 提取前50条
    preview = []
    for i in range(min(50, len(df))):
        row = {}
        for col in df.columns:
            val = df.iloc[i][col]
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = round(float(val), 4)
            elif isinstance(val, float):
                val = round(val, 4)
            row[str(col)] = val
        preview.append(row)

    result["数据预览"] = preview
    return result


# ==================== 期权风险指标 ====================

def option_risk_metrics(positions):
    """
    期权持仓风险汇总

    参数:
        positions: 期权持仓 [
            {"合约": "50ETF购6月2500", "类型": "call", "方向": "long",
             "行权价": 2.5, "数量": 10, "现价": 0.05, "Delta": 0.3, "Gamma": 0.1,
             "Vega": 0.02, "Theta": -0.01},
            ...
        ]

    返回: 风险汇总
    """
    if not positions:
        return {"error": "持仓数据不能为空"}

    total_delta = 0
    total_gamma = 0
    total_vega = 0
    total_theta = 0
    total_value = 0

    for pos in positions:
        qty = pos.get("数量", 0)
        price = pos.get("现价", 0)
        direction = 1 if pos.get("方向") == "long" else -1

        total_value += qty * price * direction
        total_delta += qty * pos.get("Delta", 0) * direction
        total_gamma += qty * pos.get("Gamma", 0) * direction
        total_vega += qty * pos.get("Vega", 0) * direction
        total_theta += qty * pos.get("Theta", 0) * direction

    # 风险解读
    risk_interpretation = []
    if abs(total_delta) > 100:
        risk_interpretation.append(f"Delta敞口较大({total_delta:.0f})，方向性风险显著")
    if abs(total_gamma) > 10:
        risk_interpretation.append(f"Gamma敞口较大({total_gamma:.1f})，Delta变化敏感")
    if total_theta < -5:
        risk_interpretation.append(f"Theta为负({total_theta:.1f}/日)，时间价值持续流失")
    elif total_theta > 5:
        risk_interpretation.append(f"Theta为正({total_theta:.1f}/日)，时间是你的朋友")

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "持仓市值": round(total_value, 2),
        "风险汇总": {
            "总Delta": round(total_delta, 2),
            "总Gamma": round(total_gamma, 4),
            "总Vega": round(total_vega, 4),
            "总Theta(日)": round(total_theta, 4),
        },
        "Delta解读": f"标的每变动1元，组合价值变动约{abs(total_delta):.0f}元" if abs(total_delta) > 0 else "Delta中性",
        "风险解读": risk_interpretation if risk_interpretation else ["风险敞口在可控范围内"],
    }


def main():
    parser = argparse.ArgumentParser(description='期权与衍生品分析系统')
    subparsers = parser.add_subparsers(dest='command')

    # Greeks计算
    greeks_parser = subparsers.add_parser('greeks', help='计算期权Greeks')
    greeks_parser.add_argument('--type', default='call', choices=['call', 'put'], help='期权类型')
    greeks_parser.add_argument('--S', type=float, required=True, help='标的价格')
    greeks_parser.add_argument('--K', type=float, required=True, help='行权价')
    greeks_parser.add_argument('--T', type=float, required=True, help='到期时间(年)')
    greeks_parser.add_argument('--r', type=float, default=0.02, help='无风险利率')
    greeks_parser.add_argument('--sigma', type=float, default=0.3, help='波动率')
    greeks_parser.add_argument('--q', type=float, default=0, help='股息率')

    # 隐含波动率
    iv_parser = subparsers.add_parser('iv', help='计算隐含波动率')
    iv_parser.add_argument('--type', default='call', choices=['call', 'put'], help='期权类型')
    iv_parser.add_argument('--price', type=float, required=True, help='期权市场价格')
    iv_parser.add_argument('--S', type=float, required=True, help='标的价格')
    iv_parser.add_argument('--K', type=float, required=True, help='行权价')
    iv_parser.add_argument('--T', type=float, required=True, help='到期时间(年)')
    iv_parser.add_argument('--r', type=float, default=0.02, help='无风险利率')

    # 策略分析
    strategy_parser = subparsers.add_parser('strategy', help='期权策略分析')
    strategy_parser.add_argument('--type', required=True,
                                 choices=['covered_call', 'protective_put', 'bull_spread',
                                          'bear_spread', 'straddle', 'strangle',
                                          'butterfly', 'iron_condor'],
                                 help='策略类型')
    strategy_parser.add_argument('--S', type=float, required=True, help='标的价格')
    strategy_parser.add_argument('--K', type=str, required=True, help='行权价列表,逗号分隔')
    strategy_parser.add_argument('--T', type=float, required=True, help='到期时间(年)')
    strategy_parser.add_argument('--r', type=float, default=0.02, help='无风险利率')
    strategy_parser.add_argument('--sigma', type=float, default=0.3, help='波动率')

    # 隐含波动率曲面
    surface_parser = subparsers.add_parser('surface', help='隐含波动率曲面分析')
    surface_parser.add_argument('--symbol', default='510050', help='标的代码')

    # 期权链
    chain_parser = subparsers.add_parser('chain', help='获取期权链数据')
    chain_parser.add_argument('--symbol', default='510050', help='标的代码')

    # 风险汇总
    risk_parser = subparsers.add_parser('risk', help='期权持仓风险汇总')
    risk_parser.add_argument('--positions', required=True, help='持仓JSON')

    args = parser.parse_args()

    if args.command == 'greeks':
        result = calculate_option_greeks(args.type, args.S, args.K, args.T, args.r, args.sigma, args.q)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'iv':
        result = implied_volatility(args.type, args.price, args.S, args.K, args.T, args.r)
        print(json.dumps({"隐含波动率": f"{result * 100:.2f}%" if result else "计算失败"}, ensure_ascii=False, indent=2))

    elif args.command == 'strategy':
        K_list = [float(k.strip()) for k in args.K.split(',')]
        result = option_strategy_analysis(args.type, args.S, K_list, args.T, args.r, args.sigma)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'surface':
        result = implied_volatility_surface(args.symbol)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'chain':
        result = get_option_chain(args.symbol)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'risk':
        positions = json.loads(args.positions)
        result = option_risk_metrics(positions)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
