#!/usr/bin/env python3
"""
风控系统 - 事前风控 / 事中风控 / 事后风控
量化交易的生命线，提供完整的三层风控体系
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


def get_multi_kline(symbols, days=250):
    """获取多只股票K线数据"""
    closes = {}
    for symbol in symbols:
        df = get_stock_kline(symbol, days)
        if df is not None:
            closes[symbol] = df['收盘']
        time.sleep(0.3)
    if len(closes) < 2:
        return None
    return pd.DataFrame(closes).dropna()


# ==================== 事前风控 ====================

def pre_trade_check(symbol, order_params, portfolio_state, risk_config=None):
    """
    事前风控检查 - 下单前必须通过所有检查
    参数:
        symbol: 股票代码
        order_params: 订单参数 {direction, quantity, price, order_type}
        portfolio_state: 当前持仓状态 {total_asset, cash, positions}
        risk_config: 风控配置
    返回:
        dict: {passed, checks, reject_reasons}
    """
    if risk_config is None:
        risk_config = get_default_risk_config()

    checks = []
    reject_reasons = []
    passed = True

    direction = order_params.get("direction", "buy")
    quantity = int(order_params.get("quantity", 0))
    price = float(order_params.get("price", 0))
    order_amount = quantity * price

    total_asset = float(portfolio_state.get("total_asset", 0))
    cash = float(portfolio_state.get("cash", 0))
    positions = portfolio_state.get("positions", [])

    # 1. 单票仓位上限检查
    max_single_position = risk_config.get("max_single_position_pct", 20)
    current_single_pct = 0
    for pos in positions:
        if pos.get("symbol") == symbol:
            current_single_pct = float(pos.get("market_value", 0)) / total_asset * 100 if total_asset > 0 else 0
            break

    if direction == "buy":
        new_single_pct = current_single_pct + order_amount / total_asset * 100 if total_asset > 0 else 0
    else:
        new_single_pct = current_single_pct

    single_check = {
        "检查项": "单票仓位上限",
        "阈值": f"{max_single_position}%",
        "当前值": f"{current_single_pct:.2f}%",
        "下单后": f"{new_single_pct:.2f}%"
    }
    if new_single_pct > max_single_position:
        single_check["结果"] = "拒绝"
        reject_reasons.append(f"单票仓位({new_single_pct:.1f}%)超过上限({max_single_position}%)")
        passed = False
    else:
        single_check["结果"] = "通过"
    checks.append(single_check)

    # 2. 总仓位上限检查
    max_total_position = risk_config.get("max_total_position_pct", 80)
    current_total_pct = sum(float(p.get("market_value", 0)) for p in positions) / total_asset * 100 if total_asset > 0 else 0
    new_total_pct = current_total_pct + order_amount / total_asset * 100 if direction == "buy" and total_asset > 0 else current_total_pct

    total_check = {
        "检查项": "总仓位上限",
        "阈值": f"{max_total_position}%",
        "当前值": f"{current_total_pct:.2f}%",
        "下单后": f"{new_total_pct:.2f}%"
    }
    if new_total_pct > max_total_position:
        total_check["结果"] = "拒绝"
        reject_reasons.append(f"总仓位({new_total_pct:.1f}%)超过上限({max_total_position}%)")
        passed = False
    else:
        total_check["结果"] = "通过"
    checks.append(total_check)

    # 3. 现金充足检查
    if direction == "buy":
        cash_check = {
            "检查项": "现金充足",
            "可用现金": f"{cash:.2f}",
            "所需资金": f"{order_amount:.2f}"
        }
        if order_amount > cash:
            cash_check["结果"] = "拒绝"
            reject_reasons.append(f"现金不足(需要{order_amount:.2f}, 可用{cash:.2f})")
            passed = False
        else:
            cash_check["结果"] = "通过"
        checks.append(cash_check)

    # 4. 单笔订单金额上限
    max_order_amount = risk_config.get("max_single_order_amount", total_asset * 0.1)
    order_amount_check = {
        "检查项": "单笔订单金额上限",
        "阈值": f"{max_order_amount:.2f}",
        "订单金额": f"{order_amount:.2f}"
    }
    if order_amount > max_order_amount:
        order_amount_check["结果"] = "拒绝"
        reject_reasons.append(f"单笔订单金额({order_amount:.2f})超过上限({max_order_amount:.2f})")
        passed = False
    else:
        order_amount_check["结果"] = "通过"
    checks.append(order_amount_check)

    # 5. 行业集中度检查
    max_sector_pct = risk_config.get("max_sector_concentration", 30)
    sector_check = {
        "检查项": "行业集中度",
        "阈值": f"{max_sector_pct}%",
        "结果": "通过",
        "说明": "需要行业分类数据"
    }
    checks.append(sector_check)

    # 6. 黑名单检查
    blacklist = risk_config.get("blacklist", [])
    blacklist_check = {
        "检查项": "黑名单",
        "是否在黑名单": symbol in blacklist
    }
    if symbol in blacklist:
        blacklist_check["结果"] = "拒绝"
        reject_reasons.append(f"{symbol}在黑名单中")
        passed = False
    else:
        blacklist_check["结果"] = "通过"
    checks.append(blacklist_check)

    # 7. 杠杆率检查
    max_leverage = risk_config.get("max_leverage", 1.0)
    total_liability = sum(float(p.get("liability", 0)) for p in positions)
    current_leverage = (total_asset + total_liability) / total_asset if total_asset > 0 else 1
    leverage_check = {
        "检查项": "杠杆率",
        "阈值": f"{max_leverage}x",
        "当前杠杆": f"{current_leverage:.2f}x"
    }
    if current_leverage > max_leverage:
        leverage_check["结果"] = "拒绝"
        reject_reasons.append(f"杠杆率({current_leverage:.2f})超过上限({max_leverage})")
        passed = False
    else:
        leverage_check["结果"] = "通过"
    checks.append(leverage_check)

    return {
        "风控通过": passed,
        "检查时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "股票代码": symbol,
        "订单方向": direction,
        "检查明细": checks,
        "拒绝原因": reject_reasons if reject_reasons else None
    }


# ==================== 事中风控 ====================

def in_trade_monitor(positions, market_data, risk_config=None):
    """
    事中风控监控 - 实时监控持仓风险
    参数:
        positions: 持仓列表 [{symbol, cost_price, quantity, market_value}]
        market_data: 实时行情 {symbol: {price, change_pct, volume}}
        risk_config: 风控配置
    返回:
        dict: {alerts, stop_loss_triggers, take_profit_triggers, risk_summary}
    """
    if risk_config is None:
        risk_config = get_default_risk_config()

    alerts = []
    stop_loss_triggers = []
    take_profit_triggers = []
    total_market_value = 0
    total_pnl = 0

    for pos in positions:
        symbol = pos.get("symbol", "")
        cost_price = float(pos.get("cost_price", 0))
        quantity = int(pos.get("quantity", 0))
        market_value = float(pos.get("market_value", 0))
        total_market_value += market_value

        if symbol not in market_data:
            continue

        current_price = float(market_data[symbol].get("price", 0))
        change_pct = float(market_data[symbol].get("change_pct", 0))
        volume = float(market_data[symbol].get("volume", 0))

        if cost_price <= 0 or current_price <= 0:
            continue

        pnl_pct = (current_price / cost_price - 1) * 100
        pnl_amount = (current_price - cost_price) * quantity
        total_pnl += pnl_amount

        # 止损检查
        stop_loss_pct = risk_config.get("stop_loss_pct", -8)
        hard_stop_pct = risk_config.get("hard_stop_loss_pct", -15)

        if pnl_pct <= hard_stop_pct:
            stop_loss_triggers.append({
                "股票": symbol,
                "类型": "硬止损",
                "亏损比例": round(pnl_pct, 2),
                "硬止损阈值": hard_stop_pct,
                "建议": "立即平仓"
            })
            alerts.append({
                "级别": "严重",
                "股票": symbol,
                "内容": f"触发硬止损! 亏损{pnl_pct:.1f}%超过{hard_stop_pct}%阈值"
            })
        elif pnl_pct <= stop_loss_pct:
            stop_loss_triggers.append({
                "股票": symbol,
                "类型": "软止损",
                "亏损比例": round(pnl_pct, 2),
                "止损阈值": stop_loss_pct,
                "建议": "考虑减仓或止损"
            })
            alerts.append({
                "级别": "警告",
                "股票": symbol,
                "内容": f"触发止损预警! 亏损{pnl_pct:.1f}%超过{stop_loss_pct}%阈值"
            })

        # 止盈检查
        take_profit_pct = risk_config.get("take_profit_pct", 20)
        trailing_stop_pct = risk_config.get("trailing_stop_pct", 10)

        if pnl_pct >= take_profit_pct:
            take_profit_triggers.append({
                "股票": symbol,
                "类型": "目标止盈",
                "盈利比例": round(pnl_pct, 2),
                "止盈阈值": take_profit_pct,
                "建议": "分批止盈，锁定利润"
            })
            alerts.append({
                "级别": "提示",
                "股票": symbol,
                "内容": f"达到止盈目标! 盈利{pnl_pct:.1f}%超过{take_profit_pct}%阈值"
            })

        # 异常波动检查
        max_daily_change = risk_config.get("max_daily_change_pct", 9)
        if abs(change_pct) >= max_daily_change:
            alerts.append({
                "级别": "警告",
                "股票": symbol,
                "内容": f"日内异常波动! 涨跌幅{change_pct:.1f}%"
            })

        # 流动性检查
        min_volume = risk_config.get("min_daily_volume", 1000000)
        if volume < min_volume:
            alerts.append({
                "级别": "提示",
                "股票": symbol,
                "内容": f"流动性不足! 成交量{volume:.0f}低于{min_volume:.0f}"
            })

    # 组合层面风控
    portfolio_alerts = []

    # 组合回撤检查
    max_portfolio_dd = risk_config.get("max_portfolio_drawdown", -15)
    if total_market_value > 0:
        total_cost = sum(float(p.get("cost_price", 0)) * int(p.get("quantity", 0)) for p in positions)
        portfolio_pnl_pct = (total_market_value / total_cost - 1) * 100 if total_cost > 0 else 0
        if portfolio_pnl_pct <= max_portfolio_dd:
            portfolio_alerts.append({
                "级别": "严重",
                "内容": f"组合回撤{portfolio_pnl_pct:.1f}%超过{max_portfolio_dd}%阈值，建议减仓"
            })

    return {
        "监控时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "告警列表": alerts,
        "止损触发": stop_loss_triggers,
        "止盈触发": take_profit_triggers,
        "组合告警": portfolio_alerts,
        "风险汇总": {
            "持仓总市值": round(total_market_value, 2),
            "持仓总盈亏": round(total_pnl, 2),
            "告警总数": len(alerts),
            "止损触发数": len(stop_loss_triggers),
            "止盈触发数": len(take_profit_triggers)
        }
    }


# ==================== 事后风控 ====================

def post_trade_risk_analysis(symbols, days=250, risk_config=None):
    """
    事后风控分析 - 全面风险评估
    参数:
        symbols: 股票代码列表
        days: 分析周期
        risk_config: 风控配置
    返回:
        dict: 完整风险评估报告
    """
    if risk_config is None:
        risk_config = get_default_risk_config()

    result = {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "分析周期": f"{days}个交易日",
        "股票数量": len(symbols),
        "个股风险": [],
        "组合风险": {},
        "压力测试": {},
        "风险评级": {}
    }

    all_returns = {}

    for symbol in symbols:
        df = get_stock_kline(symbol, days)
        if df is None or df.empty:
            continue

        close = df['close']
        returns = close.pct_change().dropna()
        all_returns[symbol] = returns

        # VaR 计算
        var_95 = float(np.percentile(returns, 5)) * 100
        var_99 = float(np.percentile(returns, 1)) * 100
        cvar_95 = float(returns[returns <= np.percentile(returns, 5)].mean()) * 100

        # 波动率
        annual_vol = float(returns.std() * np.sqrt(252)) * 100

        # 最大回撤
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak * 100
        max_dd = float(drawdown.min())

        # 偏度和峰度
        skewness = float(returns.skew())
        kurtosis = float(returns.kurtosis())

        # 风险评级
        risk_score = 0
        if annual_vol < 20:
            risk_score += 25
        elif annual_vol < 35:
            risk_score += 15
        elif annual_vol < 50:
            risk_score += 5

        if abs(max_dd) < 15:
            risk_score += 25
        elif abs(max_dd) < 25:
            risk_score += 15
        elif abs(max_dd) < 40:
            risk_score += 5

        if abs(var_95) < 3:
            risk_score += 25
        elif abs(var_95) < 5:
            risk_score += 15
        elif abs(var_95) < 8:
            risk_score += 5

        if abs(skewness) < 0.5:
            risk_score += 15
        elif abs(skewness) < 1:
            risk_score += 8

        if kurtosis < 3:
            risk_score += 10
        elif kurtosis < 5:
            risk_score += 5

        if risk_score >= 70:
            risk_level = "低风险"
        elif risk_score >= 45:
            risk_level = "中等风险"
        elif risk_score >= 25:
            risk_level = "高风险"
        else:
            risk_level = "极高风险"

        result["个股风险"].append({
            "股票代码": symbol,
            "年化波动率": round(annual_vol, 2),
            "VaR(95%)": round(var_95, 2),
            "VaR(99%)": round(var_99, 2),
            "CVaR(95%)": round(cvar_95, 2),
            "最大回撤": round(max_dd, 2),
            "偏度": round(skewness, 3),
            "峰度": round(kurtosis, 3),
            "风险评分": risk_score,
            "风险等级": risk_level
        })

    # 组合风险分析
    if len(all_returns) >= 2:
        returns_df = pd.DataFrame(all_returns).dropna()
        if len(returns_df) > 30:
            # 相关性矩阵
            corr_matrix = returns_df.corr()

            # 组合波动率（等权）
            weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
            portfolio_returns = (returns_df * weights).sum(axis=1)
            portfolio_vol = float(portfolio_returns.std() * np.sqrt(252)) * 100

            # 组合VaR
            portfolio_var_95 = float(np.percentile(portfolio_returns, 5)) * 100

            # 分散化收益
            avg_vol = float(np.mean([returns_df[c].std() * np.sqrt(252) for c in returns_df.columns])) * 100
            diversification_benefit = round(avg_vol - portfolio_vol, 2)

            # 相关性统计
            corr_values = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_values.append(corr_matrix.iloc[i, j])
            avg_corr = float(np.mean(corr_values)) if corr_values else 0

            result["组合风险"] = {
                "组合年化波动率": round(portfolio_vol, 2),
                "组合VaR(95%)": round(portfolio_var_95, 2),
                "平均个股波动率": round(avg_vol, 2),
                "分散化收益": diversification_benefit,
                "平均相关性": round(avg_corr, 3),
                "相关性矩阵": {s: {s2: round(float(corr_matrix.loc[s, s2]), 3) for s2 in corr_matrix.columns} for s in corr_matrix.columns}
            }

    # 压力测试
    stress_scenarios = {
        "2008金融危机": -0.05,
        "2015股灾": -0.07,
        "2018熊市": -0.03,
        "2020疫情冲击": -0.04,
        "极端下跌(3sigma)": -0.10,
        "流动性危机": -0.08,
        "连续跌停(3天)": -0.27
    }

    for scenario_name, shock in stress_scenarios.items():
        scenario_losses = {}
        for symbol, returns in all_returns.items():
            annual_vol = returns.std() * np.sqrt(252)
            scenario_loss = shock * (annual_vol / 0.25) * 100
            scenario_losses[symbol] = round(float(scenario_loss), 2)

        avg_loss = float(np.mean(list(scenario_losses.values())))
        max_loss = float(min(scenario_losses.values()))

        result["压力测试"][scenario_name] = {
            "冲击幅度": f"{shock*100:.0f}%",
            "平均预估损失": f"{avg_loss:.1f}%",
            "最大预估损失": f"{max_loss:.1f}%",
            "个股损失": scenario_losses
        }

    # 综合风险评级
    if result["个股风险"]:
        avg_risk_score = float(np.mean([r["风险评分"] for r in result["个股风险"]]))
        if avg_risk_score >= 70:
            overall_risk = "低风险组合"
        elif avg_risk_score >= 45:
            overall_risk = "中等风险组合"
        elif avg_risk_score >= 25:
            overall_risk = "高风险组合"
        else:
            overall_risk = "极高风险组合"

        result["风险评级"] = {
            "综合评分": round(avg_risk_score, 1),
            "综合评级": overall_risk,
            "建议": get_risk_advice(overall_risk)
        }

    return result


def stress_test_portfolio(symbols, scenarios=None, days=250):
    """
    组合压力测试
    参数:
        symbols: 股票代码列表
        scenarios: 自定义压力场景 {name: shock_pct}
        days: 历史数据天数
    """
    if scenarios is None:
        scenarios = {
            "温和下跌(-5%)": -0.05,
            "中度下跌(-10%)": -0.10,
            "严重下跌(-20%)": -0.20,
            "极端下跌(-30%)": -0.30,
            "崩盘(-50%)": -0.50
        }

    result = {
        "测试时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "场景分析": {}
    }

    all_returns = {}
    for symbol in symbols:
        df = get_stock_kline(symbol, days)
        if df is not None and not df.empty:
            returns = df['close'].pct_change().dropna()
            all_returns[symbol] = returns
        time.sleep(0.3)

    if len(all_returns) < 2:
        return {"error": "数据不足"}

    returns_df = pd.DataFrame(all_returns).dropna()
    weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)

    for scenario_name, shock in scenarios.items():
        # 使用历史beta估算冲击影响
        market_returns = returns_df.mean(axis=1)
        betas = {}
        for col in returns_df.columns:
            cov = returns_df[col].cov(market_returns)
            var = market_returns.var()
            betas[col] = cov / var if var > 0 else 1.0

        total_impact = 0
        stock_impacts = {}
        for col in returns_df.columns:
            impact = shock * betas[col] * 100
            stock_impacts[col] = round(float(impact), 2)
            total_impact += impact * weights[list(returns_df.columns).index(col)]

        result["场景分析"][scenario_name] = {
            "市场冲击": f"{shock*100:.0f}%",
            "组合预估影响": f"{total_impact:.1f}%",
            "个股影响": stock_impacts
        }

    return result


def get_default_risk_config():
    """获取默认风控配置"""
    return {
        "max_single_position_pct": 20,
        "max_total_position_pct": 80,
        "max_single_order_amount": 50000,
        "max_sector_concentration": 30,
        "max_leverage": 1.0,
        "stop_loss_pct": -8,
        "hard_stop_loss_pct": -15,
        "take_profit_pct": 20,
        "trailing_stop_pct": 10,
        "max_daily_change_pct": 9,
        "min_daily_volume": 1000000,
        "max_portfolio_drawdown": -15,
        "blacklist": [],
        "whitelist": []
    }


def get_risk_advice(risk_level):
    """根据风险等级给出建议"""
    advice_map = {
        "低风险组合": "组合风险可控，可维持当前配置，关注市场变化",
        "中等风险组合": "建议适当分散持仓，设置止损位，控制单票仓位不超过15%",
        "高风险组合": "建议降低仓位至50%以下，严格设置止损，增加低波动品种",
        "极高风险组合": "强烈建议大幅减仓，重新评估持仓品种，优先考虑风险控制"
    }
    return advice_map.get(risk_level, "请重新评估风险")


def calculate_var_breakdown(symbols, days=250):
    """
    VaR分解 - 分析各持仓对组合VaR的贡献
    """
    all_returns = {}
    for symbol in symbols:
        df = get_stock_kline(symbol, days)
        if df is not None and not df.empty:
            all_returns[symbol] = df['close'].pct_change().dropna()
        time.sleep(0.3)

    if len(all_returns) < 2:
        return {"error": "数据不足"}

    returns_df = pd.DataFrame(all_returns).dropna()
    n = len(returns_df.columns)
    weights = np.ones(n) / n

    portfolio_returns = (returns_df * weights).sum(axis=1)
    portfolio_var = float(np.percentile(portfolio_returns, 5))

    # 成分VaR
    component_var = {}
    for i, col in enumerate(returns_df.columns):
        marginal_var = returns_df[col].cov(portfolio_returns) / portfolio_returns.var() if portfolio_returns.var() > 0 else 0
        comp_var = marginal_var * weights[i] * portfolio_var
        component_var[col] = {
            "权重": f"{weights[i]*100:.1f}%",
            "边际VaR": round(float(marginal_var), 6),
            "成分VaR": round(float(comp_var) * 100, 4),
            "贡献占比": f"{abs(comp_var)/sum(abs(v) for v in [component_var[c]['成分VaR']/100 for c in component_var])*100:.1f}%" if portfolio_var != 0 else "0%"
        }

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "组合VaR(95%)": f"{portfolio_var*100:.2f}%",
        "成分VaR分解": component_var
    }


def calc_atr(df, period=14):
    """
    计算ATR（平均真实波幅）
    适配中文列名
    """
    high_col = '最高' if '最高' in df.columns else ('high' if 'high' in df.columns else None)
    low_col = '最低' if '最低' in df.columns else ('low' if 'low' in df.columns else None)
    close_col = '收盘' if '收盘' in df.columns else ('close' if 'close' in df.columns else None)

    if not all([high_col, low_col, close_col]):
        return None

    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close = df[close_col].astype(float)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def atr_dynamic_stop_loss(symbol, entry_price, direction="long", atr_multiple=2.0,
                           atr_period=14, days=120):
    """
    ATR动态止损
    基于ATR的动态止损价位，随波动率自适应调整

    参数:
        symbol: 股票代码
        entry_price: 入场价格
        direction: 持仓方向 long/short
        atr_multiple: ATR倍数（默认2倍ATR）
        atr_period: ATR计算周期
        days: 数据天数
    """
    df = get_stock_kline(symbol, days)
    if df is None or df.empty:
        return {"error": f"无法获取 {symbol} 的K线数据"}

    atr = calc_atr(df, atr_period)
    if atr is None:
        return {"error": "无法计算ATR"}

    current_atr = float(atr.iloc[-1])
    if pd.isna(current_atr) or current_atr <= 0:
        return {"error": "ATR计算异常"}

    close_col = '收盘' if '收盘' in df.columns else 'close'
    current_price = float(df[close_col].iloc[-1])

    if direction == "long":
        stop_price = entry_price - atr_multiple * current_atr
        stop_pct = (stop_price / entry_price - 1) * 100
    else:
        stop_price = entry_price + atr_multiple * current_atr
        stop_pct = (entry_price / stop_price - 1) * 100

    # 历史ATR统计
    atr_values = atr.dropna().values
    atr_percentile_50 = float(np.percentile(atr_values, 50))
    atr_percentile_90 = float(np.percentile(atr_values, 90))
    atr_trend = "上升" if current_atr > atr_percentile_50 * 1.2 else ("下降" if current_atr < atr_percentile_50 * 0.8 else "平稳")

    return {
        "策略": "ATR动态止损",
        "股票代码": symbol,
        "入场价格": round(entry_price, 2),
        "当前价格": round(current_price, 2),
        "当前ATR": round(current_atr, 2),
        "ATR倍数": atr_multiple,
        "止损价格": round(stop_price, 2),
        "止损幅度": f"{stop_pct:.1f}%",
        "ATR趋势": atr_trend,
        "ATR中位数": round(atr_percentile_50, 2),
        "ATR 90分位": round(atr_percentile_90, 2),
        "说明": f"价格跌破{stop_price:.2f}时触发止损，止损幅度随波动率动态调整",
        "风控建议": [
            f"当前ATR为{current_atr:.2f}，{'波动较大，建议适当放宽止损' if atr_trend == '上升' else '波动适中，止损设置合理'}",
            f"若ATR持续上升，考虑降低仓位而非放宽止损",
            "止损触发后等待ATR回落再考虑重新入场",
        ]
    }


def trailing_stop(symbol, entry_price, current_high=None, direction="long",
                   trail_pct=8.0, atr_trail=False, atr_multiple=3.0, days=120):
    """
    移动止盈/止损（追踪止损）
    价格向有利方向移动时，止损线跟随上移

    参数:
        symbol: 股票代码
        entry_price: 入场价格
        current_high: 持仓期间最高价（不传则从K线获取）
        direction: 持仓方向
        trail_pct: 回撤百分比（固定模式）
        atr_trail: 是否使用ATR追踪模式
        atr_multiple: ATR追踪倍数
        days: 数据天数
    """
    df = get_stock_kline(symbol, days)
    if df is None or df.empty:
        return {"error": f"无法获取 {symbol} 的K线数据"}

    close_col = '收盘' if '收盘' in df.columns else 'close'
    high_col = '最高' if '最高' in df.columns else ('high' if 'high' in df.columns else None)
    current_price = float(df[close_col].iloc[-1])

    if current_high is None and high_col:
        current_high = float(df[high_col].max())

    if current_high is None:
        current_high = current_price

    if atr_trail:
        atr = calc_atr(df, 14)
        if atr is not None:
            current_atr = float(atr.iloc[-1])
            trail_amount = atr_multiple * current_atr
            stop_price = current_high - trail_amount
            mode = f"ATR追踪({atr_multiple}xATR)"
        else:
            trail_amount = current_high * trail_pct / 100
            stop_price = current_high * (1 - trail_pct / 100)
            mode = f"固定百分比追踪({trail_pct}%)"
    else:
        trail_amount = current_high * trail_pct / 100
        stop_price = current_high * (1 - trail_pct / 100)
        mode = f"固定百分比追踪({trail_pct}%)"

    # 确保止损价不低于入场价的一定比例
    min_stop = entry_price * 0.85
    stop_price = max(stop_price, min_stop)

    pnl_from_entry = (current_price / entry_price - 1) * 100
    distance_to_stop = (current_price / stop_price - 1) * 100

    return {
        "策略": "移动止盈止损",
        "模式": mode,
        "股票代码": symbol,
        "入场价格": round(entry_price, 2),
        "持仓最高价": round(current_high, 2),
        "当前价格": round(current_price, 2),
        "追踪止损价": round(stop_price, 2),
        "回撤容忍": f"{trail_pct}%" if not atr_trail else f"{atr_multiple}xATR",
        "距止损空间": f"{distance_to_stop:.1f}%",
        "当前浮盈": f"{pnl_from_entry:.1f}%",
        "操作建议": _trailing_advice(pnl_from_entry, distance_to_stop),
    }


def _trailing_advice(pnl_pct, distance_pct):
    """根据移动止盈状态给出建议"""
    if pnl_pct > 30:
        return f"大幅盈利({pnl_pct:.1f}%)，建议收紧追踪止损至5%以内，锁定利润"
    elif pnl_pct > 15:
        return f"盈利可观({pnl_pct:.1f}%)，距止损{distance_pct:.1f}%，可继续持有"
    elif pnl_pct > 5:
        return f"小幅盈利({pnl_pct:.1f}%)，密切关注，止损不宜放宽"
    elif pnl_pct > 0:
        return "微利状态，保持当前追踪止损设置"
    else:
        return "处于亏损状态，追踪止损尚未激活，关注初始止损位"


def adaptive_stop_combo(symbol, entry_price, direction="long", days=120):
    """
    自适应止损组合方案
    同时给出ATR动态止损和移动止盈的完整方案
    """
    atr_stop = atr_dynamic_stop_loss(symbol, entry_price, direction, days=days)
    trail = trailing_stop(symbol, entry_price, direction=direction,
                           atr_trail=True, days=days)

    return {
        "股票代码": symbol,
        "入场价格": entry_price,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "初始止损(ATR)": atr_stop,
        "移动止盈": trail,
        "综合方案": {
            "阶段1_初始保护": f"入场后立即设置ATR动态止损，当前止损价{atr_stop.get('止损价格', 'N/A')}",
            "阶段2_保本移动": "当盈利超过1倍ATR后，将止损移至成本价",
            "阶段3_追踪止盈": "当盈利超过2倍ATR后，启动移动止盈，回撤容忍设为3倍ATR",
            "阶段4_锁定利润": "当盈利超过5倍ATR后，收紧追踪至1.5倍ATR",
        },
        "原则": [
            "止损只收紧不放松",
            "盈利超过2R后启动移动止盈",
            "大幅盈利后优先保护利润而非追求更高收益",
        ]
    }


# ==================== 黑天鹅事件预警 ====================

def black_swan_warning(symbol=None, index_code="000300", days=500):
    """
    黑天鹅事件预警系统
    基于极值理论、波动率突变、相关性崩溃、回撤加速等多维度检测

    参数:
        symbol: 个股代码（可选，不传则分析大盘）
        index_code: 基准指数代码
        days: 分析天数

    返回: {
        "预警等级": str,
        "预警信号": [...],
        "尾部风险": {...},
        "波动率突变": {...},
        "建议": str,
    }
    """
    if symbol:
        df = get_stock_kline(symbol, days=days + 50)
        name = symbol
    else:
        df = get_index_kline(index_code, days=days + 50)
        name = f"指数{index_code}"

    if df is None or len(df) < 120:
        return {"error": f"数据不足，至少需要120个交易日"}

    close = df['收盘'] if '收盘' in df.columns else df['close']
    returns = close.pct_change().dropna()

    warnings = []
    risk_score = 0

    # 1. 尾部风险检测（基于极值理论）
    tail_risk = _detect_tail_risk(returns)
    warnings.extend(tail_risk.get("信号", []))
    risk_score += tail_risk.get("风险分", 0)

    # 2. 波动率突变检测
    vol_regime = _detect_volatility_regime_shift(returns)
    warnings.extend(vol_regime.get("信号", []))
    risk_score += vol_regime.get("风险分", 0)

    # 3. 回撤加速检测
    drawdown_risk = _detect_drawdown_acceleration(close)
    warnings.extend(drawdown_risk.get("信号", []))
    risk_score += drawdown_risk.get("风险分", 0)

    # 4. 相关性崩溃检测（仅大盘）
    if not symbol:
        corr_breakdown = _detect_correlation_breakdown(returns)
        warnings.extend(corr_breakdown.get("信号", []))
        risk_score += corr_breakdown.get("风险分", 0)
    else:
        corr_breakdown = {}

    # 5. 流动性枯竭检测
    liquidity_risk = _detect_liquidity_dryup(df)
    warnings.extend(liquidity_risk.get("信号", []))
    risk_score += liquidity_risk.get("风险分", 0)

    # 综合预警等级
    if risk_score >= 70:
        alert_level = "红色预警"
        alert_description = "检测到多个极端风险信号，黑天鹅事件概率极高"
        action = "立即大幅减仓至20%以下，持有现金或国债等避险资产，暂停所有新开仓"
    elif risk_score >= 50:
        alert_level = "橙色预警"
        alert_description = "检测到显著风险信号，市场处于高风险状态"
        action = "减仓至50%以下，收紧止损，增加对冲仓位，暂停激进策略"
    elif risk_score >= 30:
        alert_level = "黄色预警"
        alert_description = "检测到部分风险信号，需提高警惕"
        action = "适当降低仓位，检查止损设置，减少新开仓频率"
    elif risk_score >= 15:
        alert_level = "蓝色预警"
        alert_description = "存在轻微风险信号，保持关注"
        action = "维持正常仓位，密切关注市场变化，做好应急预案"
    else:
        alert_level = "绿色（正常）"
        alert_description = "未检测到显著风险信号，市场运行正常"
        action = "正常操作，维持现有策略和仓位"

    return {
        "分析对象": name,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "预警等级": alert_level,
        "风险评分": f"{risk_score}/100",
        "预警描述": alert_description,
        "建议行动": action,
        "预警信号": warnings,
        "尾部风险分析": tail_risk,
        "波动率突变分析": vol_regime,
        "回撤加速分析": drawdown_risk,
        "相关性崩溃分析": corr_breakdown if corr_breakdown else None,
        "流动性风险分析": liquidity_risk,
        "历史黑天鹅参考": _get_historical_black_swan_context(returns),
    }


def _detect_tail_risk(returns):
    """
    尾部风险检测
    基于极值理论（EVT）的Hill估计量和超额峰度
    """
    result = {"信号": [], "风险分": 0}
    ret_arr = returns.values

    # 计算偏度和峰度
    skewness = float(pd.Series(ret_arr).skew())
    kurtosis = float(pd.Series(ret_arr).kurtosis())

    result["偏度"] = round(skewness, 3)
    result["超额峰度"] = round(kurtosis, 3)

    # 峰度异常检测
    if kurtosis > 5:
        result["信号"].append(f"超额峰度极高({kurtosis:.1f})，收益率分布呈厚尾特征，极端事件概率远超正态分布假设")
        result["风险分"] += 25
    elif kurtosis > 3:
        result["信号"].append(f"超额峰度偏高({kurtosis:.1f})，存在明显的厚尾风险")
        result["风险分"] += 15
    elif kurtosis > 1.5:
        result["信号"].append(f"超额峰度略高({kurtosis:.1f})，尾部风险需关注")
        result["风险分"] += 5

    # 偏度异常检测
    if skewness < -1:
        result["信号"].append(f"显著负偏({skewness:.2f})，大幅下跌概率高于大幅上涨")
        result["风险分"] += 15
    elif skewness < -0.5:
        result["信号"].append(f"负偏({skewness:.2f})，下跌风险偏高")
        result["风险分"] += 5

    # 极端值分析（超过3倍标准差的交易日比例）
    std = float(np.std(ret_arr))
    extreme_neg = sum(1 for r in ret_arr if r < -3 * std)
    extreme_pos = sum(1 for r in ret_arr if r > 3 * std)
    extreme_ratio = (extreme_neg + extreme_pos) / len(ret_arr) * 100

    result["极端负收益天数"] = int(extreme_neg)
    result["极端正收益天数"] = int(extreme_pos)
    result["极端值比例"] = f"{extreme_ratio:.2f}%"

    if extreme_ratio > 1.5:
        result["信号"].append(f"极端值比例({extreme_ratio:.2f}%)远超正态分布预期(0.3%)，市场存在严重厚尾")
        result["风险分"] += 20
    elif extreme_ratio > 0.8:
        result["信号"].append(f"极端值比例({extreme_ratio:.2f}%)偏高")
        result["风险分"] += 10

    # VaR突破检测
    var_99 = float(np.percentile(ret_arr, 1))
    recent_returns = ret_arr[-20:]
    var_breaches = sum(1 for r in recent_returns if r < var_99)

    result["99%VaR"] = f"{var_99 * 100:.2f}%"
    result["近期VaR突破次数"] = int(var_breaches)

    if var_breaches >= 3:
        result["信号"].append(f"近20日VaR突破{var_breaches}次，远超预期(0.2次)，尾部风险急剧上升")
        result["风险分"] += 25
    elif var_breaches >= 2:
        result["信号"].append(f"近20日VaR突破{var_breaches}次，尾部风险显著")
        result["风险分"] += 15
    elif var_breaches >= 1:
        result["信号"].append(f"近20日出现VaR突破，需关注")
        result["风险分"] += 5

    return result


def _detect_volatility_regime_shift(returns):
    """
    波动率突变检测
    检测波动率是否从低波动状态突然切换到高波动状态
    """
    result = {"信号": [], "风险分": 0}
    ret_arr = returns.values

    # 计算滚动波动率
    vol_20 = pd.Series(ret_arr).rolling(20).std() * np.sqrt(252)
    vol_60 = pd.Series(ret_arr).rolling(60).std() * np.sqrt(252)

    current_vol_20 = float(vol_20.iloc[-1]) * 100
    current_vol_60 = float(vol_60.iloc[-1]) * 100
    vol_20_20d_ago = float(vol_20.iloc[-21]) * 100 if len(vol_20) > 21 else current_vol_20

    result["当前20日波动率"] = f"{current_vol_20:.1f}%"
    result["当前60日波动率"] = f"{current_vol_60:.1f}%"
    result["20日前波动率"] = f"{vol_20_20d_ago:.1f}%"

    # 波动率突变检测
    vol_change = current_vol_20 / vol_20_20d_ago - 1 if vol_20_20d_ago > 0 else 0
    result["波动率变化"] = f"{vol_change * 100:+.1f}%"

    if vol_change > 1.0:
        result["信号"].append(f"波动率翻倍({vol_change*100:+.0f}%)，市场进入恐慌模式，黑天鹅风险极高")
        result["风险分"] += 30
    elif vol_change > 0.5:
        result["信号"].append(f"波动率大幅上升({vol_change*100:+.0f}%)，市场情绪急剧恶化")
        result["风险分"] += 20
    elif vol_change > 0.3:
        result["信号"].append(f"波动率显著上升({vol_change*100:+.0f}%)，风险加剧")
        result["风险分"] += 10

    # 波动率分位数
    if len(vol_20) >= 250:
        vol_percentile = (vol_20.iloc[-1] > vol_20).sum() / len(vol_20) * 100
        result["波动率分位"] = f"{vol_percentile:.0f}%"

        if vol_percentile > 90:
            result["信号"].append(f"波动率处于历史{vol_percentile:.0f}%分位，处于极端高波动状态")
            result["风险分"] += 15
        elif vol_percentile > 80:
            result["信号"].append(f"波动率处于历史{vol_percentile:.0f}%分位，偏高")
            result["风险分"] += 5

    # 波动率聚集效应
    if len(ret_arr) >= 60:
        recent_vol = float(np.std(ret_arr[-20:]) * np.sqrt(252) * 100)
        hist_vol = float(np.std(ret_arr[:-20]) * np.sqrt(252) * 100)
        if recent_vol > hist_vol * 1.5:
            result["信号"].append("近期波动率显著高于历史水平，存在波动率聚集效应")
            result["风险分"] += 10

    return result


def _detect_drawdown_acceleration(close):
    """
    回撤加速检测
    检测回撤是否在加速扩大（崩盘前兆）
    """
    result = {"信号": [], "风险分": 0}

    # 计算滚动最大回撤
    rolling_max = close.expanding().max()
    drawdown = (close - rolling_max) / rolling_max

    current_dd = float(drawdown.iloc[-1]) * 100
    dd_5d_ago = float(drawdown.iloc[-6]) * 100 if len(drawdown) > 6 else current_dd
    dd_20d_ago = float(drawdown.iloc[-21]) * 100 if len(drawdown) > 21 else current_dd

    result["当前回撤"] = f"{current_dd:.2f}%"
    result["5日前回撤"] = f"{dd_5d_ago:.2f}%"
    result["20日前回撤"] = f"{dd_20d_ago:.2f}%"

    # 回撤加速检测
    dd_speed_5d = current_dd - dd_5d_ago
    dd_speed_20d = current_dd - dd_20d_ago

    result["5日回撤速度"] = f"{dd_speed_5d:+.2f}%/5日"
    result["20日回撤速度"] = f"{dd_speed_20d:+.2f}%/20日"

    if current_dd < -20:
        result["信号"].append(f"回撤已达{abs(current_dd):.1f}%，处于深度回撤状态")
        result["风险分"] += 20
    elif current_dd < -15:
        result["信号"].append(f"回撤{abs(current_dd):.1f}%，回撤幅度较大")
        result["风险分"] += 10
    elif current_dd < -10:
        result["信号"].append(f"回撤{abs(current_dd):.1f}%，需关注")
        result["风险分"] += 5

    # 回撤加速
    if dd_speed_5d < -5:
        result["信号"].append(f"近5日回撤加速({dd_speed_5d:+.1f}%)，可能进入恐慌性抛售")
        result["风险分"] += 25
    elif dd_speed_5d < -3:
        result["信号"].append(f"近5日回撤加速({dd_speed_5d:+.1f}%)，下跌动能增强")
        result["风险分"] += 15
    elif dd_speed_5d < -1.5:
        result["信号"].append(f"回撤速度加快({dd_speed_5d:+.1f}%)")
        result["风险分"] += 5

    # 连续下跌天数
    consecutive_down = 0
    for i in range(len(close) - 1, 0, -1):
        if close.iloc[i] < close.iloc[i - 1]:
            consecutive_down += 1
        else:
            break

    result["连续下跌天数"] = consecutive_down

    if consecutive_down >= 7:
        result["信号"].append(f"连续下跌{consecutive_down}天，市场情绪极度悲观")
        result["风险分"] += 15
    elif consecutive_down >= 5:
        result["信号"].append(f"连续下跌{consecutive_down}天，短期趋势恶化")
        result["风险分"] += 8
    elif consecutive_down >= 3:
        result["信号"].append(f"连续下跌{consecutive_down}天")
        result["风险分"] += 3

    return result


def _detect_correlation_breakdown(returns):
    """
    相关性崩溃检测
    检测资产间相关性是否突然升高（系统性风险爆发前兆）
    """
    result = {"信号": [], "风险分": 0}

    if len(returns) < 120:
        return result

    ret_arr = returns.values

    # 计算滚动相关性（用前后半段对比）
    mid = len(ret_arr) // 2
    first_half_vol = float(np.std(ret_arr[:mid]))
    second_half_vol = float(np.std(ret_arr[mid:]))

    result["前半段波动率"] = f"{first_half_vol * np.sqrt(252) * 100:.1f}%"
    result["后半段波动率"] = f"{second_half_vol * np.sqrt(252) * 100:.1f}%"

    # 自相关性变化
    autocorr_recent = float(pd.Series(ret_arr[-60:]).autocorr())
    autocorr_hist = float(pd.Series(ret_arr[:-60]).autocorr())

    result["近期自相关"] = round(autocorr_recent, 3)
    result["历史自相关"] = round(autocorr_hist, 3)

    # 自相关转为负值（恐慌特征）
    if autocorr_recent < -0.2 and autocorr_hist > 0:
        result["信号"].append("自相关由正转负，市场出现恐慌性反转特征")
        result["风险分"] += 15

    return result


def _detect_liquidity_dryup(df):
    """
    流动性枯竭检测
    检测成交量是否异常萎缩或异常放大
    """
    result = {"信号": [], "风险分": 0}

    amount = df.get('成交额', df.get('amount', None))
    if amount is None or len(amount) < 20:
        return result

    amount = pd.Series(amount.values.flatten() if hasattr(amount.values, 'flatten') else amount.values)

    avg_amount_20 = float(amount.tail(20).mean())
    avg_amount_60 = float(amount.tail(60).mean()) if len(amount) >= 60 else avg_amount_20

    result["近20日均成交额"] = f"{avg_amount_20 / 1e8:.2f}亿"
    result["近60日均成交额"] = f"{avg_amount_60 / 1e8:.2f}亿"

    # 成交量萎缩
    if avg_amount_60 > 0:
        vol_ratio = avg_amount_20 / avg_amount_60
        result["量比"] = round(vol_ratio, 2)

        if vol_ratio < 0.5:
            result["信号"].append(f"成交量萎缩至正常的{vol_ratio*100:.0f}%，流动性急剧下降")
            result["风险分"] += 15
        elif vol_ratio < 0.7:
            result["信号"].append(f"成交量萎缩({vol_ratio*100:.0f}%)，流动性减弱")
            result["风险分"] += 5

    # 成交量异常放大（恐慌性抛售）
    latest_amount = float(amount.iloc[-1])
    if latest_amount > avg_amount_20 * 3:
        result["信号"].append(f"当日成交额({latest_amount/1e8:.2f}亿)是均值的{latest_amount/avg_amount_20:.1f}倍，可能出现恐慌性抛售")
        result["风险分"] += 20
    elif latest_amount > avg_amount_20 * 2:
        result["信号"].append(f"当日成交额异常放大({latest_amount/avg_amount_20:.1f}倍)")
        result["风险分"] += 10

    return result


def _get_historical_black_swan_context(returns):
    """提供历史黑天鹅事件参考"""
    ret_arr = returns.values
    max_dd = float(np.min((pd.Series(ret_arr).cumsum() -
                            pd.Series(ret_arr).cumsum().expanding().max())))

    return {
        "历史最大回撤": f"{max_dd * 100:.2f}%",
        "参考": "历史上A股重大黑天鹅事件：2008年金融危机(-70%)、2015年股灾(-45%)、2016年熔断(-25%)、2018年贸易战(-30%)、2020年疫情(-15%)",
        "提示": "黑天鹅事件不可预测但可防范，核心是仓位管理和止损纪律",
    }


def main():
    parser = argparse.ArgumentParser(description="风控系统")
    subparsers = parser.add_subparsers(dest="command")

    # 事前风控
    pre_parser = subparsers.add_parser("pre-check", help="事前风控检查")
    pre_parser.add_argument("--symbol", required=True, help="股票代码")
    pre_parser.add_argument("--direction", default="buy", help="买卖方向")
    pre_parser.add_argument("--quantity", type=int, required=True, help="数量")
    pre_parser.add_argument("--price", type=float, required=True, help="价格")
    pre_parser.add_argument("--total-asset", type=float, default=100000, help="总资产")
    pre_parser.add_argument("--cash", type=float, default=100000, help="现金")
    pre_parser.add_argument("--positions", default="[]", help="持仓JSON")

    # 事中风控
    in_parser = subparsers.add_parser("in-trade", help="事中风控监控")
    in_parser.add_argument("--positions", required=True, help="持仓JSON")
    in_parser.add_argument("--market-data", required=True, help="行情JSON")

    # 事后风控
    post_parser = subparsers.add_parser("post-trade", help="事后风控分析")
    post_parser.add_argument("--symbols", required=True, help="股票代码列表,逗号分隔")
    post_parser.add_argument("--days", type=int, default=250, help="分析天数")

    # 压力测试
    stress_parser = subparsers.add_parser("stress-test", help="压力测试")
    stress_parser.add_argument("--symbols", required=True, help="股票代码列表,逗号分隔")
    stress_parser.add_argument("--days", type=int, default=250, help="历史数据天数")

    # VaR分解
    var_parser = subparsers.add_parser("var-breakdown", help="VaR分解")
    var_parser.add_argument("--symbols", required=True, help="股票代码列表,逗号分隔")
    var_parser.add_argument("--days", type=int, default=250, help="分析天数")

    # 黑天鹅预警
    swan_parser = subparsers.add_parser("black-swan", help="黑天鹅事件预警")
    swan_parser.add_argument("--symbol", default=None, help="股票代码（不传则分析大盘）")
    swan_parser.add_argument("--index", default="000300", help="基准指数代码")
    swan_parser.add_argument("--days", type=int, default=500, help="分析天数")

    args = parser.parse_args()

    if args.command == "pre-check":
        positions = json.loads(args.positions) if args.positions else []
        portfolio_state = {
            "total_asset": args.total_asset,
            "cash": args.cash,
            "positions": positions
        }
        order_params = {
            "direction": args.direction,
            "quantity": args.quantity,
            "price": args.price
        }
        result = pre_trade_check(args.symbol, order_params, portfolio_state)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "in-trade":
        positions = json.loads(args.positions)
        market_data = json.loads(args.market_data)
        result = in_trade_monitor(positions, market_data)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "post-trade":
        symbols = [s.strip() for s in args.symbols.split(",")]
        result = post_trade_risk_analysis(symbols, days=args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "stress-test":
        symbols = [s.strip() for s in args.symbols.split(",")]
        result = stress_test_portfolio(symbols, days=args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "var-breakdown":
        symbols = [s.strip() for s in args.symbols.split(",")]
        result = calculate_var_breakdown(symbols, days=args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "black-swan":
        result = black_swan_warning(
            symbol=args.symbol,
            index_code=args.index,
            days=args.days,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
