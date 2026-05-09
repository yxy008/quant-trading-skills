#!/usr/bin/env python3
"""
实时风控监控面板模块 - 盘中动态风控预警、风险指标实时计算
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
    import numpy as np
except ImportError:
    np = None

from data_utils import get_stock_kline


def realtime_risk_check(positions, market_data=None):
    """
    实时风险检查
    检查当前持仓的各项风险指标

    positions: [{"symbol": "600519", "quantity": 1000, "cost": 1800, "current_price": 1850}, ...]
    market_data: {"index_change": -1.5, "market_volatility": 0.02, ...}
    """
    if not positions:
        return {"error": "请提供持仓数据"}

    alerts = []
    total_value = 0
    total_cost = 0
    total_pnl = 0

    position_details = []

    for pos in positions:
        symbol = pos.get("symbol", "")
        quantity = pos.get("quantity", 0)
        cost = pos.get("cost", 0)
        current_price = pos.get("current_price", cost)

        market_value = quantity * current_price
        cost_value = quantity * cost
        pnl = market_value - cost_value
        pnl_pct = (current_price / cost - 1) * 100 if cost > 0 else 0

        total_value += market_value
        total_cost += cost_value
        total_pnl += pnl

        # 个股风险检查
        pos_alerts = []

        # 1. 单只股票亏损超过阈值
        if pnl_pct < -10:
            pos_alerts.append({"级别": "严重", "类型": "大幅亏损", "描述": f"{symbol} 亏损 {abs(pnl_pct):.1f}%，超过10%警戒线"})
        elif pnl_pct < -5:
            pos_alerts.append({"级别": "警告", "类型": "亏损预警", "描述": f"{symbol} 亏损 {abs(pnl_pct):.1f}%，超过5%预警线"})

        # 2. 单只股票盈利超过阈值（止盈提醒）
        if pnl_pct > 30:
            pos_alerts.append({"级别": "提示", "类型": "止盈提醒", "描述": f"{symbol} 盈利 {pnl_pct:.1f}%，建议考虑止盈"})

        # 3. 价格接近成本（回本提醒）
        if abs(pnl_pct) < 1 and pnl_pct < 0:
            pos_alerts.append({"级别": "提示", "类型": "接近成本", "描述": f"{symbol} 价格接近成本，注意方向选择"})

        position_details.append({
            "股票代码": symbol,
            "持仓数量": quantity,
            "成本价": round(cost, 2),
            "当前价": round(current_price, 2),
            "市值": round(market_value, 2),
            "盈亏": round(pnl, 2),
            "盈亏比例": round(pnl_pct, 2),
            "风险预警": pos_alerts,
        })

        alerts.extend(pos_alerts)

    # 组合层面风险检查
    total_pnl_pct = (total_value / total_cost - 1) * 100 if total_cost > 0 else 0

    # 组合回撤预警
    if total_pnl_pct < -15:
        alerts.append({"级别": "严重", "类型": "组合大幅亏损", "描述": f"组合整体亏损 {abs(total_pnl_pct):.1f}%，超过15%警戒线"})
    elif total_pnl_pct < -8:
        alerts.append({"级别": "警告", "类型": "组合亏损预警", "描述": f"组合整体亏损 {abs(total_pnl_pct):.1f}%，超过8%预警线"})

    # 市场风险
    if market_data:
        index_change = market_data.get("index_change", 0)
        if index_change < -3:
            alerts.append({"级别": "严重", "类型": "市场大跌", "描述": f"大盘跌幅 {abs(index_change):.1f}%，系统性风险加剧"})
        elif index_change < -1.5:
            alerts.append({"级别": "警告", "类型": "市场下跌", "描述": f"大盘跌幅 {abs(index_change):.1f}%，注意系统性风险"})

    # 按级别排序
    severity_order = {"严重": 0, "警告": 1, "提示": 2}
    alerts.sort(key=lambda x: severity_order.get(x["级别"], 3))

    # 风险等级
    severe_count = sum(1 for a in alerts if a["级别"] == "严重")
    warning_count = sum(1 for a in alerts if a["级别"] == "警告")

    if severe_count > 0:
        risk_level = "高风险"
        risk_color = "red"
    elif warning_count > 2:
        risk_level = "中风险"
        risk_color = "orange"
    elif warning_count > 0:
        risk_level = "关注"
        risk_color = "yellow"
    else:
        risk_level = "正常"
        risk_color = "green"

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "风险等级": risk_level,
        "风险颜色": risk_color,
        "总市值": round(total_value, 2),
        "总成本": round(total_cost, 2),
        "总盈亏": round(total_pnl, 2),
        "总盈亏比例": round(total_pnl_pct, 2),
        "严重预警数": severe_count,
        "警告数": warning_count,
        "提示数": sum(1 for a in alerts if a["级别"] == "提示"),
        "预警列表": alerts,
        "持仓明细": position_details,
    }


def var_calculation(returns, confidence=0.95, method="historical"):
    """
    VaR (Value at Risk) 计算
    计算在给定置信水平下的最大可能损失

    returns: 日收益率序列
    confidence: 置信水平 (0.95, 0.99)
    method: "historical" | "parametric" | "monte_carlo"
    """
    if not returns or len(returns) < 10:
        return {"error": "数据量不足"}

    returns = np.array(returns)

    if method == "historical":
        # 历史模拟法
        var = np.percentile(returns, (1 - confidence) * 100)
        var_pct = abs(var * 100)

    elif method == "parametric":
        # 参数法（假设正态分布）
        from scipy import stats as scipy_stats
        mu = np.mean(returns)
        sigma = np.std(returns)
        z_score = scipy_stats.norm.ppf(1 - confidence)
        var = mu - z_score * sigma
        var_pct = abs(var * 100)

    elif method == "monte_carlo":
        # 蒙特卡洛模拟
        mu = np.mean(returns)
        sigma = np.std(returns)
        simulated = np.random.normal(mu, sigma, 10000)
        var = np.percentile(simulated, (1 - confidence) * 100)
        var_pct = abs(var * 100)

    else:
        var = np.percentile(returns, (1 - confidence) * 100)
        var_pct = abs(var * 100)

    # CVaR (条件VaR / 期望损失)
    cvar = np.mean(returns[returns <= var]) if sum(returns <= var) > 0 else var
    cvar_pct = abs(cvar * 100)

    return {
        "分析方法": f"{method}法",
        "置信水平": f"{confidence * 100}%",
        "VaR(日)": round(var_pct, 4),
        "CVaR(日)": round(cvar_pct, 4),
        "VaR(周)": round(var_pct * np.sqrt(5), 4),
        "VaR(月)": round(var_pct * np.sqrt(22), 4),
        "含义": f"在{confidence * 100}%置信水平下，单日最大损失不超过{var_pct:.2f}%",
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def drawdown_analysis(equity_curve):
    """
    回撤分析
    计算当前回撤、历史最大回撤等
    """
    if not equity_curve or len(equity_curve) < 5:
        return {"error": "数据量不足"}

    equity = np.array(equity_curve)
    n = len(equity)

    # 滚动最高点
    rolling_max = np.maximum.accumulate(equity)

    # 回撤序列
    drawdowns = (equity - rolling_max) / rolling_max * 100

    # 当前回撤
    current_dd = drawdowns[-1]

    # 最大回撤
    max_dd = np.min(drawdowns)
    max_dd_idx = np.argmin(drawdowns)

    # 当前回撤持续时间
    current_dd_start = n - 1
    for i in range(n - 1, -1, -1):
        if drawdowns[i] == 0:
            current_dd_start = i
            break
    current_dd_days = n - 1 - current_dd_start

    # 回撤恢复分析
    recovery_info = "已恢复"
    if current_dd < -2:
        recovery_info = f"当前回撤 {abs(current_dd):.1f}%，持续 {current_dd_days} 天"
    elif current_dd < -0.5:
        recovery_info = f"小幅回撤 {abs(current_dd):.1f}%"

    # 回撤分布
    dd_bins = {"0~-2%": 0, "-2%~-5%": 0, "-5%~-10%": 0, "-10%~-20%": 0, "-20%以上": 0}
    for dd in drawdowns:
        if dd > -2:
            dd_bins["0~-2%"] += 1
        elif dd > -5:
            dd_bins["-2%~-5%"] += 1
        elif dd > -10:
            dd_bins["-5%~-10%"] += 1
        elif dd > -20:
            dd_bins["-10%~-20%"] += 1
        else:
            dd_bins["-20%以上"] += 1

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "当前权益": round(equity[-1], 2),
        "历史最高": round(rolling_max[-1], 2),
        "当前回撤": round(current_dd, 2),
        "最大回撤": round(max_dd, 2),
        "最大回撤位置": f"第{max_dd_idx + 1}天",
        "当前回撤持续天数": current_dd_days,
        "恢复状态": recovery_info,
        "回撤分布": dd_bins,
    }


def risk_limits_check(positions, limits):
    """
    风控限额检查
    检查是否超过各项风控限额

    limits: {
        "max_single_position": 0.3,     # 单只股票最大仓位
        "max_sector_exposure": 0.4,     # 单行业最大暴露
        "max_total_leverage": 1.0,      # 最大总杠杆
        "max_daily_loss": 0.05,         # 单日最大亏损
        "max_consecutive_loss": 3,      # 最大连续亏损天数
    }
    """
    if not positions:
        return {"error": "请提供持仓数据"}

    total_value = sum(p.get("quantity", 0) * p.get("current_price", p.get("cost", 0)) for p in positions)
    if total_value <= 0:
        return {"error": "总市值为0"}

    violations = []

    # 检查单只股票仓位
    for pos in positions:
        symbol = pos.get("symbol", "")
        market_value = pos.get("quantity", 0) * pos.get("current_price", pos.get("cost", 0))
        weight = market_value / total_value

        max_single = limits.get("max_single_position", 0.3)
        if weight > max_single:
            violations.append({
                "类型": "单只股票超限",
                "股票": symbol,
                "当前权重": round(weight * 100, 1),
                "限额": round(max_single * 100, 1),
                "超出": round((weight - max_single) * 100, 1),
                "建议": f"建议减仓 {(weight - max_single) * total_value:.0f} 元",
            })

    # 检查行业暴露
    sector_exposure = {}
    for pos in positions:
        sector = pos.get("sector", "其他")
        market_value = pos.get("quantity", 0) * pos.get("current_price", pos.get("cost", 0))
        sector_exposure[sector] = sector_exposure.get(sector, 0) + market_value

    max_sector = limits.get("max_sector_exposure", 0.4)
    for sector, value in sector_exposure.items():
        weight = value / total_value
        if weight > max_sector:
            violations.append({
                "类型": "行业暴露超限",
                "行业": sector,
                "当前权重": round(weight * 100, 1),
                "限额": round(max_sector * 100, 1),
                "超出": round((weight - max_sector) * 100, 1),
                "建议": f"建议降低 {sector} 行业配置",
            })

    # 检查杠杆
    total_leverage = total_value / limits.get("nav", total_value)
    max_leverage = limits.get("max_total_leverage", 1.0)
    if total_leverage > max_leverage:
        violations.append({
            "类型": "杠杆超限",
            "当前杠杆": round(total_leverage, 2),
            "限额": max_leverage,
            "超出": round(total_leverage - max_leverage, 2),
            "建议": "建议降低杠杆水平",
        })

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "总市值": round(total_value, 2),
        "违规数量": len(violations),
        "是否合规": len(violations) == 0,
        "违规明细": violations,
        "限额配置": limits,
    }


def stress_test(positions, scenarios):
    """
    压力测试
    模拟极端市场情况下的组合表现

    scenarios: [
        {"name": "大盘跌5%", "index_change": -0.05, "sector_impacts": {"白酒": -0.08, "银行": -0.03}},
        {"name": "金融危机", "index_change": -0.15, "sector_impacts": {"白酒": -0.20, "银行": -0.10}},
    ]
    """
    if not positions:
        return {"error": "请提供持仓数据"}

    total_value = sum(p.get("quantity", 0) * p.get("current_price", p.get("cost", 0)) for p in positions)

    results = []
    for scenario in scenarios:
        scenario_name = scenario.get("name", "未命名场景")
        index_change = scenario.get("index_change", 0)
        sector_impacts = scenario.get("sector_impacts", {})

        scenario_value = 0
        position_impacts = []

        for pos in positions:
            symbol = pos.get("symbol", "")
            quantity = pos.get("quantity", 0)
            current_price = pos.get("current_price", pos.get("cost", 0))
            sector = pos.get("sector", "其他")

            # 个股跌幅 = 大盘跌幅 + 行业额外冲击
            sector_impact = sector_impacts.get(sector, index_change)
            stock_change = sector_impact * (1 + np.random.normal(0, 0.3))

            new_price = current_price * (1 + stock_change)
            new_value = quantity * new_price
            scenario_value += new_value

            position_impacts.append({
                "股票": symbol,
                "当前价": round(current_price, 2),
                "压力价": round(new_price, 2),
                "跌幅": round(stock_change * 100, 1),
                "当前市值": round(quantity * current_price, 2),
                "压力市值": round(new_value, 2),
                "损失": round(quantity * (current_price - new_price), 2),
            })

        total_loss = total_value - scenario_value
        loss_pct = (scenario_value / total_value - 1) * 100 if total_value > 0 else 0

        results.append({
            "场景": scenario_name,
            "大盘跌幅": round(index_change * 100, 1),
            "压力前市值": round(total_value, 2),
            "压力后市值": round(scenario_value, 2),
            "总损失": round(total_loss, 2),
            "损失比例": round(loss_pct, 1),
            "个股影响": position_impacts,
        })

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "当前总市值": round(total_value, 2),
        "测试场景数": len(scenarios),
        "压力测试结果": results,
    }


def main():
    parser = argparse.ArgumentParser(description="实时风控监控面板")
    subparsers = parser.add_subparsers(dest="action", help="操作")

    # 实时风险检查
    check_parser = subparsers.add_parser("check", help="实时风险检查")
    check_parser.add_argument("--positions", required=True, help="持仓数据JSON")
    check_parser.add_argument("--market", type=str, help="市场数据JSON")

    # VaR计算
    var_parser = subparsers.add_parser("var", help="VaR计算")
    var_parser.add_argument("--returns", required=True, help="收益率序列JSON")
    var_parser.add_argument("--confidence", type=float, default=0.95, help="置信水平")
    var_parser.add_argument("--method", default="historical", help="计算方法")

    # 回撤分析
    dd_parser = subparsers.add_parser("drawdown", help="回撤分析")
    dd_parser.add_argument("--equity", required=True, help="权益曲线JSON")

    # 限额检查
    limit_parser = subparsers.add_parser("limits", help="风控限额检查")
    limit_parser.add_argument("--positions", required=True, help="持仓数据JSON")
    limit_parser.add_argument("--limits", required=True, help="限额配置JSON")

    # 压力测试
    stress_parser = subparsers.add_parser("stress", help="压力测试")
    stress_parser.add_argument("--positions", required=True, help="持仓数据JSON")
    stress_parser.add_argument("--scenarios", required=True, help="测试场景JSON")

    args = parser.parse_args()

    try:
        if args.action == "check":
            positions = json.loads(args.positions)
            market = json.loads(args.market) if args.market else None
            result = realtime_risk_check(positions, market)
        elif args.action == "var":
            returns = json.loads(args.returns)
            result = var_calculation(returns, args.confidence, args.method)
        elif args.action == "drawdown":
            equity = json.loads(args.equity)
            result = drawdown_analysis(equity)
        elif args.action == "limits":
            positions = json.loads(args.positions)
            limits = json.loads(args.limits)
            result = risk_limits_check(positions, limits)
        elif args.action == "stress":
            positions = json.loads(args.positions)
            scenarios = json.loads(args.scenarios)
            result = stress_test(positions, scenarios)
        else:
            parser.print_help()
            return
    except json.JSONDecodeError as e:
        result = {"error": f"JSON解析失败: {str(e)}"}
    except Exception as e:
        result = {"error": str(e)}

    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
