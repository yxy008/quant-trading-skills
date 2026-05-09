#!/usr/bin/env python3
"""
蒙特卡洛模拟与过拟合检测
- 蒙特卡洛模拟：通过随机抽样评估策略稳健性
- 过拟合检测：参数敏感性分析、样本内外对比、退化测试
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


def monte_carlo_simulation(returns, num_simulations=1000, horizon=252,
                            initial_capital=100000, confidence=0.95):
    """
    蒙特卡洛模拟
    基于历史收益率分布，模拟未来可能的权益路径
    """
    returns_arr = np.array(returns)
    mean_ret = np.mean(returns_arr)
    std_ret = np.std(returns_arr, ddof=1)

    simulations = np.zeros((num_simulations, horizon + 1))
    simulations[:, 0] = initial_capital

    np.random.seed(42)
    for i in range(num_simulations):
        random_returns = np.random.normal(mean_ret, std_ret, horizon)
        path = initial_capital * np.cumprod(1 + random_returns)
        simulations[i, 1:] = path

    final_values = simulations[:, -1]
    final_returns = (final_values / initial_capital - 1) * 100

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    percentile_values = {}
    for p in percentiles:
        idx = int(len(final_values) * p / 100)
        percentile_values[f"P{p}"] = round(float(np.percentile(final_returns, p)), 2)

    var_95 = float(np.percentile(final_returns, 5))
    cvar_95 = float(np.mean(final_returns[final_returns <= var_95]))

    prob_profit = float(np.mean(final_returns > 0)) * 100
    prob_loss_10 = float(np.mean(final_returns < -10)) * 100
    prob_loss_20 = float(np.mean(final_returns < -20)) * 100

    median_path_idx = np.argmin(np.abs(final_returns - np.median(final_returns)))
    worst_path_idx = np.argmin(final_returns)
    best_path_idx = np.argmax(final_returns)

    path_samples = {
        "中位数路径": [round(float(v), 2) for v in simulations[median_path_idx][::21]],
        "最差路径": [round(float(v), 2) for v in simulations[worst_path_idx][::21]],
        "最优路径": [round(float(v), 2) for v in simulations[best_path_idx][::21]]
    }

    return {
        "模拟次数": num_simulations,
        "预测周期": f"{horizon}个交易日",
        "初始资金": initial_capital,
        "分位数收益": percentile_values,
        "VaR(95%)": round(var_95, 2),
        "CVaR(95%)": round(cvar_95, 2),
        "盈利概率": round(prob_profit, 1),
        "亏损超10%概率": round(prob_loss_10, 1),
        "亏损超20%概率": round(prob_loss_20, 1),
        "预期年化收益": round(float(mean_ret * 252 * 100), 2),
        "预期年化波动": round(float(std_ret * np.sqrt(252) * 100), 2),
        "样本路径": path_samples
    }


def monte_carlo_with_shocks(returns, num_simulations=500, horizon=252,
                             initial_capital=100000):
    """
    带尾部风险的蒙特卡洛模拟
    使用t分布替代正态分布，更好地捕捉肥尾风险
    """
    from scipy import stats as scipy_stats

    returns_arr = np.array(returns)
    df_t, loc, scale = scipy_stats.t.fit(returns_arr)

    simulations = np.zeros((num_simulations, horizon + 1))
    simulations[:, 0] = initial_capital

    np.random.seed(42)
    for i in range(num_simulations):
        random_returns = scipy_stats.t.rvs(df_t, loc=loc, scale=scale, size=horizon)
        path = initial_capital * np.cumprod(1 + random_returns)
        simulations[i, 1:] = path

    final_values = simulations[:, -1]
    final_returns = (final_values / initial_capital - 1) * 100

    var_95 = float(np.percentile(final_returns, 5))
    var_99 = float(np.percentile(final_returns, 1))
    cvar_95 = float(np.mean(final_returns[final_returns <= var_95]))
    cvar_99 = float(np.mean(final_returns[final_returns <= var_99]))

    return {
        "模拟次数": num_simulations,
        "分布模型": "t分布（肥尾）",
        "自由度": round(float(df_t), 2),
        "VaR(95%)": round(var_95, 2),
        "VaR(99%)": round(var_99, 2),
        "CVaR(95%)": round(cvar_95, 2),
        "CVaR(99%)": round(cvar_99, 2),
        "尾部风险说明": "t分布比正态分布更能捕捉极端事件风险"
    }


def detect_overfitting(backtest_metrics, param_sensitivity=None, in_sample_returns=None,
                        out_sample_returns=None):
    """
    过拟合检测
    综合多种方法评估策略是否存在过拟合
    """
    checks = {}
    total_score = 0
    max_score = 0

    # 1. 交易次数检查
    trade_count = backtest_metrics.get("交易总次数", backtest_metrics.get("交易次数", 0))
    max_score += 20
    if trade_count >= 30:
        checks["交易次数"] = {"结果": f"{trade_count}次 - 充足", "得分": 20, "说明": "样本量足够，统计显著"}
        total_score += 20
    elif trade_count >= 15:
        checks["交易次数"] = {"结果": f"{trade_count}次 - 一般", "得分": 12, "说明": "样本量尚可，但需谨慎"}
        total_score += 12
    elif trade_count >= 5:
        checks["交易次数"] = {"结果": f"{trade_count}次 - 偏少", "得分": 5, "说明": "样本量不足，可能存在偶然性"}
        total_score += 5
    else:
        checks["交易次数"] = {"结果": f"{trade_count}次 - 严重不足", "得分": 0, "说明": "交易次数太少，结果不可靠"}

    # 2. 夏普比率合理性检查
    sharpe = backtest_metrics.get("夏普比率", 0)
    max_score += 20
    if sharpe > 3.0:
        checks["夏普比率"] = {"结果": f"{sharpe} - 过高", "得分": 5, "说明": "夏普比率异常高，可能存在过拟合或未来函数"}
        total_score += 5
    elif sharpe > 2.0:
        checks["夏普比率"] = {"结果": f"{sharpe} - 优秀", "得分": 15, "说明": "夏普比率较高，需确认样本外表现"}
        total_score += 15
    elif sharpe > 1.0:
        checks["夏普比率"] = {"结果": f"{sharpe} - 良好", "得分": 20, "说明": "夏普比率合理"}
        total_score += 20
    elif sharpe > 0:
        checks["夏普比率"] = {"结果": f"{sharpe} - 偏低", "得分": 10, "说明": "风险调整后收益偏低"}
        total_score += 10
    else:
        checks["夏普比率"] = {"结果": f"{sharpe} - 负值", "得分": 0, "说明": "策略表现不如无风险收益"}

    # 3. 胜率检查
    win_rate = backtest_metrics.get("胜率", 0)
    max_score += 15
    if 40 <= win_rate <= 65:
        checks["胜率"] = {"结果": f"{win_rate}% - 合理", "得分": 15, "说明": "胜率在合理范围内"}
        total_score += 15
    elif win_rate > 65:
        checks["胜率"] = {"结果": f"{win_rate}% - 偏高", "得分": 8, "说明": "胜率过高，可能过度拟合历史数据"}
        total_score += 8
    else:
        checks["胜率"] = {"结果": f"{win_rate}% - 偏低", "得分": 5, "说明": "胜率偏低，依赖少数大盈利交易"}
        total_score += 5

    # 4. 最大回撤检查
    max_dd = abs(backtest_metrics.get("最大回撤", 0))
    max_score += 15
    if max_dd < 10:
        checks["最大回撤"] = {"结果": f"{max_dd}% - 很低", "得分": 10, "说明": "回撤极低，需确认是否过于保守或存在未来函数"}
        total_score += 10
    elif max_dd < 20:
        checks["最大回撤"] = {"结果": f"{max_dd}% - 可控", "得分": 15, "说明": "回撤在可接受范围"}
        total_score += 15
    elif max_dd < 35:
        checks["最大回撤"] = {"结果": f"{max_dd}% - 偏高", "得分": 8, "说明": "回撤较大，需关注风险控制"}
        total_score += 8
    else:
        checks["最大回撤"] = {"结果": f"{max_dd}% - 很高", "得分": 3, "说明": "回撤过大，策略风险较高"}

    # 5. 参数敏感性分析
    if param_sensitivity:
        max_score += 15
        cv_values = [s.get("变异系数", 0) for s in param_sensitivity.values()]
        avg_cv = np.mean(cv_values) if cv_values else 0
        if avg_cv < 0.1:
            checks["参数敏感性"] = {"结果": "低敏感", "得分": 15, "说明": "参数变化对结果影响小，策略稳健"}
            total_score += 15
        elif avg_cv < 0.3:
            checks["参数敏感性"] = {"结果": "中敏感", "得分": 10, "说明": "参数有一定影响，需注意参数选择"}
            total_score += 10
        else:
            checks["参数敏感性"] = {"结果": "高敏感", "得分": 3, "说明": "参数变化对结果影响大，存在过拟合风险"}
            total_score += 3

    # 6. 样本内外对比
    if in_sample_returns is not None and out_sample_returns is not None:
        max_score += 15
        gap = in_sample_returns - out_sample_returns
        if gap < 5:
            checks["样本内外对比"] = {"结果": f"差距{gap:.1f}% - 小", "得分": 15, "说明": "样本内外表现一致，策略泛化能力强"}
            total_score += 15
        elif gap < 15:
            checks["样本内外对比"] = {"结果": f"差距{gap:.1f}% - 中", "得分": 8, "说明": "样本外表现有所下降"}
            total_score += 8
        else:
            checks["样本内外对比"] = {"结果": f"差距{gap:.1f}% - 大", "得分": 2, "说明": "样本外表现显著下降，严重过拟合"}
            total_score += 2

    score_pct = (total_score / max_score * 100) if max_score > 0 else 0

    if score_pct >= 80:
        risk_level = "低 - 策略稳健，过拟合风险小"
    elif score_pct >= 60:
        risk_level = "中 - 存在一定过拟合风险，建议优化"
    elif score_pct >= 40:
        risk_level = "偏高 - 过拟合风险较大，需谨慎使用"
    else:
        risk_level = "高 - 严重过拟合，不建议实盘使用"

    return {
        "综合评分": round(score_pct, 1),
        "满分": max_score,
        "得分": total_score,
        "风险等级": risk_level,
        "各项检查": checks,
        "建议": _get_overfitting_advice(score_pct, checks)
    }


def _get_overfitting_advice(score, checks):
    """根据过拟合检测结果给出建议"""
    advice = []

    if score < 60:
        advice.append("策略存在明显过拟合，建议：")
        advice.append("1. 简化策略逻辑，减少参数数量")
        advice.append("2. 使用Walk-Forward优化验证参数稳定性")
        advice.append("3. 增加样本外测试周期")
        advice.append("4. 考虑加入正则化约束")
    elif score < 80:
        advice.append("策略有一定过拟合风险，建议：")
        advice.append("1. 进行样本外验证")
        advice.append("2. 检查参数敏感性")
        advice.append("3. 考虑使用更保守的参数")
    else:
        advice.append("策略过拟合风险较低，建议：")
        advice.append("1. 持续监控实盘表现")
        advice.append("2. 定期重新评估策略有效性")

    for check_name, check_result in checks.items():
        if check_result["得分"] < check_result.get("max_possible", 10):
            if check_name == "交易次数":
                advice.append("- 交易次数不足，建议延长回测周期或放宽交易条件")
            elif check_name == "夏普比率" and check_result["得分"] <= 5:
                advice.append("- 夏普比率异常，请检查是否有未来函数或数据泄露")
            elif check_name == "参数敏感性" and check_result["得分"] <= 5:
                advice.append("- 参数过于敏感，建议简化策略或使用集成方法")

    return advice


def stress_test_scenarios(returns, initial_capital=100000):
    """
    压力测试场景模拟
    模拟多种极端市场情况下的策略表现
    """
    returns_arr = np.array(returns)
    mean_ret = np.mean(returns_arr)
    std_ret = np.std(returns_arr, ddof=1)

    scenarios = {
        "2008金融危机": {
            "描述": "连续大幅下跌，波动率飙升",
            "日收益率序列": _generate_crash_sequence(mean_ret, std_ret, crash_days=30, crash_intensity=3.0),
        },
        "2015股灾": {
            "描述": "快速暴跌后反弹再下跌",
            "日收益率序列": _generate_v_shape_crash(mean_ret, std_ret, crash_days=20, rebound_days=10),
        },
        "2020疫情冲击": {
            "描述": "短期急跌后快速反弹",
            "日收益率序列": _generate_sharp_v_recovery(mean_ret, std_ret, crash_days=10, recovery_days=15),
        },
        "慢熊市": {
            "描述": "持续阴跌，波动率正常",
            "日收益率序列": _generate_slow_bear(mean_ret, std_ret, days=60, daily_drift=-0.003),
        },
        "高波动震荡": {
            "描述": "大幅双向波动，方向不明",
            "日收益率序列": _generate_high_vol_range(mean_ret, std_ret, days=60, vol_multiplier=2.5),
        },
        "流动性危机": {
            "描述": "连续跌停，无法卖出",
            "日收益率序列": _generate_liquidity_crisis(days=15, limit_down_pct=-0.10),
        },
    }

    results = {}
    for name, scenario in scenarios.items():
        scenario_returns = scenario["日收益率序列"]
        path = initial_capital * np.cumprod(1 + scenario_returns)
        final_value = path[-1]
        total_return = (final_value / initial_capital - 1) * 100
        max_dd = float(np.min(path / np.maximum.accumulate(path) - 1) * 100)

        results[name] = {
            "描述": scenario["描述"],
            "最终权益": round(float(final_value), 2),
            "总收益率": round(float(total_return), 2),
            "最大回撤": round(max_dd, 2),
            "最低权益": round(float(np.min(path)), 2),
            "是否击穿": "是" if final_value < initial_capital * 0.7 else "否",
        }

    # 综合压力评估
    worst_return = min(r["总收益率"] for r in results.values())
    worst_dd = min(r["最大回撤"] for r in results.values())
    breach_count = sum(1 for r in results.values() if r["是否击穿"] == "是")

    if worst_return > -20 and worst_dd > -30:
        resilience = "强 - 策略在极端场景下表现稳健"
    elif worst_return > -40 and worst_dd > -50:
        resilience = "中 - 策略在极端场景下有一定抗压能力"
    else:
        resilience = "弱 - 策略在极端场景下可能遭受重大损失"

    return {
        "场景分析": results,
        "综合评估": {
            "最差收益率": round(worst_return, 2),
            "最差回撤": round(worst_dd, 2),
            "击穿场景数": breach_count,
            "抗压能力": resilience,
        },
        "建议": _get_stress_advice(resilience, breach_count),
    }


def _generate_crash_sequence(mean_ret, std_ret, crash_days=30, crash_intensity=3.0):
    """生成金融危机式暴跌序列"""
    np.random.seed(12345)
    sequence = []
    for i in range(crash_days):
        shock = np.random.normal(mean_ret - crash_intensity * std_ret, std_ret * 2)
        sequence.append(min(shock, -0.02))
    for i in range(30):
        sequence.append(np.random.normal(mean_ret, std_ret * 1.5))
    return np.array(sequence)


def _generate_v_shape_crash(mean_ret, std_ret, crash_days=20, rebound_days=10):
    """生成V型暴跌反弹序列"""
    np.random.seed(23456)
    sequence = []
    for i in range(crash_days):
        sequence.append(np.random.normal(-0.03, 0.02))
    for i in range(rebound_days):
        sequence.append(np.random.normal(0.02, 0.015))
    for i in range(20):
        sequence.append(np.random.normal(mean_ret, std_ret))
    return np.array(sequence)


def _generate_sharp_v_recovery(mean_ret, std_ret, crash_days=10, recovery_days=15):
    """生成急跌急反弹序列（疫情式）"""
    np.random.seed(34567)
    sequence = []
    for i in range(crash_days):
        sequence.append(np.random.normal(-0.04, 0.025))
    for i in range(recovery_days):
        sequence.append(np.random.normal(0.03, 0.02))
    for i in range(20):
        sequence.append(np.random.normal(mean_ret, std_ret))
    return np.array(sequence)


def _generate_slow_bear(mean_ret, std_ret, days=60, daily_drift=-0.003):
    """生成慢熊市序列"""
    np.random.seed(45678)
    sequence = np.random.normal(daily_drift, std_ret, days)
    return sequence


def _generate_high_vol_range(mean_ret, std_ret, days=60, vol_multiplier=2.5):
    """生成高波动震荡序列"""
    np.random.seed(56789)
    sequence = np.random.normal(0, std_ret * vol_multiplier, days)
    return sequence


def _generate_liquidity_crisis(days=15, limit_down_pct=-0.10):
    """生成流动性危机序列（连续跌停）"""
    sequence = np.full(days, limit_down_pct)
    return sequence


def _get_stress_advice(resilience, breach_count):
    """根据压力测试结果给出建议"""
    if "强" in resilience:
        return ["策略抗压能力强，可维持当前仓位管理方案"]
    elif "中" in resilience:
        return [
            "建议设置更严格的止损线",
            "极端行情下考虑降低仓位至50%以下",
            "增加对冲工具（如期权、反向ETF）",
        ]
    else:
        return [
            "策略抗压能力不足，必须重新设计风控方案",
            "建议加入趋势过滤，熊市自动降低仓位",
            "考虑加入波动率自适应仓位管理",
            f"{breach_count}个场景出现击穿，需设置硬止损线",
        ]


def parameter_sensitivity_analysis(df, strategy_id, base_params, param_name,
                                    values, initial_capital=100000,
                                    commission_rate=0.0003, slippage=0.001):
    """
    参数敏感性分析
    测试单个参数在不同取值下的策略表现变化
    """
    from optimizer_cli import generate_signals, run_backtest

    results = []
    for val in values:
        test_params = base_params.copy()
        test_params[param_name] = val

        signals_df = generate_signals(df, strategy_id, test_params)
        bt_result = run_backtest(
            df, signals_df['signal'], initial_capital,
            commission_rate, slippage
        )

        results.append({
            "参数值": val,
            "总收益率": bt_result["总收益率"],
            "夏普比率": bt_result["夏普比率"],
            "最大回撤": bt_result["最大回撤"]
        })

    returns_list = [r["总收益率"] for r in results]
    sharpe_list = [r["夏普比率"] for r in results]

    return {
        "参数名": param_name,
        "测试值": values,
        "收益率范围": f"{min(returns_list):.1f}% ~ {max(returns_list):.1f}%",
        "收益率标准差": round(float(np.std(returns_list, ddof=1)), 2),
        "夏普范围": f"{min(sharpe_list):.2f} ~ {max(sharpe_list):.2f}",
        "敏感度": "高" if np.std(returns_list, ddof=1) > 5 else "中" if np.std(returns_list, ddof=1) > 2 else "低",
        "详细结果": results
    }


def main():
    parser = argparse.ArgumentParser(description="蒙特卡洛模拟与过拟合检测")
    parser.add_argument("--symbol", default="600519", help="股票代码")
    parser.add_argument("--days", type=int, default=500, help="历史数据天数")
    parser.add_argument("--simulations", type=int, default=1000, help="蒙特卡洛模拟次数")
    parser.add_argument("--horizon", type=int, default=252, help="预测周期（交易日）")
    parser.add_argument("--capital", type=float, default=100000, help="初始资金")
    parser.add_argument("--output", default="", help="输出JSON文件路径")

    args = parser.parse_args()

    df = get_stock_kline(args.symbol, args.days)
    if df is None or df.empty:
        print(json.dumps({"error": f"无法获取 {args.symbol} 的K线数据"}, ensure_ascii=False))
        return

    close = df['close']
    daily_returns = close.pct_change().dropna().tolist()

    mc_result = monte_carlo_simulation(
        daily_returns, args.simulations, args.horizon, args.capital
    )

    try:
        mc_shocks = monte_carlo_with_shocks(
            daily_returns, min(args.simulations, 500), args.horizon, args.capital
        )
    except Exception:
        mc_shocks = {"error": "scipy未安装，无法进行t分布模拟"}

    result = {
        "股票代码": args.symbol,
        "数据天数": len(df),
        "数据范围": f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}",
        "蒙特卡洛模拟(正态)": mc_result,
        "蒙特卡洛模拟(肥尾)": mc_shocks
    }

    output = json.dumps(result, ensure_ascii=False, indent=2, default=str)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)

    print(output)


if __name__ == "__main__":
    main()
