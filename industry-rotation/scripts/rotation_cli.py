#!/usr/bin/env python3
"""
行业轮动策略 - 行业动量分析 / 相对强弱 / 轮动信号 / 行业排名
基于行业指数表现进行轮动配置
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

from data_utils import get_industry_index_data, get_industry_etf_data, get_index_kline


# 申万一级行业指数代码映射
SW_INDUSTRY_MAP = {
    "801010": "农林牧渔", "801020": "采掘", "801030": "化工", "801040": "钢铁",
    "801050": "有色金属", "801080": "电子", "801110": "家用电器", "801120": "食品饮料",
    "801130": "纺织服装", "801140": "轻工制造", "801150": "医药生物", "801160": "公用事业",
    "801170": "交通运输", "801180": "房地产", "801200": "商业贸易", "801210": "休闲服务",
    "801230": "综合", "801710": "建筑材料", "801720": "建筑装饰", "801730": "电气设备",
    "801740": "国防军工", "801750": "计算机", "801760": "传媒", "801770": "通信",
    "801780": "银行", "801790": "非银金融", "801880": "汽车", "801890": "机械设备",
}

# 行业对应的代表性ETF
INDUSTRY_ETF_MAP = {
    "食品饮料": "512690", "医药生物": "512010", "电子": "159997",
    "计算机": "512720", "银行": "512800", "非银金融": "512070",
    "有色金属": "512400", "国防军工": "512660", "汽车": "516110",
    "房地产": "512200", "通信": "515880", "传媒": "512980",
    "化工": "159870", "电气设备": "515790", "家用电器": "159996",
}


def calc_industry_momentum(df, periods=None):
    """计算行业动量指标，适配中文列名"""
    if periods is None:
        periods = [5, 10, 20, 60, 120]
    close = df['收盘']
    momentum = {}
    for p in periods:
        if len(close) >= p:
            momentum[f'{p}日动量'] = round(float((close.iloc[-1] / close.iloc[-p] - 1) * 100), 2)
        else:
            momentum[f'{p}日动量'] = None
    return momentum


def calc_relative_strength(industry_df, benchmark_df, periods=None):
    """计算行业相对基准的相对强弱，适配中文列名"""
    if periods is None:
        periods = [5, 10, 20, 60]
    if industry_df is None or benchmark_df is None:
        return {}

    ind_close = industry_df['收盘']
    bm_close = benchmark_df['收盘']

    common_idx = ind_close.index.intersection(bm_close.index)
    if len(common_idx) < 20:
        return {}

    ind_close = ind_close.loc[common_idx]
    bm_close = bm_close.loc[common_idx]

    rs = {}
    for p in periods:
        if len(ind_close) >= p:
            ind_ret = (ind_close.iloc[-1] / ind_close.iloc[-p] - 1) * 100
            bm_ret = (bm_close.iloc[-1] / bm_close.iloc[-p] - 1) * 100
            rs[f'{p}日相对强弱'] = round(float(ind_ret - bm_ret), 2)
        else:
            rs[f'{p}日相对强弱'] = None
    return rs


def calc_industry_volatility(df, periods=None):
    """计算行业波动率，适配中文列名"""
    if periods is None:
        periods = [20, 60]
    close = df['收盘']
    returns = close.pct_change().dropna()
    vol = {}
    for p in periods:
        if len(returns) >= p:
            vol[f'{p}日波动率'] = round(float(returns.tail(p).std() * np.sqrt(252) * 100), 2)
        else:
            vol[f'{p}日波动率'] = None
    return vol


def calc_industry_turnover(df):
    """计算行业成交额变化，适配中文列名"""
    if '成交额' not in df.columns:
        return {}
    amount = df['成交额']
    turnover = {}
    if len(amount) >= 5:
        turnover['5日均成交额'] = round(float(amount.tail(5).mean()) / 1e8, 2)
    if len(amount) >= 20:
        turnover['20日均成交额'] = round(float(amount.tail(20).mean()) / 1e8, 2)
    if len(amount) >= 5:
        turnover['量比'] = round(float(amount.iloc[-1] / amount.tail(5).mean()), 2)
    return turnover


# ==================== 行业轮动分析 ====================

def analyze_industry_rotation(industries=None, days=250, top_n=5):
    """
    行业轮动分析主函数
    参数:
        industries: 行业代码列表，默认使用全部申万一级行业
        days: 数据天数
        top_n: 推荐前N个行业
    返回:
        dict: 行业轮动分析结果
    """
    if industries is None:
        industries = list(SW_INDUSTRY_MAP.keys())

    # 获取基准指数（沪深300）
    benchmark_df = get_index_kline("sh000300", days=days)

    industry_results = {}
    failed_industries = []

    for ind_code in industries:
        ind_name = SW_INDUSTRY_MAP.get(ind_code, ind_code)
        df = get_industry_index_data(ind_code, days=days)

        if df is None or len(df) < 30:
            # 尝试用ETF数据
            etf_code = INDUSTRY_ETF_MAP.get(ind_name)
            if etf_code:
                df = get_industry_etf_data(etf_code, days=days)

        if df is None or len(df) < 30:
            failed_industries.append(ind_name)
            continue

        momentum = calc_industry_momentum(df)
        volatility = calc_industry_volatility(df)
        turnover = calc_industry_turnover(df)
        rs = calc_relative_strength(df, benchmark_df) if benchmark_df is not None else {}

        # 综合评分
        score = 0.0
        score_count = 0

        if momentum.get('20日动量') is not None:
            score += momentum['20日动量']
            score_count += 1
        if momentum.get('60日动量') is not None:
            score += momentum['60日动量'] * 0.5
            score_count += 0.5
        if rs.get('20日相对强弱') is not None:
            score += rs['20日相对强弱'] * 0.8
            score_count += 0.8
        if rs.get('60日相对强弱') is not None:
            score += rs['60日相对强弱'] * 0.4
            score_count += 0.4

        composite_score = round(score / score_count, 2) if score_count > 0 else 0.0

        industry_results[ind_name] = {
            "行业代码": ind_code,
            "动量指标": momentum,
            "波动率": volatility,
            "成交指标": turnover,
            "相对强弱": rs,
            "综合评分": composite_score
        }

        time.sleep(0.3)

    if not industry_results:
        return {"error": "无法获取任何行业数据"}

    # 排名
    ranked = sorted(industry_results.items(), key=lambda x: x[1]["综合评分"], reverse=True)
    top_industries = ranked[:top_n]
    bottom_industries = ranked[-top_n:] if len(ranked) >= top_n * 2 else []

    # 轮动信号
    rotation_signals = _generate_rotation_signals(ranked)

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "分析行业数": len(industry_results),
        "失败行业": failed_industries,
        "推荐行业": [
            {
                "排名": i + 1,
                "行业": name,
                "综合评分": info["综合评分"],
                "20日动量": info["动量指标"].get("20日动量"),
                "20日相对强弱": info["相对强弱"].get("20日相对强弱"),
                "60日动量": info["动量指标"].get("60日动量"),
            }
            for i, (name, info) in enumerate(top_industries)
        ],
        "弱势行业": [
            {
                "排名": len(ranked) - i,
                "行业": name,
                "综合评分": info["综合评分"],
            }
            for i, (name, info) in enumerate(bottom_industries)
        ],
        "轮动信号": rotation_signals,
        "全部排名": [
            {"排名": i + 1, "行业": name, "综合评分": info["综合评分"]}
            for i, (name, info) in enumerate(ranked)
        ],
        "行业详情": {name: info for name, info in industry_results.items()}
    }


def _generate_rotation_signals(ranked_industries):
    """生成轮动信号"""
    if len(ranked_industries) < 3:
        return {"信号": "数据不足", "建议": "需要更多行业数据"}

    top3 = ranked_industries[:3]
    bottom3 = ranked_industries[-3:]

    top_avg = np.mean([info["综合评分"] for _, info in top3])
    bottom_avg = np.mean([info["综合评分"] for _, info in bottom3])
    spread = top_avg - bottom_avg

    signals = {
        "强势行业": [name for name, _ in top3],
        "弱势行业": [name for name, _ in bottom3],
        "强弱差值": round(float(spread), 2),
    }

    if spread > 10:
        signals["信号"] = "强轮动"
        signals["建议"] = "行业分化明显，建议超配强势行业，低配弱势行业"
    elif spread > 5:
        signals["信号"] = "中等轮动"
        signals["建议"] = "行业有一定分化，可适度倾斜配置"
    elif spread > 2:
        signals["信号"] = "弱轮动"
        signals["建议"] = "行业分化不大，建议均衡配置"
    else:
        signals["信号"] = "无明显轮动"
        signals["建议"] = "市场缺乏明确方向，建议分散配置或观望"

    return signals


# ==================== 行业动量策略回测 ====================

def backtest_rotation_strategy(industries=None, days=500, top_n=3,
                                rebalance_freq=20, initial_capital=100000):
    """
    行业轮动策略回测
    策略: 每rebalance_freq天选择动量最强的top_n个行业等权配置
    """
    if industries is None:
        industries = list(SW_INDUSTRY_MAP.keys())

    # 获取所有行业数据
    industry_data = {}
    for ind_code in industries:
        ind_name = SW_INDUSTRY_MAP.get(ind_code, ind_code)
        df = get_industry_index_data(ind_code, days=days)
        if df is not None and len(df) >= 60:
            industry_data[ind_name] = df
        time.sleep(0.2)

    if len(industry_data) < 5:
        return {"error": f"有效行业数据不足，仅{len(industry_data)}个"}

    # 对齐日期
    all_dates = None
    for df in industry_data.values():
        if all_dates is None:
            all_dates = set(df.index)
        else:
            all_dates = all_dates.intersection(set(df.index))

    all_dates = sorted(all_dates)
    if len(all_dates) < rebalance_freq + 20:
        return {"error": "共同交易日不足"}

    # 回测
    capital = initial_capital
    holdings = {}
    daily_values = []
    rebalance_dates = all_dates[::rebalance_freq]

    for i, date in enumerate(all_dates):
        # 调仓日
        if date in rebalance_dates and i >= 60:
            # 计算各行业过去20日动量
            lookback_idx = max(0, i - 20)
            momentum_scores = {}
            for name, df in industry_data.items():
                if date in df.index:
                    date_loc = df.index.get_loc(date)
                    if date_loc >= 20:
                        past_close = df['close'].iloc[date_loc - 20]
                        current_close = df['close'].iloc[date_loc]
                        momentum_scores[name] = (current_close / past_close - 1) * 100

            # 选择top_n
            ranked = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
            selected = ranked[:top_n]

            # 等权分配
            weight = 1.0 / top_n
            holdings = {}
            for name, _ in selected:
                df = industry_data[name]
                if date in df.index:
                    holdings[name] = {
                        "weight": weight,
                        "entry_price": df.loc[date, 'close'],
                        "entry_date": date
                    }

        # 计算当日组合价值
        if holdings:
            portfolio_value = 0.0
            for name, h in holdings.items():
                df = industry_data[name]
                if date in df.index:
                    current_price = df.loc[date, 'close']
                    ret = current_price / h["entry_price"] - 1
                    portfolio_value += capital * h["weight"] * (1 + ret)
                else:
                    portfolio_value += capital * h["weight"]
            daily_values.append({
                "date": str(date)[:10],
                "value": round(portfolio_value, 2),
                "return": round((portfolio_value / capital - 1) * 100, 2)
            })
        else:
            daily_values.append({
                "date": str(date)[:10],
                "value": round(capital, 2),
                "return": 0.0
            })

    if not daily_values:
        return {"error": "回测无有效数据"}

    final_value = daily_values[-1]["value"]
    total_return = (final_value / initial_capital - 1) * 100

    # 计算指标
    returns = []
    for j in range(1, len(daily_values)):
        r = (daily_values[j]["value"] / daily_values[j - 1]["value"] - 1)
        returns.append(r)

    returns_arr = np.array(returns)
    annual_return = ((1 + total_return / 100) ** (252 / len(returns_arr)) - 1) * 100 if len(returns_arr) > 0 else 0
    annual_vol = float(np.std(returns_arr) * np.sqrt(252) * 100) if len(returns_arr) > 1 else 0
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    max_dd = _calc_max_drawdown([d["value"] for d in daily_values])

    return {
        "策略名称": "行业轮动策略",
        "初始资金": initial_capital,
        "最终价值": round(final_value, 2),
        "总收益率": round(total_return, 2),
        "年化收益率": round(annual_return, 2),
        "年化波动率": round(annual_vol, 2),
        "夏普比率": round(sharpe, 2),
        "最大回撤": round(max_dd, 2),
        "调仓频率": f"每{rebalance_freq}天",
        "持仓行业数": top_n,
        "回测天数": len(daily_values),
        "净值曲线": daily_values[-100:] if len(daily_values) > 100 else daily_values
    }


def _calc_max_drawdown(values):
    """计算最大回撤"""
    peak = values[0]
    max_dd = 0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return max_dd


def main():
    parser = argparse.ArgumentParser(description='行业轮动策略')
    parser.add_argument('action', choices=['analyze', 'backtest'],
                        help='操作: analyze(行业轮动分析), backtest(策略回测)')
    parser.add_argument('--industries', default=None, help='行业代码列表,逗号分隔')
    parser.add_argument('--days', type=int, default=250, help='数据天数')
    parser.add_argument('--top', type=int, default=5, help='推荐前N个行业')
    parser.add_argument('--rebalance', type=int, default=20, help='调仓频率(天)')
    parser.add_argument('--capital', type=float, default=100000, help='初始资金')

    args = parser.parse_args()
    industries = [s.strip() for s in args.industries.split(",")] if args.industries else None

    try:
        if args.action == 'analyze':
            data = analyze_industry_rotation(industries=industries, days=args.days, top_n=args.top)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'backtest':
            data = backtest_rotation_strategy(
                industries=industries, days=args.days, top_n=args.top,
                rebalance_freq=args.rebalance, initial_capital=args.capital
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
