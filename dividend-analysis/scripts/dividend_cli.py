#!/usr/bin/env python3
"""
分红除权除息分析系统
支持分红历史查询、除权除息价格调整、分红再投资模拟、股息率分析
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

from data_utils import get_stock_kline


# ==================== 分红历史查询 ====================

def get_dividend_history(symbol):
    """
    获取股票分红历史

    参数:
        symbol: 股票代码

    返回: 分红历史数据
    """
    try:
        df = ak.stock_dividents_cninfo(symbol=symbol)
    except Exception:
        try:
            df = ak.stock_history_dividend_detail(symbol=symbol, indicator="分红")
        except Exception as e:
            return {"error": f"获取分红数据失败: {str(e)}"}

    if df is None or len(df) == 0:
        return {"error": f"未找到{symbol}的分红记录"}

    # 识别列
    date_col = plan_col = ex_date_col = amount_col = None

    for col in df.columns:
        col_lower = str(col).lower()
        if ('除权' in col or '除息' in col or 'ex' in col_lower) and ex_date_col is None:
            ex_date_col = col
        elif ('方案' in col or '预案' in col or 'plan' in col_lower) and plan_col is None:
            plan_col = col
        elif ('公告' in col or '日期' in col or 'date' in col_lower) and date_col is None:
            date_col = col
        elif ('派息' in col or '分红' in col or 'dividend' in col_lower or '金额' in col) and amount_col is None:
            amount_col = col

    dividends = []
    total_cash = 0
    total_shares_bonus = 0

    for i in range(min(20, len(df))):
        try:
            row = df.iloc[i]
            plan_text = str(row[plan_col]) if plan_col and plan_col in df.columns else ""
            ex_date = str(row[ex_date_col]) if ex_date_col and ex_date_col in df.columns else ""
            announce_date = str(row[date_col]) if date_col and date_col in df.columns else ""

            # 解析分红方案
            cash_per_10 = 0
            shares_per_10 = 0

            if plan_text:
                # 解析"10派X元"格式
                if '派' in plan_text:
                    try:
                        parts = plan_text.split('派')
                        cash_part = parts[1].split('元')[0] if '元' in parts[1] else parts[1].split(' ')[0]
                        cash_per_10 = float(cash_part.replace(' ', ''))
                    except (ValueError, IndexError):
                        pass

                # 解析"10送X股"或"10转X股"
                if '送' in plan_text:
                    try:
                        parts = plan_text.split('送')
                        share_part = parts[1].split('股')[0] if '股' in parts[1] else parts[1].split(' ')[0]
                        shares_per_10 += float(share_part.replace(' ', ''))
                    except (ValueError, IndexError):
                        pass

                if '转' in plan_text:
                    try:
                        parts = plan_text.split('转')
                        share_part = parts[1].split('股')[0] if '股' in parts[1] else parts[1].split(' ')[0]
                        shares_per_10 += float(share_part.replace(' ', ''))
                    except (ValueError, IndexError):
                        pass

            total_cash += cash_per_10
            total_shares_bonus += shares_per_10

            dividends.append({
                "公告日期": announce_date[:10] if announce_date else "N/A",
                "除权除息日": ex_date[:10] if ex_date else "N/A",
                "分红方案": plan_text or "N/A",
                "每10股派息": f"{cash_per_10:.2f}元" if cash_per_10 > 0 else "无",
                "每10股送转": f"{shares_per_10:.1f}股" if shares_per_10 > 0 else "无",
            })
        except Exception:
            continue

    return {
        "股票代码": symbol,
        "查询时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "分红记录数": len(dividends),
        "累计每10股派息": f"{total_cash:.2f}元",
        "累计每10股送转": f"{total_shares_bonus:.1f}股",
        "分红历史": dividends,
    }


# ==================== 除权除息价格调整 ====================

def adjust_price_for_dividend(symbol, days=500):
    """
    获取复权价格数据（前复权/后复权）

    参数:
        symbol: 股票代码
        days: 数据天数

    返回: 复权价格对比
    """
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'),
                                end_date=datetime.now().strftime('%Y%m%d'),
                                adjust="")
    except Exception as e:
        return {"error": f"获取K线数据失败: {str(e)}"}

    if df is None or len(df) < 10:
        return {"error": f"{symbol}数据不足"}

    # 不复权价格
    close_col = '收盘' if '收盘' in df.columns else 'close'
    close = pd.to_numeric(df[close_col], errors='coerce').dropna()

    # 前复权价格
    try:
        df_qfq = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                    start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'),
                                    end_date=datetime.now().strftime('%Y%m%d'),
                                    adjust="qfq")
        close_qfq = pd.to_numeric(df_qfq['收盘' if '收盘' in df_qfq.columns else 'close'], errors='coerce').dropna()
    except Exception:
        close_qfq = None

    # 后复权价格
    try:
        df_hfq = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                    start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'),
                                    end_date=datetime.now().strftime('%Y%m%d'),
                                    adjust="hfq")
        close_hfq = pd.to_numeric(df_hfq['收盘' if '收盘' in df_hfq.columns else 'close'], errors='coerce').dropna()
    except Exception:
        close_hfq = None

    result = {
        "股票代码": symbol,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "不复权": {
            "最新价": round(float(close.iloc[-1]), 2) if len(close) > 0 else "N/A",
            "数据条数": len(close),
        },
    }

    if close_qfq is not None and len(close_qfq) > 0:
        result["前复权"] = {
            "最新价": round(float(close_qfq.iloc[-1]), 2),
            "数据条数": len(close_qfq),
            "说明": "前复权保持最新价不变，调整历史价格，适合技术分析",
        }

    if close_hfq is not None and len(close_hfq) > 0:
        result["后复权"] = {
            "最新价": round(float(close_hfq.iloc[-1]), 2),
            "数据条数": len(close_hfq),
            "说明": "后复权保持历史价不变，调整最新价，适合计算真实收益",
        }

    # 除权除息影响分析
    if close_qfq is not None and len(close) > 0 and len(close_qfq) > 0:
        min_len = min(len(close), len(close_qfq))
        ratio = close.iloc[-min_len:].values / close_qfq.iloc[-min_len:].values
        # 复权因子
        adj_factor = float(ratio[-1]) if len(ratio) > 0 else 1.0
        result["复权因子"] = round(adj_factor, 4)
        result["累计分红影响"] = f"{(adj_factor - 1) * 100:.2f}%"
        result["说明"] = "复权因子>1表示历史上有分红送转，前复权价格低于不复权价格"

    return result


# ==================== 分红再投资模拟 ====================

def dividend_reinvestment_simulation(symbol, initial_investment=100000, years=5):
    """
    分红再投资模拟
    对比分红再投资 vs 不参与分红再投资的收益差异

    参数:
        symbol: 股票代码
        initial_investment: 初始投资金额
        years: 模拟年数

    返回: 分红再投资对比
    """
    days = years * 250

    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'),
                                end_date=datetime.now().strftime('%Y%m%d'),
                                adjust="hfq")
    except Exception as e:
        return {"error": f"获取K线数据失败: {str(e)}"}

    if df is None or len(df) < 60:
        return {"error": f"{symbol}数据不足"}

    close_col = '收盘' if '收盘' in df.columns else 'close'
    close = pd.to_numeric(df[close_col], errors='coerce').dropna()

    if len(close) < 60:
        return {"error": "有效数据不足"}

    # 获取分红数据
    try:
        div_df = ak.stock_dividents_cninfo(symbol=symbol)
    except Exception:
        div_df = None

    # 后复权价格模拟（已包含分红再投资效果）
    start_price = float(close.iloc[0])
    end_price = float(close.iloc[-1])

    shares_bought = initial_investment / start_price
    final_value = shares_bought * end_price
    total_return = (final_value / initial_investment - 1) * 100

    # 不复权价格（不含分红再投资）
    try:
        df_no_adj = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                       start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'),
                                       end_date=datetime.now().strftime('%Y%m%d'),
                                       adjust="")
        close_no_adj = pd.to_numeric(df_no_adj['收盘' if '收盘' in df_no_adj.columns else 'close'], errors='coerce').dropna()

        if len(close_no_adj) >= 60:
            start_price_no = float(close_no_adj.iloc[0])
            end_price_no = float(close_no_adj.iloc[-1])
            shares_no = initial_investment / start_price_no
            final_value_no = shares_no * end_price_no
            total_return_no = (final_value_no / initial_investment - 1) * 100

            dividend_contribution = total_return - total_return_no
        else:
            total_return_no = None
            dividend_contribution = None
    except Exception:
        total_return_no = None
        dividend_contribution = None

    result = {
        "股票代码": symbol,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "模拟参数": {
            "初始投资": f"{initial_investment:,}元",
            "模拟周期": f"{years}年",
            "起始日期": str(df.index[0])[:10] if hasattr(df.index[0], 'strftime') else str(df.index[0]),
            "结束日期": str(df.index[-1])[:10] if hasattr(df.index[-1], 'strftime') else str(df.index[-1]),
        },
        "分红再投资（后复权）": {
            "最终价值": f"{final_value:,.2f}元",
            "总收益率": f"{total_return:.2f}%",
            "年化收益率": f"{((final_value / initial_investment) ** (1 / years) - 1) * 100:.2f}%",
        },
    }

    if total_return_no is not None:
        result["不参与分红再投资（不复权）"] = {
            "最终价值": f"{final_value_no:,.2f}元",
            "总收益率": f"{total_return_no:.2f}%",
        }
        result["分红贡献"] = {
            "额外收益": f"{dividend_contribution:.2f}%",
            "说明": "分红再投资带来的额外收益，体现了复利效应",
        }

    result["建议"] = [
        "长期持有高股息股票，分红再投资可显著提升总收益",
        "后复权价格已包含分红再投资效果，回测时应使用后复权数据",
        "前复权适合技术分析，后复权适合收益计算",
    ]

    return result


# ==================== 股息率分析 ====================

def dividend_yield_analysis(symbol):
    """
    股息率分析

    参数:
        symbol: 股票代码

    返回: 股息率分析
    """
    # 获取分红历史
    div_result = get_dividend_history(symbol)
    if "error" in div_result:
        return div_result

    # 获取当前价格
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
                                end_date=datetime.now().strftime('%Y%m%d'),
                                adjust="")
        close_col = '收盘' if '收盘' in df.columns else 'close'
        current_price = float(pd.to_numeric(df[close_col], errors='coerce').dropna().iloc[-1])
    except Exception:
        current_price = None

    # 计算股息率
    dividends = div_result.get("分红历史", [])
    recent_dividends = []

    for d in dividends:
        try:
            cash_text = d.get("每10股派息", "无")
            if "元" in cash_text:
                cash_per_10 = float(cash_text.replace("元", ""))
                if current_price and current_price > 0:
                    yield_rate = (cash_per_10 / 10) / current_price * 100
                else:
                    yield_rate = None
                recent_dividends.append({
                    "除权除息日": d["除权除息日"],
                    "每10股派息": cash_text,
                    "每股派息": f"{cash_per_10 / 10:.4f}元",
                    "股息率": f"{yield_rate:.2f}%" if yield_rate is not None else "N/A",
                })
        except (ValueError, AttributeError):
            continue

    # 平均股息率
    valid_yields = []
    for d in recent_dividends:
        try:
            y = float(d["股息率"].replace("%", ""))
            valid_yields.append(y)
        except (ValueError, AttributeError):
            continue

    avg_yield = float(np.mean(valid_yields)) if valid_yields else 0

    # 股息率评级
    if avg_yield > 5:
        rating = "高股息（股息率>5%），适合红利策略"
    elif avg_yield > 3:
        rating = "中等股息（3%-5%），股息回报尚可"
    elif avg_yield > 1:
        rating = "低股息（1%-3%），股息回报一般"
    else:
        rating = "极低股息（<1%），不适合红利策略"

    return {
        "股票代码": symbol,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "当前价格": round(current_price, 2) if current_price else "N/A",
        "近5年分红": recent_dividends[:5],
        "平均股息率": f"{avg_yield:.2f}%",
        "股息率评级": rating,
        "红利策略建议": [
            "股息率>4%且稳定的股票适合长期持有",
            "关注分红连续性和增长性，而非单次高分红",
            "银行、公用事业、消费等行业股息率通常较高",
        ],
    }


def advanced_dividend_reinvestment_simulation(symbol, initial_investment=100000, years=5):
    """
    高级分红再投资模拟
    基于实际分红记录，精确模拟每次分红再买入的复利效应

    参数:
        symbol: 股票代码
        initial_investment: 初始投资金额
        years: 模拟年数

    返回: 详细的分红再投资对比报告
    """
    days = years * 250

    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'),
                                end_date=datetime.now().strftime('%Y%m%d'),
                                adjust="qfq")
    except Exception as e:
        return {"error": f"获取K线数据失败: {str(e)}"}

    if df is None or len(df) < 60:
        return {"error": f"{symbol}数据不足"}

    close_col = '收盘' if '收盘' in df.columns else 'close'
    close = pd.to_numeric(df[close_col], errors='coerce').dropna()
    df = df.set_index(pd.to_datetime(df['日期' if '日期' in df.columns else 'date']))

    if len(close) < 60:
        return {"error": "有效数据不足"}

    start_price = float(close.iloc[0])
    end_price = float(close.iloc[-1])

    shares_no_reinvest = initial_investment / start_price
    final_value_no_reinvest = shares_no_reinvest * end_price
    total_return_no = (final_value_no_reinvest / initial_investment - 1) * 100

    try:
        div_df = ak.stock_dividents_cninfo(symbol=symbol)
    except Exception:
        div_df = None

    dividend_events = []
    if div_df is not None and len(div_df) > 0:
        for i in range(min(20, len(div_df))):
            try:
                row = div_df.iloc[i]
                plan_col = None
                ex_date_col = None
                for col in div_df.columns:
                    col_lower = str(col).lower()
                    if ('除权' in col or '除息' in col or 'ex' in col_lower) and ex_date_col is None:
                        ex_date_col = col
                    elif ('方案' in col or '预案' in col or 'plan' in col_lower) and plan_col is None:
                        plan_col = col

                plan_text = str(row[plan_col]) if plan_col else ""
                ex_date_str = str(row[ex_date_col]) if ex_date_col else ""

                cash_per_10 = 0
                if plan_text and '派' in plan_text:
                    try:
                        parts = plan_text.split('派')
                        cash_part = parts[1].split('元')[0] if '元' in parts[1] else parts[1].split(' ')[0]
                        cash_per_10 = float(cash_part.replace(' ', ''))
                    except (ValueError, IndexError):
                        pass

                if cash_per_10 > 0 and ex_date_str:
                    try:
                        ex_date = pd.to_datetime(ex_date_str[:10])
                        if ex_date >= df.index[0] and ex_date <= df.index[-1]:
                            dividend_events.append({
                                "日期": ex_date,
                                "每10股派息": cash_per_10,
                                "每股派息": cash_per_10 / 10,
                            })
                    except Exception:
                        continue
            except Exception:
                continue

    dividend_events.sort(key=lambda x: x["日期"])

    shares_with_reinvest = initial_investment / start_price
    cash_dividends_received = 0
    reinvestment_log = []
    yearly_summary = {}

    for event in dividend_events:
        ex_date = event["日期"]
        div_per_share = event["每股派息"]

        cash_received = shares_with_reinvest * div_per_share
        cash_dividends_received += cash_received

        closest_idx = df.index[df.index <= ex_date]
        if len(closest_idx) == 0:
            continue
        buy_date_idx = closest_idx[-1]
        buy_price = float(close.loc[buy_date_idx])

        additional_shares = cash_received / buy_price
        shares_before = shares_with_reinvest
        shares_with_reinvest += additional_shares

        year_key = str(ex_date.year)
        if year_key not in yearly_summary:
            yearly_summary[year_key] = {"分红次数": 0, "分红总额": 0, "新增股数": 0}
        yearly_summary[year_key]["分红次数"] += 1
        yearly_summary[year_key]["分红总额"] += cash_received
        yearly_summary[year_key]["新增股数"] += additional_shares

        reinvestment_log.append({
            "除权日期": ex_date.strftime('%Y-%m-%d'),
            "每股派息": round(div_per_share, 4),
            "分红前持股": round(float(shares_before), 0),
            "收到分红": round(float(cash_received), 2),
            "再买入价": round(float(buy_price), 2),
            "新增股数": round(float(additional_shares), 0),
            "分红后持股": round(float(shares_with_reinvest), 0),
        })

    final_value_with_reinvest = shares_with_reinvest * end_price
    total_return_with = (final_value_with_reinvest / initial_investment - 1) * 100

    reinvestment_benefit = total_return_with - total_return_no
    extra_shares = shares_with_reinvest - shares_no_reinvest

    years_actual = len(close) / 252
    annual_return_no = ((final_value_no_reinvest / initial_investment) ** (1 / years_actual) - 1) * 100 if years_actual > 0 else 0
    annual_return_with = ((final_value_with_reinvest / initial_investment) ** (1 / years_actual) - 1) * 100 if years_actual > 0 else 0

    yearly_detail = []
    for year_key in sorted(yearly_summary.keys()):
        y = yearly_summary[year_key]
        yearly_detail.append({
            "年份": year_key,
            "分红次数": y["分红次数"],
            "分红总额": round(y["分红总额"], 2),
            "新增股数": round(y["新增股数"], 0),
        })

    return {
        "股票代码": symbol,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "模拟参数": {
            "初始投资": f"{initial_investment:,}元",
            "模拟周期": f"{years}年",
            "起始日期": df.index[0].strftime('%Y-%m-%d'),
            "结束日期": df.index[-1].strftime('%Y-%m-%d'),
            "起始价格": round(start_price, 2),
            "结束价格": round(end_price, 2),
        },
        "不参与分红再投资": {
            "最终持股": round(float(shares_no_reinvest), 0),
            "最终价值": round(float(final_value_no_reinvest), 2),
            "总收益率": round(total_return_no, 2),
            "年化收益率": round(annual_return_no, 2),
        },
        "分红再投资": {
            "最终持股": round(float(shares_with_reinvest), 0),
            "最终价值": round(float(final_value_with_reinvest), 2),
            "总收益率": round(total_return_with, 2),
            "年化收益率": round(annual_return_with, 2),
            "累计收到分红": round(float(cash_dividends_received), 2),
            "分红新增股数": round(float(extra_shares), 0),
        },
        "再投资增益": {
            "额外收益": round(reinvestment_benefit, 2),
            "额外股数": round(float(extra_shares), 0),
            "年化增益": round(annual_return_with - annual_return_no, 2),
            "说明": "分红再投资通过复利效应，长期可显著提升总收益",
        },
        "年度分红明细": yearly_detail,
        "分红再投资日志": reinvestment_log,
        "建议": [
            f"分红再投资带来额外{reinvestment_benefit:.1f}%收益，体现了复利的力量",
            "长期持有高股息股票并坚持分红再投资，是稳健增值的有效策略",
            "选择分红稳定、股息率高的蓝筹股进行分红再投资效果更佳",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description='分红除权除息分析系统')
    subparsers = parser.add_subparsers(dest='command')

    # 分红历史
    history_parser = subparsers.add_parser('history', help='分红历史查询')
    history_parser.add_argument('--symbol', required=True, help='股票代码')

    # 复权价格
    adjust_parser = subparsers.add_parser('adjust', help='除权除息价格调整分析')
    adjust_parser.add_argument('--symbol', required=True, help='股票代码')
    adjust_parser.add_argument('--days', type=int, default=500, help='数据天数')

    # 分红再投资模拟
    reinvest_parser = subparsers.add_parser('reinvest', help='分红再投资模拟')
    reinvest_parser.add_argument('--symbol', required=True, help='股票代码')
    reinvest_parser.add_argument('--capital', type=float, default=100000, help='初始投资金额')
    reinvest_parser.add_argument('--years', type=int, default=5, help='模拟年数')

    # 股息率分析
    yield_parser = subparsers.add_parser('yield', help='股息率分析')
    yield_parser.add_argument('--symbol', required=True, help='股票代码')

    # 高级分红再投资模拟
    advanced_parser = subparsers.add_parser('advanced', help='高级分红再投资模拟（基于实际分红记录）')
    advanced_parser.add_argument('--symbol', required=True, help='股票代码')
    advanced_parser.add_argument('--capital', type=float, default=100000, help='初始投资金额')
    advanced_parser.add_argument('--years', type=int, default=5, help='模拟年数')

    args = parser.parse_args()

    if args.command == 'history':
        result = get_dividend_history(args.symbol)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'adjust':
        result = adjust_price_for_dividend(args.symbol, args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'reinvest':
        result = dividend_reinvestment_simulation(args.symbol, args.capital, args.years)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'yield':
        result = dividend_yield_analysis(args.symbol)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'advanced':
        result = advanced_dividend_reinvestment_simulation(args.symbol, args.capital, args.years)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
