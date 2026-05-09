#!/usr/bin/env python3
"""
跨市场联动分析系统
支持AH溢价分析、北向资金流向、全球市场联动、汇率敏感性分析
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


def get_index_kline(index_code, days=500):
    """获取指数K线数据"""
    try:
        df = ak.stock_zh_index_daily_em(symbol=f"sh{index_code}" if index_code.startswith('0') else f"sz{index_code}")
        if df is not None and len(df) > 0:
            return df.tail(days)
    except Exception:
        pass
    try:
        df = ak.index_zh_a_hist(symbol=index_code, period="daily", start_date='20100101', end_date=datetime.now().strftime('%Y%m%d'))
        if df is not None and len(df) > 0:
            return df.tail(days)
    except Exception:
        pass
    return None


# ==================== AH溢价分析 ====================

def ah_premium_analysis():
    """
    AH股溢价分析
    分析A股相对H股的溢价水平

    返回: AH溢价分析结果
    """
    try:
        df = ak.stock_zh_ah_spot()
    except Exception as e:
        return {"error": f"获取AH股数据失败: {str(e)}"}

    if df is None or len(df) == 0:
        return {"error": "未获取到AH股数据"}

    # 识别列
    code_col = None
    name_col = None
    a_price_col = None
    h_price_col = None
    premium_col = None

    for col in df.columns:
        col_lower = str(col).lower()
        if '代码' in col or 'code' in col_lower:
            code_col = col
        elif '名称' in col or 'name' in col_lower:
            name_col = col
        elif 'a股' in col_lower or ('a' in col_lower and '价' in col):
            a_price_col = col
        elif 'h股' in col_lower or ('h' in col_lower and '价' in col):
            h_price_col = col
        elif '溢价' in col or 'premium' in col_lower:
            premium_col = col

    # 提取数据
    ah_list = []
    premiums = []

    for i in range(min(50, len(df))):
        row = df.iloc[i]
        try:
            code = str(row[code_col]) if code_col else ""
            name = str(row[name_col]) if name_col else ""
            a_price = float(row[a_price_col]) if a_price_col and pd.notna(row[a_price_col]) else None
            h_price = float(row[h_price_col]) if h_price_col and pd.notna(row[h_price_col]) else None
            premium = float(row[premium_col]) if premium_col and pd.notna(row[premium_col]) else None

            if premium is None and a_price and h_price and h_price > 0:
                premium = (a_price / h_price - 1) * 100

            if premium is not None:
                premiums.append(premium)

            ah_list.append({
                "代码": code,
                "名称": name,
                "A股价格": round(a_price, 2) if a_price else "N/A",
                "H股价格": round(h_price, 2) if h_price else "N/A",
                "AH溢价率": f"{premium:.1f}%" if premium is not None else "N/A",
            })
        except Exception:
            continue

    # 统计
    if premiums:
        avg_premium = float(np.mean(premiums))
        max_premium = float(np.max(premiums))
        min_premium = float(np.min(premiums))
        median_premium = float(np.median(premiums))

        # 溢价最高的
        high_premium = sorted(ah_list, key=lambda x: float(x["AH溢价率"].replace('%', '')) if x["AH溢价率"] != "N/A" else 0, reverse=True)[:5]
        # 溢价最低的
        low_premium = sorted(ah_list, key=lambda x: float(x["AH溢价率"].replace('%', '')) if x["AH溢价率"] != "N/A" else 999)[:5]
    else:
        avg_premium = max_premium = min_premium = median_premium = 0
        high_premium = low_premium = []

    # 解读
    interpretation = []
    if avg_premium > 50:
        interpretation.append(f"A股平均溢价{avg_premium:.1f}%，A股相对H股显著高估")
        interpretation.append("高溢价A股存在估值回归风险，H股相对更具性价比")
    elif avg_premium > 30:
        interpretation.append(f"A股平均溢价{avg_premium:.1f}%，处于历史中等偏高水平")
    else:
        interpretation.append(f"A股平均溢价{avg_premium:.1f}%，溢价水平合理")

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "AH股数量": len(ah_list),
        "溢价统计": {
            "平均溢价": f"{avg_premium:.1f}%",
            "中位数溢价": f"{median_premium:.1f}%",
            "最高溢价": f"{max_premium:.1f}%",
            "最低溢价": f"{min_premium:.1f}%",
        },
        "溢价最高": high_premium,
        "溢价最低（折价）": low_premium,
        "解读": interpretation,
        "投资建议": [
            "高AH溢价的A股需警惕估值回归风险",
            "低AH溢价的A股相对安全，H股折价提供安全边际",
            "可通过港股通买入折价H股获取估值修复收益",
        ],
    }


# ==================== 北向资金分析 ====================

def northbound_flow_analysis(days=30):
    """
    北向资金流向分析
    分析沪股通/深股通资金流向与A股走势关系

    参数:
        days: 分析天数

    返回: 北向资金分析
    """
    try:
        df_north = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
    except Exception:
        try:
            df_north = ak.stock_hsgt_hist_em(symbol="北向资金")
        except Exception as e:
            return {"error": f"获取北向资金数据失败: {str(e)}"}

    if df_north is None or len(df_north) == 0:
        return {"error": "未获取到北向资金数据"}

    # 提取最近数据
    df_recent = df_north.tail(days)

    # 识别列
    date_col = None
    flow_col = None

    for col in df_recent.columns:
        col_lower = str(col).lower()
        if '日期' in col or 'date' in col_lower or '时间' in col:
            date_col = col
        elif '净流入' in col or '净买入' in col or 'net' in col_lower or '资金' in col:
            flow_col = col

    if flow_col is None:
        flow_col = df_recent.columns[1] if len(df_recent.columns) > 1 else df_recent.columns[0]

    flows = pd.to_numeric(df_recent[flow_col], errors='coerce').dropna()

    if len(flows) == 0:
        return {"error": "无法解析资金流向数据"}

    total_flow = float(flows.sum())
    avg_flow = float(flows.mean())
    recent_5d = float(flows.tail(5).sum())
    recent_10d = float(flows.tail(10).sum())

    # 流入流出天数
    inflow_days = int((flows > 0).sum())
    outflow_days = int((flows < 0).sum())

    # 趋势判断
    if recent_5d > 0 and recent_10d > 0:
        trend = "北向资金持续净流入，外资看好A股"
        signal = "偏多"
    elif recent_5d < 0 and recent_10d < 0:
        trend = "北向资金持续净流出，外资撤离A股"
        signal = "偏空"
    elif recent_5d > 0:
        trend = "短期北向资金回流，关注持续性"
        signal = "中性偏多"
    else:
        trend = "北向资金流向分化，方向不明确"
        signal = "中性"

    # 每日流向
    daily_flows = []
    for i in range(min(days, len(df_recent))):
        idx = len(df_recent) - days + i if i < len(df_recent) else i
        if idx < len(df_recent):
            try:
                date_val = str(df_recent.iloc[idx][date_col])[:10] if date_col else ""
                flow_val = float(df_recent.iloc[idx][flow_col]) if pd.notna(df_recent.iloc[idx][flow_col]) else 0
                daily_flows.append({
                    "日期": date_val,
                    "净流入": f"{flow_val / 1e8:+.2f}亿",
                })
            except Exception:
                continue

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "分析周期": f"近{days}日",
        "资金流向": {
            "累计净流入": f"{total_flow / 1e8:+.2f}亿",
            "日均净流入": f"{avg_flow / 1e8:+.2f}亿",
            "近5日": f"{recent_5d / 1e8:+.2f}亿",
            "近10日": f"{recent_10d / 1e8:+.2f}亿",
            "净流入天数": inflow_days,
            "净流出天数": outflow_days,
        },
        "趋势判断": trend,
        "信号": signal,
        "每日流向": daily_flows[-10:],
        "参考": "北向资金被称为'聪明钱'，其持续流入/流出对A股有重要参考意义",
    }


# ==================== 全球市场联动 ====================

def global_market_linkage():
    """
    全球主要市场联动分析
    分析A股与美股、港股、欧股、日股的相关性

    返回: 全球市场联动分析
    """
    indices = {
        "上证指数": "000001",
        "深证成指": "399001",
        "沪深300": "000300",
    }

    # 获取A股指数数据
    a_share_data = {}
    for name, code in indices.items():
        df = get_index_kline(code, days=250)
        if df is not None and len(df) >= 60:
            close_col = '收盘' if '收盘' in df.columns else 'close'
            a_share_data[name] = df[close_col].pct_change().dropna()
        time.sleep(0.3)

    if not a_share_data:
        return {"error": "无法获取A股指数数据"}

    # 获取海外指数（通过akshare）
    global_indices = {
        "恒生指数": "HSI",
        "标普500": "SPX",
        "纳斯达克": "IXIC",
        "日经225": "N225",
    }

    global_data = {}
    for name, code in global_indices.items():
        try:
            df = ak.index_global_hist_em(symbol=code)
            if df is not None and len(df) >= 60:
                close_col = '收盘' if '收盘' in df.columns else 'close'
                global_data[name] = df[close_col].pct_change().dropna().tail(250)
        except Exception:
            pass
        time.sleep(0.3)

    # 计算相关性
    correlations = {}
    if a_share_data and global_data:
        a_returns = list(a_share_data.values())[0]
        for g_name, g_returns in global_data.items():
            min_len = min(len(a_returns), len(g_returns))
            if min_len >= 30:
                corr = round(float(np.corrcoef(a_returns[-min_len:], g_returns[-min_len:])[0, 1]), 4)
                correlations[f"A股 vs {g_name}"] = corr

    # 联动解读
    interpretation = []
    for pair, corr in correlations.items():
        if abs(corr) > 0.5:
            interpretation.append(f"{pair}高度相关({corr:.2f})，需关注海外市场波动传导")
        elif abs(corr) > 0.3:
            interpretation.append(f"{pair}中度相关({corr:.2f})，存在一定联动效应")
        else:
            interpretation.append(f"{pair}相关性较弱({corr:.2f})，独立性较强")

    # 隔夜影响分析
    overnight_impact = []
    if "恒生指数" in global_data:
        overnight_impact.append("港股开盘早于A股，恒生指数走势对A股开盘有指引作用")
    if "标普500" in global_data:
        overnight_impact.append("美股前一交易日走势影响A股次日开盘情绪")

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "相关性矩阵": correlations,
        "联动解读": interpretation,
        "隔夜影响": overnight_impact,
        "参考": [
            "A股与港股相关性通常最高（0.5-0.7），因两地经济联系紧密",
            "A股与美股相关性中等（0.2-0.4），主要通过情绪传导",
            "全球风险偏好变化时，各市场相关性会显著上升",
        ],
    }


# ==================== 汇率敏感性分析 ====================

def fx_sensitivity_analysis():
    """
    汇率敏感性分析
    分析人民币汇率与A股的关系，识别汇率敏感行业

    返回: 汇率分析结果
    """
    # 获取汇率数据
    try:
        df_fx = ak.currency_boc_sina(symbol="美元人民币")
        if df_fx is None or len(df_fx) == 0:
            df_fx = ak.fx_spot_quote()
    except Exception:
        try:
            df_fx = ak.fx_spot_quote()
        except Exception as e:
            return {"error": f"获取汇率数据失败: {str(e)}"}

    result = {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "汇率数据": {
            "数据来源": "akshare",
            "数据条数": len(df_fx) if df_fx is not None else 0,
        },
    }

    # 汇率影响分析
    result["汇率影响分析"] = {
        "人民币升值": {
            "利好行业": ["航空（美元债务减少）", "造纸（进口原料成本降低）", "旅游（出境游增加）"],
            "利空行业": ["出口型制造业", "纺织服装（出口竞争力下降）"],
            "对A股影响": "人民币升值吸引外资流入，利好A股整体估值",
        },
        "人民币贬值": {
            "利好行业": ["出口型企业", "纺织服装", "家电出口"],
            "利空行业": ["航空", "房地产（外资流出）", "进口依赖型企业"],
            "对A股影响": "人民币贬值可能导致外资流出，短期利空A股",
        },
    }

    # 汇率敏感板块
    result["汇率敏感板块"] = [
        {"板块": "航空", "敏感性": "高", "逻辑": "大量美元债务，人民币升值直接减少负债"},
        {"板块": "造纸", "敏感性": "高", "逻辑": "进口纸浆占比高，升值降低原料成本"},
        {"板块": "石油化工", "敏感性": "高", "逻辑": "原油进口依赖度高"},
        {"板块": "有色金属", "敏感性": "中", "逻辑": "国际定价商品，汇率影响进口成本"},
        {"板块": "银行", "敏感性": "中", "逻辑": "外资持股比例高，汇率影响外资流向"},
        {"板块": "房地产", "敏感性": "中", "逻辑": "汇率贬值可能引发外资流出"},
        {"板块": "出口制造", "敏感性": "高", "逻辑": "贬值利好出口，升值利空"},
    ]

    return result


# ==================== 跨境资金流向 ====================

def cross_border_capital_flow():
    """
    跨境资金流向综合分析
    整合北向资金、QFII、RQFII等外资流向

    返回: 跨境资金分析
    """
    result = {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # 北向资金
    try:
        df_north = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
        if df_north is not None and len(df_north) > 0:
            result["北向资金"] = {
                "数据状态": "获取成功",
                "数据条数": len(df_north),
                "列名": list(df_north.columns),
            }
    except Exception as e:
        result["北向资金"] = {"error": str(e)}

    # 南向资金
    try:
        df_south = ak.stock_hsgt_south_net_flow_in_em(symbol="南下")
        if df_south is not None and len(df_south) > 0:
            result["南向资金"] = {
                "数据状态": "获取成功",
                "数据条数": len(df_south),
                "列名": list(df_south.columns),
            }
    except Exception as e:
        result["南向资金"] = {"error": str(e)}

    # 解读
    result["资金流向解读"] = [
        "北向资金（外资买A股）：持续净流入表明外资看好中国资产",
        "南向资金（内资买港股）：持续净流入表明内资寻求港股低估值机会",
        "南北向资金同时流入：全球资金看好中国资产整体表现",
        "北向流出+南向流入：外资撤离A股但内资抄底港股",
    ]

    return result


def main():
    parser = argparse.ArgumentParser(description='跨市场联动分析系统')
    subparsers = parser.add_subparsers(dest='command')

    # AH溢价
    ah_parser = subparsers.add_parser('ah-premium', help='AH股溢价分析')

    # 北向资金
    north_parser = subparsers.add_parser('north-flow', help='北向资金流向分析')
    north_parser.add_argument('--days', type=int, default=30, help='分析天数')

    # 全球联动
    global_parser = subparsers.add_parser('global-link', help='全球市场联动分析')

    # 汇率分析
    fx_parser = subparsers.add_parser('fx-sensitivity', help='汇率敏感性分析')

    # 跨境资金
    cross_parser = subparsers.add_parser('cross-border', help='跨境资金流向分析')

    args = parser.parse_args()

    if args.command == 'ah-premium':
        result = ah_premium_analysis()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'north-flow':
        result = northbound_flow_analysis(days=args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'global-link':
        result = global_market_linkage()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'fx-sensitivity':
        result = fx_sensitivity_analysis()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'cross-border':
        result = cross_border_capital_flow()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
