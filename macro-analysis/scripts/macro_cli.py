#!/usr/bin/env python3
"""
宏观经济指标分析系统
支持GDP/CPI/PMI/M2/社融/利率等宏观指标获取，与股市关联分析，宏观周期判断
"""
import argparse
import json
import sys
import os
import time
from datetime import datetime

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


# ==================== 宏观指标获取 ====================

def get_macro_indicators():
    """
    获取核心宏观经济指标

    返回: 宏观指标数据
    """
    result = {
        "获取时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "指标数据": {},
    }

    # 1. GDP数据
    try:
        gdp_df = ak.macro_china_gdp()
        if gdp_df is not None and len(gdp_df) > 0:
            gdp_recent = gdp_df.tail(8)
            result["指标数据"]["GDP"] = []
            for _, row in gdp_recent.iterrows():
                result["指标数据"]["GDP"].append({
                    "时间": str(row.iloc[0]) if len(row) > 0 else "",
                    "GDP同比": f"{float(row.iloc[1]):.2f}%" if len(row) > 1 and pd.notna(row.iloc[1]) else "N/A",
                    "累计值": str(row.iloc[2]) if len(row) > 2 else "N/A",
                })
    except Exception as e:
        result["指标数据"]["GDP"] = {"error": str(e)}

    # 2. CPI数据
    try:
        cpi_df = ak.macro_china_cpi_monthly()
        if cpi_df is not None and len(cpi_df) > 0:
            cpi_recent = cpi_df.tail(12)
            result["指标数据"]["CPI"] = []
            for _, row in cpi_recent.iterrows():
                result["指标数据"]["CPI"].append({
                    "日期": str(row.iloc[0]) if len(row) > 0 else "",
                    "CPI同比": f"{float(row.iloc[1]):.2f}%" if len(row) > 1 and pd.notna(row.iloc[1]) else "N/A",
                })
    except Exception as e:
        result["指标数据"]["CPI"] = {"error": str(e)}

    # 3. PMI数据
    try:
        pmi_df = ak.macro_china_pmi()
        if pmi_df is not None and len(pmi_df) > 0:
            pmi_recent = pmi_df.tail(12)
            result["指标数据"]["PMI"] = []
            for _, row in pmi_recent.iterrows():
                result["指标数据"]["PMI"].append({
                    "日期": str(row.iloc[0]) if len(row) > 0 else "",
                    "制造业PMI": round(float(row.iloc[1]), 1) if len(row) > 1 and pd.notna(row.iloc[1]) else "N/A",
                })
    except Exception as e:
        result["指标数据"]["PMI"] = {"error": str(e)}

    # 4. 货币供应量(M2)
    try:
        m2_df = ak.macro_china_money_supply()
        if m2_df is not None and len(m2_df) > 0:
            m2_recent = m2_df.tail(12)
            result["指标数据"]["货币供应"] = []
            for _, row in m2_recent.iterrows():
                result["指标数据"]["货币供应"].append({
                    "月份": str(row.iloc[0]) if len(row) > 0 else "",
                    "M2同比": f"{float(row.iloc[1]):.2f}%" if len(row) > 1 and pd.notna(row.iloc[1]) else "N/A",
                })
    except Exception as e:
        result["指标数据"]["货币供应"] = {"error": str(e)}

    # 5. 社会融资规模
    try:
        sf_df = ak.macro_china_shrzgm()
        if sf_df is not None and len(sf_df) > 0:
            sf_recent = sf_df.tail(12)
            result["指标数据"]["社会融资"] = []
            for _, row in sf_recent.iterrows():
                result["指标数据"]["社会融资"].append({
                    "月份": str(row.iloc[0]) if len(row) > 0 else "",
                    "社融增量": f"{float(row.iloc[1]) / 1e4:.2f}万亿" if len(row) > 1 and pd.notna(row.iloc[1]) else "N/A",
                })
    except Exception as e:
        result["指标数据"]["社会融资"] = {"error": str(e)}

    # 6. LPR利率
    try:
        lpr_df = ak.macro_china_lpr()
        if lpr_df is not None and len(lpr_df) > 0:
            lpr_recent = lpr_df.tail(6)
            result["指标数据"]["LPR利率"] = []
            for _, row in lpr_recent.iterrows():
                result["指标数据"]["LPR利率"].append({
                    "日期": str(row.iloc[0]) if len(row) > 0 else "",
                    "1年期LPR": f"{float(row.iloc[1]):.2f}%" if len(row) > 1 and pd.notna(row.iloc[1]) else "N/A",
                    "5年期LPR": f"{float(row.iloc[2]):.2f}%" if len(row) > 2 and pd.notna(row.iloc[2]) else "N/A",
                })
    except Exception as e:
        result["指标数据"]["LPR利率"] = {"error": str(e)}

    return result


# ==================== 宏观与股市关联分析 ====================

def macro_market_correlation(index_code="000300", years=5):
    """
    宏观指标与股市关联分析
    分析PMI、CPI、M2等指标与大盘指数的相关性

    参数:
        index_code: 指数代码
        years: 分析年数

    返回: 关联分析结果
    """
    # 获取指数月度数据
    df_index = get_index_kline(index_code, days=years * 250)
    if df_index is None or len(df_index) < 24:
        return {"error": "指数数据不足"}

    close_col = '收盘' if '收盘' in df_index.columns else 'close'
    close = df_index[close_col]

    # 计算月度收益率
    monthly_returns = close.resample('M').last().pct_change().dropna() if hasattr(close, 'resample') else close

    # 获取PMI数据
    try:
        pmi_df = ak.macro_china_pmi()
        if pmi_df is not None and len(pmi_df) > 0:
            pmi_values = []
            for _, row in pmi_df.tail(len(monthly_returns) + 6).iterrows():
                if len(row) > 1 and pd.notna(row.iloc[1]):
                    pmi_values.append(float(row.iloc[1]))

            if len(pmi_values) >= len(monthly_returns):
                pmi_values = pmi_values[-len(monthly_returns):]
                pmi_corr = round(float(np.corrcoef(monthly_returns.values, pmi_values)[0, 1]), 4)
            else:
                pmi_corr = "数据不足"
        else:
            pmi_corr = "数据不足"
    except Exception:
        pmi_corr = "获取失败"

    # 获取M2数据
    try:
        m2_df = ak.macro_china_money_supply()
        if m2_df is not None and len(m2_df) > 0:
            m2_values = []
            for _, row in m2_df.tail(len(monthly_returns) + 6).iterrows():
                if len(row) > 1 and pd.notna(row.iloc[1]):
                    m2_values.append(float(row.iloc[1]))

            if len(m2_values) >= len(monthly_returns):
                m2_values = m2_values[-len(monthly_returns):]
                m2_corr = round(float(np.corrcoef(monthly_returns.values, m2_values)[0, 1]), 4)
            else:
                m2_corr = "数据不足"
        else:
            m2_corr = "数据不足"
    except Exception:
        m2_corr = "获取失败"

    # 获取CPI数据
    try:
        cpi_df = ak.macro_china_cpi_monthly()
        if cpi_df is not None and len(cpi_df) > 0:
            cpi_values = []
            for _, row in cpi_df.tail(len(monthly_returns) + 6).iterrows():
                if len(row) > 1 and pd.notna(row.iloc[1]):
                    cpi_values.append(float(row.iloc[1]))

            if len(cpi_values) >= len(monthly_returns):
                cpi_values = cpi_values[-len(monthly_returns):]
                cpi_corr = round(float(np.corrcoef(monthly_returns.values, cpi_values)[0, 1]), 4)
            else:
                cpi_corr = "数据不足"
        else:
            cpi_corr = "数据不足"
    except Exception:
        cpi_corr = "获取失败"

    # 关联解读
    interpretation = []
    if isinstance(pmi_corr, float):
        if pmi_corr > 0.3:
            interpretation.append(f"PMI与股市正相关({pmi_corr:.2f})，经济景气度对股市有正向驱动")
        elif pmi_corr < -0.3:
            interpretation.append(f"PMI与股市负相关({pmi_corr:.2f})，存在反直觉关系")
        else:
            interpretation.append(f"PMI与股市相关性较弱({pmi_corr:.2f})")

    if isinstance(m2_corr, float):
        if m2_corr > 0.3:
            interpretation.append(f"M2与股市正相关({m2_corr:.2f})，流动性宽松利好股市")
        elif m2_corr < -0.3:
            interpretation.append(f"M2与股市负相关({m2_corr:.2f})")
        else:
            interpretation.append(f"M2与股市相关性较弱({m2_corr:.2f})")

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "基准指数": index_code,
        "分析周期": f"近{years}年",
        "相关性": {
            "PMI_股市": pmi_corr if isinstance(pmi_corr, float) else pmi_corr,
            "M2_股市": m2_corr if isinstance(m2_corr, float) else m2_corr,
            "CPI_股市": cpi_corr if isinstance(cpi_corr, float) else cpi_corr,
        },
        "解读": interpretation,
    }


# ==================== 宏观周期判断 ====================

def macro_cycle_analysis():
    """
    宏观周期判断（美林时钟模型）
    基于经济增长和通胀两个维度判断当前所处的经济周期阶段

    返回: 周期分析结果
    """
    # 获取最新PMI（代表经济增长）
    try:
        pmi_df = ak.macro_china_pmi()
        latest_pmi = float(pmi_df.iloc[-1, 1]) if pmi_df is not None and len(pmi_df) > 0 else None
        pmi_trend = "上升" if latest_pmi and len(pmi_df) >= 6 and float(pmi_df.iloc[-1, 1]) > float(pmi_df.iloc[-4, 1]) else "下降"
    except Exception:
        latest_pmi = None
        pmi_trend = "未知"

    # 获取最新CPI（代表通胀）
    try:
        cpi_df = ak.macro_china_cpi_monthly()
        latest_cpi = float(cpi_df.iloc[-1, 1]) if cpi_df is not None and len(cpi_df) > 0 else None
        cpi_trend = "上升" if latest_cpi and len(cpi_df) >= 6 and float(cpi_df.iloc[-1, 1]) > float(cpi_df.iloc[-4, 1]) else "下降"
    except Exception:
        latest_cpi = None
        cpi_trend = "未知"

    # 美林时钟判断
    if latest_pmi is not None and latest_cpi is not None:
        if latest_pmi > 50 and latest_cpi < 2:
            cycle = "复苏期"
            cycle_desc = "经济增长加速，通胀温和，是股市最佳投资期"
            asset_allocation = "超配股票 > 债券 > 现金 > 商品"
            sector_focus = "科技、消费、金融等成长性行业"
        elif latest_pmi > 50 and latest_cpi >= 2:
            cycle = "过热期"
            cycle_desc = "经济增长强劲但通胀上升，股市仍有空间但波动加大"
            asset_allocation = "商品 > 股票 > 现金 > 债券"
            sector_focus = "资源、能源、原材料等周期性行业"
        elif latest_pmi <= 50 and latest_cpi >= 2:
            cycle = "滞胀期"
            cycle_desc = "经济放缓但通胀高企，股市承压，防御为主"
            asset_allocation = "现金 > 商品 > 债券 > 股票"
            sector_focus = "公用事业、必需消费等防御性行业"
        else:
            cycle = "衰退期"
            cycle_desc = "经济放缓和通胀下降，央行可能宽松，债券表现最佳"
            asset_allocation = "债券 > 现金 > 股票 > 商品"
            sector_focus = "高股息、医药等防御性行业"
    else:
        cycle = "无法判断"
        cycle_desc = "宏观数据不足"
        asset_allocation = "N/A"
        sector_focus = "N/A"

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "当前指标": {
            "制造业PMI": f"{latest_pmi}" if latest_pmi else "N/A",
            "PMI趋势": pmi_trend,
            "CPI同比": f"{latest_cpi}%" if latest_cpi else "N/A",
            "CPI趋势": cpi_trend,
        },
        "美林时钟": {
            "当前周期": cycle,
            "周期描述": cycle_desc,
            "资产配置建议": asset_allocation,
            "行业关注": sector_focus,
        },
        "参考": "美林时钟基于经济增长(GDP/PMI)和通胀(CPI)两个维度划分经济周期，是经典的宏观配置框架",
    }


# ==================== 政策利率分析 ====================

def interest_rate_analysis():
    """
    利率环境分析
    分析当前利率水平、利率趋势及对股市的影响
    """
    result = {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # LPR利率
    try:
        lpr_df = ak.macro_china_lpr()
        if lpr_df is not None and len(lpr_df) > 0:
            latest = lpr_df.iloc[-1]
            lpr_1y = float(latest.iloc[1]) if len(latest) > 1 and pd.notna(latest.iloc[1]) else None
            lpr_5y = float(latest.iloc[2]) if len(latest) > 2 and pd.notna(latest.iloc[2]) else None

            result["LPR利率"] = {
                "1年期": f"{lpr_1y:.2f}%" if lpr_1y else "N/A",
                "5年期": f"{lpr_5y:.2f}%" if lpr_5y else "N/A",
            }

            # 利率趋势
            if len(lpr_df) >= 4:
                prev_1y = float(lpr_df.iloc[-4, 1]) if pd.notna(lpr_df.iloc[-4, 1]) else lpr_1y
                if lpr_1y and prev_1y:
                    if lpr_1y < prev_1y:
                        result["利率趋势"] = "降息周期，利好股市估值提升"
                    elif lpr_1y > prev_1y:
                        result["利率趋势"] = "加息周期，股市估值承压"
                    else:
                        result["利率趋势"] = "利率平稳，货币政策中性"
    except Exception as e:
        result["LPR利率"] = {"error": str(e)}

    # Shibor
    try:
        shibor_df = ak.rate_interbank(market="上海银行间同业拆放利率", indicator="Shibor")
        if shibor_df is not None and len(shibor_df) > 0:
            result["Shibor"] = {}
            for _, row in shibor_df.tail(1).iterrows():
                for col in shibor_df.columns[1:6]:
                    if pd.notna(row[col]):
                        result["Shibor"][str(col)] = f"{float(row[col]):.4f}%"
    except Exception:
        pass

    return result


def main():
    parser = argparse.ArgumentParser(description='宏观经济指标分析系统')
    subparsers = parser.add_subparsers(dest='command')

    # 获取宏观指标
    indicators_parser = subparsers.add_parser('indicators', help='获取核心宏观指标')

    # 关联分析
    corr_parser = subparsers.add_parser('correlation', help='宏观与股市关联分析')
    corr_parser.add_argument('--index', default='000300', help='指数代码')
    corr_parser.add_argument('--years', type=int, default=5, help='分析年数')

    # 周期判断
    cycle_parser = subparsers.add_parser('cycle', help='宏观周期判断(美林时钟)')

    # 利率分析
    rate_parser = subparsers.add_parser('rates', help='利率环境分析')

    args = parser.parse_args()

    if args.command == 'indicators':
        result = get_macro_indicators()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'correlation':
        result = macro_market_correlation(index_code=args.index, years=args.years)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'cycle':
        result = macro_cycle_analysis()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'rates':
        result = interest_rate_analysis()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
