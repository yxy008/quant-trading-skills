#!/usr/bin/env python3
"""
ETF分析系统
支持ETF列表筛选、折溢价分析、ETF轮动策略、ETF组合构建
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


# ==================== ETF列表获取 ====================

def get_etf_list():
    """
    获取ETF列表及基本信息

    返回: ETF列表数据
    """
    try:
        df = ak.fund_etf_spot_em()
    except Exception:
        try:
            df = ak.fund_etf_category_sina(symbol="ETF基金")
        except Exception as e:
            return {"error": f"获取ETF数据失败: {str(e)}"}

    if df is None or len(df) == 0:
        return {"error": "未获取到ETF数据"}

    # 识别列
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if '代码' in col or 'code' in col_lower:
            col_map['代码'] = col
        elif '名称' in col or 'name' in col_lower:
            col_map['名称'] = col
        elif '最新' in col or '现价' in col or 'price' in col_lower:
            col_map['最新价'] = col
        elif '涨跌' in col or 'change' in col_lower:
            col_map['涨跌幅'] = col
        elif '成交' in col and ('量' in col or 'volume' in col_lower):
            col_map['成交量'] = col
        elif '成交' in col and ('额' in col or 'amount' in col_lower):
            col_map['成交额'] = col
        elif '净值' in col or 'nav' in col_lower:
            col_map['净值'] = col
        elif '折价' in col or '溢价' in col or 'discount' in col_lower:
            col_map['折溢价'] = col
        elif '规模' in col or 'size' in col_lower:
            col_map['规模'] = col

    # 分类统计
    categories = {}
    for i in range(min(200, len(df))):
        try:
            name = str(df.iloc[i][col_map.get('名称', df.columns[1])]) if '名称' in col_map else ""
            # 简单分类
            if 'ETF' in name:
                cat = "ETF"
            elif 'LOF' in name:
                cat = "LOF"
            else:
                cat = "其他"
            categories[cat] = categories.get(cat, 0) + 1
        except Exception:
            continue

    # 预览
    preview = []
    for i in range(min(30, len(df))):
        row_data = {}
        for meaning, col in col_map.items():
            if col in df.columns:
                val = df.iloc[i][col]
                if isinstance(val, (np.floating,)):
                    val = round(float(val), 4)
                elif isinstance(val, float):
                    val = round(val, 4)
                row_data[meaning] = val
        preview.append(row_data)

    return {
        "获取时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "ETF总数": len(df),
        "分类统计": categories,
        "数据列": list(df.columns),
        "识别字段": col_map,
        "数据预览": preview,
    }


# ==================== ETF筛选 ====================

def etf_screening(category=None, min_volume=None, max_premium=None, top_n=20):
    """
    ETF筛选

    参数:
        category: 类别筛选（如"宽基"、"行业"、"债券"等）
        min_volume: 最小成交额（万元）
        max_premium: 最大折溢价率（%）
        top_n: 返回数量

    返回: 筛选结果
    """
    try:
        df = ak.fund_etf_spot_em()
    except Exception as e:
        return {"error": f"获取ETF数据失败: {str(e)}"}

    if df is None or len(df) == 0:
        return {"error": "未获取到ETF数据"}

    # 识别列
    code_col = name_col = price_col = change_col = volume_col = premium_col = None

    for col in df.columns:
        col_lower = str(col).lower()
        if ('代码' in col or 'code' in col_lower) and code_col is None:
            code_col = col
        elif ('名称' in col or 'name' in col_lower) and name_col is None:
            name_col = col
        elif ('最新' in col or '现价' in col or 'price' in col_lower) and price_col is None:
            price_col = col
        elif ('涨跌' in col or 'change' in col_lower) and change_col is None:
            change_col = col
        elif ('成交' in col and ('额' in col or 'amount' in col_lower)) and volume_col is None:
            volume_col = col
        elif ('折价' in col or '溢价' in col or 'discount' in col_lower) and premium_col is None:
            premium_col = col

    candidates = []
    for i in range(min(300, len(df))):
        try:
            row = df.iloc[i]
            name = str(row[name_col]) if name_col and name_col in df.columns else ""
            code = str(row[code_col]) if code_col and code_col in df.columns else ""

            # 类别筛选
            if category:
                cat_keywords = {
                    "宽基": ["300", "500", "1000", "上证", "深证", "创业板", "科创", "A50", "MSCI"],
                    "行业": ["医药", "医疗", "科技", "芯片", "半导体", "新能源", "光伏", "军工", "消费", "白酒", "银行", "证券", "券商", "地产", "有色", "煤炭", "钢铁", "汽车"],
                    "债券": ["债", "国债", "转债", "城投"],
                    "跨境": ["恒生", "港股", "纳指", "标普", "日经", "德国", "法国"],
                    "商品": ["黄金", "白银", "原油", "豆粕", "有色"],
                    "策略": ["红利", "低波", "价值", "成长", "质量", "动量"],
                }

                keywords = cat_keywords.get(category, [])
                if keywords:
                    matched = any(kw in name for kw in keywords)
                    if not matched:
                        continue

            # 成交额筛选
            if volume_col and volume_col in df.columns and min_volume:
                vol = pd.to_numeric(row[volume_col], errors='coerce')
                if pd.notna(vol) and float(vol) < min_volume * 10000:
                    continue

            # 折溢价筛选
            if premium_col and premium_col in df.columns and max_premium is not None:
                prem = pd.to_numeric(row[premium_col], errors='coerce')
                if pd.notna(prem) and abs(float(prem)) > max_premium:
                    continue

            price = round(float(row[price_col]), 3) if price_col and price_col in df.columns and pd.notna(row[price_col]) else None
            change = round(float(row[change_col]), 2) if change_col and change_col in df.columns and pd.notna(row[change_col]) else None
            premium = round(float(row[premium_col]), 2) if premium_col and premium_col in df.columns and pd.notna(row[premium_col]) else None

            candidates.append({
                "代码": code,
                "名称": name,
                "最新价": price,
                "涨跌幅": f"{change}%" if change is not None else "N/A",
                "折溢价": f"{premium}%" if premium is not None else "N/A",
            })

            if len(candidates) >= top_n:
                break
        except Exception:
            continue

    return {
        "筛选时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "筛选条件": {
            "类别": category or "全部",
            "最小成交额": f"{min_volume}万" if min_volume else "不限",
            "最大折溢价": f"{max_premium}%" if max_premium is not None else "不限",
        },
        "结果数量": len(candidates),
        "ETF列表": candidates,
    }


# ==================== ETF折溢价分析 ====================

def etf_premium_analysis():
    """
    ETF折溢价分析
    分析ETF市场价格与净值(IOPV)的偏离程度

    返回: 折溢价分析结果
    """
    try:
        df = ak.fund_etf_spot_em()
    except Exception as e:
        return {"error": f"获取ETF数据失败: {str(e)}"}

    if df is None or len(df) == 0:
        return {"error": "未获取到ETF数据"}

    # 识别折溢价列
    premium_col = None
    name_col = None
    code_col = None

    for col in df.columns:
        col_lower = str(col).lower()
        if ('折价' in col or '溢价' in col or 'discount' in col_lower) and premium_col is None:
            premium_col = col
        elif ('名称' in col or 'name' in col_lower) and name_col is None:
            name_col = col
        elif ('代码' in col or 'code' in col_lower) and code_col is None:
            code_col = col

    if premium_col is None:
        return {"error": "未找到折溢价数据列"}

    premiums = []
    for i in range(min(200, len(df))):
        try:
            val = pd.to_numeric(df.iloc[i][premium_col], errors='coerce')
            if pd.notna(val):
                premiums.append(float(val))
        except Exception:
            continue

    if not premiums:
        return {"error": "无法解析折溢价数据"}

    premiums_arr = np.array(premiums)

    # 统计
    avg_premium = float(np.mean(premiums_arr))
    median_premium = float(np.median(premiums_arr))
    std_premium = float(np.std(premiums_arr))

    # 折价/溢价分布
    discount_count = int(np.sum(premiums_arr < -0.5))
    fair_count = int(np.sum((premiums_arr >= -0.5) & (premiums_arr <= 0.5)))
    premium_count = int(np.sum(premiums_arr > 0.5))

    # 极端折溢价
    extreme_discount = []
    extreme_premium = []

    for i in range(min(200, len(df))):
        try:
            val = float(df.iloc[i][premium_col])
            name = str(df.iloc[i][name_col]) if name_col and name_col in df.columns else ""
            code = str(df.iloc[i][code_col]) if code_col and code_col in df.columns else ""

            if val < -3:
                extreme_discount.append({"代码": code, "名称": name, "折价率": f"{val:.2f}%"})
            elif val > 3:
                extreme_premium.append({"代码": code, "名称": name, "溢价率": f"{val:.2f}%"})
        except Exception:
            continue

    # 解读
    interpretation = []
    if avg_premium < -0.5:
        interpretation.append(f"ETF整体折价{avg_premium:.2f}%，市场情绪偏悲观")
    elif avg_premium > 0.5:
        interpretation.append(f"ETF整体溢价{avg_premium:.2f}%，市场情绪偏乐观")
    else:
        interpretation.append(f"ETF整体折溢价{avg_premium:.2f}%，市场定价合理")

    if discount_count > premium_count * 2:
        interpretation.append("折价ETF数量远超溢价ETF，市场偏弱")
    elif premium_count > discount_count * 2:
        interpretation.append("溢价ETF数量远超折价ETF，市场偏热")

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "折溢价统计": {
            "平均折溢价": f"{avg_premium:.2f}%",
            "中位数": f"{median_premium:.2f}%",
            "标准差": f"{std_premium:.2f}%",
        },
        "分布": {
            "折价ETF(<-0.5%)": discount_count,
            "平价ETF(-0.5%~0.5%)": fair_count,
            "溢价ETF(>0.5%)": premium_count,
        },
        "大幅折价ETF(<-3%)": extreme_discount[:5],
        "大幅溢价ETF(>3%)": extreme_premium[:5],
        "解读": interpretation,
        "投资建议": [
            "大幅折价的ETF可能存在套利机会（需考虑流动性）",
            "大幅溢价的ETF需警惕溢价回归风险",
            "折溢价率是判断ETF定价效率的重要指标",
        ],
    }


# ==================== ETF轮动策略 ====================

def etf_rotation_strategy(etf_codes, days=120, top_n=5):
    """
    ETF轮动策略
    基于动量/波动率等指标进行ETF轮动

    参数:
        etf_codes: ETF代码列表
        days: 回看天数
        top_n: 选取数量

    返回: 轮动策略结果
    """
    etf_scores = []

    for code in etf_codes:
        try:
            df = get_stock_kline(code, days=days)
            if df is None or len(df) < 30:
                continue

            close_col = '收盘' if '收盘' in df.columns else 'close'
            close = pd.to_numeric(df[close_col], errors='coerce').dropna()

            if len(close) < 30:
                continue

            # 动量得分（近期涨幅）
            ret_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
            ret_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0
            ret_60d = (close.iloc[-1] / close.iloc[-61] - 1) * 100 if len(close) >= 61 else 0

            momentum_score = ret_5d * 0.5 + ret_20d * 0.3 + ret_60d * 0.2

            # 波动率得分（低波动加分）
            returns = close.pct_change().dropna()
            volatility = float(returns.std() * np.sqrt(252) * 100)
            vol_score = -volatility * 0.3  # 低波动加分

            # 夏普比率
            if volatility > 0:
                sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
            else:
                sharpe = 0

            total_score = momentum_score + vol_score

            etf_scores.append({
                "代码": code,
                "5日涨幅": f"{ret_5d:.2f}%",
                "20日涨幅": f"{ret_20d:.2f}%",
                "60日涨幅": f"{ret_60d:.2f}%",
                "年化波动率": f"{volatility:.1f}%",
                "夏普比率": round(sharpe, 2),
                "综合得分": round(total_score, 2),
            })
        except Exception:
            continue
        time.sleep(0.3)

    # 排序
    etf_scores.sort(key=lambda x: x["综合得分"], reverse=True)

    return {
        "策略名称": "ETF动量轮动策略",
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "策略说明": "选取近期动量最强且波动适中的ETF，定期轮动",
        "候选ETF": etf_scores[:top_n],
        "全部排名": etf_scores[:20],
        "操作建议": [
            "每周/每两周轮动一次",
            "单只ETF仓位不超过30%",
            "关注ETF流动性和折溢价",
            "市场大幅下跌时暂停轮动",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description='ETF分析系统')
    subparsers = parser.add_subparsers(dest='command')

    # ETF列表
    list_parser = subparsers.add_parser('list', help='获取ETF列表')

    # ETF筛选
    screen_parser = subparsers.add_parser('screen', help='ETF筛选')
    screen_parser.add_argument('--category', default=None, help='类别（宽基/行业/债券/跨境/商品/策略）')
    screen_parser.add_argument('--min-volume', type=float, default=None, help='最小成交额（万元）')
    screen_parser.add_argument('--max-premium', type=float, default=None, help='最大折溢价率（%）')
    screen_parser.add_argument('--top', type=int, default=20, help='返回数量')

    # 折溢价分析
    premium_parser = subparsers.add_parser('premium', help='ETF折溢价分析')

    # ETF轮动
    rotation_parser = subparsers.add_parser('rotation', help='ETF轮动策略')
    rotation_parser.add_argument('--codes', required=True, help='ETF代码列表,逗号分隔')
    rotation_parser.add_argument('--days', type=int, default=120, help='回看天数')
    rotation_parser.add_argument('--top', type=int, default=5, help='选取数量')

    args = parser.parse_args()

    if args.command == 'list':
        result = get_etf_list()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'screen':
        result = etf_screening(
            category=args.category,
            min_volume=args.min_volume,
            max_premium=args.max_premium,
            top_n=args.top,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'premium':
        result = etf_premium_analysis()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'rotation':
        codes = [c.strip() for c in args.codes.split(',')]
        result = etf_rotation_strategy(codes, days=args.days, top_n=args.top)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
