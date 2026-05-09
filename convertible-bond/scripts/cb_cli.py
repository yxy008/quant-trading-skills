#!/usr/bin/env python3
"""
可转债分析系统
支持可转债估值、转股溢价率分析、双低策略、强赎/回售/下修分析
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


# ==================== 可转债数据获取 ====================

def get_cb_list():
    """
    获取可转债列表及核心指标

    返回: 可转债列表数据
    """
    try:
        df = ak.bond_cb_jsl()
    except Exception:
        try:
            df = ak.bond_zh_cov()
        except Exception as e:
            return {"error": f"获取可转债数据失败: {str(e)}"}

    if df is None or len(df) == 0:
        return {"error": "未获取到可转债数据"}

    result = {
        "获取时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "转债总数": len(df),
        "数据列": list(df.columns),
    }

    # 提取关键字段
    key_fields = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if '代码' in col or 'code' in col_lower:
            key_fields['代码'] = col
        elif '名称' in col or 'name' in col_lower:
            key_fields['名称'] = col
        elif '现价' in col or '价格' in col or 'price' in col_lower:
            key_fields['现价'] = col
        elif '转股价' in col or 'conv_price' in col_lower:
            key_fields['转股价'] = col
        elif '正股' in col and ('价' in col or 'price' in col_lower):
            key_fields['正股价'] = col
        elif '溢价' in col or 'premium' in col_lower:
            key_fields['转股溢价率'] = col
        elif '评级' in col or 'rating' in col_lower:
            key_fields['评级'] = col
        elif '到期' in col or 'maturity' in col_lower:
            key_fields['到期时间'] = col
        elif '剩余' in col or 'remain' in col_lower:
            key_fields['剩余年限'] = col
        elif '规模' in col or 'size' in col_lower or 'amount' in col_lower:
            key_fields['规模'] = col
        elif '成交' in col or 'volume' in col_lower:
            key_fields['成交量'] = col
        elif '涨跌' in col or 'change' in col_lower:
            key_fields['涨跌幅'] = col
        elif '纯债' in col or 'bond_value' in col_lower:
            key_fields['纯债价值'] = col
        elif '强赎' in col or 'call' in col_lower:
            key_fields['强赎触发'] = col

    result["识别字段"] = {k: v for k, v in key_fields.items()}

    # 提取前30条预览
    preview = []
    for i in range(min(30, len(df))):
        row = {}
        for meaning, col in key_fields.items():
            if col in df.columns:
                val = df.iloc[i][col]
                if isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = round(float(val), 4)
                elif isinstance(val, float):
                    val = round(val, 4)
                row[meaning] = val
        preview.append(row)

    result["数据预览"] = preview
    return result


# ==================== 可转债估值分析 ====================

def cb_valuation_analysis(symbol):
    """
    单只可转债估值分析

    参数:
        symbol: 可转债代码

    返回: 估值分析结果
    """
    try:
        df = ak.bond_cb_jsl()
    except Exception:
        try:
            df = ak.bond_zh_cov()
        except Exception as e:
            return {"error": f"获取可转债数据失败: {str(e)}"}

    if df is None or len(df) == 0:
        return {"error": "未获取到可转债数据"}

    # 查找目标转债
    code_col = None
    for col in df.columns:
        if '代码' in str(col) or 'code' in str(col).lower():
            code_col = col
            break

    if code_col is None:
        return {"error": "无法识别代码列"}

    target = df[df[code_col].astype(str).str.contains(str(symbol).replace('sz', '').replace('sh', ''))]

    if len(target) == 0:
        return {"error": f"未找到可转债{symbol}"}

    row = target.iloc[0]

    # 提取关键数据
    def safe_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    # 查找各字段
    cb_price = None
    conv_price = None
    stock_price = None
    premium_rate = None
    rating = None
    maturity = None
    issue_size = None

    for col in df.columns:
        col_lower = str(col).lower()
        val = row[col]
        if '现价' in col or '价格' in col or 'price' in col_lower:
            cb_price = safe_float(val)
        elif '转股价' in col or 'conv_price' in col_lower:
            conv_price = safe_float(val)
        elif '正股' in col and ('价' in col or 'price' in col_lower):
            stock_price = safe_float(val)
        elif '溢价' in col or 'premium' in col_lower:
            premium_rate = safe_float(val)
        elif '评级' in col or 'rating' in col_lower:
            rating = str(val)
        elif '到期' in col or 'maturity' in col_lower:
            maturity = str(val)
        elif '规模' in col or 'size' in col_lower:
            issue_size = safe_float(val)

    # 计算转股价值
    if stock_price and conv_price and conv_price > 0:
        conversion_value = stock_price / conv_price * 100
    else:
        conversion_value = None

    # 计算转股溢价率
    if cb_price and conversion_value and conversion_value > 0:
        calc_premium = (cb_price / conversion_value - 1) * 100
    else:
        calc_premium = premium_rate

    # 估值分析
    analysis = []
    if calc_premium is not None:
        if calc_premium < 0:
            analysis.append(f"转股溢价率为负({calc_premium:.1f}%)，转债折价，存在套利空间")
        elif calc_premium < 10:
            analysis.append(f"转股溢价率较低({calc_premium:.1f}%)，股性较强")
        elif calc_premium < 30:
            analysis.append(f"转股溢价率适中({calc_premium:.1f}%)，股债平衡")
        elif calc_premium < 50:
            analysis.append(f"转股溢价率偏高({calc_premium:.1f}%)，债性较强")
        else:
            analysis.append(f"转股溢价率很高({calc_premium:.1f}%)，纯债属性为主")

    if cb_price:
        if cb_price < 100:
            analysis.append(f"转债价格低于面值({cb_price:.1f})，具有债底保护")
        elif cb_price < 110:
            analysis.append(f"转债价格接近面值({cb_price:.1f})，下行空间有限")
        elif cb_price < 130:
            analysis.append(f"转债价格适中({cb_price:.1f})，需关注强赎风险")
        else:
            analysis.append(f"转债价格较高({cb_price:.1f})，强赎风险加大")

    # 双低值
    if cb_price and calc_premium is not None:
        double_low = cb_price + calc_premium
    else:
        double_low = None

    return {
        "转债代码": symbol,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "基本信息": {
            "转债价格": round(cb_price, 2) if cb_price else "N/A",
            "转股价": round(conv_price, 2) if conv_price else "N/A",
            "正股价": round(stock_price, 2) if stock_price else "N/A",
            "评级": rating or "N/A",
            "到期日": maturity or "N/A",
            "规模": f"{issue_size:.2f}亿" if issue_size else "N/A",
        },
        "估值指标": {
            "转股价值": round(conversion_value, 2) if conversion_value else "N/A",
            "转股溢价率": f"{calc_premium:.2f}%" if calc_premium is not None else "N/A",
            "双低值": round(double_low, 2) if double_low else "N/A",
        },
        "分析": analysis,
        "策略建议": _get_cb_strategy_advice(cb_price, calc_premium, double_low),
    }


def _get_cb_strategy_advice(cb_price, premium, double_low):
    """根据估值给出策略建议"""
    advice = []

    if double_low is not None:
        if double_low < 120:
            advice.append("符合双低策略标准（双低值<120），可作为双低轮动候选")
        elif double_low < 130:
            advice.append("双低值尚可（120-130），可关注但非最优")
        else:
            advice.append("双低值偏高，不适合双低策略")

    if cb_price is not None:
        if cb_price < 105:
            advice.append("价格低于105，适合低价策略，下行保护充足")
        elif cb_price < 115:
            advice.append("价格在105-115之间，性价比较好")

    if premium is not None:
        if premium < 0:
            advice.append("折价状态，关注转股套利机会（需考虑转股期限制）")
        elif premium < 5:
            advice.append("低溢价，股性联动强，适合替代正股持仓")

    if cb_price is not None and cb_price > 130:
        advice.append("价格超过130，密切关注强赎公告，及时转股或卖出")

    return advice


# ==================== 双低策略 ====================

def double_low_strategy(top_n=20):
    """
    双低策略选债
    双低值 = 转债价格 + 转股溢价率
    选取双低值最低的可转债

    参数:
        top_n: 选取数量

    返回: 双低策略结果
    """
    try:
        df = ak.bond_cb_jsl()
    except Exception:
        try:
            df = ak.bond_zh_cov()
        except Exception as e:
            return {"error": f"获取可转债数据失败: {str(e)}"}

    if df is None or len(df) == 0:
        return {"error": "未获取到可转债数据"}

    # 识别关键列
    price_col = None
    premium_col = None
    code_col = None
    name_col = None
    rating_col = None
    maturity_col = None

    for col in df.columns:
        col_lower = str(col).lower()
        if ('现价' in col or '价格' in col or 'price' in col_lower) and price_col is None:
            price_col = col
        elif ('溢价' in col or 'premium' in col_lower) and premium_col is None:
            premium_col = col
        elif ('代码' in col or 'code' in col_lower) and code_col is None:
            code_col = col
        elif ('名称' in col or 'name' in col_lower) and name_col is None:
            name_col = col
        elif ('评级' in col or 'rating' in col_lower) and rating_col is None:
            rating_col = col
        elif ('到期' in col or 'maturity' in col_lower) and maturity_col is None:
            maturity_col = col

    if price_col is None or premium_col is None:
        return {"error": "无法识别价格或溢价率列"}

    # 计算双低值
    df_valid = df[[price_col, premium_col]].copy()
    df_valid = df_valid.dropna()

    prices = pd.to_numeric(df_valid[price_col], errors='coerce')
    premiums = pd.to_numeric(df_valid[premium_col], errors='coerce')

    double_low_values = prices + premiums

    # 排序取最低
    valid_idx = double_low_values.dropna().sort_values().index[:top_n * 2]

    candidates = []
    for idx in valid_idx:
        try:
            row = df.iloc[idx]
            cb_price = float(prices.iloc[idx])
            cb_premium = float(premiums.iloc[idx])
            dl_value = cb_price + cb_premium

            # 过滤条件
            if cb_price > 150 or cb_premium > 100:
                continue

            candidate = {
                "代码": str(row[code_col]) if code_col and code_col in df.columns else "",
                "名称": str(row[name_col]) if name_col and name_col in df.columns else "",
                "转债价格": round(cb_price, 2),
                "转股溢价率": f"{cb_premium:.2f}%",
                "双低值": round(dl_value, 2),
                "评级": str(row[rating_col]) if rating_col and rating_col in df.columns else "N/A",
            }
            candidates.append(candidate)

            if len(candidates) >= top_n:
                break
        except Exception:
            continue

    return {
        "策略名称": "双低策略（低价+低溢价）",
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "策略说明": "选取转债价格+转股溢价率最低的可转债，兼顾防守（低价）和进攻（低溢价）",
        "候选数量": len(candidates),
        "双低转债": candidates,
        "操作建议": [
            "每周/每月轮动一次，卖出双低值升高的，买入双低值降低的",
            "单只转债仓位不超过10%",
            "价格超过130或发布强赎公告时及时卖出",
            "优先选择评级AA及以上的转债",
        ],
    }


# ==================== 可转债条款分析 ====================

def cb_terms_analysis(symbol):
    """
    可转债条款分析（强赎/回售/下修）

    参数:
        symbol: 可转债代码

    返回: 条款分析结果
    """
    try:
        df = ak.bond_cb_jsl()
    except Exception:
        try:
            df = ak.bond_zh_cov()
        except Exception as e:
            return {"error": f"获取可转债数据失败: {str(e)}"}

    if df is None or len(df) == 0:
        return {"error": "未获取到可转债数据"}

    # 查找目标转债
    code_col = None
    for col in df.columns:
        if '代码' in str(col) or 'code' in str(col).lower():
            code_col = col
            break

    if code_col is None:
        return {"error": "无法识别代码列"}

    target = df[df[code_col].astype(str).str.contains(str(symbol).replace('sz', '').replace('sh', ''))]

    if len(target) == 0:
        return {"error": f"未找到可转债{symbol}"}

    row = target.iloc[0]

    # 提取数据
    cb_price = None
    conv_price = None
    stock_price = None
    premium = None
    maturity = None
    call_trigger = None

    for col in df.columns:
        col_lower = str(col).lower()
        val = row[col]
        try:
            val_f = float(val)
        except (ValueError, TypeError):
            val_f = None

        if ('现价' in col or '价格' in col or 'price' in col_lower) and cb_price is None:
            cb_price = val_f
        elif ('转股价' in col or 'conv_price' in col_lower) and conv_price is None:
            conv_price = val_f
        elif ('正股' in col and ('价' in col or 'price' in col_lower)) and stock_price is None:
            stock_price = val_f
        elif ('溢价' in col or 'premium' in col_lower) and premium is None:
            premium = val_f
        elif ('到期' in col or 'maturity' in col_lower) and maturity is None:
            maturity = str(val)
        elif ('强赎' in col or 'call' in col_lower) and call_trigger is None:
            call_trigger = str(val)

    # 强赎分析
    call_analysis = {}
    if stock_price and conv_price and conv_price > 0:
        price_ratio = stock_price / conv_price
        if price_ratio >= 1.3:
            call_analysis["强赎状态"] = "已触发强赎条件（正股价>=转股价*130%）"
            call_analysis["风险等级"] = "高"
            call_analysis["建议"] = "立即卖出或转股，避免被低价强赎"
        elif price_ratio >= 1.2:
            call_analysis["强赎状态"] = "接近强赎触发（正股价>=转股价*120%）"
            call_analysis["风险等级"] = "中"
            call_analysis["建议"] = "密切关注，做好卖出准备"
        elif price_ratio >= 1.1:
            call_analysis["强赎状态"] = "距离强赎有一定空间"
            call_analysis["风险等级"] = "低"
            call_analysis["建议"] = "正常持有"
        else:
            call_analysis["强赎状态"] = "距离强赎较远"
            call_analysis["风险等级"] = "极低"
            call_analysis["建议"] = "正常持有"

        call_analysis["正股/转股价比例"] = f"{price_ratio * 100:.1f}%"

    # 回售分析
    put_analysis = {}
    if cb_price and cb_price < 100:
        put_analysis["回售状态"] = "转债价格低于面值，关注回售条款"
        put_analysis["建议"] = "若进入回售期且满足条件，可博弈回售收益"
    else:
        put_analysis["回售状态"] = "转债价格高于面值，回售暂无博弈价值"

    # 下修分析
    reset_analysis = {}
    if stock_price and conv_price and conv_price > 0:
        if stock_price < conv_price * 0.85:
            reset_analysis["下修状态"] = "正股价远低于转股价，下修概率较高"
            reset_analysis["建议"] = "关注董事会提议下修公告，下修利好转债价格"
        elif stock_price < conv_price * 0.9:
            reset_analysis["下修状态"] = "正股价低于转股价，存在下修可能"
            reset_analysis["建议"] = "关注公司动态"
        else:
            reset_analysis["下修状态"] = "正股价接近或高于转股价，下修概率低"
            reset_analysis["建议"] = "无需关注下修"

    return {
        "转债代码": symbol,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "基本信息": {
            "转债价格": round(cb_price, 2) if cb_price else "N/A",
            "转股价": round(conv_price, 2) if conv_price else "N/A",
            "正股价": round(stock_price, 2) if stock_price else "N/A",
            "到期日": maturity or "N/A",
        },
        "强赎分析": call_analysis,
        "回售分析": put_analysis,
        "下修分析": reset_analysis,
        "条款知识": {
            "强赎条款": "通常为连续30日中至少15日正股价>=转股价*130%，公司有权按面值+利息赎回",
            "回售条款": "通常为最后2个计息年度，连续30日正股价<转股价*70%，持有人有权按面值+利息回售",
            "下修条款": "通常为连续30日中至少15日正股价<转股价*85%，董事会有权提议下修转股价",
        },
    }


def main():
    parser = argparse.ArgumentParser(description='可转债分析系统')
    subparsers = parser.add_subparsers(dest='command')

    # 转债列表
    list_parser = subparsers.add_parser('list', help='获取可转债列表')

    # 估值分析
    val_parser = subparsers.add_parser('valuation', help='可转债估值分析')
    val_parser.add_argument('--symbol', required=True, help='可转债代码')

    # 双低策略
    dl_parser = subparsers.add_parser('double-low', help='双低策略选债')
    dl_parser.add_argument('--top', type=int, default=20, help='选取数量')

    # 条款分析
    terms_parser = subparsers.add_parser('terms', help='可转债条款分析')
    terms_parser.add_argument('--symbol', required=True, help='可转债代码')

    args = parser.parse_args()

    if args.command == 'list':
        result = get_cb_list()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'valuation':
        result = cb_valuation_analysis(args.symbol)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'double-low':
        result = double_low_strategy(top_n=args.top)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'terms':
        result = cb_terms_analysis(args.symbol)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
