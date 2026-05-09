#!/usr/bin/env python3
"""
股票筛选器 - 基于 AkShare 真实数据
支持基础筛选 + 多因子综合评分选股
"""
import argparse
import json
import sys
import os
import time

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

from data_utils import _get_spot_df


# ==================== 因子方向定义 ====================

FACTOR_DIRECTIONS = {
    "价值因子": 1,       # PE/PB越低越好（正向）
    "动量因子": 1,       # 涨幅越大越好
    "质量因子": 1,       # 市值越大越好
    "活跃度因子": 1,     # 成交额越大越好
    "波动因子": -1,      # 振幅越小越好（反向）
}


# ==================== 基础筛选 ====================

def filter_stocks(pe_min=None, pe_max=None, pb_min=None, pb_max=None,
                  market_cap_min=None, market_cap_max=None,
                  change_min=None, change_max=None,
                  limit=20):
    """筛选股票 - 基于真实A股数据"""

    for attempt in range(3):
        try:
            df = _get_spot_df()
            if df is None or df.empty:
                continue

            df = df.copy()

            # 过滤ST股票
            if '名称' in df.columns:
                df = df[~df['名称'].str.contains('ST', na=False)]

            # 市盈率过滤
            if pe_min is not None or pe_max is not None:
                pe_col = '市盈率-动态'
                if pe_col in df.columns:
                    df[pe_col] = pd.to_numeric(df[pe_col], errors='coerce')
                    if pe_min is not None:
                        df = df[df[pe_col] >= pe_min]
                    if pe_max is not None:
                        df = df[df[pe_col] <= pe_max]

            # 市净率过滤
            if pb_min is not None or pb_max is not None:
                pb_col = '市净率'
                if pb_col in df.columns:
                    df[pb_col] = pd.to_numeric(df[pb_col], errors='coerce')
                    if pb_min is not None:
                        df = df[df[pb_col] >= pb_min]
                    if pb_max is not None:
                        df = df[df[pb_col] <= pb_max]

            # 总市值过滤
            if market_cap_min is not None or market_cap_max is not None:
                mc_col = '总市值'
                if mc_col in df.columns:
                    df[mc_col] = pd.to_numeric(df[mc_col], errors='coerce')
                    if market_cap_min is not None:
                        df = df[df[mc_col] >= market_cap_min]
                    if market_cap_max is not None:
                        df = df[df[mc_col] <= market_cap_max]

            # 涨跌幅过滤
            if change_min is not None or change_max is not None:
                chg_col = '涨跌幅'
                if chg_col in df.columns:
                    df[chg_col] = pd.to_numeric(df[chg_col], errors='coerce')
                    if change_min is not None:
                        df = df[df[chg_col] >= change_min]
                    if change_max is not None:
                        df = df[df[chg_col] <= change_max]

            # 按总市值降序排列
            if '总市值' in df.columns:
                df = df.sort_values('总市值', ascending=False)

            df = df.head(limit)

            results = []
            for _, row in df.iterrows():
                results.append({
                    "代码": str(row.get('代码', '')),
                    "名称": str(row.get('名称', '')),
                    "最新价": float(row.get('最新价', 0)) if pd.notna(row.get('最新价')) else 0,
                    "涨跌幅": float(row.get('涨跌幅', 0)) if pd.notna(row.get('涨跌幅')) else 0,
                    "市盈率-动态": float(row.get('市盈率-动态', 0)) if pd.notna(row.get('市盈率-动态')) else None,
                    "市净率": float(row.get('市净率', 0)) if pd.notna(row.get('市净率')) else None,
                    "总市值": float(row.get('总市值', 0)) if pd.notna(row.get('总市值')) else 0,
                    "成交额": float(row.get('成交额', 0)) if pd.notna(row.get('成交额')) else 0
                })

            return {
                "count": len(results),
                "stocks": results
            }

        except Exception:
            pass

        time.sleep(1)

    return {"count": 0, "stocks": [], "message": "无法获取实时数据，请稍后重试"}


# ==================== 因子计算 ====================

def _calc_value_factor(row):
    """
    价值因子评分（0-100）
    综合考虑PE和PB，低估值得高分
    """
    score = 50.0
    count = 0

    pe = row.get('市盈率-动态')
    if pe is not None and pd.notna(pe):
        pe = float(pe)
        if 0 < pe <= 10:
            score += 40
        elif 10 < pe <= 20:
            score += 30
        elif 20 < pe <= 30:
            score += 20
        elif 30 < pe <= 50:
            score += 10
        elif pe > 50:
            score += 0
        else:
            score += 5
        count += 1

    pb = row.get('市净率')
    if pb is not None and pd.notna(pb):
        pb = float(pb)
        if 0 < pb <= 1:
            score += 40
        elif 1 < pb <= 2:
            score += 30
        elif 2 < pb <= 4:
            score += 20
        elif 4 < pb <= 8:
            score += 10
        else:
            score += 0
        count += 1

    if count > 0:
        score = score / (count * 2)

    return round(min(score, 100), 1)


def _calc_momentum_factor(row):
    """
    动量因子评分（0-100）
    当日涨跌幅越大越好
    """
    chg = row.get('涨跌幅')
    if chg is None or pd.isna(chg):
        return 50.0

    chg = float(chg)
    if chg > 5:
        return 90.0
    elif chg > 3:
        return 80.0
    elif chg > 1:
        return 70.0
    elif chg > 0:
        return 60.0
    elif chg > -1:
        return 50.0
    elif chg > -3:
        return 35.0
    elif chg > -5:
        return 20.0
    else:
        return 10.0


def _calc_quality_factor(row):
    """
    质量因子评分（0-100）
    市值越大、流动性越好，得分越高
    """
    score = 50.0

    mc = row.get('总市值')
    if mc is not None and pd.notna(mc):
        mc = float(mc)
        if mc > 1e12:       # 万亿市值
            score += 30
        elif mc > 5e11:     # 5000亿
            score += 25
        elif mc > 1e11:     # 1000亿
            score += 20
        elif mc > 5e10:     # 500亿
            score += 15
        elif mc > 1e10:     # 100亿
            score += 10
        elif mc > 5e9:      # 50亿
            score += 5
        else:
            score += 0

    return round(min(score, 100), 1)


def _calc_activity_factor(row):
    """
    活跃度因子评分（0-100）
    成交额越大，活跃度越高
    """
    amount = row.get('成交额')
    if amount is None or pd.isna(amount):
        return 50.0

    amount = float(amount)
    if amount > 5e9:        # 50亿+
        return 95.0
    elif amount > 1e9:      # 10亿+
        return 85.0
    elif amount > 5e8:      # 5亿+
        return 75.0
    elif amount > 1e8:      # 1亿+
        return 65.0
    elif amount > 5e7:      # 5000万+
        return 55.0
    elif amount > 1e7:      # 1000万+
        return 45.0
    else:
        return 30.0


def _calc_volatility_factor(row):
    """
    波动因子评分（0-100）
    振幅越小越稳定，得分越高（反向因子）
    """
    amp = row.get('振幅')
    if amp is None or pd.isna(amp):
        # 尝试用涨跌幅估算
        chg = row.get('涨跌幅')
        if chg is not None and pd.notna(chg):
            amp = abs(float(chg)) * 1.5
        else:
            return 50.0

    amp = float(amp)
    if amp < 2:
        return 90.0
    elif amp < 3:
        return 80.0
    elif amp < 5:
        return 70.0
    elif amp < 7:
        return 55.0
    elif amp < 10:
        return 40.0
    else:
        return 20.0


# ==================== 因子标准化 ====================

def _normalize_rank(scores_dict):
    """
    Rank标准化 - 将原始得分转为0-100的百分位排名
    scores_dict: {股票代码: 原始得分}
    """
    if not scores_dict:
        return {}

    items = sorted(scores_dict.items(), key=lambda x: x[1])
    n = len(items)
    if n <= 1:
        return {k: 50.0 for k in scores_dict}

    result = {}
    for rank, (code, _) in enumerate(items):
        result[code] = round(rank / (n - 1) * 100, 1)

    return result


# ==================== 多因子综合选股 ====================

def multi_factor_ranking(pe_min=None, pe_max=None, pb_min=None, pb_max=None,
                          market_cap_min=None, market_cap_max=None,
                          change_min=None, change_max=None,
                          factor_weights=None, limit=20,
                          strategy="balanced"):
    """
    多因子综合评分选股
    对全市场股票进行多维度因子评分，综合排名选出最优标的

    参数:
        pe_min/pe_max: PE筛选范围
        pb_min/pb_max: PB筛选范围
        market_cap_min/max: 市值筛选范围
        change_min/max: 涨跌幅筛选范围
        factor_weights: 自定义因子权重 {"价值因子": 0.3, ...}
        limit: 返回数量
        strategy: 选股策略
            - "balanced": 均衡型，各因子等权
            - "value": 价值型，偏重PE/PB
            - "growth": 成长型，偏重动量和活跃度
            - "quality": 质量型，偏重市值和稳定性
            - "momentum": 动量型，偏重涨跌幅
    """
    # 策略预设权重
    strategy_weights = {
        "balanced": {"价值因子": 0.25, "动量因子": 0.20, "质量因子": 0.20,
                      "活跃度因子": 0.20, "波动因子": 0.15},
        "value":    {"价值因子": 0.40, "动量因子": 0.10, "质量因子": 0.15,
                      "活跃度因子": 0.15, "波动因子": 0.20},
        "growth":   {"价值因子": 0.15, "动量因子": 0.30, "质量因子": 0.15,
                      "活跃度因子": 0.25, "波动因子": 0.15},
        "quality":  {"价值因子": 0.20, "动量因子": 0.10, "质量因子": 0.35,
                      "活跃度因子": 0.15, "波动因子": 0.20},
        "momentum": {"价值因子": 0.10, "动量因子": 0.40, "质量因子": 0.10,
                      "活跃度因子": 0.25, "波动因子": 0.15},
    }

    if factor_weights is None:
        factor_weights = strategy_weights.get(strategy, strategy_weights["balanced"])

    for attempt in range(3):
        try:
            df = _get_spot_df()
            if df is None or df.empty:
                continue

            df = df.copy()

            # 过滤ST股票
            if '名称' in df.columns:
                df = df[~df['名称'].str.contains('ST', na=False)]

            # 基础筛选条件
            if pe_min is not None or pe_max is not None:
                pe_col = '市盈率-动态'
                if pe_col in df.columns:
                    df[pe_col] = pd.to_numeric(df[pe_col], errors='coerce')
                    if pe_min is not None:
                        df = df[df[pe_col] >= pe_min]
                    if pe_max is not None:
                        df = df[df[pe_col] <= pe_max]

            if pb_min is not None or pb_max is not None:
                pb_col = '市净率'
                if pb_col in df.columns:
                    df[pb_col] = pd.to_numeric(df[pb_col], errors='coerce')
                    if pb_min is not None:
                        df = df[df[pb_col] >= pb_min]
                    if pb_max is not None:
                        df = df[df[pb_col] <= pb_max]

            if market_cap_min is not None or market_cap_max is not None:
                mc_col = '总市值'
                if mc_col in df.columns:
                    df[mc_col] = pd.to_numeric(df[mc_col], errors='coerce')
                    if market_cap_min is not None:
                        df = df[df[mc_col] >= market_cap_min]
                    if market_cap_max is not None:
                        df = df[df[mc_col] <= market_cap_max]

            if change_min is not None or change_max is not None:
                chg_col = '涨跌幅'
                if chg_col in df.columns:
                    df[chg_col] = pd.to_numeric(df[chg_col], errors='coerce')
                    if change_min is not None:
                        df = df[df[chg_col] >= change_min]
                    if change_max is not None:
                        df = df[df[chg_col] <= change_max]

            if df.empty:
                return {"count": 0, "stocks": [], "message": "筛选条件过严，无符合条件的股票"}

            # 计算各因子原始得分
            raw_scores = {}
            stock_info = {}

            for _, row in df.iterrows():
                code = str(row.get('代码', ''))
                if not code:
                    continue

                raw_scores[code] = {
                    "价值因子": _calc_value_factor(row),
                    "动量因子": _calc_momentum_factor(row),
                    "质量因子": _calc_quality_factor(row),
                    "活跃度因子": _calc_activity_factor(row),
                    "波动因子": _calc_volatility_factor(row),
                }
                stock_info[code] = {
                    "代码": code,
                    "名称": str(row.get('名称', '')),
                    "最新价": float(row.get('最新价', 0)) if pd.notna(row.get('最新价')) else 0,
                    "涨跌幅": float(row.get('涨跌幅', 0)) if pd.notna(row.get('涨跌幅')) else 0,
                    "市盈率-动态": float(row.get('市盈率-动态', 0)) if pd.notna(row.get('市盈率-动态')) else None,
                    "市净率": float(row.get('市净率', 0)) if pd.notna(row.get('市净率')) else None,
                    "总市值": float(row.get('总市值', 0)) if pd.notna(row.get('总市值')) else 0,
                    "成交额": float(row.get('成交额', 0)) if pd.notna(row.get('成交额')) else 0,
                }

            if not raw_scores:
                return {"count": 0, "stocks": [], "message": "无有效股票数据"}

            # 对每个因子进行Rank标准化
            normalized = {code: {} for code in raw_scores}
            for factor_name in ["价值因子", "动量因子", "质量因子", "活跃度因子", "波动因子"]:
                factor_dict = {code: raw_scores[code][factor_name] for code in raw_scores}
                normed = _normalize_rank(factor_dict)
                for code in raw_scores:
                    normalized[code][factor_name] = normed.get(code, 50.0)

            # 综合评分 = 各因子标准化得分 * 权重 之和
            composite_scores = {}
            for code in raw_scores:
                total = 0.0
                for factor_name, weight in factor_weights.items():
                    factor_score = normalized[code].get(factor_name, 50.0)
                    total += factor_score * weight
                composite_scores[code] = round(total, 2)

            # 按综合得分降序排列
            ranked = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
            top_stocks = ranked[:limit]

            # 构建结果
            results = []
            for rank_idx, (code, score) in enumerate(top_stocks):
                info = stock_info.get(code, {})
                factor_detail = {f: round(normalized[code].get(f, 50), 1) for f in factor_weights}
                raw_detail = {f: round(raw_scores[code].get(f, 50), 1) for f in factor_weights}

                stock_result = {
                    "排名": rank_idx + 1,
                    "代码": info.get("代码", code),
                    "名称": info.get("名称", ""),
                    "最新价": info.get("最新价", 0),
                    "涨跌幅": info.get("涨跌幅", 0),
                    "市盈率-动态": info.get("市盈率-动态"),
                    "市净率": info.get("市净率"),
                    "总市值": info.get("总市值", 0),
                    "成交额": info.get("成交额", 0),
                    "综合得分": score,
                    "因子得分": factor_detail,
                    "因子原始值": raw_detail,
                }
                results.append(stock_result)

            # 统计信息
            all_scores = list(composite_scores.values())
            score_stats = {}
            if all_scores:
                score_stats = {
                    "最高分": round(max(all_scores), 2),
                    "最低分": round(min(all_scores), 2),
                    "平均分": round(np.mean(all_scores), 2),
                    "中位数": round(np.median(all_scores), 2),
                    "标准差": round(np.std(all_scores, ddof=1), 2),
                }

            return {
                "选股策略": strategy,
                "因子权重": {k: round(v * 100, 1) for k, v in factor_weights.items()},
                "候选股票数": len(composite_scores),
                "返回数量": len(results),
                "得分统计": score_stats,
                "选股结果": results,
            }

        except Exception as e:
            if attempt == 2:
                return {"count": 0, "stocks": [], "message": f"选股出错: {str(e)}"}

        time.sleep(1)

    return {"count": 0, "stocks": [], "message": "无法获取实时数据，请稍后重试"}


# ==================== 行业相对评分选股 ====================

def sector_relative_ranking(sector_name=None, limit=20):
    """
    行业内相对评分选股
    在同一行业内对股票进行多因子评分，选出行业内最优标的
    """
    try:
        # 获取行业板块成分股
        if sector_name:
            try:
                df_sector = ak.stock_board_industry_cons_em(symbol=sector_name)
                if df_sector is not None and not df_sector.empty:
                    symbols = df_sector['代码'].tolist() if '代码' in df_sector.columns else []
                else:
                    return {"error": f"未找到板块 {sector_name} 的成分股"}
            except Exception:
                return {"error": f"获取板块 {sector_name} 成分股失败"}
        else:
            return {"error": "请指定板块名称"}

        if len(symbols) < 3:
            return {"error": f"板块 {sector_name} 成分股数量不足"}

        # 获取实时行情
        df_spot = _get_spot_df()
        if df_spot is None or df_spot.empty:
            return {"error": "无法获取实时行情数据"}

        # 筛选板块内股票
        df_spot = df_spot[df_spot['代码'].isin(symbols)].copy()
        if df_spot.empty:
            return {"error": f"板块 {sector_name} 无有效行情数据"}

        # 过滤ST
        if '名称' in df_spot.columns:
            df_spot = df_spot[~df_spot['名称'].str.contains('ST', na=False)]

        # 计算因子得分
        raw_scores = {}
        stock_info = {}

        for _, row in df_spot.iterrows():
            code = str(row.get('代码', ''))
            if not code:
                continue

            raw_scores[code] = {
                "价值因子": _calc_value_factor(row),
                "动量因子": _calc_momentum_factor(row),
                "质量因子": _calc_quality_factor(row),
                "活跃度因子": _calc_activity_factor(row),
                "波动因子": _calc_volatility_factor(row),
            }
            stock_info[code] = {
                "代码": code,
                "名称": str(row.get('名称', '')),
                "最新价": float(row.get('最新价', 0)) if pd.notna(row.get('最新价')) else 0,
                "涨跌幅": float(row.get('涨跌幅', 0)) if pd.notna(row.get('涨跌幅')) else 0,
                "市盈率-动态": float(row.get('市盈率-动态', 0)) if pd.notna(row.get('市盈率-动态')) else None,
                "市净率": float(row.get('市净率', 0)) if pd.notna(row.get('市净率')) else None,
                "总市值": float(row.get('总市值', 0)) if pd.notna(row.get('总市值')) else 0,
                "成交额": float(row.get('成交额', 0)) if pd.notna(row.get('成交额')) else 0,
            }

        if not raw_scores:
            return {"error": "无有效股票数据"}

        # Rank标准化
        normalized = {code: {} for code in raw_scores}
        for factor_name in ["价值因子", "动量因子", "质量因子", "活跃度因子", "波动因子"]:
            factor_dict = {code: raw_scores[code][factor_name] for code in raw_scores}
            normed = _normalize_rank(factor_dict)
            for code in raw_scores:
                normalized[code][factor_name] = normed.get(code, 50.0)

        # 行业内等权综合评分
        weights = {"价值因子": 0.25, "动量因子": 0.20, "质量因子": 0.20,
                    "活跃度因子": 0.20, "波动因子": 0.15}

        composite_scores = {}
        for code in raw_scores:
            total = 0.0
            for factor_name, weight in weights.items():
                total += normalized[code].get(factor_name, 50.0) * weight
            composite_scores[code] = round(total, 2)

        ranked = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        top_stocks = ranked[:limit]

        results = []
        for rank_idx, (code, score) in enumerate(top_stocks):
            info = stock_info.get(code, {})
            results.append({
                "行业内排名": rank_idx + 1,
                "代码": info.get("代码", code),
                "名称": info.get("名称", ""),
                "最新价": info.get("最新价", 0),
                "涨跌幅": info.get("涨跌幅", 0),
                "市盈率-动态": info.get("市盈率-动态"),
                "市净率": info.get("市净率"),
                "总市值": info.get("总市值", 0),
                "成交额": info.get("成交额", 0),
                "行业内综合得分": score,
                "因子得分": {f: round(normalized[code].get(f, 50), 1) for f in weights},
            })

        return {
            "板块": sector_name,
            "板块内股票数": len(raw_scores),
            "返回数量": len(results),
            "选股结果": results,
        }

    except Exception as e:
        return {"error": f"行业内选股出错: {str(e)}"}


def main():
    parser = argparse.ArgumentParser(description='股票筛选器 - 支持基础筛选和多因子综合选股')
    parser.add_argument('action', choices=['filter', 'ranking', 'sector_ranking'],
                        help='操作类型: filter（基础筛选）/ ranking（多因子综合排名）/ sector_ranking（行业内排名）')
    parser.add_argument('--pe_min', type=float, help='最小市盈率（PE）')
    parser.add_argument('--pe_max', type=float, help='最大市盈率（PE）')
    parser.add_argument('--pb_min', type=float, help='最小市净率（PB）')
    parser.add_argument('--pb_max', type=float, help='最大市净率（PB）')
    parser.add_argument('--market_cap_min', type=float, help='最小总市值（单位：元）')
    parser.add_argument('--market_cap_max', type=float, help='最大总市值（单位：元）')
    parser.add_argument('--change_min', type=float, help='最小涨跌幅（%）')
    parser.add_argument('--change_max', type=float, help='最大涨跌幅（%）')
    parser.add_argument('--limit', type=int, default=20, help='返回股票数量限制，默认 20')
    parser.add_argument('--strategy', default='balanced',
                        choices=['balanced', 'value', 'growth', 'quality', 'momentum'],
                        help='选股策略: balanced(均衡)/value(价值)/growth(成长)/quality(质量)/momentum(动量)')
    parser.add_argument('--sector', type=str, help='板块名称（用于行业内排名）')

    args = parser.parse_args()

    try:
        if args.action == 'filter':
            data = filter_stocks(
                pe_min=args.pe_min, pe_max=args.pe_max,
                pb_min=args.pb_min, pb_max=args.pb_max,
                market_cap_min=args.market_cap_min, market_cap_max=args.market_cap_max,
                change_min=args.change_min, change_max=args.change_max,
                limit=args.limit
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))

        elif args.action == 'ranking':
            data = multi_factor_ranking(
                pe_min=args.pe_min, pe_max=args.pe_max,
                pb_min=args.pb_min, pb_max=args.pb_max,
                market_cap_min=args.market_cap_min, market_cap_max=args.market_cap_max,
                change_min=args.change_min, change_max=args.change_max,
                limit=args.limit, strategy=args.strategy
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))

        elif args.action == 'sector_ranking':
            data = sector_relative_ranking(
                sector_name=args.sector, limit=args.limit
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == '__main__':
    main()
