#!/usr/bin/env python3
"""
市场情绪/舆情分析模块 - 新闻情感分析、市场热度、情绪指标
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


def market_sentiment_index(symbol, days=60):
    """
    计算市场情绪综合指标
    基于价格动量、成交量、波动率等构建情绪指数
    """
    try:
        df = get_stock_kline(symbol, days=days + 30)
        if df is None or df.empty:
            return {"error": f"无法获取 {symbol} 的行情数据"}

        close_col = '收盘' if '收盘' in df.columns else ('close' if 'close' in df.columns else None)
        volume_col = '成交量' if '成交量' in df.columns else ('volume' if 'volume' in df.columns else None)
        high_col = '最高' if '最高' in df.columns else ('high' if 'high' in df.columns else None)
        low_col = '最低' if '最低' in df.columns else ('low' if 'low' in df.columns else None)

        if close_col is None:
            return {"error": "数据缺少收盘价列"}

        closes = df[close_col].values
        volumes = df[volume_col].values if volume_col else np.ones(len(closes))
        highs = df[high_col].values if high_col else closes
        lows = df[low_col].values if low_col else closes

        n = len(closes)
        if n < 20:
            return {"error": "数据量不足"}

        # 1. 价格动量得分 (0-100)
        ma5 = np.mean(closes[-5:])
        ma20 = np.mean(closes[-20:])
        momentum_score = min(100, max(0, (closes[-1] / ma20 - 1) * 500 + 50))

        # 2. 成交量情绪 (0-100)
        vol_ma5 = np.mean(volumes[-5:])
        vol_ma20 = np.mean(volumes[-20:])
        volume_score = min(100, max(0, (vol_ma5 / vol_ma20 - 1) * 200 + 50))

        # 3. 涨跌比 (0-100)
        up_days = sum(1 for i in range(1, min(20, n)) if closes[-i] > closes[-i - 1])
        advance_score = up_days / min(19, n - 1) * 100

        # 4. 波动率情绪 (0-100) - 高波动通常意味着恐慌
        returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, n)]
        volatility = np.std(returns[-20:]) * np.sqrt(252)
        vol_score = max(0, min(100, 100 - volatility * 200))

        # 5. 相对强弱 RSI
        gains = [max(0, closes[i] - closes[i - 1]) for i in range(1, n)]
        losses = [max(0, closes[i - 1] - closes[i]) for i in range(1, n)]
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rsi = 100 - 100 / (1 + avg_gain / avg_loss) if avg_loss > 0 else 100

        # 综合情绪指数
        sentiment = (momentum_score * 0.25 + volume_score * 0.2 +
                     advance_score * 0.2 + vol_score * 0.15 + rsi * 0.2)

        # 情绪等级
        if sentiment >= 70:
            level = "极度乐观"
            suggestion = "市场情绪过热，注意回调风险"
        elif sentiment >= 55:
            level = "偏乐观"
            suggestion = "市场情绪积极，可适度参与"
        elif sentiment >= 45:
            level = "中性"
            suggestion = "市场情绪平稳，观望为主"
        elif sentiment >= 30:
            level = "偏悲观"
            suggestion = "市场情绪低迷，谨慎操作"
        else:
            level = "极度悲观"
            suggestion = "市场恐慌，可能存在超跌机会"

        # 近期情绪变化
        recent_sentiments = []
        for i in range(min(10, n - 20), 0, -1):
            idx = n - i
            seg_closes = closes[max(0, idx - 20):idx]
            if len(seg_closes) >= 5:
                seg_ma5 = np.mean(seg_closes[-5:])
                seg_ma20 = np.mean(seg_closes)
                seg_mom = min(100, max(0, (seg_closes[-1] / seg_ma20 - 1) * 500 + 50))
                recent_sentiments.append(round(seg_mom, 1))

        return {
            "股票代码": symbol,
            "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "综合情绪指数": round(sentiment, 1),
            "情绪等级": level,
            "操作建议": suggestion,
            "分项得分": {
                "价格动量": round(momentum_score, 1),
                "成交量情绪": round(volume_score, 1),
                "涨跌比": round(advance_score, 1),
                "波动率情绪": round(vol_score, 1),
                "RSI": round(rsi, 1),
            },
            "近期情绪变化": recent_sentiments,
            "技术指标": {
                "MA5": round(ma5, 2),
                "MA20": round(ma20, 2),
                "当前价": round(closes[-1], 2),
                "RSI_14": round(rsi, 1),
                "20日波动率": round(volatility * 100, 1),
            },
        }
    except Exception as e:
        return {"error": str(e)}


def market_breadth(symbols, days=5):
    """
    市场宽度分析
    分析多只股票的涨跌分布
    """
    if not symbols:
        return {"error": "请提供股票列表"}

    results = []
    up_count = 0
    down_count = 0
    flat_count = 0

    for symbol in symbols[:50]:
        try:
            df = get_stock_kline(symbol, days=days + 5)
            if df is None or df.empty:
                continue

            close_col = '收盘' if '收盘' in df.columns else ('close' if 'close' in df.columns else None)
            if close_col is None:
                continue

            closes = df[close_col].values
            if len(closes) < days + 1:
                continue

            change = (closes[-1] - closes[-days - 1]) / closes[-days - 1] * 100

            results.append({
                "股票代码": symbol,
                "涨跌幅": round(change, 2),
            })

            if change > 0.5:
                up_count += 1
            elif change < -0.5:
                down_count += 1
            else:
                flat_count += 1
        except Exception:
            continue

    total = up_count + down_count + flat_count
    if total == 0:
        return {"error": "无法获取任何股票数据"}

    # 按涨跌幅排序
    results.sort(key=lambda x: x["涨跌幅"], reverse=True)

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "分析周期": f"{days}日",
        "样本数量": total,
        "上涨数量": up_count,
        "下跌数量": down_count,
        "持平数量": flat_count,
        "上涨比例": round(up_count / total * 100, 1),
        "下跌比例": round(down_count / total * 100, 1),
        "市场宽度": round((up_count - down_count) / total * 100, 1),
        "涨幅前5": results[:5],
        "跌幅前5": results[-5:][::-1],
    }


def sentiment_news_analysis(news_items):
    """
    新闻情感分析
    对新闻标题/内容进行情感评分

    news_items: [{"title": "...", "content": "..."}, ...]
    """
    if not news_items:
        return {"error": "请提供新闻数据"}

    # 中文情感词典
    positive_words = [
        "上涨", "涨停", "利好", "增长", "突破", "创新高", "盈利", "回购", "增持",
        "超预期", "业绩预增", "分红", "政策支持", "行业景气", "需求旺盛", "订单增加",
        "技术突破", "市场份额提升", "评级上调", "买入", "推荐", "看好", "强劲",
        "反弹", "复苏", "改善", "优化", "扩张", "合作", "签约", "中标",
    ]

    negative_words = [
        "下跌", "跌停", "利空", "下滑", "跌破", "创新低", "亏损", "减持", "套现",
        "不及预期", "业绩预减", "风险", "监管", "处罚", "调查", "诉讼", "债务",
        "违约", "退市", "停产", "裁员", "需求疲软", "竞争加剧", "评级下调", "卖出",
        "看空", "疲软", "低迷", "恶化", "缩减", "终止", "取消", "失败",
    ]

    analyzed = []
    total_score = 0
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for item in news_items:
        title = item.get("title", "")
        content = item.get("content", "")
        text = title + " " + content

        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)

        # 情感得分: -100 到 100
        if pos_count + neg_count > 0:
            score = (pos_count - neg_count) / (pos_count + neg_count) * 100
        else:
            score = 0

        if score > 20:
            sentiment = "正面"
            positive_count += 1
        elif score < -20:
            sentiment = "负面"
            negative_count += 1
        else:
            sentiment = "中性"
            neutral_count += 1

        total_score += score

        analyzed.append({
            "标题": title[:50],
            "情感得分": round(score, 1),
            "情感": sentiment,
            "正面词数": pos_count,
            "负面词数": neg_count,
        })

    n = len(analyzed)
    avg_score = total_score / n if n > 0 else 0

    if avg_score > 30:
        overall = "整体偏正面"
    elif avg_score > -30:
        overall = "整体中性"
    else:
        overall = "整体偏负面"

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "新闻总数": n,
        "平均情感得分": round(avg_score, 1),
        "整体评价": overall,
        "正面新闻": positive_count,
        "负面新闻": negative_count,
        "中性新闻": neutral_count,
        "正面比例": round(positive_count / n * 100, 1) if n > 0 else 0,
        "新闻明细": analyzed,
    }


def hot_sectors_analysis(sector_data):
    """
    热门板块分析

    sector_data: [{"name": "白酒", "change": 3.5, "volume_ratio": 1.5, "leading_stock": "600519"}, ...]
    """
    if not sector_data:
        return {"error": "请提供板块数据"}

    # 按涨跌幅排序
    sorted_data = sorted(sector_data, key=lambda x: x.get("change", 0), reverse=True)

    top_sectors = sorted_data[:10]
    bottom_sectors = sorted_data[-5:][::-1]

    # 资金流向分析
    total_inflow = sum(s.get("change", 0) * s.get("volume_ratio", 1) for s in sector_data)

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "板块数量": len(sector_data),
        "涨幅前10": [{"板块": s["name"], "涨跌幅": s.get("change", 0), "量比": s.get("volume_ratio", 1), "领涨股": s.get("leading_stock", "")} for s in top_sectors],
        "跌幅前5": [{"板块": s["name"], "涨跌幅": s.get("change", 0), "量比": s.get("volume_ratio", 1)} for s in bottom_sectors],
        "资金流向判断": "资金净流入" if total_inflow > 0 else "资金净流出",
    }


def fear_greed_index(market_data):
    """
    恐惧贪婪指数
    综合多个维度计算市场恐惧/贪婪程度

    market_data: {
        "market_change": 1.5,      # 大盘涨跌幅
        "put_call_ratio": 0.8,     # 看跌/看涨比率
        "volatility": 0.2,         # 波动率
        "volume_ratio": 1.2,       # 量比
        "advance_decline": 1.5,    # 涨跌比
        "new_high_low": 2.0,       # 新高/新低比
    }
    """
    if not market_data:
        return {"error": "请提供市场数据"}

    # 各维度得分 (0-100, 越高越贪婪)
    scores = {}

    # 1. 价格动量 (0-100)
    change = market_data.get("market_change", 0)
    scores["价格动量"] = min(100, max(0, 50 + change * 10))

    # 2. 看跌/看涨比率 (0-100, 越低越贪婪)
    pcr = market_data.get("put_call_ratio", 1.0)
    scores["期权情绪"] = min(100, max(0, 100 - pcr * 50))

    # 3. 波动率 (0-100, 越低越贪婪)
    vol = market_data.get("volatility", 0.2)
    scores["波动率"] = min(100, max(0, 100 - vol * 200))

    # 4. 成交量 (0-100)
    vol_ratio = market_data.get("volume_ratio", 1.0)
    scores["成交量"] = min(100, max(0, 50 + (vol_ratio - 1) * 30))

    # 5. 涨跌比 (0-100)
    ad_ratio = market_data.get("advance_decline", 1.0)
    scores["涨跌比"] = min(100, max(0, 50 + (ad_ratio - 1) * 30))

    # 6. 新高新低比 (0-100)
    nhl = market_data.get("new_high_low", 1.0)
    scores["新高新低"] = min(100, max(0, 50 + (nhl - 1) * 20))

    # 加权综合
    weights = {"价格动量": 0.25, "期权情绪": 0.15, "波动率": 0.2, "成交量": 0.15, "涨跌比": 0.15, "新高新低": 0.1}
    total = sum(scores[k] * weights[k] for k in scores)

    if total >= 75:
        level = "极度贪婪"
        desc = "市场情绪极度亢奋，可能接近顶部"
    elif total >= 60:
        level = "贪婪"
        desc = "市场情绪偏乐观"
    elif total >= 40:
        level = "中性"
        desc = "市场情绪平稳"
    elif total >= 25:
        level = "恐惧"
        desc = "市场情绪偏悲观"
    else:
        level = "极度恐惧"
        desc = "市场恐慌，可能接近底部"

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "恐惧贪婪指数": round(total, 1),
        "情绪等级": level,
        "描述": desc,
        "各维度得分": {k: round(v, 1) for k, v in scores.items()},
    }


def get_sentiment_factor(symbol, days=60):
    """
    获取情绪因子评分，供评分系统调用
    返回 -100 到 +100 的情绪因子分数
    """
    result = market_sentiment_index(symbol, days)
    if "error" in result:
        return {"情绪因子": 0, "情绪等级": "未知", "置信度": 0}

    sentiment_raw = result.get("综合情绪指数", 50)
    # 映射: 0-100 -> -100 到 +100
    sentiment_score = (sentiment_raw - 50) * 2

    level = result.get("情绪等级", "中性")
    if level in ("极度乐观", "极度悲观"):
        confidence = 85
    elif level in ("偏乐观", "偏悲观"):
        confidence = 70
    else:
        confidence = 55

    return {
        "情绪因子": round(sentiment_score, 1),
        "情绪等级": level,
        "置信度": confidence,
        "操作建议": result.get("操作建议", ""),
        "分项得分": result.get("分项得分", {}),
    }


def market_sentiment_overview(symbols, days=5):
    """
    市场情绪全景分析
    综合分析多只股票的情绪状态，给出市场整体情绪判断
    """
    if not symbols:
        return {"error": "请提供股票列表"}

    sentiment_summary = []
    levels_count = {"极度乐观": 0, "偏乐观": 0, "中性": 0, "偏悲观": 0, "极度悲观": 0}
    total_sentiment = 0
    valid_count = 0

    for symbol in symbols[:30]:
        result = market_sentiment_index(symbol, days=60)
        if "error" in result:
            continue

        sentiment = result.get("综合情绪指数", 50)
        level = result.get("情绪等级", "中性")
        levels_count[level] = levels_count.get(level, 0) + 1
        total_sentiment += sentiment
        valid_count += 1

        sentiment_summary.append({
            "股票代码": symbol,
            "情绪指数": round(sentiment, 1),
            "情绪等级": level,
        })

    if valid_count == 0:
        return {"error": "无法获取任何股票的情绪数据"}

    avg_sentiment = total_sentiment / valid_count

    # 市场整体情绪判断
    bullish_pct = (levels_count.get("极度乐观", 0) + levels_count.get("偏乐观", 0)) / valid_count * 100
    bearish_pct = (levels_count.get("极度悲观", 0) + levels_count.get("偏悲观", 0)) / valid_count * 100

    if bullish_pct > 60:
        overall = "市场整体偏乐观，多数股票情绪积极"
        risk_warning = "注意情绪过热风险，避免追高"
    elif bearish_pct > 60:
        overall = "市场整体偏悲观，多数股票情绪低迷"
        risk_warning = "关注超跌反弹机会，但需控制仓位"
    elif bullish_pct > bearish_pct:
        overall = "市场情绪分化，偏乐观略占优势"
        risk_warning = "结构性行情，精选个股"
    elif bearish_pct > bullish_pct:
        overall = "市场情绪分化，偏悲观略占优势"
        risk_warning = "防御为主，等待情绪修复"
    else:
        overall = "市场情绪中性，方向不明"
        risk_warning = "观望为主，等待方向明确"

    # 情绪极端股票
    sorted_by_sentiment = sorted(sentiment_summary, key=lambda x: x["情绪指数"], reverse=True)

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "分析股票数": valid_count,
        "平均情绪指数": round(avg_sentiment, 1),
        "整体判断": overall,
        "风险提示": risk_warning,
        "情绪分布": levels_count,
        "看多比例": round(bullish_pct, 1),
        "看空比例": round(bearish_pct, 1),
        "最乐观Top5": sorted_by_sentiment[:5],
        "最悲观Top5": sorted_by_sentiment[-5:][::-1],
    }


def main():
    parser = argparse.ArgumentParser(description="市场情绪/舆情分析")
    subparsers = parser.add_subparsers(dest="action", help="操作")

    # 个股情绪
    stock_parser = subparsers.add_parser("stock", help="个股情绪分析")
    stock_parser.add_argument("--symbol", required=True, help="股票代码")
    stock_parser.add_argument("--days", type=int, default=60, help="分析天数")

    # 市场宽度
    breadth_parser = subparsers.add_parser("breadth", help="市场宽度分析")
    breadth_parser.add_argument("--symbols", required=True, help="股票列表JSON")
    breadth_parser.add_argument("--days", type=int, default=5, help="分析周期")

    # 新闻情感
    news_parser = subparsers.add_parser("news", help="新闻情感分析")
    news_parser.add_argument("--items", required=True, help="新闻列表JSON")

    # 热门板块
    sector_parser = subparsers.add_parser("sector", help="热门板块分析")
    sector_parser.add_argument("--data", required=True, help="板块数据JSON")

    # 恐惧贪婪
    fg_parser = subparsers.add_parser("fear-greed", help="恐惧贪婪指数")
    fg_parser.add_argument("--data", required=True, help="市场数据JSON")

    args = parser.parse_args()

    try:
        if args.action == "stock":
            result = market_sentiment_index(args.symbol, args.days)
        elif args.action == "breadth":
            symbols = json.loads(args.symbols)
            result = market_breadth(symbols, args.days)
        elif args.action == "news":
            items = json.loads(args.items)
            result = sentiment_news_analysis(items)
        elif args.action == "sector":
            data = json.loads(args.data)
            result = hot_sectors_analysis(data)
        elif args.action == "fear-greed":
            data = json.loads(args.data)
            result = fear_greed_index(data)
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
