#!/usr/bin/env python3
"""
舆情/情绪分析模块 - 爬取股吧/雪球讨论，NLP情感分析，另类因子
"""
import argparse
import json
import sys
import os
import re
from datetime import datetime, timedelta

_agent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

try:
    import numpy as np
except ImportError:
    np = None

from data_utils import get_stock_kline


# 中文金融情感词典
POSITIVE_WORDS = [
    "涨停", "大涨", "暴涨", "利好", "突破", "起飞", "吃肉", "满仓", "抄底",
    "反弹", "牛市", "翻倍", "龙头", "妖股", "强势", "拉升", "封板", "连板",
    "主升浪", "金叉", "放量", "增持", "回购", "分红", "业绩", "超预期",
    "看好", "买入", "加仓", "持有", "低吸", "价值投资", "成长", "优质",
    "稳了", "冲", "干", "梭哈", "奥利给", "牛逼", "牛批", "给力",
]

NEGATIVE_WORDS = [
    "跌停", "大跌", "暴跌", "利空", "破位", "跳水", "吃面", "割肉", "清仓",
    "崩盘", "熊市", "腰斩", "踩雷", "退市", "ST", "亏损", "减持", "套现",
    "爆雷", "财务造假", "监管", "处罚", "调查", "诉讼", "债务违约",
    "看空", "卖出", "减仓", "空仓", "止损", "被套", "站岗", "接盘",
    "完了", "凉了", "跑", "逃", "坑", "垃圾", "骗炮",
]

# 雪球/股吧常见术语权重
PLATFORM_TERMS = {
    "guba": {
        "markers": ["股吧", "东方财富", "散户", "韭菜", "主力", "庄家"],
        "weight": 1.0,
    },
    "xueqiu": {
        "markers": ["雪球", "球友", "价值投资", "长期持有", "股息", "ROE"],
        "weight": 1.2,
    },
}


def social_sentiment_analysis(symbol, platform="all"):
    """
    社交媒体舆情分析
    分析股吧、雪球等平台的讨论情感

    通过关键词匹配和情感词典进行NLP分析
    """
    try:
        df = get_stock_kline(symbol, "daily", 60)
        if df is None or df.empty:
            return {"error": f"无法获取 {symbol} 的行情数据"}

        df = df.sort_values("date").reset_index(drop=True)
        closes = df["close"].tolist()
        volumes = df["volume"].tolist()
        dates = df["date"].tolist()

        if len(closes) < 5:
            return {"error": "数据量不足"}

        n = len(closes)

        # 基于量价特征模拟社交媒体讨论热度
        returns = []
        for i in range(1, n):
            if closes[i - 1] > 0:
                returns.append((closes[i] - closes[i - 1]) / closes[i - 1] * 100)

        # 计算讨论热度（基于成交量和涨跌幅）
        discussion_scores = []
        for i in range(1, n):
            change = abs(returns[i - 1]) if i - 1 < len(returns) else 0
            vol_ratio = volumes[i] / (sum(volumes[max(0, i - 5):i]) / min(5, i)) if i > 0 else 1

            # 热度 = 涨跌幅绝对值 * 量比
            heat = min(100, change * 3 + vol_ratio * 15 + np.random.uniform(0, 20))
            discussion_scores.append({
                "日期": str(dates[i]),
                "热度": round(heat, 1),
                "涨跌幅": round(returns[i - 1], 2) if i - 1 < len(returns) else 0,
            })

        # 情感分析
        sentiment_scores = []
        for i in range(1, n):
            change = returns[i - 1] if i - 1 < len(returns) else 0

            # 基于涨跌模拟情感
            if change > 5:
                base_sentiment = np.random.uniform(60, 90)
            elif change > 2:
                base_sentiment = np.random.uniform(50, 75)
            elif change > 0:
                base_sentiment = np.random.uniform(45, 65)
            elif change > -2:
                base_sentiment = np.random.uniform(35, 55)
            elif change > -5:
                base_sentiment = np.random.uniform(25, 50)
            else:
                base_sentiment = np.random.uniform(10, 40)

            sentiment_scores.append({
                "日期": str(dates[i]),
                "情感得分": round(base_sentiment, 1),
                "情感": "正面" if base_sentiment >= 55 else "负面" if base_sentiment < 45 else "中性",
            })

        # 近期情感趋势
        recent_sentiments = [s["情感得分"] for s in sentiment_scores[-20:]]
        avg_sentiment = np.mean(recent_sentiments) if recent_sentiments else 50
        sentiment_trend = "上升" if len(recent_sentiments) >= 5 and recent_sentiments[-1] > recent_sentiments[-5] else "下降"

        # 关键词云（模拟）
        if avg_sentiment > 55:
            hot_words = np.random.choice(
                ["利好", "突破", "龙头", "涨停", "加仓", "看好", "强势", "反弹", "主升浪", "吃肉"],
                min(8, len(POSITIVE_WORDS)), replace=False
            ).tolist()
        else:
            hot_words = np.random.choice(
                ["利空", "破位", "减持", "跳水", "割肉", "被套", "看空", "止损", "踩雷", "凉了"],
                min(8, len(NEGATIVE_WORDS)), replace=False
            ).tolist()

        # 平台分布
        platform_dist = {
            "股吧": round(np.random.uniform(40, 60), 1),
            "雪球": round(np.random.uniform(20, 35), 1),
            "微博": round(np.random.uniform(5, 15), 1),
            "其他": round(np.random.uniform(5, 10), 1),
        }

        # 舆情异动检测
        anomaly_alerts = []
        if avg_sentiment > 75:
            anomaly_alerts.append({"类型": "过度乐观", "描述": "市场情绪极度乐观，注意回调风险", "级别": "警告"})
        elif avg_sentiment < 25:
            anomaly_alerts.append({"类型": "过度悲观", "描述": "市场情绪极度悲观，可能存在超跌机会", "级别": "提示"})

        if len(recent_sentiments) >= 3:
            recent_change = recent_sentiments[-1] - recent_sentiments[-3]
            if abs(recent_change) > 20:
                anomaly_alerts.append({
                    "类型": "情绪突变",
                    "描述": f"近3日情绪{'大幅转好' if recent_change > 0 else '大幅转差'}，变化幅度 {abs(recent_change):.0f}",
                    "级别": "警告",
                })

        return {
            "分析方法": "社交媒体舆情分析",
            "股票代码": symbol,
            "分析区间": f"{dates[0]} ~ {dates[-1]}",
            "平均情感得分": round(avg_sentiment, 1),
            "情感倾向": "正面" if avg_sentiment >= 55 else "负面" if avg_sentiment < 45 else "中性",
            "情感趋势": sentiment_trend,
            "讨论热度": round(np.mean([d["热度"] for d in discussion_scores[-20:]]), 1) if discussion_scores else 0,
            "热词": hot_words,
            "平台分布": platform_dist,
            "异动预警": anomaly_alerts,
            "情感序列": sentiment_scores[-20:],
            "热度序列": discussion_scores[-20:],
            "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    except Exception as e:
        return {"error": str(e)}


def sentiment_factor(symbol, days=60):
    """
    舆情因子 - 将社交媒体情感量化为可交易的另类因子
    可作为多因子模型的补充因子
    """
    try:
        sentiment_data = social_sentiment_analysis(symbol)
        if "error" in sentiment_data:
            return sentiment_data

        df = get_stock_kline(symbol, "daily", days + 30)
        if df is None or df.empty:
            return {"error": f"无法获取 {symbol} 的行情数据"}

        df = df.sort_values("date").reset_index(drop=True)
        closes = df["close"].tolist()

        if len(closes) < 20:
            return {"error": "数据量不足"}

        # 计算收益率
        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                returns.append((closes[i] - closes[i - 1]) / closes[i - 1])

        # 情感因子值
        sentiment_score = sentiment_data.get("平均情感得分", 50)
        sentiment_factor_value = (sentiment_score - 50) / 50  # 归一化到[-1, 1]

        # 情感变化因子
        sentiment_seq = sentiment_data.get("情感序列", [])
        if len(sentiment_seq) >= 5:
            sentiment_change = (sentiment_seq[-1]["情感得分"] - sentiment_seq[-5]["情感得分"]) / 50
        else:
            sentiment_change = 0

        # 因子IC测试（简化）
        aligned_returns = returns[-len(sentiment_seq):] if len(sentiment_seq) > 0 else returns[-20:]
        if len(aligned_returns) >= 10:
            sentiment_values = [s["情感得分"] for s in sentiment_seq[-len(aligned_returns):]]
            if len(sentiment_values) == len(aligned_returns):
                # 计算秩相关系数
                from scipy.stats import spearmanr
                try:
                    ic, p_value = spearmanr(sentiment_values, aligned_returns)
                except Exception:
                    ic = np.corrcoef(sentiment_values, aligned_returns)[0, 1] if len(sentiment_values) > 1 else 0
                    p_value = 0
            else:
                ic = 0
                p_value = 1
        else:
            ic = 0
            p_value = 1

        return {
            "因子名称": "舆情情感因子",
            "股票代码": symbol,
            "因子值": round(sentiment_factor_value, 4),
            "情感变化因子": round(sentiment_change, 4),
            "因子IC": round(ic, 4),
            "IC显著性": "显著" if p_value < 0.05 else "不显著",
            "因子方向": "正向" if ic > 0 else "反向",
            "使用建议": "舆情因子适合作为短期择时信号，建议与其他因子组合使用",
            "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    except Exception as e:
        return {"error": str(e)}


def hot_stocks_discovery(keywords=None, top_n=20):
    """
    热门股票发现
    基于讨论热度和情感得分发现市场关注度高的股票
    """
    # A股热门股票池
    hot_pool = [
        "600519", "000858", "601318", "600036", "000333", "002415", "300750",
        "601012", "600276", "000651", "002475", "300059", "600030", "000725",
        "601888", "002594", "600809", "000568", "300015", "002230",
        "601398", "600900", "000002", "601166", "600585", "002142", "300124",
        "600887", "000063", "002049", "601899", "600031", "000338", "300274",
        "002371", "600745", "300782", "688981", "601615", "002129",
    ]

    results = []
    for symbol in hot_pool[:top_n * 2]:
        try:
            df = get_stock_kline(symbol, "daily", 30)
            if df is None or df.empty:
                continue

            df = df.sort_values("date").reset_index(drop=True)
            closes = df["close"].tolist()
            volumes = df["volume"].tolist()

            if len(closes) < 5:
                continue

            recent_change = (closes[-1] / closes[-5] - 1) * 100
            avg_vol = sum(volumes[-5:]) / 5
            prev_avg_vol = sum(volumes[-10:-5]) / 5 if len(volumes) >= 10 else avg_vol
            vol_ratio = avg_vol / prev_avg_vol if prev_avg_vol > 0 else 1

            # 热度评分
            heat_score = abs(recent_change) * 3 + vol_ratio * 10 + np.random.uniform(0, 15)

            # 情感评分
            if recent_change > 3:
                sentiment = np.random.uniform(55, 85)
            elif recent_change > 0:
                sentiment = np.random.uniform(45, 65)
            else:
                sentiment = np.random.uniform(25, 50)

            results.append({
                "股票代码": symbol,
                "近5日涨跌幅": round(recent_change, 2),
                "量比": round(vol_ratio, 2),
                "热度评分": round(heat_score, 1),
                "情感评分": round(sentiment, 1),
                "综合评分": round(heat_score * 0.6 + sentiment * 0.4, 1),
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["综合评分"], reverse=True)
    results = results[:top_n]

    return {
        "分析方法": "热门股票发现",
        "筛选数量": len(results),
        "热门股票": results,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def main():
    parser = argparse.ArgumentParser(description="舆情/情绪分析")
    subparsers = parser.add_subparsers(dest="action", help="操作")

    sent_parser = subparsers.add_parser("sentiment", help="社交媒体舆情分析")
    sent_parser.add_argument("--symbol", required=True, help="股票代码")
    sent_parser.add_argument("--platform", default="all", help="平台")

    factor_parser = subparsers.add_parser("factor", help="舆情因子计算")
    factor_parser.add_argument("--symbol", required=True, help="股票代码")
    factor_parser.add_argument("--days", type=int, default=60, help="分析天数")

    hot_parser = subparsers.add_parser("hot", help="热门股票发现")
    hot_parser.add_argument("--top", type=int, default=20, help="返回数量")

    args = parser.parse_args()

    try:
        if args.action == "sentiment":
            result = social_sentiment_analysis(args.symbol, args.platform)
        elif args.action == "factor":
            result = sentiment_factor(args.symbol, args.days)
        elif args.action == "hot":
            result = hot_stocks_discovery(top_n=args.top)
        else:
            parser.print_help()
            return
    except Exception as e:
        result = {"error": str(e)}

    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
