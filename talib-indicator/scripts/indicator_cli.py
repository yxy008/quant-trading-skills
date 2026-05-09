#!/usr/bin/env python3
"""
技术指标计算工具 - 基于 TA-Lib 和 AkShare
"""
import argparse
import json
import sys
import os
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

from data_utils import get_stock_kline as _get_stock_kline

HAS_TALIB = False
try:
    import talib
    HAS_TALIB = True
except ImportError:
    print("警告: TA-Lib 未安装，将使用纯 Python 实现")


def get_stock_kline(symbol, period='daily', days=100):
    """
    获取股票K线数据，使用前复权保证价格连续性
    """
    df = _get_stock_kline(symbol, days=days)

    if df is None or df.empty:
        return None

    df = df.reset_index()
    df['日期'] = df['日期'].astype(str)
    
    return df


def calculate_sma_python(close, timeperiod=5):
    """
    纯 Python 实现的 SMA
    """
    sma = []
    for i in range(len(close)):
        if i < timeperiod - 1:
            sma.append(None)
        else:
            sma.append(np.mean(close[i - timeperiod + 1:i + 1]))
    return sma


def calculate_sma(df, timeperiod=5):
    """
    计算简单移动平均线 (SMA)
    """
    close = df['收盘'].values.astype(np.float64)
    if HAS_TALIB:
        sma = talib.SMA(close, timeperiod=timeperiod)
        return sma.tolist()
    else:
        return calculate_sma_python(close, timeperiod)


def calculate_ema_python(close, timeperiod=12):
    """
    纯 Python 实现的 EMA
    """
    ema = []
    multiplier = 2 / (timeperiod + 1)
    for i in range(len(close)):
        if i < timeperiod - 1:
            ema.append(None)
        elif i == timeperiod - 1:
            ema.append(np.mean(close[:timeperiod]))
        else:
            ema.append((close[i] - ema[i-1]) * multiplier + ema[i-1])
    return ema


def calculate_ema(df, timeperiod=12):
    """
    计算指数移动平均线 (EMA)
    """
    close = df['收盘'].values.astype(np.float64)
    if HAS_TALIB:
        ema = talib.EMA(close, timeperiod=timeperiod)
        return ema.tolist()
    else:
        return calculate_ema_python(close, timeperiod)


def calculate_macd_python(close, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    纯 Python 实现的 MACD
    """
    ema_fast = calculate_ema_python(close, fastperiod)
    ema_slow = calculate_ema_python(close, slowperiod)
    
    macd_line = []
    for i in range(len(close)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line.append(ema_fast[i] - ema_slow[i])
        else:
            macd_line.append(None)
    
    signal_line = []
    valid_macd = [x for x in macd_line if x is not None]
    if len(valid_macd) >= signalperiod:
        ema_signal = calculate_ema_python(np.array([x for x in macd_line if x is not None]), signalperiod)
        # 对齐长度
        signal_line = [None] * (len(macd_line) - len(ema_signal)) + ema_signal
    else:
        signal_line = [None] * len(macd_line)
    
    histogram = []
    for i in range(len(macd_line)):
        if macd_line[i] is not None and signal_line[i] is not None:
            histogram.append(macd_line[i] - signal_line[i])
        else:
            histogram.append(None)
    
    return {
        'macd': macd_line,
        'macdsignal': signal_line,
        'macdhist': histogram
    }


def calculate_macd(df, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    计算 MACD
    """
    close = df['收盘'].values.astype(np.float64)
    if HAS_TALIB:
        macd, macdsignal, macdhist = talib.MACD(
            close, 
            fastperiod=fastperiod, 
            slowperiod=slowperiod, 
            signalperiod=signalperiod
        )
        return {
            'macd': macd.tolist(),
            'macdsignal': macdsignal.tolist(),
            'macdhist': macdhist.tolist()
        }
    else:
        return calculate_macd_python(close, fastperiod, slowperiod, signalperiod)


def calculate_rsi_python(close, timeperiod=14):
    """
    纯 Python 实现的 RSI
    """
    rsi = []
    deltas = np.diff(close)
    gains = []
    losses = []
    
    for i in range(len(close)):
        if i < timeperiod:
            gains.append(None)
        else:
            if i == timeperiod:
                period_deltas = deltas[:timeperiod]
                avg_gain = np.mean([x for x in period_deltas if x > 0])
                avg_loss = -np.mean([x for x in period_deltas if x < 0])
            else:
                delta = deltas[i-1]
                gain = max(delta, 0)
                loss = -min(delta, 0)
                avg_gain = (avg_gain * (timeperiod - 1) + gain) / timeperiod
                avg_loss = (avg_loss * (timeperiod - 1) + loss) / timeperiod
            
            if avg_loss == 0:
                rsi_val = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_val = 100.0 - (100.0 / (1.0 + rs))
            rsi.append(rsi_val)
    
    return [None] * timeperiod + rsi


def calculate_rsi(df, timeperiod=14):
    """
    计算相对强弱指标 (RSI)
    """
    close = df['收盘'].values.astype(np.float64)
    if HAS_TALIB:
        rsi = talib.RSI(close, timeperiod=timeperiod)
        return rsi.tolist()
    else:
        return calculate_rsi_python(close, timeperiod)


def calculate_atr_python(high, low, close, timeperiod=14):
    """
    纯 Python 实现的 ATR
    """
    tr = []
    for i in range(len(close)):
        if i == 0:
            tr_val = high[i] - low[i]
        else:
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr_val = max(tr1, tr2, tr3)
        tr.append(tr_val)
    
    atr = []
    for i in range(len(tr)):
        if i < timeperiod - 1:
            atr.append(None)
        elif i == timeperiod - 1:
            atr.append(np.mean(tr[:timeperiod]))
        else:
            atr.append((atr[i-1] * (timeperiod - 1) + tr[i]) / timeperiod)
    return atr


def calculate_atr(df, timeperiod=14):
    """
    计算平均真实波幅 (ATR)
    """
    high = df['最高'].values.astype(np.float64)
    low = df['最低'].values.astype(np.float64)
    close = df['收盘'].values.astype(np.float64)
    if HAS_TALIB:
        atr = talib.ATR(high, low, close, timeperiod=timeperiod)
        return atr.tolist()
    else:
        return calculate_atr_python(high, low, close, timeperiod)


def calculate_bbands_python(close, timeperiod=20, nbdevup=2, nbdevdn=2):
    """
    纯 Python 实现的 Bollinger Bands
    """
    middle = calculate_sma_python(close, timeperiod)
    
    upper = []
    lower = []
    for i in range(len(close)):
        if i < timeperiod - 1:
            upper.append(None)
            lower.append(None)
        else:
            std = np.std(close[i - timeperiod + 1:i + 1])
            upper.append(middle[i] + nbdevup * std)
            lower.append(middle[i] - nbdevdn * std)
    
    return {
        'upperband': upper,
        'middleband': middle,
        'lowerband': lower
    }


def calculate_bbands(df, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
    """
    计算布林带 (BBANDS)
    """
    close = df['收盘'].values.astype(np.float64)
    if HAS_TALIB:
        upperband, middleband, lowerband = talib.BBANDS(
            close, 
            timeperiod=timeperiod, 
            nbdevup=nbdevup, 
            nbdevdn=nbdevdn, 
            matype=matype
        )
        return {
            'upperband': upperband.tolist(),
            'middleband': middleband.tolist(),
            'lowerband': lowerband.tolist()
        }
    else:
        return calculate_bbands_python(close, timeperiod, nbdevup, nbdevdn)



def calculate_all_indicators(df):
    """
    计算所有常用指标
    """
    indicators = {}
    indicators['sma_5'] = calculate_sma(df, 5)
    indicators['sma_10'] = calculate_sma(df, 10)
    indicators['sma_20'] = calculate_sma(df, 20)
    indicators['ema_12'] = calculate_ema(df, 12)
    indicators['ema_26'] = calculate_ema(df, 26)
    indicators['macd'] = calculate_macd(df)
    indicators['rsi_14'] = calculate_rsi(df, 14)
    indicators['atr_14'] = calculate_atr(df, 14)
    indicators['bbands'] = calculate_bbands(df)
    return indicators


# ==================== K线形态识别 ====================

CANDLESTICK_PATTERNS = {
    "CDL2CROWS": "两只乌鸦",
    "CDL3BLACKCROWS": "三只乌鸦",
    "CDL3INSIDE": "三内部上涨下跌",
    "CDL3LINESTRIKE": "三线打击",
    "CDL3OUTSIDE": "三外部上涨下跌",
    "CDL3STARSINSOUTH": "南方三星",
    "CDL3WHITESOLDIERS": "三个白兵",
    "CDLABANDONEDBABY": "弃婴",
    "CDLADVANCEBLOCK": "大敌当前",
    "CDLBELTHOLD": "捉腰带线",
    "CDLBREAKAWAY": "脱离",
    "CDLCLOSINGMARUBOZU": "收盘缺影线",
    "CDLCONCEALBABYSWALL": "藏婴吞没",
    "CDLCOUNTERATTACK": "反击线",
    "CDLDARKCLOUDCOVER": "乌云盖顶",
    "CDLDOJI": "十字星",
    "CDLDOJISTAR": "十字星",
    "CDLDRAGONFLYDOJI": "蜻蜓十字",
    "CDLENGULFING": "吞没形态",
    "CDLEVENINGDOJISTAR": "黄昏十字星",
    "CDLEVENINGSTAR": "黄昏之星",
    "CDLGAPSIDESIDEWHITE": "向上跳空并列阳线",
    "CDLGRAVESTONEDOJI": "墓碑十字",
    "CDLHAMMER": "锤子线",
    "CDLHANGINGMAN": "上吊线",
    "CDLHARAMI": "孕线",
    "CDLHARAMICROSS": "十字孕线",
    "CDLHIGHWAVE": "风高浪大线",
    "CDLHIKKAKE": "陷阱",
    "CDLHIKKAKEMOD": "修正陷阱",
    "CDLHOMINGPIGEON": "家鸽",
    "CDLIDENTICAL3CROWS": "三胞胎乌鸦",
    "CDLINNECK": "颈内线",
    "CDLINVERTEDHAMMER": "倒锤头",
    "CDLKICKING": "反冲形态",
    "CDLKICKINGBYLENGTH": "由较长缺影线决定的反冲形态",
    "CDLLADDERBOTTOM": "梯底",
    "CDLLONGLEGGEDDOJI": "长脚十字",
    "CDLLONGLINE": "长蜡烛",
    "CDLMARUBOZU": "光头光脚",
    "CDLMATCHINGLOW": "相同低价",
    "CDLMATHOLD": "铺垫",
    "CDLMORNINGDOJISTAR": "启明十字星",
    "CDLMORNINGSTAR": "启明星",
    "CDLONNECK": "颈上线",
    "CDLPIERCING": "刺透形态",
    "CDLRICKSHAWMAN": "黄包车夫",
    "CDLRISEFALL3METHODS": "上升下降三法",
    "CDLSEPARATINGLINES": "分离线",
    "CDLSHOOTINGSTAR": "射击之星",
    "CDLSHORTLINE": "短蜡烛",
    "CDLSPINNINGTOP": "纺锤",
    "CDLSTALLEDPATTERN": "停顿形态",
    "CDLSTICKSANDWICH": "条形三明治",
    "CDLTAKURI": "探水竿",
    "CDLTASUKIGAP": "跳空并列线",
    "CDLTHRUSTING": "插入",
    "CDLTRISTAR": "三星",
    "CDLUNIQUE3RIVER": "奇特三河床",
    "CDLUPSIDEGAP2CROWS": "向上跳空两只乌鸦",
    "CDLXSIDEGAP3METHODS": "向上跳空三法",
}

# 看涨形态
BULLISH_PATTERNS = [
    "CDL3WHITESOLDIERS", "CDL3INSIDE", "CDL3OUTSIDE",
    "CDLMORNINGSTAR", "CDLMORNINGDOJISTAR", "CDLABANDONEDBABY",
    "CDLPIERCING", "CDLENGULFING", "CDLHAMMER",
    "CDLINVERTEDHAMMER", "CDLDRAGONFLYDOJI", "CDLBELTHOLD",
    "CDLHARAMI", "CDLHARAMICROSS", "CDLHOMINGPIGEON",
    "CDLLADDERBOTTOM", "CDLMATCHINGLOW", "CDLUNIQUE3RIVER",
    "CDLRISEFALL3METHODS", "CDLCOUNTERATTACK", "CDLSTICKSANDWICH",
    "CDLTAKURI", "CDLKICKING", "CDLKICKINGBYLENGTH",
    "CDLCONCEALBABYSWALL", "CDLGAPSIDESIDEWHITE",
]

# 看跌形态
BEARISH_PATTERNS = [
    "CDL3BLACKCROWS", "CDLDARKCLOUDCOVER", "CDLEVENINGSTAR",
    "CDLEVENINGDOJISTAR", "CDLSHOOTINGSTAR", "CDLHANGINGMAN",
    "CDLGRAVESTONEDOJI", "CDL2CROWS", "CDLADVANCEBLOCK",
    "CDLSTALLEDPATTERN", "CDLIDENTICAL3CROWS", "CDL3STARSINSOUTH",
    "CDLUPSIDEGAP2CROWS", "CDLONNECK", "CDLINNECK",
    "CDLTHRUSTING", "CDLBREAKAWAY",
]


def detect_patterns_python(open_p, high, low, close):
    """
    纯Python实现的K线形态识别
    识别常见的K线形态，不依赖TA-Lib
    """
    n = len(close)
    patterns = {name: [0] * n for name in CANDLESTICK_PATTERNS.values()}

    for i in range(2, n):
        o1, h1, l1, c1 = open_p[i-2], high[i-2], low[i-2], close[i-2]
        o2, h2, l2, c2 = open_p[i-1], high[i-1], low[i-1], close[i-1]
        o3, h3, l3, c3 = open_p[i], high[i], low[i], close[i]

        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        body3 = abs(c3 - o3)
        upper_shadow3 = h3 - max(o3, c3)
        lower_shadow3 = min(o3, c3) - l3
        total_range3 = h3 - l3

        # 十字星 (Doji)
        if total_range3 > 0 and body3 <= total_range3 * 0.1:
            patterns["十字星"][i] = 1

        # 蜻蜓十字 (Dragonfly Doji)
        if total_range3 > 0 and body3 <= total_range3 * 0.1 and lower_shadow3 >= total_range3 * 0.6:
            patterns["蜻蜓十字"][i] = 1

        # 墓碑十字 (Gravestone Doji)
        if total_range3 > 0 and body3 <= total_range3 * 0.1 and upper_shadow3 >= total_range3 * 0.6:
            patterns["墓碑十字"][i] = 1

        # 长脚十字 (Long-legged Doji)
        if total_range3 > 0 and body3 <= total_range3 * 0.1 and upper_shadow3 > body3 * 2 and lower_shadow3 > body3 * 2:
            patterns["长脚十字"][i] = 1

        # 锤子线 (Hammer)
        if body3 > 0 and lower_shadow3 >= body3 * 2 and upper_shadow3 <= body3 * 0.3:
            # 需要处于下跌趋势中
            if c2 < o2 and c1 < o1:
                patterns["锤子线"][i] = 1

        # 上吊线 (Hanging Man)
        if body3 > 0 and lower_shadow3 >= body3 * 2 and upper_shadow3 <= body3 * 0.3:
            if c2 > o2 and c1 > o1:
                patterns["上吊线"][i] = -1

        # 倒锤头 (Inverted Hammer)
        if body3 > 0 and upper_shadow3 >= body3 * 2 and lower_shadow3 <= body3 * 0.3:
            if c2 < o2:
                patterns["倒锤头"][i] = 1

        # 射击之星 (Shooting Star)
        if body3 > 0 and upper_shadow3 >= body3 * 2 and lower_shadow3 <= body3 * 0.3:
            if c2 > o2:
                patterns["射击之星"][i] = -1

        # 吞没形态 (Engulfing)
        if c2 < o2 and c3 > o3 and o3 <= c2 and c3 >= o2:
            patterns["吞没形态"][i] = 1  # 看涨吞没
        elif c2 > o2 and c3 < o3 and o3 >= c2 and c3 <= o2:
            patterns["吞没形态"][i] = -1  # 看跌吞没

        # 孕线 (Harami)
        if c2 > o2 and c3 < o3 and o3 >= c2 and c3 <= o2:
            patterns["孕线"][i] = -1
        elif c2 < o2 and c3 > o3 and o3 <= c2 and c3 >= o2:
            patterns["孕线"][i] = 1

        # 刺透形态 (Piercing)
        if c2 < o2 and c3 > o3 and o3 < c2 and c3 > (o2 + c2) / 2 and c3 < o2:
            patterns["刺透形态"][i] = 1

        # 乌云盖顶 (Dark Cloud Cover)
        if c2 > o2 and c3 < o3 and o3 > c2 and c3 < (o2 + c2) / 2 and c3 > o2:
            patterns["乌云盖顶"][i] = -1

        # 启明星 (Morning Star) - 需要3根K线
        if c1 < o1 and body2 <= total_range3 * 0.3 and c3 > o3 and c3 > (o1 + c1) / 2:
            patterns["启明星"][i] = 1

        # 黄昏之星 (Evening Star)
        if c1 > o1 and body2 <= total_range3 * 0.3 and c3 < o3 and c3 < (o1 + c1) / 2:
            patterns["黄昏之星"][i] = -1

        # 三个白兵 (Three White Soldiers)
        if c1 > o1 and c2 > o2 and c3 > o3 and c2 > c1 and c3 > c2:
            patterns["三个白兵"][i] = 1

        # 三只乌鸦 (Three Black Crows)
        if c1 < o1 and c2 < o2 and c3 < o3 and c2 < c1 and c3 < c2:
            patterns["三只乌鸦"][i] = -1

        # 光头光脚 (Marubozu)
        if body3 > 0 and upper_shadow3 <= body3 * 0.1 and lower_shadow3 <= body3 * 0.1:
            if c3 > o3:
                patterns["光头光脚"][i] = 1
            else:
                patterns["光头光脚"][i] = -1

        # 纺锤 (Spinning Top)
        if body3 > 0 and total_range3 > 0 and body3 <= total_range3 * 0.3 and upper_shadow3 > body3 and lower_shadow3 > body3:
            patterns["纺锤"][i] = 0

    return patterns


def detect_patterns(df):
    """
    识别K线形态
    优先使用TA-Lib，否则使用纯Python实现
    """
    open_p = df['开盘'].values.astype(np.float64)
    high = df['最高'].values.astype(np.float64)
    low = df['最低'].values.astype(np.float64)
    close = df['收盘'].values.astype(np.float64)

    if HAS_TALIB:
        patterns = {}
        for talib_name, cn_name in CANDLESTICK_PATTERNS.items():
            func = getattr(talib, talib_name, None)
            if func:
                result = func(open_p, high, low, close)
                patterns[cn_name] = result.tolist()
        return patterns
    else:
        return detect_patterns_python(open_p, high, low, close)


def find_recent_patterns(df, days=20):
    """
    查找最近的K线形态
    返回最近出现的形态及其位置和含义
    """
    patterns = detect_patterns(df)
    if not patterns:
        return {"error": "形态识别失败"}

    close_col = '收盘' if '收盘' in df.columns else 'close'
    date_col = '日期' if '日期' in df.columns else 'date'
    current_price = float(df[close_col].iloc[-1])

    found = []
    n = len(df)

    for name, signals in patterns.items():
        for i in range(max(0, n - days), n):
            if signals[i] != 0:
                date_str = str(df[date_col].iloc[i])[:10] if date_col in df.columns else f"T-{n-i-1}"
                price = float(df[close_col].iloc[i])

                if signals[i] == 1:
                    direction = "看涨"
                elif signals[i] == -1:
                    direction = "看跌"
                else:
                    direction = "中性"

                found.append({
                    "形态": name,
                    "日期": date_str,
                    "位置": f"倒数第{n-i}根K线",
                    "当时价格": round(price, 2),
                    "方向": direction,
                })

    # 按日期排序（最近的在前）
    found.sort(key=lambda x: x["位置"])

    # 统计
    bullish_count = sum(1 for f in found if f["方向"] == "看涨")
    bearish_count = sum(1 for f in found if f["方向"] == "看跌")

    if bullish_count > bearish_count:
        overall = "近期看涨形态居多"
    elif bearish_count > bullish_count:
        overall = "近期看跌形态居多"
    else:
        overall = "近期形态信号中性"

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "当前价格": round(current_price, 2),
        "分析范围": f"最近{days}个交易日",
        "发现形态数": len(found),
        "看涨形态数": bullish_count,
        "看跌形态数": bearish_count,
        "综合判断": overall,
        "形态列表": found[:20],
    }


def pattern_signal_score(df):
    """
    形态信号评分
    综合所有形态信号，给出-100到+100的评分
    """
    patterns = detect_patterns(df)
    if not patterns:
        return {"error": "形态识别失败"}

    n = len(df)
    score = 0
    details = []

    # 只看最近5根K线
    lookback = min(5, n)
    for i in range(n - lookback, n):
        for name, signals in patterns.items():
            if signals[i] != 0:
                weight = 1.0
                # 重要形态权重更高
                if name in ("启明星", "黄昏之星", "吞没形态", "三个白兵", "三只乌鸦"):
                    weight = 2.0
                elif name in ("锤子线", "上吊线", "射击之星", "乌云盖顶"):
                    weight = 1.5

                contribution = signals[i] * weight * 20
                score += contribution
                details.append({
                    "形态": name,
                    "信号": signals[i],
                    "权重": weight,
                    "贡献": round(contribution, 1),
                })

    score = max(-100, min(100, score))

    if score >= 50:
        level = "强烈看涨"
    elif score >= 20:
        level = "偏看涨"
    elif score > -20:
        level = "中性"
    elif score > -50:
        level = "偏看跌"
    else:
        level = "强烈看跌"

    return {
        "形态评分": round(score, 1),
        "信号等级": level,
        "形态明细": details,
    }


def main():
    parser = argparse.ArgumentParser(description='技术指标计算工具')
    parser.add_argument('indicator', choices=['sma', 'ema', 'macd', 'rsi', 'atr', 'bbands', 'all'],
                        help='技术指标类型')
    parser.add_argument('--symbol', required=True, help='股票代码')
    parser.add_argument('--period', default='daily', choices=['daily', 'weekly', 'monthly'])
    parser.add_argument('--days', type=int, default=100, help='获取数据的天数')
    parser.add_argument('--timeperiod', type=int, help='时间周期参数')
    
    args = parser.parse_args()
    
    try:
        df = get_stock_kline(args.symbol, args.period, args.days)
        
        if df.empty:
            print(json.dumps({'error': '未获取到数据'}, ensure_ascii=False, indent=2))
            sys.exit(1)
        
        result = {
            'symbol': args.symbol,
            'period': args.period,
            'dates': df['日期'].tolist(),
            'close': df['收盘'].tolist(),
            'high': df['最高'].tolist(),
            'low': df['最低'].tolist(),
            'volume': df['成交量'].tolist()
        }
        
        if args.indicator == 'sma':
            timeperiod = args.timeperiod if args.timeperiod else 5
            result['sma'] = calculate_sma(df, timeperiod)
            result['timeperiod'] = timeperiod
        elif args.indicator == 'ema':
            timeperiod = args.timeperiod if args.timeperiod else 12
            result['ema'] = calculate_ema(df, timeperiod)
            result['timeperiod'] = timeperiod
        elif args.indicator == 'macd':
            result['macd'] = calculate_macd(df)
        elif args.indicator == 'rsi':
            timeperiod = args.timeperiod if args.timeperiod else 14
            result['rsi'] = calculate_rsi(df, timeperiod)
            result['timeperiod'] = timeperiod
        elif args.indicator == 'atr':
            timeperiod = args.timeperiod if args.timeperiod else 14
            result['atr'] = calculate_atr(df, timeperiod)
            result['timeperiod'] = timeperiod
        elif args.indicator == 'bbands':
            timeperiod = args.timeperiod if args.timeperiod else 20
            result['bbands'] = calculate_bbands(df, timeperiod)
            result['timeperiod'] = timeperiod
        elif args.indicator == 'all':
            result['indicators'] = calculate_all_indicators(df)
        
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(json.dumps({'error': str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == '__main__':
    main()
