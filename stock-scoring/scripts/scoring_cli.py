#!/usr/bin/env python3
"""
股票综合评分工具 - stock-scoring (全真实数据版)
所有评分完全基于真实 K线数据和真实市场数据
不包含任何虚拟模拟的基本面数据
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

from data_utils import get_stock_kline, _get_spot_df


# 板块行业映射（仅用于板块分类，不包含虚拟数据）
STOCK_INDUSTRY = {
    "600519": "白酒",
    "000858": "白酒",
    "000568": "白酒",
    "600809": "白酒",
    "000001": "银行",
    "600036": "银行",
    "601318": "银行",
    "601398": "银行",
    "601939": "银行",
    "002594": "新能源",
    "300750": "新能源",
    "002466": "新能源",
    "300014": "新能源",
    "688981": "半导体",
    "603986": "半导体",
    "600887": "消费",
    "000333": "消费",
    "000651": "消费"
}


def calculate_rsi(series, period=14):
    """计算 RSI 指标"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series, fast=12, slow=26, signal=9):
    """计算 MACD 指标"""
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_technical_score(df_kline, symbol):
    """
    基于真实 K线数据 计算技术面评分（0-40）
    :param df_kline: K线数据
    :param symbol: 股票代码
    :return: dict
    """
    score = 0
    details = []
    
    if df_kline is None or len(df_kline) < 30:
        return {"得分": 0, "说明": "K线数据不足，无法计算技术面评分", "详情": []}
    
    close_prices = df_kline['收盘'] if '收盘' in df_kline.columns else df_kline['close']
    latest = close_prices.iloc[-1]
    
    # 1. 均线分析（10分）
    sma5 = close_prices.rolling(5).mean().iloc[-1]
    sma10 = close_prices.rolling(10).mean().iloc[-1]
    sma20 = close_prices.rolling(20).mean().iloc[-1]
    sma60 = close_prices.rolling(40).mean().iloc[-1] if len(close_prices) >=40 else sma20
    
    if not pd.isna(sma5) and not pd.isna(sma20):
        if latest > sma5 > sma20:
            details.append("均线排列：多头排列，+10分")
            score +=10
        elif latest < sma5 < sma20:
            details.append("均线排列：空头排列，+0分")
        elif latest > sma20:
            details.append("均线排列：价格在20日线上方，+5分")
            score +=5
        else:
            details.append("均线排列：价格在20日线下方，+2分")
            score +=2
    
    # 2. RSI 分析（10分）
    rsi_series = calculate_rsi(close_prices)
    rsi = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50
    
    if 30 < rsi <70:
        details.append(f"RSI：{round(rsi,1)}，健康区域，+10分")
        score +=10
    elif 20 <= rsi <=30:
        details.append(f"RSI：{round(rsi,1)}，超卖区域，+7分")
        score +=7
    elif 70 <= rsi <=80:
        details.append(f"RSI：{round(rsi,1)}，超买区域，+6分")
        score +=6
    else:
        details.append(f"RSI：{round(rsi,1)}，极端区域，+3分")
        score +=3
    
    #3. MACD 分析（10分）
    macd_line, signal_line, hist = calculate_macd(close_prices)
    macd = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0
    sig = signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0
    latest_hist = hist.iloc[-1] if not pd.isna(hist.iloc[-1]) else 0
    prev_hist = hist.iloc[-2] if len(hist)>=2 and not pd.isna(hist.iloc[-2]) else latest_hist
    
    if latest_hist >0 and latest_hist > prev_hist:
        details.append("MACD：动能向上，+10分")
        score +=10
    elif latest_hist >0:
        details.append("MACD：正值，+7分")
        score +=7
    elif latest_hist <0 and latest_hist < prev_hist:
        details.append("MACD：动能向下，+2分")
        score +=2
    else:
        details.append("MACD：负值，+4分")
        score +=4
    
    #4. 位置与波动率（10分）
    max_60 = close_prices.tail(40).max() if len(close_prices)>=40 else close_prices.max()
    min_60 = close_prices.tail(40).min() if len(close_prices)>=40 else close_prices.min()
    
    # 计算波动率
    returns = close_prices.pct_change().dropna()
    volatility = returns.std() *100 if len(returns)>=10 else 2
    
    if (max_60 - min_60) >0:
        position = (latest - min_60) / (max_60 - min_60)
    else:
        position =0.5
    
    if 0.3 < position <0.7:
        details.append("位置：价格位置适中，+10分")
        score +=10
    elif position <0.3:
        details.append("位置：价格接近低位，+6分")
        score +=6
    else:
        details.append("位置：价格接近高位，+5分")
        score +=5
    
    return {
        "得分": min(score,40),
        "说明": "技术面综合评分（基于真实K线）",
        "详情": details,
        "最新价": round(float(latest),2),
        "RSI": round(float(rsi),1),
        "MACD": round(float(latest_hist),3)
    }


def calculate_risk_score(df_kline, symbol):
    """
    基于真实波动率 计算风险评分（0-20，越高越安全）
    """
    if df_kline is None or len(df_kline) <20:
        return {"得分":10,"说明":"数据不足，默认风险评分","详情":["数据不足"],"波动率":0}
    
    close_prices = df_kline['收盘'] if '收盘' in df_kline.columns else df_kline['close']
    returns = close_prices.pct_change().dropna()
    volatility = returns.std() *100
    
    score =0
    details = []
    
    if volatility <1.5:
        score +=10
        details.append("波动率：极低，+10分")
    elif volatility <2.5:
        score +=8
        details.append("波动率：较低，+8分")
    elif volatility <4:
        score +=6
        details.append("波动率：中等，+6分")
    else:
        score +=4
        details.append("波动率：较高，+4分")
    
    score +=10  # 基础分
    
    return {
        "得分": min(score,20),
        "说明": "风险评分（越高越安全）",
        "详情": details,
        "波动率": round(volatility,2)
    }


# 板块动态评分缓存
_SECTOR_SCORE_CACHE = {}
_SECTOR_SCORE_CACHE_TIME = None
_SECTOR_SCORE_CACHE_TTL = 600


def _get_sector_performance():
    """
    获取所有行业板块的近期表现数据（带缓存）
    基于东方财富行业板块数据
    """
    global _SECTOR_SCORE_CACHE, _SECTOR_SCORE_CACHE_TIME
    now = time.time()
    if _SECTOR_SCORE_CACHE and _SECTOR_SCORE_CACHE_TIME:
        if now - _SECTOR_SCORE_CACHE_TIME < _SECTOR_SCORE_CACHE_TTL:
            return _SECTOR_SCORE_CACHE

    try:
        import akshare as ak
        df = ak.stock_board_industry_name_em()
        if df is not None and not df.empty:
            result = {}
            for _, row in df.iterrows():
                name = str(row.get('板块名称', ''))
                change_pct = float(row.get('涨跌幅', 0)) if pd.notna(row.get('涨跌幅')) else 0
                result[name] = {
                    '涨跌幅': change_pct,
                    '最新价': float(row.get('最新价', 0)) if pd.notna(row.get('最新价')) else 0,
                }
            _SECTOR_SCORE_CACHE = result
            _SECTOR_SCORE_CACHE_TIME = now
            return result
    except Exception:
        pass

    return _SECTOR_SCORE_CACHE if _SECTOR_SCORE_CACHE else {}


def _get_sector_trend_score(sector_name):
    """
    获取板块趋势评分（基于板块指数K线）
    返回0-10的动态评分
    """
    try:
        import akshare as ak
        df = ak.stock_board_industry_index_em(symbol=sector_name)
        if df is not None and not df.empty and len(df) >= 20:
            close = df['收盘'] if '收盘' in df.columns else df['close']
            latest = close.iloc[-1]
            ma5 = close.rolling(5).mean().iloc[-1]
            ma10 = close.rolling(10).mean().iloc[-1]
            ma20 = close.rolling(20).mean().iloc[-1]

            change_5d = (latest / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
            change_10d = (latest / close.iloc[-10] - 1) * 100 if len(close) >= 10 else 0

            trend_score = 5
            if latest > ma5 > ma10 > ma20:
                trend_score += 3
            elif latest > ma5 > ma10:
                trend_score += 2
            elif latest > ma20:
                trend_score += 1

            if change_5d > 3:
                trend_score += 1
            if change_10d > 5:
                trend_score += 1

            return min(trend_score, 10), {
                '5日涨跌幅': round(change_5d, 2),
                '10日涨跌幅': round(change_10d, 2),
                '均线状态': '多头' if latest > ma5 > ma10 > ma20 else ('偏多' if latest > ma20 else '偏弱')
            }
    except Exception:
        pass

    return 5, {}


def calculate_multi_factor_score(symbol, df_kline=None):
    """
    多因子评分（0-20）
    基于多因子模型计算因子暴露并合成评分
    """
    if df_kline is None:
        df_kline = get_stock_kline(symbol, days=120)

    if df_kline is None or len(df_kline) < 30:
        return {"得分": 10, "说明": "数据不足，使用基础因子评分", "详情": [], "因子值": {}}

    close = df_kline['收盘'] if '收盘' in df_kline.columns else df_kline['close']
    high = df_kline['最高'] if '最高' in df_kline.columns else df_kline.get('high', close)
    low = df_kline['最低'] if '最低' in df_kline.columns else df_kline.get('low', close)
    volume = df_kline.get('成交量', df_kline.get('volume', df_kline.get('成交额', df_kline.get('amount', 0)) / close))

    factor_values = {}
    details = []
    score = 10

    # 动量因子
    momentum_20 = (close.iloc[-1] / close.iloc[-min(20, len(close))] - 1) * 100
    factor_values['20日动量'] = round(momentum_20, 2)
    if momentum_20 > 10:
        score += 3
        details.append(f"20日动量{momentum_20:.1f}%，强势，+3分")
    elif momentum_20 > 3:
        score += 2
        details.append(f"20日动量{momentum_20:.1f}%，偏强，+2分")
    elif momentum_20 > 0:
        score += 1
        details.append(f"20日动量{momentum_20:.1f}%，微涨，+1分")
    elif momentum_20 > -5:
        details.append(f"20日动量{momentum_20:.1f}%，偏弱，不加分")
    else:
        score -= 1
        details.append(f"20日动量{momentum_20:.1f}%，弱势，-1分")

    # 波动率因子
    returns = close.pct_change().dropna()
    if len(returns) >= 20:
        vol_20 = returns.tail(20).std() * np.sqrt(252) * 100
        factor_values['20日年化波动率'] = round(vol_20, 2)
        if vol_20 < 25:
            score += 2
            details.append(f"年化波动率{vol_20:.1f}%，低波动，+2分")
        elif vol_20 < 40:
            score += 1
            details.append(f"年化波动率{vol_20:.1f}%，适中，+1分")
        else:
            details.append(f"年化波动率{vol_20:.1f}%，高波动，不加分")

    # 量比因子
    if len(volume) >= 5:
        vol_ratio = float(volume.iloc[-1] / volume.tail(5).mean()) if volume.tail(5).mean() > 0 else 1.0
        factor_values['5日量比'] = round(vol_ratio, 2)
        if 1.2 < vol_ratio < 3.0:
            score += 2
            details.append(f"量比{vol_ratio:.2f}，温和放量，+2分")
        elif 0.8 < vol_ratio <= 1.2:
            score += 1
            details.append(f"量比{vol_ratio:.2f}，正常，+1分")
        elif vol_ratio >= 3.0:
            details.append(f"量比{vol_ratio:.2f}，过度放量，不加分")

    # 价格位置因子
    max_60 = high.rolling(60).max().iloc[-1] if len(close) >= 60 else high.max()
    min_60 = low.rolling(60).min().iloc[-1] if len(close) >= 60 else low.min()
    if max_60 - min_60 > 0:
        position = (close.iloc[-1] - min_60) / (max_60 - min_60) * 100
        factor_values['60日价格位置'] = round(position, 1)
        if 30 < position < 70:
            score += 2
            details.append(f"价格位置{position:.0f}%，适中，+2分")
        elif 20 <= position <= 80:
            score += 1
            details.append(f"价格位置{position:.0f}%，合理，+1分")

    # 均线偏离因子
    ma20 = close.rolling(20).mean().iloc[-1]
    if ma20 > 0:
        deviation = (close.iloc[-1] / ma20 - 1) * 100
        factor_values['偏离20日均线'] = round(deviation, 2)
        if -3 < deviation < 5:
            score += 1
            details.append(f"偏离MA20 {deviation:+.1f}%，合理范围，+1分")

    score = max(0, min(score, 20))

    return {
        "得分": score,
        "说明": "多因子评分（动量/波动率/量比/位置/偏离）",
        "详情": details,
        "因子值": factor_values
    }


def calculate_sector_score(symbol):
    """
    计算板块评分（0-10）
    基于真实板块指数涨跌幅和趋势动态计算
    """
    industry = STOCK_INDUSTRY.get(symbol, "其他")
    details = []
    score = 5

    perf_data = _get_sector_performance()

    # 尝试匹配板块名称
    matched_name = None
    for name in perf_data:
        if industry in name or name in industry:
            matched_name = name
            break

    if matched_name and matched_name in perf_data:
        change_pct = perf_data[matched_name]['涨跌幅']
        details.append(f"板块{matched_name}当日涨跌幅: {change_pct:+.2f}%")

        if change_pct > 3:
            score += 3
            details.append("板块当日强势上涨，+3分")
        elif change_pct > 1:
            score += 2
            details.append("板块当日温和上涨，+2分")
        elif change_pct > 0:
            score += 1
            details.append("板块当日微涨，+1分")
        elif change_pct > -1:
            details.append("板块当日微跌，不加分")
        elif change_pct > -3:
            score -= 1
            details.append("板块当日下跌，-1分")
        else:
            score -= 2
            details.append("板块当日大幅下跌，-2分")

        # 获取板块趋势评分
        trend_score, trend_info = _get_sector_trend_score(matched_name)
        score += (trend_score - 5)
        if trend_info:
            details.append(f"板块趋势: {trend_info.get('均线状态', '--')}, "
                           f"5日涨跌{trend_info.get('5日涨跌幅', 0):+.2f}%")
    else:
        details.append(f"板块{industry}: 未获取到实时数据，使用基础评分")

    score = max(0, min(score, 10))

    return {
        "得分": score,
        "说明": f"板块评分（{industry}，基于实时数据动态计算）",
        "详情": details,
        "板块": industry
    }


def calculate_fundamental_score(symbol):
    """
    基本面评分（0-30）
    基于真实接口获取的 PE/PB/总市值 等数据进行评分
    """
    try:
        df_list = _get_spot_df()
        filtered = df_list[df_list['代码'] == symbol]
        if not filtered.empty:
            row = filtered.iloc[0]
            name = str(row['名称'])
            industry = STOCK_INDUSTRY.get(symbol, "其他")
            score = 15
            details = [f"名称：{name}", f"行业：{industry}"]

            # 市盈率评分（0-8分）：PE在10-30之间较合理，过高或过低都扣分
            pe_val = row.get('市盈率-动态')
            if pe_val is not None and pd.notna(pe_val):
                pe = float(pe_val)
                details.append(f"市盈率(动)：{pe:.2f}")
                if 0 < pe <= 15:
                    score += 8
                    details.append("PE处于低估区间，+8分")
                elif 15 < pe <= 30:
                    score += 6
                    details.append("PE处于合理区间，+6分")
                elif 30 < pe <= 50:
                    score += 4
                    details.append("PE偏高，+4分")
                elif pe > 50:
                    score += 2
                    details.append("PE过高，+2分")
                else:
                    score += 3
                    details.append("PE为负值，+3分")
            else:
                details.append("市盈率：暂无数据")

            # 市净率评分（0-7分）：PB在1-5之间较合理
            pb_val = row.get('市净率')
            if pb_val is not None and pd.notna(pb_val):
                pb = float(pb_val)
                details.append(f"市净率：{pb:.2f}")
                if 0 < pb <= 2:
                    score += 7
                    details.append("PB处于低估区间，+7分")
                elif 2 < pb <= 5:
                    score += 5
                    details.append("PB处于合理区间，+5分")
                elif 5 < pb <= 10:
                    score += 3
                    details.append("PB偏高，+3分")
                else:
                    score += 1
                    details.append("PB过高或为负，+1分")
            else:
                details.append("市净率：暂无数据")

            score = min(score, 30)

            return {
                "得分": score,
                "说明": "基本面评分（基于真实PE/PB数据）",
                "详情": details,
                "名称": name,
                "行业": industry
            }
    except Exception:
        pass

    return {"得分": 15, "说明": "基本面评分（保守）", "详情": ["无详细基本面数据"], "名称": f"股票{symbol}"}


def score_stock(symbol):
    """综合评分（含多因子维度）"""
    df_kline = get_stock_kline(symbol, days=120)
    
    tech = calculate_technical_score(df_kline, symbol)
    fund = calculate_fundamental_score(symbol)
    risk = calculate_risk_score(df_kline, symbol)
    sector = calculate_sector_score(symbol)
    multi_factor = calculate_multi_factor_score(symbol, df_kline)
    
    total = tech["得分"] + fund["得分"] + risk["得分"] + sector["得分"] + multi_factor["得分"]
    max_score = 120
    
    if total >= 96:
        suggest = "强烈建议买入"
    elif total >= 84:
        suggest = "建议买入"
    elif total >= 72:
        suggest = "谨慎关注"
    else:
        suggest = "建议观望"
    
    return {
        "代码": symbol,
        "名称": fund.get("名称", symbol),
        "总分": total,
        "满分": max_score,
        "百分制": round(total / max_score * 100, 1),
        "建议": suggest,
        "评分详情": {
            "技术面": tech,
            "基本面": fund,
            "风险": risk,
            "板块": sector,
            "多因子": multi_factor
        }
    }


def batch_score(symbols_str):
    """批量评分"""
    symbol_list = [s.strip() for s in symbols_str.split(',')]
    scores = []
    for symbol in symbol_list:
        try:
            sc = score_stock(symbol)
            scores.append(sc)
        except Exception:
            continue
    scores_sorted = sorted(scores, key=lambda x: -x["总分"])
    return {
        "日期": datetime.now().strftime('%Y-%m-%d'),
        "数量": len(scores_sorted),
        "评分列表": scores_sorted
    }


def main():
    parser = argparse.ArgumentParser(description='股票综合评分工具（全真实数据）')
    parser.add_argument('action', choices=['single', 'batch'], help='操作类型: single（单只股票）, batch（批量）')
    parser.add_argument('--symbol', help='股票代码（single）')
    parser.add_argument('--symbols', help='股票代码，逗号分隔（batch）')
    
    args = parser.parse_args()
    
    try:
        if args.action == 'single':
            if not args.symbol:
                print(json.dumps({"error":"需要 --symbol 参数"},ensure_ascii=False,indent=2))
                sys.exit(1)
            data = score_stock(args.symbol)
        elif args.action == 'batch':
            if not args.symbols:
                print(json.dumps({"error":"需要 --symbols 参数"},ensure_ascii=False,indent=2))
                sys.exit(1)
            data = batch_score(args.symbols)
        
        print(json.dumps(data, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == '__main__':
    main()
