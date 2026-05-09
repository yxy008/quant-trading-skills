#!/usr/bin/env python3
"""
日内交易分析系统
支持分钟级K线获取、日内形态识别、盘中量价分析、分时图特征分析
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


def get_minute_kline(symbol, period="5", days=5):
    """
    获取分钟级K线数据

    参数:
        symbol: 股票代码
        period: 周期 '1', '5', '15', '30', '60'
        days: 获取天数

    返回: DataFrame
    """
    try:
        df = ak.stock_zh_a_hist_min_em(
            symbol=symbol,
            period=period,
            adjust="qfq",
            start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'),
            end_date=datetime.now().strftime('%Y%m%d'),
        )
        if df is not None and len(df) > 0:
            return df
    except Exception:
        pass

    return None


# ==================== 日内量价分析 ====================

def intraday_volume_price_analysis(symbol, period="5", days=5):
    """
    日内量价关系分析
    分析盘中成交量与价格的关系，识别主力动向

    参数:
        symbol: 股票代码
        period: K线周期
        days: 分析天数

    返回: 量价分析结果
    """
    df = get_minute_kline(symbol, period=period, days=days)

    if df is None or len(df) < 20:
        return {"error": f"无法获取{symbol}的分钟K线数据"}

    # 标准化列名
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if '时间' in col or 'time' in col_lower or 'date' in col_lower:
            col_map['时间'] = col
        elif '开' in col or 'open' in col_lower:
            col_map['开盘'] = col
        elif '收' in col or 'close' in col_lower:
            col_map['收盘'] = col
        elif '高' in col or 'high' in col_lower:
            col_map['最高'] = col
        elif '低' in col or 'low' in col_lower:
            col_map['最低'] = col
        elif '量' in col or 'vol' in col_lower:
            col_map['成交量'] = col
        elif '额' in col or 'amount' in col_lower:
            col_map['成交额'] = col

    close_col = col_map.get('收盘', df.columns[2] if len(df.columns) > 2 else df.columns[0])
    vol_col = col_map.get('成交量', df.columns[4] if len(df.columns) > 4 else df.columns[-1])
    amount_col = col_map.get('成交额', df.columns[5] if len(df.columns) > 5 else df.columns[-1])
    high_col = col_map.get('最高', df.columns[3] if len(df.columns) > 3 else df.columns[0])
    low_col = col_map.get('最低', df.columns[1] if len(df.columns) > 1 else df.columns[0])

    close = pd.Series(df[close_col].values.flatten() if hasattr(df[close_col].values, 'flatten') else df[close_col].values)
    volume = pd.Series(df[vol_col].values.flatten() if hasattr(df[vol_col].values, 'flatten') else df[vol_col].values)
    amount = pd.Series(df[amount_col].values.flatten() if hasattr(df[amount_col].values, 'flatten') else df[amount_col].values)
    high = pd.Series(df[high_col].values.flatten() if hasattr(df[high_col].values, 'flatten') else df[high_col].values)
    low = pd.Series(df[low_col].values.flatten() if hasattr(df[low_col].values, 'flatten') else df[low_col].values)

    # 基本统计
    total_volume = float(volume.sum())
    total_amount = float(amount.sum())
    avg_volume = float(volume.mean())
    price_change = (float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100 if float(close.iloc[0]) > 0 else 0

    # 量价关系分类
    returns = close.pct_change().dropna()
    vol_change = volume.pct_change().dropna()

    # 放量上涨/下跌统计
    up_mask = returns > 0
    down_mask = returns < 0
    high_vol_mask = volume.iloc[1:] > volume.iloc[1:].median()

    up_high_vol = int((up_mask & high_vol_mask).sum())
    down_high_vol = int((down_mask & high_vol_mask).sum())
    up_low_vol = int((up_mask & ~high_vol_mask).sum())
    down_low_vol = int((down_mask & ~high_vol_mask).sum())

    # 量价背离检测
    divergence_signals = []
    for i in range(5, len(close)):
        recent_close = close.iloc[i - 5:i + 1]
        recent_vol = volume.iloc[i - 5:i + 1]

        price_up = recent_close.iloc[-1] > recent_close.iloc[0]
        vol_down = recent_vol.iloc[-1] < recent_vol.iloc[0]

        if price_up and vol_down:
            divergence_signals.append({
                "位置": i,
                "类型": "价涨量缩（上涨乏力）",
                "时间": str(df.iloc[i].get(col_map.get('时间', df.columns[0]), i)),
            })
        elif not price_up and not vol_down and recent_vol.iloc[-1] > recent_vol.iloc[0] * 1.5:
            divergence_signals.append({
                "位置": i,
                "类型": "价跌量增（抛压加大）",
                "时间": str(df.iloc[i].get(col_map.get('时间', df.columns[0]), i)),
            })

    # 成交量分布分析
    vol_percentiles = {
        "25%分位": float(volume.quantile(0.25)),
        "50%分位": float(volume.quantile(0.50)),
        "75%分位": float(volume.quantile(0.75)),
        "90%分位": float(volume.quantile(0.90)),
        "最大值": float(volume.max()),
    }

    # 主力动向判断
    if up_high_vol > down_high_vol * 1.5:
        main_force = "主力资金以买入为主，放量上涨明显"
    elif down_high_vol > up_high_vol * 1.5:
        main_force = "主力资金以卖出为主，放量下跌明显"
    elif up_high_vol > down_high_vol and up_low_vol > down_low_vol:
        main_force = "整体偏多，但主力动作不明显"
    elif down_high_vol > up_high_vol and down_low_vol > up_low_vol:
        main_force = "整体偏空，抛压持续"
    else:
        main_force = "多空力量均衡，方向不明确"

    return {
        "股票代码": symbol,
        "分析周期": f"{period}分钟",
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "数据条数": len(df),
        "价格变化": f"{price_change:+.2f}%",
        "总成交量": f"{total_volume / 1e4:.0f}万手" if total_volume > 1e4 else f"{total_volume:.0f}手",
        "总成交额": f"{total_amount / 1e8:.2f}亿" if total_amount > 1e8 else f"{total_amount / 1e4:.0f}万",
        "量价关系统计": {
            "放量上涨": up_high_vol,
            "放量下跌": down_high_vol,
            "缩量上涨": up_low_vol,
            "缩量下跌": down_low_vol,
        },
        "量价背离信号": divergence_signals[-10:],
        "成交量分布": vol_percentiles,
        "主力动向": main_force,
        "均价": round(float(close.mean()), 2),
        "最高价": round(float(high.max()), 2),
        "最低价": round(float(low.min()), 2),
        "振幅": f"{(float(high.max()) / float(low.min()) - 1) * 100:.2f}%",
    }


# ==================== 日内形态识别 ====================

def intraday_pattern_recognition(symbol, period="5", days=5):
    """
    日内形态识别
    识别V形反转、头肩顶底、旗形整理、突破形态等

    参数:
        symbol: 股票代码
        period: K线周期
        days: 分析天数

    返回: 形态识别结果
    """
    df = get_minute_kline(symbol, period=period, days=days)

    if df is None or len(df) < 30:
        return {"error": f"无法获取{symbol}的分钟K线数据"}

    # 标准化列名
    close_col = None
    high_col = None
    low_col = None
    for col in df.columns:
        col_lower = str(col).lower()
        if '收' in col or 'close' in col_lower:
            close_col = col
        elif '高' in col or 'high' in col_lower:
            high_col = col
        elif '低' in col or 'low' in col_lower:
            low_col = col

    if close_col is None:
        close_col = df.columns[2] if len(df.columns) > 2 else df.columns[0]
    if high_col is None:
        high_col = df.columns[3] if len(df.columns) > 3 else df.columns[0]
    if low_col is None:
        low_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    close = pd.Series(df[close_col].values.flatten() if hasattr(df[close_col].values, 'flatten') else df[close_col].values)
    high = pd.Series(df[high_col].values.flatten() if hasattr(df[high_col].values, 'flatten') else df[high_col].values)
    low = pd.Series(df[low_col].values.flatten() if hasattr(df[low_col].values, 'flatten') else df[low_col].values)

    patterns = []

    # 1. V形反转检测
    v_patterns = _detect_v_reversal(close, low, high)
    patterns.extend(v_patterns)

    # 2. 突破形态检测
    breakout_patterns = _detect_breakout(close, high, low)
    patterns.extend(breakout_patterns)

    # 3. 双底/双顶检测
    double_patterns = _detect_double_top_bottom(close, high, low)
    patterns.extend(double_patterns)

    # 4. 盘中趋势判断
    trend_analysis = _analyze_intraday_trend(close, high, low)

    return {
        "股票代码": symbol,
        "分析周期": f"{period}分钟",
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "识别形态": patterns[-10:] if patterns else [],
        "形态总数": len(patterns),
        "趋势分析": trend_analysis,
    }


def _detect_v_reversal(close, low, high, window=10):
    """V形反转检测"""
    patterns = []
    if len(close) < window * 3:
        return patterns

    for i in range(window, len(close) - window):
        left_seg = close.iloc[i - window:i]
        right_seg = close.iloc[i:i + window]
        mid_point = close.iloc[i]

        left_min = float(left_seg.min())
        right_min = float(right_seg.min())
        left_max = float(left_seg.max())
        right_max = float(right_seg.max())

        # V形底：左侧下跌，右侧上涨
        left_decline = (left_seg.iloc[0] - left_min) / left_seg.iloc[0] if left_seg.iloc[0] > 0 else 0
        right_rise = (right_max - right_seg.iloc[0]) / right_seg.iloc[0] if right_seg.iloc[0] > 0 else 0

        if left_decline > 0.01 and right_rise > 0.01:
            patterns.append({
                "位置": i,
                "类型": "V形底",
                "左侧跌幅": f"{left_decline * 100:.2f}%",
                "右侧涨幅": f"{right_rise * 100:.2f}%",
                "信号": "看涨反转",
            })

        # 倒V形顶
        left_rise = (left_max - left_seg.iloc[0]) / left_seg.iloc[0] if left_seg.iloc[0] > 0 else 0
        right_decline = (right_seg.iloc[0] - right_min) / right_seg.iloc[0] if right_seg.iloc[0] > 0 else 0

        if left_rise > 0.01 and right_decline > 0.01:
            patterns.append({
                "位置": i,
                "类型": "倒V形顶",
                "左侧涨幅": f"{left_rise * 100:.2f}%",
                "右侧跌幅": f"{right_decline * 100:.2f}%",
                "信号": "看跌反转",
            })

    return patterns


def _detect_breakout(close, high, low, window=20):
    """突破形态检测"""
    patterns = []
    if len(close) < window * 2:
        return patterns

    for i in range(window, len(close) - 5):
        hist_high = float(high.iloc[i - window:i].max())
        hist_low = float(low.iloc[i - window:i].min())
        current = float(close.iloc[i])

        # 向上突破
        if current > hist_high * 1.005:
            # 确认：后续几根K线维持在突破位上方
            if all(float(close.iloc[i + j]) > hist_high * 0.998 for j in range(1, min(5, len(close) - i))):
                patterns.append({
                    "位置": i,
                    "类型": "向上突破",
                    "突破价位": round(hist_high, 2),
                    "当前价位": round(current, 2),
                    "信号": "看涨",
                })

        # 向下突破
        if current < hist_low * 0.995:
            if all(float(close.iloc[i + j]) < hist_low * 1.002 for j in range(1, min(5, len(close) - i))):
                patterns.append({
                    "位置": i,
                    "类型": "向下突破",
                    "突破价位": round(hist_low, 2),
                    "当前价位": round(current, 2),
                    "信号": "看跌",
                })

    return patterns


def _detect_double_top_bottom(close, high, low, window=15):
    """双顶/双底检测"""
    patterns = []
    if len(close) < window * 3:
        return patterns

    for i in range(window * 2, len(close) - window):
        seg1_high = float(high.iloc[i - window * 2:i - window].max())
        seg2_high = float(high.iloc[i - window:i].max())
        middle_low = float(low.iloc[i - window:i].min())

        # 双顶
        if abs(seg1_high / seg2_high - 1) < 0.01:
            if middle_low < seg1_high * 0.97:
                patterns.append({
                    "位置": i,
                    "类型": "双顶(M头)",
                    "顶部价位": round(seg1_high, 2),
                    "颈线": round(middle_low, 2),
                    "信号": "看跌反转",
                })

        seg1_low = float(low.iloc[i - window * 2:i - window].min())
        seg2_low = float(low.iloc[i - window:i].min())
        middle_high = float(high.iloc[i - window:i].max())

        # 双底
        if abs(seg1_low / seg2_low - 1) < 0.01:
            if middle_high > seg1_low * 1.03:
                patterns.append({
                    "位置": i,
                    "类型": "双底(W底)",
                    "底部价位": round(seg1_low, 2),
                    "颈线": round(middle_high, 2),
                    "信号": "看涨反转",
                })

    return patterns


def _analyze_intraday_trend(close, high, low):
    """盘中趋势分析"""
    if len(close) < 20:
        return {"趋势": "数据不足"}

    # 计算短期均线
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()

    current = float(close.iloc[-1])
    ma5_val = float(ma5.iloc[-1])
    ma10_val = float(ma10.iloc[-1])
    ma20_val = float(ma20.iloc[-1])

    # 趋势判断
    if current > ma5_val > ma10_val > ma20_val:
        trend = "多头排列，上升趋势"
        strength = "强"
    elif current > ma5_val > ma10_val:
        trend = "短期偏多"
        strength = "中"
    elif current < ma5_val < ma10_val < ma20_val:
        trend = "空头排列，下降趋势"
        strength = "强"
    elif current < ma5_val < ma10_val:
        trend = "短期偏空"
        strength = "中"
    else:
        trend = "震荡整理"
        strength = "弱"

    # 日内波动率
    returns = close.pct_change().dropna()
    intraday_vol = float(returns.std() * 100)

    # 开盘至今表现
    open_price = float(close.iloc[0])
    day_change = (current / open_price - 1) * 100

    return {
        "趋势": trend,
        "趋势强度": strength,
        "日内波动率": f"{intraday_vol:.3f}%",
        "开盘至今涨跌": f"{day_change:+.2f}%",
        "当前价": round(current, 2),
        "MA5": round(ma5_val, 2),
        "MA10": round(ma10_val, 2),
        "MA20": round(ma20_val, 2),
        "最高价": round(float(high.max()), 2),
        "最低价": round(float(low.min()), 2),
    }


# ==================== 盘中实时监控 ====================

def realtime_intraday_monitor(symbol):
    """
    盘中实时监控
    获取实时行情并分析盘中状态

    参数:
        symbol: 股票代码

    返回: 实时监控数据
    """
    try:
        df_spot = ak.stock_zh_a_spot_em()
        spot_row = df_spot[df_spot['代码'] == symbol]

        if spot_row.empty:
            return {"error": f"未找到股票{symbol}的实时数据"}

        spot = spot_row.iloc[0]

        current = float(spot.get('最新价', 0))
        open_price = float(spot.get('今开', 0))
        high = float(spot.get('最高', 0))
        low = float(spot.get('最低', 0))
        pre_close = float(spot.get('昨收', 0))
        volume = float(spot.get('成交量', 0))
        amount = float(spot.get('成交额', 0))
        change_pct = float(spot.get('涨跌幅', 0))
        turnover = float(spot.get('换手率', 0))

        # 盘中位置分析
        if high > low > 0:
            position_in_range = (current - low) / (high - low) * 100
        else:
            position_in_range = 50

        # 相对开盘价
        vs_open = (current / open_price - 1) * 100 if open_price > 0 else 0

        # 盘中状态判断
        if position_in_range > 80:
            intraday_status = "强势运行，接近日内高点"
        elif position_in_range > 60:
            intraday_status = "偏强运行，在日内高位区间"
        elif position_in_range > 40:
            intraday_status = "在日内中枢附近运行"
        elif position_in_range > 20:
            intraday_status = "偏弱运行，在日内低位区间"
        else:
            intraday_status = "弱势运行，接近日内低点"

        # 量能判断
        if turnover > 5:
            volume_status = f"换手率{turnover:.1f}%，交投活跃"
        elif turnover > 2:
            volume_status = f"换手率{turnover:.1f}%，交投正常"
        else:
            volume_status = f"换手率{turnover:.1f}%，交投清淡"

        return {
            "股票代码": symbol,
            "名称": str(spot.get('名称', '')),
            "更新时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "实时行情": {
                "最新价": round(current, 2),
                "涨跌幅": f"{change_pct:+.2f}%",
                "今开": round(open_price, 2),
                "最高": round(high, 2),
                "最低": round(low, 2),
                "昨收": round(pre_close, 2),
            },
            "成交数据": {
                "成交量": f"{volume / 1e4:.0f}万手" if volume > 1e4 else f"{volume:.0f}手",
                "成交额": f"{amount / 1e8:.2f}亿" if amount > 1e8 else f"{amount / 1e4:.0f}万",
                "换手率": f"{turnover:.2f}%",
            },
            "盘中分析": {
                "日内位置": f"{position_in_range:.0f}%",
                "相对开盘": f"{vs_open:+.2f}%",
                "状态": intraday_status,
                "量能": volume_status,
            },
        }
    except Exception as e:
        return {"error": f"获取实时数据失败: {str(e)}"}


# ==================== 日内波动率结构 ====================

def intraday_volatility_structure(symbol, period="30", days=10):
    """
    日内波动率结构分析
    分析不同时段的波动特征

    参数:
        symbol: 股票代码
        period: K线周期
        days: 分析天数

    返回: 波动率结构
    """
    df = get_minute_kline(symbol, period=period, days=days)

    if df is None or len(df) < 20:
        return {"error": "数据不足"}

    close_col = None
    for col in df.columns:
        if '收' in str(col) or 'close' in str(col).lower():
            close_col = col
            break
    if close_col is None:
        close_col = df.columns[2] if len(df.columns) > 2 else df.columns[0]

    close = pd.Series(df[close_col].values.flatten() if hasattr(df[close_col].values, 'flatten') else df[close_col].values)
    returns = close.pct_change().dropna()

    # 按时间段分组（模拟）
    n = len(returns)
    segments = 4
    seg_size = n // segments

    time_segments = []
    for s in range(segments):
        start = s * seg_size
        end = (s + 1) * seg_size if s < segments - 1 else n
        seg_returns = returns.iloc[start:end]
        seg_vol = float(seg_returns.std() * np.sqrt(252 / (6.5 / (int(period) / 60))) * 100)

        time_segments.append({
            "时段": f"时段{s + 1}",
            "波动率": f"{seg_vol:.2f}%",
            "样本数": len(seg_returns),
            "平均收益": f"{float(seg_returns.mean()) * 100:+.3f}%",
        })

    # 波动率聚集分析
    abs_returns = np.abs(returns.values)
    autocorr = float(pd.Series(abs_returns).autocorr()) if len(abs_returns) > 1 else 0

    return {
        "股票代码": symbol,
        "分析周期": f"{period}分钟",
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "时段波动率": time_segments,
        "波动率聚集": {
            "自相关": round(autocorr, 4),
            "解读": "波动率存在聚集效应（大波动后跟大波动）" if autocorr > 0.3 else "波动率聚集不显著",
        },
        "整体波动率": f"{float(returns.std() * np.sqrt(252 / (6.5 / (int(period) / 60))) * 100):.2f}%",
    }


def main():
    parser = argparse.ArgumentParser(description='日内交易分析系统')
    subparsers = parser.add_subparsers(dest='command')

    # 量价分析
    vp_parser = subparsers.add_parser('volume-price', help='日内量价分析')
    vp_parser.add_argument('--symbol', required=True, help='股票代码')
    vp_parser.add_argument('--period', default='5', help='K线周期(1/5/15/30/60)')
    vp_parser.add_argument('--days', type=int, default=5, help='分析天数')

    # 形态识别
    pattern_parser = subparsers.add_parser('pattern', help='日内形态识别')
    pattern_parser.add_argument('--symbol', required=True, help='股票代码')
    pattern_parser.add_argument('--period', default='5', help='K线周期')
    pattern_parser.add_argument('--days', type=int, default=5, help='分析天数')

    # 实时监控
    monitor_parser = subparsers.add_parser('monitor', help='盘中实时监控')
    monitor_parser.add_argument('--symbol', required=True, help='股票代码')

    # 波动率结构
    vol_parser = subparsers.add_parser('volatility', help='日内波动率结构')
    vol_parser.add_argument('--symbol', required=True, help='股票代码')
    vol_parser.add_argument('--period', default='30', help='K线周期')
    vol_parser.add_argument('--days', type=int, default=10, help='分析天数')

    args = parser.parse_args()

    if args.command == 'volume-price':
        result = intraday_volume_price_analysis(args.symbol, period=args.period, days=args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'pattern':
        result = intraday_pattern_recognition(args.symbol, period=args.period, days=args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'monitor':
        result = realtime_intraday_monitor(args.symbol)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'volatility':
        result = intraday_volatility_structure(args.symbol, period=args.period, days=args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
