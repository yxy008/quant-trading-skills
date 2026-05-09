#!/usr/bin/env python3
"""
策略框架 - 策略定义、信号生成、策略注册
"""
import argparse
import json
import sys
import os
import ast
import re
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

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

from data_utils import get_stock_kline


class BaseStrategy(ABC):
    """策略基类"""

    def __init__(self, name, description, params=None):
        self.name = name
        self.description = description
        self.params = params or {}

    @abstractmethod
    def generate_signals(self, df):
        """
        生成交易信号
        参数:
            df: DataFrame, 包含 open/high/low/close/amount 列
        返回:
            DataFrame, 新增 signal 列: 1=买入, -1=卖出, 0=持有
        """
        pass

    def get_param_info(self):
        """获取策略参数说明"""
        return {}

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "params": self.get_param_info(),
        }


class MACrossoverStrategy(BaseStrategy):
    """均线交叉策略"""

    def __init__(self, fast_period=5, slow_period=20):
        super().__init__(
            name="均线交叉策略",
            description=f"当{fast_period}日均线上穿{slow_period}日均线时买入，下穿时卖出",
            params={"fast_period": fast_period, "slow_period": slow_period}
        )
        self.fast_period = fast_period
        self.slow_period = slow_period

    def get_param_info(self):
        return {
            "fast_period": {"type": "int", "default": 5, "min": 2, "max": 60, "label": "快线周期"},
            "slow_period": {"type": "int", "default": 20, "min": 5, "max": 120, "label": "慢线周期"}
        }

    def generate_signals(self, df):
        df = df.copy()
        close = df['close']

        df['ma_fast'] = close.rolling(window=self.fast_period).mean()
        df['ma_slow'] = close.rolling(window=self.slow_period).mean()

        df['signal'] = 0
        df['cross_up'] = (df['ma_fast'] > df['ma_slow']) & (df['ma_fast'].shift(1) <= df['ma_slow'].shift(1))
        df['cross_down'] = (df['ma_fast'] < df['ma_slow']) & (df['ma_fast'].shift(1) >= df['ma_slow'].shift(1))

        df.loc[df['cross_up'], 'signal'] = 1
        df.loc[df['cross_down'], 'signal'] = -1

        return df


class MACDStrategy(BaseStrategy):
    """MACD策略"""

    def __init__(self, fast=12, slow=26, signal_period=9):
        super().__init__(
            name="MACD策略",
            description=f"MACD({fast},{slow},{signal_period})金叉买入，死叉卖出",
            params={"fast": fast, "slow": slow, "signal_period": signal_period}
        )
        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period

    def get_param_info(self):
        return {
            "fast": {"type": "int", "default": 12, "min": 5, "max": 30, "label": "快线"},
            "slow": {"type": "int", "default": 26, "min": 10, "max": 60, "label": "慢线"},
            "signal_period": {"type": "int", "default": 9, "min": 3, "max": 20, "label": "信号线周期"}
        }

    def generate_signals(self, df):
        df = df.copy()
        close = df['close']

        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        df['dif'] = ema_fast - ema_slow
        df['dea'] = df['dif'].ewm(span=self.signal_period, adjust=False).mean()
        df['macd'] = 2 * (df['dif'] - df['dea'])

        df['signal'] = 0
        df['golden_cross'] = (df['dif'] > df['dea']) & (df['dif'].shift(1) <= df['dea'].shift(1))
        df['dead_cross'] = (df['dif'] < df['dea']) & (df['dif'].shift(1) >= df['dea'].shift(1))

        df.loc[df['golden_cross'], 'signal'] = 1
        df.loc[df['dead_cross'], 'signal'] = -1

        return df


class RSIStrategy(BaseStrategy):
    """RSI超买超卖策略"""

    def __init__(self, period=14, oversold=30, overbought=70):
        super().__init__(
            name="RSI策略",
            description=f"RSI({period})低于{oversold}买入，高于{overbought}卖出",
            params={"period": period, "oversold": oversold, "overbought": overbought}
        )
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def get_param_info(self):
        return {
            "period": {"type": "int", "default": 14, "min": 5, "max": 30, "label": "RSI周期"},
            "oversold": {"type": "int", "default": 30, "min": 10, "max": 40, "label": "超卖阈值"},
            "overbought": {"type": "int", "default": 70, "min": 60, "max": 90, "label": "超买阈值"}
        }

    def generate_signals(self, df):
        df = df.copy()
        close = df['close']

        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        df['signal'] = 0
        df.loc[df['rsi'] < self.oversold, 'signal'] = 1
        df.loc[df['rsi'] > self.overbought, 'signal'] = -1

        return df


class BollingerStrategy(BaseStrategy):
    """布林带策略"""

    def __init__(self, period=20, std_dev=2.0):
        super().__init__(
            name="布林带策略",
            description=f"价格跌破下轨买入，突破上轨卖出（周期{period}，标准差{std_dev}）",
            params={"period": period, "std_dev": std_dev}
        )
        self.period = period
        self.std_dev = std_dev

    def get_param_info(self):
        return {
            "period": {"type": "int", "default": 20, "min": 10, "max": 60, "label": "布林带周期"},
            "std_dev": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0, "label": "标准差倍数"}
        }

    def generate_signals(self, df):
        df = df.copy()
        close = df['close']

        df['ma'] = close.rolling(window=self.period).mean()
        df['std'] = close.rolling(window=self.period).std()
        df['upper'] = df['ma'] + self.std_dev * df['std']
        df['lower'] = df['ma'] - self.std_dev * df['std']

        df['signal'] = 0
        df.loc[close < df['lower'], 'signal'] = 1
        df.loc[close > df['upper'], 'signal'] = -1

        return df


class VolumeBreakoutStrategy(BaseStrategy):
    """放量突破策略"""

    def __init__(self, lookback=20, volume_multiple=1.5, price_threshold=0.03):
        super().__init__(
            name="放量突破策略",
            description=f"成交量放大{volume_multiple}倍且价格突破{lookback}日高点{price_threshold*100}%时买入",
            params={"lookback": lookback, "volume_multiple": volume_multiple, "price_threshold": price_threshold}
        )
        self.lookback = lookback
        self.volume_multiple = volume_multiple
        self.price_threshold = price_threshold

    def get_param_info(self):
        return {
            "lookback": {"type": "int", "default": 20, "min": 5, "max": 60, "label": "回看周期"},
            "volume_multiple": {"type": "float", "default": 1.5, "min": 1.0, "max": 5.0, "label": "成交量倍数"},
            "price_threshold": {"type": "float", "default": 0.03, "min": 0.01, "max": 0.10, "label": "价格突破阈值"}
        }

    def generate_signals(self, df):
        df = df.copy()
        close = df['close']
        amount = df['amount']

        df['avg_amount'] = amount.rolling(window=self.lookback).mean()
        df['high_n'] = close.rolling(window=self.lookback).max()

        df['volume_break'] = amount > df['avg_amount'] * self.volume_multiple
        df['price_break'] = close > df['high_n'].shift(1) * (1 + self.price_threshold)

        df['signal'] = 0
        df.loc[df['volume_break'] & df['price_break'], 'signal'] = 1

        df['volume_shrink'] = amount < df['avg_amount'] * 0.5
        df['price_drop'] = close < df['high_n'].shift(1) * 0.95
        df.loc[df['volume_shrink'] & df['price_drop'], 'signal'] = -1

        return df


class MultiFactorStrategy(BaseStrategy):
    """多因子综合策略 - 多个指标共振时产生信号"""

    def __init__(self, ma_fast=5, ma_slow=20, rsi_period=14, rsi_oversold=35, rsi_overbought=65):
        super().__init__(
            name="多因子综合策略",
            description="均线趋势+RSI+成交量多因子共振策略",
            params={
                "ma_fast": ma_fast, "ma_slow": ma_slow,
                "rsi_period": rsi_period, "rsi_oversold": rsi_oversold, "rsi_overbought": rsi_overbought
            }
        )
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def get_param_info(self):
        return {
            "ma_fast": {"type": "int", "default": 5, "min": 2, "max": 20, "label": "快线周期"},
            "ma_slow": {"type": "int", "default": 20, "min": 10, "max": 60, "label": "慢线周期"},
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30, "label": "RSI周期"},
            "rsi_oversold": {"type": "int", "default": 35, "min": 20, "max": 45, "label": "RSI超卖"},
            "rsi_overbought": {"type": "int", "default": 65, "min": 55, "max": 80, "label": "RSI超买"}
        }

    def generate_signals(self, df):
        df = df.copy()
        close = df['close']
        amount = df['amount']

        # 均线趋势
        df['ma_fast'] = close.rolling(window=self.ma_fast).mean()
        df['ma_slow'] = close.rolling(window=self.ma_slow).mean()
        df['trend_up'] = df['ma_fast'] > df['ma_slow']

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # 成交量
        df['vol_ma'] = amount.rolling(window=20).mean()
        df['vol_expand'] = amount > df['vol_ma'] * 1.2

        # 综合信号
        df['signal'] = 0

        # 买入条件：趋势向上 + RSI超卖区域 + 放量
        buy_cond = df['trend_up'] & (df['rsi'] < self.rsi_oversold) & df['vol_expand']
        df.loc[buy_cond, 'signal'] = 1

        # 卖出条件：趋势向下 + RSI超买区域
        sell_cond = (~df['trend_up']) & (df['rsi'] > self.rsi_overbought)
        df.loc[sell_cond, 'signal'] = -1

        return df


class TurtleTradingStrategy(BaseStrategy):
    """海龟交易策略 - 基于唐奇安通道突破"""

    def __init__(self, entry_period=20, exit_period=10, atr_period=20, atr_multiple=2.0):
        super().__init__(
            name="海龟交易策略",
            description=f"突破{entry_period}日高点买入，跌破{exit_period}日低点卖出，ATR({atr_period})止损",
            params={"entry_period": entry_period, "exit_period": exit_period,
                    "atr_period": atr_period, "atr_multiple": atr_multiple}
        )
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.atr_multiple = atr_multiple

    def get_param_info(self):
        return {
            "entry_period": {"type": "int", "default": 20, "min": 10, "max": 60, "label": "入场周期"},
            "exit_period": {"type": "int", "default": 10, "min": 5, "max": 30, "label": "离场周期"},
            "atr_period": {"type": "int", "default": 20, "min": 10, "max": 30, "label": "ATR周期"},
            "atr_multiple": {"type": "float", "default": 2.0, "min": 1.0, "max": 4.0, "label": "ATR倍数"}
        }

    def generate_signals(self, df):
        df = df.copy()
        high = df['high']
        low = df['low']
        close = df['close']

        df['entry_high'] = high.rolling(window=self.entry_period).max().shift(1)
        df['exit_low'] = low.rolling(window=self.exit_period).min().shift(1)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=self.atr_period).mean()

        df['signal'] = 0
        df['position'] = 0

        in_position = False
        for i in range(self.entry_period, len(df)):
            if not in_position:
                if df['high'].iloc[i] > df['entry_high'].iloc[i]:
                    df.loc[df.index[i], 'signal'] = 1
                    in_position = True
            else:
                stop_price = df['close'].iloc[i - 1] - self.atr_multiple * df['atr'].iloc[i]
                if df['low'].iloc[i] < df['exit_low'].iloc[i] or df['close'].iloc[i] < stop_price:
                    df.loc[df.index[i], 'signal'] = -1
                    in_position = False

        return df


class DualThrustStrategy(BaseStrategy):
    """Dual Thrust策略 - 基于前N日区间突破"""

    def __init__(self, lookback=20, k1=0.5, k2=0.5):
        super().__init__(
            name="Dual Thrust策略",
            description=f"突破前{lookback}日区间上轨买入，跌破下轨卖出（k1={k1}, k2={k2}）",
            params={"lookback": lookback, "k1": k1, "k2": k2}
        )
        self.lookback = lookback
        self.k1 = k1
        self.k2 = k2

    def get_param_info(self):
        return {
            "lookback": {"type": "int", "default": 20, "min": 5, "max": 60, "label": "回看周期"},
            "k1": {"type": "float", "default": 0.5, "min": 0.1, "max": 1.5, "label": "上轨系数"},
            "k2": {"type": "float", "default": 0.5, "min": 0.1, "max": 1.5, "label": "下轨系数"}
        }

    def generate_signals(self, df):
        df = df.copy()
        high = df['high']
        low = df['low']
        close = df['close']

        hh = high.rolling(window=self.lookback).max().shift(1)
        hc = close.rolling(window=self.lookback).max().shift(1)
        ll = low.rolling(window=self.lookback).min().shift(1)
        lc = close.rolling(window=self.lookback).min().shift(1)

        range_val = pd.concat([hh - lc, hc - ll], axis=1).max(axis=1)

        df['open'] = df.get('open', close.shift(1))
        df['upper_bound'] = df['open'] + self.k1 * range_val
        df['lower_bound'] = df['open'] - self.k2 * range_val

        df['signal'] = 0
        df.loc[high > df['upper_bound'], 'signal'] = 1
        df.loc[low < df['lower_bound'], 'signal'] = -1

        return df


class MomentumReversalStrategy(BaseStrategy):
    """动量反转策略 - 趋势跟踪+反转信号结合"""

    def __init__(self, momentum_period=20, reversal_period=5, threshold=0.05):
        super().__init__(
            name="动量反转策略",
            description=f"{momentum_period}日动量趋势+{reversal_period}日反转信号，阈值{threshold*100}%",
            params={"momentum_period": momentum_period, "reversal_period": reversal_period, "threshold": threshold}
        )
        self.momentum_period = momentum_period
        self.reversal_period = reversal_period
        self.threshold = threshold

    def get_param_info(self):
        return {
            "momentum_period": {"type": "int", "default": 20, "min": 5, "max": 60, "label": "动量周期"},
            "reversal_period": {"type": "int", "default": 5, "min": 2, "max": 20, "label": "反转周期"},
            "threshold": {"type": "float", "default": 0.05, "min": 0.01, "max": 0.15, "label": "信号阈值"}
        }

    def generate_signals(self, df):
        df = df.copy()
        close = df['close']

        df['momentum'] = close.pct_change(periods=self.momentum_period)
        df['reversal'] = -close.pct_change(periods=self.reversal_period)

        df['ma_short'] = close.rolling(window=self.reversal_period).mean()
        df['ma_long'] = close.rolling(window=self.momentum_period).mean()

        df['signal'] = 0

        buy_cond = (df['momentum'] > self.threshold) & (df['reversal'] > 0) & (df['ma_short'] > df['ma_long'])
        df.loc[buy_cond, 'signal'] = 1

        sell_cond = (df['momentum'] < -self.threshold) & (df['reversal'] < 0) & (df['ma_short'] < df['ma_long'])
        df.loc[sell_cond, 'signal'] = -1

        return df


class MeanReversionStrategy(BaseStrategy):
    """均值回归策略 - 价格偏离均线时反向交易"""

    def __init__(self, ma_period=20, entry_std=2.0, exit_std=0.5):
        super().__init__(
            name="均值回归策略",
            description=f"价格偏离{ma_period}日均线{entry_std}倍标准差时入场，回归{exit_std}倍标准差时离场",
            params={"ma_period": ma_period, "entry_std": entry_std, "exit_std": exit_std}
        )
        self.ma_period = ma_period
        self.entry_std = entry_std
        self.exit_std = exit_std

    def get_param_info(self):
        return {
            "ma_period": {"type": "int", "default": 20, "min": 10, "max": 60, "label": "均线周期"},
            "entry_std": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0, "label": "入场标准差"},
            "exit_std": {"type": "float", "default": 0.5, "min": 0.1, "max": 1.5, "label": "离场标准差"}
        }

    def generate_signals(self, df):
        df = df.copy()
        close = df['close']

        df['ma'] = close.rolling(window=self.ma_period).mean()
        df['std'] = close.rolling(window=self.ma_period).std()
        df['zscore'] = (close - df['ma']) / df['std'].replace(0, np.nan)

        df['signal'] = 0

        in_position = False
        for i in range(self.ma_period, len(df)):
            z = df['zscore'].iloc[i]
            if pd.isna(z):
                continue
            if not in_position:
                if z < -self.entry_std:
                    df.loc[df.index[i], 'signal'] = 1
                    in_position = True
                elif z > self.entry_std:
                    df.loc[df.index[i], 'signal'] = -1
                    in_position = True
            else:
                if abs(z) < self.exit_std:
                    df.loc[df.index[i], 'signal'] = 0
                    in_position = False

        return df


# 策略注册表
STRATEGY_REGISTRY = {
    "ma_cross": MACrossoverStrategy,
    "macd": MACDStrategy,
    "rsi": RSIStrategy,
    "bollinger": BollingerStrategy,
    "volume_breakout": VolumeBreakoutStrategy,
    "multi_factor": MultiFactorStrategy,
    "turtle": TurtleTradingStrategy,
    "dual_thrust": DualThrustStrategy,
    "momentum_reversal": MomentumReversalStrategy,
    "mean_reversion": MeanReversionStrategy,
}


# ==================== 市场状态识别与自适应策略 ====================

def classify_market_state(index_code="000300", days=120):
    """
    识别当前市场状态
    基于沪深300指数的趋势、波动率、成交量特征进行分类
    返回:
        dict: 市场状态分类结果
    """
    try:
        from data_utils import get_index_kline
        df = get_index_kline(index_code, days=days)
        if df is None or len(df) < 60:
            return {"状态": "未知", "置信度": 0, "说明": "数据不足"}

        close = df['收盘'] if '收盘' in df.columns else df['close']
        amount = df.get('成交额', df.get('amount', pd.Series(0, index=df.index)))

        # 计算趋势强度
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        current_price = float(close.iloc[-1])
        ma20_val = float(ma20.iloc[-1])
        ma60_val = float(ma60.iloc[-1])

        # 20日涨跌幅
        ret_20 = (close.iloc[-1] / close.iloc[-min(20, len(close))] - 1) * 100

        # 波动率
        returns = close.pct_change().dropna()
        vol_20 = float(returns.tail(20).std() * np.sqrt(252) * 100)

        # 成交量趋势
        vol_ma20 = amount.rolling(20).mean()
        vol_ratio = float(amount.iloc[-1] / vol_ma20.iloc[-1]) if vol_ma20.iloc[-1] > 0 else 1.0

        # 均线排列
        if current_price > ma20_val > ma60_val:
            alignment = "多头排列"
        elif current_price < ma20_val < ma60_val:
            alignment = "空头排列"
        else:
            alignment = "交叉震荡"

        # 市场状态判定
        if ret_20 > 5 and alignment == "多头排列" and vol_20 < 30:
            state = "牛市"
            confidence = 85
            description = "市场处于上升趋势，波动适中，适合趋势跟踪策略"
        elif ret_20 > 2 and alignment == "多头排列":
            state = "偏强震荡"
            confidence = 75
            description = "市场温和上涨，可适度参与，关注回调机会"
        elif ret_20 < -5 and alignment == "空头排列":
            state = "熊市"
            confidence = 85
            description = "市场处于下跌趋势，建议防御为主，控制仓位"
        elif ret_20 < -2 and alignment == "空头排列":
            state = "偏弱震荡"
            confidence = 75
            description = "市场偏弱，谨慎操作，等待企稳信号"
        elif abs(ret_20) <= 2:
            state = "横盘震荡"
            confidence = 70
            description = "市场方向不明，适合高抛低吸的震荡策略"
        elif vol_20 > 35:
            state = "高波动"
            confidence = 80
            description = "市场波动加剧，注意风险控制，降低仓位"
        else:
            state = "震荡"
            confidence = 65
            description = "市场方向不明，建议观望或轻仓操作"

        return {
            "状态": state,
            "置信度": confidence,
            "说明": description,
            "指标": {
                "20日涨跌幅": f"{ret_20:+.2f}%",
                "年化波动率": f"{vol_20:.1f}%",
                "均线排列": alignment,
                "量比": round(vol_ratio, 2),
                "当前价格": round(current_price, 2),
                "MA20": round(ma20_val, 2),
                "MA60": round(ma60_val, 2),
            },
            "策略建议": _get_state_strategy_advice(state),
        }
    except Exception as e:
        return {"状态": "未知", "置信度": 0, "说明": f"分析失败: {str(e)}"}


def _get_state_strategy_advice(state):
    """根据市场状态推荐策略类型"""
    advice_map = {
        "牛市": {
            "推荐策略": ["均线交叉", "MACD", "海龟交易", "放量突破"],
            "仓位建议": "70%-90%",
            "止损建议": "较宽止损（ATR*3）",
            "说明": "趋势明确，适合趋势跟踪策略，可适当提高仓位",
        },
        "偏强震荡": {
            "推荐策略": ["多因子综合", "动量反转", "Dual Thrust"],
            "仓位建议": "50%-70%",
            "止损建议": "适中止损（ATR*2）",
            "说明": "温和上涨，关注回调买入机会",
        },
        "熊市": {
            "推荐策略": ["均值回归（做空）", "RSI超卖反弹"],
            "仓位建议": "0%-30%",
            "止损建议": "严格止损（ATR*1.5）",
            "说明": "下跌趋势，以防御为主，减少操作频率",
        },
        "偏弱震荡": {
            "推荐策略": ["RSI", "布林带"],
            "仓位建议": "20%-40%",
            "止损建议": "严格止损（ATR*1.5）",
            "说明": "市场偏弱，谨慎参与，快进快出",
        },
        "横盘震荡": {
            "推荐策略": ["布林带", "均值回归", "RSI"],
            "仓位建议": "30%-50%",
            "止损建议": "适中止损（ATR*2）",
            "说明": "震荡行情，高抛低吸，注意突破方向",
        },
        "高波动": {
            "推荐策略": ["布林带", "Dual Thrust"],
            "仓位建议": "20%-40%",
            "止损建议": "宽止损（ATR*3）",
            "说明": "高波动环境，降低仓位，放宽止损避免被震出",
        },
        "震荡": {
            "推荐策略": ["多因子综合", "RSI"],
            "仓位建议": "30%-50%",
            "止损建议": "适中止损（ATR*2）",
            "说明": "方向不明，控制仓位等待方向明确",
        },
    }
    return advice_map.get(state, advice_map["震荡"])


class AdaptiveStrategy(BaseStrategy):
    """市场状态自适应策略 - 根据市场状态动态调整策略参数"""

    def __init__(self, base_strategy_id="ma_cross", index_code="000300", **params):
        super().__init__(
            name="市场自适应策略",
            description=f"基于市场状态动态调整{base_strategy_id}策略参数",
            params={"base_strategy_id": base_strategy_id, "index_code": index_code, **params}
        )
        self.base_strategy_id = base_strategy_id
        self.index_code = index_code
        self.extra_params = params

    def get_param_info(self):
        return {
            "base_strategy_id": {"type": "str", "default": "ma_cross", "label": "基础策略ID"},
            "index_code": {"type": "str", "default": "000300", "label": "参考指数代码"},
        }

    def _get_adaptive_params(self, df):
        """根据市场状态获取自适应参数"""
        market_state = classify_market_state(self.index_code, days=len(df))
        state = market_state.get("状态", "震荡")

        # 根据市场状态调整参数
        param_adjustments = {
            "牛市": {"fast_period": 5, "slow_period": 20, "rsi_oversold": 30, "rsi_overbought": 80},
            "偏强震荡": {"fast_period": 5, "slow_period": 20, "rsi_oversold": 35, "rsi_overbought": 70},
            "熊市": {"fast_period": 3, "slow_period": 10, "rsi_oversold": 20, "rsi_overbought": 60},
            "偏弱震荡": {"fast_period": 3, "slow_period": 15, "rsi_oversold": 25, "rsi_overbought": 65},
            "横盘震荡": {"fast_period": 5, "slow_period": 20, "rsi_oversold": 30, "rsi_overbought": 70},
            "高波动": {"fast_period": 10, "slow_period": 30, "rsi_oversold": 25, "rsi_overbought": 75},
            "震荡": {"fast_period": 5, "slow_period": 20, "rsi_oversold": 30, "rsi_overbought": 70},
        }

        return param_adjustments.get(state, param_adjustments["震荡"]), market_state

    def generate_signals(self, df):
        df = df.copy()
        adaptive_params, market_state = self._get_adaptive_params(df)

        # 根据基础策略ID选择策略
        if self.base_strategy_id == "ma_cross":
            strategy = MACrossoverStrategy(
                fast_period=adaptive_params.get("fast_period", 5),
                slow_period=adaptive_params.get("slow_period", 20)
            )
        elif self.base_strategy_id == "rsi":
            strategy = RSIStrategy(
                oversold=adaptive_params.get("rsi_oversold", 30),
                overbought=adaptive_params.get("rsi_overbought", 70)
            )
        elif self.base_strategy_id == "multi_factor":
            strategy = MultiFactorStrategy(
                ma_fast=adaptive_params.get("fast_period", 5),
                ma_slow=adaptive_params.get("slow_period", 20),
                rsi_oversold=adaptive_params.get("rsi_oversold", 35),
                rsi_overbought=adaptive_params.get("rsi_overbought", 65)
            )
        else:
            strategy = MACrossoverStrategy()

        result = strategy.generate_signals(df)
        result["市场状态"] = market_state.get("状态", "未知")
        result["自适应参数"] = adaptive_params

        return result


# 将自适应策略加入注册表
STRATEGY_REGISTRY["adaptive"] = AdaptiveStrategy


def get_strategy(strategy_id, **params):
    """根据ID获取策略实例"""
    if strategy_id not in STRATEGY_REGISTRY:
        return None
    strategy_cls = STRATEGY_REGISTRY[strategy_id]
    return strategy_cls(**params)


def list_strategies():
    """列出所有可用策略"""
    result = []
    for sid, cls in STRATEGY_REGISTRY.items():
        instance = cls()
        info = instance.to_dict()
        info["id"] = sid
        result.append(info)
    return result


def generate_signals(symbol, strategy_id, **params):
    """为指定股票生成交易信号"""
    strategy = get_strategy(strategy_id, **params)
    if strategy is None:
        return {"error": f"未知策略: {strategy_id}"}

    df = get_stock_kline(symbol, days=250)
    if df is None or len(df) < 60:
        return {"error": f"无法获取股票 {symbol} 的足够数据"}

    df = strategy.generate_signals(df)

    # 提取信号
    signal_rows = df[df['signal'] != 0].copy()
    signals = []
    for idx, row in signal_rows.iterrows():
        signals.append({
            "日期": idx.strftime('%Y-%m-%d'),
            "信号": "买入" if row['signal'] == 1 else "卖出",
            "收盘价": round(float(row['close']), 2)
        })

    # 最近信号
    recent_signals = signals[-10:] if len(signals) > 10 else signals

    # 当前持仓建议
    current_signal = 0
    for i in range(len(df) - 1, -1, -1):
        if df['signal'].iloc[i] != 0:
            current_signal = int(df['signal'].iloc[i])
            break

    advice_map = {1: "买入", -1: "卖出", 0: "持有观望"}
    current_advice = advice_map.get(current_signal, "持有观望")

    return {
        "股票代码": symbol,
        "策略": strategy.to_dict(),
        "数据天数": len(df),
        "信号总数": len(signals),
        "买入信号数": sum(1 for s in signals if s['信号'] == '买入'),
        "卖出信号数": sum(1 for s in signals if s['信号'] == '卖出'),
        "当前建议": current_advice,
        "最近信号": recent_signals,
        "全部信号": signals
    }


# ==================== 策略相关性矩阵 ====================

def strategy_correlation_matrix(symbol, strategy_ids=None, days=250):
    """
    策略相关性矩阵分析
    计算多个策略在同一股票上的信号相关性，识别策略拥挤风险

    参数:
        symbol: 股票代码
        strategy_ids: 要分析的策略ID列表，默认全部
        days: 分析天数

    返回: {
        "相关性矩阵": [[...], ...],
        "策略名称映射": [...],
        "拥挤度分析": {...},
        "有效策略数": int,
        "聚类分组": [...],
        "分散化建议": str,
    }
    """
    if strategy_ids is None:
        strategy_ids = list(STRATEGY_REGISTRY.keys())

    # 排除自适应策略（它是包装器）
    strategy_ids = [s for s in strategy_ids if s != "adaptive"]

    if len(strategy_ids) < 2:
        return {"error": "至少需要2个策略进行分析"}

    df = get_stock_kline(symbol, days=days + 50)
    if df is None or len(df) < 60:
        return {"error": f"无法获取股票 {symbol} 的足够数据"}

    # 为每个策略生成信号
    strategy_signals = {}
    strategy_names = {}

    for sid in strategy_ids:
        try:
            strategy = get_strategy(sid)
            if strategy is None:
                continue
            result = strategy.generate_signals(df)
            signals = result['signal'].values
            strategy_signals[sid] = signals
            strategy_names[sid] = strategy.name
        except Exception:
            continue

    if len(strategy_signals) < 2:
        return {"error": "成功运行的策略不足2个"}

    # 构建信号矩阵
    valid_ids = list(strategy_signals.keys())
    n = len(valid_ids)
    min_len = min(len(strategy_signals[sid]) for sid in valid_ids)

    signal_matrix = np.zeros((n, min_len))
    for i, sid in enumerate(valid_ids):
        signal_matrix[i] = strategy_signals[sid][-min_len:]

    # 计算相关性矩阵
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i][j] = 1.0
            else:
                corr_val = np.corrcoef(signal_matrix[i], signal_matrix[j])[0, 1]
                corr_matrix[i][j] = round(float(corr_val), 3) if not np.isnan(corr_val) else 0.0

    # 策略名称映射
    name_mapping = [{"id": sid, "name": strategy_names.get(sid, sid)} for sid in valid_ids]

    # 拥挤度分析
    crowding_analysis = _analyze_strategy_crowding(corr_matrix, valid_ids, strategy_names)

    # 有效策略数（基于相关性矩阵的特征值）
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = eigenvalues[eigenvalues > 0]
    effective_n = round(float(sum(eigenvalues) ** 2 / sum(eigenvalues ** 2)), 1)

    # 聚类分组
    clusters = _cluster_strategies(corr_matrix, valid_ids, strategy_names)

    # 分散化建议
    diversification_advice = _get_diversification_advice(corr_matrix, effective_n, n)

    return {
        "分析股票": symbol,
        "分析天数": days,
        "策略数量": n,
        "策略名称映射": name_mapping,
        "相关性矩阵": corr_matrix.tolist(),
        "拥挤度分析": crowding_analysis,
        "有效策略数": effective_n,
        "实际策略数": n,
        "策略效率": f"{effective_n / n * 100:.1f}%",
        "聚类分组": clusters,
        "分散化建议": diversification_advice,
        "高相关策略对": _find_high_corr_pairs(corr_matrix, valid_ids, strategy_names),
    }


def _analyze_strategy_crowding(corr_matrix, strategy_ids, strategy_names):
    """分析策略拥挤程度"""
    n = len(strategy_ids)

    # 计算平均相关性（排除对角线）
    upper_tri = []
    for i in range(n):
        for j in range(i + 1, n):
            upper_tri.append(corr_matrix[i][j])

    avg_corr = np.mean(upper_tri) if upper_tri else 0
    max_corr = np.max(upper_tri) if upper_tri else 0
    min_corr = np.min(upper_tri) if upper_tri else 0

    # 高相关比例（>0.7）
    high_corr_count = sum(1 for c in upper_tri if c > 0.7)
    high_corr_ratio = high_corr_count / len(upper_tri) if upper_tri else 0

    # 拥挤度等级
    if avg_corr > 0.6:
        crowding_level = "严重拥挤"
        risk = "多个策略高度相关，策略拥挤风险极高，建议大幅精简策略"
    elif avg_corr > 0.4:
        crowding_level = "中度拥挤"
        risk = "部分策略存在较高相关性，建议优化策略组合"
    elif avg_corr > 0.2:
        crowding_level = "轻度拥挤"
        risk = "策略间相关性适中，组合较为健康"
    else:
        crowding_level = "分散良好"
        risk = "策略间相关性低，组合分散化效果好"

    return {
        "平均相关性": round(float(avg_corr), 3),
        "最大相关性": round(float(max_corr), 3),
        "最小相关性": round(float(min_corr), 3),
        "高相关比例": f"{high_corr_ratio * 100:.1f}%",
        "拥挤等级": crowding_level,
        "风险提示": risk,
    }


def _cluster_strategies(corr_matrix, strategy_ids, strategy_names, threshold=0.5):
    """基于相关性对策略进行聚类分组"""
    n = len(strategy_ids)
    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True

        for j in range(n):
            if visited[j]:
                continue
            # 检查j是否与cluster中任一成员高度相关
            for c in cluster:
                if corr_matrix[c][j] > threshold:
                    cluster.append(j)
                    visited[j] = True
                    break

        cluster_info = {
            "组名": f"策略组{len(clusters) + 1}",
            "成员": [
                {"id": strategy_ids[idx], "name": strategy_names.get(strategy_ids[idx], strategy_ids[idx])}
                for idx in cluster
            ],
            "成员数": len(cluster),
            "组内平均相关": round(float(np.mean([corr_matrix[a][b]
                            for a in cluster for b in cluster if a != b])), 3) if len(cluster) > 1 else 1.0,
        }
        clusters.append(cluster_info)

    return clusters


def _find_high_corr_pairs(corr_matrix, strategy_ids, strategy_names, threshold=0.7):
    """找出高相关策略对"""
    n = len(strategy_ids)
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            if corr_matrix[i][j] > threshold:
                pairs.append({
                    "策略A": strategy_names.get(strategy_ids[i], strategy_ids[i]),
                    "策略B": strategy_names.get(strategy_ids[j], strategy_ids[j]),
                    "相关系数": round(float(corr_matrix[i][j]), 3),
                    "建议": "这两个策略高度相关，建议只保留其中一个",
                })

    pairs.sort(key=lambda x: x["相关系数"], reverse=True)
    return pairs[:10]


def _get_diversification_advice(corr_matrix, effective_n, total_n):
    """生成分散化建议"""
    efficiency = effective_n / total_n

    if efficiency > 0.8:
        return "策略组合分散化良好，各策略提供独立alpha来源，建议维持当前组合"
    elif efficiency > 0.5:
        return "策略组合存在一定冗余，建议精简高相关策略，保留互补性强的策略"
    elif efficiency > 0.3:
        return "策略组合冗余度较高，建议大幅精简，每个策略组只保留1个代表策略"
    else:
        return "策略组合严重冗余，有效策略数远少于实际策略数，建议重新设计策略体系"


# ==================== 策略代码验证与自定义回测 ====================

def validate_strategy_code(code):
    """
    策略代码语法检查
    检查Python语法、必要的函数定义等
    """
    issues = []
    warnings = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        issues.append({
            "类型": "语法错误",
            "行号": e.lineno,
            "描述": f"第{e.lineno}行: {e.msg}",
            "严重程度": "错误",
        })
        return {"是否有效": False, "问题列表": issues, "警告列表": warnings}

    has_init = False
    has_handle_bar = False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name == "init":
                has_init = True
                args = [a.arg for a in node.args.args]
                if "context" not in args:
                    issues.append({
                        "类型": "函数签名错误",
                        "行号": node.lineno,
                        "描述": "init函数应包含context参数",
                        "严重程度": "错误",
                    })
            elif node.name == "handle_bar":
                has_handle_bar = True
                args = [a.arg for a in node.args.args]
                if "context" not in args or "bar" not in args:
                    issues.append({
                        "类型": "函数签名错误",
                        "行号": node.lineno,
                        "描述": "handle_bar函数应包含context和bar参数",
                        "严重程度": "错误",
                    })

        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("os", "subprocess", "sys", "shutil"):
                    issues.append({
                        "类型": "安全限制",
                        "行号": node.lineno,
                        "描述": f"不允许导入 {alias.name} 模块",
                        "严重程度": "错误",
                    })
        elif isinstance(node, ast.ImportFrom):
            if node.module in ("os", "subprocess", "sys", "shutil"):
                issues.append({
                    "类型": "安全限制",
                    "行号": node.lineno,
                    "描述": f"不允许导入 {node.module} 模块",
                    "严重程度": "错误",
                })

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec", "__import__"):
                    issues.append({
                        "类型": "安全限制",
                        "行号": node.lineno,
                        "描述": f"不允许使用 {node.func.id}() 函数",
                        "严重程度": "错误",
                    })

    if not has_init:
        warnings.append({
            "类型": "缺少函数",
            "描述": "未定义init(context)函数，将使用默认初始化",
            "严重程度": "警告",
        })
    if not has_handle_bar:
        issues.append({
            "类型": "缺少函数",
            "描述": "未定义handle_bar(context, bar)函数，策略无法运行",
            "严重程度": "错误",
        })

    if has_handle_bar:
        return_values = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "handle_bar":
                for child in ast.walk(node):
                    if isinstance(child, ast.Return):
                        if isinstance(child.value, ast.Constant):
                            return_values.add(child.value.value)
                        elif isinstance(child.value, ast.Str):
                            return_values.add(child.value.s)

        valid_returns = {"buy", "sell", "hold"}
        invalid_returns = return_values - valid_returns
        if invalid_returns:
            warnings.append({
                "类型": "返回值",
                "描述": f"handle_bar返回了非标准值: {invalid_returns}，标准值为: buy/sell/hold",
                "严重程度": "警告",
            })

    return {
        "是否有效": len([i for i in issues if i["严重程度"] == "错误"]) == 0,
        "问题列表": issues,
        "警告列表": warnings,
    }


def run_strategy_backtest(code, symbol, start_date, end_date, initial_capital=100000, commission=0.0003):
    """
    运行策略回测
    在沙箱环境中执行用户编写的策略代码
    """
    try:
        df = get_stock_kline(symbol, days=500)
        if df is None or df.empty:
            return {"error": f"无法获取 {symbol} 的行情数据"}

        df = df.sort_values("date").reset_index(drop=True)

        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]

        if len(df) < 20:
            return {"error": "数据量不足，至少需要20个交易日"}

        sandbox_globals = {
            "__builtins__": {
                "abs": abs, "all": all, "any": any, "bool": bool,
                "dict": dict, "enumerate": enumerate, "filter": filter,
                "float": float, "int": int, "len": len, "list": list,
                "map": map, "max": max, "min": min, "range": range,
                "round": round, "set": set, "sorted": sorted, "str": str,
                "sum": sum, "tuple": tuple, "zip": zip,
                "True": True, "False": False, "None": None,
                "isinstance": isinstance, "Exception": Exception,
                "ValueError": ValueError, "TypeError": TypeError,
                "ZeroDivisionError": ZeroDivisionError,
            },
        }

        try:
            exec(code, sandbox_globals)
        except Exception as e:
            return {"error": f"策略代码执行失败: {str(e)}"}

        context = type("Context", (), {})()
        if "init" in sandbox_globals:
            try:
                sandbox_globals["init"](context)
            except Exception as e:
                return {"error": f"init函数执行失败: {str(e)}"}

        handle_bar = sandbox_globals.get("handle_bar")
        if not handle_bar:
            return {"error": "未找到handle_bar函数"}

        capital = initial_capital
        position = 0
        trades = []
        equity_curve = []
        signals = []

        closes = df["close"].tolist()
        highs = df["high"].tolist()
        lows = df["low"].tolist()
        opens = df["open"].tolist()
        volumes = df["volume"].tolist()
        dates = df["date"].tolist()

        for i in range(len(df)):
            bar = {
                "close": closes[:i + 1],
                "high": highs[:i + 1],
                "low": lows[:i + 1],
                "open": opens[:i + 1],
                "volume": volumes[:i + 1],
                "date": dates[i],
            }

            try:
                signal = handle_bar(context, bar)
            except Exception as e:
                signals.append({"日期": str(dates[i]), "信号": "error", "错误": str(e)})
                equity_curve.append(capital + position * closes[i])
                continue

            if signal not in ("buy", "sell", "hold"):
                signal = "hold"

            signals.append({"日期": str(dates[i]), "信号": signal, "价格": closes[i]})

            current_price = closes[i]

            if signal == "buy" and position == 0:
                max_shares = int(capital / (current_price * (1 + commission)))
                if max_shares > 0:
                    cost = max_shares * current_price * (1 + commission)
                    capital -= cost
                    position = max_shares
                    trades.append({
                        "日期": str(dates[i]),
                        "类型": "买入",
                        "价格": round(current_price, 2),
                        "数量": max_shares,
                        "金额": round(cost, 2),
                        "手续费": round(max_shares * current_price * commission, 2),
                    })

            elif signal == "sell" and position > 0:
                revenue = position * current_price * (1 - commission)
                capital += revenue
                trades.append({
                    "日期": str(dates[i]),
                    "类型": "卖出",
                    "价格": round(current_price, 2),
                    "数量": position,
                    "金额": round(revenue, 2),
                    "手续费": round(position * current_price * commission, 2),
                })
                position = 0

            equity = capital + position * closes[i]
            equity_curve.append(equity)

        if position > 0:
            final_price = closes[-1]
            revenue = position * final_price * (1 - commission)
            capital += revenue
            trades.append({
                "日期": str(dates[-1]),
                "类型": "平仓",
                "价格": round(final_price, 2),
                "数量": position,
                "金额": round(revenue, 2),
                "手续费": round(position * final_price * commission, 2),
            })
            position = 0

        final_equity = capital
        total_return = (final_equity / initial_capital - 1) * 100
        n = len(equity_curve)

        if n > 1:
            daily_returns = [(equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1] for i in range(1, n)]
            avg_return = np.mean(daily_returns) if daily_returns else 0
            std_return = np.std(daily_returns) if daily_returns else 0
            sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0

            peak = equity_curve[0]
            max_dd = 0
            for eq in equity_curve:
                if eq > peak:
                    peak = eq
                dd = (eq - peak) / peak * 100
                if dd < max_dd:
                    max_dd = dd

            win_count = 0
            sell_trades = [t for t in trades if t["类型"] in ("卖出", "平仓")]
            win_rate = win_count / max(len(sell_trades), 1) * 100
        else:
            sharpe = 0
            max_dd = 0
            win_rate = 0

        return {
            "策略名称": "自定义策略",
            "股票代码": symbol,
            "回测区间": f"{dates[0]} ~ {dates[-1]}",
            "初始资金": initial_capital,
            "最终权益": round(final_equity, 2),
            "总收益率": round(total_return, 2),
            "年化收益率": round(total_return / n * 252, 2) if n > 0 else 0,
            "夏普比率": round(sharpe, 2),
            "最大回撤": round(max_dd, 2),
            "胜率": round(win_rate, 1),
            "交易次数": len(trades),
            "买入次数": sum(1 for t in trades if t["类型"] == "买入"),
            "卖出次数": sum(1 for t in trades if t["类型"] in ("卖出", "平仓")),
            "交易记录": trades,
            "权益曲线": [round(e, 2) for e in equity_curve],
            "信号记录": signals,
            "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    except Exception as e:
        return {"error": f"回测执行失败: {str(e)}"}


def get_strategy_templates():
    """获取策略模板列表"""
    templates = []
    for sid, cls in STRATEGY_REGISTRY.items():
        if sid == "adaptive":
            continue
        instance = cls()
        templates.append({
            "id": sid,
            "name": instance.name,
            "description": instance.description,
            "params": instance.get_param_info(),
        })
    return templates


def get_template_code(template_id):
    """获取指定策略模板的代码示例"""
    if template_id not in STRATEGY_REGISTRY:
        return {"error": f"未找到模板: {template_id}"}

    strategy_cls = STRATEGY_REGISTRY[template_id]
    instance = strategy_cls()

    code_template = f'''# {instance.name}
# {instance.description}

def init(context):
    params = {json.dumps(instance.params, ensure_ascii=False)}
    for k, v in params.items():
        setattr(context, k, v)

def handle_bar(context, bar):
    closes = bar["close"]
    if len(closes) < 20:
        return "hold"

    # 在此编写你的策略逻辑
    # 返回 "buy" / "sell" / "hold"

    return "hold"
'''

    return {
        "id": template_id,
        "name": instance.name,
        "description": instance.description,
        "params": instance.get_param_info(),
        "code": code_template,
    }


# ==================== 策略参数自动推荐 ====================

def _get_param_grid(strategy_id):
    """获取策略的参数搜索网格"""
    grids = {
        "ma_cross": {
            "fast_period": [3, 5, 7, 10, 12, 15],
            "slow_period": [15, 20, 25, 30, 40, 50, 60],
        },
        "macd": {
            "fast": [8, 10, 12, 14],
            "slow": [20, 24, 26, 30, 34],
            "signal_period": [6, 7, 9, 11, 13],
        },
        "rsi": {
            "period": [7, 10, 14, 18, 21],
            "oversold": [20, 25, 30, 35],
            "overbought": [65, 70, 75, 80],
        },
        "bollinger": {
            "period": [15, 20, 25, 30],
            "std_dev": [1.5, 1.8, 2.0, 2.2, 2.5],
        },
        "turtle": {
            "entry_period": [15, 20, 25, 30],
            "exit_period": [8, 10, 12, 15],
        },
        "dual_thrust": {
            "lookback": [15, 20, 25, 30],
            "k1": [0.5, 0.6, 0.7, 0.8],
            "k2": [0.5, 0.6, 0.7, 0.8],
        },
        "momentum_reversal": {
            "momentum_period": [15, 20, 25, 30],
            "reversal_period": [3, 5, 7, 10],
            "threshold": [0.03, 0.05, 0.07, 0.10],
        },
        "mean_reversion": {
            "ma_period": [15, 20, 25, 30],
            "entry_std": [1.5, 2.0, 2.5],
            "exit_std": [0.3, 0.5, 0.8],
        },
        "multi_factor": {
            "ma_fast": [3, 5, 7, 10],
            "ma_slow": [15, 20, 25, 30],
            "rsi_oversold": [25, 30, 35],
            "rsi_overbought": [65, 70, 75],
        },
        "volume_breakout": {
            "volume_ratio": [1.5, 2.0, 2.5, 3.0],
            "ma_period": [15, 20, 25, 30],
        },
    }
    return grids.get(strategy_id, {})


def _generate_param_combinations(param_grid, max_combinations=200):
    """生成参数组合，限制最大组合数"""
    import itertools

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    all_combinations = list(itertools.product(*values))
    if len(all_combinations) > max_combinations:
        import random
        random.seed(42)
        all_combinations = random.sample(all_combinations, max_combinations)

    result = []
    for combo in all_combinations:
        result.append(dict(zip(keys, combo)))
    return result


def recommend_strategy_params(symbol, strategy_id, days=250, top_n=5,
                               metric="sharpe", max_combinations=200):
    """
    策略参数自动推荐
    通过网格搜索找到最优参数组合

    参数:
        symbol: 股票代码
        strategy_id: 策略ID
        days: 回测天数
        top_n: 返回前N个最优参数组合
        metric: 排序指标 (sharpe/return/calmar/win_rate)
        max_combinations: 最大参数组合数

    返回:
        dict: 推荐结果
    """
    param_grid = _get_param_grid(strategy_id)
    if not param_grid:
        return {"error": f"策略 {strategy_id} 不支持参数推荐，或未定义参数网格"}

    combinations = _generate_param_combinations(param_grid, max_combinations)
    total = len(combinations)

    # 动态导入回测模块
    import importlib.util

    backtest_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "backtest", "scripts", "backtest_cli.py"
    )

    spec = importlib.util.spec_from_file_location("backtest_cli", backtest_path)
    backtest_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backtest_module)

    results = []
    errors = 0

    for i, params in enumerate(combinations):
        try:
            bt_result = backtest_module.backtest_with_strategy(
                symbol, strategy_id,
                initial_capital=100000, days=days, **params
            )
            if 'error' in bt_result:
                errors += 1
                continue

            metrics = bt_result.get("绩效指标", {})
            results.append({
                "参数": params,
                "总收益率": metrics.get("总收益率", 0),
                "年化收益率": metrics.get("年化收益率", 0),
                "夏普比率": metrics.get("夏普比率", 0),
                "最大回撤": metrics.get("最大回撤", 0),
                "胜率": metrics.get("胜率", 0),
                "Calmar比率": metrics.get("Calmar比率", 0),
                "交易次数": metrics.get("交易总次数", 0),
                "年化波动率": metrics.get("年化波动率", 0),
            })
        except Exception:
            errors += 1
            continue

    if not results:
        return {
            "error": "所有参数组合回测均失败",
            "总组合数": total,
            "失败数": errors,
        }

    # 按指定指标排序
    metric_map = {
        "sharpe": "夏普比率",
        "return": "总收益率",
        "calmar": "Calmar比率",
        "win_rate": "胜率",
    }
    sort_key = metric_map.get(metric, "夏普比率")
    results.sort(key=lambda x: x.get(sort_key, 0) or 0, reverse=True)

    top_results = results[:top_n]

    # 生成推荐分析
    best = top_results[0]
    default_params = _get_default_params(strategy_id)

    analysis = _analyze_param_recommendation(
        strategy_id, best, default_params, top_results, sort_key
    )

    return {
        "股票代码": symbol,
        "策略ID": strategy_id,
        "策略名称": STRATEGY_REGISTRY.get(strategy_id, type(None))().name if strategy_id in STRATEGY_REGISTRY else strategy_id,
        "排序指标": sort_key,
        "回测天数": days,
        "总组合数": total,
        "成功数": len(results),
        "失败数": errors,
        "默认参数": default_params,
        "推荐参数": top_results,
        "最优参数": best["参数"],
        "推荐分析": analysis,
    }


def _get_default_params(strategy_id):
    """获取策略默认参数"""
    if strategy_id not in STRATEGY_REGISTRY:
        return {}
    instance = STRATEGY_REGISTRY[strategy_id]()
    return instance.params


def _analyze_param_recommendation(strategy_id, best, default_params, top_results, sort_key):
    """分析参数推荐结果"""
    analysis = []

    # 对比默认参数
    if default_params:
        diff_items = []
        for key, default_val in default_params.items():
            best_val = best["参数"].get(key)
            if best_val is not None and best_val != default_val:
                diff_items.append(f"{key}: {default_val} -> {best_val}")
        if diff_items:
            analysis.append(f"相比默认参数，最优参数调整: {'; '.join(diff_items)}")
        else:
            analysis.append("最优参数与默认参数一致，当前默认配置已较优")

    # 最优指标
    analysis.append(f"最优{sort_key}: {best.get(sort_key, 'N/A')}")

    # 参数稳定性分析
    if len(top_results) >= 3:
        sharpe_values = [r.get("夏普比率", 0) or 0 for r in top_results]
        sharpe_range = max(sharpe_values) - min(sharpe_values)
        if sharpe_range < 0.3:
            analysis.append("前3名夏普比率差异很小，参数选择较为稳健")
        elif sharpe_range < 1.0:
            analysis.append("前3名夏普比率有一定差异，建议选择最优参数")
        else:
            analysis.append("前3名夏普比率差异较大，参数敏感度高，需谨慎选择")

    # 回撤检查
    best_dd = best.get("最大回撤", 0) or 0
    if best_dd < -20:
        analysis.append(f"警告: 最优参数最大回撤 {best_dd}%，风险较高")
    elif best_dd < -10:
        analysis.append(f"注意: 最优参数最大回撤 {best_dd}%，需关注风险控制")

    return analysis


def main():
    parser = argparse.ArgumentParser(description='策略框架工具')
    parser.add_argument('action', choices=['list', 'signals', 'run', 'market_state', 'correlation', 'check', 'backtest', 'templates', 'template_code', 'recommend'],
                        help='操作: list(列出策略), signals(生成信号), run(运行策略), market_state(市场状态), correlation(策略相关性矩阵), check(代码检查), backtest(自定义回测), templates(模板列表), template_code(模板代码), recommend(参数推荐)')
    parser.add_argument('--symbol', default='600519', help='股票代码')
    parser.add_argument('--strategy', default='ma_cross', help='策略ID')
    parser.add_argument('--params', default='{}', help='策略参数JSON')
    parser.add_argument('--strategies', default=None, help='要分析的策略ID列表，逗号分隔，默认全部')
    parser.add_argument('--days', type=int, default=250, help='分析天数')
    parser.add_argument('--code', help='策略代码（用于check/backtest）')
    parser.add_argument('--start', help='回测开始日期')
    parser.add_argument('--end', help='回测结束日期')
    parser.add_argument('--capital', type=float, default=100000, help='初始资金')
    parser.add_argument('--id', help='模板ID（用于template_code）')

    args = parser.parse_args()

    try:
        if args.action == 'list':
            data = list_strategies()
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'signals':
            params = json.loads(args.params) if args.params else {}
            data = generate_signals(args.symbol, args.strategy, **params)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'run':
            params = json.loads(args.params) if args.params else {}
            data = generate_signals(args.symbol, args.strategy, **params)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'market_state':
            data = classify_market_state()
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'correlation':
            strategy_ids = args.strategies.split(',') if args.strategies else None
            data = strategy_correlation_matrix(args.symbol, strategy_ids, args.days)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'check':
            if not args.code:
                print(json.dumps({"error": "需要 --code 参数"}, ensure_ascii=False, indent=2))
                sys.exit(1)
            data = validate_strategy_code(args.code)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'backtest':
            if not args.code or not args.symbol:
                print(json.dumps({"error": "需要 --code 和 --symbol 参数"}, ensure_ascii=False, indent=2))
                sys.exit(1)
            data = run_strategy_backtest(args.code, args.symbol, args.start, args.end, args.capital)
            print(json.dumps(data, ensure_ascii=False, default=str))
        elif args.action == 'templates':
            data = get_strategy_templates()
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'template_code':
            if not args.id:
                print(json.dumps({"error": "需要 --id 参数"}, ensure_ascii=False, indent=2))
                sys.exit(1)
            data = get_template_code(args.id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'recommend':
            data = recommend_strategy_params(
                args.symbol, args.strategy,
                days=args.days, top_n=5, metric="sharpe"
            )
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
