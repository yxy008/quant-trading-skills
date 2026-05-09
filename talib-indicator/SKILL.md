---
name: talib-indicator
description: 基于TA-Lib库的技术指标计算工具，支持SMA、EMA、MACD、RSI、ATR、BBANDS等常用技术指标。结合akshare获取K线数据，输入股票代码和指标类型，返回计算结果。
---

# 技术指标计算工具 - TA-Lib

## 快速开始

安装依赖：
```bash
pip install ta-lib akshare pandas numpy
```

注意：TA-Lib 在 Windows 上安装可能需要预编译的wheel文件。

## 支持的技术指标

### 1. 简单移动平均线 (SMA)
```python
import talib
import numpy as np

close = np.array([...], dtype=np.float64)
sma = talib.SMA(close, timeperiod=5)
```

### 2. 指数移动平均线 (EMA)
```python
ema = talib.EMA(close, timeperiod=12)
```

### 3. MACD
```python
macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
```

### 4. 相对强弱指标 (RSI)
```python
rsi = talib.RSI(close, timeperiod=14)
```

### 5. 平均真实波幅 (ATR)
```python
atr = talib.ATR(high, low, close, timeperiod=14)
```

### 6. 布林带 (BBANDS)
```python
upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
```

## 常用股票代码

- **平安银行**: 000001
- **贵州茅台**: 600519
- **宁德时代**: 300750
- **比亚迪**: 002594
- **招商银行**: 600036

## 注意事项

1. 数据仅供学术研究，不构成投资建议
2. 需要先获取足够的历史数据才能计算指标
3. 建议添加异常处理
