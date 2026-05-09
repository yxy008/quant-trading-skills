---
name: multi-timeframe
description: 多周期分析工具，分析日线、周线、月线等多周期技术指标共振情况
---

# 多周期分析

## 功能

- 多周期K线：同时查看日线、周线、月线K线数据
- 指标共振：分析多周期技术指标（MACD、RSI、均线）的共振情况
- 周期对比：对比不同周期的趋势方向和强度
- 买卖信号：基于多周期共振生成买卖信号
- 周期转换：识别不同周期级别的支撑阻力位

## 使用方式

```bash
# 多周期分析
python scripts/timeframe_cli.py analyze --symbol 600519

# 共振分析
python scripts/timeframe_cli.py resonance --symbol 600519 --periods daily,weekly,monthly
```