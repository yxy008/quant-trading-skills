---
name: intraday-trading
description: 日内交易分析工具，分析分时图形态、日内高低点、成交量分布，辅助日内交易决策
---

# 日内交易分析

## 功能

- 分时图分析：分析日内价格走势和成交量变化
- 日内高低点：识别日内关键支撑位和阻力位
- 成交量分布：分析日内成交量在不同价位的分布
- 开盘区间：分析开盘区间的突破方向
- VWAP分析：计算日内VWAP作为交易参考

## 使用方式

```bash
# 日内分析
python scripts/intraday_cli.py analyze --symbol 600519

# 分时形态识别
python scripts/intraday_cli.py pattern --symbol 600519
```