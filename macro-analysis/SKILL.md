---
name: macro-analysis
description: 宏观经济分析工具，分析GDP、CPI、PMI、货币供应等宏观指标与股市的关系
---

# 宏观经济分析

## 功能

- 经济指标：查询GDP、CPI、PPI、PMI等核心宏观指标
- 货币政策：分析利率、存款准备金率、M2等货币政策指标
- 宏观与股市：分析宏观指标与股市走势的相关性
- 经济周期：判断当前所处的经济周期阶段
- 政策解读：分析财政政策和货币政策对市场的影响

## 使用方式

```bash
# 查询宏观指标
python scripts/macro_cli.py indicators

# 宏观与股市相关性
python scripts/macro_cli.py correlation --symbol 000300
```