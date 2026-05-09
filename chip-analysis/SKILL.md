---
name: chip-analysis
description: 筹码分布分析工具，分析股票筹码集中度、获利盘比例、成本分布，识别主力动向
---

# 筹码分布分析

## 功能

- 筹码集中度：分析股东户数变化，判断筹码集中/分散趋势
- 获利盘分析：计算不同价格区间的获利盘比例
- 成本分布：估算市场平均持仓成本
- 主力动向：通过筹码变化识别主力吸筹/出货

## 使用方式

```bash
# 分析筹码分布
python scripts/chip_cli.py analyze --symbol 600519

# 筹码集中度趋势
python scripts/chip_cli.py concentration --symbol 600519
```