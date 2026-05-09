---
name: stat-arbitrage
description: 统计套利工具，基于协整检验和配对交易策略，寻找统计套利机会
---

# 统计套利

## 功能

- 协整检验：检验股票对之间的协整关系
- 配对交易：基于价差回归的配对交易策略
- 价差分析：分析配对股票的价差分布和均值回归特性
- 套利信号：生成配对交易的入场和出场信号
- 套利回测：回测配对交易策略的历史表现

## 使用方式

```bash
# 寻找配对
python scripts/stat_arb_cli.py find_pairs --sector 银行

# 配对分析
python scripts/stat_arb_cli.py analyze --pair 600036,601166
```