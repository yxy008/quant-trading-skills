---
name: etf-analysis
description: ETF分析工具，支持ETF筛选、折溢价分析、资金流向监控、ETF轮动策略
---

# ETF分析

## 功能

- ETF筛选：按类型、规模、费率等条件筛选ETF
- 折溢价分析：分析ETF的折价/溢价情况
- 资金流向：监控ETF的资金净流入/流出
- ETF轮动：基于动量和估值进行ETF轮动策略

## 使用方式

```bash
# 筛选ETF
python scripts/etf_cli.py screen --type 股票 --min_scale 10

# ETF分析
python scripts/etf_cli.py analyze --symbol 510050
```