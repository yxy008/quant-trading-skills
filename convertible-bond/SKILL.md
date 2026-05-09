---
name: convertible-bond
description: 可转债分析工具，支持可转债估值、双低策略、折溢价分析、强赎/回售监控
---

# 可转债分析

## 功能

- 可转债查询：查询全市场可转债列表及基本信息
- 估值分析：计算转股溢价率、纯债价值、期权价值
- 双低策略：基于低价格+低溢价率的双低选债策略
- 强赎监控：监控可转债强赎触发进度
- 回售分析：分析可转债回售价值和回售收益

## 使用方式

```bash
# 查询可转债列表
python scripts/cb_cli.py list --min_price 100 --max_price 130

# 估值分析
python scripts/cb_cli.py analyze --symbol 113xxx
```