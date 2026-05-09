---
name: shareholder-trade
description: 股东交易分析工具，监控大股东增减持、高管买卖、员工持股计划等内部人交易
---

# 股东交易分析

## 功能

- 大股东增减持：监控大股东的增持和减持行为
- 高管买卖：跟踪高管买卖本公司股票的情况
- 员工持股：分析员工持股计划的进展和影响
- 内部人交易：综合内部人交易信号判断公司前景
- 增减持影响：分析增减持对股价的历史影响

## 使用方式

```bash
# 查询股东交易
python scripts/shareholder_cli.py query --symbol 600519

# 增减持扫描
python scripts/shareholder_cli.py scan --type increase
```