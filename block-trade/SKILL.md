---
name: block-trade
description: 大宗交易分析工具，监控大宗交易数据，分析折溢价率、交易对手方，识别机构动向
---

# 大宗交易分析

## 功能

- 大宗交易查询：查询个股或全市场大宗交易数据
- 折溢价分析：分析大宗交易的折价/溢价情况
- 机构动向识别：通过大宗交易识别机构买入/卖出动向
- 解禁关联分析：关联限售股解禁数据，预判大宗交易压力

## 使用方式

```bash
# 查询个股大宗交易
python scripts/block_cli.py query --symbol 600519

# 全市场大宗交易扫描
python scripts/block_cli.py scan --date 2025-01-01
```