---
name: dragon-tiger
description: 龙虎榜分析工具，分析龙虎榜上榜股票、游资席位动向、机构买卖情况
---

# 龙虎榜分析

## 功能

- 龙虎榜查询：查询每日龙虎榜上榜股票
- 席位分析：分析游资席位的买卖动向和操作风格
- 机构追踪：追踪机构专用席位的买入/卖出情况
- 上榜原因：分析股票上榜原因（涨跌幅偏离、换手率等）

## 使用方式

```bash
# 查询龙虎榜
python scripts/dragon_cli.py list --date 2025-01-01

# 席位分析
python scripts/dragon_cli.py seat --name 某某席位
```