---
name: market-microstructure
description: 市场微观结构分析工具，分析买卖盘口、价差、订单流等微观结构特征
---

# 市场微观结构分析

## 功能

- 买卖盘口：分析买卖五档的挂单量和价差
- 订单流：分析主动买入/卖出的订单流向
- 大单分析：识别大单成交和主力动向
- 流动性分析：评估股票的流动性状况
- 价差分析：分析买卖价差的变化规律

## 使用方式

```bash
# 盘口分析
python scripts/microstructure_cli.py orderbook --symbol 600519

# 订单流分析
python scripts/microstructure_cli.py flow --symbol 600519
```