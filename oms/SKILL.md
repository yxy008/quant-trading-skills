---
name: oms
description: 订单管理系统，支持订单创建、修改、撤销，订单状态跟踪和成交记录管理
---

# 订单管理系统 (OMS)

## 功能

- 订单创建：支持限价单、市价单、止损单等多种订单类型
- 订单修改：修改未成交订单的价格和数量
- 订单撤销：撤销未成交订单
- 订单查询：按状态、时间、股票代码查询订单
- 成交记录：记录和管理所有成交明细
- 订单状态：实时跟踪订单状态（待成交/部分成交/全部成交/已撤销）

## 使用方式

```bash
# 创建订单
python scripts/oms_cli.py create --symbol 600519 --type limit --price 1800 --qty 100

# 查询订单
python scripts/oms_cli.py query --status pending
```