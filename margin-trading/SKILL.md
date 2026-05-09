---
name: margin-trading
description: 融资融券分析工具，监控融资余额、融券余额变化，分析杠杆资金动向
---

# 融资融券分析

## 功能

- 融资融券余额：查询个股和全市场的融资融券余额
- 融资买入：监控融资买入额变化趋势
- 融券卖出：监控融券卖出量的变化
- 杠杆情绪：通过融资融券数据判断市场杠杆情绪
- 维持担保比例：监控维持担保比例变化

## 使用方式

```bash
# 查询融资融券
python scripts/margin_cli.py query --symbol 600519

# 全市场融资融券
python scripts/margin_cli.py market
```