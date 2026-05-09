---
name: dividend-analysis
description: 分红分析工具，分析股票历史分红记录、股息率、分红稳定性，筛选高股息标的
---

# 分红分析

## 功能

- 分红记录：查询个股历史分红记录（送股、转增、派息）
- 股息率分析：计算当前股息率和历史平均股息率
- 分红稳定性：评估公司分红的连续性和增长性
- 高股息筛选：筛选高股息率且分红稳定的股票

## 使用方式

```bash
# 查询分红记录
python scripts/dividend_cli.py history --symbol 600519

# 高股息筛选
python scripts/dividend_cli.py screen --min_yield 3 --min_years 5
```