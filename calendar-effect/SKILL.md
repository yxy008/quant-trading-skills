---
name: calendar-effect
description: 日历效应分析工具，分析A股市场的月份效应、星期效应、节前节后效应等季节性规律
---

# 日历效应分析

## 功能

- 月份效应：分析各月的历史平均收益率
- 星期效应：分析周一至周五的收益率差异
- 节前效应：分析春节、国庆等节日前后的市场表现
- 月末效应：分析月末资金面紧张对市场的影响

## 使用方式

```bash
# 分析月份效应
python scripts/calendar_cli.py monthly --symbol 000300

# 分析星期效应
python scripts/calendar_cli.py weekly --symbol 000300
```