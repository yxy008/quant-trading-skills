---
name: lockup-expiry
description: 限售股解禁分析工具，监控即将解禁的限售股，分析解禁对股价的潜在影响
---

# 限售股解禁分析

## 功能

- 解禁日历：查询即将解禁的限售股列表
- 解禁规模：分析解禁市值占流通市值的比例
- 解禁类型：区分首发解禁、定增解禁、股权激励解禁
- 影响评估：评估解禁对股价的潜在冲击
- 历史规律：分析历史解禁前后的股价表现

## 使用方式

```bash
# 查询解禁日历
python scripts/lockup_cli.py upcoming --days 30

# 解禁影响分析
python scripts/lockup_cli.py analyze --symbol 600519
```