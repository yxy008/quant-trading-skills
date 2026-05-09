---
name: trading-journal
description: 交易日志工具，记录和分析每笔交易的决策过程、执行情况和经验教训
---

# 交易日志

## 功能

- 交易记录：记录每笔交易的买入/卖出时间、价格、数量
- 决策回顾：记录交易决策的理由和依据
- 盈亏分析：自动计算每笔交易的盈亏
- 经验总结：记录交易后的经验教训
- 统计分析：按时间、股票、策略维度统计交易表现
- 行为分析：分析交易行为模式，发现改进空间

## 使用方式

```bash
# 记录交易
python scripts/journal_cli.py add --symbol 600519 --type buy --price 1800 --qty 100

# 查看日志
python scripts/journal_cli.py list --symbol 600519
```