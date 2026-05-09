---
name: auto-invest
description: 定投策略工具，支持普通定投、智能定投（估值定投、均线定投），计算定投收益和回测
---

# 定投策略工具

## 功能

- 普通定投：固定时间固定金额定投
- 智能定投：根据估值（PE/PB）或均线偏离动态调整定投金额
- 定投回测：回测定投策略的历史收益
- 止盈策略：目标收益率止盈、回撤止盈

## 使用方式

```bash
# 定投回测
python scripts/invest_cli.py backtest --symbol 000300 --amount 1000 --period monthly

# 智能定投分析
python scripts/invest_cli.py smart --symbol 000300 --method pe
```