---
name: paper-trading
description: 模拟交易工具，支持虚拟资金模拟交易，跟踪模拟持仓和收益，验证策略实盘可行性
---

# 模拟交易

## 功能

- 模拟交易：使用虚拟资金进行模拟买卖
- 持仓管理：跟踪模拟持仓和盈亏
- 收益统计：计算模拟交易的收益率、夏普比率等
- 交易记录：记录所有模拟交易明细
- 策略验证：在模拟环境中验证策略的实盘可行性

## 使用方式

```bash
# 创建模拟账户
python scripts/paper_trading_cli.py create --capital 100000

# 模拟买入
python scripts/paper_trading_cli.py buy --symbol 600519 --qty 100 --price 1800
```