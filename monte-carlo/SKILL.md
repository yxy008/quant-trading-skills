---
name: monte-carlo
description: 蒙特卡洛模拟工具，基于历史分布模拟未来收益路径，评估策略稳健性与尾部风险
---

# 蒙特卡洛模拟

## 功能

- 收益模拟：基于历史收益分布模拟未来收益路径
- VaR计算：计算在险价值(Value at Risk)
- CVaR计算：计算条件在险价值(Conditional VaR)
- 最大回撤分布：模拟最大回撤的概率分布
- 策略稳健性：评估策略在不同市场环境下的表现

## 使用方式

```bash
# 蒙特卡洛模拟
python scripts/monte_carlo_cli.py simulate --symbol 600519 --simulations 10000

# VaR计算
python scripts/monte_carlo_cli.py var --symbol 600519 --confidence 0.95
```