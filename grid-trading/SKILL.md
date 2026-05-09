---
name: grid-trading
description: 网格交易策略工具，支持等距网格、等比网格、动态网格，计算网格参数和回测收益
---

# 网格交易策略

## 功能

- 等距网格：固定价格间距的网格交易
- 等比网格：按百分比间距的网格交易
- 动态网格：根据波动率动态调整网格间距
- 网格回测：回测网格策略的历史收益
- 参数优化：自动优化网格上下限和间距

## 使用方式

```bash
# 生成网格参数
python scripts/grid_cli.py generate --symbol 600519 --upper 2000 --lower 1500 --grids 10

# 网格回测
python scripts/grid_cli.py backtest --symbol 600519 --upper 2000 --lower 1500
```