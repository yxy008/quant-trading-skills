---
name: portfolio-mgmt
description: 投资组合管理工具，支持组合创建、再平衡、风险监控、组合优化和绩效跟踪
---

# 投资组合管理

## 功能

- 组合创建：创建和管理多个投资组合
- 再平衡：根据目标权重自动计算再平衡方案
- 风险监控：实时监控组合风险指标（波动率、VaR、最大回撤）
- 组合优化：基于均值方差模型优化组合权重
- 绩效跟踪：跟踪组合收益并与基准对比
- 策略组合：管理多个策略的组合配置

## 使用方式

```bash
# 创建组合
python scripts/portfolio_cli.py create --name my_portfolio --symbols 600519,000001

# 再平衡
python scripts/portfolio_cli.py rebalance --name my_portfolio
```