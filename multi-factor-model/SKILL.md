---
name: multi-factor-model
description: 多因子模型工具，支持因子构建、因子检验、因子组合，构建多因子选股模型
---

# 多因子模型

## 功能

- 因子构建：构建价值、动量、质量、波动率等因子
- 因子检验：IC分析、分层回测、因子收益率检验
- 因子组合：多因子合成与加权
- 选股模型：基于多因子评分的选股模型
- 因子监控：监控因子表现和失效预警

## 使用方式

```bash
# 因子分析
python scripts/multi_factor_cli.py analyze --symbols 600519,000001

# 因子选股
python scripts/multi_factor_cli.py screen --factors value,momentum,quality
```