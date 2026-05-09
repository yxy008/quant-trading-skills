---
name: mean-variance
description: 均值方差优化工具，基于马科维茨模型进行投资组合优化，计算有效前沿和最优权重
---

# 均值方差优化

## 功能

- 有效前沿：计算并展示投资组合的有效前沿
- 最优权重：基于均值方差模型计算最优资产配置权重
- 夏普比率：最大化夏普比率的组合优化
- 最小方差：最小化组合方差的风险优化
- 约束优化：支持权重上下限、行业约束等

## 使用方式

```bash
# 计算有效前沿
python scripts/mv_cli.py frontier --symbols 600519,000001,002594

# 最优权重
python scripts/mv_cli.py optimize --symbols 600519,000001,002594 --objective sharpe
```