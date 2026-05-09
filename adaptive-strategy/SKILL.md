---
name: adaptive-strategy
description: 自适应策略引擎，根据市场环境自动切换策略模式（趋势/震荡/高波动），动态调整参数
---

# 自适应策略引擎

## 功能

- 市场环境识别：自动判断当前市场处于趋势、震荡还是高波动状态
- 策略自动切换：根据市场环境切换对应的交易策略
- 参数动态调整：根据波动率、成交量等指标动态调整策略参数
- 策略权重分配：多策略融合时的权重动态分配

## 使用方式

```bash
# 分析当前市场环境
python scripts/adaptive_cli.py analyze --symbol 600519

# 获取策略建议
python scripts/adaptive_cli.py recommend --symbol 600519
```