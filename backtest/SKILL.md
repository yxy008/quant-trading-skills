---
name: backtest
description: 回测验证，验证 stock-scoring 评分策略的历史有效性
---

# 回测验证 - backtest

## 功能介绍
- 使用历史数据回测 stock-scoring 评分策略
- 计算年化收益、最大回撤、胜率等核心指标
- 对比买入持有策略，验证策略有效性

## 快速开始
```bash
# 回测单只股票（默认贵州茅台，回测期1年）
python scripts/backtest_cli.py single --symbol 600519

# 回测多只股票
python scripts/backtest_cli.py multi --symbols 600519,000001,002594
```
