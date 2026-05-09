---
name: signal-winrate
description: 信号胜率分析工具，统计各类交易信号的历史胜率、盈亏比、期望收益，优化信号参数
---

# 信号胜率分析

## 功能

- 信号统计：统计各类交易信号的历史触发次数和胜率
- 盈亏比：计算信号的盈亏比和期望收益
- 信号优化：通过参数优化提高信号胜率
- 信号过滤：基于市场环境过滤低质量信号
- 信号组合：分析多信号组合的胜率提升效果

## 使用方式

```bash
# 信号胜率分析
python scripts/winrate_cli.py analyze --signal macd_cross

# 信号优化
python scripts/winrate_cli.py optimize --signal ma_cross --symbol 600519
```