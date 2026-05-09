---
name: strategy-framework
description: 策略框架工具，支持策略模板管理、策略信号生成、策略参数推荐（网格搜索优化）
---

# 策略框架

## 功能

- 策略模板：管理多种策略模板（均线交叉、MACD、布林带等）
- 信号生成：基于策略模板生成买卖信号
- 参数推荐：使用网格搜索自动推荐最优策略参数
- 策略对比：对比不同策略在同一股票上的表现
- 策略组合：多策略信号融合

## 使用方式

```bash
# 列出策略模板
python scripts/strategy_cli.py list

# 生成信号
python scripts/strategy_cli.py signal --symbol 600519 --strategy ma_cross

# 参数推荐
python scripts/strategy_cli.py recommend --symbol 600519 --strategy ma_cross
```