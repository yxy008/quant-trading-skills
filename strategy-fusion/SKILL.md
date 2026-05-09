---
name: strategy-fusion
description: 策略融合工具，将多个子策略的信号进行加权融合，生成综合交易信号
---

# 策略融合

## 功能

- 信号融合：将多个策略的信号进行加权融合
- 权重优化：基于历史表现优化各策略的融合权重
- 投票机制：多数投票和加权投票的信号决策
- 冲突处理：处理不同策略信号冲突的情况
- 融合回测：回测融合策略的综合表现

## 使用方式

```bash
# 策略融合
python scripts/fusion_cli.py fuse --strategies ma_cross,macd,rsi

# 权重优化
python scripts/fusion_cli.py optimize --strategies ma_cross,macd --symbol 600519
```