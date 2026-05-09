---
name: performance-attribution
description: 绩效归因分析工具，支持Brinson归因、因子归因、交易归因与时间序列分析
---

# 绩效归因分析

## 功能

- Brinson归因：将超额收益分解为配置效应、选择效应和交互效应
- 因子归因：将组合收益分解为各因子的贡献
- 交易归因：分析每笔交易的盈亏来源和交易行为特征
- 时间序列归因：分析不同时间段的收益来源和表现变化
- 归因报告：生成详细的绩效归因分析报告

## 使用方式

```bash
# Brinson归因
python scripts/attribution_cli.py brinson --portfolio 600519,000001 --benchmark 000300

# 因子归因
python scripts/attribution_cli.py factor --returns 0.01,-0.005,0.02
```