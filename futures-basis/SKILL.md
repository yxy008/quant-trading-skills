---
name: futures-basis
description: 期货基差分析工具，分析股指期货升贴水、期限结构、跨期套利机会
---

# 期货基差分析

## 功能

- 基差分析：计算期货与现货的基差（升贴水）
- 期限结构：分析不同到期月份合约的价格结构
- 跨期套利：识别跨期套利机会
- 交割日效应：分析交割日前后的市场行为

## 使用方式

```bash
# 基差分析
python scripts/basis_cli.py analyze --symbol IF

# 期限结构
python scripts/basis_cli.py term_structure --symbol IF
```