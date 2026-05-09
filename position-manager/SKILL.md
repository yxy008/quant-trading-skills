---
name: position-manager
description: 智能仓位管理，基于股票评分、风险承受能力，给出仓位分配和资金规划建议
---

# 智能仓位管理 - position-manager

## 功能介绍
- **评分驱动**: 结合 stock-scoring 的结果，合理分配仓位
- **风险控制**: 可设置风险承受能力（低/中/高）
- **分散投资**: 支持多只股票的仓位分配，避免单一股票风险过大

## 快速开始
```bash
# 单只股票仓位建议
python scripts/position_cli.py single --symbol 600519 --capital 100000 --risk low

# 批量分配
python scripts/position_cli.py batch --symbols 600519,000001,002594 --capital 200000 --risk medium
```
