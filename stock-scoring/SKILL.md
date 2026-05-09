---
name: stock-scoring
description: 股票综合评分系统，结合技术面、基本面、风险等多维度评分，给出投资建议
---

# 股票综合评分 - stock-scoring

## 功能介绍
- **多维度评分**: 技术面（40分）、基本面（30分）、风险（20分）、板块（10分）
- **智能建议**: 基于总分给出买入、持有、卖出建议
- **详细解读**: 每个维度的详细得分和说明

## 快速开始
```bash
# 评分单只股票
python scripts/scoring_cli.py score --symbol 600519

# 批量评分
python scripts/scoring_cli.py batch --symbols 600519,000001,002594
```
