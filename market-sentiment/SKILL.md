---
name: market-sentiment
description: 市场情绪分析工具，综合恐惧贪婪指数、波动率、资金流向等指标评估市场情绪
---

# 市场情绪分析

## 功能

- 恐惧贪婪指数：综合多维度指标计算市场情绪
- 波动率分析：分析VIX/波动率变化
- 资金情绪：通过资金流向判断市场情绪
- 舆情分析：分析新闻和社交媒体的市场情绪
- 情绪极值：识别市场情绪的极端区域

## 使用方式

```bash
# 市场情绪分析
python scripts/sentiment_cli.py analyze

# 恐惧贪婪指数
python scripts/sentiment_cli.py fear_greed
```