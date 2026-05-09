---
name: concentration-risk
description: 持仓集中度风险分析工具，评估投资组合的行业集中度、个股集中度，提供分散化建议
---

# 持仓集中度风险分析

## 功能

- 行业集中度：分析持仓在各行业的分布，识别过度集中的行业
- 个股集中度：计算单只股票占比，预警过度集中风险
- 相关性分析：分析持仓股票间的相关性，评估分散化效果
- 优化建议：提供降低集中度风险的具体建议

## 使用方式

```bash
# 分析持仓集中度
python scripts/concentration_cli.py analyze --symbols 600519,000001,002594

# 相关性矩阵
python scripts/concentration_cli.py correlation --symbols 600519,000001,002594
```