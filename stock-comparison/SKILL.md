---
name: stock-comparison
description: 股票对比工具，支持多只股票在估值、成长性、技术面、财务质量等维度的横向对比
---

# 股票对比

## 功能

- 估值对比：对比PE、PB、PS等估值指标
- 成长性对比：对比营收增长率、利润增长率
- 技术面对比：对比涨跌幅、换手率、波动率
- 财务质量对比：对比ROE、毛利率、负债率
- 综合评分：基于多维度对比的综合评分排名

## 使用方式

```bash
# 股票对比
python scripts/compare_cli.py compare --symbols 600519,000858,002304

# 行业对比
python scripts/compare_cli.py industry --sector 白酒
```