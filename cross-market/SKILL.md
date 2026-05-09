---
name: cross-market
description: 跨市场分析工具，分析A股、港股、美股之间的联动关系，AH溢价、跨市场套利机会
---

# 跨市场分析

## 功能

- AH溢价分析：分析A股与H股的溢价率变化
- 跨市场联动：分析美股/港股走势对A股的影响
- 全球资金流向：监控北向资金、南向资金流向
- 跨市场套利：识别跨市场套利机会

## 使用方式

```bash
# AH溢价分析
python scripts/cross_market_cli.py ah_premium --symbol 600519

# 北向资金分析
python scripts/cross_market_cli.py north_flow
```