---
name: options-analysis
description: 期权分析工具，支持期权定价（BS模型）、希腊字母计算、期权策略分析
---

# 期权分析

## 功能

- 期权定价：基于Black-Scholes模型计算期权理论价格
- 希腊字母：计算Delta、Gamma、Theta、Vega、Rho
- 隐含波动率：计算期权的隐含波动率
- 期权策略：分析牛市价差、熊市价差、跨式等策略
- 盈亏分析：期权策略的到期盈亏图分析

## 使用方式

```bash
# 期权定价
python scripts/options_cli.py price --symbol 510050 --strike 2.50 --type call

# 希腊字母
python scripts/options_cli.py greeks --symbol 510050 --strike 2.50
```