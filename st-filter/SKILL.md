---
name: st-filter
description: ST股票过滤器，自动识别和过滤ST、*ST、退市风险股票，保护投资组合安全
---

# ST股票过滤器

## 功能

- ST识别：自动识别ST、*ST股票
- 退市风险：识别有退市风险的股票
- 财务预警：基于财务指标预警潜在ST风险
- 自动过滤：在选股和交易中自动过滤ST股票
- ST监控：监控持仓中股票的ST状态变化

## 使用方式

```bash
# 检查ST状态
python scripts/st_filter_cli.py check --symbol 600519

# 扫描ST风险
python scripts/st_filter_cli.py scan --market all
```