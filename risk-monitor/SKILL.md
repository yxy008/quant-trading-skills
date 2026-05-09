---
name: risk-monitor
description: 风险监控工具，实时监控组合风险指标，包括VaR、CVaR、波动率、最大回撤、压力测试
---

# 风险监控

## 功能

- VaR监控：实时计算和监控在险价值
- 波动率监控：监控组合和个股的波动率变化
- 最大回撤：实时跟踪最大回撤
- 压力测试：模拟极端市场环境下的组合表现
- 风险预警：风险指标超限时自动预警
- 相关性监控：监控持仓间的相关性变化

## 使用方式

```bash
# 风险监控
python scripts/monitor_cli.py monitor --symbols 600519,000001

# 压力测试
python scripts/monitor_cli.py stress --scenario 2008
```