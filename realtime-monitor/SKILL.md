---
name: realtime-monitor
description: 实时行情监控工具，支持异动检测、涨跌停监控、成交量异常监控、价格突破监控
---

# 实时行情监控

## 功能

- 异动检测：检测价格异动、成交量异动、振幅异动
- 涨跌停监控：监控接近涨跌停的股票
- 成交量异常：检测成交量突然放大的股票
- 价格突破：监控突破关键价位的股票
- 实时推送：异动信息实时推送给用户

## 使用方式

```bash
# 启动监控
python scripts/monitor_cli.py start --symbols 600519,000001

# 异动检测
python scripts/monitor_cli.py detect --symbol 600519
```