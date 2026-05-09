---
name: trading-calendar
description: A股交易日历工具，基于AkShare真实交易日历数据。支持交易日判断、交易日区间查询、前后交易日推算、交易日历概览等功能。自动排除周末和法定节假日。
---

# 交易日历工具

## 功能

- **交易日判断**: 判断指定日期是否为A股交易日
- **交易日区间查询**: 获取指定日期范围内的所有交易日
- **前后交易日推算**: 获取指定日期的前N个/后N个交易日
- **最近交易日**: 获取最近的交易日（含今天）
- **交易日历概览**: 获取指定年份的月度交易日统计
- **交易日计数**: 获取区间内交易日数量

## 使用方式

```bash
# 判断今天是否为交易日
python scripts/calendar_cli.py check

# 判断指定日期
python scripts/calendar_cli.py check --date 2025-05-01

# 获取区间交易日
python scripts/calendar_cli.py range --start 2025-05-01 --end 2025-05-31

# 获取前5个交易日
python scripts/calendar_cli.py prev --date 2025-05-08 --n 5

# 获取后3个交易日
python scripts/calendar_cli.py next --date 2025-05-08 --n 3

# 获取最近交易日
python scripts/calendar_cli.py latest

# 获取2025年交易日历概览
python scripts/calendar_cli.py calendar --year 2025

# 获取区间交易日数量
python scripts/calendar_cli.py count --start 2025-01-01 --end 2025-12-31
```

## 数据来源

基于 AkShare 的 `tool_trade_date_hist_sina()` 接口，数据来源于新浪财经，自动包含：
- 周末休市
- 法定节假日休市
- 调休安排

## 缓存机制

交易日历数据缓存1小时，避免频繁网络请求。
