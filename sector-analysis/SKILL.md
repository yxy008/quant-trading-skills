---
name: sector-analysis
description: 行业板块分析，支持查询板块涨跌幅、板块内个股、板块估值等
---

# 行业板块分析 - sector-analysis

## 功能介绍
- **板块列表**：查看所有行业板块
- **板块详情**：查看某板块的涨跌幅、个股、估值等
- **板块对比**：对比多个板块的表现

## 快速开始

安装依赖：
```bash
pip install akshare pandas
```

## 使用方法

### CLI 使用

```bash
# 查询所有板块列表
python scripts/sector_analysis_cli.py list

# 查询板块详情（白酒）
python scripts/sector_analysis_cli.py detail --sector 白酒

# 查询板块对比
python scripts/sector_analysis_cli.py compare --sectors 银行,白酒,新能源
```
