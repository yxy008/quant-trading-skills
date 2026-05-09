---
name: financial-data
description: 股票财务数据查询，支持查询资产负债表、利润表、现金流量表、财务指标等
---

# 财务数据查询 - financial-data

## 功能介绍
- **财务指标查询**：PE、PB、ROE、ROA、营收、利润等
- **资产负债表**：查看公司资产负债情况
- **利润表**：查看公司营收利润情况
- **现金流量表**：查看公司现金流情况
- **财务健康分析**：根据财务指标给出简单分析

## 快速开始

安装依赖：
```bash
pip install akshare pandas
```

## 使用方法

### CLI 使用

```bash
# 查询财务指标
python scripts/financial_cli.py metrics --symbol 600519

# 查询利润表
python scripts/financial_cli.py income --symbol 600519

# 查询资产负债表
python scripts/financial_cli.py balance --symbol 600519

# 查询现金流量表
python scripts/financial_cli.py cash --symbol 600519
```
