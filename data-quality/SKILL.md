# 数据质量检查 (Data Quality)

## 功能概述
量化系统的数据基础保障，确保输入数据的可靠性。

## 核心能力

### 完整性检查
- 交易日历对比
- 缺失日期统计
- 数据完整率计算

### 缺失值检测
- OHLCV各列NaN检测
- 缺失比例统计

### 异常值检测
- 极端涨跌幅(>11%)
- 价格跳空(>5%)
- 成交量异常(3sigma)

### 停牌检测
- 连续多日价格不变
- 停牌区间识别

### 价格逻辑检查
- high >= low
- high >= open/close
- low <= open/close

### 数据一致性
- 多数据源对比(腾讯/东财)
- 价差分析

## 使用方式
```
python data_quality_cli.py check --symbol 600519 --days 250
python data_quality_cli.py batch --symbols 600519,000858,300750
python data_quality_cli.py consistency --symbol 600519
```
