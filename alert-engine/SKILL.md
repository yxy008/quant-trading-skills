---
name: alert-engine
description: 智能预警规则引擎，支持多条件组合、规则回测和全市场扫描，24种条件类型
---

# 智能预警规则引擎 (Alert Engine)

## 功能概述
自定义预警规则引擎，支持多条件组合、规则回测和全市场扫描。

## 核心能力

### 规则管理
- 自定义规则创建/编辑/删除
- 规则启用/禁用切换
- 规则持久化存储

### 条件类型（24种）
- 价格条件：price_above, price_below
- 涨跌幅条件：change_above, change_below
- 量比条件：volume_ratio_above, volume_ratio_below
- 振幅条件：amplitude_above
- RSI条件：rsi_above, rsi_below
- 均线条件：ma_golden_cross, ma_death_cross, price_above_ma, price_below_ma
- MACD条件：macd_golden_cross, macd_death_cross
- 新高新低：new_high_n, new_low_n
- 连续涨跌：consecutive_up, consecutive_down
- 换手率：turnover_above
- 估值条件：pe_below, pe_above, pb_below
- 市值条件：market_cap_above
- K线形态：pattern_detected

### 条件组合
- AND逻辑：所有条件同时满足
- OR逻辑：任一条件满足即可

### 严重级别
- info：信息提示
- warning：警告
- critical：严重警告

### 规则回测
- 历史触发频率统计
- 触发后N日收益分析
- 胜率统计

### 全市场扫描
- 用指定规则扫描全市场
- 找出所有触发规则的股票

## 使用方式
```
# 查看支持的条件类型
python alert_cli.py list_conditions

# 添加规则
python alert_cli.py add_rule --name "RSI超卖" --symbol 600519 --conditions '[{"type":"rsi_below","params":{"period":14,"value":30}}]' --severity warning

# 列出所有规则
python alert_cli.py list_rules

# 评估单条规则
python alert_cli.py evaluate --rule_id 1 --symbol 600519

# 评估所有规则
python alert_cli.py evaluate_all --symbol 600519

# 规则回测
python alert_cli.py backtest --rule_id 1 --symbol 600519 --days 250

# 全市场扫描
python alert_cli.py scan_market --rule_id 1 --limit 100

# 删除规则
python alert_cli.py delete_rule --rule_id 1

# 切换规则状态
python alert_cli.py toggle_rule --rule_id 1 --enabled false
```
