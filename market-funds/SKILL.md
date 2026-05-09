---
name: market-funds
description: 市场资金流向查询工具，基于 AkShare 库查询北向资金流入流出、板块资金流向、个股资金流向，帮助新手了解市场情绪。
---

# 市场资金流向 - Market Funds

## 快速开始

安装依赖：
```bash
pip install akshare pandas
```

## 功能介绍

### 1. 北向资金流向

查询沪股通、深股通的每日资金流入流出情况

### 2. 行业板块资金流向

查询各行业板块的资金净流入流出情况

### 3. 个股资金流向

查询单只股票的大单、中单、小单资金流向

## 使用方法

### CLI 使用

```bash
# 查询北向资金
python scripts/funds_cli.py northbound

# 查询板块资金流向
python scripts/funds_cli.py industry

# 查询个股资金流向
python scripts/funds_cli.py stock --symbol 000001
```

## API 说明

```python
import akshare as ak

# 北向资金
df = ak.stock_em_hsgt_north_net_flow_in()

# 行业板块资金
df = ak.stock_board_industry_em(symbol="今日涨跌排行")

# 个股资金
# 注意：部分接口可能有变化，请以 AkShare 官方文档为准
```

## 资金流向解读

- **北向资金**：
  - 大幅净流入（> 50亿）：外资看好市场
  - 大幅净流出（> -50亿）：外资谨慎或撤离

- **板块资金**：
  - 热门板块资金持续流入：可能有行情
  - 板块资金大幅流出：可能有回调风险

- **个股资金**：
  - 大单持续流入：机构看好
  - 大单持续流出：机构减仓

## 常用股票代码

- **平安银行**: 000001
- **贵州茅台**: 600519
- **招商银行**: 600036
- **宁德时代**: 300750

## 注意事项

1. 资金流向仅供参考，不构成投资建议
2. 北向资金可能因汇率、外围市场等因素波动
3. 资金流入流出需结合股价、成交量等综合判断
