---
name: stock-filter
description: 股票筛选器，基于 AkShare 库，支持按市盈率、市净率、市值范围、涨跌幅等条件筛选A股股票，帮助新手快速找到符合条件的标的。
---

# 股票筛选器 - Stock Filter

## 快速开始

安装依赖：
```bash
pip install akshare pandas
```

## 功能介绍

### 支持的筛选条件

1. **市盈率（PE）范围** - 按估值水平筛选
2. **市净率（PB）范围** - 按净资产估值筛选
3. **市值范围** - 按公司总市值筛选
4. **涨跌幅范围** - 按近期涨跌筛选
5. **成交量筛选** - 按交易活跃度筛选

### 使用场景

- 寻找低估值股票（PE/PB 双低）
- 寻找市值合适的投资标的
- 寻找近期有一定涨幅的活跃股

## 使用方法

### CLI 使用

```bash
# 筛选低PE股票（PE < 20）
python scripts/filter_cli.py filter --pe_max 20

# 筛选低估值股票（PE < 30，PB < 2）
python scripts/filter_cli.py filter --pe_max 30 --pb_max 2

# 筛选市值在 100-500 亿的股票
python scripts/filter_cli.py filter --market_cap_min 10000000000 --market_cap_max 50000000000

# 筛选近期涨幅在 5%-10% 的股票
python scripts/filter_cli.py filter --change_min 5 --change_max 10

# 组合筛选
python scripts/filter_cli.py filter --pe_max 30 --pb_max 3 --market_cap_min 10000000000 --limit 20
```

## API 说明

```python
import akshare as ak
import pandas as pd

# 获取所有A股实时行情
df = ak.stock_zh_a_spot_em()

# 筛选 PE<20 且 PB<2 的股票
filtered = df[(df['市盈率-动态'] > 0) & 
              (df['市盈率-动态'] < 20) & 
              (df['市净率'] > 0) & 
              (df['市净率'] < 2)]

# 按市值排序
filtered = filtered.sort_values('总市值', ascending=False)
```

## 估值参考指标

- **PE（市盈率）**
  - < 10: 低估值
  - 10-30: 合理范围
  - > 50: 较高估值

- **PB（市净率）**
  - < 1: 破净股
  - 1-2: 低估值
  - 2-4: 合理范围

## 常用股票代码

- **平安银行**: 000001
- **贵州茅台**: 600519
- **招商银行**: 600036
- **宁德时代**: 300750

## 注意事项

1. 筛选结果仅供参考，不构成投资建议
2. 建议结合基本面和技术面综合判断
3. PE/PB 低不代表一定会上涨，需结合行业和公司情况分析
