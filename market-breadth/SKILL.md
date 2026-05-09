---
name: market-breadth
description: 市场宽度分析工具，分析涨跌家数、新高新低、腾落线等市场宽度指标，判断市场整体强弱
---

# 市场宽度分析

## 功能

- 涨跌统计：统计全市场上涨/下跌/平盘家数
- 腾落线(ADL)：计算和展示腾落线指标
- 新高新低：统计创N日新高/新低的股票数量
- 市场宽度：综合评估市场整体强弱状态
- 极端信号：识别市场过度乐观/悲观的极端信号

## 使用方式

```bash
# 市场宽度分析
python scripts/breadth_cli.py analyze

# 腾落线
python scripts/breadth_cli.py adl --days 60
```