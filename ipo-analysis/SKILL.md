---
name: ipo-analysis
description: 新股IPO分析工具，分析新股申购价值、上市首日表现预测、新股基本面评估
---

# 新股IPO分析

## 功能

- 新股申购：查询即将申购的新股信息
- 中签率分析：分析历史中签率和申购热度
- 上市表现：分析新股上市首日涨幅规律
- 基本面评估：评估新股的行业地位和财务质量
- 破发风险：评估新股破发风险

## 使用方式

```bash
# 查询新股
python scripts/ipo_cli.py upcoming

# 新股分析
python scripts/ipo_cli.py analyze --symbol 688xxx
```