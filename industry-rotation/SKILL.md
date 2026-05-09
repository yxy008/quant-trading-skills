---
name: industry-rotation
description: 行业轮动策略工具，基于动量、估值、资金流向分析行业轮动规律，提供行业配置建议
---

# 行业轮动策略

## 功能

- 行业动量分析：计算各行业的短期/中期/长期动量
- 行业估值对比：对比各行业的PE/PB估值水平
- 资金流向监控：监控行业资金净流入/流出
- 轮动信号：基于多维度指标生成行业轮动信号
- 行业配置：根据轮动信号提供行业配置建议

## 使用方式

```bash
# 行业轮动分析
python scripts/rotation_cli.py analyze

# 行业动量排名
python scripts/rotation_cli.py momentum --period 20
```