---
name: social-sentiment
description: 社交媒体情绪分析工具，分析股吧、雪球等平台的讨论热度和情绪倾向
---

# 社交媒体情绪分析

## 功能

- 讨论热度：监控个股在社交平台的讨论热度变化
- 情绪分析：分析讨论内容的积极/消极倾向
- 舆情预警：负面舆情突然增加时自动预警
- 热度排名：全市场个股讨论热度排名
- 情绪与股价：分析社交情绪与股价的相关性

## 使用方式

```bash
# 情绪分析
python scripts/social_cli.py analyze --symbol 600519

# 热度排名
python scripts/social_cli.py ranking --top 20
```