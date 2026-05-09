---
name: ai-agent
description: AI智能选股代理，基于多因子评分模型，结合技术面、基本面、资金面进行综合评分，推荐优质股票
---

# AI智能选股代理

## 功能

- 多因子评分：综合技术面、基本面、资金面、情绪面进行评分
- 智能推荐：基于评分结果推荐优质股票
- 风险过滤：自动过滤ST股票、高风险股票
- 行业分散：确保推荐股票行业分散，降低集中风险

## 使用方式

```bash
# AI推荐股票
python scripts/agent_cli.py recommend --top 10

# 快速评分
python scripts/agent_cli.py score --symbol 600519
```