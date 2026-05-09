---
name: stock-pool
description: 候选股票池生成，无需用户输入，主动推荐符合条件的股票
---

# 候选股票池生成 - stock-pool

## 功能介绍
- 内置优质股票池（覆盖白酒、银行、新能源、半导体、消费等板块）
- 可按板块推荐，也可全市场推荐
- 可指定推荐数量

## 快速开始
```bash
# 生成默认候选池（每个板块3只，共15只）
python scripts/pool_cli.py generate

# 只生成白酒和银行的股票池
python scripts/pool_cli.py generate --sectors 白酒,银行 --count 3

# 全市场推荐20只
python scripts/pool_cli.py generate --all --count 20
```
