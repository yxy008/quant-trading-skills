---
name: data-storage
description: 数据存储与缓存管理，支持SQLite数据持久化、数据缓存、回测结果存储、报告导出（Excel/HTML）
---

# 数据存储与缓存管理

## 功能

- 数据持久化：将行情数据、财务数据存储到SQLite数据库
- 数据缓存：基于时间的智能缓存，减少重复数据请求
- 回测结果存储：保存回测结果，支持历史对比和趋势分析
- 报告导出：支持Excel和HTML格式的报告导出
- 数据清理：定期清理过期缓存和旧数据

## 使用方式

```bash
# 存储行情数据
python scripts/storage_cli.py save --symbol 600519 --type daily

# 查询缓存数据
python scripts/storage_cli.py query --symbol 600519

# 导出回测报告
python scripts/storage_cli.py export --format excel
```