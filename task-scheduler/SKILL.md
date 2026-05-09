---
name: task-scheduler
description: 定时任务调度工具，支持定时执行数据更新、策略计算、信号扫描等任务
---

# 定时任务调度

## 功能

- 定时任务：创建和管理定时执行的任务
- 任务类型：数据更新、策略计算、信号扫描、报告生成
- Cron表达式：支持灵活的Cron调度表达式
- 任务日志：记录任务执行历史和结果
- 任务链：支持任务间的依赖和串联执行

## 使用方式

```bash
# 创建定时任务
python scripts/scheduler_cli.py create --name daily_update --cron "0 18 * * *" --task data_update

# 列出任务
python scripts/scheduler_cli.py list
```