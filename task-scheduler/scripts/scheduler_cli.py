#!/usr/bin/env python3
"""
定时任务调度 - 周期性执行量化分析任务 / 任务管理 / 执行日志
支持每日/每周定时执行选股、回测、风险检查等任务
使用MySQL数据库存储
"""
import argparse
import json
import os
import sys
import time
import threading
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from db_utils import execute_query, execute_update, get_connection


def list_tasks():
    """列出所有任务"""
    rows = execute_query("SELECT * FROM tasks ORDER BY created_at DESC")
    for r in rows:
        if isinstance(r.get("config"), str):
            try:
                r["config"] = json.loads(r["config"])
            except Exception:
                pass
        if r.get("created_at"):
            r["created_at"] = r["created_at"].strftime('%Y-%m-%d %H:%M:%S') if not isinstance(r["created_at"], str) else r["created_at"]
        if r.get("updated_at"):
            r["updated_at"] = r["updated_at"].strftime('%Y-%m-%d %H:%M:%S') if not isinstance(r["updated_at"], str) else r["updated_at"]
    return rows


def add_task(name, task_type, config=None, schedule_type="daily", schedule_time="09:00"):
    """添加定时任务"""
    if config is None:
        config = {}
    config_json = json.dumps(config, ensure_ascii=False)
    task_id = execute_update(
        "INSERT INTO tasks (name, task_type, config, schedule_type, schedule_time) "
        "VALUES (%s, %s, %s, %s, %s)",
        (name, task_type, config_json, schedule_type, schedule_time)
    )
    return {"id": task_id, "message": f"任务 '{name}' 创建成功"}


def update_task(task_id, **kwargs):
    """更新任务配置"""
    allowed = {"name", "task_type", "config", "schedule_type", "schedule_time", "enabled"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return {"error": "没有可更新的字段"}

    if "config" in updates and isinstance(updates["config"], dict):
        updates["config"] = json.dumps(updates["config"], ensure_ascii=False)

    set_clause = ", ".join(f"{k} = %s" for k in updates)
    values = list(updates.values()) + [task_id]

    execute_update(f"UPDATE tasks SET {set_clause} WHERE id = %s", values)
    return {"message": f"任务 #{task_id} 更新成功"}


def delete_task(task_id):
    """删除任务"""
    execute_update("DELETE FROM task_logs WHERE task_id = %s", (task_id,))
    execute_update("DELETE FROM tasks WHERE id = %s", (task_id,))
    return {"message": f"任务 #{task_id} 已删除"}


def toggle_task(task_id):
    """启用/禁用任务"""
    task = execute_query("SELECT enabled FROM tasks WHERE id = %s", (task_id,), fetch_one=True)
    if not task:
        return {"error": f"任务 #{task_id} 不存在"}
    new_state = 0 if task["enabled"] else 1
    execute_update(
        "UPDATE tasks SET enabled = %s WHERE id = %s",
        (new_state, task_id)
    )
    return {"message": f"任务 #{task_id} 已{'启用' if new_state else '禁用'}"}


def get_task_logs(task_id=None, limit=50):
    """获取任务执行日志"""
    if task_id:
        rows = execute_query(
            "SELECT * FROM task_logs WHERE task_id = %s ORDER BY started_at DESC LIMIT %s",
            (task_id, limit)
        )
    else:
        rows = execute_query(
            "SELECT tl.*, t.name as task_name FROM task_logs tl "
            "LEFT JOIN tasks t ON tl.task_id = t.id "
            "ORDER BY tl.started_at DESC LIMIT %s",
            (limit,)
        )
    for r in rows:
        if isinstance(r.get("result"), str):
            try:
                r["result"] = json.loads(r["result"])
            except Exception:
                pass
        if r.get("started_at") and not isinstance(r["started_at"], str):
            r["started_at"] = r["started_at"].strftime('%Y-%m-%d %H:%M:%S')
        if r.get("finished_at") and not isinstance(r["finished_at"], str):
            r["finished_at"] = r["finished_at"].strftime('%Y-%m-%d %H:%M:%S')
    return rows


def get_due_tasks():
    """获取当前应该执行的任务"""
    now = datetime.now()
    current_time = now.strftime('%H:%M')
    current_weekday = now.weekday()

    rows = execute_query("SELECT * FROM tasks WHERE enabled = 1")

    due = []
    for task in rows:
        schedule_time = task.get("schedule_time", "09:00")
        schedule_type = task.get("schedule_type", "daily")

        if schedule_type == "daily":
            if schedule_time == current_time:
                due.append(task)
        elif schedule_type == "weekly":
            if schedule_time == current_time and current_weekday == 0:
                due.append(task)
        elif schedule_type == "monthly":
            if schedule_time == current_time and now.day == 1:
                due.append(task)

    return due


def execute_task(task_id):
    """执行指定任务"""
    task = execute_query("SELECT * FROM tasks WHERE id = %s", (task_id,), fetch_one=True)
    if not task:
        return {"error": f"任务 #{task_id} 不存在"}

    config = task.get("config", "{}")
    if isinstance(config, str):
        config = json.loads(config)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    log_id = execute_update(
        "INSERT INTO task_logs (task_id, status, started_at) VALUES (%s, 'running', %s)",
        (task_id, now)
    )

    try:
        result = _run_task_by_type(task["task_type"], config)
        finished_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        execute_update(
            "UPDATE task_logs SET status = 'success', result = %s, finished_at = %s WHERE id = %s",
            (json.dumps(result, ensure_ascii=False, default=str), finished_at, log_id)
        )
        return {"task_id": task_id, "log_id": log_id, "status": "success", "result": result}
    except Exception as e:
        finished_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        execute_update(
            "UPDATE task_logs SET status = 'failed', error = %s, finished_at = %s WHERE id = %s",
            (str(e), finished_at, log_id)
        )
        return {"task_id": task_id, "log_id": log_id, "status": "failed", "error": str(e)}


def _run_task_by_type(task_type, config):
    """根据任务类型执行具体逻辑"""
    symbols = config.get("symbols", [])
    days = config.get("days", 250)

    if task_type == "market_overview":
        return _task_market_overview()
    elif task_type == "stock_ranking":
        return _task_stock_ranking(symbols, days)
    elif task_type == "risk_check":
        return _task_risk_check(symbols, days)
    elif task_type == "factor_analysis":
        return _task_factor_analysis(symbols, days)
    elif task_type == "industry_rotation":
        return _task_industry_rotation(days)
    else:
        return {"message": f"未知任务类型: {task_type}"}


def _task_market_overview():
    """市场概览任务"""
    try:
        import akshare as ak
        df = ak.stock_zh_index_daily_tx(symbol="sh000001")
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            return {
                "指数": "上证指数",
                "日期": str(latest.get('date', '')),
                "收盘价": float(latest.get('close', 0)),
                "涨跌幅": float(latest.get('pct_chg', 0)),
                "成交量": float(latest.get('volume', 0)),
            }
    except Exception as e:
        return {"error": str(e)}
    return {"error": "无法获取市场数据"}


def _task_stock_ranking(symbols, days):
    """股票排名任务"""
    try:
        from multi_factor_cli import multi_factor_select
        result = multi_factor_select(symbols, days=days, top_n=10)
        return {
            "选股时间": result.get("选股时间", ""),
            "候选股票数": result.get("候选股票数", 0),
            "TOP5": [s["代码"] for s in result.get("选股结果", [])[:5]]
        }
    except Exception as e:
        return {"error": str(e)}


def _task_risk_check(symbols, days):
    """风险检查任务"""
    try:
        from risk_control_cli import pre_trade_check, get_default_risk_config
        results = {}
        risk_config = get_default_risk_config()
        for sym in symbols[:10]:
            order_params = {"direction": "buy", "quantity": 100, "price": 10}
            portfolio_state = {"total_asset": 100000, "cash": 100000, "positions": []}
            r = pre_trade_check(sym, order_params, portfolio_state, risk_config)
            results[sym] = {
                "是否通过": r.get("passed", False),
                "拒绝原因": r.get("reject_reasons", [])
            }
        return results
    except Exception as e:
        return {"error": str(e)}


def _task_factor_analysis(symbols, days):
    """因子分析任务"""
    try:
        from factor_cli import calculate_all_factors
        results = {}
        for sym in symbols[:5]:
            r = calculate_all_factors(sym, days=days)
            scores = r.get("因子评分", {})
            results[sym] = {
                "综合评分": scores.get("综合评分", 0),
                "评级": scores.get("评级", "--")
            }
        return results
    except Exception as e:
        return {"error": str(e)}


def _task_industry_rotation(days):
    """行业轮动任务"""
    try:
        from rotation_cli import analyze_industry_rotation
        result = analyze_industry_rotation(days=days, top_n=5)
        return {
            "轮动信号": result.get("轮动信号", {}).get("信号", "--"),
            "推荐行业": [r["行业"] for r in result.get("推荐行业", [])]
        }
    except Exception as e:
        return {"error": str(e)}


def run_scheduler_once():
    """执行一次调度检查"""
    due_tasks = get_due_tasks()
    results = []
    for task in due_tasks:
        result = execute_task(task["id"])
        results.append(result)
    return results


class TaskScheduler:
    """后台任务调度器"""

    def __init__(self, check_interval=60):
        self.check_interval = check_interval
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self):
        while self._running:
            try:
                run_scheduler_once()
            except Exception:
                pass
            time.sleep(self.check_interval)


_scheduler_instance = None


def get_scheduler():
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = TaskScheduler()
    return _scheduler_instance


def main():
    parser = argparse.ArgumentParser(description='定时任务调度')
    parser.add_argument('action', choices=[
        'list', 'add', 'update', 'delete', 'toggle',
        'logs', 'execute', 'run-once', 'start', 'stop'
    ], help='操作类型')
    parser.add_argument('--id', type=int, help='任务ID')
    parser.add_argument('--name', help='任务名称')
    parser.add_argument('--type', help='任务类型: market_overview/stock_ranking/risk_check/factor_analysis/industry_rotation')
    parser.add_argument('--config', default='{}', help='任务配置JSON')
    parser.add_argument('--schedule', default='daily', choices=['daily', 'weekly', 'monthly'], help='调度类型')
    parser.add_argument('--time', default='09:00', help='执行时间 HH:MM')
    parser.add_argument('--limit', type=int, default=50, help='日志数量')

    args = parser.parse_args()

    try:
        if args.action == 'list':
            data = list_tasks()
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        elif args.action == 'add':
            if not args.name or not args.type:
                print(json.dumps({"error": "请提供任务名称和类型"}, ensure_ascii=False))
                sys.exit(1)
            config = json.loads(args.config) if args.config else {}
            data = add_task(args.name, args.type, config, args.schedule, args.time)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'update':
            if not args.id:
                print(json.dumps({"error": "请提供任务ID"}, ensure_ascii=False))
                sys.exit(1)
            kwargs = {}
            if args.name:
                kwargs['name'] = args.name
            if args.type:
                kwargs['task_type'] = args.type
            if args.config:
                kwargs['config'] = json.loads(args.config)
            if args.schedule:
                kwargs['schedule_type'] = args.schedule
            if args.time:
                kwargs['schedule_time'] = args.time
            data = update_task(args.id, **kwargs)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'delete':
            if not args.id:
                print(json.dumps({"error": "请提供任务ID"}, ensure_ascii=False))
                sys.exit(1)
            data = delete_task(args.id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'toggle':
            if not args.id:
                print(json.dumps({"error": "请提供任务ID"}, ensure_ascii=False))
                sys.exit(1)
            data = toggle_task(args.id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'logs':
            data = get_task_logs(args.id, args.limit)
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        elif args.action == 'execute':
            if not args.id:
                print(json.dumps({"error": "请提供任务ID"}, ensure_ascii=False))
                sys.exit(1)
            data = execute_task(args.id)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'run-once':
            data = run_scheduler_once()
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'start':
            scheduler = get_scheduler()
            scheduler.start()
            print(json.dumps({"message": "调度器已启动"}, ensure_ascii=False))
        elif args.action == 'stop':
            scheduler = get_scheduler()
            scheduler.stop()
            print(json.dumps({"message": "调度器已停止"}, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
