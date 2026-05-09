#!/usr/bin/env python3
"""
智能预警规则引擎 - 自定义规则定义、多条件组合、规则回测
支持用户自定义预警规则，多条件AND/OR组合，历史回测验证
"""
import argparse
import json
import sys
import os
import time
import re
from datetime import datetime, timedelta

_agent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

try:
    import akshare as ak
    import pandas as pd
    import numpy as np
except ImportError:
    print("请先安装依赖: pip install akshare pandas numpy")
    sys.exit(1)

from data_utils import get_stock_kline, _get_spot_df


# ==================== 规则存储 ====================

RULES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rules.json")


def _load_rules():
    """加载预警规则"""
    if os.path.exists(RULES_FILE):
        try:
            with open(RULES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"rules": [], "next_id": 1}


def _save_rules(rules_data):
    """保存预警规则"""
    os.makedirs(os.path.dirname(RULES_FILE), exist_ok=True)
    with open(RULES_FILE, 'w', encoding='utf-8') as f:
        json.dump(rules_data, f, ensure_ascii=False, indent=2)


# ==================== 条件类型定义 ====================

CONDITION_TYPES = {
    "price_above": {"描述": "价格大于指定值", "参数": ["value"], "示例": {"value": 100}},
    "price_below": {"描述": "价格小于指定值", "参数": ["value"], "示例": {"value": 50}},
    "change_above": {"描述": "涨跌幅大于指定百分比", "参数": ["value"], "示例": {"value": 5}},
    "change_below": {"描述": "涨跌幅小于指定百分比", "参数": ["value"], "示例": {"value": -5}},
    "volume_ratio_above": {"描述": "量比大于指定值", "参数": ["value"], "示例": {"value": 2}},
    "volume_ratio_below": {"描述": "量比小于指定值", "参数": ["value"], "示例": {"value": 0.5}},
    "amplitude_above": {"描述": "振幅大于指定百分比", "参数": ["value"], "示例": {"value": 7}},
    "rsi_above": {"描述": "RSI大于指定值", "参数": ["period", "value"], "示例": {"period": 14, "value": 70}},
    "rsi_below": {"描述": "RSI小于指定值", "参数": ["period", "value"], "示例": {"period": 14, "value": 30}},
    "ma_golden_cross": {"描述": "均线金叉(短期上穿长期)", "参数": ["short_period", "long_period"], "示例": {"short_period": 5, "long_period": 20}},
    "ma_death_cross": {"描述": "均线死叉(短期下穿长期)", "参数": ["short_period", "long_period"], "示例": {"short_period": 5, "long_period": 20}},
    "macd_golden_cross": {"描述": "MACD金叉", "参数": [], "示例": {}},
    "macd_death_cross": {"描述": "MACD死叉", "参数": [], "示例": {}},
    "price_above_ma": {"描述": "价格在均线上方", "参数": ["period"], "示例": {"period": 20}},
    "price_below_ma": {"描述": "价格在均线下方", "参数": ["period"], "示例": {"period": 20}},
    "new_high_n": {"描述": "创N日新高", "参数": ["days"], "示例": {"days": 60}},
    "new_low_n": {"描述": "创N日新低", "参数": ["days"], "示例": {"days": 60}},
    "consecutive_up": {"描述": "连续上涨N日", "参数": ["days"], "示例": {"days": 5}},
    "consecutive_down": {"描述": "连续下跌N日", "参数": ["days"], "示例": {"days": 5}},
    "turnover_above": {"描述": "换手率大于指定百分比", "参数": ["value"], "示例": {"value": 10}},
    "pe_below": {"描述": "市盈率小于指定值", "参数": ["value"], "示例": {"value": 15}},
    "pe_above": {"描述": "市盈率大于指定值", "参数": ["value"], "示例": {"value": 50}},
    "pb_below": {"描述": "市净率小于指定值", "参数": ["value"], "示例": {"value": 1}},
    "market_cap_above": {"描述": "总市值大于指定值(亿)", "参数": ["value"], "示例": {"value": 100}},
    "pattern_detected": {"描述": "检测到指定K线形态", "参数": ["pattern_name"], "示例": {"pattern_name": "锤子线"}},
}


# ==================== 条件评估 ====================

def _calc_rsi(close, period=14):
    """计算RSI"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calc_macd(close, fast=12, slow=26, signal=9):
    """计算MACD"""
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _detect_kline_pattern(open_p, high, low, close, pattern_name):
    """检测K线形态"""
    n = len(close)
    if n < 3:
        return False

    o1, h1, l1, c1 = open_p.iloc[-3], high.iloc[-3], low.iloc[-3], close.iloc[-3]
    o2, h2, l2, c2 = open_p.iloc[-2], high.iloc[-2], low.iloc[-2], close.iloc[-2]
    o3, h3, l3, c3 = open_p.iloc[-1], high.iloc[-1], low.iloc[-1], close.iloc[-1]

    body3 = abs(c3 - o3)
    upper_shadow3 = h3 - max(o3, c3)
    lower_shadow3 = min(o3, c3) - l3
    total_range3 = h3 - l3

    patterns = {
        "锤子线": lambda: (body3 > 0 and lower_shadow3 >= body3 * 2
                           and upper_shadow3 <= body3 * 0.3 and c2 < o2),
        "上吊线": lambda: (body3 > 0 and lower_shadow3 >= body3 * 2
                           and upper_shadow3 <= body3 * 0.3 and c2 > o2),
        "倒锤头": lambda: (body3 > 0 and upper_shadow3 >= body3 * 2
                           and lower_shadow3 <= body3 * 0.3 and c2 < o2),
        "射击之星": lambda: (body3 > 0 and upper_shadow3 >= body3 * 2
                             and lower_shadow3 <= body3 * 0.3 and c2 > o2),
        "十字星": lambda: (total_range3 > 0 and body3 <= total_range3 * 0.1),
        "吞没形态": lambda: ((c2 < o2 and c3 > o3 and o3 <= c2 and c3 >= o2)
                             or (c2 > o2 and c3 < o3 and o3 >= c2 and c3 <= o2)),
        "启明星": lambda: (c1 < o1 and abs(c2 - o2) <= (h2 - l2) * 0.3
                           and c3 > o3 and c3 > (o1 + c1) / 2),
        "黄昏之星": lambda: (c1 > o1 and abs(c2 - o2) <= (h2 - l2) * 0.3
                             and c3 < o3 and c3 < (o1 + c1) / 2),
        "三个白兵": lambda: (c1 > o1 and c2 > o2 and c3 > o3
                             and c2 > c1 and c3 > c2),
        "三只乌鸦": lambda: (c1 < o1 and c2 < o2 and c3 < o3
                             and c2 < c1 and c3 < c2),
    }

    if pattern_name in patterns:
        try:
            return patterns[pattern_name]()
        except Exception:
            return False
    return False


def _evaluate_condition(condition, df_kline, spot_row=None):
    """
    评估单个条件是否满足
    返回: (是否满足, 当前值描述)
    """
    cond_type = condition.get("type", "")
    params = condition.get("params", {})

    if df_kline is None or len(df_kline) < 5:
        return False, "K线数据不足"

    close = df_kline['收盘'] if '收盘' in df_kline.columns else df_kline.get('close')
    high = df_kline['最高'] if '最高' in df_kline.columns else df_kline.get('high')
    low = df_kline['最低'] if '最低' in df_kline.columns else df_kline.get('low')
    open_p = df_kline['开盘'] if '开盘' in df_kline.columns else df_kline.get('open')

    if close is None or len(close) < 5:
        return False, "数据不足"

    latest = close.iloc[-1]

    try:
        if cond_type == "price_above":
            value = float(params.get("value", 0))
            return latest > value, f"当前价{latest:.2f} > {value}"

        elif cond_type == "price_below":
            value = float(params.get("value", 0))
            return latest < value, f"当前价{latest:.2f} < {value}"

        elif cond_type == "change_above":
            value = float(params.get("value", 0))
            if spot_row is not None:
                chg = float(spot_row.get('涨跌幅', 0)) if pd.notna(spot_row.get('涨跌幅')) else 0
            else:
                prev_close = close.iloc[-2] if len(close) >= 2 else close.iloc[-1]
                chg = (latest / prev_close - 1) * 100 if prev_close > 0 else 0
            return chg > value, f"涨跌幅{chg:.2f}% > {value}%"

        elif cond_type == "change_below":
            value = float(params.get("value", 0))
            if spot_row is not None:
                chg = float(spot_row.get('涨跌幅', 0)) if pd.notna(spot_row.get('涨跌幅')) else 0
            else:
                prev_close = close.iloc[-2] if len(close) >= 2 else close.iloc[-1]
                chg = (latest / prev_close - 1) * 100 if prev_close > 0 else 0
            return chg < value, f"涨跌幅{chg:.2f}% < {value}%"

        elif cond_type == "volume_ratio_above":
            value = float(params.get("value", 0))
            if spot_row is not None:
                vr = float(spot_row.get('量比', 1)) if pd.notna(spot_row.get('量比')) else 1
            else:
                volume = df_kline.get('成交量', df_kline.get('volume'))
                if volume is not None and len(volume) >= 5:
                    vr = float(volume.iloc[-1] / volume.tail(5).mean()) if volume.tail(5).mean() > 0 else 1
                else:
                    vr = 1
            return vr > value, f"量比{vr:.2f} > {value}"

        elif cond_type == "volume_ratio_below":
            value = float(params.get("value", 0))
            if spot_row is not None:
                vr = float(spot_row.get('量比', 1)) if pd.notna(spot_row.get('量比')) else 1
            else:
                volume = df_kline.get('成交量', df_kline.get('volume'))
                if volume is not None and len(volume) >= 5:
                    vr = float(volume.iloc[-1] / volume.tail(5).mean()) if volume.tail(5).mean() > 0 else 1
                else:
                    vr = 1
            return vr < value, f"量比{vr:.2f} < {value}"

        elif cond_type == "amplitude_above":
            value = float(params.get("value", 0))
            if spot_row is not None:
                amp = float(spot_row.get('振幅', 0)) if pd.notna(spot_row.get('振幅')) else 0
            else:
                amp = (high.iloc[-1] - low.iloc[-1]) / close.iloc[-2] * 100 if len(close) >= 2 else 0
            return amp > value, f"振幅{amp:.2f}% > {value}%"

        elif cond_type == "rsi_above":
            period = int(params.get("period", 14))
            value = float(params.get("value", 70))
            rsi_series = _calc_rsi(close, period)
            rsi_val = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50
            return rsi_val > value, f"RSI({period})={rsi_val:.1f} > {value}"

        elif cond_type == "rsi_below":
            period = int(params.get("period", 14))
            value = float(params.get("value", 30))
            rsi_series = _calc_rsi(close, period)
            rsi_val = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50
            return rsi_val < value, f"RSI({period})={rsi_val:.1f} < {value}"

        elif cond_type == "ma_golden_cross":
            short_p = int(params.get("short_period", 5))
            long_p = int(params.get("long_period", 20))
            if len(close) < long_p + 1:
                return False, "数据不足"
            ma_short = close.rolling(short_p).mean()
            ma_long = close.rolling(long_p).mean()
            cross_today = ma_short.iloc[-1] > ma_long.iloc[-1]
            cross_yesterday = ma_short.iloc[-2] <= ma_long.iloc[-2]
            return cross_today and cross_yesterday, f"MA{short_p}({ma_short.iloc[-1]:.2f})上穿MA{long_p}({ma_long.iloc[-1]:.2f})"

        elif cond_type == "ma_death_cross":
            short_p = int(params.get("short_period", 5))
            long_p = int(params.get("long_period", 20))
            if len(close) < long_p + 1:
                return False, "数据不足"
            ma_short = close.rolling(short_p).mean()
            ma_long = close.rolling(long_p).mean()
            cross_today = ma_short.iloc[-1] < ma_long.iloc[-1]
            cross_yesterday = ma_short.iloc[-2] >= ma_long.iloc[-2]
            return cross_today and cross_yesterday, f"MA{short_p}({ma_short.iloc[-1]:.2f})下穿MA{long_p}({ma_long.iloc[-1]:.2f})"

        elif cond_type == "macd_golden_cross":
            macd_line, signal_line, _ = _calc_macd(close)
            cross_today = macd_line.iloc[-1] > signal_line.iloc[-1]
            cross_yesterday = macd_line.iloc[-2] <= signal_line.iloc[-2]
            return cross_today and cross_yesterday, f"MACD金叉(DIF={macd_line.iloc[-1]:.3f}, DEA={signal_line.iloc[-1]:.3f})"

        elif cond_type == "macd_death_cross":
            macd_line, signal_line, _ = _calc_macd(close)
            cross_today = macd_line.iloc[-1] < signal_line.iloc[-1]
            cross_yesterday = macd_line.iloc[-2] >= signal_line.iloc[-2]
            return cross_today and cross_yesterday, f"MACD死叉(DIF={macd_line.iloc[-1]:.3f}, DEA={signal_line.iloc[-1]:.3f})"

        elif cond_type == "price_above_ma":
            period = int(params.get("period", 20))
            if len(close) < period:
                return False, "数据不足"
            ma = close.rolling(period).mean().iloc[-1]
            return latest > ma, f"当前价{latest:.2f} > MA{period}({ma:.2f})"

        elif cond_type == "price_below_ma":
            period = int(params.get("period", 20))
            if len(close) < period:
                return False, "数据不足"
            ma = close.rolling(period).mean().iloc[-1]
            return latest < ma, f"当前价{latest:.2f} < MA{period}({ma:.2f})"

        elif cond_type == "new_high_n":
            days = int(params.get("days", 60))
            if len(close) < days:
                return False, "数据不足"
            period_high = high.rolling(days).max().iloc[-2]
            return latest > period_high, f"创{days}日新高(前高{period_high:.2f}, 当前{latest:.2f})"

        elif cond_type == "new_low_n":
            days = int(params.get("days", 60))
            if len(close) < days:
                return False, "数据不足"
            period_low = low.rolling(days).min().iloc[-2]
            return latest < period_low, f"创{days}日新低(前低{period_low:.2f}, 当前{latest:.2f})"

        elif cond_type == "consecutive_up":
            days = int(params.get("days", 5))
            if len(close) < days + 1:
                return False, "数据不足"
            returns = close.pct_change().dropna().tail(days)
            return (returns > 0).sum() >= days, f"连续上涨{days}日"

        elif cond_type == "consecutive_down":
            days = int(params.get("days", 5))
            if len(close) < days + 1:
                return False, "数据不足"
            returns = close.pct_change().dropna().tail(days)
            return (returns < 0).sum() >= days, f"连续下跌{days}日"

        elif cond_type == "turnover_above":
            value = float(params.get("value", 10))
            if spot_row is not None:
                to = float(spot_row.get('换手率', 0)) if pd.notna(spot_row.get('换手率')) else 0
            else:
                to = 0
            return to > value, f"换手率{to:.2f}% > {value}%"

        elif cond_type == "pe_below":
            value = float(params.get("value", 15))
            if spot_row is not None:
                pe = float(spot_row.get('市盈率-动态', 0)) if pd.notna(spot_row.get('市盈率-动态')) else None
            else:
                pe = None
            if pe is None:
                return False, "PE数据不可用"
            return 0 < pe < value, f"PE={pe:.2f} < {value}"

        elif cond_type == "pe_above":
            value = float(params.get("value", 50))
            if spot_row is not None:
                pe = float(spot_row.get('市盈率-动态', 0)) if pd.notna(spot_row.get('市盈率-动态')) else None
            else:
                pe = None
            if pe is None:
                return False, "PE数据不可用"
            return pe > value, f"PE={pe:.2f} > {value}"

        elif cond_type == "pb_below":
            value = float(params.get("value", 1))
            if spot_row is not None:
                pb = float(spot_row.get('市净率', 0)) if pd.notna(spot_row.get('市净率')) else None
            else:
                pb = None
            if pb is None:
                return False, "PB数据不可用"
            return 0 < pb < value, f"PB={pb:.2f} < {value}"

        elif cond_type == "market_cap_above":
            value = float(params.get("value", 100))
            if spot_row is not None:
                mc = float(spot_row.get('总市值', 0)) if pd.notna(spot_row.get('总市值')) else 0
            else:
                mc = 0
            mc_yi = mc / 1e8
            return mc_yi > value, f"总市值{mc_yi:.0f}亿 > {value}亿"

        elif cond_type == "pattern_detected":
            pattern_name = params.get("pattern_name", "")
            if not pattern_name:
                return False, "未指定形态名称"
            detected = _detect_kline_pattern(open_p, high, low, close, pattern_name)
            return detected, f"检测到'{pattern_name}'形态" if detected else f"未检测到'{pattern_name}'形态"

        else:
            return False, f"未知条件类型: {cond_type}"

    except Exception as e:
        return False, f"条件评估异常: {str(e)}"


# ==================== 规则管理 ====================

def add_rule(name, symbol, conditions, logic="AND", severity="warning",
             enabled=True, description=""):
    """
    添加预警规则

    参数:
        name: 规则名称
        symbol: 监控的股票代码（支持 * 表示全市场）
        conditions: 条件列表 [{"type": "price_above", "params": {"value": 100}}, ...]
        logic: 条件组合逻辑 AND/OR
        severity: 严重级别 info/warning/critical
        enabled: 是否启用
        description: 规则描述
    """
    rules_data = _load_rules()

    # 验证条件
    for cond in conditions:
        cond_type = cond.get("type", "")
        if cond_type not in CONDITION_TYPES:
            return {"error": f"不支持的条件类型: {cond_type}，可用类型: {list(CONDITION_TYPES.keys())}"}

    rule = {
        "id": rules_data["next_id"],
        "名称": name,
        "股票代码": symbol,
        "条件": conditions,
        "逻辑": logic,
        "严重级别": severity,
        "启用": enabled,
        "描述": description,
        "创建时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "触发次数": 0,
        "最后触发": None,
    }

    rules_data["rules"].append(rule)
    rules_data["next_id"] += 1
    _save_rules(rules_data)

    return {"success": True, "规则ID": rule["id"], "规则": rule}


def list_rules(enabled_only=False):
    """列出所有规则"""
    rules_data = _load_rules()
    rules = rules_data["rules"]

    if enabled_only:
        rules = [r for r in rules if r.get("启用", True)]

    return {
        "规则总数": len(rules),
        "规则列表": rules,
    }


def delete_rule(rule_id):
    """删除规则"""
    rules_data = _load_rules()
    original_count = len(rules_data["rules"])
    rules_data["rules"] = [r for r in rules_data["rules"] if r["id"] != rule_id]

    if len(rules_data["rules"]) == original_count:
        return {"error": f"未找到规则ID: {rule_id}"}

    _save_rules(rules_data)
    return {"success": True, "message": f"规则 {rule_id} 已删除"}


def toggle_rule(rule_id, enabled=None):
    """启用/禁用规则"""
    rules_data = _load_rules()
    for rule in rules_data["rules"]:
        if rule["id"] == rule_id:
            if enabled is None:
                rule["启用"] = not rule.get("启用", True)
            else:
                rule["启用"] = enabled
            _save_rules(rules_data)
            return {"success": True, "规则ID": rule_id, "启用": rule["启用"]}
    return {"error": f"未找到规则ID: {rule_id}"}


# ==================== 规则评估 ====================

def evaluate_rule(rule, symbol=None):
    """
    评估单条规则是否触发
    返回: (是否触发, 触发详情)
    """
    if not rule.get("启用", True):
        return False, {"原因": "规则已禁用"}

    rule_symbol = rule.get("股票代码", "")
    if symbol and rule_symbol != "*" and rule_symbol != symbol:
        return False, {"原因": f"规则监控{symbol}，当前评估{rule_symbol}"}

    target_symbol = symbol or rule_symbol
    if target_symbol == "*":
        return False, {"原因": "全市场规则需要指定具体股票"}

    # 获取K线数据
    df_kline = get_stock_kline(target_symbol, days=120)
    spot_row = None

    # 尝试获取实时行情
    try:
        df_spot = _get_spot_df()
        if df_spot is not None and not df_spot.empty:
            matched = df_spot[df_spot['代码'] == target_symbol]
            if not matched.empty:
                spot_row = matched.iloc[0]
    except Exception:
        pass

    conditions = rule.get("条件", [])
    logic = rule.get("逻辑", "AND")

    condition_results = []
    for cond in conditions:
        satisfied, desc = _evaluate_condition(cond, df_kline, spot_row)
        condition_results.append({
            "条件类型": cond.get("type", ""),
            "参数": cond.get("params", {}),
            "满足": satisfied,
            "描述": desc,
        })

    if logic == "AND":
        triggered = all(r["满足"] for r in condition_results)
    elif logic == "OR":
        triggered = any(r["满足"] for r in condition_results)
    else:
        triggered = False

    detail = {
        "规则名称": rule.get("名称", ""),
        "股票代码": target_symbol,
        "组合逻辑": logic,
        "触发": triggered,
        "严重级别": rule.get("严重级别", "warning"),
        "条件详情": condition_results,
    }

    # 更新触发统计
    if triggered:
        rules_data = _load_rules()
        for r in rules_data["rules"]:
            if r["id"] == rule["id"]:
                r["触发次数"] = r.get("触发次数", 0) + 1
                r["最后触发"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                break
        _save_rules(rules_data)

    return triggered, detail


def evaluate_all_rules(symbol=None):
    """评估所有启用的规则"""
    rules_data = _load_rules()
    enabled_rules = [r for r in rules_data["rules"] if r.get("启用", True)]

    results = []
    triggered_count = 0

    for rule in enabled_rules:
        triggered, detail = evaluate_rule(rule, symbol)
        results.append(detail)
        if triggered:
            triggered_count += 1

    # 按严重级别排序
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    results.sort(key=lambda x: severity_order.get(x.get("严重级别", "info"), 2))

    return {
        "评估时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "评估规则数": len(results),
        "触发规则数": triggered_count,
        "触发率": f"{triggered_count/len(results)*100:.1f}%" if results else "0%",
        "规则结果": results,
    }


# ==================== 规则回测 ====================

def backtest_rule(rule_id, symbol=None, days=250):
    """
    对规则进行历史回测
    统计规则在历史上的触发频率和触发后的市场表现
    """
    rules_data = _load_rules()
    rule = None
    for r in rules_data["rules"]:
        if r["id"] == rule_id:
            rule = r
            break

    if rule is None:
        return {"error": f"未找到规则ID: {rule_id}"}

    target_symbol = symbol or rule.get("股票代码", "")
    if target_symbol == "*":
        return {"error": "全市场规则需要指定具体股票进行回测"}

    df_kline = get_stock_kline(target_symbol, days=days + 120)
    if df_kline is None or len(df_kline) < 60:
        return {"error": f"{target_symbol} K线数据不足"}

    close = df_kline['收盘'] if '收盘' in df_kline.columns else df_kline['close']
    conditions = rule.get("条件", [])
    logic = rule.get("逻辑", "AND")

    # 逐日回测
    trigger_dates = []
    trigger_signals = []

    for i in range(60, len(close)):
        sub_df = df_kline.iloc[:i+1].copy()

        cond_results = []
        for cond in conditions:
            satisfied, _ = _evaluate_condition(cond, sub_df, None)
            cond_results.append(satisfied)

        if logic == "AND":
            triggered = all(cond_results)
        else:
            triggered = any(cond_results)

        if triggered:
            date_idx = df_kline.index[i] if hasattr(df_kline, 'index') else i
            trigger_dates.append(str(date_idx)[:10] if hasattr(date_idx, 'strftime') else str(date_idx))
            trigger_signals.append(1)
        else:
            trigger_signals.append(0)

    # 统计
    total_signals = sum(trigger_signals)
    total_days = len(trigger_signals)

    # 计算触发后的N日收益
    forward_returns = {}
    for horizon in [1, 3, 5, 10, 20]:
        returns_list = []
        for i in range(len(trigger_signals) - horizon):
            if trigger_signals[i] == 1:
                ret = (close.iloc[i + horizon] / close.iloc[i] - 1) * 100
                returns_list.append(round(float(ret), 2))
        if returns_list:
            forward_returns[f"{horizon}日后"] = {
                "平均收益": f"{np.mean(returns_list):.2f}%",
                "胜率": f"{sum(1 for r in returns_list if r > 0)/len(returns_list)*100:.1f}%",
                "最大收益": f"{max(returns_list):.2f}%",
                "最大亏损": f"{min(returns_list):.2f}%",
                "样本数": len(returns_list),
            }

    return {
        "规则名称": rule.get("名称", ""),
        "股票代码": target_symbol,
        "回测区间": f"{trigger_dates[0] if trigger_dates else '--'} 至 {trigger_dates[-1] if trigger_dates else '--'}",
        "总交易日": total_days,
        "触发次数": total_signals,
        "触发频率": f"{total_signals/total_days*100:.2f}%" if total_days > 0 else "0%",
        "平均间隔": f"{total_days/max(total_signals,1):.0f}天",
        "触发后表现": forward_returns,
        "触发日期": trigger_dates[-20:] if len(trigger_dates) > 20 else trigger_dates,
    }


# ==================== 批量扫描 ====================

def scan_market(rule_id, limit=50):
    """
    用指定规则扫描全市场
    找出所有触发该规则的股票
    """
    rules_data = _load_rules()
    rule = None
    for r in rules_data["rules"]:
        if r["id"] == rule_id:
            rule = r
            break

    if rule is None:
        return {"error": f"未找到规则ID: {rule_id}"}

    try:
        df_spot = _get_spot_df()
        if df_spot is None or df_spot.empty:
            return {"error": "无法获取实时行情"}

        # 过滤ST
        if '名称' in df_spot.columns:
            df_spot = df_spot[~df_spot['名称'].str.contains('ST', na=False)]

        # 按成交额排序取前N只
        if '成交额' in df_spot.columns:
            df_spot['成交额'] = pd.to_numeric(df_spot['成交额'], errors='coerce')
            df_spot = df_spot.sort_values('成交额', ascending=False)

        df_spot = df_spot.head(limit)

        triggered_stocks = []
        total_checked = 0

        for _, row in df_spot.iterrows():
            code = str(row.get('代码', ''))
            if not code:
                continue

            total_checked += 1
            df_kline = get_stock_kline(code, days=120)
            if df_kline is None or len(df_kline) < 30:
                continue

            conditions = rule.get("条件", [])
            logic = rule.get("逻辑", "AND")

            cond_results = []
            for cond in conditions:
                satisfied, desc = _evaluate_condition(cond, df_kline, row)
                cond_results.append({"满足": satisfied, "描述": desc})

            if logic == "AND":
                triggered = all(r["满足"] for r in cond_results)
            else:
                triggered = any(r["满足"] for r in cond_results)

            if triggered:
                triggered_stocks.append({
                    "代码": code,
                    "名称": str(row.get('名称', '')),
                    "最新价": float(row.get('最新价', 0)) if pd.notna(row.get('最新价')) else 0,
                    "涨跌幅": float(row.get('涨跌幅', 0)) if pd.notna(row.get('涨跌幅')) else 0,
                    "条件详情": cond_results,
                })

            time.sleep(0.05)

        return {
            "规则名称": rule.get("名称", ""),
            "扫描时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "扫描股票数": total_checked,
            "触发股票数": len(triggered_stocks),
            "触发率": f"{len(triggered_stocks)/max(total_checked,1)*100:.1f}%",
            "触发股票": triggered_stocks,
        }

    except Exception as e:
        return {"error": f"全市场扫描出错: {str(e)}"}


def main():
    parser = argparse.ArgumentParser(description='智能预警规则引擎')
    parser.add_argument('action', choices=[
        'add_rule', 'list_rules', 'delete_rule', 'toggle_rule',
        'evaluate', 'evaluate_all', 'backtest', 'scan_market',
        'list_conditions'
    ], help='操作类型')

    parser.add_argument('--name', type=str, help='规则名称')
    parser.add_argument('--symbol', type=str, help='股票代码')
    parser.add_argument('--conditions', type=str, help='条件JSON字符串')
    parser.add_argument('--logic', default='AND', choices=['AND', 'OR'], help='条件组合逻辑')
    parser.add_argument('--severity', default='warning', choices=['info', 'warning', 'critical'], help='严重级别')
    parser.add_argument('--description', type=str, default='', help='规则描述')
    parser.add_argument('--rule_id', type=int, help='规则ID')
    parser.add_argument('--enabled', type=str, help='启用状态 true/false')
    parser.add_argument('--days', type=int, default=250, help='回测天数')
    parser.add_argument('--limit', type=int, default=50, help='扫描股票数量限制')
    parser.add_argument('--enabled_only', action='store_true', help='仅列出启用的规则')

    args = parser.parse_args()

    try:
        if args.action == 'list_conditions':
            data = {
                "支持的条件类型": [
                    {"类型": k, "描述": v["描述"], "参数": v["参数"], "示例": v["示例"]}
                    for k, v in CONDITION_TYPES.items()
                ]
            }
            print(json.dumps(data, ensure_ascii=False, indent=2))

        elif args.action == 'add_rule':
            if not args.name or not args.symbol or not args.conditions:
                print(json.dumps({"error": "缺少必要参数: --name, --symbol, --conditions"}, ensure_ascii=False))
                sys.exit(1)

            conditions = json.loads(args.conditions)
            data = add_rule(
                name=args.name, symbol=args.symbol,
                conditions=conditions, logic=args.logic,
                severity=args.severity, description=args.description
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))

        elif args.action == 'list_rules':
            data = list_rules(enabled_only=args.enabled_only)
            print(json.dumps(data, ensure_ascii=False, indent=2))

        elif args.action == 'delete_rule':
            if not args.rule_id:
                print(json.dumps({"error": "缺少 --rule_id"}, ensure_ascii=False))
                sys.exit(1)
            data = delete_rule(args.rule_id)
            print(json.dumps(data, ensure_ascii=False, indent=2))

        elif args.action == 'toggle_rule':
            if not args.rule_id:
                print(json.dumps({"error": "缺少 --rule_id"}, ensure_ascii=False))
                sys.exit(1)
            enabled = None
            if args.enabled is not None:
                enabled = args.enabled.lower() == 'true'
            data = toggle_rule(args.rule_id, enabled)
            print(json.dumps(data, ensure_ascii=False, indent=2))

        elif args.action == 'evaluate':
            if not args.rule_id:
                print(json.dumps({"error": "缺少 --rule_id"}, ensure_ascii=False))
                sys.exit(1)
            rules_data = _load_rules()
            rule = None
            for r in rules_data["rules"]:
                if r["id"] == args.rule_id:
                    rule = r
                    break
            if rule is None:
                print(json.dumps({"error": f"未找到规则ID: {args.rule_id}"}, ensure_ascii=False))
                sys.exit(1)
            triggered, detail = evaluate_rule(rule, args.symbol)
            print(json.dumps({"触发": triggered, "详情": detail}, ensure_ascii=False, indent=2))

        elif args.action == 'evaluate_all':
            data = evaluate_all_rules(args.symbol)
            print(json.dumps(data, ensure_ascii=False, indent=2))

        elif args.action == 'backtest':
            if not args.rule_id:
                print(json.dumps({"error": "缺少 --rule_id"}, ensure_ascii=False))
                sys.exit(1)
            data = backtest_rule(args.rule_id, args.symbol, args.days)
            print(json.dumps(data, ensure_ascii=False, indent=2))

        elif args.action == 'scan_market':
            if not args.rule_id:
                print(json.dumps({"error": "缺少 --rule_id"}, ensure_ascii=False))
                sys.exit(1)
            data = scan_market(args.rule_id, args.limit)
            print(json.dumps(data, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
