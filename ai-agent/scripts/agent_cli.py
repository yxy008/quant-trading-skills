#!/usr/bin/env python3
"""
AI Agent 增强模块
提供自然语言→策略代码、回测结果解读、异常诊断三大能力
支持 OpenAI 兼容 API 调用，内置规则引擎作为降级方案
"""
import os
import re
import json
import sys
from datetime import datetime

# 添加策略框架和回测模块路径
SKILLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
AGENT_DIR = os.path.dirname(SKILLS_DIR)
sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, os.path.join(SKILLS_DIR, "strategy-framework", "scripts"))
sys.path.insert(0, os.path.join(SKILLS_DIR, "backtest", "scripts"))
sys.path.insert(0, os.path.join(SKILLS_DIR, "multi-factor-model", "scripts"))
sys.path.insert(0, os.path.join(SKILLS_DIR, "factor-library", "scripts"))
sys.path.insert(0, os.path.join(SKILLS_DIR, "industry-rotation", "scripts"))

# AI 配置文件路径
AI_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ai_config.json")

# 默认配置
DEFAULT_CONFIG = {
    "api_base": "https://api.openai.com/v1",
    "api_key": "",
    "model": "gpt-3.5-turbo",
    "enabled": False,
    "max_tokens": 2000,
    "temperature": 0.3
}


def load_config():
    """加载 AI 配置"""
    try:
        if os.path.exists(AI_CONFIG_PATH):
            with open(AI_CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                merged = {**DEFAULT_CONFIG, **config}
                return merged
    except Exception:
        pass
    return DEFAULT_CONFIG.copy()


def save_config(config):
    """保存 AI 配置"""
    os.makedirs(os.path.dirname(AI_CONFIG_PATH), exist_ok=True)
    with open(AI_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def call_llm(system_prompt, user_prompt, timeout=20):
    """调用 LLM API"""
    config = load_config()
    if not config.get("enabled") or not config.get("api_key"):
        return None

    try:
        import urllib.request
        import urllib.error

        api_base = config["api_base"].rstrip("/")
        url = f"{api_base}/chat/completions"

        payload = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": config.get("max_tokens", 2000),
            "temperature": config.get("temperature", 0.3)
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {config['api_key']}")

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            return content
    except Exception as e:
        return None


# ==================== 能力一：自然语言 → 策略代码 ====================

# 技术指标关键词映射
INDICATOR_PATTERNS = {
    "ma_cross": {
        "keywords": ["均线", "MA", "ma", "移动平均"],
        "patterns": [
            (r"(\d+)日均线[上穿|突破|金叉].*?(\d+)日均线", "ma_cross"),
            (r"MA(\d+).*?[上穿|突破|金叉].*?MA(\d+)", "ma_cross"),
            (r"(\d+)日线[上穿|突破].*?(\d+)日线", "ma_cross"),
        ]
    },
    "macd": {
        "keywords": ["MACD", "macd", "金叉", "死叉", "DIF", "DEA"],
        "patterns": [
            (r"MACD.*?金叉", "macd"),
            (r"MACD.*?DIF.*?DEA", "macd"),
        ]
    },
    "rsi": {
        "keywords": ["RSI", "rsi", "超买", "超卖", "相对强弱"],
        "patterns": [
            (r"RSI.*?(\d+)", "rsi"),
            (r"RSI.*?超卖", "rsi"),
            (r"RSI.*?超买", "rsi"),
        ]
    },
    "bollinger": {
        "keywords": ["布林", "BOLL", "boll", "布林带", "上轨", "下轨", "中轨"],
        "patterns": [
            (r"布林带.*?下轨", "bollinger"),
            (r"布林带.*?上轨", "bollinger"),
            (r"BOLL.*?突破", "bollinger"),
        ]
    },
    "volume_breakout": {
        "keywords": ["放量", "成交量", "量能", "放量突破", "量比"],
        "patterns": [
            (r"放量.*?突破", "volume_breakout"),
            (r"成交量.*?放大", "volume_breakout"),
            (r"放量.*?(\d+)倍", "volume_breakout"),
        ]
    },
    "multi_factor": {
        "keywords": ["多因子", "综合", "多条件"],
        "patterns": [
            (r"多.*?条件", "multi_factor"),
            (r"综合.*?策略", "multi_factor"),
        ]
    }
}

# 条件关键词映射
CONDITION_KEYWORDS = {
    "and": ["且", "同时", "并且", "而且", "以及", "and", "AND"],
    "or": ["或", "或者", "or", "OR"],
    "buy": ["买入", "买", "做多", "开仓", "建仓", "进场"],
    "sell": ["卖出", "卖", "做空", "平仓", "离场", "出场"],
    "volume_up": ["放量", "成交量放大", "量能放大", "放量上涨", "量增"],
    "price_up": ["上涨", "涨", "突破", "上穿", "站上"],
    "price_down": ["下跌", "跌", "跌破", "下穿", "破位"],
}


def parse_nl_strategy(nl_text):
    """
    解析自然语言策略描述，提取关键要素
    返回结构化的策略定义
    """
    result = {
        "原始描述": nl_text,
        "解析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "识别指标": [],
        "买入条件": [],
        "卖出条件": [],
        "参数": {},
        "推荐策略": None,
        "置信度": "低"
    }

    # 1. 识别技术指标
    for indicator_id, info in INDICATOR_PATTERNS.items():
        matched = False
        for kw in info["keywords"]:
            if kw.lower() in nl_text.lower():
                matched = True
                break
        if not matched:
            for pattern, _ in info["patterns"]:
                if re.search(pattern, nl_text):
                    matched = True
                    break
        if matched:
            result["识别指标"].append(indicator_id)

    # 2. 提取参数
    # 均线周期
    ma_periods = re.findall(r'(\d+)日(?:均线|线|MA)', nl_text)
    if ma_periods:
        periods = [int(p) for p in ma_periods]
        if len(periods) >= 2:
            result["参数"]["ma_fast"] = min(periods[:2])
            result["参数"]["ma_slow"] = max(periods[:2])
        elif len(periods) == 1:
            result["参数"]["ma_fast"] = periods[0]
            result["参数"]["ma_slow"] = 20

    # RSI 周期
    rsi_match = re.search(r'RSI[\(（]?(\d+)[\)）]?', nl_text, re.IGNORECASE)
    if rsi_match:
        result["参数"]["rsi_period"] = int(rsi_match.group(1))

    # RSI 阈值
    rsi_oversold = re.search(r'RSI.*?[<小于低于]?\s*(\d+)', nl_text, re.IGNORECASE)
    if rsi_oversold:
        val = int(rsi_oversold.group(1))
        if val < 50:
            result["参数"]["rsi_oversold"] = val

    # 放量倍数
    vol_match = re.search(r'放量\s*(\d+\.?\d*)\s*倍', nl_text)
    if vol_match:
        result["参数"]["volume_multiple"] = float(vol_match.group(1))

    # MACD 参数
    macd_fast = re.search(r'MACD.*?快.*?(\d+)', nl_text, re.IGNORECASE)
    macd_slow = re.search(r'MACD.*?慢.*?(\d+)', nl_text, re.IGNORECASE)
    if macd_fast:
        result["参数"]["fast_period"] = int(macd_fast.group(1))
    if macd_slow:
        result["参数"]["slow_period"] = int(macd_slow.group(1))

    # 3. 解析买入条件
    buy_conditions = _extract_conditions(nl_text, "buy")
    result["买入条件"] = buy_conditions

    # 4. 解析卖出条件
    sell_conditions = _extract_conditions(nl_text, "sell")
    result["卖出条件"] = sell_conditions

    # 5. 推荐策略
    if result["识别指标"]:
        # 优先推荐匹配度最高的策略
        if "ma_cross" in result["识别指标"] and "volume_breakout" in result["识别指标"]:
            result["推荐策略"] = "multi_factor"
            result["置信度"] = "高"
        elif "ma_cross" in result["识别指标"]:
            result["推荐策略"] = "ma_cross"
            result["置信度"] = "高"
        elif "macd" in result["识别指标"]:
            result["推荐策略"] = "macd"
            result["置信度"] = "高"
        elif "rsi" in result["识别指标"]:
            result["推荐策略"] = "rsi"
            result["置信度"] = "高"
        elif "bollinger" in result["识别指标"]:
            result["推荐策略"] = "bollinger"
            result["置信度"] = "高"
        elif "volume_breakout" in result["识别指标"]:
            result["推荐策略"] = "volume_breakout"
            result["置信度"] = "中"
        else:
            result["推荐策略"] = result["识别指标"][0]
            result["置信度"] = "中"

    return result


def _extract_conditions(text, direction="buy"):
    """从文本中提取买卖条件"""
    conditions = []

    # 分割句子
    sentences = re.split(r'[，,。；;、\n]', text)

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        cond = {"原文": sent, "类型": "unknown"}

        # 判断买卖方向
        is_buy = any(kw in sent for kw in CONDITION_KEYWORDS["buy"])
        is_sell = any(kw in sent for kw in CONDITION_KEYWORDS["sell"])

        if direction == "buy" and is_buy:
            cond["类型"] = "buy"
        elif direction == "sell" and is_sell:
            cond["类型"] = "sell"
        elif direction == "buy" and not is_sell:
            # 买入方向，没有明确卖出词，可能是买入条件
            if any(kw in sent for kw in ["上穿", "突破", "金叉", "站上", "放量"]):
                cond["类型"] = "buy"
        elif direction == "sell" and not is_buy:
            if any(kw in sent for kw in ["下穿", "跌破", "死叉", "破位"]):
                cond["类型"] = "sell"

        if cond["类型"] != "unknown":
            # 提取具体指标
            for ind_id, info in INDICATOR_PATTERNS.items():
                for kw in info["keywords"]:
                    if kw.lower() in sent.lower():
                        cond["指标"] = ind_id
                        break
                if "指标" in cond:
                    break

            conditions.append(cond)

    return conditions


def nl_to_strategy(nl_text):
    """
    自然语言 → 策略代码
    主入口：解析自然语言，生成可执行的策略定义和回测代码
    """
    # 先用规则引擎解析
    parsed = parse_nl_strategy(nl_text)

    # 尝试用 LLM 增强
    llm_result = None
    config = load_config()
    if config.get("enabled") and config.get("api_key"):
        system_prompt = """你是一个量化交易策略专家。用户会用自然语言描述一个A股交易策略，你需要将其转换为结构化的策略定义。

请返回 JSON 格式，包含以下字段：
- strategy_name: 策略名称（中文）
- strategy_id: 策略ID（使用英文下划线命名，如 ma_cross、macd_signal、volume_breakout）
- buy_conditions: 买入条件列表，每个条件包含 indicator(指标名), operator(运算符), value(值)
- sell_conditions: 卖出条件列表，结构同上
- params: 参数字典（如 {"fast_period": 5, "slow_period": 20}）
- explanation: 策略解释说明（50字以内）
- risk_note: 风险提示（30字以内）

支持的指标：MA(均线)、MACD、RSI、KDJ、BOLL(布林带)、VOL(成交量)、ATR(真实波幅)
支持的运算符：cross_above(上穿)、cross_below(下穿)、gt(大于)、lt(小于)、gte(大于等于)、lte(小于等于)

重要约束：
1. 参数必须在合理范围内（如均线周期5-250，RSI周期6-24，MACD快线6-26、慢线12-52）
2. 避免使用未来数据（如当日收盘价不能作为当日买入条件）
3. 买入和卖出条件必须成对出现，不能只有买入没有卖出
4. 考虑A股T+1制度，当日买入次日才能卖出
5. 考虑涨跌停限制（±10%，ST股±5%），触及涨跌停时可能无法成交

示例输入："5日均线上穿20日均线买入，下穿卖出"
示例输出：
{
    "strategy_name": "双均线交叉策略",
    "strategy_id": "ma_cross",
    "buy_conditions": [{"indicator": "MA5", "operator": "cross_above", "value": "MA20"}],
    "sell_conditions": [{"indicator": "MA5", "operator": "cross_below", "value": "MA20"}],
    "params": {"fast_period": 5, "slow_period": 20},
    "explanation": "短期均线上穿长期均线视为买入信号，下穿视为卖出信号",
    "risk_note": "震荡市中可能出现频繁假信号，建议配合趋势过滤"
}

只返回 JSON，不要有其他内容。"""
        llm_result = call_llm(system_prompt, nl_text)

    # 生成策略代码
    strategy_code = _generate_strategy_code(parsed, llm_result)

    # 生成回测代码
    backtest_code = _generate_backtest_code(parsed)

    result = {
        "解析结果": parsed,
        "策略代码": strategy_code,
        "回测代码": backtest_code,
        "LLM增强": llm_result is not None,
        "生成时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    return result


def _generate_strategy_code(parsed, llm_result=None):
    """根据解析结果生成策略代码"""
    strategy_id = parsed.get("推荐策略", "ma_cross")
    params = parsed.get("参数", {})
    buy_conditions = parsed.get("买入条件", [])
    indicators = parsed.get("识别指标", [])

    lines = []
    lines.append('"""')
    lines.append(f'自动生成的交易策略')
    lines.append(f'原始描述: {parsed.get("原始描述", "")}')
    lines.append(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'策略类型: {strategy_id}')
    lines.append('"""')
    lines.append('')
    lines.append('import numpy as np')
    lines.append('import pandas as pd')
    lines.append('')
    lines.append('')
    lines.append('def generate_signals(df):')
    lines.append('    """')
    lines.append('    根据K线数据生成买卖信号')
    lines.append('    ')
    lines.append('    参数:')
    lines.append('        df: DataFrame, 需包含 open, high, low, close, volume 列')
    lines.append('    ')
    lines.append('    返回:')
    lines.append('        DataFrame, 新增 signal 列 (1=买入, -1=卖出, 0=无信号)')
    lines.append('    """')
    lines.append('    df = df.copy()')
    lines.append('    df[\'signal\'] = 0')
    lines.append('')

    # 根据策略类型生成具体代码
    if strategy_id == "ma_cross":
        fast = params.get("ma_fast", 5)
        slow = params.get("ma_slow", 20)
        lines.append(f'    # 计算均线')
        lines.append(f'    df[\'ma_fast\'] = df[\'close\'].rolling(window={fast}).mean()')
        lines.append(f'    df[\'ma_slow\'] = df[\'close\'].rolling(window={slow}).mean()')
        lines.append('')
        lines.append('    # 金叉买入信号')
        lines.append(f'    golden_cross = (df[\'ma_fast\'] > df[\'ma_slow\']) & (df[\'ma_fast\'].shift(1) <= df[\'ma_slow\'].shift(1))')
        lines.append('    df.loc[golden_cross, \'signal\'] = 1')
        lines.append('')
        lines.append('    # 死叉卖出信号')
        lines.append(f'    dead_cross = (df[\'ma_fast\'] < df[\'ma_slow\']) & (df[\'ma_fast\'].shift(1) >= df[\'ma_slow\'].shift(1))')
        lines.append('    df.loc[dead_cross, \'signal\'] = -1')

        if "volume_breakout" in indicators:
            vol_mult = params.get("volume_multiple", 1.5)
            lines.append('')
            lines.append('    # 放量过滤：只保留放量的金叉信号')
            lines.append(f'    df[\'vol_ma\'] = df[\'volume\'].rolling(window=20).mean()')
            lines.append(f'    volume_filter = df[\'volume\'] > df[\'vol_ma\'] * {vol_mult}')
            lines.append('    df.loc[df[\'signal\'] == 1, \'signal\'] = df.loc[df[\'signal\'] == 1, \'signal\'] * volume_filter.astype(int)')

    elif strategy_id == "macd":
        fast = params.get("fast_period", 12)
        slow = params.get("slow_period", 26)
        signal_p = params.get("signal_period", 9)
        lines.append(f'    # 计算 MACD')
        lines.append(f'    df[\'ema_fast\'] = df[\'close\'].ewm(span={fast}, adjust=False).mean()')
        lines.append(f'    df[\'ema_slow\'] = df[\'close\'].ewm(span={slow}, adjust=False).mean()')
        lines.append(f'    df[\'dif\'] = df[\'ema_fast\'] - df[\'ema_slow\']')
        lines.append(f'    df[\'dea\'] = df[\'dif\'].ewm(span={signal_p}, adjust=False).mean()')
        lines.append(f'    df[\'macd_hist\'] = 2 * (df[\'dif\'] - df[\'dea\'])')
        lines.append('')
        lines.append('    # MACD 金叉买入')
        lines.append('    golden_cross = (df[\'dif\'] > df[\'dea\']) & (df[\'dif\'].shift(1) <= df[\'dea\'].shift(1))')
        lines.append('    df.loc[golden_cross, \'signal\'] = 1')
        lines.append('')
        lines.append('    # MACD 死叉卖出')
        lines.append('    dead_cross = (df[\'dif\'] < df[\'dea\']) & (df[\'dif\'].shift(1) >= df[\'dea\'].shift(1))')
        lines.append('    df.loc[dead_cross, \'signal\'] = -1')

    elif strategy_id == "rsi":
        period = params.get("rsi_period", 14)
        oversold = params.get("rsi_oversold", 30)
        overbought = params.get("rsi_overbought", 70)
        lines.append(f'    # 计算 RSI')
        lines.append(f'    delta = df[\'close\'].diff()')
        lines.append(f'    gain = delta.where(delta > 0, 0)')
        lines.append(f'    loss = -delta.where(delta < 0, 0)')
        lines.append(f'    avg_gain = gain.rolling(window={period}).mean()')
        lines.append(f'    avg_loss = loss.rolling(window={period}).mean()')
        lines.append(f'    rs = avg_gain / avg_loss')
        lines.append(f'    df[\'rsi\'] = 100 - (100 / (1 + rs))')
        lines.append('')
        lines.append(f'    # RSI 超卖买入')
        lines.append(f'    df.loc[df[\'rsi\'] < {oversold}, \'signal\'] = 1')
        lines.append('')
        lines.append(f'    # RSI 超买卖出')
        lines.append(f'    df.loc[df[\'rsi\'] > {overbought}, \'signal\'] = -1')

    elif strategy_id == "bollinger":
        period = params.get("period", 20)
        std = params.get("std_dev", 2)
        lines.append(f'    # 计算布林带')
        lines.append(f'    df[\'ma\'] = df[\'close\'].rolling(window={period}).mean()')
        lines.append(f'    df[\'std\'] = df[\'close\'].rolling(window={period}).std()')
        lines.append(f'    df[\'upper\'] = df[\'ma\'] + {std} * df[\'std\']')
        lines.append(f'    df[\'lower\'] = df[\'ma\'] - {std} * df[\'std\']')
        lines.append('')
        lines.append('    # 触及下轨买入')
        lines.append('    df.loc[df[\'close\'] <= df[\'lower\'], \'signal\'] = 1')
        lines.append('')
        lines.append('    # 触及上轨卖出')
        lines.append('    df.loc[df[\'close\'] >= df[\'upper\'], \'signal\'] = -1')

    elif strategy_id == "volume_breakout":
        vol_mult = params.get("volume_multiple", 1.5)
        lines.append(f'    # 放量突破策略')
        lines.append(f'    df[\'vol_ma\'] = df[\'volume\'].rolling(window=20).mean()')
        lines.append(f'    df[\'price_ma\'] = df[\'close\'].rolling(window=20).mean()')
        lines.append('')
        lines.append(f'    # 放量且价格突破均线 → 买入')
        lines.append(f'    volume_cond = df[\'volume\'] > df[\'vol_ma\'] * {vol_mult}')
        lines.append(f'    price_cond = df[\'close\'] > df[\'price_ma\']')
        lines.append(f'    df.loc[volume_cond & price_cond, \'signal\'] = 1')
        lines.append('')
        lines.append(f'    # 缩量或跌破均线 → 卖出')
        lines.append(f'    df.loc[df[\'close\'] < df[\'price_ma\'], \'signal\'] = -1')

    elif strategy_id == "multi_factor":
        lines.append('    # 多因子综合策略')
        lines.append('    # 均线')
        fast = params.get("ma_fast", 5)
        slow = params.get("ma_slow", 20)
        lines.append(f'    df[\'ma_fast\'] = df[\'close\'].rolling(window={fast}).mean()')
        lines.append(f'    df[\'ma_slow\'] = df[\'close\'].rolling(window={slow}).mean()')
        lines.append('')
        lines.append('    # RSI')
        rsi_p = params.get("rsi_period", 14)
        lines.append(f'    delta = df[\'close\'].diff()')
        lines.append(f'    gain = delta.where(delta > 0, 0).rolling(window={rsi_p}).mean()')
        lines.append(f'    loss = (-delta.where(delta < 0, 0)).rolling(window={rsi_p}).mean()')
        lines.append(f'    df[\'rsi\'] = 100 - (100 / (1 + gain / loss))')
        lines.append('')
        lines.append('    # 成交量')
        lines.append(f'    df[\'vol_ma\'] = df[\'volume\'].rolling(window=20).mean()')
        vol_mult = params.get("volume_multiple", 1.5)
        lines.append('')
        lines.append('    # 综合买入条件：均线金叉 + RSI不超买 + 放量')
        lines.append(f'    ma_cond = (df[\'ma_fast\'] > df[\'ma_slow\']) & (df[\'ma_fast\'].shift(1) <= df[\'ma_slow\'].shift(1))')
        lines.append(f'    rsi_cond = df[\'rsi\'] < 70')
        lines.append(f'    vol_cond = df[\'volume\'] > df[\'vol_ma\'] * {vol_mult}')
        lines.append(f'    df.loc[ma_cond & rsi_cond & vol_cond, \'signal\'] = 1')
        lines.append('')
        lines.append('    # 综合卖出条件：均线死叉 或 RSI超买')
        lines.append(f'    ma_sell = (df[\'ma_fast\'] < df[\'ma_slow\']) & (df[\'ma_fast\'].shift(1) >= df[\'ma_slow\'].shift(1))')
        lines.append(f'    rsi_sell = df[\'rsi\'] > 80')
        lines.append(f'    df.loc[ma_sell | rsi_sell, \'signal\'] = -1')

    lines.append('')
    lines.append('    return df')
    lines.append('')
    lines.append('')
    lines.append('if __name__ == "__main__":')
    lines.append('    # 使用示例')
    lines.append('    import akshare as ak')
    lines.append('    symbol = "600519"')
    lines.append('    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date="20230101",')
    lines.append('                            end_date="20260101", adjust="qfq")')
    lines.append('    df.columns = [c.lower() for c in df.columns]')
    lines.append('    col_map = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",')
    lines.append('               "最低": "low", "成交量": "volume", "成交额": "amount"}')
    lines.append('    df.rename(columns=col_map, inplace=True)')
    lines.append('    result = generate_signals(df)')
    lines.append('    buy_signals = result[result[\'signal\'] == 1]')
    lines.append('    sell_signals = result[result[\'signal\'] == -1]')
    lines.append(f'    print(f"买入信号: {{len(buy_signals)}} 次")')
    lines.append(f'    print(f"卖出信号: {{len(sell_signals)}} 次")')

    return "\n".join(lines)


def _generate_backtest_code(parsed):
    """生成回测代码"""
    strategy_id = parsed.get("推荐策略", "ma_cross")
    params = parsed.get("参数", {})

    lines = []
    lines.append('"""')
    lines.append(f'自动生成的回测脚本')
    lines.append(f'策略: {strategy_id}')
    lines.append(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('"""')
    lines.append('')
    lines.append('import sys')
    lines.append('import os')
    lines.append('')
    lines.append('# 添加回测模块路径')
    lines.append('SKILLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))')
    lines.append('sys.path.insert(0, os.path.join(SKILLS_DIR, "backtest", "scripts"))')
    lines.append('sys.path.insert(0, os.path.join(SKILLS_DIR, "strategy-framework", "scripts"))')
    lines.append('')
    lines.append('from backtest_cli import backtest_with_strategy')
    lines.append('')
    lines.append('')
    lines.append('def run_backtest(symbol="600519", capital=100000, days=250):')
    lines.append('    """执行回测"""')
    lines.append(f'    strategy_id = "{strategy_id}"')
    lines.append(f'    params = {json.dumps(params, ensure_ascii=False)}')
    lines.append('')
    lines.append('    result = backtest_with_strategy(')
    lines.append('        symbol, strategy_id,')
    lines.append('        initial_capital=capital,')
    lines.append('        days=days,')
    lines.append('        position_size=1.0,')
    lines.append('        commission_rate=0.0003,')
    lines.append('        slippage=0.001,')
    lines.append('        **params')
    lines.append('    )')
    lines.append('')
    lines.append('    if "error" in result:')
    lines.append('        print(f"回测失败: {result[\'error\']}")')
    lines.append('        return')
    lines.append('')
    lines.append('    metrics = result.get("绩效指标", {})')
    lines.append('    print("=" * 50)')
    lines.append('    print("  回测结果")')
    lines.append('    print("=" * 50)')
    lines.append('    print(f"总收益率:   {metrics.get(\'总收益率\', 0):.2f}%")')
    lines.append('    print(f"年化收益率: {metrics.get(\'年化收益率\', 0):.2f}%")')
    lines.append('    print(f"夏普比率:   {metrics.get(\'夏普比率\', 0):.2f}")')
    lines.append('    print(f"最大回撤:   {metrics.get(\'最大回撤\', 0):.2f}%")')
    lines.append('    print(f"胜率:       {metrics.get(\'胜率\', 0):.2f}%")')
    lines.append('    print(f"交易次数:   {metrics.get(\'交易总次数\', 0)}")')
    lines.append('')
    lines.append('    return result')
    lines.append('')
    lines.append('')
    lines.append('if __name__ == "__main__":')
    lines.append('    run_backtest()')

    return "\n".join(lines)


def _walk_forward_analysis(trades, equity_curve):
    """
    样本外测试框架：滚动窗口验证，评估策略在未见数据上的泛化能力
    将数据分为样本内(训练)和样本外(测试)，计算表现衰减比
    返回: {窗口分析, 样本内外对比, 过拟合评分, 分析结论}
    """
    import math

    if not trades or len(trades) == 0:
        return {"窗口分析": [], "样本内外对比": {}, "过拟合评分": 0, "分析结论": ["无交易记录，无法进行样本外测试"]}

    analysis = {
        "窗口分析": [],
        "样本内外对比": {},
        "过拟合评分": 0,
        "分析结论": [],
    }

    try:
        # 按时间排序交易记录
        sorted_trades = sorted(trades, key=lambda t: str(t.get("日期", t.get("date", ""))))

        total_trades = len(sorted_trades)
        if total_trades < 20:
            analysis["分析结论"].append(f"交易次数仅{total_trades}次，样本不足，无法进行可靠的样本外测试（建议至少20次交易）")
            return analysis

        # 划分样本内(前60%)和样本外(后40%)
        split_idx = int(total_trades * 0.6)
        in_sample_trades = sorted_trades[:split_idx]
        out_sample_trades = sorted_trades[split_idx:]

        # 计算样本内指标
        is_profits = [t.get("盈亏", 0) for t in in_sample_trades if t.get("盈亏") is not None]
        oos_profits = [t.get("盈亏", 0) for t in out_sample_trades if t.get("盈亏") is not None]

        if not is_profits or not oos_profits:
            analysis["分析结论"].append("交易记录中缺少盈亏数据，无法计算样本内外对比")
            return analysis

        # 样本内指标
        is_total_return = sum(is_profits)
        is_wins = [p for p in is_profits if p > 0]
        is_losses = [p for p in is_profits if p < 0]
        is_win_rate = len(is_wins) / len(is_profits) * 100 if is_profits else 0
        is_avg_win = sum(is_wins) / len(is_wins) if is_wins else 0
        is_avg_loss = sum(is_losses) / len(is_losses) if is_losses else 0
        is_profit_factor = abs(sum(is_wins) / sum(is_losses)) if is_losses and sum(is_losses) != 0 else (999 if is_wins else 0)

        # 样本内最大回撤
        is_cumulative = 0
        is_peak = 0
        is_max_dd = 0
        for p in is_profits:
            is_cumulative += p
            if is_cumulative > is_peak:
                is_peak = is_cumulative
            dd = (is_cumulative - is_peak) if is_peak != 0 else 0
            if dd < is_max_dd:
                is_max_dd = dd

        # 样本外指标
        oos_total_return = sum(oos_profits)
        oos_wins = [p for p in oos_profits if p > 0]
        oos_losses = [p for p in oos_profits if p < 0]
        oos_win_rate = len(oos_wins) / len(oos_profits) * 100 if oos_profits else 0
        oos_avg_win = sum(oos_wins) / len(oos_wins) if oos_wins else 0
        oos_avg_loss = sum(oos_losses) / len(oos_losses) if oos_losses else 0
        oos_profit_factor = abs(sum(oos_wins) / sum(oos_losses)) if oos_losses and sum(oos_losses) != 0 else (999 if oos_wins else 0)

        # 样本外最大回撤
        oos_cumulative = 0
        oos_peak = 0
        oos_max_dd = 0
        for p in oos_profits:
            oos_cumulative += p
            if oos_cumulative > oos_peak:
                oos_peak = oos_cumulative
            dd = (oos_cumulative - oos_peak) if oos_peak != 0 else 0
            if dd < oos_max_dd:
                oos_max_dd = dd

        # 计算衰减比
        is_avg_return_per_trade = is_total_return / len(is_profits) if is_profits else 0
        oos_avg_return_per_trade = oos_total_return / len(oos_profits) if oos_profits else 0

        if is_avg_return_per_trade != 0:
            return_decay = oos_avg_return_per_trade / is_avg_return_per_trade
        else:
            return_decay = 0

        if is_win_rate > 0:
            win_rate_decay = oos_win_rate / is_win_rate
        else:
            win_rate_decay = 0

        if is_profit_factor > 0:
            pf_decay = oos_profit_factor / is_profit_factor if is_profit_factor < 999 else 1
        else:
            pf_decay = 0

        # 过拟合评分（0-100，越高越可能过拟合）
        overfit_score = 0
        if return_decay < 0.5:
            overfit_score += 40
            analysis["分析结论"].append(f"样本外平均收益衰减严重（衰减比{return_decay:.2f}），策略在未见数据上几乎失效，严重过拟合")
        elif return_decay < 0.7:
            overfit_score += 25
            analysis["分析结论"].append(f"样本外收益明显衰减（衰减比{return_decay:.2f}），策略泛化能力不足，存在过拟合风险")
        elif return_decay < 0.9:
            overfit_score += 10
            analysis["分析结论"].append(f"样本外收益轻微衰减（衰减比{return_decay:.2f}），策略泛化能力可接受")

        if win_rate_decay < 0.7:
            overfit_score += 25
            analysis["分析结论"].append(f"样本外胜率大幅下降（{oos_win_rate:.1f}% vs {is_win_rate:.1f}%），信号质量在样本外显著恶化")
        elif win_rate_decay < 0.85:
            overfit_score += 10
            analysis["分析结论"].append(f"样本外胜率有所下降（{oos_win_rate:.1f}% vs {is_win_rate:.1f}%），需关注信号稳定性")

        if oos_max_dd < is_max_dd * 1.5:
            overfit_score += 15
            analysis["分析结论"].append(f"样本外回撤显著扩大（{oos_max_dd:.1f}% vs {is_max_dd:.1f}%），风险控制在样本外失效")

        if oos_total_return < 0 and is_total_return > 0:
            overfit_score += 20
            analysis["分析结论"].append("样本内盈利但样本外亏损，策略完全过拟合，不具备实盘价值")

        overfit_score = min(100, overfit_score)
        analysis["过拟合评分"] = overfit_score

        # 过拟合等级
        if overfit_score >= 60:
            analysis["分析结论"].append(f"过拟合评分{overfit_score}/100，策略泛化能力差，不建议实盘使用")
        elif overfit_score >= 30:
            analysis["分析结论"].append(f"过拟合评分{overfit_score}/100，策略存在一定过拟合，实盘前需进一步验证")
        else:
            analysis["分析结论"].append(f"过拟合评分{overfit_score}/100，策略泛化能力良好，样本内外表现一致")

        analysis["样本内外对比"] = {
            "样本内": {
                "交易次数": len(is_profits),
                "总收益": f"{is_total_return:.2f}%",
                "胜率": f"{is_win_rate:.1f}%",
                "平均盈利": f"{is_avg_win:.2f}%",
                "平均亏损": f"{is_avg_loss:.2f}%",
                "盈利因子": round(is_profit_factor, 2),
                "最大回撤": f"{is_max_dd:.2f}%",
            },
            "样本外": {
                "交易次数": len(oos_profits),
                "总收益": f"{oos_total_return:.2f}%",
                "胜率": f"{oos_win_rate:.1f}%",
                "平均盈利": f"{oos_avg_win:.2f}%",
                "平均亏损": f"{oos_avg_loss:.2f}%",
                "盈利因子": round(oos_profit_factor, 2),
                "最大回撤": f"{oos_max_dd:.2f}%",
            },
            "衰减比": {
                "收益衰减": round(return_decay, 2),
                "胜率衰减": round(win_rate_decay, 2),
                "盈利因子衰减": round(pf_decay, 2),
            },
        }

    except Exception as e:
        analysis["分析结论"] = [f"样本外测试分析异常: {str(e)}"]

    return analysis


# ==================== 能力二：回测结果解读 ====================

def _performance_attribution(metrics, trades, equity_curve):
    """
    绩效归因分析：将策略收益分解为多个来源
    包含：收益分解、交易归因、回撤特征分析、风险调整收益评估
    返回: {收益分解, 交易归因, 回撤特征, 风险调整评估, 归因洞察}
    """
    import math

    total_return = metrics.get("总收益率", 0)
    annual_return = metrics.get("年化收益率", 0)
    sharpe = metrics.get("夏普比率", 0)
    max_dd = metrics.get("最大回撤", 0)
    win_rate = metrics.get("胜率", 0)
    total_trades = metrics.get("交易总次数", 0)
    calmar = metrics.get("卡玛比率", 0)
    excess_return = metrics.get("超额收益", 0)

    attribution = {
        "收益分解": {},
        "交易归因": {},
        "回撤特征": {},
        "风险调整评估": {},
        "归因洞察": []
    }

    # ===== 1. 收益分解 =====
    # 将总收益分解为：无风险收益 + 市场Beta收益 + Alpha超额收益
    risk_free_rate = 3.0  # 假设无风险利率3%
    market_return_est = total_return - excess_return if excess_return != 0 else total_return * 0.6

    # 估算Beta：通过超额收益与总收益的关系反推
    if total_return != 0 and abs(total_return) > 0.01:
        implied_beta = (total_return - excess_return) / total_return if excess_return != 0 else 0.8
        implied_beta = max(0.2, min(implied_beta, 2.0))
    else:
        implied_beta = 0.8

    market_contribution = market_return_est
    alpha_contribution = total_return - market_contribution

    attribution["收益分解"] = {
        "总收益": f"{total_return:.2f}%",
        "无风险收益基准": f"{risk_free_rate:.1f}%",
        "市场Beta贡献": f"{market_contribution:.2f}%",
        "Alpha超额贡献": f"{alpha_contribution:.2f}%",
        "估算Beta": round(implied_beta, 2),
        "超额收益": f"{excess_return:.2f}%",
    }

    # 收益来源判断
    if alpha_contribution > 5:
        attribution["归因洞察"].append(f"策略Alpha贡献{alpha_contribution:.1f}%，选股/择时能力突出，收益主要来自主动管理能力")
    elif alpha_contribution > 0:
        attribution["归因洞察"].append(f"策略Alpha贡献{alpha_contribution:.1f}%，有一定主动收益能力，但市场Beta仍是主要驱动力")
    else:
        attribution["归因洞察"].append(f"策略Alpha为负({alpha_contribution:.1f}%)，收益完全依赖市场上涨，缺乏独立获利能力")

    # ===== 2. 交易归因 =====
    if trades and len(trades) > 0:
        profits = [t.get("盈亏", 0) for t in trades if t.get("盈亏") is not None]
        if profits:
            wins = [p for p in profits if p > 0]
            losses = [p for p in profits if p < 0]

            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else (999 if wins else 0)
            expectancy = (win_rate / 100 * avg_win + (1 - win_rate / 100) * avg_loss) if total_trades > 0 else 0

            # 盈亏分布
            pnl_distribution = {
                "大幅亏损(<-5%)": 0,
                "中等亏损(-5%~-2%)": 0,
                "小幅亏损(-2%~0)": 0,
                "小幅盈利(0~2%)": 0,
                "中等盈利(2%~5%)": 0,
                "大幅盈利(>5%)": 0,
            }
            for p in profits:
                if p < -5:
                    pnl_distribution["大幅亏损(<-5%)"] += 1
                elif p < -2:
                    pnl_distribution["中等亏损(-5%~-2%)"] += 1
                elif p < 0:
                    pnl_distribution["小幅亏损(-2%~0)"] += 1
                elif p < 2:
                    pnl_distribution["小幅盈利(0~2%)"] += 1
                elif p < 5:
                    pnl_distribution["中等盈利(2%~5%)"] += 1
                else:
                    pnl_distribution["大幅盈利(>5%)"] += 1

            # 连续盈亏分析
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            current_streak = 0
            current_type = None
            for p in profits:
                if p > 0:
                    if current_type == "win":
                        current_streak += 1
                    else:
                        current_streak = 1
                        current_type = "win"
                    max_consecutive_wins = max(max_consecutive_wins, current_streak)
                elif p < 0:
                    if current_type == "loss":
                        current_streak += 1
                    else:
                        current_streak = 1
                        current_type = "loss"
                    max_consecutive_losses = max(max_consecutive_losses, current_streak)

            # 最大单笔盈亏
            max_single_win = max(profits) if profits else 0
            max_single_loss = min(profits) if profits else 0

            attribution["交易归因"] = {
                "总交易次数": total_trades,
                "盈利次数": len(wins),
                "亏损次数": len(losses),
                "胜率": f"{win_rate:.2f}%",
                "平均盈利": f"{avg_win:.2f}%",
                "平均亏损": f"{avg_loss:.2f}%",
                "盈亏比": round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else "N/A",
                "盈利因子": round(profit_factor, 2),
                "期望收益": f"{expectancy:.2f}%",
                "最大单笔盈利": f"{max_single_win:.2f}%",
                "最大单笔亏损": f"{max_single_loss:.2f}%",
                "最大连续盈利次数": max_consecutive_wins,
                "最大连续亏损次数": max_consecutive_losses,
                "盈亏分布": pnl_distribution,
            }

            # 交易归因洞察
            if profit_factor > 2:
                attribution["归因洞察"].append(f"盈利因子{profit_factor:.1f}，盈利交易的总收益远超亏损交易，策略盈亏结构健康")
            elif profit_factor > 1.2:
                attribution["归因洞察"].append(f"盈利因子{profit_factor:.1f}，盈亏结构合理但安全边际有限")
            else:
                attribution["归因洞察"].append(f"盈利因子仅{profit_factor:.1f}，盈利不足以覆盖亏损，策略盈亏结构存在根本性问题")

            if max_consecutive_losses >= 5:
                attribution["归因洞察"].append(f"最大连续亏损{max_consecutive_losses}次，策略在不利市场环境下可能持续回撤，需加强风控")

            if expectancy > 1:
                attribution["归因洞察"].append(f"每笔交易期望收益{expectancy:.1f}%，策略具有正向统计优势")
            elif expectancy > 0:
                attribution["归因洞察"].append(f"期望收益仅{expectancy:.1f}%，统计优势微弱，交易成本可能侵蚀全部利润")
            else:
                attribution["归因洞察"].append(f"期望收益为负({expectancy:.1f}%)，长期执行必然亏损")

            # 盈亏分布分析
            large_win_pct = pnl_distribution["大幅盈利(>5%)"] / total_trades * 100 if total_trades > 0 else 0
            large_loss_pct = pnl_distribution["大幅亏损(<-5%)"] / total_trades * 100 if total_trades > 0 else 0
            if large_loss_pct > 20:
                attribution["归因洞察"].append(f"大幅亏损交易占比{large_loss_pct:.0f}%，尾部风险突出，建议设置硬止损限制单笔亏损")
            if large_win_pct > 30:
                attribution["归因洞察"].append(f"大幅盈利交易占比{large_win_pct:.0f}%，策略具有捕捉大趋势的能力，这是核心优势")

    # ===== 3. 回撤特征分析 =====
    if equity_curve and len(equity_curve) > 1:
        values = []
        for e in equity_curve:
            if isinstance(e, dict):
                v = e.get("权益", e.get("value", 0))
            else:
                v = float(e) if e is not None else 0
            values.append(v)

        if len(values) > 1:
            # 计算回撤序列
            peak = values[0]
            drawdowns = []
            recovery_info = []
            in_drawdown = False
            dd_start = 0
            max_dd_val = 0
            max_dd_idx = 0
            dd_duration = 0
            current_dd_duration = 0

            for i, v in enumerate(values):
                if v > peak:
                    if in_drawdown:
                        recovery_info.append({
                            "回撤起始": dd_start,
                            "回撤结束": i,
                            "持续周期": i - dd_start,
                            "回撤幅度": f"{(min(values[dd_start:i+1]) / peak - 1) * 100:.2f}%",
                        })
                        in_drawdown = False
                    peak = v
                    current_dd_duration = 0
                else:
                    if not in_drawdown:
                        dd_start = i
                        in_drawdown = True
                    current_dd_duration += 1

                dd = (v - peak) / peak * 100
                drawdowns.append(dd)
                if dd < max_dd_val:
                    max_dd_val = dd
                    max_dd_idx = i
                    dd_duration = current_dd_duration

            # 如果最后还在回撤中
            if in_drawdown:
                recovery_info.append({
                    "回撤起始": dd_start,
                    "回撤结束": len(values) - 1,
                    "持续周期": len(values) - 1 - dd_start,
                    "回撤幅度": f"{(min(values[dd_start:]) / peak - 1) * 100:.2f}%",
                    "状态": "尚未恢复",
                })

            # 回撤统计
            dd_array = [d for d in drawdowns if d < 0]
            avg_dd = sum(dd_array) / len(dd_array) if dd_array else 0

            # 计算回撤恢复时间
            recovery_durations = [r["持续周期"] for r in recovery_info if r.get("状态") != "尚未恢复"]
            avg_recovery = sum(recovery_durations) / len(recovery_durations) if recovery_durations else 0
            max_recovery = max(recovery_durations) if recovery_durations else 0

            attribution["回撤特征"] = {
                "最大回撤": f"{max_dd_val:.2f}%",
                "最大回撤持续周期": dd_duration,
                "平均回撤": f"{avg_dd:.2f}%",
                "回撤次数": len(recovery_info),
                "平均恢复周期": round(avg_recovery, 1),
                "最长恢复周期": max_recovery,
                "回撤明细": recovery_info[-5:],
            }

            # 回撤洞察
            if dd_duration > 60:
                attribution["归因洞察"].append(f"最大回撤持续{dd_duration}个周期，恢复时间过长，策略在长期下跌中缺乏自我保护机制")
            elif dd_duration > 30:
                attribution["归因洞察"].append(f"最大回撤持续{dd_duration}个周期，恢复周期偏长，建议加入市场状态判断减少无效持仓")

            if max_dd_val < -30:
                attribution["归因洞察"].append(f"最大回撤{max_dd_val:.1f}%远超警戒线，策略风险敞口过大，必须引入仓位管理和止损机制")
            elif max_dd_val < -20:
                attribution["归因洞察"].append(f"最大回撤{max_dd_val:.1f}%偏高，建议设置回撤阈值自动减仓")

            if avg_recovery > 40:
                attribution["归因洞察"].append(f"平均回撤恢复需{avg_recovery:.0f}个周期，策略修复能力弱，需优化出场信号")

    # ===== 4. 风险调整收益评估 =====
    # 索提诺比率估算（只考虑下行波动）
    sortino_est = 0
    if sharpe > 0 and max_dd < 0:
        # 简化估算：索提诺 ≈ 夏普 * (总波动 / 下行波动)，下行波动通常约为总波动的70%
        sortino_est = round(sharpe / 0.7, 2)

    # 收益回撤比
    return_dd_ratio = abs(total_return / max_dd) if max_dd != 0 else (999 if total_return > 0 else 0)

    # 卡尔玛比率评级
    calmar_rating = ""
    if calmar > 2:
        calmar_rating = "优秀"
    elif calmar > 1:
        calmar_rating = "良好"
    elif calmar > 0.5:
        calmar_rating = "一般"
    else:
        calmar_rating = "较差"

    attribution["风险调整评估"] = {
        "夏普比率": f"{sharpe:.2f}",
        "卡玛比率": f"{calmar:.2f}",
        "卡玛比率评级": calmar_rating,
        "估算索提诺比率": sortino_est,
        "收益回撤比": round(return_dd_ratio, 2),
        "年化收益率": f"{annual_return:.2f}%",
        "年化波动率估算": f"{abs(annual_return / sharpe):.1f}%" if sharpe != 0 else "N/A",
    }

    # 风险调整洞察
    if calmar > 2:
        attribution["归因洞察"].append(f"卡玛比率{calmar:.1f}，每承担1%回撤获得{calmar:.1f}%收益，风险收益交换效率优秀")
    elif calmar < 0.5:
        attribution["归因洞察"].append(f"卡玛比率仅{calmar:.1f}，风险收益交换效率低，策略承担了过多风险却未获得相应回报")

    if return_dd_ratio < 1:
        attribution["归因洞察"].append(f"收益回撤比仅{return_dd_ratio:.1f}，收益甚至不足以覆盖最大回撤，策略风险收益结构严重失衡")

    # ===== 5. 综合归因总结 =====
    # 判断收益主要驱动因素
    if total_return > 0:
        if alpha_contribution > total_return * 0.5:
            primary_driver = "主动管理能力（Alpha驱动型）"
        elif market_contribution > total_return * 0.7:
            primary_driver = "市场上涨（Beta驱动型）"
        else:
            primary_driver = "市场与主动管理共同驱动"
    else:
        primary_driver = "策略失效，收益为负"

    attribution["归因总结"] = {
        "收益主要驱动": primary_driver,
        "策略风格判断": _classify_strategy_style(win_rate, total_trades, max_dd, sharpe),
        "核心优势": _identify_core_strength(attribution),
        "核心短板": _identify_core_weakness(attribution),
    }

    return attribution


def _classify_strategy_style(win_rate, total_trades, max_dd, sharpe):
    """根据指标判断策略风格"""
    if win_rate > 55 and total_trades > 20:
        return "高胜率型 - 追求高胜率，单笔盈利可能较小"
    elif win_rate < 45 and sharpe > 1:
        return "趋势跟踪型 - 低胜率高盈亏比，依靠少数大盈利覆盖多次小亏损"
    elif total_trades < 10:
        return "低频交易型 - 交易次数少，注重机会质量"
    elif total_trades > 50:
        return "高频交易型 - 交易频繁，依赖统计优势积累收益"
    elif max_dd < -25:
        return "高风险偏好型 - 愿意承受较大回撤以追求高收益"
    else:
        return "均衡型 - 各项指标较为均衡，无明显风格偏向"


def _identify_core_strength(attribution):
    """识别策略核心优势"""
    strengths = []
    trade_attr = attribution.get("交易归因", {})
    risk_attr = attribution.get("风险调整评估", {})

    profit_factor = trade_attr.get("盈利因子", 0)
    if isinstance(profit_factor, (int, float)) and profit_factor > 2:
        strengths.append("盈亏结构优秀")

    calmar = risk_attr.get("卡玛比率", "")
    try:
        calmar_val = float(calmar)
        if calmar_val > 1.5:
            strengths.append("风险调整收益出色")
    except (ValueError, TypeError):
        pass

    dd_attr = attribution.get("回撤特征", {})
    max_dd_str = dd_attr.get("最大回撤", "0%")
    try:
        max_dd_val = float(max_dd_str.replace("%", ""))
        if max_dd_val > -10:
            strengths.append("回撤控制优秀")
    except (ValueError, TypeError):
        pass

    return "、".join(strengths) if strengths else "无明显突出优势"


def _identify_core_weakness(attribution):
    """识别策略核心短板"""
    weaknesses = []
    trade_attr = attribution.get("交易归因", {})
    dd_attr = attribution.get("回撤特征", {})

    profit_factor = trade_attr.get("盈利因子", 0)
    if isinstance(profit_factor, (int, float)) and profit_factor < 1.2:
        weaknesses.append("盈亏结构失衡")

    max_dd_str = dd_attr.get("最大回撤", "0%")
    try:
        max_dd_val = float(max_dd_str.replace("%", ""))
        if max_dd_val < -25:
            weaknesses.append("回撤控制不足")
    except (ValueError, TypeError):
        pass

    dd_duration = dd_attr.get("最大回撤持续周期", 0)
    if isinstance(dd_duration, (int, float)) and dd_duration > 60:
        weaknesses.append("回撤恢复过慢")

    max_consecutive = trade_attr.get("最大连续亏损次数", 0)
    if isinstance(max_consecutive, (int, float)) and max_consecutive >= 5:
        weaknesses.append("连续亏损风险高")

    return "、".join(weaknesses) if weaknesses else "无明显突出短板"


def interpret_backtest(backtest_result):
    """
    解读回测结果，指出问题和改进建议
    """
    metrics = backtest_result.get("绩效指标", {})
    strategy_info = backtest_result.get("策略", {})

    total_return = metrics.get("总收益率", 0)
    annual_return = metrics.get("年化收益率", 0)
    sharpe = metrics.get("夏普比率", 0)
    max_dd = metrics.get("最大回撤", 0)
    win_rate = metrics.get("胜率", 0)
    total_trades = metrics.get("交易总次数", 0)
    calmar = metrics.get("卡玛比率", 0)
    excess_return = metrics.get("超额收益", 0)

    interpretation = {
        "解读时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "策略名称": strategy_info.get("name", "未知"),
        "综合评级": "",
        "亮点": [],
        "风险点": [],
        "改进建议": [],
        "详细分析": {}
    }

    # 综合评级
    score = 0
    if total_return > 0:
        score += 1
    if sharpe > 1:
        score += 1
    if max_dd > -20:
        score += 1
    if win_rate > 50:
        score += 1
    if total_trades >= 5:
        score += 1

    if score >= 4:
        interpretation["综合评级"] = "优秀 - 策略表现良好，各项指标均衡"
    elif score >= 3:
        interpretation["综合评级"] = "良好 - 策略整体可行，但存在改进空间"
    elif score >= 2:
        interpretation["综合评级"] = "一般 - 策略需要优化，部分指标不理想"
    else:
        interpretation["综合评级"] = "较差 - 策略存在明显问题，建议重新设计"

    # 收益分析
    return_analysis = {
        "总收益率": f"{total_return:.2f}%",
        "年化收益率": f"{annual_return:.2f}%",
        "超额收益": f"{excess_return:.2f}%"
    }

    if total_return > 30:
        interpretation["亮点"].append(f"总收益率 {total_return:.2f}% 表现优异，远超市场平均水平")
    elif total_return > 10:
        interpretation["亮点"].append(f"总收益率 {total_return:.2f}% 表现良好，跑赢大多数投资者")
    elif total_return > 0:
        interpretation["风险点"].append(f"总收益率仅 {total_return:.2f}%，收益水平偏低，需考虑策略有效性")
    else:
        interpretation["风险点"].append(f"总收益率为负 ({total_return:.2f}%)，策略在当前市场环境下失效")

    # 风险分析
    risk_analysis = {
        "最大回撤": f"{max_dd:.2f}%",
        "夏普比率": f"{sharpe:.2f}",
        "卡玛比率": f"{calmar:.2f}"
    }

    if sharpe > 2:
        interpretation["亮点"].append(f"夏普比率 {sharpe:.2f} 非常优秀，风险调整后收益极高")
    elif sharpe > 1:
        interpretation["亮点"].append(f"夏普比率 {sharpe:.2f} 良好，风险收益比合理")
    elif sharpe > 0.5:
        interpretation["风险点"].append(f"夏普比率仅 {sharpe:.2f}，风险调整后收益偏低")
    else:
        interpretation["风险点"].append(f"夏普比率 {sharpe:.2f} 过低，承担的风险与收益不匹配")

    if max_dd < -30:
        interpretation["风险点"].append(f"最大回撤 {max_dd:.2f}% 过大，策略风险控制能力不足")
        interpretation["改进建议"].append("建议加入止损机制，将单笔最大亏损控制在 5%-8%")
        interpretation["改进建议"].append("考虑加入仓位管理，根据波动率动态调整仓位")
    elif max_dd < -20:
        interpretation["风险点"].append(f"最大回撤 {max_dd:.2f}% 偏高，需加强风险控制")
        interpretation["改进建议"].append("建议设置移动止盈止损，保护已有利润")
    elif max_dd < -10:
        interpretation["亮点"].append(f"最大回撤仅 {max_dd:.2f}%，风险控制能力良好")

    # 胜率分析
    win_analysis = {
        "胜率": f"{win_rate:.2f}%",
        "交易总次数": total_trades
    }

    if win_rate > 60:
        interpretation["亮点"].append(f"胜率 {win_rate:.2f}% 较高，策略信号准确度高")
    elif win_rate > 45:
        pass  # 正常范围
    elif win_rate > 30:
        interpretation["风险点"].append(f"胜率仅 {win_rate:.2f}%，信号准确度偏低")
        interpretation["改进建议"].append("建议增加信号过滤条件，如趋势过滤、成交量确认")
    else:
        interpretation["风险点"].append(f"胜率仅 {win_rate:.2f}%，策略信号可靠性严重不足")

    if total_trades < 5:
        interpretation["风险点"].append(f"交易次数仅 {total_trades} 次，样本量不足，统计意义有限")
        interpretation["改进建议"].append("建议放宽交易条件或延长回测周期，增加交易次数")
    elif total_trades > 100:
        interpretation["风险点"].append(f"交易次数 {total_trades} 次过多，可能存在过度交易")
        interpretation["改进建议"].append("建议增加交易频率限制，避免过度交易导致手续费侵蚀收益")

    # 通用改进建议
    if sharpe < 1:
        interpretation["改进建议"].append("考虑加入大盘趋势过滤，只在上升趋势中做多")
    if total_return < 20:
        interpretation["改进建议"].append("建议尝试多周期参数优化，找到最佳参数组合")
    if max_dd < -15:
        interpretation["改进建议"].append("建议加入波动率自适应仓位，高波动时降低仓位")

    interpretation["详细分析"] = {
        "收益分析": return_analysis,
        "风险分析": risk_analysis,
        "胜率分析": win_analysis
    }

    # 绩效归因分析
    trades = backtest_result.get("交易记录", [])
    equity_curve = backtest_result.get("权益曲线", [])
    attribution = _performance_attribution(metrics, trades, equity_curve)
    interpretation["详细分析"]["绩效归因"] = attribution
    interpretation["归因洞察"] = attribution.get("归因洞察", [])

    # 压力测试分析
    stress_test = _stress_test_analysis(metrics, trades, equity_curve)
    interpretation["详细分析"]["压力测试"] = stress_test
    interpretation["压力测试评级"] = stress_test.get("综合评级", "")

    # 交易成本效率分析
    cost_efficiency = _analyze_strategy_cost_efficiency(metrics, trades)
    interpretation["详细分析"]["交易成本"] = cost_efficiency
    interpretation["成本效率评估"] = cost_efficiency.get("成本效率评估", "")

    # 样本外测试（过拟合检测）
    walk_forward = _walk_forward_analysis(trades, equity_curve)
    interpretation["详细分析"]["样本外测试"] = walk_forward
    interpretation["过拟合评分"] = walk_forward.get("过拟合评分", 0)
    # 将过拟合警告合并到风险点
    for conclusion in walk_forward.get("分析结论", []):
        if "过拟合" in conclusion or "衰减" in conclusion:
            interpretation["风险点"].append(conclusion)

    # 尝试用 LLM 增强解读
    config = load_config()
    if config.get("enabled") and config.get("api_key"):
        # 收集交易层面数据
        trade_analysis = {}
        if trades and len(trades) > 0:
            profits = [t.get("盈亏", 0) for t in trades if t.get("盈亏") is not None]
            if profits:
                wins = [p for p in profits if p > 0]
                losses = [p for p in profits if p < 0]
                trade_analysis["盈利次数"] = len(wins)
                trade_analysis["亏损次数"] = len(losses)
                trade_analysis["平均盈利"] = f"{sum(wins)/len(wins):.2f}%" if wins else "N/A"
                trade_analysis["平均亏损"] = f"{sum(losses)/len(losses):.2f}%" if losses else "N/A"
                trade_analysis["盈亏比"] = round(abs(sum(wins)/max(sum(losses), 1)), 2) if wins and losses else "N/A"
                trade_analysis["最大单笔盈利"] = f"{max(profits):.2f}%"
                trade_analysis["最大单笔亏损"] = f"{min(profits):.2f}%"

        # 回撤分析
        dd_analysis = {}
        if equity_curve and len(equity_curve) > 1:
            values = [e.get("权益", 0) for e in equity_curve if e.get("权益") is not None]
            if len(values) > 1:
                peak = values[0]
                max_dd_val = 0
                dd_duration = 0
                current_dd_duration = 0
                for v in values:
                    if v > peak:
                        peak = v
                        current_dd_duration = 0
                    else:
                        current_dd_duration += 1
                    dd = (v - peak) / peak * 100
                    if dd < max_dd_val:
                        max_dd_val = dd
                        dd_duration = current_dd_duration
                dd_analysis["最大回撤幅度"] = f"{max_dd_val:.2f}%"
                dd_analysis["最长回撤持续期"] = f"{dd_duration}个周期"

        system_prompt = """你是量化交易策略首席分析师。请根据以下回测数据和绩效归因分析，生成一份专业的策略评估报告。

输出格式（严格按此结构）：
【综合评级】给出策略等级（优秀/合格/需优化），并一句话说明理由
【收益来源】基于归因数据，说明收益主要来自市场Beta还是主动Alpha，评价收益质量
【风险画像】评价风险控制水平，关注最大回撤幅度、持续时间和恢复能力
【交易特征】评价交易行为，关注胜率、盈亏比、盈利因子、连续亏损风险
【归因洞察】基于绩效归因数据，指出策略的核心优势和关键短板
【优化方向】给出2-3条最优先的改进方向，每条包含具体参数建议
【免责声明】以上分析仅供参考，不构成投资建议

评估基准：
- 优秀：年化>15%，夏普>1.5，回撤<15%，胜率>55%，盈亏比>2
- 合格：年化>8%，夏普>0.8，回撤<25%，胜率>45%，盈亏比>1.2
- 需优化：低于上述标准

要求：语言专业精炼，总字数控制在350字以内。"""
        user_prompt = json.dumps({
            "策略名称": strategy_info.get("name", ""),
            "绩效概览": {
                "总收益率": f"{total_return:.2f}%",
                "年化收益率": f"{annual_return:.2f}%",
                "夏普比率": f"{sharpe:.2f}",
                "最大回撤": f"{max_dd:.2f}%",
                "胜率": f"{win_rate:.2f}%",
                "交易次数": total_trades,
                "卡玛比率": f"{calmar:.2f}",
                "超额收益": f"{excess_return:.2f}%"
            },
            "交易分析": trade_analysis,
            "回撤分析": dd_analysis,
            "绩效归因": {
                "收益分解": attribution.get("收益分解", {}),
                "交易归因": attribution.get("交易归因", {}),
                "回撤特征": attribution.get("回撤特征", {}),
                "风险调整评估": attribution.get("风险调整评估", {}),
                "归因总结": attribution.get("归因总结", {}),
            },
            "规则引擎预判": {
                "亮点": interpretation.get("亮点", []),
                "风险点": interpretation.get("风险点", []),
                "改进建议": interpretation.get("改进建议", [])
            }
        }, ensure_ascii=False)
        llm_insight = call_llm(system_prompt, user_prompt)
        if llm_insight:
            interpretation["AI深度解读"] = llm_insight

    return interpretation


def _analyze_trading_cost(price, shares, is_sell=True, turnover=None, amplitude=None):
    """
    交易成本分析：计算单笔交易的完整成本构成
    包含佣金、印花税、过户费、滑点估算、冲击成本估算
    返回: {成本明细, 总成本, 成本占比, 优化建议}
    """
    if not price or not shares or price <= 0 or shares <= 0:
        return {"总成本": 0, "成本占比": "0%", "成本明细": {}, "优化建议": []}

    trade_amount = price * shares

    # A股交易费率标准
    commission_rate = 0.00025  # 佣金万2.5
    min_commission = 5.0  # 最低佣金5元
    stamp_tax_rate = 0.001 if is_sell else 0  # 印花税千1（仅卖出）
    transfer_fee_rate = 0.00002  # 过户费万0.2

    # 佣金
    commission = max(trade_amount * commission_rate, min_commission)

    # 印花税
    stamp_tax = trade_amount * stamp_tax_rate

    # 过户费
    transfer_fee = max(trade_amount * transfer_fee_rate, 1.0)

    # 滑点估算（基于振幅和换手率）
    slippage_rate = 0.0005  # 默认滑点万5
    if amplitude is not None and turnover is not None:
        if amplitude > 5 and turnover > 5:
            slippage_rate = 0.002  # 高波动高换手，滑点千2
        elif amplitude > 3 and turnover > 3:
            slippage_rate = 0.001  # 中等波动，滑点千1
        elif amplitude < 2 and turnover < 2:
            slippage_rate = 0.0003  # 低波动低换手，滑点万3
    slippage = trade_amount * slippage_rate

    # 冲击成本估算（大额交易对价格的影响）
    impact_rate = 0
    if trade_amount > 1000000:  # 超过100万
        impact_rate = 0.001  # 千1冲击
    elif trade_amount > 500000:  # 超过50万
        impact_rate = 0.0005  # 万5冲击
    elif trade_amount > 100000:  # 超过10万
        impact_rate = 0.0002  # 万2冲击
    impact_cost = trade_amount * impact_rate

    total_cost = commission + stamp_tax + transfer_fee + slippage + impact_cost
    cost_pct = (total_cost / trade_amount * 100) if trade_amount > 0 else 0

    cost_detail = {
        "交易金额": round(trade_amount, 2),
        "佣金": round(commission, 2),
        "印花税": round(stamp_tax, 2),
        "过户费": round(transfer_fee, 2),
        "滑点估算": round(slippage, 2),
        "冲击成本": round(impact_cost, 2),
        "总成本": round(total_cost, 2),
        "成本占比": f"{cost_pct:.3f}%",
    }

    suggestions = []
    if cost_pct > 0.5:
        suggestions.append(f"单笔交易成本占比{cost_pct:.2f}%偏高，建议增加单笔交易金额以摊薄固定成本")
    if commission == min_commission and trade_amount < 20000:
        suggestions.append(f"交易金额{trade_amount:.0f}元低于2万元，佣金按最低5元收取，实际费率偏高，建议提高单笔交易规模")
    if slippage_rate >= 0.001:
        suggestions.append("当前市场波动较大，滑点成本较高，建议使用限价单减少滑点损失")
    if impact_rate > 0:
        suggestions.append(f"交易金额较大（{trade_amount/10000:.0f}万），存在冲击成本，建议分批建仓/减仓")
    if is_sell and stamp_tax > 0:
        suggestions.append(f"卖出需缴纳印花税{stamp_tax:.0f}元，频繁交易会显著增加成本，建议降低换手率")

    return {
        "成本明细": cost_detail,
        "总成本": round(total_cost, 2),
        "成本占比": f"{cost_pct:.3f}%",
        "优化建议": suggestions,
    }


def _analyze_strategy_cost_efficiency(metrics, trades):
    """
    策略层面交易成本效率分析
    评估交易成本对策略整体收益的侵蚀程度
    返回: {成本效率评估, 成本侵蚀分析, 换手率评估, 优化建议}
    """
    total_return = metrics.get("总收益率", 0)
    total_trades = metrics.get("交易总次数", 0)
    win_rate = metrics.get("胜率", 0)
    annual_return = metrics.get("年化收益率", 0)

    analysis = {
        "成本效率评估": "",
        "成本侵蚀分析": {},
        "换手率评估": "",
        "优化建议": [],
    }

    if not trades or len(trades) == 0:
        analysis["成本效率评估"] = "无交易记录，无法评估"
        return analysis

    # 估算总交易成本
    total_amount = 0
    total_estimated_cost = 0
    buy_count = 0
    sell_count = 0

    for t in trades:
        price = t.get("价格", 0)
        shares = t.get("数量", 0)
        direction = t.get("方向", "")
        if price and shares and price > 0 and shares > 0:
            amount = price * shares
            total_amount += amount
            is_sell = direction in ("卖出", "sell", "SELL")
            if is_sell:
                sell_count += 1
            else:
                buy_count += 1
            cost_result = _analyze_trading_cost(price, shares, is_sell=is_sell)
            total_estimated_cost += cost_result.get("总成本", 0)

    if total_amount > 0:
        cost_ratio = total_estimated_cost / total_amount * 100
        analysis["成本侵蚀分析"] = {
            "总交易金额": round(total_amount, 2),
            "估算总成本": round(total_estimated_cost, 2),
            "成本占交易额比": f"{cost_ratio:.3f}%",
            "买入次数": buy_count,
            "卖出次数": sell_count,
            "总交易次数": total_trades,
        }

        # 成本对收益的侵蚀
        if total_return != 0:
            # 假设初始资金为总交易金额的1/交易次数（粗略估算）
            estimated_capital = total_amount / max(total_trades, 1) * 2
            cost_erosion = total_estimated_cost / estimated_capital * 100 if estimated_capital > 0 else 0
            analysis["成本侵蚀分析"]["估算初始资金"] = round(estimated_capital, 2)
            analysis["成本侵蚀分析"]["成本对总收益侵蚀"] = f"{cost_erosion:.2f}%"

            if cost_erosion > total_return * 0.3:
                analysis["成本效率评估"] = "成本严重侵蚀收益"
                analysis["优化建议"].append(f"交易成本侵蚀了约{cost_erosion:.1f}%的收益，超过总收益的30%，策略盈利能力被交易成本严重削弱")
            elif cost_erosion > total_return * 0.1:
                analysis["成本效率评估"] = "成本有一定影响"
                analysis["优化建议"].append(f"交易成本约占收益的{cost_erosion/total_return*100:.0f}%，建议关注成本控制")
            else:
                analysis["成本效率评估"] = "成本控制良好"
        else:
            analysis["成本效率评估"] = "收益为零或负，成本影响需结合其他指标判断"

    # 换手率评估
    if total_trades > 0:
        avg_trades_per_stock = total_trades / max(len(set(t.get("代码", "") for t in trades)), 1)
        if total_trades > 50:
            analysis["换手率评估"] = "交易频率偏高"
            analysis["优化建议"].append(f"总交易{total_trades}次，换手率偏高，频繁交易导致成本累积，建议精选信号减少无效交易")
        elif total_trades > 20:
            analysis["换手率评估"] = "交易频率适中"
        else:
            analysis["换手率评估"] = "交易频率较低，成本可控"

    # 胜率与成本的关系
    if win_rate is not None and win_rate < 40:
        analysis["优化建议"].append(f"胜率仅{win_rate:.1f}%，低胜率策略对交易成本更敏感，每笔亏损交易都叠加了成本损失")

    return analysis


# ==================== 能力三：异常诊断 ====================

def _stress_test_analysis(metrics, trades, equity_curve):
    """
    压力测试分析：模拟极端市场环境下策略的表现
    基于回测数据中的最差表现推断压力场景下的风险敞口
    返回: {压力场景模拟, 风险指标, 压力测试建议, 综合评级}
    """
    total_return = metrics.get("总收益率", 0)
    max_dd = metrics.get("最大回撤", 0)
    sharpe = metrics.get("夏普比率", 0)
    win_rate = metrics.get("胜率", 0)
    total_trades = metrics.get("交易总次数", 0)
    annual_return = metrics.get("年化收益率", 0)

    analysis = {
        "压力场景模拟": [],
        "风险指标": {},
        "压力测试建议": [],
        "综合评级": "",
    }

    # ===== 1. 基于历史最差表现推断压力场景 =====
    worst_case_dd = max_dd
    worst_case_consecutive_losses = 0
    worst_case_recovery = 0

    if trades and len(trades) > 0:
        profits = [t.get("盈亏", 0) for t in trades if t.get("盈亏") is not None]
        if profits:
            # 最大连续亏损
            current_streak = 0
            for p in profits:
                if p < 0:
                    current_streak += 1
                    worst_case_consecutive_losses = max(worst_case_consecutive_losses, current_streak)
                else:
                    current_streak = 0

            # 最差连续N笔交易的总亏损
            if len(profits) >= 5:
                worst_5_streak = float('inf')
                for i in range(len(profits) - 4):
                    streak_sum = sum(profits[i:i+5])
                    if streak_sum < worst_5_streak:
                        worst_5_streak = streak_sum
                if worst_5_streak < 0:
                    analysis["压力场景模拟"].append({
                        "场景": "连续5笔交易最差表现",
                        "累计亏损": f"{worst_5_streak:.2f}%",
                        "说明": "模拟策略在连续不利信号下的最大可能亏损",
                    })

            if len(profits) >= 10:
                worst_10_streak = float('inf')
                for i in range(len(profits) - 9):
                    streak_sum = sum(profits[i:i+10])
                    if streak_sum < worst_10_streak:
                        worst_10_streak = streak_sum
                if worst_10_streak < 0:
                    analysis["压力场景模拟"].append({
                        "场景": "连续10笔交易最差表现",
                        "累计亏损": f"{worst_10_streak:.2f}%",
                        "说明": "模拟策略在持续不利市场环境下的累计亏损",
                    })

    # 权益曲线分析
    if equity_curve and len(equity_curve) > 1:
        values = []
        for e in equity_curve:
            if isinstance(e, dict):
                v = e.get("权益", e.get("value", 0))
            else:
                v = float(e) if e is not None else 0
            values.append(v)

        if len(values) > 1:
            # 最大回撤及恢复
            peak = values[0]
            max_dd_val = 0
            dd_start = 0
            dd_end = 0
            recovery_end = 0
            in_drawdown = False
            current_dd_start = 0

            for i, v in enumerate(values):
                if v > peak:
                    if in_drawdown:
                        recovery_end = i
                        in_drawdown = False
                    peak = v
                else:
                    if not in_drawdown:
                        current_dd_start = i
                        in_drawdown = True
                    dd = (v - peak) / peak * 100
                    if dd < max_dd_val:
                        max_dd_val = dd
                        dd_start = current_dd_start
                        dd_end = i

            if max_dd_val < 0:
                worst_case_recovery = recovery_end - dd_end if recovery_end > dd_end else len(values) - dd_end

            # 模拟市场暴跌场景：假设在最大回撤基础上再恶化50%
            crash_scenario_dd = max_dd_val * 1.5
            analysis["压力场景模拟"].append({
                "场景": "市场暴跌（历史最大回撤 x 1.5）",
                "预估回撤": f"{crash_scenario_dd:.2f}%",
                "说明": "模拟类似2008年或2015年股灾级别的极端行情",
            })

            # 模拟高波动场景：假设波动率翻倍
            if sharpe != 0 and annual_return != 0:
                normal_vol = abs(annual_return / sharpe)
                high_vol_dd = max_dd_val * 1.8  # 波动率翻倍时回撤通常放大1.5-2倍
                analysis["压力场景模拟"].append({
                    "场景": "波动率翻倍（高波动市场）",
                    "预估回撤": f"{high_vol_dd:.2f}%",
                    "说明": f"模拟波动率从{normal_vol:.1f}%升至{normal_vol*2:.1f}%时的回撤",
                })

    # ===== 2. 风险指标计算 =====
    # VaR估算（简化版：基于最大回撤）
    var_95 = abs(max_dd) * 0.6 if max_dd < 0 else 5
    var_99 = abs(max_dd) * 0.85 if max_dd < 0 else 8

    # 条件VaR（CVaR/Expected Shortfall）
    cvar_95 = abs(max_dd) * 0.8 if max_dd < 0 else 10

    # 最大可能亏损（基于最差连续交易）
    max_potential_loss = abs(max_dd) * 1.3 if max_dd < 0 else 15

    analysis["风险指标"] = {
        "VaR(95%)": f"{var_95:.1f}%",
        "VaR(99%)": f"{var_99:.1f}%",
        "CVaR(95%)": f"{cvar_95:.1f}%",
        "最大可能亏损估算": f"{max_potential_loss:.1f}%",
        "历史最大回撤": f"{abs(max_dd):.1f}%",
        "最大连续亏损次数": worst_case_consecutive_losses,
        "最长恢复周期": worst_case_recovery,
    }

    # ===== 3. 压力测试建议 =====
    suggestions = []

    # 基于压力测试结果给出仓位建议
    if abs(max_dd) > 30:
        suggestions.append({
            "优先级": "高",
            "建议": "极端行情下最大回撤可能超过45%，建议将单策略资金上限控制在总资产的30%以内",
            "具体措施": "设置策略级别止损：当策略回撤超过20%时暂停交易，重新评估市场环境",
        })
        suggestions.append({
            "优先级": "高",
            "建议": "引入波动率自适应仓位：当市场波动率(ATR)超过历史均值2倍时，仓位降至正常的30%",
            "具体措施": "监控VIX/ATR指标，波动率突破阈值时自动减仓",
        })
    elif abs(max_dd) > 20:
        suggestions.append({
            "优先级": "中",
            "建议": "压力场景下回撤可能达到30-40%，建议设置策略级别回撤预警线",
            "具体措施": "当回撤超过15%时减仓至50%，超过25%时暂停交易",
        })
    else:
        suggestions.append({
            "优先级": "低",
            "建议": "当前策略回撤控制较好，压力场景下回撤预计在可控范围内",
            "具体措施": "维持现有风控措施，定期复盘压力测试结果",
        })

    # 连续亏损应对
    if worst_case_consecutive_losses >= 5:
        suggestions.append({
            "优先级": "高",
            "建议": f"历史最大连续亏损{worst_case_consecutive_losses}次，压力场景下可能更多",
            "具体措施": "设置连续亏损熔断机制：连续亏损3次暂停1天，连续亏损5次暂停1周",
        })

    # 流动性风险
    if total_trades < 10:
        suggestions.append({
            "优先级": "中",
            "建议": "交易次数较少，流动性压力测试数据不足",
            "具体措施": "建议延长回测周期或放宽交易条件，获取更多样本评估流动性风险",
        })

    # 黑天鹅应对
    suggestions.append({
        "优先级": "中",
        "建议": "配置尾部风险对冲：考虑配置5-10%的逆相关性资产或期权保护",
        "具体措施": "可关注黄金ETF、国债ETF等避险资产，在极端行情下提供对冲保护",
    })

    analysis["压力测试建议"] = suggestions

    # ===== 4. 综合评级 =====
    stress_score = 0
    if abs(max_dd) < 15:
        stress_score += 3
    elif abs(max_dd) < 25:
        stress_score += 2
    elif abs(max_dd) < 35:
        stress_score += 1

    if worst_case_consecutive_losses < 4:
        stress_score += 2
    elif worst_case_consecutive_losses < 7:
        stress_score += 1

    if worst_case_recovery < 30:
        stress_score += 2
    elif worst_case_recovery < 60:
        stress_score += 1

    if sharpe > 1:
        stress_score += 1

    if stress_score >= 6:
        analysis["综合评级"] = "抗压能力优秀 - 策略在极端行情下预计仍能保持较好的风险控制"
    elif stress_score >= 4:
        analysis["综合评级"] = "抗压能力良好 - 策略有一定韧性，但极端行情下需加强风控"
    elif stress_score >= 2:
        analysis["综合评级"] = "抗压能力一般 - 策略在压力场景下可能面临较大回撤，建议加强防护措施"
    else:
        analysis["综合评级"] = "抗压能力较弱 - 策略对极端行情敏感，强烈建议引入多层次风控机制"

    return analysis


def diagnose_anomaly(strategy_id, backtest_result, market_context=None):
    """
    异常诊断：当策略回撤超预期时，分析原因
    """
    metrics = backtest_result.get("绩效指标", {})
    trades = backtest_result.get("交易记录", [])
    equity_curve = backtest_result.get("权益曲线", [])

    max_dd = metrics.get("最大回撤", 0)
    sharpe = metrics.get("夏普比率", 0)
    win_rate = metrics.get("胜率", 0)
    total_return = metrics.get("总收益率", 0)

    diagnosis = {
        "诊断时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "策略ID": strategy_id,
        "异常等级": "正常",
        "异常指标": [],
        "可能原因": [],
        "诊断建议": [],
        "详细分析": {}
    }

    # 判断异常等级
    anomaly_count = 0
    if max_dd < -25:
        diagnosis["异常指标"].append(f"最大回撤 {max_dd:.2f}% 严重超标（阈值 -25%）")
        anomaly_count += 2
    elif max_dd < -15:
        diagnosis["异常指标"].append(f"最大回撤 {max_dd:.2f}% 偏高（阈值 -15%）")
        anomaly_count += 1

    if sharpe < 0.3:
        diagnosis["异常指标"].append(f"夏普比率 {sharpe:.2f} 过低（阈值 0.3）")
        anomaly_count += 1

    if win_rate < 35:
        diagnosis["异常指标"].append(f"胜率 {win_rate:.2f}% 偏低（阈值 35%）")
        anomaly_count += 1

    if total_return < -10:
        diagnosis["异常指标"].append(f"总收益 {total_return:.2f}% 为负")
        anomaly_count += 1

    if anomaly_count >= 3:
        diagnosis["异常等级"] = "严重异常"
    elif anomaly_count >= 2:
        diagnosis["异常等级"] = "中度异常"
    elif anomaly_count >= 1:
        diagnosis["异常等级"] = "轻度异常"

    # 分析可能原因
    if max_dd < -20:
        diagnosis["可能原因"].append("策略缺乏有效的止损机制，单笔亏损过大")
        diagnosis["可能原因"].append("可能在市场剧烈波动期间持有重仓")
        diagnosis["诊断建议"].append("立即加入硬止损：单笔亏损超过 5% 强制平仓")
        diagnosis["诊断建议"].append("加入波动率过滤器：VIX/ATR 过高时降低仓位或暂停交易")

    if sharpe < 0.5:
        diagnosis["可能原因"].append("收益波动过大，风险调整后收益不足")
        diagnosis["可能原因"].append("策略可能在震荡市中频繁发出错误信号")
        diagnosis["诊断建议"].append("加入趋势过滤器：只在明确趋势中交易")
        diagnosis["诊断建议"].append("考虑降低交易频率，只参与高确定性机会")

    if win_rate < 40:
        diagnosis["可能原因"].append("策略信号质量差，假突破信号过多")
        diagnosis["可能原因"].append("入场时机选择不当，追高买入")
        diagnosis["诊断建议"].append("增加信号确认机制：如需要连续2根K线确认")
        diagnosis["诊断建议"].append("加入成交量确认：只参与放量突破")

    # 分析交易记录中的亏损交易
    if trades:
        losing_trades = [t for t in trades if t.get("盈亏", 0) < 0]
        if losing_trades:
            avg_loss = sum(t.get("盈亏", 0) for t in losing_trades) / len(losing_trades)
            max_single_loss = min(t.get("盈亏", 0) for t in losing_trades)

            loss_analysis = {
                "亏损交易数": len(losing_trades),
                "平均亏损": f"{avg_loss:.2f}",
                "最大单笔亏损": f"{max_single_loss:.2f}",
                "亏损占比": f"{len(losing_trades) / len(trades) * 100:.1f}%"
            }
            diagnosis["详细分析"]["亏损分析"] = loss_analysis

            if abs(max_single_loss) > abs(avg_loss) * 3:
                diagnosis["可能原因"].append("存在极端亏损交易，可能是黑天鹅事件或策略缺陷")
                diagnosis["诊断建议"].append("检查最大亏损交易发生时的市场环境，针对性优化")

    # 分析权益曲线
    if equity_curve and len(equity_curve) > 10:
        # 计算回撤持续时间
        peak = equity_curve[0]
        max_dd_duration = 0
        current_dd_duration = 0
        for val in equity_curve:
            if val >= peak:
                peak = val
                current_dd_duration = 0
            else:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)

        diagnosis["详细分析"]["最长回撤周期"] = f"{max_dd_duration} 个交易日"

        if max_dd_duration > 60:
            diagnosis["可能原因"].append(f"回撤持续时间过长（{max_dd_duration}天），策略可能在长期下跌中持续亏损")
            diagnosis["诊断建议"].append("加入市场环境判断：熊市中降低仓位或暂停交易")

    # 压力测试分析
    stress_test = _stress_test_analysis(metrics, trades, equity_curve)
    diagnosis["详细分析"]["压力测试"] = stress_test
    diagnosis["压力测试评级"] = stress_test.get("综合评级", "")
    # 将压力测试建议合并到诊断建议中
    for s in stress_test.get("压力测试建议", []):
        if s.get("优先级") == "高":
            diagnosis["诊断建议"].append(s.get("建议", ""))

    # 交易成本效率分析
    cost_efficiency = _analyze_strategy_cost_efficiency(metrics, trades)
    diagnosis["详细分析"]["交易成本"] = cost_efficiency
    diagnosis["成本效率评估"] = cost_efficiency.get("成本效率评估", "")
    for s in cost_efficiency.get("优化建议", []):
        diagnosis["诊断建议"].append(s)

    # 样本外测试（过拟合检测）
    walk_forward = _walk_forward_analysis(trades, equity_curve)
    diagnosis["详细分析"]["样本外测试"] = walk_forward
    diagnosis["过拟合评分"] = walk_forward.get("过拟合评分", 0)
    for conclusion in walk_forward.get("分析结论", []):
        if "过拟合" in conclusion:
            diagnosis["可能原因"].append(conclusion)
            diagnosis["诊断建议"].append("策略可能过拟合，建议进行样本外验证或简化策略参数")

    # 尝试用 LLM 增强诊断
    config = load_config()
    if config.get("enabled") and config.get("api_key"):
        # 收集详细诊断数据
        diag_data = {
            "策略ID": strategy_id,
            "规则引擎判定": {
                "异常等级": diagnosis["异常等级"],
                "异常指标": diagnosis["异常指标"],
                "可能原因": diagnosis["可能原因"],
                "诊断建议": diagnosis["诊断建议"],
            },
            "绩效指标": {
                "最大回撤": f"{max_dd:.2f}%",
                "夏普比率": f"{sharpe:.2f}",
                "胜率": f"{win_rate:.2f}%",
                "总收益": f"{total_return:.2f}%"
            },
            "详细分析": diagnosis.get("详细分析", {})
        }

        # 交易层面深度分析
        if trades:
            profits = [t.get("盈亏", 0) for t in trades if t.get("盈亏") is not None]
            if profits:
                wins = [p for p in profits if p > 0]
                losses = [p for p in profits if p < 0]
                diag_data["交易深度分析"] = {
                    "盈利交易数": len(wins),
                    "亏损交易数": len(losses),
                    "平均盈利": f"{sum(wins)/len(wins):.2f}%" if wins else "N/A",
                    "平均亏损": f"{sum(losses)/len(losses):.2f}%" if losses else "N/A",
                    "盈亏比": round(abs(sum(wins)/max(abs(sum(losses)), 0.01)), 2) if wins and losses else "N/A",
                    "最大单笔盈利": f"{max(profits):.2f}%",
                    "最大单笔亏损": f"{min(profits):.2f}%",
                    "连续亏损最大次数": _count_consecutive_losses(profits),
                }

        # 市场环境上下文
        if market_context:
            diag_data["市场环境"] = market_context

        system_prompt = """你是量化交易风控总监。用户的策略出现了异常表现，请从专业风控角度进行全面诊断。

诊断框架（五层穿透分析）：
1. 策略层：策略逻辑缺陷？过度拟合？参数敏感？信号滞后？
2. 市场层：当前市场环境是否不利？趋势/震荡/高波动？
3. 行业层：是否集中在弱势行业？行业轮动影响？
4. 风控层：止损有效性？仓位管理合理性？最大回撤是否可控？
5. 执行层：滑点、冲击成本、流动性影响？

异常严重度判定标准：
- 严重：回撤>25% 或 夏普<0.3 或 连续亏损>8次
- 中等：回撤15-25% 或 夏普0.3-0.8 或 胜率<40%
- 轻微：回撤10-15% 或 夏普0.8-1.0

输出格式（严格按此结构）：
【异常定级】给出异常等级（严重/中等/轻微），并说明触发条件
【根因分析】按可能性排序，列出2-3个最可能的根本原因
【止损评估】评价当前止损机制的有效性，给出具体改进数值
【修复方案】3条按优先级排序的修复建议，每条包含具体参数和预期效果
【风控总结】一句话总结核心问题和最优先行动项
【免责声明】以上分析仅供参考，不构成投资建议

要求：语言专业精准，像风控报告，总字数控制在350字以内。"""
        user_prompt = json.dumps(diag_data, ensure_ascii=False)
        llm_diag = call_llm(system_prompt, user_prompt)
        if llm_diag:
            diagnosis["AI深度诊断"] = llm_diag

    return diagnosis


def _count_consecutive_losses(profits):
    """计算最大连续亏损次数"""
    max_consecutive = 0
    current = 0
    for p in profits:
        if p < 0:
            current += 1
            max_consecutive = max(max_consecutive, current)
        else:
            current = 0
    return max_consecutive


# ==================== 综合 AI 助手接口 ====================

def ai_chat(message, context=None):
    """
    AI 对话接口：根据用户消息类型自动路由到对应能力
    """
    message_lower = message.lower().strip()

    # 判断意图
    intent = _classify_intent(message_lower)

    result = {
        "意图": intent,
        "回复": "",
        "数据": None,
        "时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    if intent == "strategy_generate":
        # 自然语言 → 策略
        strategy_result = nl_to_strategy(message)
        result["数据"] = strategy_result
        result["回复"] = _format_strategy_reply(strategy_result)

    elif intent == "backtest_interpret":
        # 回测解读
        if context and "backtest" in context:
            interpretation = interpret_backtest(context["backtest"])
            result["数据"] = interpretation
            result["回复"] = _format_interpretation_reply(interpretation)
        else:
            result["回复"] = "请先执行回测，然后我可以帮你解读回测结果。你可以说'帮我分析回测结果'并提供回测数据。"

    elif intent == "anomaly_diagnose":
        # 异常诊断
        if context and "backtest" in context:
            strategy_id = context.get("strategy_id", "unknown")
            diagnosis = diagnose_anomaly(strategy_id, context["backtest"])
            result["数据"] = diagnosis
            result["回复"] = _format_diagnosis_reply(diagnosis)
        else:
            result["回复"] = "请先执行回测，当回测出现异常回撤时，我可以帮你诊断原因。"

    elif intent == "market_query":
        result["回复"] = "我可以帮你查询市场行情。请告诉我你想了解哪只股票（如600519），或者想看大盘趋势。"

    elif intent == "strategy_query":
        result["回复"] = "当前系统支持以下策略：\n1. 均线交叉(ma_cross) - 快慢均线金叉死叉\n2. MACD(macd) - MACD金叉死叉\n3. RSI(rsi) - 超买超卖\n4. 布林带(bollinger) - 触及上下轨\n5. 放量突破(volume_breakout) - 放量突破均线\n6. 多因子(multi_factor) - 综合多条件\n\n你可以说'帮我写一个5日均线上穿20日均线且放量的策略'来生成策略代码。"

    else:
        result["回复"] = _generate_general_reply(message, intent)

    return result


def _classify_intent(message):
    """分类用户意图"""
    # 策略生成相关
    strategy_keywords = [
        "策略", "均线", "MACD", "RSI", "布林", "放量", "金叉", "死叉",
        "上穿", "下穿", "突破", "帮我写", "生成策略", "编写策略",
        "写一个", "创建一个", "设计一个", "做一个"
    ]
    if any(kw.lower() in message.lower() for kw in strategy_keywords):
        return "strategy_generate"

    # 回测解读相关
    interpret_keywords = [
        "解读", "分析回测", "回测结果", "回测报告", "帮我看看",
        "分析一下", "表现如何", "怎么样", "好不好"
    ]
    if any(kw in message for kw in interpret_keywords):
        return "backtest_interpret"

    # 异常诊断相关
    anomaly_keywords = [
        "回撤", "亏损", "异常", "诊断", "出问题", "不对",
        "为什么亏", "怎么亏", "原因", "排查"
    ]
    if any(kw in message for kw in anomaly_keywords):
        return "anomaly_diagnose"

    # 市场查询
    market_keywords = [
        "行情", "大盘", "涨跌", "走势", "趋势", "市场",
        "今天", "现在", "最新"
    ]
    if any(kw in message for kw in market_keywords):
        return "market_query"

    # 策略查询
    strategy_query_keywords = [
        "有哪些策略", "支持什么", "什么策略", "可用策略",
        "策略列表", "功能"
    ]
    if any(kw in message for kw in strategy_query_keywords):
        return "strategy_query"

    return "general"


def _format_strategy_reply(strategy_result):
    """格式化策略生成回复"""
    parsed = strategy_result.get("解析结果", {})
    strategy_id = parsed.get("推荐策略", "未知")
    confidence = parsed.get("置信度", "低")
    indicators = parsed.get("识别指标", [])
    params = parsed.get("参数", {})

    reply_parts = []
    reply_parts.append(f"我已解析你的策略描述，置信度：{confidence}")

    if indicators:
        indicator_names = {
            "ma_cross": "均线交叉", "macd": "MACD", "rsi": "RSI",
            "bollinger": "布林带", "volume_breakout": "放量突破",
            "multi_factor": "多因子综合"
        }
        ind_names = [indicator_names.get(i, i) for i in indicators]
        reply_parts.append(f"识别到的技术指标：{'、'.join(ind_names)}")

    if params:
        param_strs = []
        for k, v in params.items():
            param_labels = {
                "ma_fast": f"快线周期={v}", "ma_slow": f"慢线周期={v}",
                "rsi_period": f"RSI周期={v}", "rsi_oversold": f"RSI超卖={v}",
                "volume_multiple": f"放量倍数={v}倍"
            }
            param_strs.append(param_labels.get(k, f"{k}={v}"))
        reply_parts.append(f"提取的参数：{'，'.join(param_strs)}")

    reply_parts.append(f"推荐策略：{strategy_id}")

    # 过拟合风险评估
    overfit_risk = _assess_overfit_risk(parsed)
    if overfit_risk["level"] != "低":
        reply_parts.append(f"\n  {overfit_risk['warning']}")
        if overfit_risk.get("suggestions"):
            for sug in overfit_risk["suggestions"]:
                reply_parts.append(f"  {sug}")

    reply_parts.append("策略代码和回测代码已生成，你可以在下方查看完整代码。")

    return "\n".join(reply_parts)


def _assess_overfit_risk(parsed):
    """
    评估策略过拟合风险
    返回: {level, warning, suggestions}
    """
    params = parsed.get("参数", {})
    indicators = parsed.get("识别指标", [])
    buy_conditions = parsed.get("买入条件", [])
    sell_conditions = parsed.get("卖出条件", [])

    param_count = len(params)
    indicator_count = len(indicators)
    condition_count = len(buy_conditions) + len(sell_conditions)

    risk_score = 0
    reasons = []

    if param_count > 5:
        risk_score += 3
        reasons.append(f"参数数量({param_count}个)偏多")
    elif param_count > 3:
        risk_score += 1
        reasons.append(f"参数数量({param_count}个)适中")

    if indicator_count > 3:
        risk_score += 2
        reasons.append(f"使用{indicator_count}个指标，组合复杂度高")
    elif indicator_count > 1:
        risk_score += 1

    if condition_count > 4:
        risk_score += 2
        reasons.append(f"买卖条件({condition_count}条)过多，可能过度拟合历史数据")

    # 检查是否有过于精确的参数值
    for k, v in params.items():
        if isinstance(v, (int, float)) and v < 3:
            risk_score += 1
            reasons.append(f"参数'{k}'={v}过小，对噪音敏感")
            break

    if risk_score >= 5:
        level = "高"
        warning = f"过拟合风险：{level}（{'；'.join(reasons)}）"
        suggestions = [
            "建议减少参数数量，优先保留核心参数",
            "建议使用Walk-Forward验证评估策略稳定性",
            "建议在样本外数据(最近1年)上单独验证",
        ]
    elif risk_score >= 3:
        level = "中"
        warning = f"过拟合风险：{level}（{'；'.join(reasons)}）"
        suggestions = [
            "建议在样本外数据上验证策略表现",
            "可考虑简化部分条件，降低复杂度",
        ]
    else:
        level = "低"
        warning = ""
        suggestions = []

    return {"level": level, "warning": warning, "suggestions": suggestions}


def _format_interpretation_reply(interpretation):
    """格式化回测解读回复"""
    parts = []
    parts.append(f"综合评级：{interpretation.get('综合评级', '')}")

    highlights = interpretation.get("亮点", [])
    if highlights:
        parts.append("\n亮点：")
        for h in highlights[:3]:
            parts.append(f"  + {h}")

    risks = interpretation.get("风险点", [])
    if risks:
        parts.append("\n风险点：")
        for r in risks[:3]:
            parts.append(f"  - {r}")

    suggestions = interpretation.get("改进建议", [])
    if suggestions:
        parts.append("\n改进建议：")
        for s in suggestions[:3]:
            parts.append(f"  > {s}")

    ai_insight = interpretation.get("AI深度解读", "")
    if ai_insight:
        parts.append(f"\nAI 深度分析：\n{ai_insight}")

    return "\n".join(parts)


def _format_diagnosis_reply(diagnosis):
    """格式化异常诊断回复"""
    parts = []
    parts.append(f"异常等级：{diagnosis.get('异常等级', '正常')}")

    indicators = diagnosis.get("异常指标", [])
    if indicators:
        parts.append("\n异常指标：")
        for ind in indicators:
            parts.append(f"  ! {ind}")

    causes = diagnosis.get("可能原因", [])
    if causes:
        parts.append("\n可能原因：")
        for c in causes[:3]:
            parts.append(f"  ? {c}")

    suggestions = diagnosis.get("诊断建议", [])
    if suggestions:
        parts.append("\n诊断建议：")
        for s in suggestions[:3]:
            parts.append(f"  > {s}")

    ai_diag = diagnosis.get("AI深度诊断", "")
    if ai_diag:
        parts.append(f"\nAI 深度诊断：\n{ai_diag}")

    return "\n".join(parts)


def _generate_general_reply(message, intent):
    """生成通用回复，优先使用LLM，降级使用硬编码回复"""
    config = load_config()
    if config.get("enabled") and config.get("api_key"):
        system_prompt = """你是A股量化交易平台的AI助手，运行在一个集成了数据获取、策略回测、风险控制、智能推荐等50个技能模块的专业平台上。

你的身份和能力：
- 你可以解答股票相关的问题，包括行情分析、技术指标解读、基本面评估
- 你可以帮助用户理解和使用平台的各项功能
- 你可以提供投资知识科普，但必须声明不构成投资建议
- 你了解A股市场的基本规则（T+1、涨跌停、交易时间等）

回复要求：
1. 用中文回复，简洁专业，语气友好
2. 如果用户问的是平台功能，引导用户使用对应功能
3. 如果用户问的是投资建议，必须声明"以上分析仅供参考，不构成投资建议"
4. 如果用户问的是你无法回答的问题，诚实告知并引导到你能帮助的领域
5. 回复控制在200字以内，除非用户要求详细分析

平台核心功能速查：
- 智能推荐助手：自然语言描述需求，AI多维度推荐股票
- 策略生成：描述交易思路，自动生成策略代码
- 回测分析：上传策略进行历史回测，评估绩效
- 风险控制：多维度风险评估和仓位管理
- 数据获取：实时行情、历史K线、财务数据"""
        user_prompt = f"用户说：{message}\n请用中文简洁回复。"
        llm_reply = call_llm(system_prompt, user_prompt)
        if llm_reply:
            return llm_reply

    replies = {
        "greeting": "你好！我是你的 AI 股票分析助手。我可以帮你：\n1. 将自然语言描述转为策略代码（如'帮我写一个5日均线上穿20日均线且放量的策略'）\n2. 解读回测结果，指出问题和改进方向\n3. 诊断策略异常回撤的原因\n\n请告诉我你需要什么帮助？",
        "help": "我可以帮你做以下事情：\n- 策略生成：描述你的交易思路，我帮你生成代码\n- 回测解读：分析回测报告，指出亮点和风险\n- 异常诊断：当策略回撤过大时，分析原因\n\n试试说'帮我写一个MACD金叉买入的策略'",
    }

    if any(kw in message for kw in ["你好", "hi", "hello", "嗨"]):
        return replies["greeting"]
    elif any(kw in message for kw in ["帮助", "help", "能做什么", "功能"]):
        return replies["help"]
    else:
        return f"收到你的消息。{replies['help']}"


# ==================== 每日市场简报 ====================

def generate_market_briefing():
    """
    生成当日A股市场简报
    返回: {briefing_text, market_data, generated_time}
    """
    current_date = datetime.now().strftime('%Y年%m月%d日')
    briefing = {
        "generated_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "briefing_text": "",
        "market_data": {}
    }

    try:
        from data_utils import _get_spot_df
        spot_df = _get_spot_df()
        if spot_df is not None and not spot_df.empty:
            df = spot_df.copy()

            # 大盘指数数据
            index_codes = ['000001', '399001', '399006', '000688']
            index_names = {'000001': '上证指数', '399001': '深证成指', '399006': '创业板指', '000688': '科创50'}
            index_data = {}
            for code in index_codes:
                row = df[df['代码'] == code]
                if not row.empty:
                    r = row.iloc[0]
                    index_data[code] = {
                        "name": index_names.get(code, code),
                        "price": _safe_float(r.get('最新价')),
                        "change_pct": _safe_float(r.get('涨跌幅')),
                        "amplitude": _safe_float(r.get('振幅')),
                    }
            briefing["market_data"]["indices"] = index_data

            # 涨跌统计
            if '涨跌幅' in df.columns:
                chg = pd.to_numeric(df['涨跌幅'], errors='coerce').dropna()
                total = len(chg)
                up = int((chg > 0).sum())
                down = int((chg < 0).sum())
                briefing["market_data"]["up_count"] = up
                briefing["market_data"]["down_count"] = down
                briefing["market_data"]["flat_count"] = int((chg == 0).sum())
                briefing["market_data"]["up_down_ratio"] = round(up / max(down, 1), 2)
                briefing["market_data"]["avg_change"] = round(float(chg.mean()), 2)
                briefing["market_data"]["median_change"] = round(float(chg.median()), 2)
                briefing["market_data"]["limit_up"] = int((chg >= 9.9).sum())
                briefing["market_data"]["limit_down"] = int((chg <= -9.9).sum())

                # 涨幅分布
                bins = [(-100, -5), (-5, -2), (-2, 0), (0, 2), (2, 5), (5, 100)]
                dist = {}
                for lo, hi in bins:
                    cnt = int(((chg > lo) & (chg <= hi)).sum())
                    if cnt > 0:
                        dist[f"{lo}%~{hi}%"] = cnt
                briefing["market_data"]["change_distribution"] = dist

            # 成交额统计
            if '成交额' in df.columns:
                amt = pd.to_numeric(df['成交额'], errors='coerce').dropna()
                total_amt = amt.sum()
                if total_amt > 1e12:
                    briefing["market_data"]["total_amount"] = f"{total_amt/1e12:.2f}万亿"
                else:
                    briefing["market_data"]["total_amount"] = f"{total_amt/1e8:.0f}亿"
                briefing["market_data"]["avg_amount_per_stock"] = f"{amt.mean()/1e8:.1f}亿"

            # 热门板块
            if '涨跌幅' in df.columns and '行业' in df.columns:
                sector_chg = df.groupby('行业')['涨跌幅'].apply(
                    lambda x: pd.to_numeric(x, errors='coerce').mean()
                ).dropna().sort_values(ascending=False)
                briefing["market_data"]["top_sectors"] = [
                    {"name": s, "change": round(float(v), 2)}
                    for s, v in sector_chg.head(5).items()
                ]
                briefing["market_data"]["bottom_sectors"] = [
                    {"name": s, "change": round(float(v), 2)}
                    for s, v in sector_chg.tail(5).items()
                ]

            # 涨幅前5个股
            if '涨跌幅' in df.columns and '名称' in df.columns:
                top_stocks = df.nlargest(5, '涨跌幅')
                briefing["market_data"]["top_gainers"] = [
                    {"name": str(r['名称']), "code": str(r['代码']), "change": _safe_float(r['涨跌幅'])}
                    for _, r in top_stocks.iterrows()
                ]
    except Exception as e:
        briefing["market_data"]["error"] = str(e)

    # LLM 生成自然语言简报
    config = load_config()
    if config.get("enabled") and config.get("api_key"):
        md = briefing["market_data"]
        system_prompt = f"""你是A股市场首席策略分析师。请根据以下市场数据，生成一份专业的当日A股市场简报。

当前日期：{current_date}

输出格式（严格按此结构）：
【市场定调】用一句话定性今日市场（如"放量普涨"、"缩量分化"、"弱势震荡"），并给出市场温度（热/温/冷）
【指数扫描】简述各主要指数表现，指出领涨和领跌指数
【情绪仪表】涨跌比、涨停跌停数、成交额变化趋势，判断市场情绪（亢奋/正常/恐慌）
【板块轮动】热点板块及驱动逻辑，弱势板块及原因
【资金风向】成交额变化、个股活跃度判断
【策略建议】基于当前市场状态，给出1-2条仓位或操作建议
【免责声明】以上分析仅供参考，不构成投资建议

要求：
- 语言专业精炼，像券商研究所的每日复盘报告
- 数据引用要准确，基于提供的实际数据
- 总字数控制在300字以内"""
        user_prompt = json.dumps(md, ensure_ascii=False, default=str)
        llm_briefing = call_llm(system_prompt, user_prompt)
        if llm_briefing:
            briefing["briefing_text"] = llm_briefing
    else:
        md = briefing["market_data"]
        parts = [f"{'='*40}", f"  {current_date} A股市场简报", f"{'='*40}", ""]
        indices = md.get("indices", {})
        for code, info in indices.items():
            if info.get("price") is not None:
                chg = info.get("change_pct", 0) or 0
                arrow = "+" if chg > 0 else ""
                parts.append(f"  {info['name']:<6s} {info['price']:>8.2f}  {arrow}{chg:+.2f}%")
        parts.append("")
        parts.append(f"  上涨: {md.get('up_count', '?')}家  下跌: {md.get('down_count', '?')}家  涨跌比: {md.get('up_down_ratio', '?')}")
        parts.append(f"  涨停: {md.get('limit_up', '?')}家  跌停: {md.get('limit_down', '?')}家")
        if md.get("total_amount"):
            parts.append(f"  成交额: {md['total_amount']}")
        top_sectors = md.get("top_sectors", [])
        if top_sectors:
            parts.append(f"\n  热门板块: {', '.join(s['name'] for s in top_sectors[:3])}")
        parts.append(f"\n  以上分析仅供参考，不构成投资建议。")
        briefing["briefing_text"] = "\n".join(parts)

    return briefing


# ==================== 配置管理 ====================

def get_ai_config():
    """获取当前 AI 配置"""
    config = load_config()
    # 隐藏 API Key
    safe_config = config.copy()
    if safe_config.get("api_key"):
        key = safe_config["api_key"]
        if len(key) > 8:
            safe_config["api_key"] = key[:4] + "****" + key[-4:]
        else:
            safe_config["api_key"] = "****"
    return safe_config


def update_ai_config(new_config):
    """更新 AI 配置"""
    config = load_config()
    for k, v in new_config.items():
        if k in config:
            config[k] = v
    save_config(config)
    return {"status": "ok", "message": "配置已更新"}


def test_ai_connection():
    """测试 AI 连接"""
    config = load_config()
    if not config.get("enabled"):
        return {"status": "disabled", "message": "AI 功能未启用"}
    if not config.get("api_key"):
        return {"status": "no_key", "message": "未配置 API Key"}

    try:
        result = call_llm("你是一个助手", "请回复'连接成功'")
        if result:
            return {"status": "ok", "message": "AI 连接正常", "response": result[:100]}
        else:
            return {"status": "error", "message": "AI 返回为空，请检查配置"}
    except Exception as e:
        return {"status": "error", "message": f"连接失败: {str(e)}"}


# 默认推荐股票池（沪深300成分股中流动性好的标的）
DEFAULT_RECOMMEND_POOL = [
    "600519", "000858", "000568", "600809", "000596",
    "600036", "601318", "000333", "600276", "300750",
    "601012", "600900", "000651", "002415", "300059",
    "600030", "000002", "601166", "600585", "000725",
    "002475", "300124", "600887", "000063", "601888",
    "600309", "002714", "300015", "600436", "000538"
]


def _generate_trading_review(result):
    """
    交易复盘报告生成：综合所有分析维度，生成结构化的交易复盘报告
    报告结构：市场概况 -> 多维度分析 -> 风险提示 -> 操作建议 -> 推荐标的
    返回: {报告标题, 生成时间, 市场概况, 多维度分析, 风险提示, 综合操作建议, 推荐标的详情}
    """
    from datetime import datetime

    report = {
        "报告标题": "AI量化交易复盘报告",
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "市场概况": {},
        "多维度分析": {},
        "风险提示": [],
        "综合操作建议": [],
        "推荐标的详情": [],
    }

    # 1. 市场概况
    market_overview = result.get("市场概况", {})
    report["市场概况"] = {
        "上涨家数": market_overview.get("上涨家数", "N/A"),
        "下跌家数": market_overview.get("下跌家数", "N/A"),
        "涨停家数": market_overview.get("涨停家数", "N/A"),
        "跌停家数": market_overview.get("跌停家数", "N/A"),
        "市场情绪": market_overview.get("市场情绪", "N/A"),
    }

    # 2. 多维度分析
    multi_analysis = {}

    # 市场状态
    regime = result.get("市场状态", {})
    if regime:
        multi_analysis["市场状态"] = {
            "当前状态": regime.get("当前状态", "N/A"),
            "建议仓位": regime.get("建议仓位", "N/A"),
        }

    # 多周期共振
    resonance = result.get("多周期共振", {})
    if resonance:
        multi_analysis["多周期共振"] = {
            "共振状态": resonance.get("共振状态", "N/A"),
            "共振评分": resonance.get("共振评分", "N/A"),
            "操作含义": resonance.get("操作含义", "N/A"),
        }

    # 量价关系
    vol_price = result.get("量价关系", {})
    if vol_price:
        multi_analysis["量价关系"] = {
            "综合判断": vol_price.get("综合判断", "N/A"),
            "量价评分": vol_price.get("量价评分", "N/A"),
            "放量突破": vol_price.get("放量突破", {}).get("突破等级", "N/A"),
            "量价背离": vol_price.get("量价背离", {}).get("背离等级", "N/A"),
        }

    # 财报季分析
    earnings = result.get("财报季分析", {})
    if earnings:
        multi_analysis["财报季"] = {
            "当前阶段": earnings.get("财报阶段", "N/A"),
            "阶段特征": earnings.get("阶段特征", "N/A"),
        }

    # 日历效应
    calendar = result.get("日历效应", {})
    if calendar:
        multi_analysis["日历效应"] = {
            "月初月末": calendar.get("月初月末效应", {}).get("阶段", "N/A"),
            "季节性": calendar.get("季节性规律", {}).get("规律", "N/A"),
        }

    # 板块轮动
    rotation = result.get("板块轮动", {})
    if rotation:
        multi_analysis["板块轮动"] = {
            "轮动速度": rotation.get("轮动速度", "N/A"),
            "轮动方向": rotation.get("轮动方向", "N/A"),
            "强势板块": rotation.get("强势板块", [])[:3] if isinstance(rotation.get("强势板块"), list) else "N/A",
        }

    # 市场择时
    timing = result.get("市场择时", {})
    if timing:
        multi_analysis["市场择时"] = {
            "择时评分": timing.get("择时评分", "N/A"),
            "操作建议": timing.get("操作建议", "N/A"),
        }

    # 风险监控
    risk = result.get("风险监控", {})
    if risk:
        multi_analysis["风险监控"] = {
            "风险等级": risk.get("风险等级", "N/A"),
            "最大回撤": risk.get("最大回撤", "N/A"),
        }

    report["多维度分析"] = multi_analysis

    # 3. 风险提示
    risk_warnings = []

    # 从风险监控获取
    if risk:
        risk_alerts = risk.get("风险提示", [])
        if isinstance(risk_alerts, list):
            risk_warnings.extend(risk_alerts)

    # 从财报季获取
    if earnings:
        earnings_risks = earnings.get("风险提示", [])
        if isinstance(earnings_risks, list):
            risk_warnings.extend(earnings_risks)

    # 从量价关系获取
    if vol_price:
        vol_risks = vol_price.get("处理建议", [])
        if isinstance(vol_risks, list):
            risk_warnings.extend(vol_risks)

    # 从极端行情获取
    extreme = result.get("极端行情应对", {})
    if extreme:
        extreme_risks = extreme.get("风险提示", [])
        if isinstance(extreme_risks, list):
            risk_warnings.extend(extreme_risks)

    # 去重
    seen = set()
    unique_warnings = []
    for w in risk_warnings:
        if w not in seen:
            seen.add(w)
            unique_warnings.append(w)
    report["风险提示"] = unique_warnings[:10]

    # 4. 综合操作建议
    suggestions = []

    # 市场状态建议
    if regime:
        regime_advice = regime.get("操作建议", "")
        if regime_advice:
            suggestions.append(f"[市场状态] {regime_advice}")

    # 多周期共振建议
    if resonance:
        resonance_advice = resonance.get("操作含义", "")
        if resonance_advice:
            suggestions.append(f"[趋势共振] {resonance_advice}")

    # 量价关系建议
    if vol_price:
        vol_advice = vol_price.get("综合判断", "")
        if vol_advice:
            suggestions.append(f"[量价关系] {vol_advice}")

    # 财报季建议
    if earnings:
        earnings_advice = earnings.get("策略建议", [])
        if isinstance(earnings_advice, list) and earnings_advice:
            suggestions.append(f"[财报季] {earnings_advice[0]}")

    # 日历效应建议
    if calendar:
        cal_advice = calendar.get("综合建议", [])
        if isinstance(cal_advice, list) and cal_advice:
            suggestions.append(f"[日历效应] {cal_advice[0]}")

    # 市场择时建议
    if timing:
        timing_advice = timing.get("操作建议", "")
        if timing_advice:
            suggestions.append(f"[择时] {timing_advice}")

    # 自适应策略建议
    adaptive = result.get("自适应策略", {})
    if adaptive:
        adaptive_advice = adaptive.get("策略建议", [])
        if isinstance(adaptive_advice, list) and adaptive_advice:
            suggestions.append(f"[自适应] {adaptive_advice[0]}")

    report["综合操作建议"] = suggestions

    # 5. 推荐标的详情
    recommended = result.get("推荐股票", [])
    for stock in recommended[:5]:
        detail = {
            "代码": stock.get("代码", "N/A"),
            "名称": stock.get("名称", "N/A"),
            "最新价": stock.get("最新价", "N/A"),
            "涨跌幅": stock.get("涨跌幅", "N/A"),
            "综合评分": stock.get("综合评分", "N/A"),
            "推荐理由": stock.get("推荐理由", "N/A"),
        }
        report["推荐标的详情"].append(detail)

    return report


def _calendar_effect_analyzer():
    """
    日历效应与季节性分析：月初/月末效应、周内效应、节前节后效应、季节性规律
    返回: {当前日期, 月初月末效应, 周内效应, 季节性规律, 综合建议}
    """
    from datetime import datetime

    now = datetime.now()
    month = now.month
    day = now.day
    weekday = now.weekday()  # 0=周一, 6=周日

    calendar = {
        "当前日期": now.strftime("%Y-%m-%d"),
        "星期": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][weekday],
        "月初月末效应": {},
        "周内效应": {},
        "季节性规律": {},
        "综合建议": [],
    }

    # 1. 月初月末效应
    if day <= 5:
        calendar["月初月末效应"] = {
            "阶段": "月初",
            "特征": "月初资金面相对宽松，机构配置需求增加，历史上上涨概率较高",
            "建议": "月初适合积极布局，关注机构增持方向",
        }
    elif day >= 25:
        calendar["月初月末效应"] = {
            "阶段": "月末",
            "特征": "月末资金面偏紧，机构可能锁定收益或调整仓位，波动加大",
            "建议": "月末注意控制仓位，避免追高，关注逆回购利率变化",
        }
    else:
        calendar["月初月末效应"] = {
            "阶段": "月中",
            "特征": "月中市场运行相对平稳，趋势延续性较强",
            "建议": "按正常策略执行，关注趋势方向",
        }

    # 2. 周内效应
    weekday_effects = {
        0: {"效应": "周一效应", "特征": "周末消息面集中消化，开盘波动较大，容易出现跳空", "建议": "周一早盘不宜追涨杀跌，等待市场消化消息后再操作"},
        1: {"效应": "周二", "特征": "市场回归理性，趋势较为明确", "建议": "周二适合执行交易计划"},
        2: {"效应": "周三", "特征": "周中转折点，部分短线资金开始调仓", "建议": "关注盘中资金流向变化"},
        3: {"效应": "周四", "特征": "部分短线资金提前离场规避周末风险，下午容易出现跳水", "建议": "周四下午注意控制风险，不宜重仓过周末"},
        4: {"效应": "周五效应", "特征": "市场对周末消息面有预期，尾盘容易出现异动", "建议": "周五关注尾盘资金动向，可适度布局周末利好预期标的"},
        5: {"效应": "周六（非交易日）", "特征": "", "建议": ""},
        6: {"效应": "周日（非交易日）", "特征": "", "建议": ""},
    }

    calendar["周内效应"] = weekday_effects.get(weekday, {})

    # 3. 季节性规律
    seasonal_patterns = {
        1: {"规律": "春季躁动预热期", "特征": "年初资金面宽松，机构开始布局全年，春季躁动行情酝酿", "建议": "关注年报预告超预期标的，布局全年主线"},
        2: {"规律": "春季躁动进行时", "特征": "历史上2月上涨概率最高（约70%），两会政策预期升温", "建议": "积极参与，关注政策受益板块和业绩超预期标的"},
        3: {"规律": "两会行情", "特征": "两会期间政策密集发布，主题投资活跃，但会后有回调风险", "建议": "两会前布局政策受益板块，会中注意利好兑现风险"},
        4: {"规律": "决断期", "特征": "一季报披露+年报收官，业绩驱动行情，是全年方向的重要决断期", "建议": "基于一季报数据调整全年配置方向"},
        5: {"规律": "五穷", "特征": "历史统计5月上涨概率较低，市场进入业绩空窗期，Sell in May效应", "建议": "适当降低仓位，防御为主，关注高股息和消费防御板块"},
        6: {"规律": "六绝", "特征": "年中资金面偏紧，市场情绪低迷，但也是布局下半年的好时机", "建议": "逢低布局下半年景气行业，关注半年报预告"},
        7: {"规律": "七翻身", "特征": "半年报行情开启，资金面改善，历史上7月反弹概率较高", "建议": "关注半年报超预期标的，适度提高仓位"},
        8: {"规律": "半年报行情", "特征": "半年报密集披露，机构调仓换股，业绩分化加大", "建议": "跟随机构调仓方向，关注业绩超预期行业"},
        9: {"规律": "金九", "特征": "消费旺季+国庆前备货，消费板块活跃，但需注意国庆前资金离场", "建议": "关注消费板块机会，国庆前适当降低仓位"},
        10: {"规律": "银十", "特征": "国庆后资金回流，三季报披露，市场活跃度提升", "建议": "积极参与三季报行情，关注全年业绩确定性高的标的"},
        11: {"规律": "年末布局期", "特征": "机构开始为下一年布局，调仓换股频繁", "建议": "关注机构增持方向，为来年春季行情做准备"},
        12: {"规律": "年末收官", "特征": "年末排名行情+资金回笼压力，市场分化加大", "建议": "防御为主，关注低估值蓝筹，为来年布局"},
    }

    calendar["季节性规律"] = seasonal_patterns.get(month, {})

    # 4. 综合建议
    calendar["综合建议"] = []

    # 结合月初月末
    if day <= 5:
        calendar["综合建议"].append("月初资金面宽松，适合积极布局")
    elif day >= 25:
        calendar["综合建议"].append("月末注意控制仓位和流动性风险")

    # 结合周内
    if weekday == 0:
        calendar["综合建议"].append("周一消化周末消息，不宜追涨杀跌")
    elif weekday == 3:
        calendar["综合建议"].append("周四注意下午可能的短线资金离场")
    elif weekday == 4:
        calendar["综合建议"].append("周五关注尾盘异动和周末消息面预期")

    # 结合季节性
    seasonal = seasonal_patterns.get(month, {})
    if seasonal.get("建议"):
        calendar["综合建议"].append(seasonal["建议"])

    return calendar


def _earnings_season_analyzer():
    """
    财报季效应分析：基于当前日期判断财报季阶段，给出策略调整建议
    A股财报披露规律：一季报4月、半年报7-8月、三季报10月、年报1-4月
    返回: {财报阶段, 阶段特征, 策略建议, 风险提示}
    """
    from datetime import datetime

    now = datetime.now()
    month = now.month
    day = now.day

    earnings = {
        "当前日期": now.strftime("%Y-%m-%d"),
        "财报阶段": "",
        "阶段特征": "",
        "策略建议": [],
        "风险提示": [],
    }

    # 判断财报季阶段
    if month == 1 and day <= 31:
        earnings["财报阶段"] = "年报业绩预告期（1月）"
        earnings["阶段特征"] = "上市公司密集发布年报业绩预告，超预期个股受追捧，低于预期个股承压"
        earnings["策略建议"] = [
            "关注业绩预告超预期的个股，尤其是景气度向上的行业",
            "回避业绩预告可能暴雷的标的，尤其是高估值+业绩不确定的",
            "业绩预告是选股的重要参考，优先选择业绩确定性高的标的",
        ]
        earnings["风险提示"] = [
            "业绩预告不及预期可能导致股价大幅下跌",
            "部分公司可能延迟或模糊披露，需警惕信息不对称风险",
        ]
    elif month == 2 or month == 3:
        earnings["财报阶段"] = "年报披露密集期（2-3月）"
        earnings["阶段特征"] = "年报正式披露，业绩兑现期，利好出尽/利空出尽效应明显"
        earnings["策略建议"] = [
            "年报披露后关注分红方案，高股息标的具有防御价值",
            "业绩符合预期但股价未涨的标的可能存在补涨机会",
            "年报数据是全年基本面分析的基础，重点关注ROE和现金流",
        ]
        earnings["风险提示"] = [
            "利好出尽：业绩好但股价已提前反映，披露后可能回调",
            "年报审计意见需关注，非标意见是重大风险信号",
        ]
    elif month == 4:
        if day <= 15:
            earnings["财报阶段"] = "年报收官+一季报预告期（4月上旬）"
            earnings["阶段特征"] = "年报披露收尾，一季报预告开始，市场关注点从去年业绩转向今年预期"
            earnings["策略建议"] = [
                "一季报是全年业绩的风向标，重点关注一季报超预期的行业",
                "年报披露末尾需警惕业绩变脸的公司",
            ]
        else:
            earnings["财报阶段"] = "一季报披露密集期（4月下旬）"
            earnings["阶段特征"] = "一季报密集披露，业绩驱动行情明显，个股分化加大"
            earnings["策略建议"] = [
                "一季报超预期个股往往能引领二季度行情",
                "关注一季报中营收和利润双增的标的",
            ]
            earnings["风险提示"] = [
                "一季报不及预期可能导致全年预期下调，股价承压",
            ]
    elif month == 5 or month == 6:
        earnings["财报阶段"] = "业绩空窗期（5-6月）"
        earnings["阶段特征"] = "财报披露结束，市场进入业绩空窗期，题材炒作活跃，主题投资占主导"
        earnings["策略建议"] = [
            "业绩空窗期题材股活跃，可适度参与主题投资",
            "基于一季报筛选全年业绩确定性高的标的，逢低布局",
            "关注政策驱动型机会，如产业政策、区域政策等",
        ]
        earnings["风险提示"] = [
            "题材炒作风险较大，注意区分真成长和纯炒作",
            "缺乏业绩验证窗口，概念股容易暴涨暴跌",
        ]
    elif month == 7:
        if day <= 15:
            earnings["财报阶段"] = "半年报业绩预告期（7月上旬）"
            earnings["阶段特征"] = "中小板/创业板强制披露半年报预告，业绩预期修正期"
            earnings["策略建议"] = [
                "半年报预告是检验全年业绩预期的关键节点",
                "关注预告超预期的中小市值标的",
            ]
        else:
            earnings["财报阶段"] = "半年报披露期（7月下旬-8月）"
            earnings["阶段特征"] = "半年报开始披露，业绩验证期，市场关注业绩兑现情况"
            earnings["策略建议"] = [
                "半年报是全年业绩的重要验证点，超预期标的值得重点关注",
                "关注半年报中毛利率和净利率的变化趋势",
            ]
    elif month == 8:
        earnings["财报阶段"] = "半年报披露密集期（8月）"
        earnings["阶段特征"] = "半年报密集披露，业绩分化加大，机构调仓换股频繁"
        earnings["策略建议"] = [
            "半年报数据是下半年投资的重要依据",
            "关注机构持仓变化，跟随聪明资金布局",
        ]
        earnings["风险提示"] = [
            "半年报不及预期可能导致机构集中减持",
        ]
    elif month == 9:
        earnings["财报阶段"] = "业绩空窗期（9月）"
        earnings["阶段特征"] = "半年报结束，三季报尚未开始，叠加国庆长假效应"
        earnings["策略建议"] = [
            "基于半年报筛选优质标的，为四季度布局",
            "国庆前市场通常偏谨慎，控制仓位",
        ]
    elif month == 10:
        earnings["财报阶段"] = "三季报披露期（10月）"
        earnings["阶段特征"] = "三季报披露，全年业绩基本明朗，机构开始布局下一年"
        earnings["策略建议"] = [
            "三季报是全年业绩的最终确认，重点关注全年业绩指引",
            "机构开始为下一年调仓，关注机构增持方向",
        ]
        earnings["风险提示"] = [
            "三季报不及预期意味着全年业绩可能低于预期",
        ]
    elif month == 11 or month == 12:
        earnings["财报阶段"] = "业绩空窗期+年末效应（11-12月）"
        earnings["阶段特征"] = "业绩空窗期，年末排名行情，机构调仓锁定收益"
        earnings["策略建议"] = [
            "年末机构排名行情，关注机构重仓股的表现",
            "为来年春季行情布局，关注估值合理+业绩确定性高的标的",
            "年末资金面偏紧，注意流动性风险",
        ]
        earnings["风险提示"] = [
            "年末机构调仓可能导致重仓股波动加大",
            "部分机构可能锁定收益导致强势股回调",
        ]

    return earnings


def _volume_price_analyzer(spot_df):
    """
    量价关系深度分析：识别经典量价形态
    形态识别：放量突破、缩量回调、量价背离、底部放量、高位滞涨
    返回: {量价评分, 放量突破, 缩量回调, 量价背离, 底部放量, 高位滞涨, 综合判断}
    """
    import pandas as pd
    import numpy as np

    if spot_df is None or spot_df.empty:
        return {"量价评分": 50, "放量突破": {}, "缩量回调": {}, "量价背离": {}, "底部放量": {}, "高位滞涨": {}, "综合判断": "", "处理建议": ["数据不足，无法进行量价关系分析"]}

    vp = {
        "量价评分": 50,
        "放量突破": {},
        "缩量回调": {},
        "量价背离": {},
        "底部放量": {},
        "高位滞涨": {},
        "综合判断": "",
        "处理建议": [],
    }

    try:
        df = spot_df.copy()
        total = len(df)

        chg_today = pd.to_numeric(df['涨跌幅'], errors='coerce').fillna(0)
        chg_5d = pd.to_numeric(df.get('5日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        chg_20d = pd.to_numeric(df.get('20日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        chg_60d = pd.to_numeric(df.get('60日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        turnover = pd.to_numeric(df['换手率'], errors='coerce').fillna(0)
        vol_ratio = pd.to_numeric(df['量比'], errors='coerce').fillna(1)
        amplitude = pd.to_numeric(df['振幅'], errors='coerce').fillna(0)

        # 1. 放量突破：涨幅>3% + 换手率>5% + 量比>1.5
        breakout = df[(chg_today > 3) & (turnover > 5) & (vol_ratio > 1.5)]
        breakout_count = len(breakout)

        if breakout_count > total * 0.05:
            breakout_level = "大量"
        elif breakout_count > total * 0.02:
            breakout_level = "适中"
        else:
            breakout_level = "稀少"

        vp["放量突破"] = {
            "放量突破股票数": breakout_count,
            "占比": f"{breakout_count/total*100:.2f}%" if total > 0 else "0%",
            "突破等级": breakout_level,
            "含义": "放量突破表示资金主动买入，是强势信号" if breakout_count > 0 else "暂无放量突破信号",
        }

        # 2. 缩量回调：跌幅<2% + 换手率<2% + 量比<0.8
        pullback = df[(chg_today < 0) & (chg_today > -2) & (turnover < 2) & (vol_ratio < 0.8)]
        pullback_count = len(pullback)

        if pullback_count > total * 0.1:
            pullback_level = "普遍"
        elif pullback_count > total * 0.05:
            pullback_level = "适中"
        else:
            pullback_level = "稀少"

        vp["缩量回调"] = {
            "缩量回调股票数": pullback_count,
            "占比": f"{pullback_count/total*100:.2f}%" if total > 0 else "0%",
            "回调等级": pullback_level,
            "含义": "缩量回调表示抛压减轻，可能是洗盘而非出货" if pullback_count > 0 else "暂无缩量回调信号",
        }

        # 3. 量价背离
        # 价涨量缩：今日涨但换手率低于5日均值（用换手率<2%且涨>1%近似）
        price_up_vol_down = df[(chg_today > 1) & (turnover < 2) & (vol_ratio < 0.7)]
        pu_count = len(price_up_vol_down)

        # 价跌量增：今日跌但量比>1.5
        price_down_vol_up = df[(chg_today < -1) & (vol_ratio > 1.5)]
        pd_count = len(price_down_vol_up)

        divergence_total = pu_count + pd_count
        if divergence_total > total * 0.08:
            divergence_level = "严重"
        elif divergence_total > total * 0.04:
            divergence_level = "关注"
        else:
            divergence_level = "正常"

        vp["量价背离"] = {
            "价涨量缩(上涨乏力)": f"{pu_count}只",
            "价跌量增(下跌放量)": f"{pd_count}只",
            "背离等级": divergence_level,
            "含义": "量价背离是趋势可能反转的预警信号",
        }

        if divergence_level == "严重":
            vp["处理建议"].append("量价背离严重，上涨乏力+下跌放量并存，市场方向可能即将转变")

        # 4. 底部放量：60日跌幅>20% + 今日涨幅>2% + 量比>1.5
        bottom_breakout = df[(chg_60d < -20) & (chg_today > 2) & (vol_ratio > 1.5)]
        bottom_count = len(bottom_breakout)

        if bottom_count > total * 0.03:
            bottom_level = "较多"
        elif bottom_count > 0:
            bottom_level = "个别"
        else:
            bottom_level = "无"

        vp["底部放量"] = {
            "底部放量反弹股票数": bottom_count,
            "底部等级": bottom_level,
            "含义": "超跌后放量反弹，可能是底部信号，但需确认持续性" if bottom_count > 0 else "暂无底部放量信号",
        }

        if bottom_count > 0:
            bottom_names = bottom_breakout['名称'].head(5).tolist()
            vp["处理建议"].append(f"底部放量反弹标的: {', '.join(str(n) for n in bottom_names)}，关注是否形成底部形态")

        # 5. 高位放量滞涨：20日涨幅>20% + 今日涨幅<2% + 量比>1.5
        top_stall = df[(chg_20d > 20) & (chg_today < 2) & (chg_today > -2) & (vol_ratio > 1.5)]
        stall_count = len(top_stall)

        if stall_count > total * 0.03:
            stall_level = "较多（警惕出货）"
        elif stall_count > 0:
            stall_level = "个别"
        else:
            stall_level = "无"

        vp["高位滞涨"] = {
            "高位放量滞涨股票数": stall_count,
            "滞涨等级": stall_level,
            "含义": "高位放量但涨幅有限，可能是主力出货信号" if stall_count > 0 else "暂无高位滞涨信号",
        }

        if stall_count > total * 0.03:
            vp["处理建议"].append(f"高位放量滞涨股票{stall_count}只，警惕主力出货，持有相关标的建议减仓")

        # 6. 综合判断
        score = 50
        if breakout_level == "大量":
            score += 15
        elif breakout_level == "适中":
            score += 8

        if pullback_level == "普遍":
            score += 10

        if divergence_level == "严重":
            score -= 15
        elif divergence_level == "关注":
            score -= 8

        if stall_count > total * 0.03:
            score -= 10

        vp["量价评分"] = max(0, min(100, score))

        if score >= 70:
            vp["综合判断"] = "量价关系健康，资金主动买入积极，市场做多意愿强"
        elif score >= 55:
            vp["综合判断"] = "量价关系正常，市场运行平稳"
        elif score >= 40:
            vp["综合判断"] = "量价关系偏弱，存在背离或滞涨信号，需谨慎"
        else:
            vp["综合判断"] = "量价关系恶化，放量下跌或高位滞涨明显，建议防御"

    except Exception as e:
        vp["处理建议"] = [f"量价关系分析异常: {str(e)}"]
        vp["量价评分"] = 50

    return vp


def _multi_timeframe_resonance(spot_df):
    """
    多时间框架共振分析：基于5日/20日/60日涨跌幅模拟短中长周期趋势共振
    分析维度：短期趋势、中期趋势、长期趋势、共振判断、共振强度
    返回: {共振评分, 短期趋势, 中期趋势, 长期趋势, 共振状态, 操作含义}
    """
    import pandas as pd
    import numpy as np

    if spot_df is None or spot_df.empty:
        return {"共振评分": 50, "短期趋势": {}, "中期趋势": {}, "长期趋势": {}, "共振状态": "无法判断", "操作含义": [], "处理建议": ["数据不足，无法进行多时间框架分析"]}

    resonance = {
        "共振评分": 50,
        "短期趋势": {},
        "中期趋势": {},
        "长期趋势": {},
        "共振状态": "无法判断",
        "操作含义": [],
        "处理建议": [],
    }

    try:
        df = spot_df.copy()
        total = len(df)

        chg_5d = pd.to_numeric(df.get('5日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        chg_20d = pd.to_numeric(df.get('20日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        chg_60d = pd.to_numeric(df.get('60日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        chg_today = pd.to_numeric(df['涨跌幅'], errors='coerce').fillna(0)

        # 1. 短期趋势（5日）
        avg_5d = chg_5d.mean()
        up_5d_ratio = (chg_5d > 0).sum() / total * 100 if total > 0 else 0

        if avg_5d > 3:
            short_trend = "强势上涨"
            short_score = 80
        elif avg_5d > 1:
            short_trend = "温和上涨"
            short_score = 65
        elif avg_5d > -1:
            short_trend = "横盘震荡"
            short_score = 50
        elif avg_5d > -3:
            short_trend = "温和下跌"
            short_score = 35
        else:
            short_trend = "强势下跌"
            short_score = 20

        resonance["短期趋势"] = {
            "周期": "5日（周线级别）",
            "平均涨跌幅": f"{avg_5d:.2f}%",
            "上涨比例": f"{up_5d_ratio:.1f}%",
            "趋势方向": short_trend,
            "趋势评分": short_score,
        }

        # 2. 中期趋势（20日）
        avg_20d = chg_20d.mean()
        up_20d_ratio = (chg_20d > 0).sum() / total * 100 if total > 0 else 0

        if avg_20d > 8:
            mid_trend = "强势上涨"
            mid_score = 80
        elif avg_20d > 3:
            mid_trend = "温和上涨"
            mid_score = 65
        elif avg_20d > -3:
            mid_trend = "横盘震荡"
            mid_score = 50
        elif avg_20d > -8:
            mid_trend = "温和下跌"
            mid_score = 35
        else:
            mid_trend = "强势下跌"
            mid_score = 20

        resonance["中期趋势"] = {
            "周期": "20日（月线级别）",
            "平均涨跌幅": f"{avg_20d:.2f}%",
            "上涨比例": f"{up_20d_ratio:.1f}%",
            "趋势方向": mid_trend,
            "趋势评分": mid_score,
        }

        # 3. 长期趋势（60日）
        avg_60d = chg_60d.mean()
        up_60d_ratio = (chg_60d > 0).sum() / total * 100 if total > 0 else 0

        if avg_60d > 15:
            long_trend = "强势上涨"
            long_score = 80
        elif avg_60d > 5:
            long_trend = "温和上涨"
            long_score = 65
        elif avg_60d > -5:
            long_trend = "横盘震荡"
            long_score = 50
        elif avg_60d > -15:
            long_trend = "温和下跌"
            long_score = 35
        else:
            long_trend = "强势下跌"
            long_score = 20

        resonance["长期趋势"] = {
            "周期": "60日（季线级别）",
            "平均涨跌幅": f"{avg_60d:.2f}%",
            "上涨比例": f"{up_60d_ratio:.1f}%",
            "趋势方向": long_trend,
            "趋势评分": long_score,
        }

        # 4. 多周期共振判断
        trends = [short_trend, mid_trend, long_trend]
        up_trends = sum(1 for t in trends if "上涨" in t)
        down_trends = sum(1 for t in trends if "下跌" in t)
        sideways = sum(1 for t in trends if "震荡" in t)

        if up_trends == 3:
            resonance["共振状态"] = "三周期共振向上（最强做多信号）"
            resonance_score = 90
        elif up_trends == 2 and down_trends == 0:
            resonance["共振状态"] = "两周期共振向上（偏多信号）"
            resonance_score = 75
        elif down_trends == 3:
            resonance["共振状态"] = "三周期共振向下（最强做空/离场信号）"
            resonance_score = 10
        elif down_trends == 2 and up_trends == 0:
            resonance["共振状态"] = "两周期共振向下（偏空信号）"
            resonance_score = 25
        elif up_trends == 1 and down_trends == 1:
            resonance["共振状态"] = "多周期分歧（趋势不明，谨慎操作）"
            resonance_score = 50
        elif sideways >= 2:
            resonance["共振状态"] = "多周期横盘（等待方向选择）"
            resonance_score = 45
        else:
            resonance["共振状态"] = "周期混合信号（需结合其他指标判断）"
            resonance_score = 50

        # 5. 操作含义
        if resonance_score >= 75:
            resonance["操作含义"] = [
                "多周期共振向上，是最佳做多时机",
                "可适当提高仓位，顺势而为",
                "止损可放宽，让利润奔跑",
                "关注强势板块中的领涨个股",
            ]
        elif resonance_score >= 60:
            resonance["操作含义"] = [
                "多数周期偏多，可适度参与",
                "仓位控制在60%-80%",
                "关注短期趋势是否与中长期一致",
            ]
        elif resonance_score >= 40:
            resonance["操作含义"] = [
                "多周期信号不一致，建议观望或轻仓",
                "仓位控制在30%-50%",
                "等待更明确的共振信号再加大仓位",
            ]
        elif resonance_score >= 25:
            resonance["操作含义"] = [
                "多数周期偏空，建议降低仓位",
                "仓位控制在20%以下",
                "已有盈利及时锁定，不宜新开仓位",
            ]
        else:
            resonance["操作含义"] = [
                "三周期共振向下，强烈建议离场观望",
                "空仓或极轻仓（<10%）",
                "现金为王，等待趋势反转信号",
                "不要逆势抄底，尊重趋势的力量",
            ]

        resonance["共振评分"] = resonance_score

    except Exception as e:
        resonance["处理建议"] = [f"多时间框架分析异常: {str(e)}"]
        resonance["共振评分"] = 50

    return resonance


def _sector_rotation_quantifier(spot_df):
    """
    板块轮动节奏量化：量化行业轮动的速度、强度和方向
    分析维度：轮动速度指数、轮动强度指数、轮动方向、动量持续性、轮动策略
    返回: {轮动评分, 轮动速度, 轮动强度, 轮动方向, 动量持续性, 轮动策略}
    """
    import pandas as pd
    import numpy as np

    if spot_df is None or spot_df.empty or '行业' not in spot_df.columns:
        return {"轮动评分": 50, "轮动速度": {}, "轮动强度": {}, "轮动方向": {}, "动量持续性": {}, "轮动策略": [], "处理建议": ["数据不足或缺少行业分类，无法进行板块轮动量化分析"]}

    rotation = {
        "轮动评分": 50,
        "轮动速度": {},
        "轮动强度": {},
        "轮动方向": {},
        "动量持续性": {},
        "轮动策略": [],
        "处理建议": [],
    }

    try:
        df = spot_df.copy()

        chg_today = pd.to_numeric(df['涨跌幅'], errors='coerce').fillna(0)
        chg_5d = pd.to_numeric(df.get('5日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        chg_20d = pd.to_numeric(df.get('20日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        amount = pd.to_numeric(df['成交额'], errors='coerce').fillna(0)

        # 按行业聚合
        sector_stats = df.groupby('行业').agg(
            平均涨跌幅=('涨跌幅', lambda x: pd.to_numeric(x, errors='coerce').mean()),
            平均5日涨跌=('5日涨跌幅', lambda x: pd.to_numeric(x, errors='coerce').mean()),
            平均20日涨跌=('20日涨跌幅', lambda x: pd.to_numeric(x, errors='coerce').mean()),
            股票数=('代码', 'count'),
            总成交额=('成交额', lambda x: pd.to_numeric(x, errors='coerce').sum()),
            上涨家数=('涨跌幅', lambda x: (pd.to_numeric(x, errors='coerce') > 0).sum()),
            下跌家数=('涨跌幅', lambda x: (pd.to_numeric(x, errors='coerce') < 0).sum()),
        ).reset_index()

        sector_stats = sector_stats[sector_stats['股票数'] >= 3]  # 过滤样本太少的行业

        if len(sector_stats) < 3:
            rotation["处理建议"].append("有效行业数量不足，无法进行轮动分析")
            return rotation

        # 1. 轮动速度指数
        # 基于今日涨跌幅排名与5日涨跌幅排名的变化
        sector_stats['今日排名'] = sector_stats['平均涨跌幅'].rank(ascending=False)
        sector_stats['5日排名'] = sector_stats['平均5日涨跌'].rank(ascending=False)
        sector_stats['排名变化'] = abs(sector_stats['今日排名'] - sector_stats['5日排名'])

        avg_rank_change = sector_stats['排名变化'].mean()
        max_possible_change = len(sector_stats) - 1
        rotation_speed = (avg_rank_change / max_possible_change * 100) if max_possible_change > 0 else 0

        if rotation_speed > 60:
            speed_level = "高速轮动"
        elif rotation_speed > 35:
            speed_level = "中速轮动"
        elif rotation_speed > 15:
            speed_level = "低速轮动"
        else:
            speed_level = "趋势延续"

        rotation["轮动速度"] = {
            "轮动速度指数": f"{rotation_speed:.1f}",
            "平均排名变化": f"{avg_rank_change:.1f}位",
            "轮动速度等级": speed_level,
        }

        # 2. 轮动强度指数
        # 领涨行业与领跌行业的涨跌幅差距
        top3_chg = sector_stats.nlargest(3, '平均涨跌幅')['平均涨跌幅'].mean()
        bottom3_chg = sector_stats.nsmallest(3, '平均涨跌幅')['平均涨跌幅'].mean()
        strength_gap = top3_chg - bottom3_chg

        if strength_gap > 5:
            strength_level = "极强"
        elif strength_gap > 3:
            strength_level = "强"
        elif strength_gap > 1.5:
            strength_level = "中等"
        else:
            strength_level = "弱"

        rotation["轮动强度"] = {
            "领涨行业均涨幅": f"{top3_chg:.2f}%",
            "领跌行业均跌幅": f"{bottom3_chg:.2f}%",
            "强弱差距": f"{strength_gap:.2f}%",
            "轮动强度等级": strength_level,
        }

        # 3. 轮动方向判断
        # 资金流向：成交额占比变化
        total_amount = sector_stats['总成交额'].sum()
        sector_stats['成交占比'] = sector_stats['总成交额'] / total_amount * 100

        # 今日领涨行业
        top_sectors = sector_stats.nlargest(3, '平均涨跌幅')
        bottom_sectors = sector_stats.nsmallest(3, '平均涨跌幅')

        rotation["轮动方向"] = {
            "资金流入行业": top_sectors[['行业', '平均涨跌幅', '成交占比']].to_dict('records'),
            "资金流出行业": bottom_sectors[['行业', '平均涨跌幅', '成交占比']].to_dict('records'),
            "方向判断": "",
        }

        # 判断轮动方向
        if len(top_sectors) > 0 and len(bottom_sectors) > 0:
            top_avg_20d = top_sectors['平均20日涨跌'].mean()
            bottom_avg_20d = bottom_sectors['平均20日涨跌'].mean()

            if top_avg_20d > 5 and bottom_avg_20d < -5:
                rotation["轮动方向"]["方向判断"] = "强者恒强：资金持续流入强势行业，弱势行业继续失血"
            elif top_avg_20d < 0 and bottom_avg_20d < -10:
                rotation["轮动方向"]["方向判断"] = "超跌反弹：前期弱势行业出现反弹，可能是短期修复"
            elif top_avg_20d > 10:
                rotation["轮动方向"]["方向判断"] = "趋势强化：领涨行业趋势明确，可顺势而为"
            else:
                rotation["轮动方向"]["方向判断"] = "轮动切换：行业涨跌交替，无明显持续性方向"

        # 4. 动量持续性
        # 5日领涨行业在20日是否也领涨
        top5_5d = set(sector_stats.nlargest(3, '平均5日涨跌')['行业'].tolist())
        top5_20d = set(sector_stats.nlargest(3, '平均20日涨跌')['行业'].tolist())
        overlap = len(top5_5d & top5_20d)

        if overlap >= 2:
            momentum_level = "强持续"
        elif overlap >= 1:
            momentum_level = "弱持续"
        else:
            momentum_level = "无持续"

        rotation["动量持续性"] = {
            "5日领涨行业": list(top5_5d),
            "20日领涨行业": list(top5_20d),
            "重叠数": overlap,
            "动量持续性": momentum_level,
        }

        # 5. 轮动策略建议
        if speed_level == "高速轮动":
            rotation["轮动策略"].append("行业轮动速度极快，追热点风险大，建议以持有为主，减少换仓")
            rotation["轮动策略"].append("高速轮动环境下，分散配置优于集中押注")
        elif speed_level == "趋势延续":
            rotation["轮动策略"].append("行业趋势延续性强，可适度集中配置强势行业")
            if momentum_level == "强持续":
                rotation["轮动策略"].append("动量持续性强的行业可继续持有，顺势而为")

        if strength_level in ["极强", "强"]:
            rotation["轮动策略"].append(f"行业分化剧烈（强弱差距{strength_gap:.1f}%），选对行业比选对个股更重要")

        if momentum_level == "强持续":
            rotation["轮动策略"].append(f"领涨行业动量持续性强，建议关注: {', '.join(list(top5_5d)[:3])}")

        # 综合评分
        score = 50
        if speed_level == "趋势延续":
            score += 15
        elif speed_level == "高速轮动":
            score -= 10

        if strength_level in ["极强", "强"]:
            score += 10

        if momentum_level == "强持续":
            score += 10
        elif momentum_level == "无持续":
            score -= 10

        rotation["轮动评分"] = max(0, min(100, score))

    except Exception as e:
        rotation["处理建议"] = [f"板块轮动量化分析异常: {str(e)}"]
        rotation["轮动评分"] = 50

    return rotation


def _trading_psychology_assistant(spot_df, market_timing=None):
    """
    交易心理辅助：检测常见交易心理偏差并给出行为纠偏建议
    检测维度：贪婪指标、恐惧指标、过度自信、锚定效应、纪律性评分
    返回: {心理评分, 贪婪检测, 恐惧检测, 过度自信, 锚定效应, 纪律性, 纠偏建议}
    """
    import pandas as pd
    import numpy as np

    if spot_df is None or spot_df.empty:
        return {"心理评分": 50, "贪婪检测": {}, "恐惧检测": {}, "过度自信": {}, "锚定效应": {}, "纪律性": {}, "纠偏建议": ["数据不足，无法进行交易心理分析"]}

    psych = {
        "心理评分": 50,
        "贪婪检测": {},
        "恐惧检测": {},
        "过度自信": {},
        "锚定效应": {},
        "纪律性": {},
        "纠偏建议": [],
    }

    try:
        df = spot_df.copy()
        total = len(df)

        chg_today = pd.to_numeric(df['涨跌幅'], errors='coerce').fillna(0)
        chg_5d = pd.to_numeric(df.get('5日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        chg_20d = pd.to_numeric(df.get('20日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        chg_60d = pd.to_numeric(df.get('60日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        turnover = pd.to_numeric(df['换手率'], errors='coerce').fillna(0)
        vol_ratio = pd.to_numeric(df['量比'], errors='coerce').fillna(1)
        amplitude = pd.to_numeric(df['振幅'], errors='coerce').fillna(0)

        # 1. 贪婪指标检测
        # 追高冲动：连续大涨后今日继续放量追入
        chase_high = df[(chg_5d > 15) & (chg_today > 3) & (vol_ratio > 1.5)]
        chase_count = len(chase_high)

        # 过度交易倾向：高换手+高振幅
        overtrade = df[(turnover > 10) & (amplitude > 8)]
        overtrade_count = len(overtrade)

        greed_score = 100
        if chase_count > total * 0.05:
            greed_score -= 25
            greed_level = "严重"
        elif chase_count > total * 0.02:
            greed_score -= 15
            greed_level = "偏高"
        elif overtrade_count > total * 0.03:
            greed_score -= 10
            greed_level = "关注"
        else:
            greed_level = "正常"

        psych["贪婪检测"] = {
            "追高股票数": chase_count,
            "过度交易股票数": overtrade_count,
            "贪婪等级": greed_level,
            "贪婪评分": greed_score,
        }

        if greed_level == "严重":
            psych["纠偏建议"].append("市场追高情绪严重，此时追入风险极大，建议等待回调再入场")
            psych["纠偏建议"].append("记住：'别人贪婪时我恐惧'，当前应保持冷静")
        elif greed_level == "偏高":
            psych["纠偏建议"].append("市场存在追高倾向，注意不要被短期涨幅冲昏头脑")

        # 2. 恐惧指标检测
        # 恐慌抛售：连续下跌后今日继续放量下跌
        panic_sell = df[(chg_5d < -15) & (chg_today < -3) & (vol_ratio > 1.5)]
        panic_count = len(panic_sell)

        # 过度保守：低换手+低波动（市场有机会但不敢参与）
        overcautious = df[(turnover < 0.5) & (abs(chg_today) < 1) & (chg_20d > 5)]
        cautious_count = len(overcautious)

        fear_score = 100
        if panic_count > total * 0.05:
            fear_score -= 25
            fear_level = "严重"
        elif panic_count > total * 0.02:
            fear_score -= 15
            fear_level = "偏高"
        elif cautious_count > total * 0.1:
            fear_score -= 10
            fear_level = "关注"
        else:
            fear_level = "正常"

        psych["恐惧检测"] = {
            "恐慌抛售股票数": panic_count,
            "过度保守股票数": cautious_count,
            "恐惧等级": fear_level,
            "恐惧评分": fear_score,
        }

        if fear_level == "严重":
            psych["纠偏建议"].append("市场恐慌情绪严重，但需注意：'别人恐惧时我贪婪'，优质标的的恐慌抛售可能是机会")
            psych["纠偏建议"].append("不要被恐慌情绪主导，理性分析基本面，区分系统性风险和个股错杀")
        elif fear_level == "偏高":
            psych["纠偏建议"].append("市场存在恐慌情绪，注意区分理性避险和过度恐慌")

        # 3. 过度自信检测
        # 连续盈利后容易过度自信：市场连续上涨时容易放松风控
        up_streak = (chg_20d > 10).sum()
        up_streak_ratio = up_streak / total * 100 if total > 0 else 0

        overconf_score = 100
        if up_streak_ratio > 50:
            overconf_score -= 20
            overconf_level = "偏高"
        elif up_streak_ratio > 30:
            overconf_score -= 10
            overconf_level = "关注"
        else:
            overconf_level = "正常"

        psych["过度自信"] = {
            "20日上涨超10%股票占比": f"{up_streak_ratio:.1f}%",
            "过度自信等级": overconf_level,
            "过度自信评分": overconf_score,
        }

        if overconf_level == "偏高":
            psych["纠偏建议"].append("市场连续上涨，容易产生过度自信。请严格执行止损纪律，不要因为近期盈利而放松风控")
            psych["纠偏建议"].append("回顾历史：最大的亏损往往发生在连续盈利之后")

        # 4. 锚定效应检测
        # 锚定效应：执着于买入价/历史高点，不愿止损
        # 检测：60日跌幅大但今日缩量（持有者不愿割肉）
        anchored = df[(chg_60d < -30) & (turnover < 1) & (chg_today < 0)]
        anchored_count = len(anchored)

        anchor_score = 100
        if anchored_count > total * 0.05:
            anchor_score -= 20
            anchor_level = "严重"
        elif anchored_count > total * 0.02:
            anchor_score -= 10
            anchor_level = "关注"
        else:
            anchor_level = "正常"

        psych["锚定效应"] = {
            "锚定股票数(深套缩量)": anchored_count,
            "锚定等级": anchor_level,
            "锚定评分": anchor_score,
        }

        if anchor_level == "严重":
            psych["纠偏建议"].append("检测到较多深套缩量股票，可能存在锚定效应——执着于成本价不愿止损")
            psych["纠偏建议"].append("提醒：止损不是承认错误，而是保护本金的必要手段。沉没成本不应影响决策")

        # 5. 纪律性评分
        # 基于市场状态评估当前应遵守的交易纪律
        discipline_score = 100

        # 波动大时需要更严格的纪律
        vol_std = chg_today.std()
        if vol_std > 4:
            discipline_score -= 15

        # 趋势不明时需要更谨慎
        up_ratio = (chg_today > 0).sum() / total * 100 if total > 0 else 50
        if 40 < up_ratio < 60:
            discipline_score -= 10

        psych["纪律性"] = {
            "纪律评分": discipline_score,
            "当前纪律要求": "",
            "纪律清单": [
                "每笔交易前设定止损位，绝不裸奔",
                "单只股票仓位不超过总资金的20%",
                "日内不追涨杀跌，按计划执行",
                "连续亏损3笔后暂停交易，复盘反思",
                "盈利后不急于加仓，等待确认信号",
            ],
        }

        if discipline_score < 70:
            psych["纪律性"]["当前纪律要求"] = "高纪律要求：市场环境复杂，必须严格执行每一条纪律"
        elif discipline_score < 85:
            psych["纪律性"]["当前纪律要求"] = "中等纪律要求：保持基本纪律，可适度灵活"
        else:
            psych["纪律性"]["当前纪律要求"] = "正常纪律要求：按常规纪律执行即可"

        # 6. 综合心理评分
        composite = (greed_score * 0.25 + fear_score * 0.25 + overconf_score * 0.2 + anchor_score * 0.15 + discipline_score * 0.15)
        psych["心理评分"] = round(composite, 1)

        if composite >= 80:
            psych["心理状态"] = "理性"
        elif composite >= 60:
            psych["心理状态"] = "基本理性"
        elif composite >= 40:
            psych["心理状态"] = "情绪化"
        else:
            psych["心理状态"] = "极度情绪化"

        if psych["心理状态"] in ["情绪化", "极度情绪化"]:
            psych["纠偏建议"].append("当前心理状态不佳，建议减少交易频率，避免情绪化决策")

    except Exception as e:
        psych["纠偏建议"] = [f"交易心理分析异常: {str(e)}"]
        psych["心理评分"] = 50

    return psych


def _equity_curve_manager(spot_df, market_timing=None, portfolio_value=1000000, current_drawdown=0):
    """
    资金曲线管理：基于资金曲线的动态风险管理
    管理维度：动态Kelly调整、回撤控制、盈利加仓/亏损减仓、资金曲线健康度
    返回: {资金曲线评分, Kelly调整, 回撤控制, 加仓/减仓规则, 健康度评估}
    """
    import pandas as pd
    import numpy as np

    if spot_df is None or spot_df.empty:
        return {"资金曲线评分": 50, "Kelly调整": {}, "回撤控制": {}, "加仓减仓规则": {}, "健康度评估": {}, "处理建议": ["数据不足，无法进行资金曲线管理"]}

    manager = {
        "资金曲线评分": 50,
        "Kelly调整": {},
        "回撤控制": {},
        "加仓减仓规则": {},
        "健康度评估": {},
        "处理建议": [],
    }

    try:
        df = spot_df.copy()
        total = len(df)

        chg_today = pd.to_numeric(df['涨跌幅'], errors='coerce').fillna(0)
        chg_60d = pd.to_numeric(df.get('60日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        amplitude = pd.to_numeric(df['振幅'], errors='coerce').fillna(0)

        # 1. 动态Kelly调整
        # Kelly公式: f = (p*b - q) / b
        # 简化：基于市场胜率估算最优仓位
        up_ratio = (chg_today > 0).sum() / total if total > 0 else 0.5
        avg_win = chg_today[chg_today > 0].mean() if (chg_today > 0).sum() > 0 else 2.0
        avg_loss = abs(chg_today[chg_today < 0].mean()) if (chg_today < 0).sum() > 0 else 2.0

        if avg_loss > 0:
            win_loss_ratio = avg_win / avg_loss
            kelly_f = (up_ratio * win_loss_ratio - (1 - up_ratio)) / win_loss_ratio
            kelly_f = max(0, min(0.5, kelly_f))  # 限制在0-50%
        else:
            kelly_f = 0.25

        # 半Kelly更稳健
        half_kelly = kelly_f / 2

        manager["Kelly调整"] = {
            "市场胜率": f"{up_ratio*100:.1f}%",
            "盈亏比": f"{win_loss_ratio:.2f}",
            "全Kelly仓位": f"{kelly_f*100:.1f}%",
            "半Kelly仓位(推荐)": f"{half_kelly*100:.1f}%",
            "说明": "半Kelly在保留大部分收益的同时大幅降低回撤风险",
        }

        # 2. 回撤控制
        # 基于60日涨跌幅模拟回撤状态
        avg_60d = chg_60d.mean()
        worst_60d = chg_60d.min()
        deep_dd_count = (chg_60d < -30).sum()

        dd_score = 100
        if worst_60d < -50:
            dd_score -= 30
            dd_level = "严重"
        elif worst_60d < -30:
            dd_score -= 20
            dd_level = "较大"
        elif worst_60d < -15:
            dd_score -= 10
            dd_level = "中等"
        else:
            dd_level = "轻微"

        manager["回撤控制"] = {
            "平均60日涨跌": f"{avg_60d:.2f}%",
            "最差60日涨跌": f"{worst_60d:.2f}%",
            "深度回撤股票数": deep_dd_count,
            "回撤等级": dd_level,
            "回撤评分": dd_score,
        }

        # 回撤控制规则
        if dd_level == "严重":
            manager["处理建议"].append("资金曲线出现严重回撤，建议立即降低仓位至20%以下")
            manager["处理建议"].append("暂停所有新增交易，专注修复资金曲线")
        elif dd_level == "较大":
            manager["处理建议"].append("资金曲线回撤较大，建议降低仓位至40%以下")
            manager["处理建议"].append("收紧止损，每笔交易风险控制在总资金的1%以内")
        elif dd_level == "中等":
            manager["处理建议"].append("资金曲线有回撤，注意控制单笔风险")

        # 3. 盈利加仓/亏损减仓规则
        manager["加仓减仓规则"] = {
            "盈利加仓条件": [
                "资金曲线创新高后回撤不超过5%",
                "连续3笔交易盈利",
                "市场择时评分>60",
            ],
            "盈利加仓方式": [
                "每次加仓不超过总资金的10%",
                "金字塔加仓：首次50%，二次30%，三次20%",
                "加仓后止损上移至盈亏平衡点",
            ],
            "亏损减仓条件": [
                "资金曲线从高点回撤超过10%",
                "连续3笔交易亏损",
                "市场择时评分<40",
            ],
            "亏损减仓方式": [
                "立即降低仓位至原来的50%",
                "暂停新开仓，只平仓不减仓的标的可保留",
                "回撤超过15%时强制休息1周",
            ],
        }

        # 4. 资金曲线健康度评估
        health_score = 100

        # 波动率影响
        vol_std = chg_today.std()
        if vol_std > 5:
            health_score -= 20
        elif vol_std > 3:
            health_score -= 10

        # 回撤影响
        health_score -= (100 - dd_score) * 0.5

        # 趋势影响
        if avg_60d < -20:
            health_score -= 15
        elif avg_60d < -10:
            health_score -= 8

        health_score = max(0, min(100, health_score))

        if health_score >= 80:
            health_level = "健康"
        elif health_score >= 60:
            health_level = "亚健康"
        elif health_score >= 40:
            health_level = "需关注"
        else:
            health_level = "危险"

        manager["健康度评估"] = {
            "健康评分": round(health_score, 1),
            "健康等级": health_level,
            "波动率影响": f"涨跌幅标准差{vol_std:.2f}%",
            "回撤影响": f"回撤评分{dd_score}",
            "趋势影响": f"60日平均涨跌{avg_60d:.2f}%",
        }

        if health_level == "危险":
            manager["处理建议"].append("资金曲线处于危险状态，强烈建议暂停交易，全面复盘")
        elif health_level == "需关注":
            manager["处理建议"].append("资金曲线需要关注，建议降低交易频率和仓位")

        # 5. 结合择时信号
        if market_timing:
            timing_score = market_timing.get("择时评分", 50)
            if timing_score < 40:
                manager["处理建议"].append(f"择时评分{timing_score}分，建议与资金曲线管理联动，进一步降低仓位")

        # 综合评分
        composite = health_score * 0.5 + dd_score * 0.3 + (100 - abs(50 - half_kelly * 100)) * 0.2
        manager["资金曲线评分"] = round(composite, 1)

    except Exception as e:
        manager["处理建议"] = [f"资金曲线管理分析异常: {str(e)}"]
        manager["资金曲线评分"] = 50

    return manager


def _dynamic_hedge_advisor(spot_df, market_timing=None, portfolio_value=1000000):
    """
    动态对冲建议：基于市场状态给出对冲策略建议
    分析维度：对冲必要性、对冲工具选择、对冲比例、成本估算、动态调整规则
    返回: {对冲评分, 对冲必要性, 对冲方案, 成本估算, 调整规则}
    """
    import pandas as pd
    import numpy as np

    if spot_df is None or spot_df.empty:
        return {"对冲评分": 100, "对冲必要性": "无法判断", "对冲方案": [], "成本估算": {}, "调整规则": [], "处理建议": ["数据不足，无法给出对冲建议"]}

    hedge = {
        "对冲评分": 100,
        "对冲必要性": "低",
        "对冲方案": [],
        "成本估算": {},
        "调整规则": [],
        "处理建议": [],
    }

    try:
        df = spot_df.copy()
        total = len(df)

        chg_today = pd.to_numeric(df['涨跌幅'], errors='coerce').fillna(0)
        amplitude = pd.to_numeric(df['振幅'], errors='coerce').fillna(0)
        chg_60d = pd.to_numeric(df.get('60日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)

        # 1. 对冲必要性评估
        down_ratio = (chg_today < 0).sum() / total * 100 if total > 0 else 0
        avg_chg = chg_today.mean()
        vol_std = chg_today.std()
        deep_drawdown = (chg_60d < -30).sum()

        hedge_score = 100
        need_hedge = False

        if down_ratio > 60 and avg_chg < -2:
            hedge_score -= 30
            need_hedge = True
            hedge["对冲必要性"] = "高"
            hedge["处理建议"].append("市场普跌严重，系统性风险极高，强烈建议进行对冲保护")
        elif down_ratio > 50 and avg_chg < -1:
            hedge_score -= 20
            need_hedge = True
            hedge["对冲必要性"] = "中高"
            hedge["处理建议"].append("市场偏弱，系统性风险较高，建议考虑对冲")
        elif vol_std > 4:
            hedge_score -= 15
            need_hedge = True
            hedge["对冲必要性"] = "中"
            hedge["处理建议"].append("市场波动加剧，建议适度对冲以降低组合波动")
        elif deep_drawdown > total * 0.1:
            hedge_score -= 10
            need_hedge = True
            hedge["对冲必要性"] = "中"
            hedge["处理建议"].append(f"{deep_drawdown}只股票60日跌幅超30%，组合回撤风险大，建议对冲")
        else:
            hedge["对冲必要性"] = "低"
            hedge["处理建议"].append("市场运行正常，系统性风险可控，暂不需要对冲")

        # 2. 对冲方案设计
        if need_hedge:
            if hedge_score < 60:
                hedge_ratio = 0.6
                hedge["对冲方案"] = [
                    {
                        "工具": "股指期货空单（IF/IC/IM）",
                        "对冲比例": f"{hedge_ratio*100:.0f}%",
                        "说明": "市场风险极高，建议用股指期货对冲60%的持仓市值",
                        "操作": f"做空{hedge_ratio*portfolio_value/1e4:.0f}万市值的股指期货",
                    },
                    {
                        "工具": "买入看跌期权（保护性Put）",
                        "对冲比例": "30%-50%",
                        "说明": "买入虚值看跌期权，为组合提供下行保护",
                        "操作": "买入对应市值的虚值看跌期权，行权价选择当前价95%位置",
                    },
                ]
            elif hedge_score < 75:
                hedge_ratio = 0.3
                hedge["对冲方案"] = [
                    {
                        "工具": "股指期货空单（IF/IC/IM）",
                        "对冲比例": f"{hedge_ratio*100:.0f}%",
                        "说明": "市场风险较高，建议用股指期货对冲30%的持仓市值",
                        "操作": f"做空{hedge_ratio*portfolio_value/1e4:.0f}万市值的股指期货",
                    },
                    {
                        "工具": "ETF融券卖出",
                        "对冲比例": "20%-30%",
                        "说明": "融券卖出宽基ETF，成本较低，适合中小资金",
                        "操作": "融券卖出沪深300ETF或中证500ETF",
                    },
                ]
            else:
                hedge_ratio = 0.15
                hedge["对冲方案"] = [
                    {
                        "工具": "ETF融券卖出",
                        "对冲比例": f"{hedge_ratio*100:.0f}%",
                        "说明": "市场风险可控，适度对冲即可",
                        "操作": f"融券卖出{hedge_ratio*portfolio_value/1e4:.0f}万市值的宽基ETF",
                    },
                ]

        # 3. 成本估算
        hedge["成本估算"] = {
            "股指期货": "保证金约12%-15%，年化资金成本约3%-5%",
            "看跌期权": "期权费约组合市值的1%-3%/月，取决于波动率",
            "ETF融券": "融券费率约年化8%-10%，部分ETF可能更高",
            "说明": "对冲成本是必要的'保险费'，极端行情下对冲收益远超成本",
        }

        # 4. 动态调整规则
        hedge["调整规则"] = [
            "市场反弹超过3%：减少对冲比例至原来的一半",
            "市场继续下跌超过5%：增加对冲比例至原来的1.5倍",
            "波动率回落至正常水平：逐步平仓对冲头寸",
            "出现明确反转信号：立即平仓所有对冲头寸",
            "每月评估一次对冲效果，优化对冲工具和比例",
        ]

        # 5. 结合择时信号
        if market_timing:
            timing_score = market_timing.get("择时评分", 50)
            if timing_score < 40:
                hedge["处理建议"].append(f"择时系统评分{timing_score}分（偏空），与对冲建议一致，建议执行对冲")
            elif timing_score > 60 and hedge["对冲必要性"] in ["低"]:
                hedge["处理建议"].append(f"择时系统评分{timing_score}分（偏多），当前无需对冲")

        hedge["对冲评分"] = max(0, min(100, hedge_score))

    except Exception as e:
        hedge["处理建议"] = [f"动态对冲分析异常: {str(e)}"]
        hedge["对冲评分"] = 0

    return hedge


def _market_timing_system(spot_df):
    """
    市场择时系统：多维度判断当前市场是否适合进场/离场
    维度：趋势强度、波动率信号、情绪信号、资金流信号、综合择时评分
    返回: {择时评分, 趋势分析, 波动率信号, 情绪信号, 资金流信号, 操作建议}
    """
    import pandas as pd
    import numpy as np

    if spot_df is None or spot_df.empty:
        return {"择时评分": 50, "操作建议": ["数据不足，无法进行市场择时判断"]}

    timing = {
        "择时评分": 50,
        "趋势分析": {},
        "波动率信号": {},
        "情绪信号": {},
        "资金流信号": {},
        "操作建议": [],
    }

    try:
        df = spot_df.copy()
        total = len(df)

        chg_today = pd.to_numeric(df['涨跌幅'], errors='coerce').fillna(0)
        amplitude = pd.to_numeric(df['振幅'], errors='coerce').fillna(0)
        turnover = pd.to_numeric(df['换手率'], errors='coerce').fillna(0)
        vol_ratio = pd.to_numeric(df['量比'], errors='coerce').fillna(1)
        amount = pd.to_numeric(df['成交额'], errors='coerce').fillna(0)
        chg_5d = pd.to_numeric(df.get('5日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        chg_20d = pd.to_numeric(df.get('20日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        chg_60d = pd.to_numeric(df.get('60日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)

        # 1. 趋势强度分析
        up_count = (chg_today > 0).sum()
        down_count = (chg_today < 0).sum()
        flat_count = (chg_today == 0).sum()

        up_ratio = up_count / total * 100 if total > 0 else 0
        avg_chg = chg_today.mean()
        median_chg = chg_today.median()

        # 趋势强度评分
        trend_score = 50
        if up_ratio > 60:
            trend_score += 20
        elif up_ratio > 50:
            trend_score += 10
        elif up_ratio < 40:
            trend_score -= 20
        elif up_ratio < 45:
            trend_score -= 10

        if avg_chg > 1:
            trend_score += 10
        elif avg_chg < -1:
            trend_score -= 10

        timing["趋势分析"] = {
            "上涨家数": up_count,
            "下跌家数": down_count,
            "平盘家数": flat_count,
            "上涨比例": f"{up_ratio:.1f}%",
            "平均涨跌幅": f"{avg_chg:.2f}%",
            "中位数涨跌幅": f"{median_chg:.2f}%",
            "趋势评分": trend_score,
        }

        # 2. 波动率信号（类VIX指标）
        vol_std = chg_today.std()
        amp_mean = amplitude.mean()
        amp_std = amplitude.std()

        vol_score = 50
        if vol_std > 5:
            vol_score -= 20
            vol_level = "极高"
        elif vol_std > 3:
            vol_score -= 10
            vol_level = "偏高"
        elif vol_std < 1.5:
            vol_score += 10
            vol_level = "偏低"
        else:
            vol_level = "正常"

        timing["波动率信号"] = {
            "涨跌幅标准差": f"{vol_std:.2f}%",
            "平均振幅": f"{amp_mean:.2f}%",
            "波动率等级": vol_level,
            "波动率评分": vol_score,
        }

        # 3. 情绪信号
        limit_up = (chg_today > 9.5).sum()
        limit_down = (chg_today < -9.5).sum()
        limit_ratio = (limit_up - limit_down) / total * 100 if total > 0 else 0

        avg_vol_ratio = vol_ratio.mean()
        high_vol = (vol_ratio > 2).sum()

        sentiment_score = 50
        if limit_ratio > 3:
            sentiment_score += 15
        elif limit_ratio > 1:
            sentiment_score += 8
        elif limit_ratio < -3:
            sentiment_score -= 20
        elif limit_ratio < -1:
            sentiment_score -= 10

        if avg_vol_ratio > 1.5:
            sentiment_score += 5
        elif avg_vol_ratio < 0.7:
            sentiment_score -= 5

        timing["情绪信号"] = {
            "涨停家数": limit_up,
            "跌停家数": limit_down,
            "涨跌停净差": limit_up - limit_down,
            "平均量比": f"{avg_vol_ratio:.2f}",
            "放量股票(量比>2)": high_vol,
            "情绪评分": sentiment_score,
        }

        # 4. 资金流信号
        total_amount = amount.sum()
        avg_amount = amount.mean()
        amount_median = amount.median()

        # 成交额集中度（大资金流向）
        top20_amount = amount.nlargest(int(total * 0.2)).sum() if total > 0 else 0
        concentration = top20_amount / total_amount * 100 if total_amount > 0 else 0

        flow_score = 50
        if concentration > 60:
            flow_score += 10
        elif concentration < 40:
            flow_score -= 10

        timing["资金流信号"] = {
            "总成交额": f"{total_amount/1e8:.2f}亿",
            "平均成交额": f"{avg_amount/1e8:.4f}亿",
            "前20%成交集中度": f"{concentration:.1f}%",
            "资金流评分": flow_score,
        }

        # 5. 综合择时评分
        composite = trend_score * 0.35 + vol_score * 0.25 + sentiment_score * 0.25 + flow_score * 0.15
        timing["择时评分"] = round(composite, 1)

        if composite >= 70:
            timing["择时等级"] = "积极"
            timing["仓位建议"] = "80%-100%"
            timing["操作建议"] = [
                "市场环境良好，趋势向上，适合积极操作",
                "可适当提高仓位，关注强势板块",
                "止损可适当放宽至5%-8%",
            ]
        elif composite >= 55:
            timing["择时等级"] = "中性偏多"
            timing["仓位建议"] = "60%-80%"
            timing["操作建议"] = [
                "市场环境尚可，但需保持谨慎",
                "维持中等仓位，精选个股",
                "止损设置在3%-5%",
            ]
        elif composite >= 45:
            timing["择时等级"] = "中性"
            timing["仓位建议"] = "40%-60%"
            timing["操作建议"] = [
                "市场方向不明，建议控制仓位",
                "以持有为主，减少新增买入",
                "止损收紧至3%以内",
            ]
        elif composite >= 30:
            timing["择时等级"] = "偏空"
            timing["仓位建议"] = "20%-40%"
            timing["操作建议"] = [
                "市场偏弱，建议降低仓位",
                "优先减仓弱势标的，保留强势股",
                "止损收紧至2%以内，有盈利及时锁定",
            ]
        else:
            timing["择时等级"] = "防御"
            timing["仓位建议"] = "0%-20%"
            timing["操作建议"] = [
                "市场环境恶劣，建议轻仓或空仓",
                "现金为王，等待市场企稳",
                "如必须持仓，仅保留最核心的防御标的",
            ]

    except Exception as e:
        timing["操作建议"] = [f"市场择时分析异常: {str(e)}"]
        timing["择时评分"] = 50

    return timing


def _multi_strategy_allocation(candidates, spot_df, total_capital=1000000):
    """
    多策略资金分配优化：基于风险预算的多策略资金分配
    策略类型：价值策略、成长策略、动量策略、防御策略、均衡策略
    返回: {策略分配方案, 风险预算, 策略间相关性, 分配建议}
    """
    import pandas as pd
    import numpy as np

    if not candidates or spot_df is None or spot_df.empty:
        return {"策略分配方案": [], "风险预算": {}, "策略间相关性": {}, "分配建议": ["数据不足，无法进行多策略资金分配"]}

    allocation = {
        "策略分配方案": [],
        "风险预算": {},
        "策略间相关性": {},
        "分配建议": [],
    }

    try:
        df = spot_df.copy()

        # 将候选股票按风格分类
        strategies = {
            "价值策略": [],
            "成长策略": [],
            "动量策略": [],
            "防御策略": [],
            "均衡策略": [],
        }

        for c in candidates:
            code = c.get('代码', '')
            pe = c.get('市盈率')
            chg_60d = c.get('60日涨跌幅')
            chg_today = c.get('涨跌幅', 0)
            turnover = c.get('换手率', 0)

            row = df[df['代码'] == code]
            if row.empty:
                strategies["均衡策略"].append(c)
                continue

            r = row.iloc[0]
            pe_val = float(r.get('市盈率-动态', 0) or 0)
            pb_val = float(r.get('市净率', 0) or 0)
            chg_60d_val = float(r.get('60日涨跌幅', 0) or 0)
            chg_today_val = float(r.get('涨跌幅', 0) or 0)
            turnover_val = float(r.get('换手率', 0) or 0)

            # 风格分类
            if pe_val > 0 and pe_val < 15 and pb_val > 0 and pb_val < 2:
                strategies["价值策略"].append(c)
            elif chg_60d_val > 20 and turnover_val > 3:
                strategies["动量策略"].append(c)
            elif chg_60d_val > 10 and pe_val > 20:
                strategies["成长策略"].append(c)
            elif pe_val > 0 and pe_val < 20 and chg_60d_val < 10:
                strategies["防御策略"].append(c)
            else:
                strategies["均衡策略"].append(c)

        # 确保每个策略至少有一只股票
        all_assigned = []
        for strategy_name, stocks in strategies.items():
            all_assigned.extend(stocks)

        unassigned = [c for c in candidates if c not in all_assigned]
        for c in unassigned:
            strategies["均衡策略"].append(c)

        # 计算各策略的风险特征
        strategy_risks = {}
        for name, stocks in strategies.items():
            if stocks:
                chgs = []
                for s in stocks:
                    code = s.get('代码', '')
                    row = df[df['代码'] == code]
                    if not row.empty:
                        chg = float(row.iloc[0].get('涨跌幅', 0) or 0)
                        chgs.append(chg)
                if chgs:
                    strategy_risks[name] = {
                        "股票数": len(stocks),
                        "平均涨跌幅": np.mean(chgs),
                        "波动率": np.std(chgs) if len(chgs) > 1 else 2.0,
                        "最大涨幅": max(chgs),
                        "最大跌幅": min(chgs),
                    }
                else:
                    strategy_risks[name] = {"股票数": len(stocks), "波动率": 3.0}
            else:
                strategy_risks[name] = {"股票数": 0, "波动率": 0}

        # 风险平价分配：权重与波动率成反比
        active_strategies = {k: v for k, v in strategy_risks.items() if v["股票数"] > 0 and v.get("波动率", 0) > 0}

        if active_strategies:
            inv_vol = {k: 1.0 / max(v.get("波动率", 1), 0.5) for k, v in active_strategies.items()}
            total_inv_vol = sum(inv_vol.values())

            for name in active_strategies:
                weight = inv_vol[name] / total_inv_vol * 100
                risk_budget = weight  # 风险预算等于权重（风险平价）

                capital = total_capital * weight / 100
                stock_count = active_strategies[name]["股票数"]
                per_stock = capital / stock_count if stock_count > 0 else 0

                allocation["策略分配方案"].append({
                    "策略": name,
                    "资金权重": f"{weight:.1f}%",
                    "分配金额": f"{capital/10000:.1f}万",
                    "股票数": stock_count,
                    "单只金额": f"{per_stock/10000:.2f}万",
                    "风险特征": f"波动率{active_strategies[name].get('波动率', 0):.2f}%",
                })

            allocation["风险预算"] = {
                "分配方法": "风险平价（Risk Parity）",
                "说明": "各策略分配的风险预算相等，波动率低的策略获得更多资金",
                "总资金": f"{total_capital/10000:.0f}万",
            }

        # 策略间相关性（基于涨跌幅方向）
        correlations = []
        strategy_names = list(active_strategies.keys())
        for i in range(len(strategy_names)):
            for j in range(i + 1, len(strategy_names)):
                si = strategy_names[i]
                sj = strategy_names[j]
                avg_i = active_strategies[si].get("平均涨跌幅", 0)
                avg_j = active_strategies[sj].get("平均涨跌幅", 0)

                if avg_i * avg_j > 0:
                    corr_type = "正相关"
                elif avg_i * avg_j < 0:
                    corr_type = "负相关（分散效果好）"
                else:
                    corr_type = "不相关"

                correlations.append({
                    "策略对": f"{si} vs {sj}",
                    "相关性": corr_type,
                })

        allocation["策略间相关性"] = {
            "相关性分析": correlations,
            "分散化评估": "",
        }

        # 分散化评估
        neg_corr = sum(1 for c in correlations if "负相关" in c.get("相关性", ""))
        if neg_corr > 0:
            allocation["策略间相关性"]["分散化评估"] = f"存在{neg_corr}对负相关策略，多策略分散效果良好"
        else:
            allocation["策略间相关性"]["分散化评估"] = "策略间多为正相关，分散效果有限，建议增加不同风格的策略"

        # 分配建议
        total_weight = sum(float(s["资金权重"].replace("%", "")) for s in allocation["策略分配方案"])
        if total_weight > 0:
            max_weight_strategy = max(allocation["策略分配方案"], key=lambda x: float(x["资金权重"].replace("%", "")))
            allocation["分配建议"].append(f"资金分配最多的策略是'{max_weight_strategy['策略']}'({max_weight_strategy['资金权重']})，波动率最低，风险调整后收益潜力最大")

        allocation["分配建议"].append("以上分配基于风险平价原则，实际执行时可根据市场状态和个人偏好调整")

    except Exception as e:
        allocation["分配建议"] = [f"多策略资金分配异常: {str(e)}"]

    return allocation


def _extreme_market_handler(spot_df, candidates=None):
    """
    极端行情应对机制：涨跌停、极端波动、停牌等异常情况的系统化应对
    检测维度：涨跌停风险、极端波动、流动性危机、连续异常、应对策略
    返回: {极端行情评分, 涨跌停分析, 极端波动, 流动性危机, 连续异常, 应对策略, 处理建议}
    """
    import pandas as pd
    import numpy as np

    if spot_df is None or spot_df.empty:
        return {"极端行情评分": 100, "涨跌停分析": {}, "极端波动": {}, "流动性危机": {}, "连续异常": {}, "应对策略": [], "处理建议": ["数据不足，无法进行极端行情分析"]}

    handler = {
        "极端行情评分": 100,
        "涨跌停分析": {},
        "极端波动": {},
        "流动性危机": {},
        "连续异常": {},
        "应对策略": [],
        "处理建议": [],
    }

    try:
        df = spot_df.copy()
        total = len(df)

        chg_today = pd.to_numeric(df['涨跌幅'], errors='coerce').fillna(0)
        amplitude = pd.to_numeric(df['振幅'], errors='coerce').fillna(0)
        turnover = pd.to_numeric(df['换手率'], errors='coerce').fillna(0)
        vol_ratio = pd.to_numeric(df['量比'], errors='coerce').fillna(1)
        chg_5d = pd.to_numeric(df.get('5日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)

        # 1. 涨跌停分析
        limit_up_mask = chg_today > 9.5
        limit_down_mask = chg_today < -9.5

        limit_up_stocks = df[limit_up_mask]
        limit_down_stocks = df[limit_down_mask]

        limit_up_count = len(limit_up_stocks)
        limit_down_count = len(limit_down_stocks)

        limit_analysis = {
            "涨停股票数": limit_up_count,
            "跌停股票数": limit_down_count,
            "涨停占比": f"{limit_up_count/total*100:.2f}%" if total > 0 else "0%",
            "跌停占比": f"{limit_down_count/total*100:.2f}%" if total > 0 else "0%",
        }

        # 涨停板流动性风险评估
        if limit_up_count > 0:
            limit_up_turnover = limit_up_stocks['换手率'].astype(float)
            low_turnover_limit_up = (limit_up_turnover < 1).sum()
            limit_analysis["涨停低换手(<1%)"] = f"{low_turnover_limit_up}只（封板牢固）"
            limit_analysis["涨停高换手(>5%)"] = f"{(limit_up_turnover > 5).sum()}只（封板松动风险）"

        # 跌停板流动性风险
        if limit_down_count > 0:
            limit_down_turnover = limit_down_stocks['换手率'].astype(float)
            no_turnover_limit_down = (limit_down_turnover < 0.3).sum()
            limit_analysis["跌停无量(<0.3%)"] = f"{no_turnover_limit_down}只（流动性枯竭，无法卖出）"

        handler["涨跌停分析"] = limit_analysis

        # 涨跌停风险评分
        if limit_down_count > total * 0.05:
            handler["极端行情评分"] -= 25
            handler["处理建议"].append(f"跌停股票占比{limit_down_count/total*100:.1f}%，市场恐慌情绪严重，建议暂停买入操作")
        elif limit_down_count > total * 0.02:
            handler["极端行情评分"] -= 15
            handler["处理建议"].append(f"跌停股票{limit_down_count}只，市场存在恐慌，建议谨慎操作")

        if limit_up_count > total * 0.1:
            handler["极端行情评分"] -= 10
            handler["处理建议"].append(f"涨停股票占比{limit_up_count/total*100:.1f}%，市场过热，追高风险大")

        # 2. 极端波动检测
        extreme_amp = df[amplitude > 15]
        extreme_amp_count = len(extreme_amp)

        extreme_vol = {
            "极端振幅股票(>15%)": extreme_amp_count,
            "极端振幅占比": f"{extreme_amp_count/total*100:.2f}%" if total > 0 else "0%",
            "平均振幅": f"{amplitude.mean():.2f}%",
            "最大振幅": f"{amplitude.max():.2f}%",
        }

        if extreme_amp_count > total * 0.05:
            extreme_vol["波动等级"] = "极端"
            handler["极端行情评分"] -= 20
            handler["处理建议"].append(f"{extreme_amp_count}只股票振幅超15%，市场极端波动，建议暂停交易或大幅降低仓位")
        elif extreme_amp_count > total * 0.02:
            extreme_vol["波动等级"] = "剧烈"
            handler["极端行情评分"] -= 10
        else:
            extreme_vol["波动等级"] = "正常"

        handler["极端波动"] = extreme_vol

        # 3. 流动性危机检测
        # 缩量跌停+无量下跌的组合
        liquidity_crisis = df[(chg_today < -7) & (turnover < 0.5)]
        crisis_count = len(liquidity_crisis)

        handler["流动性危机"] = {
            "无量暴跌股票": crisis_count,
            "危机等级": "严重" if crisis_count > 10 else ("关注" if crisis_count > 3 else "无"),
        }

        if crisis_count > 10:
            handler["极端行情评分"] -= 20
            handler["处理建议"].append(f"{crisis_count}只股票无量暴跌，存在流动性危机，持有相关标的需高度警惕")
        elif crisis_count > 3:
            handler["极端行情评分"] -= 10

        # 4. 连续异常检测
        # 连续大涨后今日放量滞涨（出货信号）
        continuous_abnormal = df[
            (chg_5d > 20) & (chg_today < 2) & (vol_ratio > 1.5)
        ]
        abnormal_count = len(continuous_abnormal)

        # 连续大跌后今日继续放量下跌（恐慌蔓延）
        continuous_panic = df[
            (chg_5d < -20) & (chg_today < -3) & (vol_ratio > 1.5)
        ]
        panic_count = len(continuous_panic)

        handler["连续异常"] = {
            "连续大涨后转弱": f"{abnormal_count}只（可能出货）",
            "连续大跌后续跌": f"{panic_count}只（恐慌蔓延）",
        }

        if panic_count > total * 0.03:
            handler["极端行情评分"] -= 15
            handler["处理建议"].append(f"{panic_count}只股票连续大跌后继续放量下跌，恐慌情绪蔓延，不宜抄底")

        # 5. 推荐标的极端行情检查
        if candidates:
            cand_codes = [c.get('代码', '') for c in candidates[:5]]
            cand_df = df[df['代码'].isin(cand_codes)]

            if len(cand_df) > 0:
                cand_limit = cand_df[(cand_df['涨跌幅'].astype(float) > 9.5) | (cand_df['涨跌幅'].astype(float) < -9.5)]
                if len(cand_limit) > 0:
                    names = cand_limit['名称'].tolist()
                    handler["处理建议"].append(f"推荐标的中{', '.join(str(n) for n in names)}处于涨跌停状态，需关注流动性风险")

        # 6. 应对策略
        score = handler["极端行情评分"]

        if score >= 85:
            handler["应对策略"] = [
                "市场运行正常，按常规策略执行",
                "维持正常仓位和止损止盈设置",
            ]
        elif score >= 65:
            handler["应对策略"] = [
                "市场出现局部异常，适当降低仓位至60%",
                "收紧止损幅度至3%以内",
                "避免追高买入，优先选择回调标的",
            ]
        elif score >= 45:
            handler["应对策略"] = [
                "市场异常程度较高，降低仓位至40%以下",
                "止损收紧至2%以内，有盈利及时锁定",
                "暂停新增买入，以持有和减仓为主",
                "关注跌停股票是否出现流动性危机",
            ]
        else:
            handler["应对策略"] = [
                "市场处于极端状态，建议仓位降至20%以下或空仓",
                "立即检查持仓中是否有跌停/无量下跌标的",
                "暂停所有买入操作，优先保护本金",
                "等待市场企稳后再考虑重新入场",
                "极端行情下现金为王，保留充足流动性",
            ]

        handler["极端行情评分"] = max(0, min(100, score))

        if score < 50:
            handler["极端行情等级"] = "极端"
        elif score < 70:
            handler["极端行情等级"] = "异常"
        elif score < 85:
            handler["极端行情等级"] = "关注"
        else:
            handler["极端行情等级"] = "正常"

    except Exception as e:
        handler["处理建议"] = [f"极端行情分析异常: {str(e)}"]
        handler["极端行情评分"] = 0

    return handler


def _realtime_risk_monitor(spot_df, candidates=None):
    """
    盘中实时风险监控增强：实时VaR、回撤监控、集中度风险、尾部风险
    检测维度：VaR计算、回撤估算、行业集中度、流动性风险、尾部风险
    返回: {风险评分, VaR分析, 回撤监控, 集中度风险, 流动性风险, 尾部风险, 风险预警, 处理建议}
    """
    import pandas as pd
    import numpy as np

    if spot_df is None or spot_df.empty:
        return {"风险评分": 100, "VaR分析": {}, "回撤监控": {}, "集中度风险": {}, "流动性风险": {}, "尾部风险": {}, "风险预警": [], "处理建议": ["数据不足，无法进行实时风险监控"]}

    monitor = {
        "风险评分": 100,
        "VaR分析": {},
        "回撤监控": {},
        "集中度风险": {},
        "流动性风险": {},
        "尾部风险": {},
        "风险预警": [],
        "处理建议": [],
    }

    try:
        df = spot_df.copy()
        total = len(df)

        chg_today = pd.to_numeric(df['涨跌幅'], errors='coerce').fillna(0)
        chg_60d = pd.to_numeric(df.get('60日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        turnover = pd.to_numeric(df['换手率'], errors='coerce').fillna(0)
        amount = pd.to_numeric(df.get('成交额', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        amplitude = pd.to_numeric(df['振幅'], errors='coerce').fillna(0)

        # 1. 实时VaR计算（历史模拟法，基于涨跌幅分布）
        chg_clean = chg_today.dropna()
        if len(chg_clean) > 0:
            var_95 = np.percentile(chg_clean, 5)
            var_99 = np.percentile(chg_clean, 1)
            cvar_95 = chg_clean[chg_clean <= var_95].mean() if (chg_clean <= var_95).any() else var_95

            var_level = "高" if var_95 < -5 else ("中" if var_95 < -3 else "低")

            monitor["VaR分析"] = {
                "VaR_95": f"{var_95:.2f}%",
                "VaR_99": f"{var_99:.2f}%",
                "CVaR_95": f"{cvar_95:.2f}%",
                "VaR等级": var_level,
                "说明": f"在95%置信水平下，单日最大损失不超过{abs(var_95):.2f}%",
            }

            if var_level == "高":
                monitor["风险评分"] -= 20
                monitor["风险预警"].append(f"VaR风险高(VaR_95={var_95:.2f}%)，单日潜在损失较大")
            elif var_level == "中":
                monitor["风险评分"] -= 10

        # 2. 实时回撤监控（基于60日涨跌幅推断最大回撤）
        chg_60d_clean = chg_60d.dropna()
        if len(chg_60d_clean) > 0:
            worst_60d = chg_60d_clean.min()
            avg_60d = chg_60d_clean.mean()
            deep_drawdown = (chg_60d_clean < -30).sum()

            dd_level = "严重" if worst_60d < -40 else ("较大" if worst_60d < -25 else ("中等" if worst_60d < -15 else "轻微"))

            monitor["回撤监控"] = {
                "最差60日涨跌": f"{worst_60d:.2f}%",
                "平均60日涨跌": f"{avg_60d:.2f}%",
                "深度回撤股票数(<-30%)": deep_drawdown,
                "回撤等级": dd_level,
            }

            if dd_level == "严重":
                monitor["风险评分"] -= 15
                monitor["风险预警"].append(f"回撤风险严重，{deep_drawdown}只股票60日跌幅超30%")
            elif dd_level == "较大":
                monitor["风险评分"] -= 10

        # 3. 行业集中度风险
        if '行业' in df.columns:
            sector_amount = df.groupby('行业')['成交额'].apply(
                lambda x: pd.to_numeric(x, errors='coerce').sum()
            ).dropna()
            total_amount = sector_amount.sum()

            if total_amount > 0:
                top_sector = sector_amount.nlargest(3)
                top3_ratio = top_sector.sum() / total_amount * 100
                max_sector = top_sector.index[0]
                max_ratio = top_sector.iloc[0] / total_amount * 100

                conc_level = "高" if top3_ratio > 60 else ("中" if top3_ratio > 40 else "低")

                monitor["集中度风险"] = {
                    "前3行业成交占比": f"{top3_ratio:.1f}%",
                    "最大行业": max_sector,
                    "最大行业占比": f"{max_ratio:.1f}%",
                    "集中度等级": conc_level,
                }

                if conc_level == "高":
                    monitor["风险评分"] -= 15
                    monitor["风险预警"].append(f"行业集中度过高，前3行业成交占比{top3_ratio:.1f}%，系统性风险较大")

        # 4. 流动性风险
        low_turnover = df[(turnover < 0.5) & (amount > 0)]
        low_liq_ratio = len(low_turnover) / total * 100 if total > 0 else 0

        liq_level = "紧张" if low_liq_ratio > 20 else ("偏紧" if low_liq_ratio > 10 else "正常")

        monitor["流动性风险"] = {
            "低换手率股票数(<0.5%)": len(low_turnover),
            "低流动性占比": f"{low_liq_ratio:.1f}%",
            "流动性等级": liq_level,
        }

        if liq_level == "紧张":
            monitor["风险评分"] -= 15
            monitor["风险预警"].append(f"流动性风险高，{low_liq_ratio:.1f}%的股票换手率低于0.5%")
        elif liq_level == "偏紧":
            monitor["风险评分"] -= 8

        # 5. 尾部风险（偏度和峰度）
        if len(chg_clean) > 10:
            skewness = chg_clean.skew()
            kurtosis = chg_clean.kurtosis()

            tail_risk = "高" if abs(skewness) > 1 or kurtosis > 5 else ("中" if abs(skewness) > 0.5 or kurtosis > 3 else "低")

            monitor["尾部风险"] = {
                "偏度": round(float(skewness), 2),
                "峰度": round(float(kurtosis), 2),
                "尾部风险等级": tail_risk,
                "说明": "",
            }

            if skewness < -0.5:
                monitor["尾部风险"]["说明"] = "涨跌幅分布左偏，极端下跌风险较高"
                monitor["风险评分"] -= 10
                monitor["风险预警"].append(f"尾部风险：涨跌幅左偏({skewness:.2f})，极端下跌概率较高")
            elif skewness > 0.5:
                monitor["尾部风险"]["说明"] = "涨跌幅分布右偏，极端上涨机会较多"
            if kurtosis > 5:
                monitor["尾部风险"]["说明"] += "，峰度高，极端行情概率增加"
                monitor["风险评分"] -= 10
                monitor["风险预警"].append(f"尾部风险：峰度{kurtosis:.2f}，厚尾特征明显，极端行情概率高")

        # 6. 推荐股票专项风险
        if candidates:
            cand_codes = [c.get('代码', '') for c in candidates[:5]]
            cand_df = df[df['代码'].isin(cand_codes)]

            if len(cand_df) > 0:
                cand_avg_chg = cand_df['涨跌幅'].astype(float).mean()
                cand_max_dd = cand_df.get('60日涨跌幅', pd.Series(0, index=cand_df.index)).astype(float).min()

                if cand_max_dd < -30:
                    monitor["风险预警"].append(f"推荐标的中存在60日跌幅超30%的股票，需关注个股风险")

        # 7. 综合风险评估
        score = monitor["风险评分"]
        if score >= 85:
            monitor["风险等级"] = "低风险"
            monitor["处理建议"].append("当前市场风险水平较低，可正常运作策略")
        elif score >= 65:
            monitor["风险等级"] = "中风险"
            monitor["处理建议"].append("市场存在一定风险，建议适当降低仓位或收紧止损")
        elif score >= 45:
            monitor["风险等级"] = "高风险"
            monitor["处理建议"].append("市场风险较高，建议降低仓位至50%以下，严格止损")
        else:
            monitor["风险等级"] = "极高风险"
            monitor["处理建议"].append("市场风险极高，建议大幅降低仓位或暂时观望")

        monitor["风险评分"] = max(0, min(100, score))

    except Exception as e:
        monitor["处理建议"] = [f"实时风险监控异常: {str(e)}"]
        monitor["风险评分"] = 0

    return monitor


def _alpha_decay_detection(spot_df, factor_ic=None):
    """
    策略Alpha衰减检测：监控因子有效性的衰减趋势
    检测维度：因子拥挤度、因子收益衰减、信号质量变化、换手率异常
    返回: {衰减评分, 拥挤度分析, 收益衰减, 信号质量, 衰减预警, 处理建议}
    """
    import pandas as pd
    import numpy as np

    if spot_df is None or spot_df.empty:
        return {"衰减评分": 100, "拥挤度分析": {}, "收益衰减": {}, "信号质量": {}, "衰减预警": [], "处理建议": ["数据不足，无法进行Alpha衰减检测"]}

    detection = {
        "衰减评分": 100,
        "拥挤度分析": {},
        "收益衰减": {},
        "信号质量": {},
        "衰减预警": [],
        "处理建议": [],
    }

    try:
        df = spot_df.copy()
        total = len(df)

        chg_today = pd.to_numeric(df['涨跌幅'], errors='coerce').fillna(0)
        turnover = pd.to_numeric(df['换手率'], errors='coerce').fillna(0)
        vol_ratio = pd.to_numeric(df['量比'], errors='coerce').fillna(1)
        amount = pd.to_numeric(df.get('成交额', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        chg_5d = pd.to_numeric(df.get('5日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        chg_20d = pd.to_numeric(df.get('20日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        chg_60d = pd.to_numeric(df.get('60日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)

        # 1. 因子拥挤度分析
        # 高换手率股票集中度：如果大量资金集中在少数高换手股票，说明因子拥挤
        high_turnover = df[turnover > 10]
        high_turnover_amount = high_turnover.get('成交额', pd.Series(0, index=high_turnover.index)).astype(float).sum() if len(high_turnover) > 0 else 0
        total_amount = amount.sum()

        if total_amount > 0:
            concentration = high_turnover_amount / total_amount * 100
            top10_amount = amount.nlargest(10).sum()
            top10_concentration = top10_amount / total_amount * 100

            crowd_level = "高" if concentration > 30 else ("中" if concentration > 15 else "低")

            detection["拥挤度分析"] = {
                "高换手股票资金占比": f"{concentration:.1f}%",
                "前10成交额集中度": f"{top10_concentration:.1f}%",
                "拥挤等级": crowd_level,
                "高换手股票数": len(high_turnover),
            }

            if crowd_level == "高":
                detection["衰减评分"] -= 20
                detection["衰减预警"].append("因子拥挤度高，资金过度集中于少数高换手股票，Alpha可能快速衰减")
            elif crowd_level == "中":
                detection["衰减评分"] -= 10
                detection["衰减预警"].append("因子拥挤度中等，需持续关注")

        # 2. 因子收益衰减检测
        # 基于不同周期涨跌幅的分布特征推断因子收益变化
        decay_signals = []

        # 检测中期趋势因子是否失效（60日涨幅与20日涨幅的关系）
        if len(chg_60d.dropna()) > 0 and len(chg_20d.dropna()) > 0:
            # 如果60日强势股在近20日表现转弱，说明趋势因子可能衰减
            strong_60d = df[chg_60d > 20]
            if len(strong_60d) > 5:
                avg_20d_of_strong = strong_60d['20日涨跌幅'].astype(float).mean()
                if avg_20d_of_strong < 0:
                    decay_signals.append(f"60日强势股近20日平均涨幅{avg_20d_of_strong:.1f}%，趋势因子可能衰减")
                    detection["衰减评分"] -= 15

        # 检测动量因子是否失效（5日涨幅与当日涨幅的关系）
        if len(chg_5d.dropna()) > 0:
            strong_5d = df[chg_5d > 10]
            if len(strong_5d) > 5:
                avg_today_of_strong = strong_5d['涨跌幅'].astype(float).mean()
                if avg_today_of_strong < -1:
                    decay_signals.append(f"5日强势股今日平均涨幅{avg_today_of_strong:.1f}%，短期动量因子可能衰减")
                    detection["衰减评分"] -= 10

        # 检测低估值因子是否失效
        if '市盈率-动态' in df.columns:
            pe = pd.to_numeric(df['市盈率-动态'], errors='coerce')
            low_pe = df[(pe > 0) & (pe < 15)]
            if len(low_pe) > 5:
                avg_chg_low_pe = low_pe['涨跌幅'].astype(float).mean()
                if avg_chg_low_pe < -0.5:
                    decay_signals.append(f"低估值股票今日平均涨幅{avg_chg_low_pe:.1f}%，价值因子可能衰减")
                    detection["衰减评分"] -= 8

        detection["收益衰减"] = {
            "衰减信号": decay_signals if decay_signals else ["未检测到明显因子收益衰减"],
            "检测方法": "基于不同周期涨跌幅的因子收益对比",
        }

        # 3. 信号质量检测
        # 基于量价关系的异常程度
        quality_signals = []

        # 检测放量滞涨（量比高但涨幅小，可能是出货信号）
        volume_stall = df[(vol_ratio > 2) & (chg_today.abs() < 1) & (turnover > 3)]
        if len(volume_stall) > total * 0.03:
            quality_signals.append(f"{len(volume_stall)}只股票放量滞涨，信号质量下降")
            detection["衰减评分"] -= 10

        # 检测缩量急跌（量比低但跌幅大，可能是流动性危机）
        volume_crash = df[(vol_ratio < 0.5) & (chg_today < -5)]
        if len(volume_crash) > total * 0.02:
            quality_signals.append(f"{len(volume_crash)}只股票缩量急跌，市场信号失真")
            detection["衰减评分"] -= 10

        detection["信号质量"] = {
            "质量信号": quality_signals if quality_signals else ["信号质量正常"],
            "放量滞涨数": len(volume_stall),
            "缩量急跌数": len(volume_crash),
        }

        # 4. 综合衰减评估
        score = detection["衰减评分"]
        if score >= 90:
            detection["衰减等级"] = "无衰减"
            detection["处理建议"].append("因子Alpha未见明显衰减，策略可正常运行")
        elif score >= 70:
            detection["衰减等级"] = "轻微衰减"
            detection["处理建议"].append("因子Alpha出现轻微衰减迹象，建议降低相关因子权重10%-20%")
        elif score >= 50:
            detection["衰减等级"] = "中度衰减"
            detection["处理建议"].append("因子Alpha中度衰减，建议降低相关因子权重30%-50%，增加新因子替代")
        else:
            detection["衰减等级"] = "严重衰减"
            detection["处理建议"].append("因子Alpha严重衰减，建议暂停使用相关因子，重新进行因子挖掘")

        # 5. 因子IC趋势（如果有历史IC数据）
        if factor_ic and isinstance(factor_ic, dict):
            ic_summary = factor_ic.get("IC汇总", {})
            avg_ic = ic_summary.get("平均Rank_IC", 0)
            if isinstance(avg_ic, (int, float)):
                if abs(avg_ic) < 0.02:
                    detection["衰减预警"].append(f"当前平均Rank IC仅{avg_ic:.4f}，因子预测能力极弱")
                    detection["衰减评分"] = max(0, detection["衰减评分"] - 15)

        detection["衰减评分"] = max(0, min(100, score))

    except Exception as e:
        detection["处理建议"] = [f"Alpha衰减检测异常: {str(e)}"]
        detection["衰减评分"] = 0

    return detection


def _data_quality_pipeline(spot_df):
    """
    数据预处理与质量管道：统一的数据清洗、异常值检测、缺失值处理
    检测维度：缺失值分析、异常值检测(3σ+MAD)、数据一致性、过期数据检测
    返回: {质量评分, 缺失值报告, 异常值报告, 数据一致性, 清洗后数据, 处理建议}
    """
    import pandas as pd
    import numpy as np

    if spot_df is None or spot_df.empty:
        return {"质量评分": 0, "缺失值报告": {}, "异常值报告": {}, "数据一致性": {}, "清洗后数据": None, "处理建议": ["数据为空，无法进行质量评估"]}

    pipeline = {
        "质量评分": 100,
        "缺失值报告": {},
        "异常值报告": {},
        "数据一致性": {},
        "清洗后数据": None,
        "处理建议": [],
    }

    try:
        df = spot_df.copy()
        total_rows = len(df)

        # 关键字段列表
        key_fields = ['涨跌幅', '最新价', '换手率', '量比', '振幅', '市盈率-动态']
        optional_fields = ['60日涨跌幅', '20日涨跌幅', '5日涨跌幅', '年初至今涨跌幅', '成交额', '市净率']

        # 1. 缺失值分析
        missing_report = {}
        for field in key_fields + optional_fields:
            if field in df.columns:
                missing_count = df[field].isna().sum()
                missing_pct = missing_count / total_rows * 100
                if missing_pct > 0:
                    missing_report[field] = {
                        "缺失数量": missing_count,
                        "缺失比例": f"{missing_pct:.1f}%",
                        "严重程度": "严重" if missing_pct > 20 else ("中等" if missing_pct > 5 else "轻微"),
                    }

        pipeline["缺失值报告"] = missing_report

        # 缺失值扣分
        severe_missing = sum(1 for v in missing_report.values() if v["严重程度"] == "严重")
        moderate_missing = sum(1 for v in missing_report.values() if v["严重程度"] == "中等")
        pipeline["质量评分"] -= severe_missing * 15 + moderate_missing * 5

        # 2. 异常值检测（基于涨跌幅的3σ和MAD方法）
        outlier_report = {}
        if '涨跌幅' in df.columns:
            chg = pd.to_numeric(df['涨跌幅'], errors='coerce').dropna()

            if len(chg) > 0:
                # 3σ方法
                mean_chg = chg.mean()
                std_chg = chg.std()
                if std_chg > 0:
                    upper_3s = mean_chg + 3 * std_chg
                    lower_3s = mean_chg - 3 * std_chg
                    outliers_3s = ((chg > upper_3s) | (chg < lower_3s)).sum()
                    outlier_report["3σ异常值"] = {
                        "检测数量": outliers_3s,
                        "占比": f"{outliers_3s/len(chg)*100:.2f}%",
                        "上界": f"{upper_3s:.2f}%",
                        "下界": f"{lower_3s:.2f}%",
                    }

                # MAD方法（更鲁棒）
                median_chg = chg.median()
                mad = (chg - median_chg).abs().median()
                if mad > 0:
                    upper_mad = median_chg + 5 * mad
                    lower_mad = median_chg - 5 * mad
                    outliers_mad = ((chg > upper_mad) | (chg < lower_mad)).sum()
                    outlier_report["MAD异常值"] = {
                        "检测数量": outliers_mad,
                        "占比": f"{outliers_mad/len(chg)*100:.2f}%",
                        "上界": f"{upper_mad:.2f}%",
                        "下界": f"{lower_mad:.2f}%",
                    }

                # 极端值检测（涨跌停附近）
                limit_up = (chg > 9.5).sum()
                limit_down = (chg < -9.5).sum()
                if limit_up > 0 or limit_down > 0:
                    outlier_report["涨跌停检测"] = {
                        "涨停数量": limit_up,
                        "跌停数量": limit_down,
                        "涨停占比": f"{limit_up/len(chg)*100:.2f}%",
                        "跌停占比": f"{limit_down/len(chg)*100:.2f}%",
                    }

        pipeline["异常值报告"] = outlier_report

        # 异常值扣分
        if outlier_report.get("3σ异常值", {}).get("检测数量", 0) > total_rows * 0.05:
            pipeline["质量评分"] -= 10
            pipeline["处理建议"].append("涨跌幅异常值比例较高，可能存在数据错误或极端行情")

        # 3. 数据一致性检查
        consistency = {}
        if '最新价' in df.columns and '涨跌幅' in df.columns:
            price = pd.to_numeric(df['最新价'], errors='coerce')
            # 检查价格为0或负数的记录
            invalid_price = ((price <= 0) | price.isna()).sum()
            if invalid_price > 0:
                consistency["无效价格"] = f"{invalid_price}条记录价格为0或缺失"
                pipeline["质量评分"] -= 10

        if '换手率' in df.columns:
            turnover = pd.to_numeric(df['换手率'], errors='coerce')
            # 换手率异常（>50%或<0）
            abnormal_turnover = ((turnover > 50) | (turnover < 0)).sum()
            if abnormal_turnover > 0:
                consistency["异常换手率"] = f"{abnormal_turnover}条记录换手率异常"
                pipeline["质量评分"] -= 5

        if '市盈率-动态' in df.columns:
            pe = pd.to_numeric(df['市盈率-动态'], errors='coerce')
            # 市盈率为负或极端值
            extreme_pe = ((pe < -1000) | (pe > 10000)).sum()
            if extreme_pe > 0:
                consistency["极端市盈率"] = f"{extreme_pe}条记录市盈率极端"
                pipeline["质量评分"] -= 3

        pipeline["数据一致性"] = consistency

        # 4. 智能填充缺失值
        cleaned_df = df.copy()
        fill_actions = []

        for field in key_fields:
            if field in cleaned_df.columns:
                before = cleaned_df[field].isna().sum()
                if before > 0:
                    numeric_vals = pd.to_numeric(cleaned_df[field], errors='coerce')
                    if field in ('涨跌幅', '振幅'):
                        numeric_vals = numeric_vals.fillna(0)
                    elif field == '量比':
                        numeric_vals = numeric_vals.fillna(1)
                    elif field in ('换手率',):
                        numeric_vals = numeric_vals.fillna(numeric_vals.median() if numeric_vals.notna().any() else 0)
                    elif field in ('市盈率-动态',):
                        numeric_vals = numeric_vals.fillna(numeric_vals.median() if numeric_vals.notna().any() else 0)
                    else:
                        numeric_vals = numeric_vals.fillna(numeric_vals.median() if numeric_vals.notna().any() else 0)
                    cleaned_df[field] = numeric_vals
                    after = cleaned_df[field].isna().sum()
                    if before > after:
                        fill_actions.append(f"{field}: 填充{before - after}条缺失值")

        if fill_actions:
            pipeline["处理建议"].extend(fill_actions)

        pipeline["清洗后数据"] = cleaned_df

        # 5. 质量评级
        score = pipeline["质量评分"]
        if score >= 90:
            pipeline["质量评级"] = "优秀"
        elif score >= 70:
            pipeline["质量评级"] = "良好"
        elif score >= 50:
            pipeline["质量评级"] = "一般"
        else:
            pipeline["质量评级"] = "较差"

        if score < 70:
            pipeline["处理建议"].append(f"数据质量评级为'{pipeline['质量评级']}'，建议检查数据源或等待数据更新")

        pipeline["质量评分"] = max(0, min(100, score))

    except Exception as e:
        pipeline["处理建议"] = [f"数据质量管道异常: {str(e)}"]
        pipeline["质量评分"] = 0

    return pipeline


def _cross_market_analysis(spot_df):
    """
    跨市场联动分析：分析不同市场/板块之间的强弱对比和资金流向
    分析维度：沪市vs深市、大盘vs小盘、板块间相关性、市场风格分化
    返回: {市场强弱对比, 大小盘风格, 板块相关性, 资金流向推断, 分析建议}
    """
    import pandas as pd
    import numpy as np

    if spot_df is None or spot_df.empty:
        return {"市场强弱对比": {}, "大小盘风格": {}, "板块相关性": {}, "资金流向推断": {}, "分析建议": ["数据不足，无法进行跨市场分析"]}

    analysis = {
        "市场强弱对比": {},
        "大小盘风格": {},
        "板块相关性": {},
        "资金流向推断": {},
        "分析建议": [],
    }

    try:
        df = spot_df.copy()

        chg_today = pd.to_numeric(df['涨跌幅'], errors='coerce').fillna(0)
        turnover = pd.to_numeric(df['换手率'], errors='coerce').fillna(0)
        amount = pd.to_numeric(df.get('成交额', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        vol_ratio = pd.to_numeric(df['量比'], errors='coerce').fillna(1)
        chg_60d = pd.to_numeric(df.get('60日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)

        # 1. 沪市 vs 深市 强弱对比
        sh_mask = df['代码'].str.startswith('6')
        sz_main_mask = df['代码'].str.startswith('0')
        sz_gem_mask = df['代码'].str.startswith('3')

        sh_df = df[sh_mask]
        sz_main_df = df[sz_main_mask]
        sz_gem_df = df[sz_gem_mask]

        market_comparison = {}

        for name, sub_df in [("上证主板", sh_df), ("深证主板", sz_main_df), ("创业板", sz_gem_df)]:
            if len(sub_df) > 0:
                avg_chg = sub_df['涨跌幅'].astype(float).mean()
                avg_turnover = sub_df['换手率'].astype(float).mean()
                avg_amount = sub_df.get('成交额', pd.Series(0, index=sub_df.index)).astype(float).mean()
                up_ratio = (sub_df['涨跌幅'].astype(float) > 0).sum() / len(sub_df) * 100
                avg_chg_60d = sub_df.get('60日涨跌幅', pd.Series(0, index=sub_df.index)).astype(float).mean()

                market_comparison[name] = {
                    "股票数量": len(sub_df),
                    "平均涨跌幅": f"{avg_chg:.2f}%",
                    "上涨比例": f"{up_ratio:.1f}%",
                    "平均换手率": f"{avg_turnover:.2f}%",
                    "平均成交额": f"{avg_amount/1e8:.2f}亿",
                    "60日平均涨跌": f"{avg_chg_60d:.2f}%",
                }

        analysis["市场强弱对比"] = market_comparison

        # 强弱判断
        if len(sh_df) > 0 and len(sz_gem_df) > 0:
            sh_avg = sh_df['涨跌幅'].astype(float).mean()
            gem_avg = sz_gem_df['涨跌幅'].astype(float).mean()
            diff = sh_avg - gem_avg

            if diff > 1:
                analysis["分析建议"].append(f"上证主板强于创业板(差值{diff:.2f}%)，市场风格偏向价值/蓝筹，建议关注低估值蓝筹股")
            elif diff < -1:
                analysis["分析建议"].append(f"创业板强于上证主板(差值{abs(diff):.2f}%)，市场风格偏向成长/题材，建议关注高成长标的")
            else:
                analysis["分析建议"].append("上证主板与创业板表现接近，市场风格均衡")

        # 2. 大小盘风格分化
        # 基于成交额划分大小盘
        if len(df) > 0 and amount.sum() > 0:
            amount_median = amount.median()
            large_cap = df[amount > amount_median * 3]
            mid_cap = df[(amount > amount_median) & (amount <= amount_median * 3)]
            small_cap = df[amount <= amount_median]

            style_analysis = {}
            for name, sub_df in [("大盘股(成交额>3倍中位数)", large_cap), ("中盘股", mid_cap), ("小盘股(成交额<=中位数)", small_cap)]:
                if len(sub_df) > 0:
                    avg_chg = sub_df['涨跌幅'].astype(float).mean()
                    up_ratio = (sub_df['涨跌幅'].astype(float) > 0).sum() / len(sub_df) * 100
                    avg_turnover = sub_df['换手率'].astype(float).mean()

                    style_analysis[name] = {
                        "股票数量": len(sub_df),
                        "平均涨跌幅": f"{avg_chg:.2f}%",
                        "上涨比例": f"{up_ratio:.1f}%",
                        "平均换手率": f"{avg_turnover:.2f}%",
                    }

            analysis["大小盘风格"] = style_analysis

            if len(large_cap) > 0 and len(small_cap) > 0:
                large_avg = large_cap['涨跌幅'].astype(float).mean()
                small_avg = small_cap['涨跌幅'].astype(float).mean()
                style_diff = large_avg - small_avg

                if style_diff > 1:
                    analysis["分析建议"].append(f"大盘股强于小盘股(差值{style_diff:.2f}%)，资金偏好蓝筹，小盘股可能承压")
                elif style_diff < -1:
                    analysis["分析建议"].append(f"小盘股强于大盘股(差值{abs(style_diff):.2f}%)，资金偏好题材，市场风险偏好较高")

        # 3. 板块间相关性（基于涨跌幅的近似相关性）
        sector_groups = {}
        sector_keywords = {
            "银行": ["银行"],
            "证券": ["证券", "券商"],
            "医药": ["医药", "药", "生物", "医疗"],
            "半导体": ["半导体", "芯片"],
            "新能源": ["新能源", "锂电", "光伏", "风电"],
            "白酒": ["酒", "茅台"],
            "汽车": ["汽车"],
            "军工": ["军工", "航天"],
            "煤炭": ["煤炭", "煤业"],
            "有色": ["有色", "矿业", "黄金"],
        }

        for _, row in df.iterrows():
            name = str(row.get('名称', ''))
            chg = float(row.get('涨跌幅', 0))
            for sector_name, keywords in sector_keywords.items():
                if any(kw in name for kw in keywords):
                    if sector_name not in sector_groups:
                        sector_groups[sector_name] = []
                    sector_groups[sector_name].append(chg)
                    break

        sector_avg_chg = {}
        for sector, chgs in sector_groups.items():
            if len(chgs) >= 3:
                sector_avg_chg[sector] = sum(chgs) / len(chgs)

        # 计算板块间相关性
        sector_correlations = []
        sector_names = list(sector_avg_chg.keys())
        for i in range(len(sector_names)):
            for j in range(i + 1, len(sector_names)):
                si = sector_names[i]
                sj = sector_names[j]
                # 基于涨跌幅方向的近似相关性
                chg_diff = abs(sector_avg_chg[si] - sector_avg_chg[sj])
                max_chg = max(abs(sector_avg_chg[si]), abs(sector_avg_chg[sj]), 0.01)
                approx_corr = 1 - min(chg_diff / max_chg, 1)
                sector_correlations.append({
                    "板块对": f"{si} vs {sj}",
                    "近似相关性": round(approx_corr, 2),
                    "联动特征": "同向" if sector_avg_chg[si] * sector_avg_chg[sj] > 0 else "背离",
                })

        analysis["板块相关性"] = {
            "板块间关系": sector_correlations[:10] if sector_correlations else [],
            "强势板块": [s for s, v in sorted(sector_avg_chg.items(), key=lambda x: x[1], reverse=True)[:3]] if sector_avg_chg else [],
            "弱势板块": [s for s, v in sorted(sector_avg_chg.items(), key=lambda x: x[1])[:3]] if sector_avg_chg else [],
        }

        # 4. 资金流向推断（基于成交额和量比）
        sh_amount = sh_df.get('成交额', pd.Series(0, index=sh_df.index)).astype(float).sum() if len(sh_df) > 0 else 0
        sz_amount = sz_main_df.get('成交额', pd.Series(0, index=sz_main_df.index)).astype(float).sum() if len(sz_main_df) > 0 else 0
        gem_amount = sz_gem_df.get('成交额', pd.Series(0, index=sz_gem_df.index)).astype(float).sum() if len(sz_gem_df) > 0 else 0

        total_amount = sh_amount + sz_amount + gem_amount
        if total_amount > 0:
            analysis["资金流向推断"] = {
                "上证主板成交占比": f"{sh_amount/total_amount*100:.1f}%",
                "深证主板成交占比": f"{sz_amount/total_amount*100:.1f}%",
                "创业板成交占比": f"{gem_amount/total_amount*100:.1f}%",
                "资金偏好": "沪市" if sh_amount > sz_amount + gem_amount else ("深市" if sz_amount + gem_amount > sh_amount * 1.5 else "均衡"),
            }

            if gem_amount / total_amount > 0.3:
                analysis["分析建议"].append("创业板成交占比较高，资金活跃于成长股，市场风险偏好较高")
            elif sh_amount / total_amount > 0.5:
                analysis["分析建议"].append("上证主板成交占比较高，资金偏好蓝筹，市场风格偏防御")

        # 5. 综合建议
        if analysis["板块相关性"].get("强势板块"):
            strong = analysis["板块相关性"]["强势板块"]
            analysis["分析建议"].append(f"当前强势板块: {', '.join(strong)}，建议关注这些板块中的优质标的")

        if analysis["板块相关性"].get("弱势板块"):
            weak = analysis["板块相关性"]["弱势板块"]
            analysis["分析建议"].append(f"当前弱势板块: {', '.join(weak)}，可关注是否存在超跌反弹机会")

    except Exception as e:
        analysis["分析建议"] = [f"跨市场分析异常: {str(e)}"]

    return analysis


def _event_driven_analysis(spot_df):
    """
    事件驱动分析框架：基于量价异常推断可能的事件驱动信号
    检测维度：异常收益事件、信息冲击事件、板块联动事件、连续异动事件
    注意：本分析基于量价数据推断，不等同于真实事件确认，需结合公告信息验证
    返回: {异常收益事件, 信息冲击事件, 板块联动事件, 连续异动事件, 事件汇总, 分析建议}
    """
    import pandas as pd
    import numpy as np

    if spot_df is None or spot_df.empty:
        return {"异常收益事件": [], "信息冲击事件": [], "板块联动事件": [], "连续异动事件": [], "事件汇总": {}, "分析建议": ["数据不足，无法进行事件驱动分析"]}

    analysis = {
        "异常收益事件": [],
        "信息冲击事件": [],
        "板块联动事件": [],
        "连续异动事件": [],
        "事件汇总": {},
        "分析建议": [],
    }

    try:
        df = spot_df.copy()

        chg_today = pd.to_numeric(df['涨跌幅'], errors='coerce').fillna(0)
        turnover = pd.to_numeric(df['换手率'], errors='coerce').fillna(0)
        vol_ratio = pd.to_numeric(df['量比'], errors='coerce').fillna(1)
        amplitude = pd.to_numeric(df['振幅'], errors='coerce').fillna(0)
        amount = pd.to_numeric(df.get('成交额', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        chg_5d = pd.to_numeric(df.get('5日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)

        total = len(df)

        # 1. 异常收益事件（涨跌幅超过3倍标准差）
        chg_mean = chg_today.mean()
        chg_std = chg_today.std()
        if chg_std > 0:
            upper_3sigma = chg_mean + 3 * chg_std
            lower_3sigma = chg_mean - 3 * chg_std

            extreme_up = df[(chg_today > upper_3sigma) & (chg_today > 5)]
            extreme_down = df[(chg_today < lower_3sigma) & (chg_today < -5)]

            for _, row in extreme_up.head(10).iterrows():
                analysis["异常收益事件"].append({
                    "代码": str(row.get('代码', '')),
                    "名称": str(row.get('名称', '')),
                    "涨跌幅": f"{float(row['涨跌幅']):.2f}%",
                    "事件类型": "异常大涨",
                    "可能原因": "重大利好公告/业绩超预期/行业政策利好/资金炒作",
                })

            for _, row in extreme_down.head(10).iterrows():
                analysis["异常收益事件"].append({
                    "代码": str(row.get('代码', '')),
                    "名称": str(row.get('名称', '')),
                    "涨跌幅": f"{float(row['涨跌幅']):.2f}%",
                    "事件类型": "异常大跌",
                    "可能原因": "重大利空公告/业绩不及预期/股东减持/行业政策利空",
                })

        # 2. 信息冲击事件（异常放量+大幅涨跌）
        info_impact = df[
            (vol_ratio > 2.5) &
            ((chg_today > 5) | (chg_today < -5)) &
            (turnover > 3)
        ].sort_values('涨跌幅', key=lambda x: abs(x), ascending=False)

        for _, row in info_impact.head(10).iterrows():
            direction = "利好冲击" if float(row['涨跌幅']) > 0 else "利空冲击"
            analysis["信息冲击事件"].append({
                "代码": str(row.get('代码', '')),
                "名称": str(row.get('名称', '')),
                "涨跌幅": f"{float(row['涨跌幅']):.2f}%",
                "量比": f"{float(row['量比']):.1f}",
                "换手率": f"{float(row['换手率']):.2f}%",
                "事件类型": direction,
                "信息强度": "强" if float(row['量比']) > 4 else "中",
                "可能原因": f"重大信息驱动{direction}，量比{float(row['量比']):.1f}表明市场反应剧烈",
            })

        # 3. 板块联动事件（同行业多只股票同时异动）
        # 基于股票名称或代码前缀推断行业
        sector_groups = {}
        for _, row in df.iterrows():
            code = str(row.get('代码', ''))
            name = str(row.get('名称', ''))
            chg = float(row.get('涨跌幅', 0))
            vol_r = float(row.get('量比', 1))

            # 简单行业推断：基于名称关键词
            sector = "其他"
            sector_keywords = {
                "银行": ["银行"],
                "证券": ["证券", "券商"],
                "保险": ["保险"],
                "房地产": ["地产", "房产", "万科", "保利"],
                "医药": ["医药", "药", "生物", "医疗"],
                "半导体": ["半导体", "芯片", "微电子"],
                "新能源": ["新能源", "锂电", "光伏", "风电", "储能"],
                "白酒": ["酒", "茅台", "五粮液"],
                "汽车": ["汽车", "比亚迪", "长城", "吉利"],
                "军工": ["军工", "航天", "航空", "兵器"],
                "煤炭": ["煤炭", "煤业", "神华"],
                "有色": ["有色", "矿业", "黄金", "铜", "铝"],
                "电力": ["电力", "能源", "核电"],
            }

            for sector_name, keywords in sector_keywords.items():
                if any(kw in name for kw in keywords):
                    sector = sector_name
                    break

            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append({
                "代码": code,
                "名称": name,
                "涨跌幅": chg,
                "量比": vol_r,
            })

        # 检测板块联动：同一行业多只股票同向大幅波动
        for sector, stocks in sector_groups.items():
            if len(stocks) < 3:
                continue
            up_stocks = [s for s in stocks if s["涨跌幅"] > 3]
            down_stocks = [s for s in stocks if s["涨跌幅"] < -3]

            if len(up_stocks) >= 3:
                avg_chg = sum(s["涨跌幅"] for s in up_stocks) / len(up_stocks)
                analysis["板块联动事件"].append({
                    "板块": sector,
                    "联动方向": "集体上涨",
                    "涉及股票数": len(up_stocks),
                    "平均涨幅": f"{avg_chg:.2f}%",
                    "代表股票": ", ".join(s["名称"] for s in up_stocks[:3]),
                    "可能原因": f"{sector}板块集体走强，可能存在行业政策利好或资金集中流入",
                })
            elif len(down_stocks) >= 3:
                avg_chg = sum(s["涨跌幅"] for s in down_stocks) / len(down_stocks)
                analysis["板块联动事件"].append({
                    "板块": sector,
                    "联动方向": "集体下跌",
                    "涉及股票数": len(down_stocks),
                    "平均跌幅": f"{avg_chg:.2f}%",
                    "代表股票": ", ".join(s["名称"] for s in down_stocks[:3]),
                    "可能原因": f"{sector}板块集体走弱，可能存在行业政策利空或资金集中流出",
                })

        # 4. 连续异动事件（5日涨幅极端+今日继续放量）
        continuous = df[
            ((chg_5d > 20) | (chg_5d < -20)) &
            (vol_ratio > 1.5)
        ].sort_values('5日涨跌幅', key=lambda x: abs(x), ascending=False)

        for _, row in continuous.head(10).iterrows():
            direction_5d = "连续上涨" if float(row.get('5日涨跌幅', 0)) > 0 else "连续下跌"
            analysis["连续异动事件"].append({
                "代码": str(row.get('代码', '')),
                "名称": str(row.get('名称', '')),
                "5日涨跌幅": f"{float(row.get('5日涨跌幅', 0)):.2f}%",
                "今日涨跌幅": f"{float(row['涨跌幅']):.2f}%",
                "量比": f"{float(row['量比']):.1f}",
                "事件类型": direction_5d,
                "可能原因": f"连续{direction_5d}且持续放量，可能存在持续性事件驱动，需关注后续公告",
            })

        # 5. 事件汇总
        total_events = (
            len(analysis["异常收益事件"]) +
            len(analysis["信息冲击事件"]) +
            len(analysis["板块联动事件"]) +
            len(analysis["连续异动事件"])
        )

        high_impact_count = len([e for e in analysis["信息冲击事件"] if e.get("信息强度") == "强"])

        analysis["事件汇总"] = {
            "总事件数": total_events,
            "异常收益事件": len(analysis["异常收益事件"]),
            "信息冲击事件": len(analysis["信息冲击事件"]),
            "板块联动事件": len(analysis["板块联动事件"]),
            "连续异动事件": len(analysis["连续异动事件"]),
            "高强度信息冲击": high_impact_count,
            "事件活跃度": "高" if total_events > 15 else ("中" if total_events > 5 else "低"),
        }

        # 6. 分析建议
        if high_impact_count > 3:
            analysis["分析建议"].append(f"检测到{high_impact_count}个高强度信息冲击事件，市场信息流动活跃，建议关注相关公告")
        if len(analysis["板块联动事件"]) > 2:
            analysis["分析建议"].append("多个板块出现联动效应，市场呈现结构性行情，建议关注强势板块")
        if len(analysis["连续异动事件"]) > 5:
            analysis["分析建议"].append("连续异动股票较多，可能存在持续性事件驱动机会，建议深入分析异动原因")
        if total_events == 0:
            analysis["分析建议"].append("未检测到明显事件驱动信号，市场处于信息平静期")

        analysis["分析建议"].append("以上事件信号基于量价异常推断，仅供参考，实际投资决策需结合公司公告和基本面信息验证")

    except Exception as e:
        analysis["分析建议"] = [f"事件驱动分析异常: {str(e)}"]

    return analysis


def _behavioral_finance_factors(spot_df):
    """
    行为金融学因子分析：基于日线数据构建行为金融指标
    分析维度：反转效应、锚定效应、羊群效应、恐慌指数、过度自信
    返回: {行为因子汇总, 反转效应, 锚定效应, 羊群效应, 恐慌指数, 分析建议}
    """
    import pandas as pd
    import numpy as np

    if spot_df is None or spot_df.empty:
        return {"行为因子汇总": {}, "反转效应": {}, "锚定效应": {}, "羊群效应": {}, "恐慌指数": {}, "分析建议": ["数据不足，无法进行行为金融分析"]}

    analysis = {
        "行为因子汇总": {},
        "反转效应": {},
        "锚定效应": {},
        "羊群效应": {},
        "恐慌指数": {},
        "分析建议": [],
    }

    try:
        df = spot_df.copy()

        chg_today = pd.to_numeric(df['涨跌幅'], errors='coerce').fillna(0)
        chg_5d = pd.to_numeric(df.get('5日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        chg_20d = pd.to_numeric(df.get('20日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        chg_60d = pd.to_numeric(df.get('60日涨跌幅', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        turnover = pd.to_numeric(df['换手率'], errors='coerce').fillna(0)
        amplitude = pd.to_numeric(df['振幅'], errors='coerce').fillna(0)
        vol_ratio = pd.to_numeric(df['量比'], errors='coerce').fillna(1)

        total = len(df)

        # 1. 反转效应（过度反应检测）
        # 短期大涨后回调概率高（过度反应），短期大跌后反弹概率高
        extreme_up_5d = (chg_5d > 15).sum()
        extreme_down_5d = (chg_5d < -15).sum()
        extreme_up_20d = (chg_20d > 30).sum()
        extreme_down_20d = (chg_20d < -30).sum()

        reversal_signals = []
        if extreme_up_5d > total * 0.05:
            reversal_signals.append(f"{extreme_up_5d}只股票5日涨幅超15%，存在过度反应后的回调风险")
        if extreme_down_5d > total * 0.05:
            reversal_signals.append(f"{extreme_down_5d}只股票5日跌幅超15%，存在超跌反弹机会")
        if extreme_up_20d > total * 0.05:
            reversal_signals.append(f"{extreme_up_20d}只股票20日涨幅超30%，中期过度上涨风险较高")

        analysis["反转效应"] = {
            "5日极端上涨(>15%)": f"{extreme_up_5d}只 ({extreme_up_5d/total*100:.1f}%)",
            "5日极端下跌(<-15%)": f"{extreme_down_5d}只 ({extreme_down_5d/total*100:.1f}%)",
            "20日极端上涨(>30%)": f"{extreme_up_20d}只 ({extreme_up_20d/total*100:.1f}%)",
            "20日极端下跌(<-30%)": f"{extreme_down_20d}只 ({extreme_down_20d/total*100:.1f}%)",
            "反转信号": reversal_signals if reversal_signals else ["未检测到明显反转信号"],
        }

        # 2. 锚定效应（52周高低点锚定）
        # 基于60日涨跌幅推断相对位置
        # 60日涨幅极高=接近高点锚定，60日跌幅极大=接近低点锚定
        near_high = (chg_60d > 40).sum()
        near_low = (chg_60d < -40).sum()
        mid_range = ((chg_60d > -20) & (chg_60d < 20)).sum()

        anchor_signals = []
        if near_high > total * 0.03:
            anchor_signals.append(f"{near_high}只股票60日涨幅超40%，可能接近52周高点，存在锚定效应下的获利了结压力")
        if near_low > total * 0.03:
            anchor_signals.append(f"{near_low}只股票60日跌幅超40%，可能接近52周低点，存在锚定效应下的抄底心理支撑")

        analysis["锚定效应"] = {
            "接近高点(60日涨>40%)": f"{near_high}只 ({near_high/total*100:.1f}%)",
            "接近低点(60日跌>40%)": f"{near_low}只 ({near_low/total*100:.1f}%)",
            "中间区域(-20%~20%)": f"{mid_range}只 ({mid_range/total*100:.1f}%)",
            "锚定信号": anchor_signals if anchor_signals else ["未检测到明显锚定效应信号"],
        }

        # 3. 羊群效应
        # 涨跌幅集中度：如果大部分股票同涨同跌，说明羊群效应强
        up_count = (chg_today > 0).sum()
        down_count = (chg_today < 0).sum()
        flat_count = (chg_today == 0).sum()

        if total > 0:
            up_ratio = up_count / total
            down_ratio = down_count / total
            concentration = max(up_ratio, down_ratio)

            # 涨跌幅离散度
            chg_std = chg_today[chg_today != 0].std() if len(chg_today[chg_today != 0]) > 1 else 0

            herd_level = "强" if concentration > 0.7 else ("中" if concentration > 0.55 else "弱")
            herd_signals = []
            if concentration > 0.7:
                herd_signals.append(f"涨跌集中度{concentration*100:.0f}%，羊群效应强，市场情绪主导，个股分化小")
            elif concentration < 0.55:
                herd_signals.append(f"涨跌集中度{concentration*100:.0f}%，羊群效应弱，个股分化大，选股能力更重要")

            analysis["羊群效应"] = {
                "上涨家数": f"{up_count}只 ({up_ratio*100:.1f}%)",
                "下跌家数": f"{down_count}只 ({down_ratio*100:.1f}%)",
                "平盘家数": f"{flat_count}只 ({flat_count/total*100:.1f}%)",
                "涨跌集中度": f"{concentration*100:.1f}%",
                "羊群效应等级": herd_level,
                "涨跌幅标准差": f"{chg_std:.2f}%",
                "羊群信号": herd_signals,
            }

        # 4. 恐慌指数（基于跌幅和振幅）
        panic_threshold = -5
        high_amp_threshold = 8

        panic_stocks = ((chg_today < panic_threshold) & (amplitude > high_amp_threshold)).sum()
        panic_ratio = panic_stocks / total * 100 if total > 0 else 0

        # 恐慌指数：放量下跌+高振幅的股票占比
        panic_volume = ((chg_today < -3) & (vol_ratio > 1.5)).sum()

        panic_level = "高" if panic_ratio > 10 else ("中" if panic_ratio > 5 else "低")
        panic_signals = []
        if panic_ratio > 10:
            panic_signals.append(f"恐慌指数高({panic_ratio:.1f}%)，市场恐慌情绪蔓延，可能出现非理性抛售")
        elif panic_ratio > 5:
            panic_signals.append(f"恐慌指数中等({panic_ratio:.1f}%)，局部恐慌，需关注个股风险")
        else:
            panic_signals.append(f"恐慌指数低({panic_ratio:.1f}%)，市场情绪稳定")

        analysis["恐慌指数"] = {
            "恐慌股票数(跌>5%且振幅>8%)": f"{panic_stocks}只",
            "恐慌占比": f"{panic_ratio:.1f}%",
            "放量下跌股票数(跌>3%且量比>1.5)": f"{panic_volume}只",
            "恐慌等级": panic_level,
            "恐慌信号": panic_signals,
        }

        # 5. 行为因子汇总
        summary_signals = []
        if concentration > 0.7:
            summary_signals.append("羊群效应强：市场同涨同跌，趋势策略有效性降低，反转策略可能更有效")
        if panic_ratio > 10:
            summary_signals.append("恐慌情绪高：行为金融学表明此时往往是中长期布局良机（逆向投资）")
        if extreme_up_5d > total * 0.05:
            summary_signals.append("过度反应明显：短期追高风险大，建议等待回调后再介入")

        analysis["行为因子汇总"] = {
            "核心信号": summary_signals if summary_signals else ["行为金融因子未检测到极端信号，市场行为较为理性"],
            "策略启示": [],
        }

        # 策略启示
        if concentration > 0.7 and panic_ratio > 5:
            analysis["行为因子汇总"]["策略启示"].append("高羊群+高恐慌=市场非理性，逆向投资策略可能获得超额收益")
        elif concentration < 0.55 and panic_ratio < 5:
            analysis["行为因子汇总"]["策略启示"].append("低羊群+低恐慌=市场理性，基本面选股策略有效性较高")
        elif concentration > 0.7 and panic_ratio < 5:
            analysis["行为因子汇总"]["策略启示"].append("高羊群+低恐慌=趋势市，动量策略可能更有效")

        # 综合建议
        if panic_ratio > 10:
            analysis["分析建议"].append("市场恐慌情绪较高，行为金融学建议保持冷静，避免恐慌性抛售，可关注被错杀的优质标的")
        if extreme_up_5d > total * 0.08:
            analysis["分析建议"].append("短期过度上涨股票较多，存在回调风险，建议不要追高，等待回调机会")
        if extreme_down_5d > total * 0.08:
            analysis["分析建议"].append("短期过度下跌股票较多，存在超跌反弹机会，可关注基本面良好的超跌标的")

    except Exception as e:
        analysis["分析建议"] = [f"行为金融分析异常: {str(e)}"]

    return analysis


def _microstructure_analysis(spot_df):
    """
    市场微观结构分析：基于日线数据估算流动性、价格冲击和波动特征
    分析维度：流动性深度、价格冲击成本、波动率聚类、交易活跃度模式
    返回: {流动性分析, 价格冲击估算, 波动特征, 微观结构风险, 分析建议}
    """
    import pandas as pd
    import numpy as np

    if spot_df is None or spot_df.empty:
        return {"流动性分析": {}, "价格冲击估算": {}, "波动特征": {}, "微观结构风险": {}, "分析建议": ["数据不足，无法进行微观结构分析"]}

    analysis = {
        "流动性分析": {},
        "价格冲击估算": {},
        "波动特征": {},
        "微观结构风险": {},
        "分析建议": [],
    }

    try:
        df = spot_df.copy()

        # 提取关键字段
        turnover = pd.to_numeric(df['换手率'], errors='coerce').fillna(0)
        volume = pd.to_numeric(df.get('成交量', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        amount = pd.to_numeric(df.get('成交额', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        amplitude = pd.to_numeric(df['振幅'], errors='coerce').fillna(0)
        price = pd.to_numeric(df['最新价'], errors='coerce').fillna(0)
        chg = pd.to_numeric(df['涨跌幅'], errors='coerce').fillna(0)
        vol_ratio = pd.to_numeric(df['量比'], errors='coerce').fillna(1)

        valid = price > 0

        # 1. 流动性分析
        # 换手率分层
        high_liq = (turnover > 5).sum()
        mid_liq = ((turnover > 1) & (turnover <= 5)).sum()
        low_liq = ((turnover > 0.1) & (turnover <= 1)).sum()
        frozen = (turnover <= 0.1).sum()

        total = max(high_liq + mid_liq + low_liq + frozen, 1)
        analysis["流动性分析"] = {
            "高流动性(换手>5%)": f"{high_liq}只 ({high_liq/total*100:.1f}%)",
            "中等流动性(1-5%)": f"{mid_liq}只 ({mid_liq/total*100:.1f}%)",
            "低流动性(0.1-1%)": f"{low_liq}只 ({low_liq/total*100:.1f}%)",
            "冻结状态(<0.1%)": f"{frozen}只 ({frozen/total*100:.1f}%)",
            "市场平均换手率": f"{turnover[valid].mean():.2f}%",
            "换手率中位数": f"{turnover[valid].median():.2f}%",
        }

        # 2. 价格冲击估算（基于Amihud非流动性指标）
        # Amihud = |return| / dollar_volume
        if (amount[valid] > 0).sum() > 0:
            amihud = np.abs(chg[valid]) / (amount[valid] / 1e8).clip(lower=0.01)
            amihud_clean = amihud[~np.isinf(amihud) & (amihud < 100)]
            if len(amihud_clean) > 0:
                avg_impact = amihud_clean.mean()
                analysis["价格冲击估算"] = {
                    "Amihud非流动性指标": round(avg_impact, 4),
                    "指标含义": "每亿元成交额引起的价格变化百分比，值越大流动性越差",
                    "冲击成本等级": "低" if avg_impact < 0.01 else ("中" if avg_impact < 0.05 else "高"),
                    "估算滑点(100万)": f"{avg_impact * 0.01:.4f}%",
                    "估算滑点(500万)": f"{avg_impact * 0.05:.4f}%",
                }
                if avg_impact > 0.05:
                    analysis["分析建议"].append("市场整体流动性偏弱，大额交易可能产生显著价格冲击，建议控制单笔交易规模")
            else:
                analysis["价格冲击估算"] = {"状态": "数据不足，无法估算"}
        else:
            analysis["价格冲击估算"] = {"状态": "缺少成交额数据"}

        # 3. 波动特征分析
        amp_valid = amplitude[valid]
        if len(amp_valid) > 0:
            amp_mean = amp_valid.mean()
            amp_std = amp_valid.std()
            amp_skew = amp_valid.skew() if len(amp_valid) > 2 else 0

            # 波动率聚类检测：高波动股票占比
            high_vol = (amp_valid > amp_mean + amp_std).sum()
            low_vol = (amp_valid < amp_mean - amp_std).sum()

            analysis["波动特征"] = {
                "平均振幅": f"{amp_mean:.2f}%",
                "振幅标准差": f"{amp_std:.2f}%",
                "振幅偏度": round(amp_skew, 2),
                "高波动股票占比": f"{high_vol/len(amp_valid)*100:.1f}%",
                "低波动股票占比": f"{low_vol/len(amp_valid)*100:.1f}%",
                "波动特征": "右偏(极端上涨多)" if amp_skew > 0.5 else ("左偏(极端下跌多)" if amp_skew < -0.5 else "对称"),
            }

            if amp_skew > 1:
                analysis["分析建议"].append("振幅分布严重右偏，存在极端上涨个股，追高风险较大")
            elif amp_skew < -1:
                analysis["分析建议"].append("振幅分布严重左偏，存在极端下跌个股，市场恐慌情绪较重")

        # 4. 微观结构风险
        risk_signals = []

        # 量价背离检测
        vol_chg_corr = np.corrcoef(volume[valid], chg[valid])[0, 1] if len(volume[valid]) > 2 else 0
        if not np.isnan(vol_chg_corr):
            if vol_chg_corr < -0.3:
                risk_signals.append(f"量价负相关({vol_chg_corr:.2f})，放量下跌特征明显，市场信心不足")
            elif vol_chg_corr > 0.5:
                risk_signals.append(f"量价强正相关({vol_chg_corr:.2f})，放量上涨特征明显，市场情绪积极")

        # 异常放量检测
        abnormal_vol = (vol_ratio > 2).sum()
        if abnormal_vol > total * 0.1:
            risk_signals.append(f"{abnormal_vol}只股票异常放量(量比>2)，占比{abnormal_vol/total*100:.1f}%，可能存在信息事件或操纵行为")

        # 极端振幅检测
        extreme_amp = (amplitude[valid] > 10).sum()
        if extreme_amp > 0:
            risk_signals.append(f"{extreme_amp}只股票振幅超10%，存在极端波动风险")

        analysis["微观结构风险"] = {
            "风险信号": risk_signals if risk_signals else ["未检测到明显微观结构风险信号"],
            "量价相关系数": round(vol_chg_corr, 3) if not np.isnan(vol_chg_corr) else "N/A",
            "异常放量股票数": int(abnormal_vol),
            "极端波动股票数": int(extreme_amp),
        }

        # 5. 综合建议
        if high_liq / total > 0.5:
            analysis["分析建议"].append("市场整体流动性充裕，适合短线交易策略")
        elif low_liq / total > 0.5:
            analysis["分析建议"].append("市场整体流动性不足，建议降低交易频率，优先选择高流动性标的")

        if analysis["价格冲击估算"].get("冲击成本等级") == "高":
            analysis["分析建议"].append("价格冲击成本较高，大资金操作需采用算法交易（TWAP/VWAP）降低冲击")

    except Exception as e:
        analysis["分析建议"] = [f"微观结构分析异常: {str(e)}"]

    return analysis


def _portfolio_optimization(strategies):
    """
    策略组合优化：基于风险平价的多策略权重分配
    输入多个策略的回测指标，输出最优资金分配方案
    返回: {优化权重, 组合指标, 相关性分析, 优化建议}
    """
    import math

    if not strategies or len(strategies) < 2:
        return {"优化权重": {}, "组合指标": {}, "相关性分析": {}, "优化建议": ["至少需要2个策略才能进行组合优化"]}

    analysis = {
        "优化权重": {},
        "组合指标": {},
        "相关性分析": {},
        "优化建议": [],
    }

    try:
        # 提取各策略的风险指标
        strategy_data = []
        for i, s in enumerate(strategies):
            metrics = s.get("metrics", s)
            name = s.get("name", f"策略{i+1}")

            sharpe = float(metrics.get("sharpe_ratio", metrics.get("夏普比率", 0)))
            max_dd = float(metrics.get("max_drawdown", metrics.get("最大回撤", 0)))
            annual_return = float(metrics.get("annual_return", metrics.get("年化收益率", 0)))
            volatility = float(metrics.get("volatility", metrics.get("年化波动率", 0)))
            win_rate = float(metrics.get("win_rate", metrics.get("胜率", 0)))

            # 如果波动率为0，用最大回撤估算
            if volatility <= 0 and max_dd < 0:
                volatility = abs(max_dd) * 1.5

            strategy_data.append({
                "name": name,
                "sharpe": sharpe,
                "max_dd": abs(max_dd),
                "annual_return": annual_return,
                "volatility": volatility if volatility > 0 else 10,
                "win_rate": win_rate,
            })

        # 风险平价权重：每个策略的权重与其风险贡献成反比
        total_inv_vol = sum(1.0 / max(s["volatility"], 0.1) for s in strategy_data)
        risk_parity_weights = {}
        for s in strategy_data:
            inv_vol = 1.0 / max(s["volatility"], 0.1)
            risk_parity_weights[s["name"]] = round(inv_vol / total_inv_vol * 100)

        # 归一化确保总和为100
        total_rp = sum(risk_parity_weights.values())
        if total_rp != 100:
            diff = 100 - total_rp
            max_key = max(risk_parity_weights, key=risk_parity_weights.get)
            risk_parity_weights[max_key] += diff

        # Sharpe加权：夏普比率越高权重越大
        total_sharpe = sum(max(s["sharpe"], 0.01) for s in strategy_data)
        sharpe_weights = {}
        for s in strategy_data:
            sharpe_weights[s["name"]] = round(max(s["sharpe"], 0.01) / total_sharpe * 100)

        total_sw = sum(sharpe_weights.values())
        if total_sw != 100:
            diff = 100 - total_sw
            max_key = max(sharpe_weights, key=sharpe_weights.get)
            sharpe_weights[max_key] += diff

        # 综合权重：风险平价50% + Sharpe加权50%
        combined_weights = {}
        for s in strategy_data:
            combined_weights[s["name"]] = round(
                risk_parity_weights[s["name"]] * 0.5 + sharpe_weights[s["name"]] * 0.5
            )

        total_cw = sum(combined_weights.values())
        if total_cw != 100:
            diff = 100 - total_cw
            max_key = max(combined_weights, key=combined_weights.get)
            combined_weights[max_key] += diff

        analysis["优化权重"] = {
            "风险平价权重": risk_parity_weights,
            "Sharpe加权权重": sharpe_weights,
            "综合推荐权重": combined_weights,
        }

        # 计算组合指标
        combined_return = sum(
            combined_weights[s["name"]] / 100 * s["annual_return"]
            for s in strategy_data
        )
        combined_vol = math.sqrt(sum(
            (combined_weights[s["name"]] / 100) ** 2 * s["volatility"] ** 2
            for s in strategy_data
        ))
        combined_sharpe = combined_return / combined_vol if combined_vol > 0 else 0
        combined_max_dd = sum(
            combined_weights[s["name"]] / 100 * s["max_dd"]
            for s in strategy_data
        )

        analysis["组合指标"] = {
            "组合年化收益": f"{combined_return:.2f}%",
            "组合年化波动": f"{combined_vol:.2f}%",
            "组合夏普比率": round(combined_sharpe, 2),
            "组合最大回撤": f"{combined_max_dd:.2f}%",
            "策略数量": len(strategies),
        }

        # 相关性分析（基于夏普和回撤的相似度）
        if len(strategy_data) >= 2:
            correlations = []
            for i in range(len(strategy_data)):
                for j in range(i + 1, len(strategy_data)):
                    si = strategy_data[i]
                    sj = strategy_data[j]
                    # 基于风险特征的近似相关性
                    sharpe_sim = 1 - abs(si["sharpe"] - sj["sharpe"]) / max(abs(si["sharpe"]) + abs(sj["sharpe"]), 0.01)
                    dd_sim = 1 - abs(si["max_dd"] - sj["max_dd"]) / max(si["max_dd"] + sj["max_dd"], 0.01)
                    approx_corr = (sharpe_sim + dd_sim) / 2
                    correlations.append({
                        "策略对": f"{si['name']} vs {sj['name']}",
                        "近似相关性": round(approx_corr, 2),
                        "分散化效果": "好" if approx_corr < 0.5 else ("一般" if approx_corr < 0.8 else "差"),
                    })

            analysis["相关性分析"] = {
                "策略间相关性": correlations,
                "平均相关性": round(sum(c["近似相关性"] for c in correlations) / len(correlations), 2) if correlations else 0,
            }

            avg_corr = analysis["相关性分析"]["平均相关性"]
            if avg_corr < 0.3:
                analysis["优化建议"].append("策略间相关性低，组合分散化效果优秀，能有效降低整体风险")
            elif avg_corr < 0.6:
                analysis["优化建议"].append("策略间存在一定相关性，组合有一定分散化效果，建议增加低相关策略")
            else:
                analysis["优化建议"].append("策略间相关性较高，组合分散化效果有限，建议引入不同风格的策略（如趋势+反转+套利）")

        # 权重集中度检查
        max_weight = max(combined_weights.values())
        if max_weight > 50:
            max_name = max(combined_weights, key=combined_weights.get)
            analysis["优化建议"].append(f"组合权重过于集中在'{max_name}'({max_weight}%)，建议适当分散以降低单一策略风险")

        # 夏普比率检查
        low_sharpe = [s["name"] for s in strategy_data if s["sharpe"] < 0.3]
        if low_sharpe:
            analysis["优化建议"].append(f"以下策略夏普比率偏低：{', '.join(low_sharpe)}，建议评估是否保留在组合中")

    except Exception as e:
        analysis["优化建议"] = [f"组合优化分析异常: {str(e)}"]

    return analysis


def _signal_tracking(candidates, spot_df):
    """
    推荐信号回溯验证与因子暴露分析
    注意：本分析基于历史收益数据，存在前视偏差，不等同于前向信号跟踪
    实际信号准确率需通过多次推荐记录的时间序列跟踪来验证
    返回: {因子暴露验证, 风格一致性检测, 回溯表现, 前视偏差警告, 跟踪建议}
    """
    import pandas as pd
    import numpy as np

    if not candidates or spot_df is None or spot_df.empty:
        return {"因子暴露验证": {}, "风格一致性检测": {}, "回溯表现": {}, "前视偏差警告": "无推荐数据", "跟踪建议": ["无推荐数据，无法进行信号分析"]}

    analysis = {
        "因子暴露验证": {},
        "风格一致性检测": {},
        "回溯表现": {},
        "前视偏差警告": "",
        "跟踪建议": [],
    }

    try:
        # 前视偏差警告
        analysis["前视偏差警告"] = "以下分析基于历史收益数据（5日/20日/60日涨跌幅），反映的是推荐股票过去的表现，不等同于推荐后的未来收益。实际信号准确率需要通过多次推荐记录的时间序列跟踪来验证。建议建立推荐日志，定期统计推荐后N日的实际收益。"

        # 构建代码到行数据的映射
        code_map = {}
        for _, row in spot_df.iterrows():
            code_map[str(row.get('代码', ''))] = row

        # 因子暴露验证：检查推荐股票在各因子维度上的暴露是否合理
        factor_exposures = {}
        all_pe = []
        all_pb = []
        all_turnover = []
        all_amplitude = []
        all_chg_60d = []

        for _, row in spot_df.iterrows():
            try:
                pe = float(row.get('市盈率-动态', 0))
                if pe > 0 and not np.isnan(pe):
                    all_pe.append(pe)
                pb = float(row.get('市净率', 0))
                if pb > 0 and not np.isnan(pb):
                    all_pb.append(pb)
                to = float(row.get('换手率', 0))
                if not np.isnan(to):
                    all_turnover.append(to)
                amp = float(row.get('振幅', 0))
                if not np.isnan(amp):
                    all_amplitude.append(amp)
                chg = float(row.get('60日涨跌幅', 0))
                if not np.isnan(chg):
                    all_chg_60d.append(chg)
            except (ValueError, TypeError):
                pass

        # 计算全市场分位数
        def calc_percentile(values, target):
            if not values:
                return 50
            return sum(1 for v in values if v <= target) / len(values) * 100

        rec_pe = []
        rec_pb = []
        rec_turnover = []
        rec_amplitude = []
        rec_chg_60d = []

        for c in candidates:
            code = str(c.get("代码", ""))
            row = code_map.get(code)
            if row is None:
                continue
            try:
                pe = float(row.get('市盈率-动态', 0))
                if pe > 0 and not np.isnan(pe):
                    rec_pe.append(pe)
                pb = float(row.get('市净率', 0))
                if pb > 0 and not np.isnan(pb):
                    rec_pb.append(pb)
                to = float(row.get('换手率', 0))
                if not np.isnan(to):
                    rec_turnover.append(to)
                amp = float(row.get('振幅', 0))
                if not np.isnan(amp):
                    rec_amplitude.append(amp)
                chg = float(row.get('60日涨跌幅', 0))
                if not np.isnan(chg):
                    rec_chg_60d.append(chg)
            except (ValueError, TypeError):
                pass

        # 因子暴露分析
        if rec_pe and all_pe:
            avg_rec_pe = sum(rec_pe) / len(rec_pe)
            avg_mkt_pe = sum(all_pe) / len(all_pe)
            pct_pe = calc_percentile(all_pe, avg_rec_pe)
            factor_exposures["市盈率"] = {
                "推荐均值": round(avg_rec_pe, 1),
                "市场均值": round(avg_mkt_pe, 1),
                "全市场分位": f"{pct_pe:.0f}%",
                "暴露方向": "偏低估值" if pct_pe < 40 else ("偏高估值" if pct_pe > 60 else "中性"),
            }

        if rec_pb and all_pb:
            avg_rec_pb = sum(rec_pb) / len(rec_pb)
            avg_mkt_pb = sum(all_pb) / len(all_pb)
            pct_pb = calc_percentile(all_pb, avg_rec_pb)
            factor_exposures["市净率"] = {
                "推荐均值": round(avg_rec_pb, 2),
                "市场均值": round(avg_mkt_pb, 2),
                "全市场分位": f"{pct_pb:.0f}%",
                "暴露方向": "偏低估值" if pct_pb < 40 else ("偏高估值" if pct_pb > 60 else "中性"),
            }

        if rec_turnover and all_turnover:
            avg_rec_to = sum(rec_turnover) / len(rec_turnover)
            avg_mkt_to = sum(all_turnover) / len(all_turnover)
            pct_to = calc_percentile(all_turnover, avg_rec_to)
            factor_exposures["换手率"] = {
                "推荐均值": f"{avg_rec_to:.2f}%",
                "市场均值": f"{avg_mkt_to:.2f}%",
                "全市场分位": f"{pct_to:.0f}%",
                "暴露方向": "高活跃度" if pct_to > 60 else ("低活跃度" if pct_to < 40 else "中性"),
            }

        if rec_amplitude and all_amplitude:
            avg_rec_amp = sum(rec_amplitude) / len(rec_amplitude)
            avg_mkt_amp = sum(all_amplitude) / len(all_amplitude)
            pct_amp = calc_percentile(all_amplitude, avg_rec_amp)
            factor_exposures["振幅"] = {
                "推荐均值": f"{avg_rec_amp:.2f}%",
                "市场均值": f"{avg_mkt_amp:.2f}%",
                "全市场分位": f"{pct_amp:.0f}%",
                "暴露方向": "高波动" if pct_amp > 60 else ("低波动" if pct_amp < 40 else "中性"),
            }

        if rec_chg_60d and all_chg_60d:
            avg_rec_chg = sum(rec_chg_60d) / len(rec_chg_60d)
            avg_mkt_chg = sum(all_chg_60d) / len(all_chg_60d)
            pct_chg = calc_percentile(all_chg_60d, avg_rec_chg)
            factor_exposures["60日动量"] = {
                "推荐均值": f"{avg_rec_chg:.2f}%",
                "市场均值": f"{avg_mkt_chg:.2f}%",
                "全市场分位": f"{pct_chg:.0f}%",
                "暴露方向": "强动量" if pct_chg > 60 else ("弱动量" if pct_chg < 40 else "中性"),
            }

        analysis["因子暴露验证"] = factor_exposures

        # 风格一致性检测：检查推荐股票的风格是否与用户偏好一致
        style_checks = []
        momentum_pct = float(factor_exposures.get("60日动量", {}).get("全市场分位", "50").replace("%", ""))
        value_pe_pct = float(factor_exposures.get("市盈率", {}).get("全市场分位", "50").replace("%", ""))

        if momentum_pct > 70:
            style_checks.append("推荐股票整体偏向强动量风格，在趋势市中表现较好，但在市场反转时可能面临较大回撤")
        elif momentum_pct < 30:
            style_checks.append("推荐股票整体偏向弱动量/反转风格，可能在市场反弹时表现较好")

        if value_pe_pct < 30:
            style_checks.append("推荐股票整体偏向低估值/价值风格，防御性较强但成长性可能不足")
        elif value_pe_pct > 70:
            style_checks.append("推荐股票整体偏向高估值/成长风格，进攻性较强但估值风险较高")

        analysis["风格一致性检测"] = {
            "风格特征": style_checks if style_checks else ["推荐股票风格较为均衡，无明显风格偏向"],
            "风格漂移风险": "低" if len(style_checks) <= 1 else "中",
        }

        # 回溯表现（附带前视偏差警告）
        periods = {"5日": "5日涨跌幅", "20日": "20日涨跌幅", "60日": "60日涨跌幅"}
        for period, col in periods.items():
            sig_rets = []
            mkt_rets = []
            for _, row in spot_df.iterrows():
                try:
                    ret = float(row.get(col, 0))
                    if not np.isnan(ret):
                        mkt_rets.append(ret)
                except (ValueError, TypeError):
                    pass

            for c in candidates:
                code = str(c.get("代码", ""))
                row = code_map.get(code)
                if row is None:
                    continue
                try:
                    ret = float(row.get(col, 0))
                    if not np.isnan(ret):
                        sig_rets.append(ret)
                except (ValueError, TypeError):
                    pass

            if sig_rets and mkt_rets:
                avg_sig = sum(sig_rets) / len(sig_rets)
                avg_mkt = sum(mkt_rets) / len(mkt_rets)
                wins = sum(1 for r in sig_rets if r > 0)
                analysis["回溯表现"][period] = {
                    "推荐平均收益": f"{avg_sig:.2f}%",
                    "市场平均收益": f"{avg_mkt:.2f}%",
                    "超额收益": f"{avg_sig - avg_mkt:+.2f}%",
                    "胜率": f"{wins / len(sig_rets) * 100:.1f}%" if sig_rets else "N/A",
                    "样本数": len(sig_rets),
                }

        # 跟踪建议
        analysis["跟踪建议"].append("建议建立推荐日志文件，记录每次推荐的股票代码和日期，定期（每周/每月）统计推荐后5日、20日、60日的实际收益")
        analysis["跟踪建议"].append("可通过对比推荐股票等权组合与沪深300指数的走势，评估推荐信号的实战价值")
        analysis["跟踪建议"].append("建议设置推荐信号有效期（如5个交易日），过期后重新评估")

        # 因子暴露合理性检查
        if factor_exposures:
            extreme_exposures = []
            for factor, data in factor_exposures.items():
                pct_str = data.get("全市场分位", "50%")
                try:
                    pct = float(pct_str.replace("%", ""))
                    if pct > 80 or pct < 20:
                        extreme_exposures.append(f"{factor}(分位{pct:.0f}%)")
                except ValueError:
                    pass
            if extreme_exposures:
                analysis["跟踪建议"].append(f"以下因子暴露较为极端：{', '.join(extreme_exposures)}，建议检查是否与投资目标一致")

    except Exception as e:
        analysis["跟踪建议"] = [f"信号分析异常: {str(e)}"]

    return analysis


def _parameter_sensitivity(spot_df, base_weights, top_n=5):
    """
    策略参数敏感性分析：评估评分权重微调对推荐结果的影响
    对每个权重维度进行±10%、±20%扰动，计算推荐列表变化率
    返回: {敏感度分析, 参数悬崖检测, 鲁棒性评分, 分析建议}
    """
    import pandas as pd
    import numpy as np

    if spot_df is None or spot_df.empty:
        return {"敏感度分析": {}, "参数悬崖检测": [], "鲁棒性评分": 0, "分析建议": ["数据不足，无法进行参数敏感性分析"]}

    analysis = {
        "敏感度分析": {},
        "参数悬崖检测": [],
        "鲁棒性评分": 0,
        "分析建议": [],
    }

    try:
        # 获取基准推荐列表
        base_candidates = _get_top_candidates(spot_df, base_weights, top_n)
        base_codes = set(c["代码"] for c in base_candidates)

        if not base_codes:
            analysis["分析建议"].append("基准推荐列表为空，无法进行敏感性分析")
            return analysis

        # 对每个权重维度进行扰动测试
        weight_dims = ["trend", "value", "activity", "momentum", "risk"]
        perturbations = [-0.2, -0.1, 0.1, 0.2]

        total_sensitivity = 0
        dim_count = 0

        for dim in weight_dims:
            dim_sensitivities = []
            for pert in perturbations:
                # 创建扰动后的权重
                perturbed_weights = dict(base_weights)
                perturbed_weights[dim] = max(1, perturbed_weights[dim] + int(perturbed_weights[dim] * pert))

                # 重新归一化
                total = sum(perturbed_weights.values())
                perturbed_weights = {k: round(v / total * 100) for k, v in perturbed_weights.items()}

                # 获取扰动后的推荐列表
                perturbed_candidates = _get_top_candidates(spot_df, perturbed_weights, top_n)
                perturbed_codes = set(c["代码"] for c in perturbed_candidates)

                # 计算变化率
                if base_codes:
                    overlap = len(base_codes & perturbed_codes)
                    change_rate = 1 - (overlap / len(base_codes))
                    dim_sensitivities.append(change_rate)

            if dim_sensitivities:
                avg_sensitivity = sum(dim_sensitivities) / len(dim_sensitivities)
                max_sensitivity = max(dim_sensitivities)
                analysis["敏感度分析"][dim] = {
                    "平均变化率": round(avg_sensitivity, 3),
                    "最大变化率": round(max_sensitivity, 3),
                    "敏感等级": "高" if avg_sensitivity > 0.4 else ("中" if avg_sensitivity > 0.2 else "低"),
                }
                total_sensitivity += avg_sensitivity
                dim_count += 1

                # 参数悬崖检测：小扰动(±10%)导致大变化(>40%)
                small_pert_changes = [dim_sensitivities[i] for i, p in enumerate(perturbations) if abs(p) <= 0.1]
                if small_pert_changes and max(small_pert_changes) > 0.4:
                    analysis["参数悬崖检测"].append(f"{dim}维度存在参数悬崖：±10%权重变化导致推荐列表变化{max(small_pert_changes)*100:.0f}%，该参数极不稳定")

        # 鲁棒性评分（0-100，越高越鲁棒）
        if dim_count > 0:
            avg_total_sensitivity = total_sensitivity / dim_count
            robustness = max(0, min(100, round((1 - avg_total_sensitivity) * 100)))
            analysis["鲁棒性评分"] = robustness

            if robustness >= 80:
                analysis["分析建议"].append(f"参数鲁棒性评分{robustness}/100，模型对参数变化不敏感，推荐结果稳定可靠")
            elif robustness >= 60:
                analysis["分析建议"].append(f"参数鲁棒性评分{robustness}/100，模型有一定鲁棒性，但部分维度敏感需关注")
            elif robustness >= 40:
                analysis["分析建议"].append(f"参数鲁棒性评分{robustness}/100，模型对参数较敏感，建议谨慎使用并定期复核权重")
            else:
                analysis["分析建议"].append(f"参数鲁棒性评分{robustness}/100，模型极不稳定，参数微调即导致推荐大变，存在严重过拟合风险")

        # 悬崖检测建议
        if analysis["参数悬崖检测"]:
            cliff_dims = [d for d in analysis["参数悬崖检测"] if "参数悬崖" in d]
            if cliff_dims:
                analysis["分析建议"].append("检测到参数悬崖，建议：1) 使用更稳健的权重估计方法 2) 增加样本量 3) 考虑使用集成方法平滑权重")

    except Exception as e:
        analysis["分析建议"] = [f"参数敏感性分析异常: {str(e)}"]

    return analysis


def _get_top_candidates(spot_df, weights, top_n=5):
    """
    使用指定权重计算Top-N候选股票（简化版评分，用于敏感性分析）
    """
    import pandas as pd
    import numpy as np

    try:
        df = spot_df.copy()
        df = df[df['最新价'].notna() & (df['最新价'] > 0)].copy()
        df = df[~df['名称'].str.contains('ST', na=False)].copy()

        if df.empty:
            return []

        chg_today = df['涨跌幅'].fillna(0).astype(float)
        turnover = df['换手率'].fillna(0).astype(float)
        vol_ratio = df['量比'].fillna(1).astype(float)
        pe = df['市盈率-动态'].fillna(0).astype(float)
        pb = df.get('市净率', pd.Series(0, index=df.index)).fillna(0).astype(float)
        amplitude = df['振幅'].fillna(0).astype(float)
        chg_60d = df.get('60日涨跌幅', pd.Series(0, index=df.index)).fillna(0).astype(float)
        chg_ytd = df.get('年初至今涨跌幅', pd.Series(0, index=df.index)).fillna(0).astype(float)

        # 趋势得分
        trend_score = pd.Series(5, index=df.index)
        trend_score[chg_60d > 20] += 12
        trend_score[(chg_60d > 5) & (chg_60d <= 20)] += 8
        trend_score[(chg_60d > -5) & (chg_60d <= 5)] += 5
        trend_score[(chg_60d > -20) & (chg_60d <= -5)] += 3
        trend_score[chg_ytd > 30] += 8
        trend_score[(chg_ytd > 10) & (chg_ytd <= 30)] += 5
        trend_score[(chg_ytd > -10) & (chg_ytd <= 10)] += 3
        trend_score = trend_score.clip(upper=25)

        # 估值得分
        value_score = pd.Series(5, index=df.index)
        value_score[(pe > 0) & (pe < 15)] += 12
        value_score[(pe >= 15) & (pe < 25)] += 9
        value_score[(pe >= 25) & (pe < 40)] += 6
        value_score[(pe >= 40) & (pe < 60)] += 3
        value_score[(pb > 0) & (pb < 1.5)] += 8
        value_score[(pb >= 1.5) & (pb < 3)] += 5
        value_score[(pb >= 3) & (pb < 5)] += 3
        value_score = value_score.clip(upper=25)

        # 活跃度得分
        activity_score = (turnover * 1.5 + vol_ratio * 4).clip(upper=20)

        # 动量得分
        momentum_score = pd.Series(3, index=df.index)
        momentum_score[chg_today > 5] = 13
        momentum_score[(chg_today > 2) & (chg_today <= 5)] = 10
        momentum_score[(chg_today > 0) & (chg_today <= 2)] = 7
        momentum_score[(chg_today > -2) & (chg_today <= 0)] = 5
        momentum_score[(chg_today > -5) & (chg_today <= -2)] = 3

        # 风险得分
        risk_score = pd.Series(3, index=df.index)
        risk_score[(amplitude > 2) & (amplitude < 5)] = 12
        risk_score[((amplitude > 1) & (amplitude <= 2)) | ((amplitude >= 5) & (amplitude < 8))] = 8
        risk_score[amplitude >= 8] = 4

        total_score = (
            trend_score * weights.get('trend', 25) / 25 +
            value_score * weights.get('value', 25) / 25 +
            activity_score * weights.get('activity', 20) / 20 +
            momentum_score * weights.get('momentum', 15) / 15 +
            risk_score * weights.get('risk', 15) / 15
        ) * 20

        total_score = total_score.clip(upper=99).round().astype(int)

        result_df = pd.DataFrame({
            '代码': df['代码'].astype(str),
            '名称': df['名称'].fillna(df['代码']).astype(str),
            '综合评分': total_score,
        })

        result_df = result_df.sort_values('综合评分', ascending=False)
        result_df = result_df.head(top_n)

        candidates = []
        for _, row in result_df.iterrows():
            candidates.append({
                "代码": str(row['代码']),
                "名称": str(row['名称']),
                "综合评分": int(row['综合评分']),
            })

        return candidates
    except Exception:
        return []


def _factor_ic_analysis(spot_df):
    """
    因子有效性统计检验：计算各因子的Rank IC、IC_IR、因子收益率
    基于横截面数据评估因子对未来收益的预测能力
    返回: {因子检验结果, 有效因子列表, 无效因子列表, 权重调整建议}
    """
    import pandas as pd
    import numpy as np

    if spot_df is None or spot_df.empty:
        return {"因子检验结果": {}, "有效因子列表": [], "无效因子列表": [], "权重调整建议": []}

    analysis = {
        "因子检验结果": {},
        "有效因子列表": [],
        "无效因子列表": [],
        "权重调整建议": [],
    }

    try:
        df = spot_df.copy()

        # 定义因子和对应的列名
        factor_map = {
            "市盈率(价值)": "市盈率-动态",
            "市净率(价值)": "市净率",
            "换手率(活跃度)": "换手率",
            "量比(活跃度)": "量比",
            "振幅(风险)": "振幅",
            "当日涨跌(动量)": "涨跌幅",
        }

        # 未来收益代理：60日涨跌幅（如果可用）
        forward_return_col = None
        if '60日涨跌幅' in df.columns:
            forward_return_col = '60日涨跌幅'
        elif '年初至今涨跌幅' in df.columns:
            forward_return_col = '年初至今涨跌幅'

        if forward_return_col is None:
            analysis["权重调整建议"].append("缺少未来收益数据（60日涨跌幅），无法进行IC分析，保持默认权重")
            return analysis

        # 准备数据
        forward_ret = pd.to_numeric(df[forward_return_col], errors='coerce')
        valid_mask = forward_ret.notna()

        for factor_name, col_name in factor_map.items():
            if col_name not in df.columns:
                continue

            factor_values = pd.to_numeric(df[col_name], errors='coerce')
            factor_mask = factor_values.notna() & valid_mask

            if factor_mask.sum() < 30:
                continue

            f_vals = factor_values[factor_mask]
            f_ret = forward_ret[factor_mask]

            # Rank IC：因子排名与未来收益排名的Spearman相关系数
            f_rank = f_vals.rank()
            ret_rank = f_ret.rank()
            n = len(f_rank)

            if n < 10:
                continue

            # Spearman秩相关系数
            d_sq = ((f_rank - ret_rank) ** 2).sum()
            rank_ic = 1 - (6 * d_sq) / (n * (n**2 - 1))

            # 因子收益率：因子值最高组 vs 最低组的未来收益差
            top_quantile = f_vals >= f_vals.quantile(0.8)
            bottom_quantile = f_vals <= f_vals.quantile(0.2)
            top_ret = f_ret[top_quantile].mean() if top_quantile.sum() > 0 else 0
            bottom_ret = f_ret[bottom_quantile].mean() if bottom_quantile.sum() > 0 else 0
            factor_return = top_ret - bottom_ret

            # IC符号一致性（正IC表示因子值越高未来收益越高）
            ic_direction = "正向" if rank_ic > 0 else "反向"

            # 有效性判断
            abs_ic = abs(rank_ic)
            if abs_ic >= 0.05:
                effectiveness = "强有效"
                analysis["有效因子列表"].append(factor_name)
            elif abs_ic >= 0.02:
                effectiveness = "弱有效"
                analysis["有效因子列表"].append(factor_name)
            else:
                effectiveness = "无效"
                analysis["无效因子列表"].append(factor_name)

            analysis["因子检验结果"][factor_name] = {
                "Rank_IC": round(rank_ic, 4),
                "IC绝对值": round(abs_ic, 4),
                "有效性": effectiveness,
                "IC方向": ic_direction,
                "因子收益率": f"{factor_return:.2f}%",
                "样本数": int(n),
            }

        # 生成权重调整建议
        valid_count = len(analysis["有效因子列表"])
        invalid_count = len(analysis["无效因子列表"])

        if invalid_count > 0:
            invalid_names = "、".join(analysis["无效因子列表"])
            analysis["权重调整建议"].append(f"以下因子IC不显著，建议降低权重或剔除：{invalid_names}")

        if valid_count > 0:
            valid_names = "、".join(analysis["有效因子列表"])
            analysis["权重调整建议"].append(f"以下因子通过IC检验，可维持或提高权重：{valid_names}")

        if valid_count == 0:
            analysis["权重调整建议"].append("所有因子均未通过IC检验，当前市场环境下因子模型可能失效，建议降低仓位观望")

        # 计算IC_IR（所有因子的IC均值/IC标准差）
        ic_values = [abs(v["Rank_IC"]) for v in analysis["因子检验结果"].values()]
        if len(ic_values) >= 2:
            ic_mean = np.mean(ic_values)
            ic_std = np.std(ic_values, ddof=1)
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            analysis["IC_IR汇总"] = {
                "平均|IC|": round(ic_mean, 4),
                "IC标准差": round(ic_std, 4),
                "IC_IR": round(ic_ir, 2),
            }
            if ic_ir > 0.5:
                analysis["权重调整建议"].append(f"因子整体IC_IR={ic_ir:.2f}，因子体系稳定性良好")
            elif ic_ir > 0.2:
                analysis["权重调整建议"].append(f"因子整体IC_IR={ic_ir:.2f}，因子体系稳定性一般，需持续监控")
            else:
                analysis["权重调整建议"].append(f"因子整体IC_IR={ic_ir:.2f}偏低，因子体系不稳定，建议降低模型依赖")

    except Exception as e:
        analysis["权重调整建议"] = [f"因子IC分析异常: {str(e)}"]

    return analysis


def _detect_market_regime(spot_df):
    """
    检测当前市场状态（牛市/熊市/震荡）
    基于上证指数(000001)的60日涨跌幅判断
    返回: {regime, index_60d_chg, index_name, index_price}
    """
    regime = "震荡"
    index_60d_chg = 0.0
    index_name = "上证指数"
    index_price = None

    try:
        row = spot_df[spot_df['代码'] == '000001']
        if not row.empty:
            r = row.iloc[0]
            index_price = _safe_float(r.get('最新价'))
            chg_60d = _safe_float(r.get('60日涨跌幅'))
            if chg_60d is not None:
                index_60d_chg = chg_60d
                if chg_60d > 15:
                    regime = "牛市"
                elif chg_60d > 5:
                    regime = "偏强震荡"
                elif chg_60d < -15:
                    regime = "熊市"
                elif chg_60d < -5:
                    regime = "偏弱震荡"
                else:
                    regime = "震荡"
    except Exception:
        pass

    return {
        "regime": regime,
        "index_60d_chg": index_60d_chg,
        "index_name": index_name,
        "index_price": index_price,
    }


def _get_regime_weights(regime, risk_level="中等"):
    """
    根据市场状态和风险偏好返回动态权重
    权重总和为100，分配到五个维度
    """
    # 基础权重（震荡市 + 中等风险）
    base_weights = {
        "trend": 25, "value": 25, "activity": 20, "momentum": 15, "risk": 15
    }

    # 市场状态调整
    if regime == "牛市":
        base_weights = {"trend": 30, "value": 15, "activity": 22, "momentum": 20, "risk": 13}
    elif regime == "偏强震荡":
        base_weights = {"trend": 28, "value": 20, "activity": 20, "momentum": 18, "risk": 14}
    elif regime == "偏弱震荡":
        base_weights = {"trend": 20, "value": 28, "activity": 18, "momentum": 12, "risk": 22}
    elif regime == "熊市":
        base_weights = {"trend": 15, "value": 30, "activity": 15, "momentum": 10, "risk": 30}

    # 风险偏好微调
    if risk_level == "高":
        base_weights["momentum"] = min(base_weights["momentum"] + 5, 30)
        base_weights["activity"] = min(base_weights["activity"] + 3, 25)
        base_weights["risk"] = max(base_weights["risk"] - 5, 5)
    elif risk_level == "低":
        base_weights["value"] = min(base_weights["value"] + 5, 35)
        base_weights["risk"] = min(base_weights["risk"] + 5, 35)
        base_weights["momentum"] = max(base_weights["momentum"] - 5, 5)

    # 归一化确保总和为100
    total = sum(base_weights.values())
    return {k: round(v / total * 100) for k, v in base_weights.items()}


def _adaptive_strategy_adjustment(spot_df, regime_info, microstructure, behavioral, cross_market):
    """
    自适应动态策略调整：根据综合市场环境动态调整策略参数
    整合市场状态、微观结构、行为金融、跨市场分析的结果
    返回: {市场综合评估, 动态仓位建议, 动态止损止盈, 策略类型推荐, 因子权重调整, 操作建议}
    """
    import pandas as pd
    import numpy as np

    if spot_df is None or spot_df.empty:
        return {"市场综合评估": {}, "动态仓位建议": {}, "动态止损止盈": {}, "策略类型推荐": {}, "因子权重调整": {}, "操作建议": ["数据不足，无法进行自适应策略调整"]}

    adjustment = {
        "市场综合评估": {},
        "动态仓位建议": {},
        "动态止损止盈": {},
        "策略类型推荐": {},
        "因子权重调整": {},
        "操作建议": [],
    }

    try:
        df = spot_df.copy()
        chg_today = pd.to_numeric(df['涨跌幅'], errors='coerce').fillna(0)
        amplitude = pd.to_numeric(df['振幅'], errors='coerce').fillna(0)
        turnover = pd.to_numeric(df['换手率'], errors='coerce').fillna(0)

        # 1. 市场综合评估
        regime = regime_info.get("regime", "震荡")
        index_60d = regime_info.get("index_60d_chg", 0)

        # 微观结构信号
        micro_signals = []
        if microstructure:
            liq = microstructure.get("流动性分析", {})
            vol_cluster = microstructure.get("波动率聚类", {})
            if liq:
                liq_level = liq.get("流动性评级", "")
                if liq_level == "紧张":
                    micro_signals.append("流动性偏紧")
                elif liq_level == "充裕":
                    micro_signals.append("流动性充裕")
            if vol_cluster:
                vol_state = vol_cluster.get("波动状态", "")
                if vol_state == "高波动":
                    micro_signals.append("高波动环境")

        # 行为金融信号
        behavior_signals = []
        if behavioral:
            herd = behavioral.get("羊群效应", {})
            panic = behavioral.get("恐慌指数", {})
            if herd:
                herd_level = herd.get("羊群效应等级", "")
                if herd_level == "强":
                    behavior_signals.append("羊群效应强")
            if panic:
                panic_level = panic.get("恐慌等级", "")
                if panic_level == "高":
                    behavior_signals.append("恐慌情绪高")

        # 跨市场信号
        cross_signals = []
        if cross_market:
            fund_flow = cross_market.get("资金流向推断", {})
            if fund_flow:
                preference = fund_flow.get("资金偏好", "")
                if preference != "均衡":
                    cross_signals.append(f"资金偏好{preference}")

        # 综合评分
        score = 50
        if regime == "牛市":
            score += 20
        elif regime == "偏强震荡":
            score += 10
        elif regime == "偏弱震荡":
            score -= 10
        elif regime == "熊市":
            score -= 20

        if "高波动环境" in micro_signals:
            score -= 10
        if "流动性偏紧" in micro_signals:
            score -= 10
        if "流动性充裕" in micro_signals:
            score += 5
        if "羊群效应强" in behavior_signals:
            score -= 5
        if "恐慌情绪高" in behavior_signals:
            score -= 10

        score = max(0, min(100, score))

        env_level = "积极" if score >= 70 else ("中性" if score >= 40 else "谨慎")

        adjustment["市场综合评估"] = {
            "综合评分": f"{score}/100",
            "环境等级": env_level,
            "市场状态": regime,
            "指数60日涨跌": f"{index_60d:.2f}%",
            "微观结构信号": micro_signals if micro_signals else ["无明显异常"],
            "行为金融信号": behavior_signals if behavior_signals else ["无明显异常"],
            "跨市场信号": cross_signals if cross_signals else ["无明显异常"],
        }

        # 2. 动态仓位建议
        base_position = 60
        if regime == "牛市":
            base_position = 80
        elif regime == "偏强震荡":
            base_position = 70
        elif regime == "偏弱震荡":
            base_position = 50
        elif regime == "熊市":
            base_position = 30

        if "高波动环境" in micro_signals:
            base_position -= 15
        if "流动性偏紧" in micro_signals:
            base_position -= 10
        if "恐慌情绪高" in behavior_signals:
            base_position -= 10
        if "羊群效应强" in behavior_signals:
            base_position -= 5

        base_position = max(10, min(100, base_position))

        adjustment["动态仓位建议"] = {
            "建议仓位": f"{base_position}%",
            "现金保留": f"{100 - base_position}%",
            "仓位等级": "重仓" if base_position >= 70 else ("中等" if base_position >= 40 else "轻仓"),
            "调整依据": [],
        }

        if base_position >= 70:
            adjustment["动态仓位建议"]["调整依据"].append("市场环境积极，可适度提高仓位以获取更多收益")
        elif base_position >= 40:
            adjustment["动态仓位建议"]["调整依据"].append("市场环境中性，保持中等仓位，灵活应对")
        else:
            adjustment["动态仓位建议"]["调整依据"].append("市场环境谨慎，建议降低仓位控制风险")

        # 3. 动态止损止盈
        avg_amplitude = amplitude.mean() if len(amplitude) > 0 else 3
        avg_chg_abs = chg_today.abs().mean() if len(chg_today) > 0 else 2

        base_stop_loss = max(3, avg_amplitude * 1.5)
        base_take_profit = max(5, avg_amplitude * 3)

        if regime == "熊市":
            base_stop_loss *= 0.8
            base_take_profit *= 0.7
        elif regime == "牛市":
            base_stop_loss *= 1.2
            base_take_profit *= 1.3

        if "高波动环境" in micro_signals:
            base_stop_loss *= 1.3
            base_take_profit *= 1.2

        adjustment["动态止损止盈"] = {
            "建议止损幅度": f"{base_stop_loss:.1f}%",
            "建议止盈幅度": f"{base_take_profit:.1f}%",
            "止损说明": f"基于市场平均振幅{avg_amplitude:.1f}%动态计算，{'高波动环境下适当放宽' if '高波动环境' in micro_signals else '正常波动环境'}",
            "止盈说明": f"基于市场平均振幅{avg_amplitude:.1f}%动态计算，{regime}环境下{'适当提高止盈目标' if regime in ('牛市', '偏强震荡') else '适当降低止盈目标'}",
        }

        # 4. 策略类型推荐
        strategies = []

        if regime in ("牛市", "偏强震荡"):
            strategies.append({
                "策略": "趋势跟踪策略",
                "适用度": "高",
                "说明": "市场处于上升趋势，趋势跟踪策略有效性较高，建议关注强势股回调买入机会",
            })
            strategies.append({
                "策略": "动量策略",
                "适用度": "高",
                "说明": "强势市场下动量效应明显，可关注近期涨幅领先的标的",
            })
        elif regime in ("熊市", "偏弱震荡"):
            strategies.append({
                "策略": "价值防御策略",
                "适用度": "高",
                "说明": "市场偏弱，建议关注低估值、高股息的价值防御标的",
            })
            strategies.append({
                "策略": "逆向投资策略",
                "适用度": "中",
                "说明": "弱市中可关注超跌优质标的的反弹机会，但需严格控制仓位",
            })
        else:
            strategies.append({
                "策略": "均衡配置策略",
                "适用度": "高",
                "说明": "震荡市中建议均衡配置，兼顾进攻与防守",
            })
            strategies.append({
                "策略": "波段操作策略",
                "适用度": "中",
                "说明": "震荡市适合波段操作，高抛低吸，注意把握节奏",
            })

        if "恐慌情绪高" in behavior_signals:
            strategies.append({
                "策略": "逆向布局策略",
                "适用度": "中",
                "说明": "恐慌情绪下优质标的可能被错杀，可小仓位分批布局，但需耐心等待市场企稳",
            })

        if "羊群效应强" in behavior_signals:
            strategies.append({
                "策略": "独立选股策略",
                "适用度": "中",
                "说明": "羊群效应强时市场同涨同跌，精选个股的能力更加重要，避免盲目跟风",
            })

        adjustment["策略类型推荐"] = {
            "推荐策略": strategies,
            "核心思路": "",
        }

        if env_level == "积极":
            adjustment["策略类型推荐"]["核心思路"] = "进攻为主，防守为辅，积极把握市场机会"
        elif env_level == "中性":
            adjustment["策略类型推荐"]["核心思路"] = "攻守兼备，灵活调整，注重选股质量"
        else:
            adjustment["策略类型推荐"]["核心思路"] = "防守为主，进攻为辅，严格控制风险"

        # 5. 因子权重调整
        current_weights = _get_regime_weights(regime, "中等")

        weight_adjustments = []
        if "高波动环境" in micro_signals:
            current_weights["risk"] = min(current_weights["risk"] + 5, 35)
            current_weights["momentum"] = max(current_weights["momentum"] - 3, 5)
            weight_adjustments.append("高波动环境：提高风险控制权重，降低动量权重")

        if "流动性偏紧" in micro_signals:
            current_weights["activity"] = max(current_weights["activity"] - 3, 5)
            weight_adjustments.append("流动性偏紧：降低资金活跃度权重")

        if "羊群效应强" in behavior_signals:
            current_weights["value"] = min(current_weights["value"] + 3, 35)
            weight_adjustments.append("羊群效应强：提高估值权重，注重安全边际")

        total = sum(current_weights.values())
        current_weights = {k: round(v / total * 100) for k, v in current_weights.items()}

        adjustment["因子权重调整"] = {
            "调整后权重": current_weights,
            "调整说明": weight_adjustments if weight_adjustments else ["当前环境无需额外调整权重"],
        }

        # 6. 操作建议
        if env_level == "积极":
            adjustment["操作建议"].append("市场环境积极，可适当提高仓位至70%以上，关注强势板块龙头")
            adjustment["操作建议"].append(f"止损设置建议{base_stop_loss:.1f}%，止盈设置建议{base_take_profit:.1f}%")
        elif env_level == "中性":
            adjustment["操作建议"].append("市场环境中性，建议仓位控制在40%-70%，均衡配置")
            adjustment["操作建议"].append("关注板块轮动节奏，避免追高，逢低布局优质标的")
        else:
            adjustment["操作建议"].append("市场环境谨慎，建议仓位控制在40%以下，以防御为主")
            adjustment["操作建议"].append("可关注高股息、低估值防御标的，保留充足现金等待更好机会")

        if "恐慌情绪高" in behavior_signals:
            adjustment["操作建议"].append("恐慌情绪下避免恐慌性抛售，理性评估持仓，优质标的可继续持有")

        adjustment["操作建议"].append("以上建议基于当前市场环境动态生成，市场环境变化时需重新评估")

    except Exception as e:
        adjustment["操作建议"] = [f"自适应策略调整异常: {str(e)}"]

    return adjustment


def _fast_score_stocks(spot_df, preference="", risk_level="中等", top_n=5, price_range="", market=""):
    """
    快速评分：使用实时行情数据对全市场股票进行评分
    spot_df: 已获取的实时行情DataFrame
    market: "sh"=上证, "sz"=深证, ""=全部
    """
    import numpy as np
    import pandas as pd

    if spot_df is None or spot_df.empty:
        print("[AI推荐] 实时行情数据为空")
        return [], {}, {}

    print(f"[AI推荐] 全市场股票数量: {len(spot_df)}")

    # 市场状态检测（在过滤前，使用全量数据判断大盘状态）
    market_regime = _detect_market_regime(spot_df)
    print(f"[AI推荐] 市场状态: {market_regime['regime']} (上证60日涨跌: {market_regime['index_60d_chg']:+.1f}%)")

    # 因子有效性统计检验
    factor_ic = _factor_ic_analysis(spot_df)
    invalid_factors = factor_ic.get("无效因子列表", [])
    if invalid_factors:
        print(f"[AI推荐] 因子IC检验 - 无效因子: {', '.join(invalid_factors)}")
    ic_ir_summary = factor_ic.get("IC_IR汇总", {})
    if ic_ir_summary:
        print(f"[AI推荐] 因子IC_IR: {ic_ir_summary.get('IC_IR', 'N/A')}")

    # 根据市场状态动态调整权重
    weights = _get_regime_weights(market_regime['regime'], risk_level)
    # 根据因子IC结果微调权重：无效因子降权
    if invalid_factors:
        for f in invalid_factors:
            if "市盈率" in f or "市净率" in f:
                weights['value'] = max(weights['value'] - 8, 5)
            if "换手率" in f or "量比" in f:
                weights['activity'] = max(weights['activity'] - 8, 5)
            if "振幅" in f:
                weights['risk'] = max(weights['risk'] - 8, 5)
            if "涨跌" in f:
                weights['momentum'] = max(weights['momentum'] - 8, 5)
        # 重新归一化
        total = sum(weights.values())
        weights = {k: round(v / total * 100) for k, v in weights.items()}
    print(f"[AI推荐] 动态权重: 趋势={weights['trend']} 估值={weights['value']} 活跃度={weights['activity']} 动量={weights['momentum']} 风险={weights['risk']}")

    # 参数敏感性分析
    sensitivity = _parameter_sensitivity(spot_df, weights, top_n)
    print(f"[AI推荐] 参数鲁棒性评分: {sensitivity.get('鲁棒性评分', 'N/A')}/100")

    # 根据市场筛选
    if market == "sh":
        spot_df = spot_df[spot_df['代码'].str.startswith('6')].copy()
        market_label = "上证指数"
    elif market == "sz":
        spot_df = spot_df[spot_df['代码'].str.startswith(('0', '3'))].copy()
        market_label = "深证成指"
    else:
        market_label = "全市场"

    print(f"[AI推荐] 市场筛选({market_label})后: {len(spot_df)}只")

    # 基础过滤：排除价格异常、ST股票
    spot_df = spot_df[spot_df['最新价'].notna() & (spot_df['最新价'] > 0)].copy()
    # 排除名称中包含ST的股票
    spot_df = spot_df[~spot_df['名称'].str.contains('ST', na=False)].copy()
    print(f"[AI推荐] 基础过滤后: {len(spot_df)}只")

    if spot_df.empty:
        print("[AI推荐] 过滤后无可用股票")
        return [], {}, {}

    # 价格过滤（支持字符串桶和(lo, hi)元组两种格式）
    if price_range:
        if isinstance(price_range, (list, tuple)) and len(price_range) == 2:
            lo, hi = float(price_range[0]), float(price_range[1])
            spot_df = spot_df[(spot_df['最新价'] >= lo) & (spot_df['最新价'] <= hi)]
            print(f"[AI推荐] 价格过滤({lo}-{hi}元)后: {len(spot_df)}只")
        else:
            price_ranges = {
                "0-2": (0, 2), "3-5": (3, 5), "6-10": (6, 10),
                "11-20": (11, 20), "21-50": (21, 50),
                "51-100": (51, 100), "100+": (100, float('inf')),
            }
            pr = price_ranges.get(price_range)
            if pr:
                lo, hi = pr
                spot_df = spot_df[(spot_df['最新价'] >= lo) & (spot_df['最新价'] <= hi)]
                print(f"[AI推荐] 价格过滤({price_range})后: {len(spot_df)}只")

    if spot_df.empty:
        return [], {}, {}

    # 向量化计算评分（多周期多因子模型，避免逐行循环）
    chg_today = spot_df['涨跌幅'].fillna(0).astype(float)
    turnover = spot_df['换手率'].fillna(0).astype(float)
    vol_ratio = spot_df['量比'].fillna(1).astype(float)
    pe = spot_df['市盈率-动态'].fillna(0).astype(float)
    pb = spot_df.get('市净率', pd.Series(0, index=spot_df.index)).fillna(0).astype(float)
    amplitude = spot_df['振幅'].fillna(0).astype(float)
    price = spot_df['最新价'].astype(float)
    chg_60d = spot_df.get('60日涨跌幅', pd.Series(0, index=spot_df.index)).fillna(0).astype(float)
    chg_ytd = spot_df.get('年初至今涨跌幅', pd.Series(0, index=spot_df.index)).fillna(0).astype(float)

    # ===== 1. 中期趋势得分（0-25分）：基于60日涨跌幅 + 年初至今涨跌幅 =====
    trend_score = pd.Series(5, index=spot_df.index)
    # 60日趋势
    trend_score[chg_60d > 20] += 12
    trend_score[(chg_60d > 5) & (chg_60d <= 20)] += 8
    trend_score[(chg_60d > -5) & (chg_60d <= 5)] += 5
    trend_score[(chg_60d > -20) & (chg_60d <= -5)] += 3
    # 年初至今趋势
    trend_score[chg_ytd > 30] += 8
    trend_score[(chg_ytd > 10) & (chg_ytd <= 30)] += 5
    trend_score[(chg_ytd > -10) & (chg_ytd <= 10)] += 3
    trend_score = trend_score.clip(upper=25)

    # ===== 2. 估值得分（0-25分）：基于市盈率 + 市净率 =====
    value_score = pd.Series(5, index=spot_df.index)
    # 市盈率评估
    value_score[(pe > 0) & (pe < 15)] += 12
    value_score[(pe >= 15) & (pe < 25)] += 9
    value_score[(pe >= 25) & (pe < 40)] += 6
    value_score[(pe >= 40) & (pe < 60)] += 3
    # 市净率评估
    value_score[(pb > 0) & (pb < 1.5)] += 8
    value_score[(pb >= 1.5) & (pb < 3)] += 5
    value_score[(pb >= 3) & (pb < 5)] += 3
    value_score = value_score.clip(upper=25)

    # ===== 3. 活跃度得分（0-20分）：基于换手率 + 量比 =====
    activity_score = (turnover * 1.5 + vol_ratio * 4).clip(upper=20)

    # ===== 4. 短期动量得分（0-15分）：基于今日涨跌幅（权重降低） =====
    momentum_score = pd.Series(3, index=spot_df.index)
    momentum_score[chg_today > 5] = 13
    momentum_score[(chg_today > 2) & (chg_today <= 5)] = 10
    momentum_score[(chg_today > 0) & (chg_today <= 2)] = 7
    momentum_score[(chg_today > -2) & (chg_today <= 0)] = 5
    momentum_score[(chg_today > -5) & (chg_today <= -2)] = 3

    # ===== 5. 风险控制得分（0-15分）：基于振幅（适中为好） =====
    risk_score = pd.Series(3, index=spot_df.index)
    risk_score[(amplitude > 2) & (amplitude < 5)] = 12
    risk_score[((amplitude > 1) & (amplitude <= 2)) | ((amplitude >= 5) & (amplitude < 8))] = 8
    risk_score[amplitude >= 8] = 4

    total_score = (
        trend_score * weights['trend'] / 25 +
        value_score * weights['value'] / 25 +
        activity_score * weights['activity'] / 20 +
        momentum_score * weights['momentum'] / 15 +
        risk_score * weights['risk'] / 15
    ) * 20

    total_score = total_score.clip(upper=99).round().astype(int)

    # 构建结果
    result_df = pd.DataFrame({
        '代码': spot_df['代码'].astype(str),
        '名称': spot_df['名称'].fillna(spot_df['代码']).astype(str),
        '综合评分': total_score,
        '最新价': price,
        '涨跌幅': chg_today,
        '换手率': turnover,
        '市盈率': pe,
        '市净率': pb,
        '60日涨跌幅': chg_60d,
        '年初至今涨跌幅': chg_ytd,
        '量比': vol_ratio,
        '振幅': amplitude,
    })

    # 按评分排序取top_n
    result_df = result_df.sort_values('综合评分', ascending=False)
    result_df = result_df.head(max(top_n, 20))

    # 评级
    def get_rating(s):
        if s >= 70: return "强烈推荐"
        if s >= 55: return "推荐"
        if s >= 40: return "关注"
        if s >= 25: return "观望"
        return "回避"

    scored = []
    for _, row in result_df.iterrows():
        code = str(row['代码'])
        name = str(row['名称'])
        if name == 'nan' or name == '' or name == code:
            name = code
        t_val = row['换手率']
        p_val = row['市盈率']
        pb_val = row['市净率']
        turnover = round(float(t_val), 2) if pd.notna(t_val) and float(t_val) > 0 else None
        pe = round(float(p_val), 2) if pd.notna(p_val) and float(p_val) > 0 else None
        pb = round(float(pb_val), 2) if pd.notna(pb_val) and float(pb_val) > 0 else None
        chg_60d_val = row['60日涨跌幅']
        chg_ytd_val = row['年初至今涨跌幅']
        chg_60d = round(float(chg_60d_val), 2) if pd.notna(chg_60d_val) else None
        chg_ytd = round(float(chg_ytd_val), 2) if pd.notna(chg_ytd_val) else None
        scored.append({
            "代码": code,
            "名称": name,
            "综合评分": int(row['综合评分']),
            "评级": get_rating(int(row['综合评分'])),
            "最新价": round(float(row['最新价']), 2),
            "涨跌幅": round(float(row['涨跌幅']), 2),
            "换手率": turnover,
            "市盈率": pe,
            "市净率": pb,
            "60日涨跌幅": chg_60d,
            "年初至今涨跌幅": chg_ytd,
        })

    print(f"[AI推荐] 评分完成，共{len(scored)}只股票入围")
    return scored, factor_ic, sensitivity


def _fallback_kline_score(pool, top_n=5):
    """
    降级方案：使用K线数据逐只评分（当实时行情不可用时）
    """
    candidates = []
    import pandas as pd
    try:
        from data_utils import get_stock_kline
    except Exception as e:
        print(f"[AI推荐] 导入get_stock_kline失败: {e}")
        return candidates

    # 尝试从实时行情获取名称映射
    name_map = {}
    try:
        from data_utils import _get_spot_df
        spot_df = _get_spot_df()
        if spot_df is not None and not spot_df.empty:
            for _, row in spot_df.iterrows():
                code = str(row.get('代码', ''))
                raw_name = row.get('名称', '')
                if not pd.isna(raw_name) and str(raw_name).strip() != '' and str(raw_name).strip() != 'nan':
                    name_map[code] = str(raw_name).strip()
    except Exception:
        pass

    success_count = 0
    for sym in pool[:max(top_n, 10)]:
        try:
            df = get_stock_kline(sym, days=30)
            if df is None or len(df) < 5:
                continue

            # 兼容中英文列名
            close_col = '收盘' if '收盘' in df.columns else ('close' if 'close' in df.columns else None)
            vol_col = '成交量' if '成交量' in df.columns else ('volume' if 'volume' in df.columns else None)
            if close_col is None:
                print(f"[AI推荐] K线降级: {sym} 无收盘列, 实际列名: {list(df.columns)[:8]}")
                continue

            close = df[close_col].astype(float)
            volume = df[vol_col].astype(float) if vol_col else close * 0 + 1
            chg_5d = (close.iloc[-1] / close.iloc[-min(5, len(close))] - 1) * 100
            chg_20d = (close.iloc[-1] / close.iloc[-min(20, len(close))] - 1) * 100
            vol_ratio = volume.iloc[-1] / volume.tail(5).mean() if len(volume) >= 5 else 1
            score = min(round(abs(chg_5d) * 3 + chg_20d * 2 + vol_ratio * 5 + 30), 95)
            rating = "推荐" if score >= 55 else "关注" if score >= 40 else "观望"
            stock_name = name_map.get(sym, sym)
            candidates.append({
                "代码": sym, "名称": stock_name, "综合评分": score, "评级": rating,
                "最新价": float(close.iloc[-1]), "涨跌幅": round(chg_5d, 2),
                "换手率": 0, "市盈率": 0,
            })
            success_count += 1
        except Exception as e:
            print(f"[AI推荐] K线降级评分失败 {sym}: {e}")
    candidates.sort(key=lambda x: x["综合评分"], reverse=True)
    print(f"[AI推荐] K线降级评分完成，成功{success_count}只")
    return candidates


def ai_recommend_stocks(preference="", risk_level="中等", top_n=5, price_range="", market=""):
    """
    AI智能推荐股票
    从全市场实时行情中动态筛选并评分，无需硬编码股票池
    market: "sh"=上证, "sz"=深证, ""=全部
    """
    result = {
        "推荐时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "推荐方法": "实时行情多因子模型",
        "推荐股票": [],
        "分析说明": ""
    }

    market_label = {"sh": "上证指数", "sz": "深证成指"}.get(market, "全市场")
    print(f"[AI推荐] 市场筛选: {market_label}")

    # 第一步：获取实时行情数据（一次API调用）
    try:
        from data_utils import _get_spot_df
        spot_df = _get_spot_df()
    except Exception as e:
        print(f"[AI推荐] 获取实时行情失败: {e}")
        spot_df = None

    # 数据质量管道：清洗和评估数据质量
    data_quality = _data_quality_pipeline(spot_df)
    result["数据质量"] = data_quality
    if data_quality.get("清洗后数据") is not None:
        spot_df = data_quality["清洗后数据"]
        print(f"[AI推荐] 数据质量评分: {data_quality['质量评分']}/100 ({data_quality.get('质量评级', '未知')})")

    # 市场择时系统：判断当前市场是否适合进场
    market_timing = _market_timing_system(spot_df)
    result["市场择时"] = market_timing
    print(f"[AI推荐] 市场择时评分: {market_timing['择时评分']}/100 ({market_timing.get('择时等级', '未知')})")

    # 动态对冲建议：基于市场状态给出对冲策略
    hedge_advice = _dynamic_hedge_advisor(spot_df, market_timing)
    result["对冲建议"] = hedge_advice
    if hedge_advice.get("对冲必要性") not in ["低", "无法判断"]:
        print(f"[AI推荐] 对冲建议: {hedge_advice['对冲必要性']}必要性，对冲评分{hedge_advice['对冲评分']}/100")

    # 资金曲线管理：动态Kelly调整、回撤控制、加仓减仓规则
    equity_mgr = _equity_curve_manager(spot_df, market_timing)
    result["资金曲线管理"] = equity_mgr
    print(f"[AI推荐] 资金曲线健康度: {equity_mgr['健康度评估'].get('健康等级', '未知')} ({equity_mgr['健康度评估'].get('健康评分', 0)}分)")

    # 交易心理辅助：检测贪婪/恐惧/过度自信/锚定效应等行为偏差
    psych_assist = _trading_psychology_assistant(spot_df, market_timing)
    result["交易心理"] = psych_assist
    print(f"[AI推荐] 交易心理状态: {psych_assist.get('心理状态', '未知')} ({psych_assist['心理评分']}分)")

    # 板块轮动节奏量化：轮动速度、强度、方向、动量持续性
    sector_rotation = _sector_rotation_quantifier(spot_df)
    result["板块轮动"] = sector_rotation
    print(f"[AI推荐] 板块轮动评分: {sector_rotation['轮动评分']}/100 ({sector_rotation['轮动速度'].get('轮动速度等级', '未知')})")

    # 多时间框架共振分析：短中长周期趋势共振判断
    timeframe_resonance = _multi_timeframe_resonance(spot_df)
    result["多周期共振"] = timeframe_resonance
    print(f"[AI推荐] 多周期共振: {timeframe_resonance['共振状态']} ({timeframe_resonance['共振评分']}分)")

    # 量价关系深度分析：识别放量突破/缩量回调/量价背离/底部放量/高位滞涨
    vol_price = _volume_price_analyzer(spot_df)
    result["量价关系"] = vol_price
    print(f"[AI推荐] 量价关系: {vol_price['综合判断']} ({vol_price['量价评分']}分)")

    # 财报季效应分析：基于当前日期判断财报季阶段
    earnings_season = _earnings_season_analyzer()
    result["财报季分析"] = earnings_season
    print(f"[AI推荐] 财报季阶段: {earnings_season['财报阶段']}")

    # 日历效应与季节性分析：月初月末效应、周内效应、季节性规律
    calendar_effect = _calendar_effect_analyzer()
    result["日历效应"] = calendar_effect
    print(f"[AI推荐] 日历效应: {calendar_effect['月初月末效应'].get('阶段', '')} + {calendar_effect['季节性规律'].get('规律', '')}")

    # 第二步：快速评分（向量化计算，全市场股票参与）
    candidates, factor_ic, sensitivity = _fast_score_stocks(spot_df, preference, risk_level, top_n, price_range, market)

    # 将因子IC分析结果加入返回
    result["因子IC分析"] = factor_ic
    # 将参数敏感性分析结果加入返回
    result["参数敏感性分析"] = sensitivity

    # Alpha衰减检测
    alpha_decay = _alpha_decay_detection(spot_df, factor_ic)
    result["Alpha衰减检测"] = alpha_decay

    # 第三步：降级方案（K线数据逐只评分，尊重价格范围）
    if not candidates:
        result["推荐方法"] = "K线多因子模型"
        fallback_pool = []
        if spot_df is not None and not spot_df.empty:
            # 先应用价格过滤
            filtered = _build_filtered_df(spot_df, market, price_range)
            if filtered is not None and not filtered.empty:
                fallback_pool = filtered['代码'].head(50).tolist()
            elif market == "sh":
                fallback_pool = spot_df[spot_df['代码'].str.startswith('6')]['代码'].head(50).tolist()
            elif market == "sz":
                fallback_pool = spot_df[spot_df['代码'].str.startswith(('0', '3'))]['代码'].head(50).tolist()
            else:
                fallback_pool = spot_df['代码'].head(50).tolist()
        candidates = _fallback_kline_score(fallback_pool, top_n)

    # 第四步：最终兜底（尊重价格范围过滤）
    if not candidates:
        result["推荐方法"] = "基础股票池"
        # 构建与评分函数一致的过滤条件
        filtered_df = _build_filtered_df(spot_df, market, price_range)
        if filtered_df is not None and not filtered_df.empty:
            for _, row in filtered_df.head(top_n).iterrows():
                c = str(row.get('代码', ''))
                n = str(row.get('名称', ''))
                if n == 'nan' or n == '':
                    n = c
                candidates.append({
                    "代码": c, "名称": n, "综合评分": 50, "评级": "--",
                    "最新价": round(float(row.get('最新价', 0) or 0), 2),
                    "涨跌幅": round(float(row.get('涨跌幅', 0) or 0), 2),
                    "换手率": _safe_float(row.get('换手率')),
                    "市盈率": _safe_float(row.get('市盈率-动态')),
                })

    # 确保至少有 top_n 个候选（同样尊重价格范围）
    if len(candidates) < top_n:
        filtered_df = _build_filtered_df(spot_df, market, price_range)
        if filtered_df is not None and not filtered_df.empty:
            existing_codes = {c.get('代码', '') for c in candidates}
            for _, row in filtered_df.iterrows():
                if len(candidates) >= top_n:
                    break
                c = str(row.get('代码', ''))
                if c not in existing_codes:
                    n = str(row.get('名称', ''))
                    if n == 'nan' or n == '':
                        n = c
                    candidates.append({
                        "代码": c, "名称": n,
                        "综合评分": 30, "评级": "观望",
                        "最新价": round(float(row.get('最新价', 0) or 0), 2),
                        "涨跌幅": round(float(row.get('涨跌幅', 0) or 0), 2),
                        "换手率": _safe_float(row.get('换手率')),
                        "市盈率": _safe_float(row.get('市盈率-动态')),
                    })

    # 第五步：生成多维度推荐理由
    for c in candidates[:top_n]:
        code = c.get('代码', '')
        chg = c.get('涨跌幅', 0)
        pe = c.get('市盈率')
        pb = c.get('市净率')
        turnover = c.get('换手率')
        chg_60d = c.get('60日涨跌幅')
        chg_ytd = c.get('年初至今涨跌幅')
        score = c.get('综合评分', 0)

        reason_parts = []

        # 1. 中期趋势分析（60日 + 年初至今）
        trend_desc = []
        if chg_60d is not None:
            if chg_60d > 20:
                trend_desc.append(f"近60日上涨{chg_60d:+.1f}%")
            elif chg_60d > 5:
                trend_desc.append(f"近60日上涨{chg_60d:+.1f}%")
            elif chg_60d > -5:
                trend_desc.append(f"近60日横盘整理({chg_60d:+.1f}%)")
            else:
                trend_desc.append(f"近60日回调{chg_60d:.1f}%")
        if chg_ytd is not None:
            if chg_ytd > 10:
                trend_desc.append(f"年内累计上涨{chg_ytd:+.1f}%")
            elif chg_ytd > -10:
                trend_desc.append(f"年内表现平稳({chg_ytd:+.1f}%)")
            else:
                trend_desc.append(f"年内跌幅{chg_ytd:.1f}%")
        if trend_desc:
            reason_parts.append("中期趋势：" + "，".join(trend_desc))

        # 2. 短期动量分析（当日涨跌幅，权重已降低）
        if chg > 5:
            reason_parts.append(f"今日强势上涨{chg:+.2f}%，短期动能充足")
        elif chg > 2:
            reason_parts.append(f"今日上涨{chg:+.2f}%，走势偏多")
        elif chg > 0:
            reason_parts.append(f"今日微涨{chg:+.2f}%，平稳运行")
        elif chg > -2:
            reason_parts.append(f"今日微跌{chg:.2f}%，窄幅震荡")
        elif chg > -5:
            reason_parts.append(f"今日下跌{chg:.2f}%，短期承压")
        else:
            reason_parts.append(f"今日跌幅{chg:.2f}%，注意短期风险")

        # 3. 估值分析（市盈率 + 市净率）
        val_desc = []
        if pe is not None and pe > 0:
            if pe < 15:
                val_desc.append(f"市盈率{pe:.1f}处于低估区间")
            elif pe < 25:
                val_desc.append(f"市盈率{pe:.1f}估值合理偏低")
            elif pe < 40:
                val_desc.append(f"市盈率{pe:.1f}估值中性")
            elif pe < 60:
                val_desc.append(f"市盈率{pe:.1f}估值偏高")
            else:
                val_desc.append(f"市盈率{pe:.1f}估值较高")
        if pb is not None and pb > 0:
            if pb < 1.5:
                val_desc.append(f"市净率{pb:.2f}低于净资产")
            elif pb < 3:
                val_desc.append(f"市净率{pb:.2f}合理")
            elif pb < 5:
                val_desc.append(f"市净率{pb:.2f}偏高")
        if val_desc:
            reason_parts.append("估值：" + "，".join(val_desc))

        # 4. 资金活跃度分析
        if turnover is not None and turnover > 0:
            if turnover > 8:
                reason_parts.append(f"换手率{turnover:.1f}%较高，资金高度活跃")
            elif turnover > 3:
                reason_parts.append(f"换手率{turnover:.1f}%适中，交投活跃")
            else:
                reason_parts.append(f"换手率{turnover:.1f}%偏低，筹码稳定")

        # 5. 综合评估
        if score >= 70:
            reason_parts.append("综合评分优秀，多维度指标表现突出，建议重点关注")
        elif score >= 55:
            reason_parts.append("综合评分良好，整体表现稳健，适合纳入观察池")
        elif score >= 40:
            reason_parts.append("综合评分中等，可适当关注等待更好时机")

        c["推荐理由"] = "；".join(reason_parts) if reason_parts else "数据不足，建议进一步分析"

        # 仓位建议（基于价格和振幅估算ATR）
        price_val = c.get('最新价', 0)
        if price_val > 0:
            estimated_atr = price_val * 0.03
            pos_advice = _calc_position_advice(price_val, estimated_atr, 30)
            c["仓位建议"] = pos_advice["suggested_position_pct"]
            c["止损参考"] = f"{pos_advice['stop_loss']}元"
            c["止盈参考"] = f"{pos_advice['take_profit']}元"

    result["推荐股票"] = candidates[:top_n]

    # 盘中实时风险监控
    risk_monitor = _realtime_risk_monitor(spot_df, candidates[:top_n])
    result["实时风险监控"] = risk_monitor

    # 极端行情应对机制
    extreme_handler = _extreme_market_handler(spot_df, candidates[:top_n])
    result["极端行情应对"] = extreme_handler

    # 多策略资金分配优化
    multi_strategy = _multi_strategy_allocation(candidates[:top_n], spot_df)
    result["多策略分配"] = multi_strategy

    # 相关性/集中度检查
    diversity_check = _check_portfolio_diversity(candidates[:top_n], spot_df)
    if diversity_check:
        result["风险提示"] = diversity_check

    # 行业轮动分析
    rotation = _analyze_industry_rotation(spot_df)
    result["行业轮动"] = rotation

    # 市场微观结构分析
    microstructure = _microstructure_analysis(spot_df)
    result["微观结构分析"] = microstructure

    # 行为金融学因子分析
    behavioral = _behavioral_finance_factors(spot_df)
    result["行为金融分析"] = behavioral

    # 事件驱动分析
    event_driven = _event_driven_analysis(spot_df)
    result["事件驱动分析"] = event_driven

    # 跨市场联动分析
    cross_market = _cross_market_analysis(spot_df)
    result["跨市场分析"] = cross_market

    # 自适应动态策略调整（整合所有分析结果）
    adaptive = _adaptive_strategy_adjustment(spot_df, market_regime, microstructure, behavioral, cross_market)
    result["自适应策略"] = adaptive

    # 实盘信号跟踪与准确率统计
    signal_tracking = _signal_tracking(result["推荐股票"], spot_df)
    result["信号跟踪"] = signal_tracking

    result["分析说明"] = f"基于{market_label}实时行情数据，通过中期趋势、估值水平、资金活跃度、短期动量、风险控制五维度多因子模型综合评分，为您筛选出{len(result['推荐股票'])}只优质标的。"

    # 交易复盘报告生成：综合所有分析维度，生成结构化复盘报告
    review_report = _generate_trading_review(result)
    result["复盘报告"] = review_report

    return result


def _analyze_industry_rotation(spot_df):
    """
    行业轮动分析：基于实时行情数据识别行业强弱和轮动信号
    通过行业平均涨跌幅、行业内个股分化度、行业动量变化判断轮动方向
    返回: {强势行业, 弱势行业, 轮动信号, 行业排名, 分析说明}
    """
    import pandas as pd
    import numpy as np

    if spot_df is None or spot_df.empty:
        return {"强势行业": [], "弱势行业": [], "轮动信号": "数据不足", "分析说明": []}

    analysis = {
        "强势行业": [],
        "弱势行业": [],
        "轮动信号": "无明显轮动",
        "行业排名": [],
        "分析说明": [],
    }

    try:
        df = spot_df.copy()
        if '行业' not in df.columns or '涨跌幅' not in df.columns:
            return analysis

        # 过滤无效行业
        df = df[df['行业'].notna()]
        df = df[df['行业'] != '']
        df = df[df['行业'] != 'nan']

        # 计算各行业平均涨跌幅
        sector_chg = df.groupby('行业')['涨跌幅'].apply(
            lambda x: pd.to_numeric(x, errors='coerce').mean()
        ).dropna().sort_values(ascending=False)

        if sector_chg.empty:
            return analysis

        # 计算行业内个股数量（确保样本量足够）
        sector_count = df.groupby('行业').size()
        valid_sectors = sector_count[sector_count >= 3].index
        sector_chg = sector_chg[sector_chg.index.isin(valid_sectors)]

        # 行业排名
        for sector, chg in sector_chg.items():
            count = sector_count.get(sector, 0)
            analysis["行业排名"].append({
                "行业": sector,
                "平均涨跌幅": round(float(chg), 2),
                "样本数": int(count),
            })

        # 强势行业（涨幅前5）
        top5 = sector_chg.head(5)
        for sector, chg in top5.items():
            analysis["强势行业"].append({
                "行业": sector,
                "平均涨跌幅": round(float(chg), 2),
            })

        # 弱势行业（跌幅前5）
        bottom5 = sector_chg.tail(5)
        for sector, chg in bottom5.items():
            analysis["弱势行业"].append({
                "行业": sector,
                "平均涨跌幅": round(float(chg), 2),
            })

        # ===== 轮动信号检测 =====
        signals = []

        # 1. 行业分化度：强势行业与弱势行业的差距
        if len(top5) > 0 and len(bottom5) > 0:
            top_avg = top5.mean()
            bottom_avg = bottom5.mean()
            spread = top_avg - bottom_avg

            if spread > 5:
                signals.append(f"行业分化显著（强弱差{spread:.1f}%），市场呈现明显的结构性行情")
            elif spread > 2:
                signals.append(f"行业有一定分化（强弱差{spread:.1f}%），存在结构性机会")
            else:
                signals.append(f"行业分化较小（强弱差{spread:.1f}%），市场趋于同涨同跌")

        # 2. 行业风格判断：成长 vs 价值
        growth_sectors = ["电子", "计算机", "传媒", "通信", "医药生物", "电气设备", "国防军工"]
        value_sectors = ["银行", "非银金融", "房地产", "钢铁", "采掘", "建筑装饰", "建筑材料"]

        growth_chg = sector_chg[sector_chg.index.isin(growth_sectors)]
        value_chg = sector_chg[sector_chg.index.isin(value_sectors)]

        if len(growth_chg) > 0 and len(value_chg) > 0:
            growth_avg = growth_chg.mean()
            value_avg = value_chg.mean()
            style_diff = growth_avg - value_avg

            if style_diff > 2:
                signals.append(f"成长风格占优（成长-价值={style_diff:.1f}%），资金偏好高成长赛道")
                analysis["轮动信号"] = "成长风格主导"
            elif style_diff < -2:
                signals.append(f"价值风格占优（价值-成长={abs(style_diff):.1f}%），资金转向防御性板块")
                analysis["轮动信号"] = "价值风格主导"
            else:
                signals.append("成长与价值风格均衡，无明显风格偏向")

        # 3. 防御性行业 vs 周期性行业
        defensive_sectors = ["食品饮料", "医药生物", "公用事业", "农林牧渔"]
        cyclical_sectors = ["有色金属", "化工", "钢铁", "汽车", "机械设备"]

        defensive_chg = sector_chg[sector_chg.index.isin(defensive_sectors)]
        cyclical_chg = sector_chg[sector_chg.index.isin(cyclical_sectors)]

        if len(defensive_chg) > 0 and len(cyclical_chg) > 0:
            def_avg = defensive_chg.mean()
            cyc_avg = cyclical_chg.mean()
            dc_diff = def_avg - cyc_avg

            if dc_diff > 2:
                signals.append("防御性行业走强，市场风险偏好下降，资金寻求确定性")
                if analysis["轮动信号"] == "无明显轮动":
                    analysis["轮动信号"] = "防御风格轮动"
            elif dc_diff < -2:
                signals.append("周期性行业走强，市场风险偏好上升，资金追逐弹性")
                if analysis["轮动信号"] == "无明显轮动":
                    analysis["轮动信号"] = "周期风格轮动"

        # 4. 行业动量持续性判断
        if '60日涨跌幅' in df.columns:
            sector_chg_60d = df.groupby('行业')['60日涨跌幅'].apply(
                lambda x: pd.to_numeric(x, errors='coerce').mean()
            ).dropna()

            # 对比当日涨跌与60日趋势
            common_sectors = sector_chg.index.intersection(sector_chg_60d.index)
            if len(common_sectors) > 0:
                momentum_continue = 0
                momentum_reverse = 0
                for s in common_sectors:
                    if (sector_chg[s] > 0 and sector_chg_60d[s] > 0) or (sector_chg[s] < 0 and sector_chg_60d[s] < 0):
                        momentum_continue += 1
                    else:
                        momentum_reverse += 1

                total = momentum_continue + momentum_reverse
                if total > 0:
                    continue_ratio = momentum_continue / total * 100
                    if continue_ratio > 70:
                        signals.append(f"行业动量延续性强（{continue_ratio:.0f}%行业延续趋势），趋势策略有效")
                    elif continue_ratio < 30:
                        signals.append(f"行业动量反转明显（{continue_ratio:.0f}%行业延续趋势），市场风格可能切换")

        analysis["分析说明"] = signals

    except Exception as e:
        analysis["分析说明"] = [f"行业轮动分析异常: {str(e)}"]

    return analysis


def _check_portfolio_diversity(stocks, spot_df):
    """
    检查推荐组合的行业集中度风险
    返回风险提示文本，如果分散度良好则返回None
    """
    if not stocks or len(stocks) < 3:
        return None

    # 获取每只股票的行业
    industry_count = {}
    stock_industries = {}
    for s in stocks:
        code = s.get('代码', '')
        industry = None
        if spot_df is not None and not spot_df.empty:
            row = spot_df[spot_df['代码'] == code]
            if not row.empty:
                industry = str(row.iloc[0].get('行业', ''))
                if industry in ('nan', '', 'None'):
                    industry = None
        if industry:
            industry_count[industry] = industry_count.get(industry, 0) + 1
            stock_industries[code] = industry

    if not industry_count:
        return None

    total = len(stocks)
    max_industry = max(industry_count, key=industry_count.get)
    max_count = industry_count[max_industry]
    concentration = max_count / total * 100

    if concentration > 50:
        same_industry_stocks = [s for s in stocks if stock_industries.get(s.get('代码', '')) == max_industry]
        codes_str = '、'.join([s.get('名称', s.get('代码', '')) for s in same_industry_stocks])
        return f"行业集中度预警：{max_industry}行业占比{concentration:.0f}%（{codes_str}），建议跨行业分散配置以降低单一行业风险"

    if concentration > 30:
        return f"行业集中度提示：{max_industry}行业占比{concentration:.0f}%，建议适当分散到其他行业"

    return None


def _build_filtered_df(spot_df, market, price_range):
    """构建与_fast_score_stocks一致的过滤条件，供兜底逻辑复用"""
    import pandas as pd
    if spot_df is None or spot_df.empty:
        return None
    df = spot_df.copy()
    if market == "sh":
        df = df[df['代码'].str.startswith('6')]
    elif market == "sz":
        df = df[df['代码'].str.startswith(('0', '3'))]
    df = df[df['最新价'].notna() & (df['最新价'] > 0)]
    df = df[~df['名称'].str.contains('ST', na=False)]
    if price_range:
        if isinstance(price_range, (list, tuple)) and len(price_range) == 2:
            lo, hi = float(price_range[0]), float(price_range[1])
            df = df[(df['最新价'] >= lo) & (df['最新价'] <= hi)]
        else:
            price_ranges = {
                "0-2": (0, 2), "3-5": (3, 5), "6-10": (6, 10),
                "11-20": (11, 20), "21-50": (21, 50),
                "51-100": (51, 100), "100+": (100, float('inf')),
            }
            pr = price_ranges.get(price_range)
            if pr:
                lo, hi = pr
                df = df[(df['最新价'] >= lo) & (df['最新价'] <= hi)]
    return df


def _safe_float(val):
    """安全转换为float，NaN/None返回None（JSON中为null）"""
    import pandas as pd
    if val is None:
        return None
    if pd.isna(val):
        return None
    try:
        v = float(val)
        if v != v:  # NaN check
            return None
        return round(v, 2)
    except (ValueError, TypeError):
        return None


# ==================== 联网搜索能力 ====================

def analyze_search_need(query, conversation_history=""):
    """
    分析用户查询是否需要联网搜索（金融/股票领域）
    返回: {need_web_search, search_reason, confidence}
    """
    instruction = """你是一个专注于A股量化交易的查询分析专家。请分析用户的查询，判断是否需要联网搜索来获取最新、最准确的信息。

需要联网搜索的情况包括：
1. 实时行情 - 包含"最新价"、"实时"、"现在"、"当前"、"盘中"、"今天"等词汇
2. 新闻事件 - 包含"新闻"、"公告"、"消息"、"政策"、"利好"、"利空"等
3. 财报数据 - 包含"财报"、"季报"、"年报"、"业绩"、"营收"、"利润"等最新财务数据
4. 研报观点 - 包含"研报"、"评级"、"目标价"、"分析师"、"推荐"等
5. 市场情绪 - 包含"情绪"、"热度"、"讨论"、"舆论"、"大家都在说"等
6. 行业动态 - 包含"行业"、"板块"、"政策"、"监管"、"新规"等
7. 宏观经济 - 包含"GDP"、"CPI"、"利率"、"降息"、"加息"、"央行"等
8. 龙虎榜/资金 - 包含"龙虎榜"、"北向资金"、"主力"、"游资"、"机构"等

不需要联网搜索的情况：
- 用户已有明确数据，只需分析计算（如"帮我算一下PE"、"回测这个策略"）
- 纯技术分析（如"画一下均线"、"MACD金叉了吗"）
- 系统内已有数据可回答的问题

请返回JSON格式：
{
    "need_web_search": true/false,
    "search_reason": "需要搜索的原因（中文，简洁）",
    "confidence": 0.0-1.0
}"""

    prompt = f"""### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history if conversation_history else "无"}

### 用户查询 ###
{query}

### 分析结果 ###
"""
    return call_llm(instruction, prompt)


def rewrite_search_query(query, search_type="综合"):
    """
    将用户查询改写为适合金融搜索引擎检索的形式
    返回: {rewritten_query, search_keywords, search_intent, suggested_sources}
    """
    instruction = """你是一个专注于金融信息检索的搜索优化专家。请将用户的查询改写为更适合搜索引擎检索的形式。

改写技巧：
1. 添加股票代码 - 如"600519 贵州茅台"、"000858 五粮液"
2. 添加时间范围 - 如"2025年"、"今天"、"本周"、"最新"
3. 使用关键词组合 - 将长句拆分为核心关键词
4. 明确搜索意图 - 行情查询/新闻搜索/研报查找/政策追踪
5. 去除口语化表达 - 转换为标准金融搜索词
6. 添加金融术语 - 增加同义词或专业词汇（如"市盈率 PE"）

请返回JSON格式：
{
    "rewritten_query": "改写后的搜索查询",
    "search_keywords": ["关键词1", "关键词2", "关键词3"],
    "search_intent": "搜索意图（中文）",
    "suggested_sources": ["建议搜索的网站类型"]
}"""

    prompt = f"""### 指令 ###
{instruction}

### 原始查询 ###
{query}

### 搜索类型 ###
{search_type}

### 改写结果 ###
"""
    return call_llm(instruction, prompt)


def plan_search_strategy(query, search_type="综合"):
    """
    为用户的金融查询制定详细的搜索策略
    返回: {primary_keywords, extended_keywords, search_platforms, time_range}
    """
    current_date = datetime.now().strftime('%Y年%m月%d日')
    instruction = f"""你是一个金融信息搜索策略专家。请为用户的查询制定详细的搜索策略。

当前日期：{current_date}

搜索策略包括：
1. 主要搜索词 - 核心关键词（股票代码、公司名称、关键事件）
2. 扩展搜索词 - 相关词汇和同义词（行业术语、关联概念）
3. 搜索平台 - 推荐的金融信息平台
4. 时间范围 - 具体的搜索时间范围

可选的金融搜索平台：
- 东方财富(eastmoney.com) - 综合财经资讯
- 雪球(xueqiu.com) - 投资者社区讨论
- 同花顺(10jqka.com.cn) - 行情与新闻
- 巨潮资讯(cninfo.com.cn) - 官方公告
- 上交所/深交所官网 - 官方数据
- 财联社(cls.cn) - 快讯
- 华尔街见闻(wallstreetcn.com) - 宏观与市场
- 新浪财经(finance.sina.com.cn) - 综合财经

请返回JSON格式：
{{
    "primary_keywords": ["主要关键词"],
    "extended_keywords": ["扩展关键词"],
    "search_platforms": ["搜索平台"],
    "time_range": "具体的时间范围"
}}"""

    prompt = f"""### 指令 ###
{instruction}

### 用户查询 ###
{query}

### 搜索类型 ###
{search_type}

### 搜索策略 ###
"""
    return call_llm(instruction, prompt)


# ==================== 智能推荐助手（对话式） ====================

def parse_recommend_query(text, existing_context=None):
    """
    从自然语言中解析推荐参数
    返回: {market, price_range, top_n, risk_level, preference, extra_dimensions, is_followup, intent}
    price_range 为 (lo, hi) 元组或空字符串
    """
    import re
    params = {
        "market": "",
        "price_range": "",
        "top_n": 5,
        "risk_level": "中等",
        "preference": "",
        "extra_dimensions": [],
        "is_followup": False,
        "intent": "recommend",
    }

    # 合并上下文
    if existing_context:
        for k in ["market", "price_range", "top_n", "risk_level", "preference"]:
            if k in existing_context:
                params[k] = existing_context[k]
        params["extra_dimensions"] = list(existing_context.get("extra_dimensions", []))

    # 判断是否为追问（补充维度）
    followup_patterns = [
        r'(再|还|也|另外|额外|加上|补充).*(根据|基于|用|从|按).*(评估|分析|推荐|看)',
        r'(加上|补充|增加|追加).*(维度|指标|因子|数据)',
        r'(再|还).*(看看|算算|评估|分析)',
    ]
    for pat in followup_patterns:
        if re.search(pat, text):
            params["is_followup"] = True
            break

    # 解析市场
    if re.search(r'上证|沪市|上海|6开头', text):
        params["market"] = "sh"
    elif re.search(r'深证|深市|深圳|0开头|3开头|创业板', text):
        params["market"] = "sz"
    elif re.search(r'全市场|全部|所有|不限', text):
        params["market"] = ""

    # 解析价格范围（生成连续区间元组）
    price_match = re.search(r'(\d+)\s*[-~至到]\s*(\d+)\s*元', text)
    if price_match:
        lo, hi = int(price_match.group(1)), int(price_match.group(2))
        params["price_range"] = (lo, hi)
    elif re.search(r'(\d+)元以下|低于(\d+)元|不超过(\d+)元|(\d+)元以内', text):
        m = re.search(r'(\d+)元以下|低于(\d+)元|不超过(\d+)元|(\d+)元以内', text)
        nums = re.findall(r'(\d+)', m.group())
        if nums:
            val = int(nums[0])
            params["price_range"] = (0, val)
    elif re.search(r'(\d+)元以上|高于(\d+)元|超过(\d+)元', text):
        m = re.search(r'(\d+)元以上|高于(\d+)元|超过(\d+)元', text)
        nums = re.findall(r'(\d+)', m.group())
        if nums:
            val = int(nums[0])
            params["price_range"] = (val, float('inf'))

    # 解析推荐数量
    num_match = re.search(r'(推荐|选|筛选|找|要|给我|帮我).*?(\d+)\s*[只个支]', text)
    if num_match:
        params["top_n"] = int(num_match.group(2))
    elif re.search(r'(\d+)\s*[只个支]', text):
        m = re.search(r'(\d+)\s*[只个支]', text)
        params["top_n"] = int(m.group(1))

    # 解析风险偏好
    if re.search(r'低风险|稳健|保守|安全', text):
        params["risk_level"] = "低"
    elif re.search(r'高风险|激进|进取|冒险', text):
        params["risk_level"] = "高"

    # 解析板块偏好
    sector_keywords = {
        "科技": r'科技|AI|人工智能|软件|互联网|IT',
        "消费": r'消费|零售|食品|饮料',
        "医药": r'医药|医疗|制药|生物|健康',
        "金融": r'金融|银行|保险|券商|证券',
        "新能源": r'新能源|光伏|风电|锂电|储能|太阳能',
        "白酒": r'白酒|酒',
        "半导体": r'半导体|芯片|集成电路',
        "军工": r'军工|国防|航天|航空',
        "汽车": r'汽车|新能源车|电动车|整车',
        "家电": r'家电|电器|家居',
        "电力": r'电力|发电|电网',
        "地产": r'地产|房地产|房产|物业',
    }
    for sector, pat in sector_keywords.items():
        if re.search(pat, text):
            params["preference"] = sector
            break

    # 解析额外维度
    dimension_keywords = {
        "K线技术分析": r'K线|k线|技术分析|技术面|均线|MACD|KDJ|RSI|布林|BOLL',
        "成交量分析": r'成交量|放量|缩量|量能|堆量',
        "资金流向": r'资金流向|主力资金|北向资金|大单|净流入|净流出',
        "财务基本面": r'财务|基本面|营收|利润|ROE|净资产|负债|现金流',
        "行业对比": r'行业对比|同行|板块对比|行业排名',
        "历史波动率": r'历史波动|波动率|标准差|回撤',
    }
    for dim_name, pat in dimension_keywords.items():
        if re.search(pat, text):
            if dim_name not in params["extra_dimensions"]:
                params["extra_dimensions"].append(dim_name)

    # 限制数量
    params["top_n"] = max(1, min(params["top_n"], 20))

    return params


def ai_recommend_assist(query, context=None):
    """
    智能推荐助手：对话式推荐
    支持自然语言输入、多轮对话补充维度、个股分析
    """
    import re

    # 解析参数
    parsed = parse_recommend_query(query, context)

    # 格式化价格标签
    def _price_label(pr):
        if isinstance(pr, (list, tuple)) and len(pr) == 2:
            lo, hi = pr
            if hi == float('inf'):
                return f"{lo}元以上"
            return f"{lo}-{hi}元"
        return pr if pr else "不限"

    price_label = _price_label(parsed["price_range"])
    market_label = {"sh": "上证指数", "sz": "深证成指"}.get(parsed["market"], "全市场")

    # ===== 意图识别：判断是推荐、分析还是追问 =====
    # 检测是否包含股票代码（6位数字）
    code_match = re.search(r'\b(\d{6})\b', query)
    # 检测是否为个股分析请求
    analyze_patterns = [
        r'(分析|评估|看看|诊断|解读).*(股票|个股|这只|那个)',
        r'(帮我|给我|请).*(分析|评估|看看|诊断|解读)',
        r'(怎么看|怎么样|如何).*(股票|这只|那个)',
        r'分析.*\d{6}',
    ]
    is_analyze = False
    if code_match:
        is_analyze = True
    else:
        for pat in analyze_patterns:
            if re.search(pat, query):
                is_analyze = True
                break

    # 如果上下文是分析模式，追问也按分析处理
    if context and context.get("intent") == "analyze":
        is_analyze = True
        parsed["is_followup"] = True

    # ===== 个股分析模式 =====
    if is_analyze:
        return _handle_stock_analysis(query, parsed, context, code_match)

    # ===== 推荐模式 =====
    reply_parts = []

    if parsed["is_followup"] and context:
        new_dims = parsed["extra_dimensions"]
        old_dims = context.get("extra_dimensions", [])
        added_dims = [d for d in new_dims if d not in old_dims]
        if added_dims:
            reply_parts.append(f"好的，我在原有维度基础上，额外加入 {', '.join(added_dims)} 维度重新评估。")
        else:
            reply_parts.append("好的，我按照你的补充要求重新评估。")
    else:
        reply_parts.append(f"正在从{market_label}中为你筛选，价格范围：{price_label}，推荐{parsed['top_n']}只股票。")

    try:
        result = ai_recommend_stocks(
            preference=parsed["preference"],
            risk_level=parsed["risk_level"],
            top_n=parsed["top_n"],
            price_range=parsed["price_range"],
            market=parsed["market"],
        )
        stocks = result.get("推荐股票", [])
    except Exception as e:
        return {
            "reply": f"推荐过程出错：{str(e)}",
            "parsed_params": parsed,
            "recommend_result": [],
            "context": parsed,
        }

    if parsed["extra_dimensions"] and stocks:
        stocks = _apply_extra_dimensions(stocks, parsed["extra_dimensions"])

    default_dims = ["中期趋势", "估值水平", "资金活跃度", "短期动量", "风险控制"]
    all_dims = default_dims + parsed["extra_dimensions"]
    dim_desc = "、".join(all_dims)

    reply_parts.append(f"已通过{dim_desc}等维度综合评估，为你找到以下{len(stocks)}只标的：")

    # 行业集中度风险提示
    risk_warning = result.get("风险提示", "")
    if risk_warning:
        reply_parts.append(f"\n  {risk_warning}")

    new_context = dict(parsed)

    return {
        "reply": "\n".join(reply_parts),
        "parsed_params": {
            "market": market_label if parsed["market"] else "全市场",
            "price_range": price_label,
            "top_n": parsed["top_n"],
            "risk_level": parsed["risk_level"],
            "preference": parsed["preference"] if parsed["preference"] else "",
            "extra_dimensions": parsed["extra_dimensions"],
        },
        "recommend_result": stocks,
        "context": new_context,
    }


def _handle_stock_analysis(query, parsed, context, code_match):
    """处理个股分析请求"""
    import re

    reply_parts = []
    stock_code = None
    stock_name = ""

    # 提取股票代码
    if code_match:
        stock_code = code_match.group(1)
    elif context and context.get("stock_code"):
        stock_code = context["stock_code"]

    # 尝试从行情数据中查找股票名称
    if stock_code:
        try:
            from data_utils import _get_spot_df
            spot_df = _get_spot_df()
            if spot_df is not None and not spot_df.empty:
                match_row = spot_df[spot_df['代码'] == stock_code]
                if not match_row.empty:
                    stock_name = str(match_row.iloc[0].get('名称', ''))
        except Exception:
            pass

    if not stock_code:
        return {
            "reply": "请提供具体的股票代码（6位数字），我才能为你进行分析。",
            "parsed_params": parsed,
            "recommend_result": [],
            "context": parsed,
        }

    display_name = f"{stock_name}({stock_code})" if stock_name else stock_code

    if parsed.get("is_followup") and context:
        reply_parts.append(f"好的，我继续对 {display_name} 进行补充分析。")
    else:
        reply_parts.append(f"正在对 {display_name} 进行多维度分析...")

    # 获取K线数据
    kline_data = None
    try:
        from data_utils import get_stock_kline
        kline_data = get_stock_kline(stock_code, days=120)
    except Exception as e:
        reply_parts.append(f"获取K线数据时出错：{str(e)}")

    # 获取实时行情
    spot_info = {}
    try:
        from data_utils import _get_spot_df
        spot_df = _get_spot_df()
        if spot_df is not None and not spot_df.empty:
            match_row = spot_df[spot_df['代码'] == stock_code]
            if not match_row.empty:
                row = match_row.iloc[0]
                spot_info = {
                    "最新价": row.get('最新价'),
                    "涨跌幅": row.get('涨跌幅'),
                    "换手率": row.get('换手率'),
                    "量比": row.get('量比'),
                    "市盈率-动态": row.get('市盈率-动态'),
                    "市净率": row.get('市净率'),
                    "振幅": row.get('振幅'),
                    "60日涨跌幅": row.get('60日涨跌幅'),
                    "年初至今涨跌幅": row.get('年初至今涨跌幅'),
                }
    except Exception:
        pass

    # 构建分析结果
    analysis = _build_stock_analysis(stock_code, stock_name, kline_data, spot_info, parsed)

    reply_parts.append(analysis["summary"])

    if analysis.get("ai_summary"):
        reply_parts.append("")
        reply_parts.append("【AI 综合研判】")
        reply_parts.append(analysis["ai_summary"])

    new_context = dict(parsed)
    new_context["intent"] = "analyze"
    new_context["stock_code"] = stock_code
    new_context["stock_name"] = stock_name

    return {
        "reply": "\n".join(reply_parts),
        "parsed_params": {
            "market": "",
            "price_range": "",
            "top_n": 0,
            "risk_level": parsed.get("risk_level", ""),
            "preference": "",
            "extra_dimensions": parsed.get("extra_dimensions", []),
        },
        "recommend_result": [],
        "analysis_result": analysis,
        "context": new_context,
    }


def _multi_timeframe_analysis(kline_data):
    """
    多时间框架分析：在日线、周线、月线三个周期上分析趋势一致性
    基于日线数据合成周线和月线，检测多周期共振与背离
    返回: {各周期趋势, 共振判断, 背离检测, 综合建议}
    """
    import pandas as pd
    import numpy as np

    if kline_data is None or kline_data.empty:
        return {"各周期趋势": {}, "共振判断": "数据不足", "背离检测": [], "综合建议": []}

    analysis = {
        "各周期趋势": {},
        "共振判断": "数据不足",
        "背离检测": [],
        "综合建议": [],
    }

    try:
        df = kline_data.copy()
        close_col = '收盘' if '收盘' in df.columns else ('close' if 'close' in df.columns else None)
        if not close_col or len(df) < 20:
            return analysis

        df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
        df = df.dropna(subset=[close_col])
        if len(df) < 20:
            return analysis

        close = df[close_col]

        # ===== 1. 日线趋势分析 =====
        daily_trend = _calc_timeframe_trend(close, "日线")
        analysis["各周期趋势"]["日线"] = daily_trend

        # ===== 2. 周线趋势分析 =====
        if len(close) >= 25:
            weekly_close = close.iloc[::5]  # 每5个交易日取一个
            if len(weekly_close) >= 5:
                weekly_trend = _calc_timeframe_trend(weekly_close, "周线")
                analysis["各周期趋势"]["周线"] = weekly_trend

        # ===== 3. 月线趋势分析 =====
        if len(close) >= 60:
            monthly_close = close.iloc[::20]  # 每20个交易日取一个
            if len(monthly_close) >= 3:
                monthly_trend = _calc_timeframe_trend(monthly_close, "月线")
                analysis["各周期趋势"]["月线"] = monthly_trend

        # ===== 4. 多周期共振判断 =====
        trends = analysis["各周期趋势"]
        trend_directions = {}
        for tf, info in trends.items():
            trend_directions[tf] = info.get("趋势方向", "未知")

        # 统计各周期方向
        up_count = sum(1 for d in trend_directions.values() if d == "上涨")
        down_count = sum(1 for d in trend_directions.values() if d == "下跌")
        total_count = len(trend_directions)

        if total_count >= 2:
            if up_count == total_count:
                analysis["共振判断"] = "多周期共振向上"
                analysis["综合建议"].append("日线、周线、月线趋势一致向上，多头排列，中期上涨趋势确立，适合顺势做多")
            elif down_count == total_count:
                analysis["共振判断"] = "多周期共振向下"
                analysis["综合建议"].append("日线、周线、月线趋势一致向下，空头排列，中期下跌趋势确立，建议回避或轻仓")
            elif up_count > down_count:
                analysis["共振判断"] = "偏多但存在分歧"
                analysis["综合建议"].append(f"多数周期看涨（{up_count}/{total_count}），但存在分歧周期，趋势不够稳固，注意风险控制")
            elif down_count > up_count:
                analysis["共振判断"] = "偏空但存在分歧"
                analysis["综合建议"].append(f"多数周期看跌（{down_count}/{total_count}），但存在分歧周期，可能有反弹机会但不宜重仓")
            else:
                analysis["共振判断"] = "多周期方向分歧"
                analysis["综合建议"].append("各周期方向不一致，市场处于震荡格局，建议观望或短线操作")

        # ===== 5. 背离检测 =====
        if "日线" in trends and "周线" in trends:
            daily_dir = trends["日线"].get("趋势方向", "")
            weekly_dir = trends["周线"].get("趋势方向", "")
            if daily_dir == "上涨" and weekly_dir == "下跌":
                analysis["背离检测"].append("日线上涨但周线下跌：短期反弹可能受制于中期下降趋势，反弹高度有限")
            elif daily_dir == "下跌" and weekly_dir == "上涨":
                analysis["背离检测"].append("日线下跌但周线上涨：短期回调可能是中期上升趋势中的正常调整，关注支撑位")

        if "周线" in trends and "月线" in trends:
            weekly_dir = trends["周线"].get("趋势方向", "")
            monthly_dir = trends["月线"].get("趋势方向", "")
            if weekly_dir == "上涨" and monthly_dir == "下跌":
                analysis["背离检测"].append("周线上涨但月线下跌：中期反弹面临长期下降趋势压力，可能是熊市反弹")
            elif weekly_dir == "下跌" and monthly_dir == "上涨":
                analysis["背离检测"].append("周线下跌但月线上涨：中期回调在长期上升趋势中，可能是较好的加仓时机")

        # ===== 6. 趋势强度综合评估 =====
        strengths = []
        for tf, info in trends.items():
            strength = info.get("趋势强度", 0)
            if strength is not None:
                strengths.append(strength)

        if strengths:
            avg_strength = sum(strengths) / len(strengths)
            if avg_strength > 70:
                analysis["综合建议"].append(f"多周期平均趋势强度{avg_strength:.0f}，趋势动能充沛")
            elif avg_strength > 40:
                analysis["综合建议"].append(f"多周期平均趋势强度{avg_strength:.0f}，趋势力度一般")
            else:
                analysis["综合建议"].append(f"多周期平均趋势强度{avg_strength:.0f}，趋势动能不足，可能处于震荡或转折阶段")

    except Exception as e:
        analysis["综合建议"] = [f"多时间框架分析异常: {str(e)}"]

    return analysis


def _calc_timeframe_trend(close_series, label):
    """
    计算单个时间框架的趋势方向和强度
    基于均线排列、价格位置、近期涨跌幅综合判断
    """
    import numpy as np

    result = {
        "周期": label,
        "趋势方向": "未知",
        "趋势强度": 0,
        "分析说明": [],
    }

    try:
        close = close_series.astype(float)
        n = len(close)

        if n < 5:
            return result

        current_price = close.iloc[-1]

        # 计算均线
        ma5 = close.iloc[-5:].mean() if n >= 5 else None
        ma10 = close.iloc[-10:].mean() if n >= 10 else None
        ma20 = close.iloc[-20:].mean() if n >= 20 else None

        # 均线排列判断
        if ma5 is not None and ma10 is not None and ma20 is not None:
            if ma5 > ma10 > ma20:
                result["趋势方向"] = "上涨"
                result["趋势强度"] = 80
                result["分析说明"].append("均线多头排列（MA5>MA10>MA20），趋势强劲向上")
            elif ma5 < ma10 < ma20:
                result["趋势方向"] = "下跌"
                result["趋势强度"] = 20
                result["分析说明"].append("均线空头排列（MA5<MA10<MA20），趋势明确向下")
            elif ma5 > ma10 and ma10 < ma20:
                result["趋势方向"] = "震荡偏多"
                result["趋势强度"] = 55
                result["分析说明"].append("短期均线金叉但中长期均线压制，震荡偏多")
            elif ma5 < ma10 and ma10 > ma20:
                result["趋势方向"] = "震荡偏空"
                result["趋势强度"] = 45
                result["分析说明"].append("短期均线死叉但中长期均线支撑，震荡偏空")
            else:
                result["趋势方向"] = "震荡"
                result["趋势强度"] = 50
                result["分析说明"].append("均线交织，方向不明，处于震荡格局")
        elif ma5 is not None and ma10 is not None:
            if ma5 > ma10:
                result["趋势方向"] = "短期偏多"
                result["趋势强度"] = 60
            else:
                result["趋势方向"] = "短期偏空"
                result["趋势强度"] = 40
        else:
            # 数据不足，仅基于近期涨跌判断
            if n >= 3:
                recent_chg = (close.iloc[-1] / close.iloc[-3] - 1) * 100
                if recent_chg > 2:
                    result["趋势方向"] = "短期上涨"
                    result["趋势强度"] = 65
                elif recent_chg < -2:
                    result["趋势方向"] = "短期下跌"
                    result["趋势强度"] = 35
                else:
                    result["趋势方向"] = "横盘"
                    result["趋势强度"] = 50

        # 价格相对均线位置
        if ma20 is not None and current_price is not None:
            price_vs_ma20 = (current_price / ma20 - 1) * 100
            if price_vs_ma20 > 10:
                result["分析说明"].append(f"价格高于MA20达{price_vs_ma20:.1f}%，短期偏离较大，注意回调风险")
            elif price_vs_ma20 < -10:
                result["分析说明"].append(f"价格低于MA20达{abs(price_vs_ma20):.1f}%，超跌明显，关注反弹机会")

        # 近期涨跌幅
        if n >= 5:
            chg_5 = (close.iloc[-1] / close.iloc[-5] - 1) * 100
            if abs(chg_5) > 5:
                result["分析说明"].append(f"近5周期涨跌幅{chg_5:.1f}%，短期波动较大")

    except Exception:
        pass

    return result


def _analyze_market_sentiment(spot_info, kline_data=None):
    """
    市场情绪分析：基于可用数据推断市场情绪状态
    通过涨跌幅、振幅、换手率、量比等指标构建情绪代理指数
    返回: {情绪判断, 恐慌贪婪指数, 市场分歧度, 情绪趋势, 分析说明}
    """
    if not spot_info:
        return {"情绪判断": "数据不足", "恐慌贪婪指数": 50, "分析说明": []}

    chg = _safe_float(spot_info.get("涨跌幅"))
    amplitude = _safe_float(spot_info.get("振幅"))
    turnover = _safe_float(spot_info.get("换手率"))
    vol_ratio = _safe_float(spot_info.get("量比"))
    chg_60d = _safe_float(spot_info.get("60日涨跌幅"))
    chg_ytd = _safe_float(spot_info.get("年初至今涨跌幅"))

    analysis = {
        "情绪判断": "中性",
        "恐慌贪婪指数": 50,
        "市场分歧度": "正常",
        "情绪趋势": "平稳",
        "分析说明": [],
    }

    # ===== 1. 恐慌贪婪指数（代理） =====
    # 基于多个指标构建0-100的恐慌贪婪指数，50为中性
    # > 70 贪婪，< 30 恐慌
    fear_greed = 50
    signal_count = 0

    if chg is not None:
        if chg > 5:
            fear_greed += 20
            signal_count += 1
        elif chg > 3:
            fear_greed += 12
            signal_count += 1
        elif chg > 1:
            fear_greed += 5
            signal_count += 1
        elif chg < -5:
            fear_greed -= 20
            signal_count += 1
        elif chg < -3:
            fear_greed -= 12
            signal_count += 1
        elif chg < -1:
            fear_greed -= 5
            signal_count += 1

    if vol_ratio is not None:
        if vol_ratio > 2:
            fear_greed += 10  # 极度放量可能意味着情绪高涨
            signal_count += 1
        elif vol_ratio > 1.5:
            fear_greed += 5
            signal_count += 1
        elif vol_ratio < 0.5:
            fear_greed -= 5  # 极度缩量可能意味着恐慌观望
            signal_count += 1

    if chg_60d is not None:
        if chg_60d > 30:
            fear_greed += 15
            signal_count += 1
        elif chg_60d > 15:
            fear_greed += 8
            signal_count += 1
        elif chg_60d < -20:
            fear_greed -= 15
            signal_count += 1
        elif chg_60d < -10:
            fear_greed -= 8
            signal_count += 1

    if chg_ytd is not None:
        if chg_ytd > 20:
            fear_greed += 10
            signal_count += 1
        elif chg_ytd < -15:
            fear_greed -= 10
            signal_count += 1

    # 归一化
    if signal_count > 0:
        fear_greed = 50 + (fear_greed - 50) / signal_count

    fear_greed = max(0, min(100, round(fear_greed)))
    analysis["恐慌贪婪指数"] = fear_greed

    if fear_greed >= 75:
        analysis["情绪判断"] = "极度贪婪"
        analysis["分析说明"].append("市场情绪极度亢奋，短期可能出现过热回调风险")
    elif fear_greed >= 60:
        analysis["情绪判断"] = "偏贪婪"
        analysis["分析说明"].append("市场情绪偏乐观，追高需谨慎")
    elif fear_greed >= 40:
        analysis["情绪判断"] = "中性"
    elif fear_greed >= 25:
        analysis["情绪判断"] = "偏恐慌"
        analysis["分析说明"].append("市场情绪偏悲观，可能存在超跌反弹机会")
    else:
        analysis["情绪判断"] = "极度恐慌"
        analysis["分析说明"].append("市场情绪极度悲观，恐慌性抛售可能酝酿反转机会")

    # ===== 2. 市场分歧度分析 =====
    if amplitude is not None and turnover is not None:
        if amplitude > 8 and turnover > 8:
            analysis["市场分歧度"] = "极度分歧"
            analysis["分析说明"].append("高振幅+高换手，多空分歧白热化，短期方向选择在即")
        elif amplitude > 5 and turnover > 5:
            analysis["市场分歧度"] = "分歧较大"
            analysis["分析说明"].append("振幅和换手率偏高，市场对当前价位存在较大分歧")
        elif amplitude > 3 and turnover > 3:
            analysis["市场分歧度"] = "有一定分歧"
        elif amplitude < 2 and turnover < 2:
            analysis["市场分歧度"] = "共识较强"
            analysis["分析说明"].append("低振幅低换手，市场对当前价位认可度较高")

    # ===== 3. 情绪趋势分析 =====
    if kline_data is not None and not kline_data.empty:
        try:
            import pandas as pd
            kline = kline_data.copy()
            close_col = '收盘' if '收盘' in kline.columns else ('close' if 'close' in kline.columns else None)
            if close_col and len(kline) >= 5:
                close = kline[close_col].astype(float)
                # 连续涨跌天数
                consecutive_up = 0
                consecutive_down = 0
                for i in range(len(close) - 1, 0, -1):
                    if close.iloc[i] > close.iloc[i - 1]:
                        if consecutive_down == 0:
                            consecutive_up += 1
                        else:
                            break
                    elif close.iloc[i] < close.iloc[i - 1]:
                        if consecutive_up == 0:
                            consecutive_down += 1
                        else:
                            break
                    else:
                        break

                if consecutive_up >= 5:
                    analysis["情绪趋势"] = "持续亢奋"
                    analysis["分析说明"].append(f"连续上涨{consecutive_up}天，短期情绪持续高涨，注意获利回吐")
                elif consecutive_up >= 3:
                    analysis["情绪趋势"] = "短期偏暖"
                    analysis["分析说明"].append(f"连续上涨{consecutive_up}天，情绪温和向好")
                elif consecutive_down >= 5:
                    analysis["情绪趋势"] = "持续低迷"
                    analysis["分析说明"].append(f"连续下跌{consecutive_down}天，悲观情绪蔓延，关注超跌反弹")
                elif consecutive_down >= 3:
                    analysis["情绪趋势"] = "短期偏冷"
                    analysis["分析说明"].append(f"连续下跌{consecutive_down}天，情绪偏弱")

                # 近5日涨跌比
                if len(close) >= 5:
                    up_days = sum(1 for i in range(len(close) - 5, len(close)) if close.iloc[i] > close.iloc[i - 1])
                    if up_days >= 4:
                        analysis["分析说明"].append("近5日4天上涨，短期多头氛围浓厚")
                    elif up_days <= 1:
                        analysis["分析说明"].append("近5日仅1天上涨，短期空头氛围浓厚")
        except Exception:
            pass

    # ===== 4. 情绪面综合研判 =====
    if fear_greed >= 75 and analysis["市场分歧度"] in ("极度分歧", "分歧较大"):
        analysis["分析说明"].append("[风险提示] 贪婪情绪+高分歧，往往是阶段性顶部的特征，建议谨慎追高")
    elif fear_greed <= 25 and analysis["市场分歧度"] in ("极度分歧", "分歧较大"):
        analysis["分析说明"].append("[机会提示] 恐慌情绪+高分歧，可能是阶段性底部区域，关注企稳信号")

    return analysis


def _build_stock_analysis(stock_code, stock_name, kline_data, spot_info, parsed):
    """构建个股多维度分析报告"""
    import pandas as pd
    import numpy as np

    analysis = {
        "stock_code": stock_code,
        "stock_name": stock_name,
        "dimensions": {},
        "summary": "",
    }

    dims = []

    # 1. 实时行情维度
    if spot_info:
        price = spot_info.get("最新价")
        chg = spot_info.get("涨跌幅")
        turnover = spot_info.get("换手率")
        vol_ratio = spot_info.get("量比")
        pe = spot_info.get("市盈率-动态")
        pb = spot_info.get("市净率")
        amplitude = spot_info.get("振幅")

        realtime_parts = []
        if price is not None:
            realtime_parts.append(f"最新价：{price}元")
        if chg is not None:
            direction = "上涨" if chg > 0 else "下跌" if chg < 0 else "平盘"
            realtime_parts.append(f"涨跌幅：{direction}{abs(chg):.2f}%")
        if turnover is not None:
            realtime_parts.append(f"换手率：{turnover}%")
        if vol_ratio is not None:
            realtime_parts.append(f"量比：{vol_ratio}")
        if amplitude is not None:
            realtime_parts.append(f"振幅：{amplitude}%")

        analysis["dimensions"]["实时行情"] = "；".join(realtime_parts)
        dims.append("实时行情")

    # 2. 估值得分
    if spot_info:
        pe = spot_info.get("市盈率-动态")
        pb = spot_info.get("市净率")
        val_parts = []
        if pe is not None and pe > 0:
            if pe < 15:
                val_parts.append(f"市盈率{pe:.1f}，处于低估区间，估值优势明显")
            elif pe < 25:
                val_parts.append(f"市盈率{pe:.1f}，估值合理偏低")
            elif pe < 40:
                val_parts.append(f"市盈率{pe:.1f}，估值中性")
            elif pe < 60:
                val_parts.append(f"市盈率{pe:.1f}，估值偏高")
            else:
                val_parts.append(f"市盈率{pe:.1f}，估值较高，需谨慎")
        if pb is not None and pb > 0:
            if pb < 1.5:
                val_parts.append(f"市净率{pb:.2f}，低于净资产折价区间")
            elif pb < 3:
                val_parts.append(f"市净率{pb:.2f}，估值合理")
            else:
                val_parts.append(f"市净率{pb:.2f}，溢价较高")
        if val_parts:
            analysis["dimensions"]["估值水平"] = "；".join(val_parts)
            dims.append("估值水平")

    # 3. K线技术分析
    if kline_data is not None and not kline_data.empty:
        kline = kline_data.copy()
        close_col = '收盘' if '收盘' in kline.columns else ('close' if 'close' in kline.columns else None)
        vol_col = '成交量' if '成交量' in kline.columns else ('volume' if 'volume' in kline.columns else None)
        if close_col:
            close = kline[close_col].astype(float)

            # 均线分析
            ma5 = close.tail(5).mean()
            ma10 = close.tail(10).mean()
            ma20 = close.tail(20).mean()
            ma60 = close.tail(60).mean() if len(close) >= 60 else close.mean()

            current_price = close.iloc[-1]
            tech_parts = []

            # 均线排列
            if ma5 > ma10 > ma20:
                tech_parts.append("均线多头排列（MA5>MA10>MA20），中期趋势向好")
            elif ma5 < ma10 < ma20:
                tech_parts.append("均线空头排列（MA5<MA10<MA20），中期趋势偏弱")
            else:
                tech_parts.append("均线交织，处于震荡整理格局")

            # 价格与均线关系
            if current_price > ma20:
                tech_parts.append(f"股价站上20日均线({ma20:.2f})，短期偏强")
            else:
                tech_parts.append(f"股价位于20日均线({ma20:.2f})下方，短期承压")

            # 近期涨跌
            if len(close) >= 5:
                chg_5d = (close.iloc[-1] / close.iloc[-5] - 1) * 100
                tech_parts.append(f"近5日涨跌幅：{chg_5d:+.2f}%")
            if len(close) >= 20:
                chg_20d = (close.iloc[-1] / close.iloc[-20] - 1) * 100
                tech_parts.append(f"近20日涨跌幅：{chg_20d:+.2f}%")

            # MACD 简易判断
            if len(close) >= 26:
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                dif = ema12 - ema26
                dea = dif.ewm(span=9, adjust=False).mean()
                macd_bar = 2 * (dif - dea)
                if dif.iloc[-1] > dea.iloc[-1] and macd_bar.iloc[-1] > 0:
                    tech_parts.append("MACD金叉状态，红柱放大，动能偏多")
                elif dif.iloc[-1] < dea.iloc[-1]:
                    tech_parts.append("MACD死叉状态，动能偏空")
                else:
                    tech_parts.append("MACD处于临界状态")

            analysis["dimensions"]["K线技术分析"] = "；".join(tech_parts)
            dims.append("K线技术分析")

            # 成交量分析
            if vol_col:
                vol = kline[vol_col].astype(float)
                vol_5 = vol.tail(5).mean()
                vol_20 = vol.tail(20).mean() if len(vol) >= 20 else vol.mean()
                vol_ratio_val = vol_5 / vol_20 if vol_20 > 0 else 1
                vol_parts = []
                if vol_ratio_val > 1.5:
                    vol_parts.append(f"近5日均量是20日均量的{vol_ratio_val:.1f}倍，明显放量")
                elif vol_ratio_val > 1.0:
                    vol_parts.append(f"近5日均量略高于20日均量({vol_ratio_val:.1f}倍)，温和放量")
                else:
                    vol_parts.append(f"近5日均量低于20日均量({vol_ratio_val:.1f}倍)，缩量状态")
                analysis["dimensions"]["成交量分析"] = "；".join(vol_parts)
                dims.append("成交量分析")

            # 波动率分析
            if len(close) >= 20:
                returns = close.pct_change().dropna()
                if len(returns) >= 20:
                    daily_vol = returns.tail(20).std()
                    annual_vol = daily_vol * np.sqrt(252) * 100
                    vol_parts = []
                    if annual_vol < 20:
                        vol_parts.append(f"年化波动率{annual_vol:.1f}%，波动较低，适合稳健型投资者")
                    elif annual_vol < 40:
                        vol_parts.append(f"年化波动率{annual_vol:.1f}%，波动适中")
                    else:
                        vol_parts.append(f"年化波动率{annual_vol:.1f}%，波动较高，注意风险控制")
                    analysis["dimensions"]["历史波动率"] = "；".join(vol_parts)
                    dims.append("历史波动率")

            # ATR 计算与仓位管理
            if '最高' in kline.columns and '最低' in kline.columns:
                high = kline['最高'].astype(float)
                low = kline['最低'].astype(float)
                prev_close = close.shift(1)
                tr1 = high - low
                tr2 = (high - prev_close).abs()
                tr3 = (low - prev_close).abs()
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr14 = true_range.tail(14).mean()
                if atr14 > 0 and current_price > 0:
                    atr_pct = (atr14 / current_price) * 100
                    position_advice = _calc_position_advice(current_price, atr14, annual_vol if 'annual_vol' in dir() else 30)
                    pos_parts = [
                        f"ATR(14)={atr14:.2f}元（日均波幅{atr_pct:.1f}%）",
                        f"建议单笔仓位：{position_advice['suggested_position_pct']}（{position_advice['method']}）",
                        f"止损参考位：{position_advice['stop_loss']:.2f}元（-{position_advice['stop_loss_pct']:.1f}%）",
                        f"止盈参考位：{position_advice['take_profit']:.2f}元（+{position_advice['take_profit_pct']:.1f}%）",
                    ]
                    analysis["dimensions"]["仓位管理"] = "；".join(pos_parts)
                    dims.append("仓位管理")
                    analysis["position_advice"] = position_advice

    # 4. 中期趋势
    if spot_info:
        chg_60d = spot_info.get("60日涨跌幅")
        chg_ytd = spot_info.get("年初至今涨跌幅")
        trend_parts = []
        if chg_60d is not None:
            if chg_60d > 20:
                trend_parts.append(f"近60日上涨{chg_60d:+.1f}%，中期趋势强劲")
            elif chg_60d > 5:
                trend_parts.append(f"近60日上涨{chg_60d:+.1f}%，中期趋势向好")
            elif chg_60d > -5:
                trend_parts.append(f"近60日横盘({chg_60d:+.1f}%)，方向待选择")
            else:
                trend_parts.append(f"近60日下跌{chg_60d:.1f}%，中期趋势偏弱")
        if chg_ytd is not None:
            if chg_ytd > 10:
                trend_parts.append(f"年内累计上涨{chg_ytd:+.1f}%")
            elif chg_ytd > -10:
                trend_parts.append(f"年内表现平稳({chg_ytd:+.1f}%)")
            else:
                trend_parts.append(f"年内跌幅{chg_ytd:.1f}%")
        if trend_parts:
            analysis["dimensions"]["中期趋势"] = "；".join(trend_parts)
            dims.append("中期趋势")

    # 5. 资金面深度分析
    if spot_info:
        capital_flow = _analyze_capital_flow(spot_info, kline_data)
        flow_parts = []
        flow_direction = capital_flow.get("资金流向判断", "中性")
        main_force = capital_flow.get("主力动向", "")
        flow_parts.append(f"资金流向：{flow_direction}")
        if main_force and main_force != "无明显主力迹象":
            flow_parts.append(f"主力动向：{main_force}")
        for note in capital_flow.get("分析说明", []):
            flow_parts.append(note)
        for alert in capital_flow.get("异常检测", []):
            flow_parts.append(f"[异常预警] {alert}")
        if flow_parts:
            analysis["dimensions"]["资金面分析"] = "；".join(flow_parts)
            dims.append("资金面分析")
        analysis["capital_flow"] = capital_flow

    # 6. 市场情绪分析
    if spot_info:
        sentiment = _analyze_market_sentiment(spot_info, kline_data)
        sent_parts = []
        sent_parts.append(f"情绪判断：{sentiment.get('情绪判断', '中性')}（恐慌贪婪指数：{sentiment.get('恐慌贪婪指数', 50)}）")
        sent_parts.append(f"市场分歧度：{sentiment.get('市场分歧度', '正常')}")
        sent_parts.append(f"情绪趋势：{sentiment.get('情绪趋势', '平稳')}")
        for note in sentiment.get("分析说明", []):
            sent_parts.append(note)
        if sent_parts:
            analysis["dimensions"]["市场情绪"] = "；".join(sent_parts)
            dims.append("市场情绪")
        analysis["sentiment"] = sentiment

    # 7. 交易成本估算
    if spot_info:
        price = _safe_float(spot_info.get("最新价"))
        amplitude = _safe_float(spot_info.get("振幅"))
        turnover = _safe_float(spot_info.get("换手率"))
        if price and price > 0:
            # 以1手（100股）为基准估算交易成本
            cost_analysis = _analyze_trading_cost(price, 100, is_sell=True, turnover=turnover, amplitude=amplitude)
            cost_parts = []
            cost_parts.append(f"买入+卖出往返成本约{cost_analysis.get('成本占比', '0%')}")
            for s in cost_analysis.get("优化建议", [])[:2]:
                cost_parts.append(s)
            if cost_parts:
                analysis["dimensions"]["交易成本"] = "；".join(cost_parts)
                dims.append("交易成本")
            analysis["trading_cost"] = cost_analysis

    # 8. 多时间框架分析
    if kline_data is not None and not kline_data.empty:
        mtf = _multi_timeframe_analysis(kline_data)
        mtf_parts = []
        mtf_parts.append(f"共振判断：{mtf.get('共振判断', '数据不足')}")
        for tf_name, tf_info in mtf.get("各周期趋势", {}).items():
            mtf_parts.append(f"{tf_name}：{tf_info.get('趋势方向', '未知')}（强度{tf_info.get('趋势强度', 0)}）")
        for alert in mtf.get("背离检测", []):
            mtf_parts.append(f"[背离] {alert}")
        for s in mtf.get("综合建议", []):
            mtf_parts.append(s)
        if mtf_parts:
            analysis["dimensions"]["多周期分析"] = "；".join(mtf_parts)
            dims.append("多周期分析")
        analysis["multi_timeframe"] = mtf

    # 生成综合摘要
    summary_parts = [f"【{stock_name}({stock_code}) 多维度分析报告】" if stock_name else f"【{stock_code} 多维度分析报告】"]
    summary_parts.append(f"分析维度：{'、'.join(dims)}")
    summary_parts.append("")

    for dim_name in dims:
        if dim_name in analysis["dimensions"]:
            summary_parts.append(f"【{dim_name}】{analysis['dimensions'][dim_name]}")

    analysis["summary"] = "\n".join(summary_parts)

    # LLM 自然语言总结
    config = load_config()
    if config.get("enabled") and config.get("api_key"):
        # 收集关键数值供LLM参考
        key_metrics = {}
        if spot_info:
            for k in ["最新价", "涨跌幅", "换手率", "量比", "市盈率-动态", "市净率", "60日涨跌幅", "年初至今涨跌幅"]:
                v = spot_info.get(k)
                if v is not None:
                    key_metrics[k] = _safe_float(v)
        if kline_data is not None and not kline_data.empty:
            kline = kline_data.copy()
            ccol = '收盘' if '收盘' in kline.columns else ('close' if 'close' in kline.columns else None)
            if ccol:
                close_series = kline[ccol].astype(float)
                key_metrics["当前价"] = round(float(close_series.iloc[-1]), 2)
                key_metrics["MA5"] = round(float(close_series.tail(5).mean()), 2)
                key_metrics["MA10"] = round(float(close_series.tail(10).mean()), 2)
                key_metrics["MA20"] = round(float(close_series.tail(20).mean()), 2)
                if len(close_series) >= 60:
                    key_metrics["MA60"] = round(float(close_series.tail(60).mean()), 2)
                if len(close_series) >= 5:
                    key_metrics["5日涨跌"] = round(float((close_series.iloc[-1] / close_series.iloc[-5] - 1) * 100), 2)
                if len(close_series) >= 20:
                    key_metrics["20日涨跌"] = round(float((close_series.iloc[-1] / close_series.iloc[-20] - 1) * 100), 2)
                    returns = close_series.pct_change().dropna().tail(20)
                    key_metrics["20日波动率"] = round(float(returns.std() * 100), 2)

        dims_text = "\n".join([f"【{d}】{analysis['dimensions'].get(d, '')}" for d in dims])
        metrics_text = json.dumps(key_metrics, ensure_ascii=False)

        system_prompt = """你是A股资深量化分析师。请根据以下多维度分析数据和关键指标，生成一份专业的个股研判报告。

输出格式（严格按此结构）：
【综合研判】一句话概括整体多空判断，并说明核心理由（30字以内）
【关键价位】列出2-3个关键技术价位（支撑位、压力位、目标位），给出具体数字
【仓位建议】基于ATR波动率，给出建议仓位比例和止损止盈价位
【操作参考】给出1-2条操作层面的参考意见（如"回踩XX元可关注"、"突破XX元可考虑"）
【风险提示】指出当前最需要关注的1个风险点
【免责声明】以上分析仅供参考，不构成投资建议

要求：
- 语言专业但不晦涩，像资深分析师给客户做简报
- 价位判断要基于提供的MA均线数据和近期高低点
- 仓位建议要参考ATR波动率数据，波动越大仓位应越小
- 总字数控制在250字以内"""
        user_prompt = f"""股票：{stock_name}({stock_code})

关键指标：
{metrics_text}

各维度分析：
{dims_text}

仓位管理数据：
{json.dumps(analysis.get('position_advice', {}), ensure_ascii=False)}

请按格式输出研判报告。"""
        llm_summary = call_llm(system_prompt, user_prompt)
        if llm_summary:
            analysis["ai_summary"] = llm_summary

    return analysis


def _calc_position_advice(current_price, atr14, annual_vol=30):
    """
    基于ATR波动率计算仓位建议和止损止盈价位
    返回: {suggested_position_pct, method, stop_loss, stop_loss_pct, take_profit, take_profit_pct}
    """
    atr_pct = (atr14 / current_price) * 100

    # 基于波动率的仓位计算：仓位 = 目标波动率 / 实际波动率
    target_vol = 15
    vol_based_position = min(target_vol / max(annual_vol, 5) * 100, 100)

    # 基于ATR的风险仓位：假设单笔风险承受为总资金的2%，止损设在2倍ATR
    risk_per_trade = 2.0
    atr_stop_multiple = 2.0
    atr_based_position = (risk_per_trade / (atr_pct * atr_stop_multiple)) * 100

    # 综合两种方法取较小值（更保守）
    suggested_pct = min(vol_based_position, atr_based_position)

    # 根据波动率分档给出建议
    if atr_pct < 2:
        suggested_pct = min(suggested_pct, 30)
        method = "低波动标的，可适度提高仓位至30%"
    elif atr_pct < 4:
        suggested_pct = min(suggested_pct, 20)
        method = "中等波动，建议仓位控制在20%以内"
    elif atr_pct < 6:
        suggested_pct = min(suggested_pct, 10)
        method = "波动偏高，建议仓位不超过10%"
    else:
        suggested_pct = min(suggested_pct, 5)
        method = "高波动标的，严格控制仓位在5%以内"

    suggested_pct = max(round(suggested_pct, 1), 1.0)

    # 止损止盈价位（基于ATR）
    stop_loss = round(current_price - atr14 * 2, 2)
    stop_loss_pct = round((stop_loss / current_price - 1) * 100, 1)
    take_profit = round(current_price + atr14 * 3, 2)
    take_profit_pct = round((take_profit / current_price - 1) * 100, 1)

    return {
        "suggested_position_pct": f"{suggested_pct}%",
        "method": method,
        "stop_loss": stop_loss,
        "stop_loss_pct": stop_loss_pct,
        "take_profit": take_profit,
        "take_profit_pct": take_profit_pct,
        "atr14": round(atr14, 2),
        "atr_pct": round(atr_pct, 1),
    }


def _apply_extra_dimensions(stocks, extra_dimensions):
    """对推荐结果应用额外维度进行补充评分调整"""
    import pandas as pd

    for dim in extra_dimensions:
        if dim == "K线技术分析":
            stocks = _apply_kline_dimension(stocks)
        elif dim == "成交量分析":
            stocks = _apply_volume_dimension(stocks)
        elif dim == "资金流向":
            stocks = _apply_fund_flow_dimension(stocks)
        elif dim == "财务基本面":
            stocks = _apply_fundamental_dimension(stocks)
        elif dim == "历史波动率":
            stocks = _apply_volatility_dimension(stocks)

    # 重新排序
    stocks.sort(key=lambda x: x.get("综合评分", 0), reverse=True)
    return stocks


def _apply_kline_dimension(stocks):
    """K线技术分析维度：基于均线趋势和MACD信号调整评分"""
    try:
        from data_utils import get_stock_kline
        import pandas as pd
        import numpy as np

        for s in stocks:
            code = s.get("代码", "")
            try:
                df = get_stock_kline(code, days=60)
                if df is None or df.empty or len(df) < 20:
                    s["推荐理由"] = (s.get("推荐理由", "") + "；K线数据不足，无法进行技术评估")
                    continue

                close = df['收盘'].astype(float)
                ma5 = close.rolling(5).mean().iloc[-1]
                ma10 = close.rolling(10).mean().iloc[-1]
                ma20 = close.rolling(20).mean().iloc[-1]
                latest_close = close.iloc[-1]

                # 均线多头排列加分
                kline_bonus = 0
                kline_reason = ""
                if ma5 > ma10 > ma20 and latest_close > ma5:
                    kline_bonus = 8
                    kline_reason = "均线多头排列，短期趋势向好"
                elif ma5 > ma10:
                    kline_bonus = 5
                    kline_reason = "短期均线金叉，走势偏多"
                elif ma5 < ma10 < ma20:
                    kline_bonus = -5
                    kline_reason = "均线空头排列，短期承压"
                elif latest_close > ma20:
                    kline_bonus = 3
                    kline_reason = "股价站上20日均线，中期趋势偏多"
                else:
                    kline_bonus = -3
                    kline_reason = "股价低于20日均线，中期趋势偏弱"

                # MACD评估
                ema12 = close.ewm(span=12).mean()
                ema26 = close.ewm(span=26).mean()
                dif = ema12 - ema26
                dea = dif.ewm(span=9).mean()
                macd_bar = 2 * (dif - dea)
                if dif.iloc[-1] > dea.iloc[-1] and macd_bar.iloc[-1] > 0:
                    kline_bonus += 3
                    kline_reason += "，MACD金叉状态"

                s["综合评分"] = min(99, s.get("综合评分", 0) + kline_bonus)
                s["推荐理由"] = (s.get("推荐理由", "") + f"；K线技术：{kline_reason}")

            except Exception as e:
                s["推荐理由"] = (s.get("推荐理由", "") + f"；K线数据获取异常")

    except Exception as e:
        print(f"[K线维度] 评估异常: {e}")

    return stocks


def _apply_volume_dimension(stocks):
    """成交量分析维度"""
    try:
        from data_utils import get_stock_kline
        import pandas as pd

        for s in stocks:
            code = s.get("代码", "")
            try:
                df = get_stock_kline(code, days=30)
                if df is None or df.empty or len(df) < 10:
                    continue

                vol = df['成交量'].astype(float)
                avg_vol_5 = vol.tail(5).mean()
                avg_vol_20 = vol.tail(20).mean()
                vol_ratio = avg_vol_5 / avg_vol_20 if avg_vol_20 > 0 else 1

                if vol_ratio > 1.5:
                    s["综合评分"] = min(99, s.get("综合评分", 0) + 5)
                    s["推荐理由"] = (s.get("推荐理由", "") + "；成交量：近5日放量明显，资金关注度提升")
                elif vol_ratio > 1.2:
                    s["综合评分"] = min(99, s.get("综合评分", 0) + 2)
                    s["推荐理由"] = (s.get("推荐理由", "") + "；成交量：温和放量，交投趋于活跃")
                elif vol_ratio < 0.5:
                    s["综合评分"] = max(1, s.get("综合评分", 0) - 3)
                    s["推荐理由"] = (s.get("推荐理由", "") + "；成交量：近期缩量明显，市场关注度下降")

            except Exception:
                pass
    except Exception as e:
        print(f"[成交量维度] 评估异常: {e}")

    return stocks


def _analyze_capital_flow(spot_info, kline_data=None):
    """
    资金面深度分析：主力动向、资金流向、异常交易检测
    基于可用数据（涨跌幅、换手率、量比、成交额、振幅）进行多维度推断
    返回: {资金流向判断, 主力动向, 异常检测, 综合评分, 分析说明}
    """
    if not spot_info:
        return {"资金流向判断": "数据不足", "综合评分": 0, "分析说明": []}

    chg = _safe_float(spot_info.get("涨跌幅"))
    turnover = _safe_float(spot_info.get("换手率"))
    vol_ratio = _safe_float(spot_info.get("量比"))
    amount = _safe_float(spot_info.get("成交额"))
    amplitude = _safe_float(spot_info.get("振幅"))
    price = _safe_float(spot_info.get("最新价"))

    analysis = {
        "资金流向判断": "中性",
        "主力动向": "无明显主力迹象",
        "异常检测": [],
        "综合评分": 0,
        "分析说明": [],
    }

    score = 0

    # ===== 1. 资金流向判断 =====
    # 基于涨跌幅+换手率+量比综合判断资金方向
    if chg is not None and turnover is not None and vol_ratio is not None:
        if chg > 3 and turnover > 5 and vol_ratio > 1.5:
            analysis["资金流向判断"] = "强势流入"
            score += 15
            analysis["分析说明"].append(f"放量上涨{chg:.1f}%，换手率{turnover}%，量比{vol_ratio}，资金大举流入")
        elif chg > 1 and turnover > 3 and vol_ratio > 1.2:
            analysis["资金流向判断"] = "温和流入"
            score += 8
            analysis["分析说明"].append(f"价量齐升，换手率{turnover}%，资金温和流入")
        elif chg > 0 and vol_ratio > 1.0:
            analysis["资金流向判断"] = "小幅流入"
            score += 3
            analysis["分析说明"].append("股价微涨伴随温和放量，资金小幅净流入")
        elif chg < -3 and turnover > 5 and vol_ratio > 1.5:
            analysis["资金流向判断"] = "明显流出"
            score -= 15
            analysis["分析说明"].append(f"放量下跌{abs(chg):.1f}%，换手率{turnover}%，资金出逃迹象明显")
        elif chg < -1 and turnover > 3 and vol_ratio > 1.2:
            analysis["资金流向判断"] = "温和流出"
            score -= 8
            analysis["分析说明"].append("价跌量增，资金温和流出")
        elif chg < 0 and vol_ratio < 0.8:
            analysis["资金流向判断"] = "缩量回调"
            score -= 2
            analysis["分析说明"].append("缩量下跌，抛压有限，可能是正常调整")
        elif abs(chg) < 1 and vol_ratio < 0.8:
            analysis["资金流向判断"] = "交投清淡"
            analysis["分析说明"].append("缩量横盘，市场关注度低，资金观望")

    # ===== 2. 主力动向判断 =====
    # 通过振幅+换手率+量比组合判断是否有主力活动
    if amplitude is not None and turnover is not None and vol_ratio is not None:
        if amplitude > 8 and turnover > 8 and vol_ratio > 2:
            analysis["主力动向"] = "主力高度活跃，疑似对倒或拉升出货"
            analysis["异常检测"].append("高振幅+高换手+高量比，主力活动迹象明显，需警惕操纵风险")
            score += 5  # 主力活跃可能带来机会，但也伴随风险
        elif amplitude > 5 and turnover > 5 and vol_ratio > 1.5:
            analysis["主力动向"] = "主力较为活跃，资金博弈激烈"
            analysis["分析说明"].append("振幅和换手率偏高，多空分歧加大，主力资金参与度较高")
            score += 3
        elif amplitude > 3 and turnover > 3 and vol_ratio > 1.2:
            analysis["主力动向"] = "有主力资金参与迹象"
            analysis["分析说明"].append("盘中振幅和量能配合，可能有主力资金在运作")
            score += 1
        elif turnover < 1 and vol_ratio < 0.6:
            analysis["主力动向"] = "主力资金参与度低"
            analysis["分析说明"].append("换手率极低，量能萎缩，主力资金关注度不足")

    # ===== 3. 成交额分析 =====
    if amount is not None:
        if amount > 50:
            analysis["分析说明"].append(f"日成交额{amount:.1f}亿元，流动性充裕，大资金进出方便")
            score += 3
        elif amount > 10:
            analysis["分析说明"].append(f"日成交额{amount:.1f}亿元，流动性良好")
            score += 1
        elif amount > 3:
            analysis["分析说明"].append(f"日成交额{amount:.1f}亿元，流动性一般")
        elif amount > 0:
            analysis["分析说明"].append(f"日成交额仅{amount:.2f}亿元，流动性偏弱，大资金进出受限")
            score -= 2

    # ===== 4. 异常交易检测 =====
    # 检测可能的主力对倒、拉高出货等异常行为
    if chg is not None and amplitude is not None and turnover is not None:
        # 高开低走放量：可能是拉高出货
        if chg < -2 and amplitude > 6 and turnover > 5:
            analysis["异常检测"].append("高开低走放量，疑似主力拉高出货，短期注意风险")
            score -= 10
        # 低开高走放量：可能是洗盘后拉升
        if chg > 2 and amplitude > 6 and turnover > 5:
            analysis["异常检测"].append("低开高走放量，疑似洗盘后拉升，关注后续持续性")
            score += 5
        # 尾盘急拉或急跌（通过振幅和涨跌幅关系推断）
        if abs(chg) > 5 and amplitude > 10:
            analysis["异常检测"].append("日内波动剧烈，可能存在尾盘异动或突发事件影响")

    # ===== 5. K线辅助分析 =====
    if kline_data is not None and not kline_data.empty:
        try:
            import pandas as pd
            kline = kline_data.copy()
            vol_col = '成交量' if '成交量' in kline.columns else ('volume' if 'volume' in kline.columns else None)
            close_col = '收盘' if '收盘' in kline.columns else ('close' if 'close' in kline.columns else None)

            if vol_col and close_col and len(kline) >= 5:
                vol = kline[vol_col].astype(float)
                close = kline[close_col].astype(float)

                # 近5日成交量趋势
                vol_5_avg = vol.tail(5).mean()
                vol_prev_5_avg = vol.tail(10).head(5).mean() if len(vol) >= 10 else vol_5_avg
                vol_trend = (vol_5_avg / vol_prev_5_avg - 1) * 100 if vol_prev_5_avg > 0 else 0

                if vol_trend > 30:
                    analysis["分析说明"].append(f"近5日均量较前5日增长{vol_trend:.0f}%，资金关注度持续提升")
                    score += 3
                elif vol_trend < -30:
                    analysis["分析说明"].append(f"近5日均量较前5日萎缩{abs(vol_trend):.0f}%，资金关注度下降")
                    score -= 2

                # 量价配合分析
                chg_5d = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
                if chg_5d > 3 and vol_trend > 20:
                    analysis["分析说明"].append("近5日量价齐升，资金持续流入，短期动能充足")
                    score += 5
                elif chg_5d < -3 and vol_trend > 20:
                    analysis["分析说明"].append("近5日放量下跌，资金持续流出，短期压力较大")
                    score -= 5
                elif chg_5d > 3 and vol_trend < -10:
                    analysis["分析说明"].append("近5日缩量上涨，上涨动力减弱，注意回调风险")
                    score -= 2
        except Exception:
            pass

    analysis["综合评分"] = max(-30, min(30, score))

    return analysis


def _apply_fund_flow_dimension(stocks):
    """资金流向维度（增强版：基于多维度资金面分析）"""
    for s in stocks:
        spot_info = {
            "涨跌幅": s.get("涨跌幅"),
            "换手率": s.get("换手率"),
            "量比": s.get("量比"),
            "成交额": s.get("成交额"),
            "振幅": s.get("振幅"),
            "最新价": s.get("最新价"),
        }
        flow = _analyze_capital_flow(spot_info)
        flow_score = flow.get("综合评分", 0)
        flow_direction = flow.get("资金流向判断", "中性")
        main_force = flow.get("主力动向", "")

        s["综合评分"] = max(1, min(99, s.get("综合评分", 0) + flow_score))

        reason_parts = [f"资金面：{flow_direction}"]
        if main_force and main_force != "无明显主力迹象":
            reason_parts.append(main_force)
        s["推荐理由"] = (s.get("推荐理由", "") + "；" + "，".join(reason_parts))

        # 存储资金面分析详情
        s["资金面分析"] = flow

    return stocks


def _apply_fundamental_dimension(stocks):
    """财务基本面维度（基于市盈率和市净率综合评估）"""
    for s in stocks:
        pe = s.get("市盈率")
        pb = s.get("市净率")
        reasons = []
        bonus = 0

        if pe is not None and pe > 0:
            if pe < 10:
                bonus += 5
                reasons.append("市盈率极低，盈利能力强")
            elif pe < 20:
                bonus += 3
                reasons.append("市盈率合理，估值有支撑")

        if pb is not None and pb > 0:
            if pb < 1:
                bonus += 5
                reasons.append("破净状态，资产价值凸显")
            elif pb < 2:
                bonus += 2
                reasons.append("市净率较低，安全边际充足")

        if reasons:
            s["综合评分"] = min(99, s.get("综合评分", 0) + bonus)
            s["推荐理由"] = (s.get("推荐理由", "") + "；基本面：" + "，".join(reasons))

    return stocks


def _apply_volatility_dimension(stocks):
    """历史波动率维度"""
    try:
        from data_utils import get_stock_kline
        import pandas as pd
        import numpy as np

        for s in stocks:
            code = s.get("代码", "")
            try:
                df = get_stock_kline(code, days=60)
                if df is None or df.empty or len(df) < 20:
                    continue

                close = df['收盘'].astype(float)
                returns = close.pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # 年化波动率

                if volatility < 0.2:
                    s["综合评分"] = min(99, s.get("综合评分", 0) + 3)
                    s["推荐理由"] = (s.get("推荐理由", "") + f"；波动率：年化波动{volatility:.1%}，走势稳健低波动")
                elif volatility < 0.4:
                    s["推荐理由"] = (s.get("推荐理由", "") + f"；波动率：年化波动{volatility:.1%}，波动适中")
                else:
                    s["综合评分"] = max(1, s.get("综合评分", 0) - 3)
                    s["推荐理由"] = (s.get("推荐理由", "") + f"；波动率：年化波动{volatility:.1%}，高波动需注意风险")

            except Exception:
                pass
    except Exception as e:
        print(f"[波动率维度] 评估异常: {e}")

    return stocks


if __name__ == "__main__":
    # 测试自然语言解析
    test_text = "帮我写一个5日均线上穿20日均线且放量1.5倍的策略"
    print(f"测试输入: {test_text}")
    print("=" * 60)

    result = nl_to_strategy(test_text)
    print(f"解析结果: {json.dumps(result['解析结果'], ensure_ascii=False, indent=2)}")
    print(f"\n策略代码长度: {len(result['策略代码'])} 字符")
    print(f"回测代码长度: {len(result['回测代码'])} 字符")
