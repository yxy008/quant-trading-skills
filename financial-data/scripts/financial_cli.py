#!/usr/bin/env python3
"""
财务数据查询工具 - financial-data (全真实数据版)
所有数据均来自真实接口，不包含任何虚拟模拟数据
"""
import argparse
import json
import sys
import os
from datetime import datetime, timedelta

# 添加agent目录到路径以导入data_utils
_agent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

try:
    import akshare as ak
    import pandas as pd
except ImportError:
    print("请先安装依赖: pip install akshare pandas")
    sys.exit(1)

from data_utils import get_stock_kline, get_financial_data, _get_spot_df


def get_stock_name_and_info(symbol):
    """
    获取股票名称和基本信息（真实接口，有fallback）
    :param symbol: 6位股票代码
    :return: dict
    """
    try:
        df_list = _get_spot_df()
        filtered = df_list[df_list['代码'] == symbol]
        if not filtered.empty:
            row = filtered.iloc[0]
            return {
                "名称": str(row['名称']),
                "行业": str(row['行业']) if '行业' in df_list.columns else "未知",
                "数据源": "真实接口-东方财富实时行情"
            }
    except Exception:
        pass
    
    # 如果获取失败，返回最小化的fallback（仅包含名称和行业，且明确标注）
    name_map = {
        "600519": "贵州茅台",
        "000001": "平安银行",
        "002594": "比亚迪",
        "300750": "宁德时代",
        "600036": "招商银行",
        "601318": "中国平安",
        "601398": "工商银行",
        "601939": "建设银行"
    }
    industry_map = {
        "600519": "白酒",
        "000001": "银行",
        "002594": "新能源",
        "300750": "新能源",
        "600036": "银行",
        "601318": "保险",
        "601398": "银行",
        "601939": "银行"
    }
    return {
        "名称": name_map.get(symbol, f"股票{symbol}"),
        "行业": industry_map.get(symbol, "未知"),
        "数据源": "Fallback数据"
    }


def calculate_indicators_from_kline(symbol, df_kline):
    """
    从 K线数据 计算真实指标
    :param symbol: 股票代码
    :param df_kline: K线数据 DataFrame
    :return: dict 指标
    """
    if df_kline is None or len(df_kline) < 20:
        return None
    
    latest = df_kline.iloc[-1]
    prev_close = df_kline.iloc[-2]['收盘'] if len(df_kline) >= 2 else latest['收盘']
    
    change_pct = (latest['收盘'] - prev_close) / prev_close * 100 if prev_close > 0 else 0
    
    df_recent = df_kline.tail(30)
    returns = df_recent['收盘'].pct_change().dropna()
    volatility = returns.std() * 100 if len(returns) > 5 else 0
    
    has_amount = '成交额' in df_recent.columns
    has_volume = '成交量' in df_recent.columns
    if has_amount:
        avg_amount = df_recent['成交额'].mean()
    elif has_volume:
        avg_amount = df_recent['成交量'].mean()
    else:
        avg_amount = 0

    latest_amount = float(latest.get('成交额', latest.get('成交量', 0)))
    latest_close = float(latest.get('收盘', 0))

    return {
        "最新价": round(latest_close, 2),
        "涨跌幅": round(change_pct, 2),
        "成交量": int(latest_amount),
        "成交额": round(latest_amount * latest_close, 2),
        "30日平均成交量": round(float(avg_amount), 0),
        "波动率": round(volatility, 2),
        "K线数据量": len(df_kline),
        "最新日期": str(latest.name)[:10] if hasattr(latest, 'name') else ""
    }


def get_financial_metrics(symbol):
    """
    获取综合财务指标（全真实数据）
    :param symbol: 股票代码
    :return: dict
    """
    result = {"代码": symbol}
    
    # 1. 获取名称和基本信息
    name_info = get_stock_name_and_info(symbol)
    result.update(name_info)
    
    # 2. 获取 K线数据
    df_kline = get_stock_kline(symbol, days=365)
    
    if df_kline is not None and not df_kline.empty:
        # 从 K线 计算真实指标
        kline_indicators = calculate_indicators_from_kline(symbol, df_kline)
        if kline_indicators is not None:
            result.update(kline_indicators)
            result["数据质量"] = "完整真实数据"
        else:
            result["数据质量"] = "部分真实数据"
    else:
        result["数据质量"] = "K线数据不可用"
        result["提示"] = "暂时无法获取 K线 数据，请稍后重试"
    
    return result


def get_income_statement(symbol):
    """
    利润表（真实数据接口）
    :param symbol: 股票代码
    :return: dict
    """
    try:
        df_profit = ak.stock_profit_sheet_by_yearly_em(symbol=symbol)
        if df_profit is not None and not df_profit.empty:
            # 取最新一年的数据
            latest = df_profit.iloc[0]
            result = {
                "代码": symbol,
                "报告期": str(latest.get('报告期', '未知')),
                "数据源": "真实接口-东方财富",
                "数据质量": "完整"
            }
            # 只保留真实存在的字段
            for col in df_profit.columns:
                val = latest.get(col)
                if val is not None and pd.notna(val):
                    result[str(col)] = val
            return result
    except Exception:
        pass
    
    # 接口不可用时返回明确提示
    return {
        "代码": symbol,
        "提示": "利润表接口暂时不可用，请稍后重试",
        "数据源": "无（接口失败）"
    }


def get_balance_sheet(symbol):
    """
    资产负债表（真实数据接口）
    :param symbol: 股票代码
    :return: dict
    """
    try:
        df_balance = ak.stock_balance_sheet_by_yearly_em(symbol=symbol)
        if df_balance is not None and not df_balance.empty:
            latest = df_balance.iloc[0]
            result = {
                "代码": symbol,
                "报告期": str(latest.get('报告期', '未知')),
                "数据源": "真实接口-东方财富",
                "数据质量": "完整"
            }
            for col in df_balance.columns:
                val = latest.get(col)
                if val is not None and pd.notna(val):
                    result[str(col)] = val
            return result
    except Exception:
        pass
    
    return {
        "代码": symbol,
        "提示": "资产负债表接口暂时不可用，请稍后重试",
        "数据源": "无（接口失败）"
    }


def get_cash_flow(symbol):
    """
    现金流量表（真实数据接口）
    :param symbol: 股票代码
    :return: dict
    """
    try:
        df_cash = ak.stock_cash_flow_sheet_by_yearly_em(symbol=symbol)
        if df_cash is not None and not df_cash.empty:
            latest = df_cash.iloc[0]
            result = {
                "代码": symbol,
                "报告期": str(latest.get('报告期', '未知')),
                "数据源": "真实接口-东方财富",
                "数据质量": "完整"
            }
            for col in df_cash.columns:
                val = latest.get(col)
                if val is not None and pd.notna(val):
                    result[str(col)] = val
            return result
    except Exception:
        pass
    
    return {
        "代码": symbol,
        "提示": "现金流量表接口暂时不可用，请稍后重试",
        "数据源": "无（接口失败）"
    }


def main():
    parser = argparse.ArgumentParser(description='财务数据查询工具（全真实数据）')
    parser.add_argument('action', choices=['metrics', 'income', 'balance', 'cash'],
                        help='操作类型: metrics（财务指标）, income（利润表）, balance（资产负债表）, cash（现金流量表）')
    parser.add_argument('--symbol', required=True, help='6位股票代码')
    
    args = parser.parse_args()
    
    try:
        if args.action == 'metrics':
            data = get_financial_metrics(args.symbol)
        elif args.action == 'income':
            data = get_income_statement(args.symbol)
        elif args.action == 'balance':
            data = get_balance_sheet(args.symbol)
        elif args.action == 'cash':
            data = get_cash_flow(args.symbol)
        
        print(json.dumps(data, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == '__main__':
    main()
