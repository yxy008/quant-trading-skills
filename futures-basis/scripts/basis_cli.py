#!/usr/bin/env python3
"""
期货基差与升贴水分析系统
支持基差计算、升贴水结构分析、期现套利、基差交易策略
"""
import argparse
import json
import sys
import os
import time
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


# ==================== 期货基差计算 ====================

def calculate_basis(futures_price, spot_price):
    """
    计算基差

    参数:
        futures_price: 期货价格
        spot_price: 现货价格

    返回: 基差分析
    """
    basis = futures_price - spot_price
    basis_pct = (basis / spot_price) * 100

    if basis > 0:
        structure = "期货升水（Contango）"
        implication = "市场预期未来价格上涨，或持有成本较高"
    elif basis < 0:
        structure = "期货贴水（Backwardation）"
        implication = "市场预期未来价格下跌，或现货供应紧张"
    else:
        structure = "平水"
        implication = "期货与现货价格一致"

    return {
        "计算时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "期货价格": futures_price,
        "现货价格": spot_price,
        "基差": round(basis, 2),
        "基差率": f"{basis_pct:.2f}%",
        "升贴水结构": structure,
        "市场含义": implication,
    }


# ==================== 股指期货基差分析 ====================

def index_futures_basis():
    """
    股指期货基差分析
    获取IF/IC/IH/IM四大股指期货的基差数据

    返回: 股指期货基差分析
    """
    try:
        df = ak.stock_index_spot_em()
    except Exception as e:
        return {"error": f"获取指数数据失败: {str(e)}"}

    if df is None or len(df) == 0:
        return {"error": "未获取到指数数据"}

    # 主要指数
    target_indices = {
        "上证指数": "000001",
        "沪深300": "000300",
        "中证500": "000905",
        "中证1000": "000852",
        "上证50": "000016",
    }

    spot_data = {}
    for name, code in target_indices.items():
        for i in range(len(df)):
            row_code = str(df.iloc[i].get('代码', ''))
            if code in row_code:
                spot_data[name] = {
                    "代码": code,
                    "最新价": float(df.iloc[i].get('最新价', 0)),
                    "涨跌幅": f"{df.iloc[i].get('涨跌幅', 0)}%",
                }
                break

    # 期货合约对应关系
    futures_mapping = {
        "IF": {"标的": "沪深300", "乘数": 300},
        "IC": {"标的": "中证500", "乘数": 200},
        "IH": {"标的": "上证50", "乘数": 300},
        "IM": {"标的": "中证1000", "乘数": 200},
    }

    # 尝试获取期货数据
    try:
        futures_df = ak.futures_zh_daily_sina(symbol="IF0")
    except Exception:
        futures_df = None

    basis_analysis = []
    for fut_code, info in futures_mapping.items():
        underlying = info["标的"]
        if underlying in spot_data:
            spot_price = spot_data[underlying]["最新价"]
            # 期货价格需要从实际数据获取，这里做说明
            basis_analysis.append({
                "期货合约": fut_code,
                "标的指数": underlying,
                "现货价格": spot_price,
                "乘数": info["乘数"],
                "说明": "期货价格需从实时行情获取，基差=期货-现货",
            })

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "现货指数": spot_data,
        "期货合约": basis_analysis,
        "基差交易策略": {
            "正向套利": "当期货大幅升水时，买入现货（或ETF）同时卖出期货",
            "反向套利": "当期货大幅贴水时，买入期货同时卖出现货",
            "基差回归": "基差偏离均值时，做多贴水合约或做空升水合约",
        },
        "基差对市场的影响": [
            "期货持续贴水：市场悲观，对冲需求大，可能是底部信号",
            "期货持续升水：市场乐观，投机需求大，需警惕过热",
            "基差收敛：临近交割日，期货价格向现货靠拢",
            "基差扩大：市场分歧加大，波动率上升",
        ],
    }


# ==================== 商品期货基差 ====================

def commodity_futures_basis():
    """
    商品期货基差分析
    分析主要商品期货的升贴水结构

    返回: 商品期货基差分析
    """
    # 主要商品期货品种
    commodities = [
        {"品种": "螺纹钢", "代码": "RB", "交易所": "上期所", "单位": "10吨/手"},
        {"品种": "铁矿石", "代码": "I", "交易所": "大商所", "单位": "100吨/手"},
        {"品种": "原油", "代码": "SC", "交易所": "上期能源", "单位": "1000桶/手"},
        {"品种": "黄金", "代码": "AU", "交易所": "上期所", "单位": "1000克/手"},
        {"品种": "铜", "代码": "CU", "交易所": "上期所", "单位": "5吨/手"},
        {"品种": "豆粕", "代码": "M", "交易所": "大商所", "单位": "10吨/手"},
        {"品种": "PTA", "代码": "TA", "交易所": "郑商所", "单位": "5吨/手"},
        {"品种": "甲醇", "代码": "MA", "交易所": "郑商所", "单位": "10吨/手"},
    ]

    # 基差分析框架
    basis_framework = {
        "正向市场（Contango）": {
            "特征": "远月合约价格高于近月",
            "原因": "仓储成本+资金成本+保险成本",
            "策略": "卖出近月买入远月的熊市套利",
        },
        "反向市场（Backwardation）": {
            "特征": "远月合约价格低于近月",
            "原因": "现货供应紧张，便利收益高",
            "策略": "买入近月卖出远月的牛市套利",
        },
    }

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "主要品种": commodities,
        "基差分析框架": basis_framework,
        "基差交易要点": [
            "基差=现货价格-期货价格（国内常用定义）",
            "基差走强：现货相对期货上涨，利好现货多头",
            "基差走弱：现货相对期货下跌，利好期货多头",
            "临近交割月基差趋于收敛，是套利的安全期",
            "关注仓单变化，仓单增加通常压制近月价格",
        ],
        "季节性基差规律": {
            "螺纹钢": "11-12月冬储期间基差走强",
            "豆粕": "4-5月南美大豆上市基差走弱",
            "原油": "夏季出行旺季基差走强",
            "黄金": "节假日消费旺季基差走强",
        },
    }


def main():
    parser = argparse.ArgumentParser(description='期货基差与升贴水分析系统')
    subparsers = parser.add_subparsers(dest='command')

    # 基差计算
    calc_parser = subparsers.add_parser('calc', help='基差计算')
    calc_parser.add_argument('--futures', type=float, required=True, help='期货价格')
    calc_parser.add_argument('--spot', type=float, required=True, help='现货价格')

    # 股指期货基差
    index_parser = subparsers.add_parser('index', help='股指期货基差分析')

    # 商品期货基差
    commodity_parser = subparsers.add_parser('commodity', help='商品期货基差分析')

    args = parser.parse_args()

    if args.command == 'calc':
        result = calculate_basis(args.futures, args.spot)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'index':
        result = index_futures_basis()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == 'commodity':
        result = commodity_futures_basis()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
