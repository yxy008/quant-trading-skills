#!/usr/bin/env python3
"""
订单管理系统(OMS) - 订单拆分、TWAP/VWAP算法交易、撤单改单
"""
import argparse
import json
import sys
import os
import math
from datetime import datetime, timedelta

_agent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

try:
    import numpy as np
except ImportError:
    np = None

from data_utils import get_stock_kline


def twap_split(total_qty, total_slots, start_price=None, price_range=None):
    """
    TWAP（时间加权平均价格）拆单算法
    将大单拆分为多个时间片段的等量小单

    total_qty: 总数量（股）
    total_slots: 时间片数
    start_price: 起始价格
    price_range: 价格波动范围 (low, high)
    """
    if total_qty <= 0 or total_slots <= 0:
        return {"error": "数量和片数必须大于0"}

    lot_size = 100
    total_lots = total_qty // lot_size

    if total_lots < total_slots:
        return {"error": f"总手数({total_lots})小于时间片数({total_slots})，无法拆分"}

    # 基础每片手数
    base_lots = total_lots // total_slots
    remainder = total_lots % total_slots

    slots = []
    cumulative_qty = 0

    for i in range(total_slots):
        slot_lots = base_lots + (1 if i < remainder else 0)
        slot_qty = slot_lots * lot_size
        cumulative_qty += slot_qty

        # 模拟价格波动
        if start_price and price_range:
            price_low, price_high = price_range
            slot_price = start_price + np.random.uniform(
                (price_low - start_price) * 0.3,
                (price_high - start_price) * 0.3
            )
        elif start_price:
            slot_price = start_price * (1 + np.random.uniform(-0.005, 0.005))
        else:
            slot_price = None

        slots.append({
            "片序号": i + 1,
            "数量": slot_qty,
            "手数": slot_lots,
            "预估价格": round(slot_price, 2) if slot_price else None,
            "累计数量": cumulative_qty,
            "进度": round(cumulative_qty / total_qty * 100, 1),
        })

    avg_price = None
    if start_price:
        prices = [s["预估价格"] for s in slots if s["预估价格"]]
        if prices:
            avg_price = round(sum(prices) / len(prices), 2)

    return {
        "算法": "TWAP",
        "总数量": total_qty,
        "总手数": total_lots,
        "时间片数": total_slots,
        "每片基础手数": base_lots,
        "预估均价": avg_price,
        "拆分明细": slots,
        "生成时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def vwap_split(symbol, total_qty, total_slots, days=5):
    """
    VWAP（成交量加权平均价格）拆单算法
    根据历史成交量分布来分配各时间片的订单量

    symbol: 股票代码
    total_qty: 总数量
    total_slots: 时间片数
    days: 用于估算成交量分布的历史天数
    """
    if total_qty <= 0 or total_slots <= 0:
        return {"error": "数量和片数必须大于0"}

    lot_size = 100
    total_lots = total_qty // lot_size

    if total_lots < total_slots:
        return {"error": f"总手数({total_lots})小于时间片数({total_slots})，无法拆分"}

    # 获取历史成交量分布
    try:
        df = get_stock_kline(symbol, "daily", days + 10)
        if df is not None and not df.empty:
            volumes = df["volume"].tolist()[-days:]
            total_vol = sum(volumes)
            if total_vol > 0:
                vol_weights = [v / total_vol for v in volumes]
            else:
                vol_weights = [1.0 / days] * days
        else:
            vol_weights = [1.0 / days] * days
    except Exception:
        vol_weights = [1.0 / days] * days

    # 将日成交量分布映射到时间片
    slot_weights = []
    slots_per_day = max(1, total_slots // len(vol_weights))
    for w in vol_weights:
        for _ in range(slots_per_day):
            slot_weights.append(w / slots_per_day)
    # 截断或填充
    slot_weights = slot_weights[:total_slots]
    if len(slot_weights) < total_slots:
        avg_w = 1.0 / total_slots
        slot_weights.extend([avg_w] * (total_slots - len(slot_weights)))

    # 归一化
    weight_sum = sum(slot_weights)
    slot_weights = [w / weight_sum for w in slot_weights]

    # 分配手数
    allocated = []
    remaining = total_lots
    for i in range(total_slots):
        if i == total_slots - 1:
            slot_lots = remaining
        else:
            slot_lots = max(1, int(total_lots * slot_weights[i]))
            slot_lots = min(slot_lots, remaining - (total_slots - i - 1))
        remaining -= slot_lots
        allocated.append(slot_lots)

    slots = []
    cumulative_qty = 0
    for i, slot_lots in enumerate(allocated):
        slot_qty = slot_lots * lot_size
        cumulative_qty += slot_qty
        slots.append({
            "片序号": i + 1,
            "数量": slot_qty,
            "手数": slot_lots,
            "权重": round(slot_weights[i] * 100, 2),
            "累计数量": cumulative_qty,
            "进度": round(cumulative_qty / total_qty * 100, 1),
        })

    return {
        "算法": "VWAP",
        "股票代码": symbol,
        "总数量": total_qty,
        "总手数": total_lots,
        "时间片数": total_slots,
        "历史参考天数": days,
        "拆分明细": slots,
        "生成时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def iceberg_order(total_qty, visible_qty, price=None):
    """
    冰山订单
    只显示一小部分订单量，隐藏真实大单意图

    total_qty: 总数量
    visible_qty: 每次显示的数量
    price: 限价
    """
    if total_qty <= 0 or visible_qty <= 0:
        return {"error": "数量必须大于0"}

    if visible_qty > total_qty:
        visible_qty = total_qty

    lot_size = 100
    total_lots = total_qty // lot_size
    visible_lots = max(1, visible_qty // lot_size)

    waves = math.ceil(total_lots / visible_lots)

    wave_details = []
    remaining = total_lots
    for i in range(waves):
        wave_lots = min(visible_lots, remaining)
        wave_qty = wave_lots * lot_size
        remaining -= wave_lots
        wave_details.append({
            "波次": i + 1,
            "显示数量": wave_qty,
            "显示手数": wave_lots,
            "隐藏数量": total_qty - wave_qty * (i + 1),
            "累计成交": wave_qty * (i + 1),
            "进度": round(wave_qty * (i + 1) / total_qty * 100, 1),
        })

    return {
        "算法": "冰山订单",
        "总数量": total_qty,
        "每次显示": visible_qty,
        "波次数": waves,
        "限价": price,
        "执行计划": wave_details,
        "生成时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def smart_order(symbol, side, total_qty, urgency="normal", days=5):
    """
    智能下单
    根据市场情况自动选择最优下单算法

    urgency: 紧急程度 (low/normal/high)
    """
    try:
        df = get_stock_kline(symbol, "daily", days + 10)
        if df is not None and not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
            closes = df["close"].tolist()
            volumes = df["volume"].tolist()

            if len(closes) >= 5:
                avg_vol = sum(volumes[-5:]) / 5
                recent_change = (closes[-1] / closes[-5] - 1) * 100

                # 计算市场冲击成本
                participation_rate = total_qty / (avg_vol * 100) if avg_vol > 0 else 0.01
                if participation_rate > 0.1:
                    recommendation = "建议使用VWAP算法，降低市场冲击"
                    algo = "vwap"
                    slots = max(10, int(participation_rate * 50))
                elif participation_rate > 0.05:
                    recommendation = "建议使用TWAP算法，均衡执行"
                    algo = "twap"
                    slots = 8
                else:
                    recommendation = "可直接市价/限价下单"
                    algo = "direct"
                    slots = 1

                if urgency == "high":
                    slots = max(1, slots // 2)
                    recommendation += "（紧急模式：减少时间片）"
                elif urgency == "low":
                    slots = slots * 2
                    recommendation += "（低 urgency：增加时间片以降低冲击）"

                return {
                    "股票代码": symbol,
                    "方向": side,
                    "总数量": total_qty,
                    "紧急程度": urgency,
                    "日均成交量": int(avg_vol),
                    "参与率": round(participation_rate * 100, 2),
                    "推荐算法": algo,
                    "建议时间片": slots,
                    "建议": recommendation,
                    "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                }
    except Exception:
        pass

    return {
        "股票代码": symbol,
        "方向": side,
        "总数量": total_qty,
        "紧急程度": urgency,
        "推荐算法": "twap",
        "建议时间片": 5,
        "建议": "默认使用TWAP算法",
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def modify_order(order_id, new_price=None, new_qty=None):
    """
    改单
    修改已有订单的价格或数量
    """
    changes = {}
    if new_price is not None:
        changes["新价格"] = new_price
    if new_qty is not None:
        changes["新数量"] = new_qty

    return {
        "success": True,
        "订单ID": order_id,
        "修改内容": changes,
        "修改时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "提示": "实际改单需先撤原单再重新下单",
    }


def order_book_summary(orders):
    """
    订单簿汇总
    分析当前订单簿状态
    """
    if not orders:
        return {"订单总数": 0, "提示": "无活跃订单"}

    pending = [o for o in orders if o.get("状态") in ("待提交", "已提交", "部分成交")]
    filled = [o for o in orders if o.get("状态") == "全部成交"]
    cancelled = [o for o in orders if o.get("状态") == "已撤销"]

    buy_orders = [o for o in orders if o.get("方向") == "买入"]
    sell_orders = [o for o in orders if o.get("方向") == "卖出"]

    total_buy_qty = sum(o.get("数量", 0) for o in buy_orders)
    total_sell_qty = sum(o.get("数量", 0) for o in sell_orders)

    return {
        "订单总数": len(orders),
        "待处理": len(pending),
        "已成交": len(filled),
        "已撤销": len(cancelled),
        "买入订单": len(buy_orders),
        "卖出订单": len(sell_orders),
        "买入总量": total_buy_qty,
        "卖出总量": total_sell_qty,
        "净买入": total_buy_qty - total_sell_qty,
        "汇总时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def main():
    parser = argparse.ArgumentParser(description="订单管理系统(OMS)")
    subparsers = parser.add_subparsers(dest="action", help="操作")

    twap_parser = subparsers.add_parser("twap", help="TWAP拆单")
    twap_parser.add_argument("--qty", type=int, required=True, help="总数量")
    twap_parser.add_argument("--slots", type=int, required=True, help="时间片数")
    twap_parser.add_argument("--price", type=float, help="起始价格")
    twap_parser.add_argument("--range", help="价格范围JSON [low, high]")

    vwap_parser = subparsers.add_parser("vwap", help="VWAP拆单")
    vwap_parser.add_argument("--symbol", required=True, help="股票代码")
    vwap_parser.add_argument("--qty", type=int, required=True, help="总数量")
    vwap_parser.add_argument("--slots", type=int, required=True, help="时间片数")
    vwap_parser.add_argument("--days", type=int, default=5, help="历史参考天数")

    iceberg_parser = subparsers.add_parser("iceberg", help="冰山订单")
    iceberg_parser.add_argument("--qty", type=int, required=True, help="总数量")
    iceberg_parser.add_argument("--visible", type=int, required=True, help="显示数量")
    iceberg_parser.add_argument("--price", type=float, help="限价")

    smart_parser = subparsers.add_parser("smart", help="智能下单建议")
    smart_parser.add_argument("--symbol", required=True, help="股票代码")
    smart_parser.add_argument("--side", default="buy", help="买卖方向")
    smart_parser.add_argument("--qty", type=int, required=True, help="总数量")
    smart_parser.add_argument("--urgency", default="normal", help="紧急程度")

    modify_parser = subparsers.add_parser("modify", help="改单")
    modify_parser.add_argument("--order_id", required=True, help="订单ID")
    modify_parser.add_argument("--price", type=float, help="新价格")
    modify_parser.add_argument("--qty", type=int, help="新数量")

    summary_parser = subparsers.add_parser("summary", help="订单簿汇总")
    summary_parser.add_argument("--orders", default="[]", help="订单列表JSON")

    args = parser.parse_args()

    try:
        if args.action == "twap":
            price_range = json.loads(args.range) if args.range else None
            result = twap_split(args.qty, args.slots, args.price, price_range)
        elif args.action == "vwap":
            result = vwap_split(args.symbol, args.qty, args.slots, args.days)
        elif args.action == "iceberg":
            result = iceberg_order(args.qty, args.visible, args.price)
        elif args.action == "smart":
            result = smart_order(args.symbol, args.side, args.qty, args.urgency)
        elif args.action == "modify":
            result = modify_order(args.order_id, args.price, args.qty)
        elif args.action == "summary":
            orders = json.loads(args.orders)
            result = order_book_summary(orders)
        else:
            parser.print_help()
            return
    except json.JSONDecodeError as e:
        result = {"error": f"JSON解析失败: {str(e)}"}
    except Exception as e:
        result = {"error": str(e)}

    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
