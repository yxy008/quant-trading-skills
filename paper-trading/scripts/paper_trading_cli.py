#!/usr/bin/env python3
"""
模拟交易系统 - 虚拟账户、订单管理、实时盈亏跟踪
"""
import argparse
import json
import sys
import os
import time
from datetime import datetime, timedelta
from collections import defaultdict

_agent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("请先安装依赖: pip install pandas numpy")
    sys.exit(1)

from data_utils import get_stock_kline, get_realtime_quote
from db_utils import get_db_connection


class PaperTradingAccount:
    """模拟交易账户"""

    def __init__(self, account_id=None, initial_capital=100000, account_name="默认账户"):
        self.account_id = account_id or f"PT_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.account_name = account_name
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {symbol: {"shares": int, "avg_cost": float, "market_value": float}}
        self.orders = []  # 订单历史
        self.trades = []  # 成交历史
        self.created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._load_from_db()

    def _load_from_db(self):
        """从数据库加载账户数据"""
        try:
            conn = get_db_connection()
            if conn is None:
                return
            cursor = conn.cursor()
            cursor.execute(
                "SELECT cash, initial_capital, account_name FROM paper_accounts WHERE account_id = %s",
                (self.account_id,)
            )
            row = cursor.fetchone()
            if row:
                self.cash = float(row[0])
                self.initial_capital = float(row[1])
                self.account_name = row[2]

            cursor.execute(
                "SELECT symbol, shares, avg_cost FROM paper_positions WHERE account_id = %s",
                (self.account_id,)
            )
            for row in cursor.fetchall():
                self.positions[row[0]] = {
                    "shares": int(row[1]),
                    "avg_cost": float(row[2]),
                    "market_value": 0
                }

            cursor.execute(
                "SELECT order_id, symbol, order_type, direction, price, quantity, "
                "filled_quantity, status, created_at FROM paper_orders "
                "WHERE account_id = %s ORDER BY created_at DESC LIMIT 200",
                (self.account_id,)
            )
            self.orders = [
                {
                    "order_id": row[0], "symbol": row[1], "order_type": row[2],
                    "direction": row[3], "price": float(row[4]), "quantity": int(row[5]),
                    "filled_quantity": int(row[6]), "status": row[7], "created_at": str(row[8])
                }
                for row in cursor.fetchall()
            ]

            cursor.execute(
                "SELECT trade_id, symbol, direction, price, quantity, amount, "
                "fee_detail, traded_at FROM paper_trades "
                "WHERE account_id = %s ORDER BY traded_at DESC LIMIT 200",
                (self.account_id,)
            )
            self.trades = [
                {
                    "trade_id": row[0], "symbol": row[1], "direction": row[2],
                    "price": float(row[3]), "quantity": int(row[4]),
                    "amount": float(row[5]), "fee_detail": row[6], "traded_at": str(row[7])
                }
                for row in cursor.fetchall()
            ]

            cursor.close()
            conn.close()
        except Exception:
            pass

    def _save_account(self):
        """保存账户到数据库"""
        try:
            conn = get_db_connection()
            if conn is None:
                return
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO paper_accounts (account_id, account_name, cash, initial_capital)
                   VALUES (%s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE cash = VALUES(cash), account_name = VALUES(account_name)""",
                (self.account_id, self.account_name, self.cash, self.initial_capital)
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception:
            pass

    def _save_position(self, symbol):
        """保存持仓到数据库"""
        try:
            conn = get_db_connection()
            if conn is None:
                return
            cursor = conn.cursor()
            if symbol in self.positions and self.positions[symbol]["shares"] > 0:
                pos = self.positions[symbol]
                cursor.execute(
                    """INSERT INTO paper_positions (account_id, symbol, shares, avg_cost)
                       VALUES (%s, %s, %s, %s)
                       ON DUPLICATE KEY UPDATE shares = VALUES(shares), avg_cost = VALUES(avg_cost)""",
                    (self.account_id, symbol, pos["shares"], pos["avg_cost"])
                )
            else:
                cursor.execute(
                    "DELETE FROM paper_positions WHERE account_id = %s AND symbol = %s",
                    (self.account_id, symbol)
                )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception:
            pass

    def _save_order(self, order):
        """保存订单到数据库"""
        try:
            conn = get_db_connection()
            if conn is None:
                return
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO paper_orders (account_id, order_id, symbol, order_type, direction,
                   price, quantity, filled_quantity, status)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE filled_quantity = VALUES(filled_quantity),
                   status = VALUES(status)""",
                (self.account_id, order["order_id"], order["symbol"], order["order_type"],
                 order["direction"], order["price"], order["quantity"],
                 order["filled_quantity"], order["status"])
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception:
            pass

    def _save_trade(self, trade):
        """保存成交记录到数据库"""
        try:
            conn = get_db_connection()
            if conn is None:
                return
            cursor = conn.cursor()
            fee_json = json.dumps(trade.get("fee_detail", {}), ensure_ascii=False)
            cursor.execute(
                """INSERT INTO paper_trades (account_id, trade_id, symbol, direction, price,
                   quantity, amount, fee_detail)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (self.account_id, trade["trade_id"], trade["symbol"], trade["direction"],
                 trade["price"], trade["quantity"], trade["amount"], fee_json)
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception:
            pass

    def place_order(self, symbol, direction, quantity, order_type="market", price=None):
        """
        下单
        参数:
            symbol: 股票代码
            direction: buy/sell
            quantity: 数量（股）
            order_type: market/limit
            price: 限价（限价单时需要）
        """
        symbol = symbol.strip()
        quantity = int(quantity)

        if quantity < 100 or quantity % 100 != 0:
            return {"error": "数量必须为100的整数倍"}

        if order_type == "limit" and price is None:
            return {"error": "限价单需要指定价格"}

        order_id = f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "order_type": order_type,
            "direction": direction,
            "price": float(price) if price else 0,
            "quantity": quantity,
            "filled_quantity": 0,
            "status": "pending",
            "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        self.orders.insert(0, order)
        self._save_order(order)

        # 市价单立即成交
        if order_type == "market":
            self._execute_order(order)

        return {"success": True, "order": order}

    def _execute_order(self, order):
        """执行订单成交"""
        symbol = order["symbol"]
        direction = order["direction"]
        quantity = order["quantity"]

        quote = get_realtime_quote(symbol)
        if quote is None:
            order["status"] = "rejected"
            order["message"] = "无法获取实时行情"
            self._save_order(order)
            return

        current_price = quote.get("最新价", quote.get("latest_price", 0))
        if current_price <= 0:
            order["status"] = "rejected"
            order["message"] = "无效的实时价格"
            self._save_order(order)
            return

        # 计算费用
        trade_amount = quantity * current_price
        commission = max(trade_amount * 0.0003, 5)
        stamp_tax = trade_amount * 0.001 if direction == "sell" else 0
        transfer_fee = max(trade_amount * 0.00001, 0.1)
        total_fee = commission + stamp_tax + transfer_fee

        fee_detail = {
            "佣金": round(commission, 2),
            "印花税": round(stamp_tax, 2),
            "过户费": round(transfer_fee, 2),
            "合计": round(total_fee, 2)
        }

        if direction == "buy":
            total_cost = trade_amount + total_fee
            if total_cost > self.cash:
                order["status"] = "rejected"
                order["message"] = f"资金不足，需要 {total_cost:.2f}，可用 {self.cash:.2f}"
                self._save_order(order)
                return

            self.cash -= total_cost
            if symbol in self.positions:
                old_shares = self.positions[symbol]["shares"]
                old_cost = self.positions[symbol]["avg_cost"]
                new_shares = old_shares + quantity
                new_cost = (old_cost * old_shares + trade_amount) / new_shares
                self.positions[symbol]["shares"] = new_shares
                self.positions[symbol]["avg_cost"] = new_cost
            else:
                self.positions[symbol] = {
                    "shares": quantity,
                    "avg_cost": current_price,
                    "market_value": trade_amount
                }
        else:
            if symbol not in self.positions or self.positions[symbol]["shares"] < quantity:
                order["status"] = "rejected"
                order["message"] = f"持仓不足，需要 {quantity}股，持有 {self.positions.get(symbol, {}).get('shares', 0)}股"
                self._save_order(order)
                return

            net_amount = trade_amount - total_fee
            self.cash += net_amount
            self.positions[symbol]["shares"] -= quantity
            if self.positions[symbol]["shares"] <= 0:
                del self.positions[symbol]

        order["filled_quantity"] = quantity
        order["status"] = "filled"
        order["filled_price"] = current_price
        order["fee_detail"] = fee_detail
        self._save_order(order)

        trade_id = f"TRD_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        trade = {
            "trade_id": trade_id,
            "symbol": symbol,
            "direction": direction,
            "price": current_price,
            "quantity": quantity,
            "amount": trade_amount,
            "fee_detail": fee_detail,
            "traded_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.trades.insert(0, trade)
        self._save_trade(trade)

        self._save_account()
        self._save_position(symbol)

    def cancel_order(self, order_id):
        """撤销订单"""
        for order in self.orders:
            if order["order_id"] == order_id:
                if order["status"] in ("filled", "cancelled", "rejected"):
                    return {"error": f"订单状态为 {order['status']}，无法撤销"}
                order["status"] = "cancelled"
                self._save_order(order)
                return {"success": True, "order": order}
        return {"error": "订单不存在"}

    def update_market_values(self):
        """更新持仓市值"""
        for symbol in list(self.positions.keys()):
            quote = get_realtime_quote(symbol)
            price = quote.get("最新价", quote.get("latest_price", 0)) if quote else 0
            if price > 0:
                self.positions[symbol]["market_value"] = (
                    self.positions[symbol]["shares"] * price
                )

    def get_summary(self):
        """获取账户摘要"""
        self.update_market_values()

        total_market_value = sum(p["market_value"] for p in self.positions.values())
        total_equity = self.cash + total_market_value
        total_return = (total_equity / self.initial_capital - 1) * 100

        # 计算各持仓盈亏
        position_list = []
        for symbol, pos in self.positions.items():
            cost = pos["shares"] * pos["avg_cost"]
            pnl = pos["market_value"] - cost
            pnl_pct = (pnl / cost * 100) if cost > 0 else 0
            position_list.append({
                "股票代码": symbol,
                "持仓数量": pos["shares"],
                "成本价": round(pos["avg_cost"], 2),
                "市值": round(pos["market_value"], 2),
                "盈亏": round(pnl, 2),
                "盈亏比例": round(pnl_pct, 2)
            })

        return {
            "账户ID": self.account_id,
            "账户名称": self.account_name,
            "初始资金": self.initial_capital,
            "可用资金": round(self.cash, 2),
            "持仓市值": round(total_market_value, 2),
            "总资产": round(total_equity, 2),
            "总收益率": round(total_return, 2),
            "持仓明细": position_list,
            "创建时间": self.created_at
        }

    def get_positions(self):
        """获取持仓列表"""
        self.update_market_values()
        result = []
        for symbol, pos in self.positions.items():
            cost = pos["shares"] * pos["avg_cost"]
            pnl = pos["market_value"] - cost
            pnl_pct = (pnl / cost * 100) if cost > 0 else 0
            result.append({
                "股票代码": symbol,
                "持仓数量": pos["shares"],
                "成本价": round(pos["avg_cost"], 2),
                "当前价": round(pos["market_value"] / pos["shares"], 2) if pos["shares"] > 0 else 0,
                "市值": round(pos["market_value"], 2),
                "盈亏": round(pnl, 2),
                "盈亏比例": round(pnl_pct, 2)
            })
        return result

    def get_orders(self, limit=50):
        """获取订单列表"""
        return self.orders[:limit]

    def get_trades(self, limit=50):
        """获取成交列表"""
        return self.trades[:limit]


# 全局账户缓存
_accounts = {}


def get_or_create_account(account_id=None, initial_capital=100000, account_name="默认账户"):
    """获取或创建模拟账户"""
    if account_id and account_id in _accounts:
        return _accounts[account_id]

    account = PaperTradingAccount(
        account_id=account_id,
        initial_capital=initial_capital,
        account_name=account_name
    )
    _accounts[account.account_id] = account
    return account


def init_paper_trading_tables():
    """初始化模拟交易数据库表"""
    try:
        conn = get_db_connection()
        if conn is None:
            return False
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_accounts (
                account_id VARCHAR(64) PRIMARY KEY,
                account_name VARCHAR(128) DEFAULT '默认账户',
                cash DECIMAL(16,2) DEFAULT 100000,
                initial_capital DECIMAL(16,2) DEFAULT 100000,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_positions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                account_id VARCHAR(64) NOT NULL,
                symbol VARCHAR(16) NOT NULL,
                shares INT DEFAULT 0,
                avg_cost DECIMAL(10,4) DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uk_account_symbol (account_id, symbol)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_orders (
                id INT AUTO_INCREMENT PRIMARY KEY,
                account_id VARCHAR(64) NOT NULL,
                order_id VARCHAR(64) NOT NULL UNIQUE,
                symbol VARCHAR(16) NOT NULL,
                order_type VARCHAR(16) DEFAULT 'market',
                direction VARCHAR(8) NOT NULL,
                price DECIMAL(10,4) DEFAULT 0,
                quantity INT DEFAULT 0,
                filled_quantity INT DEFAULT 0,
                status VARCHAR(16) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_account (account_id),
                INDEX idx_status (status)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INT AUTO_INCREMENT PRIMARY KEY,
                account_id VARCHAR(64) NOT NULL,
                trade_id VARCHAR(64) NOT NULL UNIQUE,
                symbol VARCHAR(16) NOT NULL,
                direction VARCHAR(8) NOT NULL,
                price DECIMAL(10,4) DEFAULT 0,
                quantity INT DEFAULT 0,
                amount DECIMAL(16,2) DEFAULT 0,
                fee_detail JSON,
                traded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_account (account_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"初始化模拟交易表失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="模拟交易系统")
    subparsers = parser.add_subparsers(dest="action", help="操作")

    # 初始化
    subparsers.add_parser("init", help="初始化数据库表")

    # 创建账户
    create_parser = subparsers.add_parser("create", help="创建模拟账户")
    create_parser.add_argument("--capital", type=float, default=100000, help="初始资金")
    create_parser.add_argument("--name", default="默认账户", help="账户名称")

    # 下单
    order_parser = subparsers.add_parser("order", help="下单")
    order_parser.add_argument("--account", default=None, help="账户ID")
    order_parser.add_argument("--symbol", required=True, help="股票代码")
    order_parser.add_argument("--direction", required=True, choices=["buy", "sell"], help="买卖方向")
    order_parser.add_argument("--quantity", type=int, required=True, help="数量")
    order_parser.add_argument("--type", default="market", choices=["market", "limit"], help="订单类型")
    order_parser.add_argument("--price", type=float, default=None, help="限价")

    # 查询
    summary_parser = subparsers.add_parser("summary", help="账户摘要")
    summary_parser.add_argument("--account", default=None, help="账户ID")

    positions_parser = subparsers.add_parser("positions", help="持仓列表")
    positions_parser.add_argument("--account", default=None, help="账户ID")

    orders_parser = subparsers.add_parser("orders", help="订单列表")
    orders_parser.add_argument("--account", default=None, help="账户ID")

    trades_parser = subparsers.add_parser("trades", help="成交列表")
    trades_parser.add_argument("--account", default=None, help="账户ID")

    # 撤销
    cancel_parser = subparsers.add_parser("cancel", help="撤销订单")
    cancel_parser.add_argument("--account", default=None, help="账户ID")
    cancel_parser.add_argument("--order_id", required=True, help="订单ID")

    args = parser.parse_args()

    if args.action == "init":
        success = init_paper_trading_tables()
        print(json.dumps({"success": success}, ensure_ascii=False))

    elif args.action == "create":
        account = get_or_create_account(
            initial_capital=args.capital,
            account_name=args.name
        )
        print(json.dumps(account.get_summary(), ensure_ascii=False, indent=2))

    elif args.action == "order":
        account = get_or_create_account(account_id=args.account)
        result = account.place_order(
            symbol=args.symbol,
            direction=args.direction,
            quantity=args.quantity,
            order_type=args.type,
            price=args.price
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.action == "summary":
        account = get_or_create_account(account_id=args.account)
        print(json.dumps(account.get_summary(), ensure_ascii=False, indent=2))

    elif args.action == "positions":
        account = get_or_create_account(account_id=args.account)
        print(json.dumps(account.get_positions(), ensure_ascii=False, indent=2))

    elif args.action == "orders":
        account = get_or_create_account(account_id=args.account)
        print(json.dumps(account.get_orders(), ensure_ascii=False, indent=2, default=str))

    elif args.action == "trades":
        account = get_or_create_account(account_id=args.account)
        print(json.dumps(account.get_trades(), ensure_ascii=False, indent=2, default=str))

    elif args.action == "cancel":
        account = get_or_create_account(account_id=args.account)
        result = account.cancel_order(args.order_id)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
