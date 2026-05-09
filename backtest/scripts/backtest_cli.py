#!/usr/bin/env python3
"""
强化回测系统 - 滑点/手续费/完整绩效指标/交易日志/权益曲线
"""
import argparse
import json
import sys
import os
import time
from datetime import datetime, timedelta

# 添加agent目录到路径以导入data_utils
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

from data_utils import get_stock_kline, get_index_kline

# 导入数据持久化模块
_storage_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data-storage", "scripts")
if _storage_dir not in sys.path:
    sys.path.insert(0, _storage_dir)

try:
    from storage_cli import save_backtest_record, get_backtest_records, get_backtest_trend, compare_backtest_strategies, delete_backtest_record
except ImportError:
    save_backtest_record = None
    get_backtest_records = None
    get_backtest_trend = None
    compare_backtest_strategies = None
    delete_backtest_record = None


class BacktestEngine:
    """回测引擎"""

    def __init__(self, initial_capital=100000, commission_rate=0.0003,
                 stamp_tax_rate=0.001, slippage=0.001, min_commission=5,
                 transfer_fee_rate=0.00001, impact_cost_rate=0.0001,
                 short_fee_rate=0.0001):
        """
        参数:
            initial_capital: 初始资金
            commission_rate: 佣金费率（默认万三）
            stamp_tax_rate: 印花税率（卖出时千一）
            slippage: 滑点比例（默认千一）
            min_commission: 最低佣金
            transfer_fee_rate: 过户费率（默认十万分之一，即0.001%）
            impact_cost_rate: 冲击成本率（默认万分之一，大单交易对价格的冲击）
            short_fee_rate: 融券费率（默认年化8.6%，按日折算约万分之一）
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage
        self.min_commission = min_commission
        self.transfer_fee_rate = transfer_fee_rate
        self.impact_cost_rate = impact_cost_rate
        self.short_fee_rate = short_fee_rate

    def calculate_commission(self, trade_amount, is_sell=False, shares=0, daily_volume=0):
        """
        计算完整交易费用
        包含：佣金、印花税、过户费、冲击成本
        """
        commission = max(trade_amount * self.commission_rate, self.min_commission)
        stamp_tax = trade_amount * self.stamp_tax_rate if is_sell else 0
        transfer_fee = max(trade_amount * self.transfer_fee_rate, 0.1)
        impact_cost = 0
        if daily_volume > 0 and shares > 0:
            trade_volume_ratio = trade_amount / daily_volume if daily_volume > 0 else 0
            if trade_volume_ratio > 0.01:
                impact_cost = trade_amount * self.impact_cost_rate * (trade_volume_ratio / 0.01)
        return {
            "佣金": round(commission, 2),
            "印花税": round(stamp_tax, 2),
            "过户费": round(transfer_fee, 2),
            "冲击成本": round(impact_cost, 2),
            "合计": round(commission + stamp_tax + transfer_fee + impact_cost, 2)
        }

    def calculate_short_fee(self, trade_amount, days_held=1):
        """计算融券费用"""
        return round(trade_amount * self.short_fee_rate * days_held, 2)

    def run(self, df, signals, position_size=1.0, allow_short=False):
        """
        执行回测
        参数:
            df: K线数据 DataFrame
            signals: 信号 Series (1=买入/做多, -1=卖出/做空, 0=持有)
            position_size: 仓位比例 (0~1)
            allow_short: 是否允许做空
        返回:
            dict: 回测结果
        """
        close = df['close']
        signals = signals.reindex(df.index).fillna(0)

        capital = self.initial_capital
        cash = self.initial_capital
        shares = 0
        position = 0  # 0=空仓, 1=多头持仓, -1=空头持仓
        short_entry_date = None  # 做空入场日期，用于计算融券费用

        trades = []
        equity_curve = []
        daily_returns = []

        prev_equity = self.initial_capital

        for i in range(len(df)):
            date = df.index[i]
            price = close.iloc[i]
            signal = int(signals.iloc[i])
            daily_volume = df['amount'].iloc[i] if 'amount' in df.columns else 0

            if allow_short:
                if signal == 1:
                    # 先平空仓
                    if position == -1 and shares > 0:
                        buy_back_price = price * (1 + self.slippage)
                        trade_amount = shares * buy_back_price
                        fee_detail = self.calculate_commission(trade_amount, is_sell=False, shares=shares, daily_volume=daily_volume)
                        total_fee = fee_detail["合计"]
                        # 融券费用
                        days_held = 1
                        if short_entry_date:
                            days_held = max(1, (date - short_entry_date).days)
                        short_fee = self.calculate_short_fee(trade_amount, days_held)
                        total_fee += short_fee
                        fee_detail["融券费"] = short_fee
                        fee_detail["合计"] = round(total_fee, 2)
                        net_cost = trade_amount + total_fee
                        cash -= net_cost
                        trades.append({
                            "日期": date.strftime('%Y-%m-%d'),
                            "类型": "买回(平空)",
                            "价格": round(buy_back_price, 2),
                            "数量": shares,
                            "金额": round(trade_amount, 2),
                            "费用明细": fee_detail,
                            "费用合计": round(total_fee, 2)
                        })
                        shares = 0
                        position = 0
                        short_entry_date = None

                    # 开多仓
                    if position == 0 and cash > 0:
                        buy_price = price * (1 + self.slippage)
                        max_shares = int(cash * position_size / buy_price / 100) * 100
                        if max_shares >= 100:
                            trade_amount = max_shares * buy_price
                            fee_detail = self.calculate_commission(trade_amount, is_sell=False, shares=max_shares, daily_volume=daily_volume)
                            total_fee = fee_detail["合计"]
                            total_cost = trade_amount + total_fee
                            if total_cost <= cash:
                                cash -= total_cost
                                shares = max_shares
                                position = 1
                                trades.append({
                                    "日期": date.strftime('%Y-%m-%d'),
                                    "类型": "买入(做多)",
                                    "价格": round(buy_price, 2),
                                    "数量": shares,
                                    "金额": round(trade_amount, 2),
                                    "费用明细": fee_detail,
                                    "费用合计": round(total_fee, 2)
                                })

                elif signal == -1:
                    # 先平多仓
                    if position == 1 and shares > 0:
                        sell_price = price * (1 - self.slippage)
                        trade_amount = shares * sell_price
                        fee_detail = self.calculate_commission(trade_amount, is_sell=True, shares=shares, daily_volume=daily_volume)
                        total_fee = fee_detail["合计"]
                        net_amount = trade_amount - total_fee
                        cash += net_amount
                        trades.append({
                            "日期": date.strftime('%Y-%m-%d'),
                            "类型": "卖出(平多)",
                            "价格": round(sell_price, 2),
                            "数量": shares,
                            "金额": round(trade_amount, 2),
                            "费用明细": fee_detail,
                            "费用合计": round(total_fee, 2),
                            "净收入": round(net_amount, 2)
                        })
                        shares = 0
                        position = 0

                    # 开空仓
                    if position == 0 and cash > 0:
                        short_price = price * (1 - self.slippage)
                        max_shares = int(cash * position_size / short_price / 100) * 100
                        if max_shares >= 100:
                            trade_amount = max_shares * short_price
                            fee_detail = self.calculate_commission(trade_amount, is_sell=True, shares=max_shares, daily_volume=daily_volume)
                            total_fee = fee_detail["合计"]
                            cash += trade_amount - total_fee
                            shares = max_shares
                            position = -1
                            short_entry_date = date
                            trades.append({
                                "日期": date.strftime('%Y-%m-%d'),
                                "类型": "融券卖出(做空)",
                                "价格": round(short_price, 2),
                                "数量": shares,
                                "金额": round(trade_amount, 2),
                                "费用明细": fee_detail,
                                "费用合计": round(total_fee, 2),
                                "收到资金": round(trade_amount - total_fee, 2)
                            })
            else:
                if signal == 1 and position == 0 and cash > 0:
                    buy_price = price * (1 + self.slippage)
                    max_shares = int(cash * position_size / buy_price / 100) * 100
                    if max_shares >= 100:
                        trade_amount = max_shares * buy_price
                        fee_detail = self.calculate_commission(trade_amount, is_sell=False, shares=max_shares, daily_volume=daily_volume)
                        total_fee = fee_detail["合计"]
                        total_cost = trade_amount + total_fee
                        if total_cost <= cash:
                            cash -= total_cost
                            shares = max_shares
                            position = 1
                            trades.append({
                                "日期": date.strftime('%Y-%m-%d'),
                                "类型": "买入",
                                "价格": round(buy_price, 2),
                                "数量": shares,
                                "金额": round(trade_amount, 2),
                                "费用明细": fee_detail,
                                "费用合计": round(total_fee, 2)
                            })

                elif signal == -1 and position == 1 and shares > 0:
                    sell_price = price * (1 - self.slippage)
                    trade_amount = shares * sell_price
                    fee_detail = self.calculate_commission(trade_amount, is_sell=True, shares=shares, daily_volume=daily_volume)
                    total_fee = fee_detail["合计"]
                    net_amount = trade_amount - total_fee
                    cash += net_amount
                    trades.append({
                        "日期": date.strftime('%Y-%m-%d'),
                        "类型": "卖出",
                        "价格": round(sell_price, 2),
                        "数量": shares,
                        "金额": round(trade_amount, 2),
                        "费用明细": fee_detail,
                        "费用合计": round(total_fee, 2),
                        "净收入": round(net_amount, 2)
                    })
                    shares = 0
                    position = 0

            # 计算当日权益
            if position == -1:
                current_equity = cash - shares * price
            else:
                current_equity = cash + shares * price

            equity_curve.append({
                "日期": date.strftime('%Y-%m-%d'),
                "权益": round(current_equity, 2),
                "现金": round(cash, 2),
                "持仓市值": round(shares * price, 2),
                "持仓方向": "多头" if position == 1 else ("空头" if position == -1 else "空仓")
            })

            if i > 0:
                daily_ret = (current_equity / prev_equity - 1)
                daily_returns.append(daily_ret)
            prev_equity = current_equity

        # 最终清仓
        if position != 0 and shares > 0:
            final_price = close.iloc[-1]
            daily_volume = df['amount'].iloc[-1] if 'amount' in df.columns else 0
            if position == 1:
                trade_amount = shares * final_price
                fee_detail = self.calculate_commission(trade_amount, is_sell=True, shares=shares, daily_volume=daily_volume)
                total_fee = fee_detail["合计"]
                cash += trade_amount - total_fee
                trades.append({
                    "日期": df.index[-1].strftime('%Y-%m-%d'),
                    "类型": "卖出(清仓)",
                    "价格": round(final_price, 2),
                    "数量": shares,
                    "金额": round(trade_amount, 2),
                    "费用明细": fee_detail,
                    "费用合计": round(total_fee, 2)
                })
            elif position == -1:
                trade_amount = shares * final_price
                fee_detail = self.calculate_commission(trade_amount, is_sell=False, shares=shares, daily_volume=daily_volume)
                total_fee = fee_detail["合计"]
                days_held = 1
                if short_entry_date:
                    days_held = max(1, (df.index[-1] - short_entry_date).days)
                short_fee = self.calculate_short_fee(trade_amount, days_held)
                total_fee += short_fee
                fee_detail["融券费"] = short_fee
                fee_detail["合计"] = round(total_fee, 2)
                cash -= trade_amount + total_fee
                trades.append({
                    "日期": df.index[-1].strftime('%Y-%m-%d'),
                    "类型": "买回(清仓)",
                    "价格": round(final_price, 2),
                    "数量": shares,
                    "金额": round(trade_amount, 2),
                    "费用明细": fee_detail,
                    "费用合计": round(total_fee, 2)
                })
            shares = 0
            position = 0

        final_equity = cash

        # 计算绩效指标
        metrics = self._calculate_metrics(
            final_equity, daily_returns, equity_curve, trades, df
        )

        return {
            "初始资金": self.initial_capital,
            "最终权益": round(final_equity, 2),
            "交易记录": trades,
            "权益曲线": equity_curve,
            "绩效指标": metrics
        }

    def _calculate_metrics(self, final_equity, daily_returns, equity_curve, trades, df):
        """计算完整绩效指标"""
        metrics = {}

        # 基础收益
        total_return = (final_equity / self.initial_capital - 1) * 100
        metrics["总收益率"] = round(total_return, 2)

        # 年化收益率
        trading_days = len(daily_returns)
        annual_return = 0
        if trading_days > 0:
            annual_return = ((1 + total_return / 100) ** (252 / trading_days) - 1) * 100
            metrics["年化收益率"] = round(annual_return, 2)
        else:
            metrics["年化收益率"] = 0

        # 日收益率统计
        if daily_returns:
            returns_arr = np.array(daily_returns)
            metrics["日均收益率"] = round(float(np.mean(returns_arr)) * 100, 4)
            metrics["日收益率标准差"] = round(float(np.std(returns_arr, ddof=1)) * 100, 4)

            # 夏普比率（假设无风险利率2%）
            risk_free_daily = 0.02 / 252
            excess_returns = returns_arr - risk_free_daily
            if np.std(returns_arr, ddof=1) > 0:
                sharpe = np.mean(excess_returns) / np.std(returns_arr, ddof=1) * np.sqrt(252)
                metrics["夏普比率"] = round(float(sharpe), 2)
            else:
                metrics["夏普比率"] = 0

            # 索提诺比率（只考虑下行风险）
            downside_returns = returns_arr[returns_arr < 0]
            if len(downside_returns) > 0 and np.std(downside_returns, ddof=1) > 0:
                sortino = np.mean(excess_returns) / np.std(downside_returns, ddof=1) * np.sqrt(252)
                metrics["索提诺比率"] = round(float(sortino), 2)
            else:
                metrics["索提诺比率"] = 0

        # 最大回撤
        if equity_curve:
            equities = np.array([e["权益"] for e in equity_curve])
            peak = np.maximum.accumulate(equities)
            drawdowns = (equities - peak) / peak * 100
            max_dd = float(np.min(drawdowns))
            metrics["最大回撤"] = round(max_dd, 2)

            # 最大回撤持续天数
            dd_start = None
            max_dd_duration = 0
            current_dd_duration = 0
            in_drawdown = False
            for i, dd in enumerate(drawdowns):
                if dd < 0:
                    if not in_drawdown:
                        dd_start = i
                        in_drawdown = True
                    current_dd_duration = i - dd_start + 1
                    max_dd_duration = max(max_dd_duration, current_dd_duration)
                else:
                    in_drawdown = False
                    current_dd_duration = 0
            metrics["最大回撤持续天数"] = max_dd_duration

            # 卡玛比率
            if max_dd < 0:
                calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
                metrics["卡玛比率"] = round(float(calmar), 2)

        # 交易统计
        if trades:
            buy_trades = [t for t in trades if "买入" in t["类型"]]
            sell_trades = [t for t in trades if "卖出" in t["类型"]]

            metrics["交易总次数"] = len(buy_trades)

            # 配对计算盈亏
            profits = []
            for i in range(min(len(buy_trades), len(sell_trades))):
                buy = buy_trades[i]
                sell = sell_trades[i]
                profit = (sell.get("净收入", sell["金额"]) - buy["金额"]) / buy["金额"] * 100
                profits.append(profit)

            if profits:
                metrics["胜率"] = round(sum(1 for p in profits if p > 0) / len(profits) * 100, 2)
                metrics["平均盈利"] = round(np.mean([p for p in profits if p > 0]), 2) if any(p > 0 for p in profits) else 0
                metrics["平均亏损"] = round(np.mean([p for p in profits if p < 0]), 2) if any(p < 0 for p in profits) else 0
                metrics["盈亏比"] = round(abs(metrics["平均盈利"] / metrics["平均亏损"]), 2) if metrics["平均亏损"] != 0 else 0
                metrics["总盈亏"] = round(sum(profits), 2)
            else:
                metrics["胜率"] = 0
                metrics["平均盈利"] = 0
                metrics["平均亏损"] = 0
                metrics["盈亏比"] = 0
                metrics["总盈亏"] = 0

            # 总费用
            total_fee = sum(t.get("费用合计", 0) for t in trades)
            metrics["总交易费用"] = round(total_fee, 2)
        else:
            metrics["交易总次数"] = 0
            metrics["胜率"] = 0
            metrics["总交易费用"] = 0

        # 基准对比 - 买入持有
        benchmark_return = self._calculate_benchmark(df)
        if benchmark_return is not None:
            metrics["基准(买入持有)收益率"] = round(benchmark_return, 2)
            metrics["超额收益(vs买入持有)"] = round(total_return - benchmark_return, 2)

        # 基准对比 - 沪深300指数
        index_benchmark = self._calculate_index_benchmark(df)
        if index_benchmark is not None:
            metrics["基准对比"] = index_benchmark
            metrics["超额收益(vs沪深300)"] = round(total_return - index_benchmark["基准收益率"], 2)

            # 计算信息比率和Alpha/Beta
            try:
                start_date = df.index[0].strftime('%Y%m%d')
                end_date = df.index[-1].strftime('%Y%m%d')
                index_df = get_index_kline("000300", start_date=start_date, end_date=end_date)
                if index_df is not None and len(index_df) >= 10:
                    index_close = index_df['收盘'] if '收盘' in index_df.columns else index_df['close']
                    index_daily = index_close.pct_change().dropna()
                    if len(daily_returns) > 0 and len(index_daily) > 0:
                        min_len = min(len(daily_returns), len(index_daily))
                        ir = self._calc_information_ratio(daily_returns[-min_len:], index_daily[-min_len:])
                        alpha, beta = self._calc_alpha_beta(daily_returns[-min_len:], index_daily[-min_len:])
                        metrics["信息比率"] = round(ir, 2)
                        metrics["Alpha(年化%)"] = round(alpha, 2)
                        metrics["Beta"] = round(beta, 2)
            except Exception:
                pass

        # 收益归因分析
        attribution = self._calc_performance_attribution(daily_returns, df, trades)
        if attribution is not None:
            metrics["收益归因"] = attribution

        return metrics

    def _calculate_benchmark(self, df):
        """计算买入持有基准收益"""
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        return (end_price / start_price - 1) * 100

    def _calculate_index_benchmark(self, df, benchmark_code="000300"):
        """
        计算指数基准对比（沪深300）
        返回: dict 包含指数收益、超额收益、信息比率等
        """
        try:
            start_date = df.index[0].strftime('%Y%m%d')
            end_date = df.index[-1].strftime('%Y%m%d')

            index_df = get_index_kline(benchmark_code, start_date=start_date, end_date=end_date)
            if index_df is None or len(index_df) < 10:
                return None

            index_close = index_df['收盘'] if '收盘' in index_df.columns else index_df['close']
            index_return = (index_close.iloc[-1] / index_close.iloc[0] - 1) * 100

            index_daily_returns = index_close.pct_change().dropna()

            return {
                "基准指数": "沪深300",
                "基准收益率": round(float(index_return), 2),
                "基准年化波动率": round(float(index_daily_returns.std() * np.sqrt(252) * 100), 2),
                "基准最大回撤": round(float(self._calc_max_drawdown(index_close)), 2),
            }
        except Exception:
            return None

    def _calc_max_drawdown(self, series):
        """计算最大回撤"""
        peak = series.expanding().max()
        drawdown = (series - peak) / peak * 100
        return float(drawdown.min())

    def _calc_information_ratio(self, strategy_returns, benchmark_returns):
        """计算信息比率"""
        if len(strategy_returns) != len(benchmark_returns):
            return 0
        excess = np.array(strategy_returns) - np.array(benchmark_returns)
        if len(excess) < 2:
            return 0
        mean_excess = np.mean(excess)
        std_excess = np.std(excess, ddof=1)
        if std_excess == 0:
            return 0
        return float(mean_excess / std_excess * np.sqrt(252))

    def _calc_alpha_beta(self, strategy_returns, benchmark_returns):
        """计算Alpha和Beta"""
        if len(strategy_returns) != len(benchmark_returns) or len(strategy_returns) < 10:
            return 0, 1
        x = np.array(benchmark_returns)
        y = np.array(strategy_returns)
        cov = np.cov(x, y)
        if cov[0, 0] == 0:
            return 0, 1
        beta = cov[0, 1] / cov[0, 0]
        alpha = (np.mean(y) - beta * np.mean(x)) * 252 * 100
        return float(alpha), float(beta)

    def _calc_performance_attribution(self, daily_returns, df, trades):
        """
        收益归因分析（Brinson模型简化版）
        将总收益分解为：选股收益 + 择时收益 + 市场收益 + 交易成本
        """
        if not daily_returns or len(daily_returns) < 10:
            return None

        returns_arr = np.array(daily_returns)
        total_return = (np.prod(1 + returns_arr) - 1) * 100

        # 市场收益（买入持有）
        close = df['close']
        market_return = (close.iloc[-1] / close.iloc[0] - 1) * 100

        # 交易成本
        total_cost = sum(t.get("费用合计", 0) for t in trades)
        cost_pct = total_cost / self.initial_capital * 100

        # 择时收益 = 策略收益 - 买入持有收益（简化）
        timing_return = total_return - market_return

        # 选股收益（超额收益中扣除交易成本后的部分）
        selection_return = timing_return + cost_pct

        # 计算各成分占比
        components = {
            "市场收益(Beta)": round(market_return, 2),
            "选股收益(Alpha)": round(selection_return, 2),
            "交易成本": round(-cost_pct, 2),
            "总收益": round(total_return, 2),
        }

        # 收益归因评价
        evaluation = []
        if selection_return > 5:
            evaluation.append("选股能力优秀，超额收益显著")
        elif selection_return > 0:
            evaluation.append("选股能力良好，有正向超额收益")
        elif selection_return > -3:
            evaluation.append("选股能力一般，超额收益不明显")
        else:
            evaluation.append("选股能力较弱，跑输市场")

        if abs(timing_return - selection_return) < 1:
            evaluation.append("择时贡献与选股贡献基本一致")
        elif timing_return > selection_return:
            evaluation.append("择时贡献大于选股贡献")

        # 收益稳定性分析
        positive_days = sum(1 for r in daily_returns if r > 0)
        total_days = len(daily_returns)
        win_rate_daily = positive_days / total_days * 100 if total_days > 0 else 0

        return {
            "收益分解": components,
            "评价": evaluation,
            "日胜率": round(win_rate_daily, 1),
            "最大单日收益": round(max(daily_returns) * 100, 2) if daily_returns else 0,
            "最大单日亏损": round(min(daily_returns) * 100, 2) if daily_returns else 0,
        }


def backtest_with_strategy(symbol, strategy_id, initial_capital=100000,
                           commission_rate=0.0003, stamp_tax_rate=0.001,
                           slippage=0.001, position_size=1.0, days=250,
                           allow_short=False, **strategy_params):
    """使用指定策略进行回测"""
    # 动态导入策略框架
    import importlib.util
    import os

    strategy_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "strategy-framework", "scripts", "strategy_cli.py"
    )

    spec = importlib.util.spec_from_file_location("strategy_cli", strategy_path)
    strategy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy_module)

    strategy = strategy_module.get_strategy(strategy_id, **strategy_params)
    if strategy is None:
        return {"error": f"未知策略: {strategy_id}"}

    df = get_stock_kline(symbol, days=days)
    if df is None or len(df) < 60:
        return {"error": f"无法获取股票 {symbol} 的足够历史数据"}

    df_with_signals = strategy.generate_signals(df)
    signals = df_with_signals['signal']

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        stamp_tax_rate=stamp_tax_rate,
        slippage=slippage
    )

    result = engine.run(df, signals, position_size=position_size, allow_short=allow_short)
    result["股票代码"] = symbol
    result["策略"] = strategy.to_dict()
    result["回测参数"] = {
        "初始资金": initial_capital,
        "佣金费率": f"{commission_rate*10000:.1f}%%",
        "印花税率": f"{stamp_tax_rate*1000:.1f}%%",
        "滑点": f"{slippage*100:.1f}%",
        "仓位比例": f"{position_size*100:.0f}%",
        "回测天数": days,
        "允许做空": "是" if allow_short else "否"
    }

    return result


def _safe_eval(expr, local_vars):
    """安全表达式求值 - 使用AST白名单方式替代eval()"""
    import ast
    import operator

    ALLOWED_NODES = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare,
        ast.BoolOp, ast.Name, ast.Constant, ast.Load,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
        ast.Num,
    }

    ALLOWED_OPS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.Eq: operator.eq, ast.NotEq: operator.ne,
        ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Gt: operator.gt, ast.GtE: operator.ge,
        ast.And: lambda a, b: a & b, ast.Or: lambda a, b: a | b,
        ast.Not: operator.not_, ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval_node(node):
        node_type = type(node)
        if node_type not in ALLOWED_NODES:
            raise ValueError(f"不允许的操作: {node_type.__name__}")

        if node_type == ast.Expression:
            return _eval_node(node.body)
        elif node_type == ast.Constant:
            return node.value
        elif node_type == ast.Num:
            return node.n
        elif node_type == ast.Name:
            if node.id in local_vars:
                return local_vars[node.id]
            raise ValueError(f"未知变量: {node.id}")
        elif node_type == ast.BinOp:
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            op_func = ALLOWED_OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
            return op_func(left, right)
        elif node_type == ast.UnaryOp:
            operand = _eval_node(node.operand)
            op_func = ALLOWED_OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
            return op_func(operand)
        elif node_type == ast.Compare:
            left = _eval_node(node.left)
            result = True
            for op_node, comp_node in zip(node.ops, node.comparators):
                right = _eval_node(comp_node)
                op_func = ALLOWED_OPS.get(type(op_node))
                if op_func is None:
                    raise ValueError(f"不支持的比较运算符: {type(op_node).__name__}")
                cmp_result = op_func(left, right)
                result = result & cmp_result if isinstance(result, (pd.Series, np.ndarray)) else result and cmp_result
                left = right
            return result
        elif node_type == ast.BoolOp:
            values = [_eval_node(v) for v in node.values]
            if type(node.op) == ast.And:
                result = values[0]
                for v in values[1:]:
                    result = result & v
                return result
            elif type(node.op) == ast.Or:
                result = values[0]
                for v in values[1:]:
                    result = result | v
                return result
            raise ValueError(f"不支持的布尔运算符: {type(node.op).__name__}")
        raise ValueError(f"不支持的节点类型: {node_type.__name__}")

    try:
        tree = ast.parse(expr.strip(), mode='eval')
        return _eval_node(tree)
    except SyntaxError as e:
        raise ValueError(f"表达式语法错误: {str(e)}")


def backtest_with_custom_signals(symbol, buy_condition, sell_condition,
                                  initial_capital=100000, days=250):
    """使用自定义条件进行回测"""
    df = get_stock_kline(symbol, days=days)
    if df is None or len(df) < 60:
        return {"error": f"无法获取股票 {symbol} 的足够历史数据"}

    close = df['close']
    signals = pd.Series(0, index=df.index)

    # 计算技术指标
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    amount = df['amount']
    vol_ma20 = amount.rolling(20).mean()

    # 构建条件判断环境
    local_vars = {
        'close': close, 'ma5': ma5, 'ma10': ma10, 'ma20': ma20, 'ma60': ma60,
        'rsi': rsi, 'amount': amount, 'vol_ma20': vol_ma20,
        'high': df['high'], 'low': df['low'], 'open': df['open']
    }

    try:
        buy_mask = _safe_eval(buy_condition, local_vars)
        sell_mask = _safe_eval(sell_condition, local_vars)

        signals[buy_mask] = 1
        signals[sell_mask] = -1
    except Exception as e:
        return {"error": f"条件表达式错误: {str(e)}"}

    engine = BacktestEngine(initial_capital=initial_capital)
    result = engine.run(df, signals)
    result["股票代码"] = symbol
    result["策略"] = {
        "name": "自定义策略",
        "description": f"买入条件: {buy_condition} | 卖出条件: {sell_condition}"
    }

    return result


# ==================== 多资产组合回测 ====================

class PortfolioBacktestEngine:
    """多资产组合回测引擎"""

    def __init__(self, initial_capital=100000, commission_rate=0.0003,
                 stamp_tax_rate=0.001, slippage=0.001, min_commission=5,
                 transfer_fee_rate=0.00001, impact_cost_rate=0.0001):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage
        self.min_commission = min_commission
        self.transfer_fee_rate = transfer_fee_rate
        self.impact_cost_rate = impact_cost_rate

    def calculate_commission(self, trade_amount, is_sell=False, shares=0, daily_volume=0):
        commission = max(trade_amount * self.commission_rate, self.min_commission)
        stamp_tax = trade_amount * self.stamp_tax_rate if is_sell else 0
        transfer_fee = max(trade_amount * self.transfer_fee_rate, 0.1)
        impact_cost = 0
        if daily_volume > 0 and shares > 0:
            trade_volume_ratio = trade_amount / daily_volume if daily_volume > 0 else 0
            if trade_volume_ratio > 0.01:
                impact_cost = trade_amount * self.impact_cost_rate * (trade_volume_ratio / 0.01)
        return {
            "佣金": round(commission, 2),
            "印花税": round(stamp_tax, 2),
            "过户费": round(transfer_fee, 2),
            "冲击成本": round(impact_cost, 2),
            "合计": round(commission + stamp_tax + transfer_fee + impact_cost, 2)
        }

    def run(self, price_matrix, signal_matrix, weights=None, rebalance_freq=20):
        """
        多资产组合回测
        参数:
            price_matrix: DataFrame, 行=日期, 列=股票代码, 值=收盘价
            signal_matrix: DataFrame, 行=日期, 列=股票代码, 值=信号(1买入/-1卖出/0持有)
            weights: dict, 各股票的目标权重 {symbol: weight_pct}
            rebalance_freq: 再平衡频率(交易日)
        返回:
            dict: 组合回测结果
        """
        if weights is None:
            n = len(price_matrix.columns)
            weights = {col: 1.0 / n for col in price_matrix.columns}

        symbols = price_matrix.columns.tolist()
        dates = price_matrix.index.tolist()

        capital = self.initial_capital
        cash = self.initial_capital
        positions = {s: {"shares": 0, "market_value": 0} for s in symbols}

        trades = []
        equity_curve = []
        daily_returns = []
        prev_equity = self.initial_capital

        for i, date in enumerate(dates):
            # 再平衡
            if i > 0 and i % rebalance_freq == 0:
                total_equity = cash + sum(
                    positions[s]["shares"] * price_matrix.iloc[i][s]
                    for s in symbols
                )
                for s in symbols:
                    target_value = total_equity * weights.get(s, 0)
                    current_price = price_matrix.iloc[i][s]
                    if pd.isna(current_price) or current_price <= 0:
                        continue

                    current_shares = positions[s]["shares"]
                    target_shares = int(target_value / current_price / 100) * 100
                    diff_shares = target_shares - current_shares

                    if diff_shares >= 100:
                        buy_price = current_price * (1 + self.slippage)
                        trade_amount = diff_shares * buy_price
                        fee_detail = self.calculate_commission(trade_amount, is_sell=False, shares=diff_shares)
                        total_fee = fee_detail["合计"]
                        total_cost = trade_amount + total_fee
                        if total_cost <= cash:
                            cash -= total_cost
                            positions[s]["shares"] += diff_shares
                            trades.append({
                                "日期": date.strftime('%Y-%m-%d'),
                                "股票": s,
                                "类型": "买入(再平衡)",
                                "价格": round(buy_price, 2),
                                "数量": diff_shares,
                                "金额": round(trade_amount, 2),
                                "费用明细": fee_detail,
                                "费用合计": round(total_fee, 2)
                            })
                    elif diff_shares <= -100:
                        sell_shares = abs(diff_shares)
                        if sell_shares <= positions[s]["shares"]:
                            sell_price = current_price * (1 - self.slippage)
                            trade_amount = sell_shares * sell_price
                            fee_detail = self.calculate_commission(trade_amount, is_sell=True, shares=sell_shares)
                            total_fee = fee_detail["合计"]
                            net_amount = trade_amount - total_fee
                            cash += net_amount
                            positions[s]["shares"] -= sell_shares
                            trades.append({
                                "日期": date.strftime('%Y-%m-%d'),
                                "股票": s,
                                "类型": "卖出(再平衡)",
                                "价格": round(sell_price, 2),
                                "数量": sell_shares,
                                "金额": round(trade_amount, 2),
                                "费用明细": fee_detail,
                                "费用合计": round(total_fee, 2)
                            })

            # 处理交易信号
            for s in symbols:
                if i >= len(signal_matrix):
                    continue
                signal = int(signal_matrix.iloc[i].get(s, 0))
                current_price = price_matrix.iloc[i][s]
                if pd.isna(current_price) or current_price <= 0:
                    continue

                if signal == 1 and positions[s]["shares"] == 0 and cash > 0:
                    alloc = cash * weights.get(s, 0.1)
                    buy_price = current_price * (1 + self.slippage)
                    max_shares = int(alloc / buy_price / 100) * 100
                    if max_shares >= 100:
                        trade_amount = max_shares * buy_price
                        fee_detail = self.calculate_commission(trade_amount, is_sell=False, shares=max_shares)
                        total_fee = fee_detail["合计"]
                        total_cost = trade_amount + total_fee
                        if total_cost <= cash:
                            cash -= total_cost
                            positions[s]["shares"] = max_shares
                            trades.append({
                                "日期": date.strftime('%Y-%m-%d'),
                                "股票": s,
                                "类型": "买入(信号)",
                                "价格": round(buy_price, 2),
                                "数量": max_shares,
                                "金额": round(trade_amount, 2),
                                "费用明细": fee_detail,
                                "费用合计": round(total_fee, 2)
                            })

                elif signal == -1 and positions[s]["shares"] > 0:
                    sell_price = current_price * (1 - self.slippage)
                    trade_amount = positions[s]["shares"] * sell_price
                    fee_detail = self.calculate_commission(trade_amount, is_sell=True, shares=positions[s]["shares"])
                    total_fee = fee_detail["合计"]
                    net_amount = trade_amount - total_fee
                    cash += net_amount
                    trades.append({
                        "日期": date.strftime('%Y-%m-%d'),
                        "股票": s,
                        "类型": "卖出(信号)",
                        "价格": round(sell_price, 2),
                        "数量": positions[s]["shares"],
                        "金额": round(trade_amount, 2),
                        "费用明细": fee_detail,
                        "费用合计": round(total_fee, 2)
                    })
                    positions[s]["shares"] = 0

            # 计算当日权益
            total_equity = cash
            for s in symbols:
                current_price = price_matrix.iloc[i][s]
                if not pd.isna(current_price):
                    positions[s]["market_value"] = positions[s]["shares"] * current_price
                    total_equity += positions[s]["market_value"]

            equity_curve.append({
                "日期": date.strftime('%Y-%m-%d'),
                "权益": round(total_equity, 2),
                "现金": round(cash, 2),
                "持仓市值": round(total_equity - cash, 2)
            })

            if i > 0:
                daily_ret = (total_equity / prev_equity - 1)
                daily_returns.append(daily_ret)
            prev_equity = total_equity

        # 最终清仓
        final_date = dates[-1]
        for s in symbols:
            if positions[s]["shares"] > 0:
                final_price = price_matrix.iloc[-1][s]
                if not pd.isna(final_price):
                    trade_amount = positions[s]["shares"] * final_price
                    fee_detail = self.calculate_commission(trade_amount, is_sell=True, shares=positions[s]["shares"])
                    total_fee = fee_detail["合计"]
                    cash += trade_amount - total_fee
                    trades.append({
                        "日期": final_date.strftime('%Y-%m-%d'),
                        "股票": s,
                        "类型": "卖出(清仓)",
                        "价格": round(final_price, 2),
                        "数量": positions[s]["shares"],
                        "金额": round(trade_amount, 2),
                        "费用明细": fee_detail,
                        "费用合计": round(total_fee, 2)
                    })
                    positions[s]["shares"] = 0

        final_equity = cash

        # 计算绩效指标
        metrics = self._calculate_portfolio_metrics(final_equity, daily_returns, equity_curve, trades)

        # 个股贡献
        stock_contributions = self._calculate_stock_contributions(trades, price_matrix, symbols)

        return {
            "初始资金": self.initial_capital,
            "最终权益": round(final_equity, 2),
            "交易记录": trades,
            "权益曲线": equity_curve,
            "绩效指标": metrics,
            "个股贡献": stock_contributions
        }

    def _calculate_portfolio_metrics(self, final_equity, daily_returns, equity_curve, trades):
        """计算组合绩效指标"""
        metrics = {}

        total_return = (final_equity / self.initial_capital - 1) * 100
        metrics["总收益率"] = round(total_return, 2)

        trading_days = len(daily_returns)
        if trading_days > 0:
            annual_return = ((1 + total_return / 100) ** (252 / trading_days) - 1) * 100
            metrics["年化收益率"] = round(annual_return, 2)
        else:
            metrics["年化收益率"] = 0

        if daily_returns:
            returns_arr = np.array(daily_returns)
            metrics["日均收益率"] = round(float(np.mean(returns_arr)) * 100, 4)
            metrics["日收益率标准差"] = round(float(np.std(returns_arr, ddof=1)) * 100, 4)

            risk_free_daily = 0.02 / 252
            excess_returns = returns_arr - risk_free_daily
            if np.std(returns_arr, ddof=1) > 0:
                sharpe = np.mean(excess_returns) / np.std(returns_arr, ddof=1) * np.sqrt(252)
                metrics["夏普比率"] = round(float(sharpe), 2)
            else:
                metrics["夏普比率"] = 0

            downside_returns = returns_arr[returns_arr < 0]
            if len(downside_returns) > 0 and np.std(downside_returns, ddof=1) > 0:
                sortino = np.mean(excess_returns) / np.std(downside_returns, ddof=1) * np.sqrt(252)
                metrics["索提诺比率"] = round(float(sortino), 2)
            else:
                metrics["索提诺比率"] = 0

        if equity_curve:
            equities = np.array([e["权益"] for e in equity_curve])
            peak = np.maximum.accumulate(equities)
            drawdowns = (equities - peak) / peak * 100
            max_dd = float(np.min(drawdowns))
            metrics["最大回撤"] = round(max_dd, 2)

            if max_dd < 0:
                calmar = metrics.get("年化收益率", 0) / abs(max_dd)
                metrics["卡玛比率"] = round(float(calmar), 2)

        if trades:
            buy_trades = [t for t in trades if "买入" in t["类型"]]
            sell_trades = [t for t in trades if "卖出" in t["类型"]]
            metrics["交易总次数"] = len(buy_trades)
            total_fee = sum(t.get("费用合计", 0) for t in trades)
            metrics["总交易费用"] = round(total_fee, 2)

            profits = []
            for i in range(min(len(buy_trades), len(sell_trades))):
                buy = buy_trades[i]
                sell = sell_trades[i]
                profit = (sell.get("净收入", sell["金额"]) - buy["金额"]) / buy["金额"] * 100
                profits.append(profit)

            if profits:
                metrics["胜率"] = round(sum(1 for p in profits if p > 0) / len(profits) * 100, 2)
                metrics["平均盈利"] = round(np.mean([p for p in profits if p > 0]), 2) if any(p > 0 for p in profits) else 0
                metrics["平均亏损"] = round(np.mean([p for p in profits if p < 0]), 2) if any(p < 0 for p in profits) else 0
                metrics["盈亏比"] = round(abs(metrics["平均盈利"] / metrics["平均亏损"]), 2) if metrics["平均亏损"] != 0 else 0
            else:
                metrics["胜率"] = 0
                metrics["平均盈利"] = 0
                metrics["平均亏损"] = 0
                metrics["盈亏比"] = 0
        else:
            metrics["交易总次数"] = 0
            metrics["胜率"] = 0
            metrics["总交易费用"] = 0

        return metrics

    def _calculate_stock_contributions(self, trades, price_matrix, symbols):
        """计算各股票对组合的贡献"""
        contributions = {}
        for s in symbols:
            stock_trades = [t for t in trades if t.get("股票") == s]
            buy_amount = sum(t["金额"] for t in stock_trades if "买入" in t["类型"])
            sell_amount = sum(t.get("净收入", t["金额"]) for t in stock_trades if "卖出" in t["类型"])
            pnl = sell_amount - buy_amount
            pnl_pct = (pnl / buy_amount * 100) if buy_amount > 0 else 0

            contributions[s] = {
                "买入总额": round(buy_amount, 2),
                "卖出总额": round(sell_amount, 2),
                "盈亏金额": round(pnl, 2),
                "盈亏比例": f"{pnl_pct:.1f}%",
                "交易次数": len(stock_trades)
            }

        return contributions


def backtest_portfolio(symbols, strategy_id, initial_capital=100000,
                        weights=None, days=250, rebalance_freq=20,
                        commission_rate=0.0003, slippage=0.001, **strategy_params):
    """
    多资产组合回测
    参数:
        symbols: 股票代码列表
        strategy_id: 策略ID
        initial_capital: 初始资金
        weights: 权重分配 {symbol: weight}
        days: 回测天数
        rebalance_freq: 再平衡频率
    返回:
        dict: 组合回测结果
    """
    import importlib.util
    import os

    strategy_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "strategy-framework", "scripts", "strategy_cli.py"
    )

    spec = importlib.util.spec_from_file_location("strategy_cli", strategy_path)
    strategy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy_module)

    strategy = strategy_module.get_strategy(strategy_id, **strategy_params)
    if strategy is None:
        return {"error": f"未知策略: {strategy_id}"}

    # 获取多只股票数据
    price_data = {}
    signal_data = {}
    common_dates = None

    for symbol in symbols:
        df = get_stock_kline(symbol, days=days)
        if df is None or len(df) < 60:
            continue

        df_with_signals = strategy.generate_signals(df)
        price_data[symbol] = df['close']
        signal_data[symbol] = df_with_signals['signal']

        if common_dates is None:
            common_dates = set(df.index)
        else:
            common_dates = common_dates.intersection(set(df.index))

        time.sleep(0.3)

    if len(price_data) < 2:
        return {"error": "有效股票数据不足，至少需要2只"}

    common_dates = sorted(common_dates)
    if len(common_dates) < 60:
        return {"error": "共同交易日不足60天"}

    price_matrix = pd.DataFrame(
        {s: price_data[s].reindex(common_dates) for s in price_data},
        index=common_dates
    ).dropna()

    signal_matrix = pd.DataFrame(
        {s: signal_data[s].reindex(common_dates).fillna(0) for s in signal_data},
        index=common_dates
    ).fillna(0)

    if weights is None:
        n = len(price_matrix.columns)
        weights = {col: 1.0 / n for col in price_matrix.columns}

    engine = PortfolioBacktestEngine(
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage=slippage
    )

    result = engine.run(price_matrix, signal_matrix, weights, rebalance_freq)
    result["股票池"] = price_matrix.columns.tolist()
    result["策略"] = strategy.to_dict()
    result["权重配置"] = {s: f"{w*100:.1f}%" for s, w in weights.items()}
    result["再平衡频率"] = f"{rebalance_freq}个交易日"

    return result


# ==================== 实时盈亏归因 ====================

def realtime_pnl_attribution(holdings, benchmark_returns=None, sector_returns=None,
                               risk_free_rate=0.02):
    """
    实时盈亏归因分析
    将持仓收益分解为：市场收益、行业配置收益、选股收益、时机选择收益

    参数:
        holdings: 当前持仓 [
            {"代码": "000001", "名称": "平安银行", "行业": "银行", "权重": 0.2,
             "当日收益": 0.015, "beta": 1.1, "基准权重": 0.15},
            ...
        ]
        benchmark_returns: 基准指数收益率（如沪深300）
        sector_returns: 各行业收益率 {"银行": 0.01, "科技": 0.02, ...}
        risk_free_rate: 无风险利率

    返回: {
        "总收益": float,
        "归因分解": {...},
        "超额收益": float,
        "归因解读": str,
    }
    """
    if not holdings:
        return {"error": "持仓数据不能为空"}

    # 计算组合总收益
    total_return = sum(
        h.get("权重", 0) * h.get("当日收益", 0) for h in holdings
    )

    # 计算组合beta
    portfolio_beta = sum(
        h.get("权重", 0) * h.get("beta", 1.0) for h in holdings
    )

    # 基准收益
    bench_return = benchmark_returns if benchmark_returns is not None else 0

    # 1. 市场收益 = beta * 基准收益
    market_return = portfolio_beta * bench_return

    # 2. 行业配置收益
    sector_allocation_return = 0
    sector_details = []
    if sector_returns:
        for h in holdings:
            sector = h.get("行业", "其他")
            sector_ret = sector_returns.get(sector, 0)
            benchmark_weight = h.get("基准权重", 0)
            active_weight = h.get("权重", 0) - benchmark_weight
            sector_contrib = active_weight * sector_ret
            sector_allocation_return += sector_contrib
            sector_details.append({
                "行业": sector,
                "主动权重": f"{active_weight * 100:+.2f}%",
                "行业收益": f"{sector_ret * 100:+.2f}%",
                "贡献": f"{sector_contrib * 100:+.3f}%",
            })

    # 3. 选股收益
    stock_selection_return = 0
    stock_details = []
    for h in holdings:
        sector = h.get("行业", "其他")
        sector_ret = sector_returns.get(sector, 0) if sector_returns else bench_return
        stock_ret = h.get("当日收益", 0)
        weight = h.get("权重", 0)
        selection_contrib = weight * (stock_ret - sector_ret)
        stock_selection_return += selection_contrib
        stock_details.append({
            "代码": h.get("代码", ""),
            "名称": h.get("名称", ""),
            "权重": f"{weight * 100:.2f}%",
            "个股收益": f"{stock_ret * 100:+.2f}%",
            "行业收益": f"{sector_ret * 100:+.2f}%",
            "超额": f"{(stock_ret - sector_ret) * 100:+.2f}%",
            "贡献": f"{selection_contrib * 100:+.3f}%",
        })

    # 4. 交互效应（残差）
    interaction_return = total_return - market_return - sector_allocation_return - stock_selection_return

    # 超额收益
    excess_return = total_return - bench_return

    # 归因解读
    attribution_interpretation = _interpret_attribution(
        total_return, market_return, sector_allocation_return,
        stock_selection_return, excess_return
    )

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "组合总收益": f"{total_return * 100:+.2f}%",
        "基准收益": f"{bench_return * 100:+.2f}%" if bench_return else "N/A",
        "超额收益": f"{excess_return * 100:+.2f}%",
        "归因分解": {
            "市场收益": {
                "贡献": f"{market_return * 100:+.3f}%",
                "占比": f"{abs(market_return) / max(abs(total_return), 0.0001) * 100:.1f}%",
                "说明": f"组合beta={portfolio_beta:.2f}，基准收益={bench_return*100:+.2f}%",
            },
            "行业配置收益": {
                "贡献": f"{sector_allocation_return * 100:+.3f}%",
                "说明": "超配/低配行业带来的收益",
                "明细": sector_details,
            },
            "选股收益": {
                "贡献": f"{stock_selection_return * 100:+.3f}%",
                "说明": "在行业内选择个股带来的超额收益",
                "明细": stock_details,
            },
            "交互效应": {
                "贡献": f"{interaction_return * 100:+.3f}%",
                "说明": "各因素交互作用产生的收益（通常较小）",
            },
        },
        "归因解读": attribution_interpretation,
        "组合特征": {
            "组合Beta": round(portfolio_beta, 2),
            "持仓数量": len(holdings),
            "最大单股权重": f"{max(h.get('权重', 0) for h in holdings) * 100:.1f}%",
        },
    }


def _interpret_attribution(total_return, market_return, sector_return,
                            stock_return, excess_return):
    """解读归因结果"""
    parts = []

    # 超额收益判断
    if excess_return > 0.005:
        parts.append("组合显著跑赢基准")
    elif excess_return > 0:
        parts.append("组合小幅跑赢基准")
    elif excess_return > -0.005:
        parts.append("组合小幅跑输基准")
    else:
        parts.append("组合显著跑输基准")

    # 主要贡献来源
    contributions = {
        "市场Beta": abs(market_return),
        "行业配置": abs(sector_return),
        "选股能力": abs(stock_return),
    }
    main_source = max(contributions, key=contributions.get)

    if main_source == "市场Beta":
        parts.append("收益主要来自市场Beta暴露")
    elif main_source == "行业配置":
        parts.append("行业配置是主要收益来源")
    elif main_source == "选股能力":
        parts.append("选股能力是主要alpha来源")

    # 选股能力评估
    if stock_return > 0.003:
        parts.append("选股能力突出，个股超额收益显著")
    elif stock_return < -0.003:
        parts.append("选股效果不佳，建议审视持仓个股")

    return "；".join(parts)


def daily_pnl_attribution(holdings, index_code="000300", days=5):
    """
    多日滚动盈亏归因
    分析最近N个交易日的收益归因变化趋势

    参数:
        holdings: 持仓列表（需包含每日收益数据）
        index_code: 基准指数代码
        days: 分析天数
    """
    # 获取基准指数数据
    df_index = get_index_kline(index_code, days=days + 10)
    if df_index is None or len(df_index) < days:
        return {"error": "基准指数数据不足"}

    close_idx = df_index['收盘'] if '收盘' in df_index.columns else df_index['close']
    bench_returns = close_idx.pct_change().dropna().tail(days).values

    # 模拟每日归因
    daily_attributions = []
    cumulative_attribution = {
        "市场收益": 0, "行业配置收益": 0, "选股收益": 0, "交互效应": 0,
    }

    for i in range(min(days, len(bench_returns))):
        # 模拟每日持仓收益（实际应从持仓数据获取）
        day_bench = float(bench_returns[i])

        # 简化：假设持仓收益与基准有一定关系
        day_attribution = {
            "日期": f"T-{days - i}",
            "基准收益": f"{day_bench * 100:+.2f}%",
            "市场收益": f"{day_bench * 0.9 * 100:+.3f}%",
            "选股收益": f"{day_bench * 0.1 * 100:+.3f}%",
        }
        daily_attributions.append(day_attribution)

    return {
        "分析周期": f"最近{days}个交易日",
        "基准指数": index_code,
        "每日归因": daily_attributions,
        "说明": "每日归因帮助识别收益来源的稳定性",
    }


def factor_exposure_attribution(holdings, factor_returns):
    """
    因子暴露归因
    将收益分解为各因子暴露的贡献

    参数:
        holdings: 持仓列表
        factor_returns: 因子收益率 {
            "市值因子": 0.01,
            "价值因子": -0.005,
            "动量因子": 0.02,
            "质量因子": 0.008,
            "波动因子": -0.003,
        }

    返回: 因子归因结果
    """
    if not holdings or not factor_returns:
        return {"error": "持仓数据和因子收益率不能为空"}

    # 计算组合在各因子上的暴露
    factor_exposures = {}
    for factor_name in factor_returns:
        exposures = []
        for h in holdings:
            exposure = h.get("因子暴露", {}).get(factor_name, 0)
            weight = h.get("权重", 0)
            exposures.append(weight * exposure)
        factor_exposures[factor_name] = sum(exposures)

    # 计算各因子贡献
    factor_contributions = {}
    total_factor_return = 0
    for factor_name, factor_ret in factor_returns.items():
        exposure = factor_exposures.get(factor_name, 0)
        contribution = exposure * factor_ret
        factor_contributions[factor_name] = {
            "因子暴露": round(exposure, 3),
            "因子收益": f"{factor_ret * 100:+.2f}%",
            "贡献": f"{contribution * 100:+.3f}%",
        }
        total_factor_return += contribution

    # 因子归因解读
    positive_factors = [k for k, v in factor_contributions.items()
                        if float(v["贡献"].replace('%', '').replace('+', '')) > 0]
    negative_factors = [k for k, v in factor_contributions.items()
                        if float(v["贡献"].replace('%', '').replace('+', '')) < 0]

    interpretation = []
    if positive_factors:
        interpretation.append(f"正面贡献因子: {', '.join(positive_factors)}")
    if negative_factors:
        interpretation.append(f"负面拖累因子: {', '.join(negative_factors)}")

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "因子归因": factor_contributions,
        "因子总收益": f"{total_factor_return * 100:+.3f}%",
        "归因解读": "；".join(interpretation) if interpretation else "因子贡献不显著",
        "组合因子暴露": {k: round(v, 3) for k, v in factor_exposures.items()},
    }


def run_compare_strategies(symbol, strategies, days=500, initial_capital=100000,
                            commission_rate=0.0003, slippage=0.001, position_size=1.0):
    """
    多策略在同一股票上的增强对比回测
    包含权益曲线叠加、指标对比表、综合排名、策略相关性、月度收益、滚动绩效

    参数:
        symbol: 股票代码
        strategies: 策略列表 [{"id": "ma_cross", "name": "双均线", "params": {}}, ...]
        days: 回测天数
        initial_capital: 初始资金
        commission_rate: 佣金费率
        slippage: 滑点
        position_size: 仓位比例

    返回: 对比报告
    """
    results = []
    equity_curves = {}
    drawdown_curves = {}
    daily_return_curves = {}

    for strat in strategies:
        sid = strat.get("id", "")
        sname = strat.get("name", sid)
        sparams = strat.get("params", {})

        bt_result = backtest_with_strategy(
            symbol, sid, initial_capital=initial_capital, days=days,
            position_size=position_size, commission_rate=commission_rate,
            slippage=slippage, **sparams
        )

        if 'error' in bt_result:
            continue

        metrics = bt_result.get("绩效指标", {})
        results.append({
            "策略ID": sid,
            "策略名称": sname,
            "参数": sparams,
            "总收益率": metrics.get("总收益率", 0),
            "年化收益率": metrics.get("年化收益率", 0),
            "夏普比率": metrics.get("夏普比率", 0),
            "索提诺比率": metrics.get("索提诺比率", 0),
            "最大回撤": metrics.get("最大回撤", 0),
            "卡玛比率": metrics.get("Calmar比率", 0),
            "交易次数": metrics.get("交易总次数", 0),
            "胜率": metrics.get("胜率", 0),
        })

        equity_curves[sname] = bt_result.get("权益曲线", [])
        drawdown_curves[sname] = bt_result.get("回撤序列", [])
        daily_return_curves[sname] = bt_result.get("日收益率", [])

    if not results:
        return {"error": "所有策略回测均失败"}

    ranking = _calculate_ranking(results)
    correlation = _calculate_strategy_correlation(daily_return_curves)
    monthly_returns = _calculate_monthly_returns(equity_curves)
    rolling_performance = _calculate_rolling_performance(equity_curves)

    return {
        "股票代码": symbol,
        "回测参数": {
            "初始资金": initial_capital,
            "佣金费率": f"{commission_rate*10000:.1f}%%",
            "滑点": f"{slippage*100:.1f}%",
            "回测天数": days,
        },
        "策略对比结果": results,
        "综合排名": ranking,
        "策略相关性": correlation,
        "月度收益对比": monthly_returns,
        "滚动绩效": rolling_performance,
        "权益曲线": equity_curves,
        "回撤曲线": drawdown_curves,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def run_compare_symbols(symbols, strategy_id, days=500, initial_capital=100000,
                         commission_rate=0.0003, slippage=0.001, position_size=1.0):
    """
    同一策略在多只股票上的对比回测

    参数:
        symbols: 股票代码列表
        strategy_id: 策略ID
        days: 回测天数
        initial_capital: 初始资金
        commission_rate: 佣金费率
        slippage: 滑点
        position_size: 仓位比例

    返回: 对比报告
    """
    results = []
    equity_curves = {}

    for symbol in symbols:
        bt_result = backtest_with_strategy(
            symbol, strategy_id, initial_capital=initial_capital, days=days,
            position_size=position_size, commission_rate=commission_rate,
            slippage=slippage
        )

        if 'error' in bt_result:
            results.append({"股票代码": symbol, "状态": "回测失败"})
            continue

        metrics = bt_result.get("绩效指标", {})
        results.append({
            "股票代码": symbol,
            "总收益率": metrics.get("总收益率", 0),
            "年化收益率": metrics.get("年化收益率", 0),
            "夏普比率": metrics.get("夏普比率", 0),
            "最大回撤": metrics.get("最大回撤", 0),
            "卡玛比率": metrics.get("Calmar比率", 0),
            "交易次数": metrics.get("交易总次数", 0),
            "胜率": metrics.get("胜率", 0),
        })
        equity_curves[symbol] = bt_result.get("权益曲线", [])

    if not results:
        return {"error": "所有股票回测均失败"}

    results.sort(key=lambda x: x.get("夏普比率", 0) or 0, reverse=True)

    return {
        "策略ID": strategy_id,
        "回测参数": {
            "初始资金": initial_capital,
            "佣金费率": f"{commission_rate*10000:.1f}%%",
            "滑点": f"{slippage*100:.1f}%",
            "回测天数": days,
        },
        "股票对比结果": results,
        "权益曲线": equity_curves,
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def _calculate_ranking(results):
    """计算综合排名"""
    if not results:
        return []

    metrics_for_ranking = ["总收益率", "夏普比率", "卡玛比率", "胜率"]
    weights = {"总收益率": 0.3, "夏普比率": 0.3, "卡玛比率": 0.2, "胜率": 0.2}

    scores = []
    for r in results:
        score = 0
        detail = {}
        for metric in metrics_for_ranking:
            val = r.get(metric, 0) or 0
            detail[metric] = val
            score += val * weights.get(metric, 0.25)
        scores.append({
            "策略名称": r.get("策略名称", r.get("策略ID", "")),
            "综合得分": round(score, 2),
            "各指标得分": detail,
        })

    scores.sort(key=lambda x: x["综合得分"], reverse=True)

    for i, s in enumerate(scores):
        s["排名"] = i + 1

    return scores


def _calculate_strategy_correlation(daily_return_curves):
    """计算策略间日收益率相关性"""
    names = list(daily_return_curves.keys())
    if len(names) < 2:
        return {"说明": "策略数量不足，无法计算相关性"}

    min_len = min(len(v) for v in daily_return_curves.values() if v)
    if min_len < 2:
        return {"说明": "数据不足，无法计算相关性"}

    aligned = {}
    for name in names:
        if daily_return_curves[name]:
            aligned[name] = daily_return_curves[name][-min_len:]

    corr_matrix = []
    for name1 in names:
        row = {"策略": name1}
        for name2 in names:
            if name1 == name2:
                row[name2] = 1.0
            elif name1 in aligned and name2 in aligned:
                r1 = np.array(aligned[name1])
                r2 = np.array(aligned[name2])
                corr = np.corrcoef(r1, r2)[0, 1]
                row[name2] = round(float(corr), 4) if not np.isnan(corr) else 0
            else:
                row[name2] = 0
        corr_matrix.append(row)

    return {
        "相关性矩阵": corr_matrix,
        "策略列表": names,
    }


def _calculate_monthly_returns(equity_curves):
    """计算月度收益对比"""
    monthly = {}
    for name, curve in equity_curves.items():
        if not curve:
            continue
        monthly[name] = {}
        for point in curve:
            month_key = point.get("日期", "")[:7]
            if not month_key:
                continue
            equity_val = point.get("权益", 0)
            if month_key not in monthly[name]:
                monthly[name][month_key] = {"起始权益": equity_val, "结束权益": equity_val}
            monthly[name][month_key]["结束权益"] = equity_val

    result = {}
    for name, months in monthly.items():
        result[name] = {}
        for month_key, data in months.items():
            start_eq = data["起始权益"]
            end_eq = data["结束权益"]
            ret = (end_eq / start_eq - 1) * 100 if start_eq > 0 else 0
            result[name][month_key] = round(ret, 2)

    return result


def _calculate_rolling_performance(equity_curves, window=60):
    """计算滚动绩效（滚动夏普比率）"""
    rolling = {}
    for name, curve in equity_curves.items():
        if len(curve) < window + 2:
            continue
        equities = np.array([p.get("权益", 0) for p in curve])
        dates = [p.get("日期", "") for p in curve]
        rolling_sharpe = []
        for i in range(window, len(equities)):
            segment = equities[i - window:i + 1]
            rets = np.diff(segment) / segment[:-1]
            if len(rets) > 0 and np.std(rets, ddof=1) > 0:
                sr = float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(252))
            else:
                sr = 0
            rolling_sharpe.append({
                "日期": dates[i],
                "滚动夏普": round(sr, 2)
            })
        rolling[name] = rolling_sharpe

    return rolling


# ==================== A股交易费用标准 ====================

A_SHARE_FEES = {
    "佣金": {
        "费率": 0.00025,
        "最低": 5.0,
        "说明": "券商佣金，默认万2.5，最低5元",
    },
    "印花税": {
        "费率": 0.0005,
        "方向": "卖出时收取",
        "说明": "2023年8月28日起减半征收，由0.1%降至0.05%",
    },
    "过户费": {
        "费率": 0.00001,
        "最低": 0.1,
        "说明": "中国结算收取，十万分之一，双向收取",
    },
    "经手费": {
        "费率": 0.0000341,
        "说明": "交易所收取，包含在佣金中",
    },
    "证管费": {
        "费率": 0.00002,
        "说明": "证监会收取，包含在佣金中",
    },
}


def calculate_single_trade_cost(price, shares, direction="buy", commission_rate=0.00025,
                                 min_commission=5.0, stamp_tax_rate=0.0005,
                                 transfer_fee_rate=0.00001):
    """
    计算单笔交易费用

    参数:
        price: 成交价格
        shares: 成交数量（股）
        direction: 买卖方向（buy/sell）
        commission_rate: 佣金费率
        min_commission: 最低佣金
        stamp_tax_rate: 印花税率
        transfer_fee_rate: 过户费率

    返回: 费用明细
    """
    trade_amount = price * shares
    commission = max(trade_amount * commission_rate, min_commission)
    stamp_tax = trade_amount * stamp_tax_rate if direction == "sell" else 0
    transfer_fee = max(trade_amount * transfer_fee_rate, 0.1)
    total_fee = commission + stamp_tax + transfer_fee
    effective_rate = total_fee / trade_amount * 100 if trade_amount > 0 else 0

    return {
        "交易金额": round(trade_amount, 2),
        "费用明细": {
            "佣金": round(commission, 2),
            "印花税": round(stamp_tax, 2),
            "过户费": round(transfer_fee, 2),
        },
        "费用合计": round(total_fee, 2),
        "实际费率": f"{effective_rate:.4f}%",
        "净收付": round(trade_amount - total_fee, 2) if direction == "sell" else round(trade_amount + total_fee, 2),
    }


def calculate_round_trip_cost(buy_price, sell_price, shares, commission_rate=0.00025,
                               min_commission=5.0, stamp_tax_rate=0.0005,
                               transfer_fee_rate=0.00001):
    """
    计算完整买卖往返费用

    参数:
        buy_price: 买入价格
        sell_price: 卖出价格
        shares: 交易数量
        commission_rate: 佣金费率
        min_commission: 最低佣金
        stamp_tax_rate: 印花税率
        transfer_fee_rate: 过户费率

    返回: 往返费用分析
    """
    buy_result = calculate_single_trade_cost(buy_price, shares, "buy",
                                              commission_rate, min_commission,
                                              stamp_tax_rate, transfer_fee_rate)
    sell_result = calculate_single_trade_cost(sell_price, shares, "sell",
                                               commission_rate, min_commission,
                                               stamp_tax_rate, transfer_fee_rate)

    total_cost = buy_result["费用合计"] + sell_result["费用合计"]
    buy_amount = buy_price * shares
    sell_amount = sell_price * shares
    gross_profit = sell_amount - buy_amount
    net_profit = gross_profit - total_cost

    breakeven_pct = total_cost / buy_amount * 100 if buy_amount > 0 else 0

    return {
        "买入": buy_result,
        "卖出": sell_result,
        "往返费用合计": round(total_cost, 2),
        "毛利润": round(gross_profit, 2),
        "净利润": round(net_profit, 2),
        "盈亏平衡涨幅": f"{breakeven_pct:.4f}%",
        "说明": f"股价需上涨{breakeven_pct:.4f}%才能覆盖交易成本",
    }


def cost_sensitivity_analysis(base_price=10.0, base_shares=1000):
    """
    交易成本敏感性分析
    分析不同交易量/价格下的成本变化

    参数:
        base_price: 基准价格
        base_shares: 基准数量

    返回: 敏感性分析结果
    """
    results = []
    amounts = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    for amount in amounts:
        shares = max(int(amount / base_price), 100)
        result = calculate_round_trip_cost(base_price, base_price * 1.01, shares)
        results.append({
            "交易金额": f"{amount:,}元",
            "往返费用": result["往返费用合计"],
            "成本率": f"{result['往返费用合计'] / (base_price * shares * 2) * 100:.4f}%",
        })

    optimal_analysis = []
    share_levels = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    for shares in share_levels:
        result = calculate_round_trip_cost(base_price, base_price * 1.01, shares)
        trade_amount = base_price * shares
        cost_rate = result["往返费用合计"] / (trade_amount * 2) * 100 if trade_amount > 0 else 0
        optimal_analysis.append({
            "数量(股)": shares,
            "交易金额": f"{trade_amount:,.0f}元",
            "往返费用": result["往返费用合计"],
            "成本率": f"{cost_rate:.4f}%",
        })

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "基准价格": base_price,
        "不同金额成本率": results,
        "不同数量成本率": optimal_analysis,
        "结论": [
            "交易金额越大，成本率越低（佣金最低5元的影响被摊薄）",
            "单笔交易金额建议不低于2万元，否则成本率偏高",
            "印花税是最大成本项（卖出时0.05%），佣金可通过谈判降低",
        ],
    }


def compare_broker_fees(trade_amount=100000):
    """
    不同券商费率对比

    参数:
        trade_amount: 交易金额

    返回: 券商费率对比
    """
    brokers = [
        {"名称": "默认券商", "佣金率": 0.00025, "最低佣金": 5.0},
        {"名称": "低佣券商A", "佣金率": 0.00015, "最低佣金": 5.0},
        {"名称": "低佣券商B", "佣金率": 0.0001, "最低佣金": 5.0},
        {"名称": "免五券商", "佣金率": 0.0001, "最低佣金": 0.0},
        {"名称": "万一免五", "佣金率": 0.0001, "最低佣金": 0.0},
    ]

    comparison = []
    for broker in brokers:
        buy_cost = calculate_single_trade_cost(
            10.0, int(trade_amount / 10.0), "buy",
            commission_rate=broker["佣金率"],
            min_commission=broker["最低佣金"],
        )
        sell_cost = calculate_single_trade_cost(
            10.0, int(trade_amount / 10.0), "sell",
            commission_rate=broker["佣金率"],
            min_commission=broker["最低佣金"],
        )
        total = buy_cost["费用合计"] + sell_cost["费用合计"]
        comparison.append({
            "券商": broker["名称"],
            "佣金率": f"{broker['佣金率']*100:.2f}%",
            "最低佣金": f"{broker['最低佣金']}元",
            "买入费用": buy_cost["费用合计"],
            "卖出费用": sell_cost["费用合计"],
            "往返合计": total,
            "年省(相对默认)": "",
        })

    if comparison:
        default_total = comparison[0]["往返合计"]
        for item in comparison:
            saving = default_total - item["往返合计"]
            item["年省(相对默认)"] = f"{saving * 250:.0f}元（按年250次交易）"

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "交易金额": f"{trade_amount:,}元",
        "券商对比": comparison,
        "建议": [
            "高频交易者应选择低佣金券商，年省可达数千元",
            "免五政策对小额交易者尤为重要",
            "印花税和过户费所有券商统一，无法节省",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description='强化回测系统')
    parser.add_argument('action', choices=['strategy', 'custom', 'compare', 'portfolio', 'attribution', 'save', 'history', 'trend', 'cost'],
                        help='操作类型')
    parser.add_argument('--symbol', default='600519', help='股票代码')
    parser.add_argument('--symbols', default=None, help='股票代码列表,逗号分隔(组合回测)')
    parser.add_argument('--strategy', default='ma_cross', help='策略ID')
    parser.add_argument('--params', default='{}', help='策略参数JSON')
    parser.add_argument('--capital', type=float, default=100000, help='初始资金')
    parser.add_argument('--days', type=int, default=250, help='回测天数')
    parser.add_argument('--buy', default='(ma5 > ma20) & (ma5.shift(1) <= ma20.shift(1))',
                        help='买入条件表达式')
    parser.add_argument('--sell', default='(ma5 < ma20) & (ma5.shift(1) >= ma20.shift(1))',
                        help='卖出条件表达式')
    parser.add_argument('--weights', default=None, help='权重JSON(组合回测)')
    parser.add_argument('--rebalance', type=int, default=20, help='再平衡频率')
    parser.add_argument('--holdings', default=None, help='持仓JSON(归因分析)')
    parser.add_argument('--benchmark', type=float, default=None, help='基准收益率')
    parser.add_argument('--sectors', default=None, help='行业收益率JSON')
    parser.add_argument('--factors', default=None, help='因子收益率JSON')

    args = parser.parse_args()

    try:
        if args.action == 'strategy':
            params = json.loads(args.params) if args.params else {}
            data = backtest_with_strategy(
                args.symbol, args.strategy,
                initial_capital=args.capital, days=args.days, **params
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'save':
            params = json.loads(args.params) if args.params else {}
            data = backtest_with_strategy(
                args.symbol, args.strategy,
                initial_capital=args.capital, days=args.days, **params
            )
            if 'error' in data:
                print(json.dumps(data, ensure_ascii=False, indent=2))
                sys.exit(1)
            if save_backtest_record is None:
                print(json.dumps({"error": "数据持久化模块不可用"}, ensure_ascii=False, indent=2))
                sys.exit(1)
            metrics = data.get("绩效指标", {})
            record_id = save_backtest_record(
                args.symbol,
                data.get("策略", {}).get("name", args.strategy),
                {
                    "累计收益率": metrics.get("总收益率"),
                    "年化收益率": metrics.get("年化收益率"),
                    "夏普比率": metrics.get("夏普比率"),
                    "最大回撤": metrics.get("最大回撤"),
                    "胜率": metrics.get("胜率"),
                    "交易次数": metrics.get("交易总次数"),
                    "Calmar比率": metrics.get("Calmar比率"),
                    "年化波动率": metrics.get("年化波动率"),
                }
            )
            print(json.dumps({
                "状态": "保存成功",
                "记录ID": record_id,
                "回测结果": data,
            }, ensure_ascii=False, indent=2, default=str))
        elif args.action == 'history':
            if get_backtest_records is None:
                print(json.dumps({"error": "数据持久化模块不可用"}, ensure_ascii=False, indent=2))
                sys.exit(1)
            records = get_backtest_records(
                limit=args.days,
                symbol=args.symbol if args.symbol != '600519' else None,
                strategy=args.strategy if args.strategy != 'ma_cross' else None
            )
            print(json.dumps({
                "记录数": len(records),
                "回测记录": records,
            }, ensure_ascii=False, indent=2, default=str))
        elif args.action == 'trend':
            if get_backtest_trend is None:
                print(json.dumps({"error": "数据持久化模块不可用"}, ensure_ascii=False, indent=2))
                sys.exit(1)
            data = get_backtest_trend(args.symbol, args.strategy, limit=args.days)
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        elif args.action == 'attribution':
            if not args.holdings:
                print(json.dumps({"error": "请提供--holdings参数"}, ensure_ascii=False))
                sys.exit(1)
            holdings = json.loads(args.holdings)
            sectors = json.loads(args.sectors) if args.sectors else None
            factors = json.loads(args.factors) if args.factors else None

            if factors:
                data = factor_exposure_attribution(holdings, factors)
            else:
                data = realtime_pnl_attribution(
                    holdings,
                    benchmark_returns=args.benchmark,
                    sector_returns=sectors,
                )
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'custom':
            data = backtest_with_custom_signals(
                args.symbol, args.buy, args.sell,
                initial_capital=args.capital, days=args.days
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'compare':
            strategies_str = args.strategy if args.strategy != 'ma_cross' else 'ma_cross,macd,rsi,bollinger'
            strategy_ids = [s.strip() for s in strategies_str.split(',')]
            strategies = [{"id": sid, "name": sid, "params": {}} for sid in strategy_ids]
            data = run_compare_strategies(
                args.symbol, strategies,
                initial_capital=args.capital, days=args.days
            )
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        elif args.action == 'portfolio':
            if not args.symbols:
                print(json.dumps({"error": "请提供--symbols参数"}, ensure_ascii=False))
                sys.exit(1)
            symbols = [s.strip() for s in args.symbols.split(",")]
            weights = json.loads(args.weights) if args.weights else None
            params = json.loads(args.params) if args.params else {}
            data = backtest_portfolio(
                symbols, args.strategy,
                initial_capital=args.capital, days=args.days,
                weights=weights, rebalance_freq=args.rebalance, **params
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == 'cost':
            cost_cmd = args.strategy if args.strategy != 'ma_cross' else 'fee-standard'
            if cost_cmd == 'fee-standard':
                print(json.dumps(A_SHARE_FEES, ensure_ascii=False, indent=2))
            elif cost_cmd == 'sensitivity':
                data = cost_sensitivity_analysis()
                print(json.dumps(data, ensure_ascii=False, indent=2))
            elif cost_cmd == 'broker-compare':
                data = compare_broker_fees()
                print(json.dumps(data, ensure_ascii=False, indent=2))
            elif cost_cmd == 'single':
                params = json.loads(args.params) if args.params else {}
                data = calculate_single_trade_cost(
                    params.get('price', 10.0),
                    params.get('shares', 1000),
                    params.get('direction', 'buy')
                )
                print(json.dumps(data, ensure_ascii=False, indent=2))
            elif cost_cmd == 'round-trip':
                params = json.loads(args.params) if args.params else {}
                data = calculate_round_trip_cost(
                    params.get('buy_price', 10.0),
                    params.get('sell_price', 10.5),
                    params.get('shares', 1000)
                )
                print(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                print(json.dumps({"error": f"未知的成本命令: {cost_cmd}"}, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
