#!/usr/bin/env python3
"""
策略信号推送通知模块 - 支持邮件和钉钉机器人推送
"""
import argparse
import json
import sys
import os
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional, Dict, List

_agent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

try:
    import requests
except ImportError:
    requests = None

from db_utils import get_db_connection


class NotificationConfig:
    """通知配置管理"""

    def __init__(self):
        self.config = self._load_config()

    def _load_config(self):
        """从数据库加载配置"""
        config = {
            "email": {"enabled": False, "smtp_host": "", "smtp_port": 465,
                       "sender": "", "password": "", "receivers": []},
            "dingtalk": {"enabled": False, "webhook_url": "", "secret": ""},
            "rules": {"signal_change": True, "daily_summary": True,
                      "risk_alert": True, "trade_notify": True}
        }
        try:
            conn = get_db_connection()
            if conn is None:
                return config
            cursor = conn.cursor()
            cursor.execute("SELECT config_key, config_value FROM notify_config")
            for row in cursor.fetchall():
                key = row[0]
                value = row[1]
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    pass
                keys = key.split(".")
                target = config
                for k in keys[:-1]:
                    if k not in target:
                        target[k] = {}
                    target = target[k]
                target[keys[-1]] = value
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"加载通知配置失败: {e}")
        return config

    def save_config(self, section, key, value):
        """保存配置到数据库"""
        try:
            conn = get_db_connection()
            if conn is None:
                return False
            cursor = conn.cursor()
            config_key = f"{section}.{key}"
            config_value = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
            cursor.execute(
                """INSERT INTO notify_config (config_key, config_value)
                   VALUES (%s, %s)
                   ON DUPLICATE KEY UPDATE config_value = VALUES(config_value)""",
                (config_key, config_value)
            )
            conn.commit()
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            print(f"保存通知配置失败: {e}")
            return False

    def get_email_config(self):
        return self.config.get("email", {})

    def get_dingtalk_config(self):
        return self.config.get("dingtalk", {})

    def get_rules(self):
        return self.config.get("rules", {})

    def is_rule_enabled(self, rule_name):
        return self.config.get("rules", {}).get(rule_name, False)


class EmailNotifier:
    """邮件通知"""

    def __init__(self, config: NotificationConfig):
        self.config = config
        self.email_config = config.get_email_config()

    def send(self, subject, content, content_type="html"):
        """发送邮件"""
        if not self.email_config.get("enabled"):
            return {"success": False, "error": "邮件通知未启用"}

        smtp_host = self.email_config.get("smtp_host", "")
        smtp_port = self.email_config.get("smtp_port", 465)
        sender = self.email_config.get("sender", "")
        password = self.email_config.get("password", "")
        receivers = self.email_config.get("receivers", [])

        if not all([smtp_host, sender, password, receivers]):
            return {"success": False, "error": "邮件配置不完整"}

        try:
            msg = MIMEMultipart()
            msg["From"] = sender
            msg["To"] = ", ".join(receivers)
            msg["Subject"] = subject
            msg.attach(MIMEText(content, content_type, "utf-8"))

            with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
                server.login(sender, password)
                server.sendmail(sender, receivers, msg.as_string())

            return {"success": True, "message": f"邮件已发送至 {len(receivers)} 个收件人"}
        except Exception as e:
            return {"success": False, "error": f"邮件发送失败: {str(e)}"}


class DingTalkNotifier:
    """钉钉机器人通知"""

    def __init__(self, config: NotificationConfig):
        self.config = config
        self.dt_config = config.get_dingtalk_config()

    def send(self, title, content):
        """发送钉钉消息"""
        if not self.dt_config.get("enabled"):
            return {"success": False, "error": "钉钉通知未启用"}

        webhook_url = self.dt_config.get("webhook_url", "")
        if not webhook_url:
            return {"success": False, "error": "钉钉Webhook地址未配置"}

        if requests is None:
            return {"success": False, "error": "请安装requests库: pip install requests"}

        try:
            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "title": title,
                    "text": f"## {title}\n\n{content}\n\n> 发送时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                }
            }

            secret = self.dt_config.get("secret", "")
            url = webhook_url
            if secret:
                import hmac
                import hashlib
                import base64
                import urllib.parse
                timestamp = str(round(time.time() * 1000))
                sign_str = f"{timestamp}\n{secret}"
                sign = base64.b64encode(
                    hmac.new(secret.encode(), sign_str.encode(), hashlib.sha256).digest()
                ).decode()
                url = f"{webhook_url}&timestamp={timestamp}&sign={urllib.parse.quote(sign)}"

            resp = requests.post(url, json=payload, timeout=10)
            result = resp.json()

            if result.get("errcode") == 0:
                return {"success": True, "message": "钉钉消息已发送"}
            else:
                return {"success": False, "error": f"钉钉发送失败: {result.get('errmsg', '未知错误')}"}
        except Exception as e:
            return {"success": False, "error": f"钉钉发送失败: {str(e)}"}


class SignalNotifier:
    """策略信号通知服务"""

    def __init__(self):
        self.config = NotificationConfig()
        self.email = EmailNotifier(self.config)
        self.dingtalk = DingTalkNotifier(self.config)

    def notify_signal_change(self, strategy_name, symbol, signal_type, price, reason=""):
        """通知策略信号变化"""
        if not self.config.is_rule_enabled("signal_change"):
            return {"success": False, "error": "信号变化通知未启用"}

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        signal_text = "买入" if signal_type == "buy" else "卖出"
        emoji = "🔴" if signal_type == "buy" else "🟢"

        subject = f"[{emoji}交易信号] {strategy_name} - {symbol} {signal_text}信号"
        content = f"""
        <h2>策略信号通知</h2>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse;width:100%;max-width:500px;">
            <tr><td style="background:#f5f5f5;"><b>策略名称</b></td><td>{strategy_name}</td></tr>
            <tr><td style="background:#f5f5f5;"><b>股票代码</b></td><td>{symbol}</td></tr>
            <tr><td style="background:#f5f5f5;"><b>信号类型</b></td><td style="color:{'#ef5350' if signal_type == 'buy' else '#26a69a'};font-weight:bold;">{signal_text}</td></tr>
            <tr><td style="background:#f5f5f5;"><b>当前价格</b></td><td>{price:.2f}</td></tr>
            <tr><td style="background:#f5f5f5;"><b>信号原因</b></td><td>{reason or '策略自动生成'}</td></tr>
            <tr><td style="background:#f5f5f5;"><b>触发时间</b></td><td>{now}</td></tr>
        </table>
        <p style="color:#999;font-size:12px;">此为自动通知，请勿回复。如需调整通知设置，请登录系统。</p>
        """

        dingtalk_content = (
            f"- **策略**: {strategy_name}\n"
            f"- **股票**: {symbol}\n"
            f"- **信号**: <font color=\"{'#ef5350' if signal_type == 'buy' else '#26a69a'}\">{signal_text}</font>\n"
            f"- **价格**: {price:.2f}\n"
            f"- **原因**: {reason or '策略自动生成'}\n"
            f"- **时间**: {now}"
        )

        results = {"email": None, "dingtalk": None}
        results["email"] = self.email.send(subject, content)
        results["dingtalk"] = self.dingtalk.send(subject, dingtalk_content)

        return {
            "success": results["email"].get("success") or results["dingtalk"].get("success"),
            "results": results
        }

    def notify_daily_summary(self, summary_data):
        """发送每日摘要"""
        if not self.config.is_rule_enabled("daily_summary"):
            return {"success": False, "error": "每日摘要通知未启用"}

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        date_str = datetime.now().strftime('%Y-%m-%d')

        subject = f"[每日摘要] 量化交易日报 - {date_str}"

        # 构建HTML内容
        content = f"<h2>量化交易日报 - {date_str}</h2>"

        # 账户概况
        account = summary_data.get("account", {})
        if account:
            content += "<h3>账户概况</h3>"
            content += "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse;'>"
            content += "<tr style='background:#f5f5f5;'><td>总资产</td><td>可用资金</td><td>持仓市值</td><td>总收益率</td></tr>"
            content += f"<tr><td>{account.get('总资产', 0):,.2f}</td><td>{account.get('可用资金', 0):,.2f}</td><td>{account.get('持仓市值', 0):,.2f}</td><td>{account.get('总收益率', 0):.2f}%</td></tr>"
            content += "</table>"

        # 持仓明细
        positions = summary_data.get("positions", [])
        if positions:
            content += "<h3>当前持仓</h3>"
            content += "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse;'>"
            content += "<tr style='background:#f5f5f5;'><td>股票</td><td>数量</td><td>成本</td><td>现价</td><td>盈亏</td><td>盈亏%</td></tr>"
            for p in positions:
                pnl_color = "#26a69a" if p.get("盈亏", 0) >= 0 else "#ef5350"
                content += f"<tr><td>{p.get('股票代码', '')}</td><td>{p.get('持仓数量', 0)}</td><td>{p.get('成本价', 0):.2f}</td><td>{p.get('当前价', 0):.2f}</td><td style='color:{pnl_color};'>{p.get('盈亏', 0):+,.2f}</td><td style='color:{pnl_color};'>{p.get('盈亏比例', 0):+.2f}%</td></tr>"
            content += "</table>"

        # 今日信号
        signals = summary_data.get("signals", [])
        if signals:
            content += "<h3>今日信号</h3>"
            content += "<ul>"
            for s in signals:
                content += f"<li>{s.get('策略', '')} - {s.get('股票', '')}: {s.get('信号', '')} @ {s.get('价格', 0):.2f}</li>"
            content += "</ul>"

        content += f"<p style='color:#999;font-size:12px;'>生成时间: {now}</p>"

        dingtalk_content = f"**账户概况**\n"
        if account:
            dingtalk_content += (
                f"- 总资产: {account.get('总资产', 0):,.2f}\n"
                f"- 总收益率: {account.get('总收益率', 0):.2f}%\n"
            )
        if positions:
            dingtalk_content += f"\n**持仓** ({len(positions)}只)\n"
            for p in positions[:5]:
                dingtalk_content += f"- {p.get('股票代码', '')}: {p.get('盈亏比例', 0):+.2f}%\n"
            if len(positions) > 5:
                dingtalk_content += f"- ... 等共{len(positions)}只\n"

        results = {"email": None, "dingtalk": None}
        results["email"] = self.email.send(subject, content)
        results["dingtalk"] = self.dingtalk.send(subject, dingtalk_content)

        return {
            "success": results["email"].get("success") or results["dingtalk"].get("success"),
            "results": results
        }

    def notify_risk_alert(self, alert_type, message, details=None):
        """发送风险告警"""
        if not self.config.is_rule_enabled("risk_alert"):
            return {"success": False, "error": "风险告警通知未启用"}

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        subject = f"[风险告警] {alert_type} - {now}"
        content = f"""
        <h2 style="color:#ef5350;">风险告警</h2>
        <p><b>告警类型:</b> {alert_type}</p>
        <p><b>告警内容:</b> {message}</p>
        <p><b>触发时间:</b> {now}</p>
        """

        if details:
            content += "<h3>详细信息</h3><ul>"
            for k, v in details.items():
                content += f"<li><b>{k}:</b> {v}</li>"
            content += "</ul>"

        content += "<p style='color:#999;font-size:12px;'>此为自动告警，请及时关注并处理。</p>"

        dingtalk_content = (
            f"## <font color='#ef5350'>风险告警</font>\n\n"
            f"- **类型**: {alert_type}\n"
            f"- **内容**: {message}\n"
            f"- **时间**: {now}\n"
        )
        if details:
            for k, v in details.items():
                dingtalk_content += f"- **{k}**: {v}\n"

        results = {"email": None, "dingtalk": None}
        results["email"] = self.email.send(subject, content)
        results["dingtalk"] = self.dingtalk.send(subject, dingtalk_content)

        return {
            "success": results["email"].get("success") or results["dingtalk"].get("success"),
            "results": results
        }

    def notify_trade(self, symbol, direction, quantity, price, fee, pnl=None):
        """通知交易执行"""
        if not self.config.is_rule_enabled("trade_notify"):
            return {"success": False, "error": "交易通知未启用"}

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        direction_text = "买入" if direction == "buy" else "卖出"
        amount = quantity * price

        subject = f"[交易通知] {direction_text} {symbol} {quantity}股 @ {price:.2f}"
        content = f"""
        <h2>交易执行通知</h2>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse;width:100%;max-width:500px;">
            <tr><td style="background:#f5f5f5;"><b>股票代码</b></td><td>{symbol}</td></tr>
            <tr><td style="background:#f5f5f5;"><b>交易方向</b></td><td style="color:{'#ef5350' if direction == 'buy' else '#26a69a'};font-weight:bold;">{direction_text}</td></tr>
            <tr><td style="background:#f5f5f5;"><b>成交价格</b></td><td>{price:.2f}</td></tr>
            <tr><td style="background:#f5f5f5;"><b>成交数量</b></td><td>{quantity}股</td></tr>
            <tr><td style="background:#f5f5f5;"><b>成交金额</b></td><td>{amount:,.2f}</td></tr>
            <tr><td style="background:#f5f5f5;"><b>交易费用</b></td><td>{fee:.2f}</td></tr>
        """

        if pnl is not None:
            pnl_color = "#26a69a" if pnl >= 0 else "#ef5350"
            content += f"<tr><td style=\"background:#f5f5f5;\"><b>实现盈亏</b></td><td style=\"color:{pnl_color};font-weight:bold;\">{pnl:+,.2f}</td></tr>"

        content += f"""
            <tr><td style="background:#f5f5f5;"><b>交易时间</b></td><td>{now}</td></tr>
        </table>
        <p style="color:#999;font-size:12px;">此为自动通知，请勿回复。</p>
        """

        dingtalk_content = (
            f"- **股票**: {symbol}\n"
            f"- **方向**: <font color=\"{'#ef5350' if direction == 'buy' else '#26a69a'}\">{direction_text}</font>\n"
            f"- **价格**: {price:.2f}\n"
            f"- **数量**: {quantity}股\n"
            f"- **金额**: {amount:,.2f}\n"
            f"- **费用**: {fee:.2f}\n"
        )
        if pnl is not None:
            dingtalk_content += f"- **盈亏**: {pnl:+,.2f}\n"
        dingtalk_content += f"- **时间**: {now}"

        results = {"email": None, "dingtalk": None}
        results["email"] = self.email.send(subject, content)
        results["dingtalk"] = self.dingtalk.send(subject, dingtalk_content)

        return {
            "success": results["email"].get("success") or results["dingtalk"].get("success"),
            "results": results
        }


def init_notify_tables():
    """初始化通知配置数据库表"""
    try:
        conn = get_db_connection()
        if conn is None:
            return False
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notify_config (
                id INT AUTO_INCREMENT PRIMARY KEY,
                config_key VARCHAR(128) NOT NULL UNIQUE,
                config_value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"初始化通知表失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="策略信号推送通知")
    subparsers = parser.add_subparsers(dest="action", help="操作")

    # 初始化
    subparsers.add_parser("init", help="初始化数据库表")

    # 配置
    config_parser = subparsers.add_parser("config", help="配置通知参数")
    config_parser.add_argument("--section", required=True, choices=["email", "dingtalk", "rules"],
                               help="配置分类")
    config_parser.add_argument("--key", required=True, help="配置键")
    config_parser.add_argument("--value", required=True, help="配置值(JSON格式)")

    # 查看配置
    subparsers.add_parser("show-config", help="查看当前配置")

    # 发送测试
    test_parser = subparsers.add_parser("test", help="发送测试通知")
    test_parser.add_argument("--channel", default="all", choices=["email", "dingtalk", "all"],
                             help="通知渠道")

    # 发送信号通知
    signal_parser = subparsers.add_parser("signal", help="发送信号变化通知")
    signal_parser.add_argument("--strategy", required=True, help="策略名称")
    signal_parser.add_argument("--symbol", required=True, help="股票代码")
    signal_parser.add_argument("--type", required=True, choices=["buy", "sell"], help="信号类型")
    signal_parser.add_argument("--price", type=float, required=True, help="当前价格")
    signal_parser.add_argument("--reason", default="", help="信号原因")

    # 发送每日摘要
    summary_parser = subparsers.add_parser("summary", help="发送每日摘要")
    summary_parser.add_argument("--data", default="{}", help="摘要数据(JSON)")

    # 发送风险告警
    alert_parser = subparsers.add_parser("alert", help="发送风险告警")
    alert_parser.add_argument("--type", required=True, help="告警类型")
    alert_parser.add_argument("--message", required=True, help="告警内容")
    alert_parser.add_argument("--details", default="{}", help="详细信息(JSON)")

    # 发送交易通知
    trade_parser = subparsers.add_parser("trade", help="发送交易通知")
    trade_parser.add_argument("--symbol", required=True, help="股票代码")
    trade_parser.add_argument("--direction", required=True, choices=["buy", "sell"], help="交易方向")
    trade_parser.add_argument("--quantity", type=int, required=True, help="数量")
    trade_parser.add_argument("--price", type=float, required=True, help="价格")
    trade_parser.add_argument("--fee", type=float, default=0, help="费用")
    trade_parser.add_argument("--pnl", type=float, default=None, help="实现盈亏")

    args = parser.parse_args()

    if args.action == "init":
        success = init_notify_tables()
        print(json.dumps({"success": success}, ensure_ascii=False))

    elif args.action == "config":
        config = NotificationConfig()
        try:
            value = json.loads(args.value)
        except json.JSONDecodeError:
            value = args.value
        success = config.save_config(args.section, args.key, value)
        print(json.dumps({"success": success}, ensure_ascii=False))

    elif args.action == "show-config":
        config = NotificationConfig()
        print(json.dumps(config.config, ensure_ascii=False, indent=2))

    elif args.action == "test":
        notifier = SignalNotifier()
        if args.channel in ("email", "all"):
            result = notifier.email.send(
                "测试邮件 - 量化交易系统",
                "<h2>测试邮件</h2><p>这是一封来自量化交易系统的测试邮件。</p><p>如果您收到此邮件，说明邮件配置正确。</p>"
            )
            print("邮件测试:", json.dumps(result, ensure_ascii=False))
        if args.channel in ("dingtalk", "all"):
            result = notifier.dingtalk.send(
                "测试消息 - 量化交易系统",
                "这是一条来自量化交易系统的测试消息。\n如果您收到此消息，说明钉钉配置正确。"
            )
            print("钉钉测试:", json.dumps(result, ensure_ascii=False))

    elif args.action == "signal":
        notifier = SignalNotifier()
        result = notifier.notify_signal_change(
            args.strategy, args.symbol, args.type, args.price, args.reason
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.action == "summary":
        notifier = SignalNotifier()
        try:
            data = json.loads(args.data)
        except json.JSONDecodeError:
            data = {}
        result = notifier.notify_daily_summary(data)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.action == "alert":
        notifier = SignalNotifier()
        try:
            details = json.loads(args.details)
        except json.JSONDecodeError:
            details = {}
        result = notifier.notify_risk_alert(args.type, args.message, details)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.action == "trade":
        notifier = SignalNotifier()
        result = notifier.notify_trade(
            args.symbol, args.direction, args.quantity, args.price, args.fee, args.pnl
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
