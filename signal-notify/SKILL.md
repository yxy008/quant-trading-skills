---
name: signal-notify
description: 信号通知工具，支持邮件、微信、钉钉等多渠道推送交易信号和预警信息
---

# 信号通知

## 功能

- 邮件通知：通过邮件发送交易信号和预警
- 微信通知：通过微信推送消息
- 钉钉通知：通过钉钉机器人推送消息
- 信号管理：管理各类交易信号的订阅和推送规则
- 通知模板：自定义通知消息模板

## 使用方式

```bash
# 发送通知
python scripts/notify_cli.py send --channel email --message "买入信号: 600519"

# 配置通知渠道
python scripts/notify_cli.py config --channel wechat
```