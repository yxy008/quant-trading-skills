# 风控系统 (Risk Control)

## 功能概述
量化交易的生命线，提供完整的三层风控体系。

## 核心能力

### 事前风控 (Pre-Trade)
- 单票仓位上限检查
- 总仓位上限检查
- 现金充足检查
- 单笔订单金额上限
- 行业集中度检查
- 黑名单检查
- 杠杆率检查

### 事中风控 (In-Trade)
- 实时止损/止盈监控
- 硬止损/软止损分级
- 异常波动告警
- 流动性监控
- 组合回撤监控

### 事后风控 (Post-Trade)
- VaR/CVaR计算
- 波动率分析
- 最大回撤分析
- 偏度/峰度分析
- 风险评级
- 压力测试
- VaR分解

## 使用方式
```
python risk_control_cli.py pre-check --symbol 600519 --direction buy --quantity 100 --price 1800 --total-asset 100000 --cash 50000
python risk_control_cli.py post-trade --symbols 600519,000858,300750 --days 250
python risk_control_cli.py stress-test --symbols 600519,000858,300750
python risk_control_cli.py var-breakdown --symbols 600519,000858,300750
```
