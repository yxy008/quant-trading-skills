# 量化交易 Skills 集合

A股量化交易技能（Skills）集合，涵盖数据获取、技术分析、策略回测、风险控制、AI 智能推荐等领域，共 **50 个 Skill**。

---

## 安装

```bash
# 克隆仓库
git clone https://github.com/yxy008/quant-trading-skills.git
cd quant-trading-skills

# 安装依赖
pip install -r requirements.txt
```

**环境要求**: Python >= 3.10

---

## Skills 列表

### 数据获取

| Skill | 说明 |
|-------|------|
| [akshare-stock](akshare-stock/) | A股实时行情、历史K线、财务数据获取 |
| [financial-data](financial-data/) | 财务报表数据（利润表、资产负债表、现金流量表） |
| [stock-info](stock-info/) | 股票基本信息查询（公司概况、股本结构等） |
| [news-announcements](news-announcements/) | 新闻公告、研报数据获取 |
| [market-funds](market-funds/) | 市场资金流向、北向资金、融资融券数据 |
| [dragon-tiger](dragon-tiger/) | 龙虎榜数据获取与分析 |

### 技术分析

| Skill | 说明 |
|-------|------|
| [talib-indicator](talib-indicator/) | TA-Lib 技术指标计算（MACD、RSI、布林带等） |
| [market-trend](market-trend/) | 市场趋势分析（均线系统、趋势强度） |
| [multi-timeframe](multi-timeframe/) | 多周期分析（日/周/月线联动） |
| [market-breadth](market-breadth/) | 市场宽度分析（涨跌家数、新高新低） |

### 选股与评分

| Skill | 说明 |
|-------|------|
| [stock-scoring](stock-scoring/) | 股票综合评分系统（技术面+基本面+风险多维度） |
| [stock-filter](stock-filter/) | 多条件股票筛选器 |
| [stock-pool](stock-pool/) | 股票池管理与维护 |
| [multi-factor-model](multi-factor-model/) | 多因子选股模型 |
| [factor-library](factor-library/) | 量化因子库（动量、价值、质量等因子） |
| [ml-factors](ml-factors/) | 机器学习因子挖掘 |

### 策略与回测

| Skill | 说明 |
|-------|------|
| [backtest](backtest/) | 策略回测验证（年化收益、最大回撤、夏普比率） |
| [backtrader](backtrader/) | Backtrader 回测框架集成 |
| [strategy-framework](strategy-framework/) | 策略开发框架 |
| [strategy-ide](strategy-ide/) | 策略在线编辑器 |
| [parameter-optimizer](parameter-optimizer/) | 策略参数优化（网格搜索、遗传算法） |
| [monte-carlo](monte-carlo/) | 蒙特卡洛模拟与过拟合检测 |
| [industry-rotation](industry-rotation/) | 行业轮动策略 |

### 交易执行

| Skill | 说明 |
|-------|------|
| [paper-trading](paper-trading/) | 模拟交易（虚拟资金，真实行情） |
| [live-trading](live-trading/) | 实盘交易接口 |
| [algo-trading](algo-trading/) | 算法交易（TWAP、VWAP、冰山订单） |
| [oms](oms/) | 订单管理系统 |

### 风险管理

| Skill | 说明 |
|-------|------|
| [risk-control](risk-control/) | 风险控制（事前检查、事中监控、事后分析） |
| [risk-metrics](risk-metrics/) | 风险指标计算（VaR、CVaR、贝塔等） |
| [risk-monitor](risk-monitor/) | 风控实时监控面板 |
| [position-manager](position-manager/) | 仓位管理与动态调整 |

### 组合管理

| Skill | 说明 |
|-------|------|
| [portfolio-mgmt](portfolio-mgmt/) | 投资组合管理 |
| [portfolio-optimizer](portfolio-optimizer/) | 组合优化（均值方差、风险平价、Black-Litterman） |
| [multi-asset](multi-asset/) | 多资产配置（股票、ETF、债券） |
| [performance-attribution](performance-attribution/) | 绩效归因分析（Brinson、因子归因） |

### AI 智能

| Skill | 说明 |
|-------|------|
| [ai-agent](ai-agent/) | AI 智能助手（自然语言选股、智能推荐、个股分析） |

### 市场分析

| Skill | 说明 |
|-------|------|
| [sector-analysis](sector-analysis/) | 板块分析（行业热度、板块轮动） |
| [stock-sector](stock-sector/) | 股票板块分类与成分股查询 |
| [stock-comparison](stock-comparison/) | 多股票横向对比分析 |
| [market-sentiment](market-sentiment/) | 市场情绪指标分析 |
| [social-sentiment](social-sentiment/) | 社交媒体情绪分析 |

### 监控与通知

| Skill | 说明 |
|-------|------|
| [realtime-monitor](realtime-monitor/) | 实时行情监控与异动检测 |
| [signal-notify](signal-notify/) | 交易信号通知（微信、邮件、钉钉） |
| [websocket-push](websocket-push/) | WebSocket 实时数据推送 |
| [task-scheduler](task-scheduler/) | 定时任务调度 |

### 工具

| Skill | 说明 |
|-------|------|
| [data-quality](data-quality/) | 数据质量检查与清洗 |
| [data-storage](data-storage/) | 数据持久化存储 |
| [report-export](report-export/) | 分析报告导出（PDF、Excel） |
| [dashboard-plus](dashboard-plus/) | 仪表盘增强组件 |
| [hot-reload](hot-reload/) | 策略热重载 |

---

## 使用示例

```python
# 获取股票实时行情
from akshare-stock.scripts.stock_cli import get_realtime_quote
data = get_realtime_quote("600519")

# 股票综合评分
from stock-scoring.scripts.scoring_cli import score_stock
result = score_stock("600519")

# 策略回测
from backtest.scripts.backtest_cli import run_backtest
result = run_backtest("600519", strategy="ma_cross")

# AI 智能推荐
from ai-agent.scripts.agent_cli import ai_recommend_stocks
result = ai_recommend_stocks(preference="科技", risk_level="中等", top_n=5)
```

---

## 免责声明

本项目仅供学习研究使用，不构成任何投资建议。股市有风险，投资需谨慎。使用者需自行承担交易风险。

---

## 许可证

[MIT License](LICENSE)
