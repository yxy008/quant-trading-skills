#!/usr/bin/env python3
"""
持仓集中度风险分析系统
支持个股集中度、行业集中度、HHI指数、集中度预警
"""
import argparse
import json
import sys
import os
import time
from datetime import datetime

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


# ==================== 行业分类映射 ====================

INDUSTRY_MAP = {
    "银行": "金融",
    "保险": "金融",
    "证券": "金融",
    "券商": "金融",
    "信托": "金融",
    "白酒": "食品饮料",
    "啤酒": "食品饮料",
    "乳业": "食品饮料",
    "调味品": "食品饮料",
    "食品": "食品饮料",
    "饮料": "食品饮料",
    "医药": "医药生物",
    "医疗": "医药生物",
    "生物": "医药生物",
    "制药": "医药生物",
    "芯片": "电子",
    "半导体": "电子",
    "集成电路": "电子",
    "电子": "电子",
    "光伏": "新能源",
    "风电": "新能源",
    "锂电": "新能源",
    "新能源": "新能源",
    "储能": "新能源",
    "汽车": "汽车",
    "整车": "汽车",
    "零部件": "汽车",
    "房地产": "房地产",
    "地产": "房地产",
    "建筑": "建筑装饰",
    "建材": "建筑材料",
    "水泥": "建筑材料",
    "钢铁": "钢铁",
    "有色": "有色金属",
    "煤炭": "煤炭",
    "石油": "石油石化",
    "石化": "石油石化",
    "化工": "基础化工",
    "化学": "基础化工",
    "军工": "国防军工",
    "航空": "交通运输",
    "机场": "交通运输",
    "港口": "交通运输",
    "铁路": "交通运输",
    "电力": "公用事业",
    "水务": "公用事业",
    "燃气": "公用事业",
    "环保": "公用事业",
    "通信": "通信",
    "5G": "通信",
    "计算机": "计算机",
    "软件": "计算机",
    "互联网": "传媒",
    "传媒": "传媒",
    "游戏": "传媒",
    "家电": "家用电器",
    "电器": "家用电器",
    "纺织": "纺织服装",
    "服装": "纺织服装",
    "农业": "农林牧渔",
    "养殖": "农林牧渔",
    "零售": "商贸零售",
    "商业": "商贸零售",
    "旅游": "社会服务",
    "酒店": "社会服务",
    "机械": "机械设备",
    "设备": "机械设备",
}


def classify_industry(name):
    """根据股票名称推断行业"""
    for keyword, industry in INDUSTRY_MAP.items():
        if keyword in name:
            return industry
    return "其他"


# ==================== 持仓集中度分析 ====================

def concentration_analysis(positions):
    """
    持仓集中度分析

    参数:
        positions: 持仓列表，每项包含 {"代码": str, "名称": str, "市值": float}

    返回: 集中度分析结果
    """
    if not positions:
        return {"error": "持仓数据为空"}

    total_value = sum(p["市值"] for p in positions)

    if total_value <= 0:
        return {"error": "持仓总市值为0"}

    # 个股集中度
    stock_concentration = []
    for p in positions:
        weight = p["市值"] / total_value * 100
        stock_concentration.append({
            "代码": p["代码"],
            "名称": p["名称"],
            "市值": round(p["市值"], 2),
            "权重": f"{weight:.2f}%",
        })

    stock_concentration.sort(key=lambda x: float(x["权重"].replace("%", "")), reverse=True)

    # Top N 集中度
    top1_weight = float(stock_concentration[0]["权重"].replace("%", "")) if stock_concentration else 0
    top3_weight = sum(float(s["权重"].replace("%", "")) for s in stock_concentration[:3])
    top5_weight = sum(float(s["权重"].replace("%", "")) for s in stock_concentration[:5])

    # HHI指数（赫芬达尔指数）
    weights = [float(s["权重"].replace("%", "")) / 100 for s in stock_concentration]
    hhi = sum(w ** 2 for w in weights) * 10000

    # 行业集中度
    industry_weights = {}
    for p in positions:
        industry = classify_industry(p["名称"])
        weight = p["市值"] / total_value * 100
        industry_weights[industry] = industry_weights.get(industry, 0) + weight

    industry_concentration = [
        {"行业": ind, "权重": f"{w:.2f}%"}
        for ind, w in sorted(industry_weights.items(), key=lambda x: x[1], reverse=True)
    ]

    top_industry_weight = float(industry_concentration[0]["权重"].replace("%", "")) if industry_concentration else 0

    # 行业HHI
    ind_weights = [float(i["权重"].replace("%", "")) / 100 for i in industry_concentration]
    industry_hhi = sum(w ** 2 for w in ind_weights) * 10000

    # 风险评级
    warnings = []
    risk_level = "低"

    if top1_weight > 20:
        warnings.append(f"单只股票{stock_concentration[0]['名称']}权重{top1_weight:.1f}%过高（>20%），建议分散")
        risk_level = "高"
    elif top1_weight > 15:
        warnings.append(f"单只股票权重{top1_weight:.1f}%偏高（>15%），关注集中度风险")
        risk_level = "中"

    if top3_weight > 50:
        warnings.append(f"前3大持仓权重{top3_weight:.1f}%过高（>50%），组合分散度不足")
        if risk_level != "高":
            risk_level = "高"
    elif top3_weight > 40:
        warnings.append(f"前3大持仓权重{top3_weight:.1f}%偏高（>40%）")
        if risk_level == "低":
            risk_level = "中"

    if top5_weight > 70:
        warnings.append(f"前5大持仓权重{top5_weight:.1f}%过高（>70%）")
        if risk_level != "高":
            risk_level = "高"

    if hhi > 2000:
        warnings.append(f"个股HHI指数{hhi:.0f}过高（>2000），集中度风险显著")
    elif hhi > 1500:
        warnings.append(f"个股HHI指数{hhi:.0f}偏高（>1500）")

    if top_industry_weight > 40:
        warnings.append(f"最大行业'{industry_concentration[0]['行业']}'权重{top_industry_weight:.1f}%过高（>40%），行业集中度风险大")
        if risk_level != "高":
            risk_level = "高"
    elif top_industry_weight > 30:
        warnings.append(f"最大行业权重{top_industry_weight:.1f}%偏高（>30%）")
        if risk_level == "低":
            risk_level = "中"

    if len(positions) < 5:
        warnings.append(f"持仓数量仅{len(positions)}只，数量过少，分散度不足")
        if risk_level == "低":
            risk_level = "中"

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "持仓概况": {
            "持仓数量": len(positions),
            "持仓总市值": round(total_value, 2),
        },
        "个股集中度": {
            "排名": stock_concentration,
            "Top1权重": f"{top1_weight:.2f}%",
            "Top3权重": f"{top3_weight:.2f}%",
            "Top5权重": f"{top5_weight:.2f}%",
            "HHI指数": round(hhi, 0),
        },
        "行业集中度": {
            "行业分布": industry_concentration,
            "最大行业权重": f"{top_industry_weight:.2f}%",
            "行业HHI": round(industry_hhi, 0),
        },
        "风险等级": risk_level,
        "预警": warnings if warnings else ["持仓集中度在合理范围内"],
        "参考标准": {
            "个股HHI": "HHI<1000低集中，1000-1800中等，>1800高集中",
            "行业HHI": "行业HHI<1500分散良好，>2500集中度过高",
            "单只上限": "单只股票建议不超过15%-20%",
            "行业上限": "单一行业建议不超过30%-40%",
        },
    }


# ==================== 集中度优化建议 ====================

def concentration_optimization(positions, max_single_weight=15, max_industry_weight=30):
    """
    集中度优化建议
    给出降低集中度的具体操作建议

    参数:
        positions: 持仓列表
        max_single_weight: 单只最大权重（%）
        max_industry_weight: 单行业最大权重（%）

    返回: 优化建议
    """
    analysis = concentration_analysis(positions)
    if "error" in analysis:
        return analysis

    suggestions = []
    total_value = sum(p["市值"] for p in positions)

    # 超限个股
    stock_conc = analysis["个股集中度"]["排名"]
    for s in stock_conc:
        weight = float(s["权重"].replace("%", ""))
        if weight > max_single_weight:
            excess = weight - max_single_weight
            reduce_amount = total_value * excess / 100
            suggestions.append({
                "类型": "个股超限",
                "对象": f"{s['名称']}({s['代码']})",
                "当前权重": f"{weight:.1f}%",
                "建议权重": f"{max_single_weight}%",
                "建议减仓": f"{reduce_amount:,.0f}元",
            })

    # 超限行业
    industry_conc = analysis["行业集中度"]["行业分布"]
    for ind in industry_conc:
        weight = float(ind["权重"].replace("%", ""))
        if weight > max_industry_weight:
            suggestions.append({
                "类型": "行业超限",
                "对象": ind["行业"],
                "当前权重": f"{weight:.1f}%",
                "建议权重": f"{max_industry_weight}%",
                "建议": f"减少{ind['行业']}行业持仓或增加其他行业配置",
            })

    return {
        "分析时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "优化参数": {
            "单只上限": f"{max_single_weight}%",
            "行业上限": f"{max_industry_weight}%",
        },
        "当前风险等级": analysis["风险等级"],
        "优化建议": suggestions if suggestions else [{"类型": "无需调整", "对象": "全部持仓", "说明": "当前持仓集中度在合理范围内"}],
        "分散化建议": [
            "持仓数量建议8-20只，太少集中度高，太多难以跟踪",
            "行业分布建议覆盖5个以上不同行业",
            "大盘/中盘/小盘风格适度分散",
            "可考虑配置ETF实现一键分散",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description='持仓集中度风险分析系统')
    subparsers = parser.add_subparsers(dest='command')

    # 集中度分析
    analyze_parser = subparsers.add_parser('analyze', help='持仓集中度分析')
    analyze_parser.add_argument('--positions', required=True, help='持仓JSON，格式: [{"代码":"xxx","名称":"xxx","市值":xxx}]')

    # 优化建议
    opt_parser = subparsers.add_parser('optimize', help='集中度优化建议')
    opt_parser.add_argument('--positions', required=True, help='持仓JSON')
    opt_parser.add_argument('--max-single', type=float, default=15, help='单只最大权重(%)')
    opt_parser.add_argument('--max-industry', type=float, default=30, help='单行业最大权重(%)')

    args = parser.parse_args()

    try:
        if args.command == 'analyze':
            positions = json.loads(args.positions)
            result = concentration_analysis(positions)
            print(json.dumps(result, ensure_ascii=False, indent=2))

        elif args.command == 'optimize':
            positions = json.loads(args.positions)
            result = concentration_optimization(positions, args.max_single, args.max_industry)
            print(json.dumps(result, ensure_ascii=False, indent=2))

        else:
            parser.print_help()
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"持仓JSON解析失败: {str(e)}"}, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
