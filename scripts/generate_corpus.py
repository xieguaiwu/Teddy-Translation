#!/usr/bin/env python3
"""
Large-scale synthetic Chinese-English parallel corpus generator.

Produces realistic, diverse sentence pairs simulating news and literary
domains for SMT training. Uses template-based generation with bilingual
vocabularies, grammar rule maps, and Zipf-distributed word frequencies.

Usage:
    python scripts/generate_corpus.py --size 10000 --seed 42
    python scripts/generate_corpus.py --size 5000 --news-ratio 0.5 --output-dir my_corpus
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# Bilingual Lexicon
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BilingualEntry:
    """A single aligned Chinese-English lexical entry."""
    zh: str
    en: str
    pos: str           # part of speech tag
    domain: str        # "news", "lit", "common"
    freq_weight: float = 1.0  # relative frequency (higher = more common)


# ── Common function words / structures ───────────────────────────────────────

COMMON_PARTICLES = [
    BilingualEntry("的", "of", "part", "common", 12.0),
    BilingualEntry("了", "", "part", "common", 10.0),  # perfective aspect
    BilingualEntry("着", "", "part", "common", 4.0),   # progressive
    BilingualEntry("过", "", "part", "common", 3.0),   # experiential
    BilingualEntry("和", "and", "conj", "common", 8.0),
    BilingualEntry("但是", "but", "conj", "common", 5.0),
    BilingualEntry("或者", "or", "conj", "common", 4.0),
    BilingualEntry("因为", "because", "conj", "common", 6.0),
    BilingualEntry("所以", "therefore", "conj", "common", 5.0),
    BilingualEntry("虽然", "although", "conj", "common", 4.0),
    BilingualEntry("如果", "if", "conj", "common", 5.0),
    BilingualEntry("但是", "but", "conj", "common", 5.0),
    BilingualEntry("而且", "and also", "conj", "common", 4.0),
]

COMMON_ADVERBS = [
    BilingualEntry("非常", "very", "adv", "common", 8.0),
    BilingualEntry("很", "very", "adv", "common", 10.0),
    BilingualEntry("十分", "extremely", "adv", "common", 5.0),
    BilingualEntry("特别", "especially", "adv", "common", 5.0),
    BilingualEntry("一直", "always", "adv", "common", 6.0),
    BilingualEntry("经常", "often", "adv", "common", 5.0),
    BilingualEntry("已经", "already", "adv", "common", 7.0),
    BilingualEntry("可能", "possibly", "adv", "common", 6.0),
    BilingualEntry("也许", "perhaps", "adv", "common", 4.0),
    BilingualEntry("几乎", "almost", "adv", "common", 4.0),
    BilingualEntry("仍然", "still", "adv", "common", 5.0),
    BilingualEntry("逐渐", "gradually", "adv", "common", 4.0),
    BilingualEntry("迅速", "rapidly", "adv", "common", 4.0),
    BilingualEntry("完全", "completely", "adv", "common", 4.0),
    BilingualEntry("确实", "indeed", "adv", "common", 4.0),
    BilingualEntry("当然", "of course", "adv", "common", 5.0),
    BilingualEntry("显然", "obviously", "adv", "common", 4.0),
    BilingualEntry("实际上", "actually", "adv", "common", 5.0),
    BilingualEntry("最终", "finally", "adv", "common", 4.0),
    BilingualEntry("首先", "first", "adv", "common", 5.0),
]

COMMON_MEASURE = [
    BilingualEntry("个", "", "measure", "common", 10.0),
    BilingualEntry("种", "kind of", "measure", "common", 6.0),
    BilingualEntry("些", "some", "measure", "common", 7.0),
    BilingualEntry("年", "year", "measure", "common", 8.0),
    BilingualEntry("月", "month", "measure", "common", 6.0),
    BilingualEntry("日", "day", "measure", "common", 6.0),
    BilingualEntry("次", "time(s)", "measure", "common", 7.0),
    BilingualEntry("倍", "times", "measure", "common", 4.0),
]

COMMON_PRONOUNS = [
    BilingualEntry("我", "I", "pron", "common", 10.0),
    BilingualEntry("你", "you", "pron", "common", 9.0),
    BilingualEntry("他", "he", "pron", "common", 8.0),
    BilingualEntry("她", "she", "pron", "common", 7.0),
    BilingualEntry("它", "it", "pron", "common", 6.0),
    BilingualEntry("我们", "we", "pron", "common", 8.0),
    BilingualEntry("你们", "you", "pron", "common", 5.0),
    BilingualEntry("他们", "they", "pron", "common", 7.0),
    BilingualEntry("她们", "they", "pron", "common", 3.0),
    BilingualEntry("自己", "oneself", "pron", "common", 5.0),
    BilingualEntry("这", "this", "pron", "common", 8.0),
    BilingualEntry("那", "that", "pron", "common", 7.0),
    BilingualEntry("这些", "these", "pron", "common", 6.0),
    BilingualEntry("那些", "those", "pron", "common", 5.0),
    BilingualEntry("谁", "who", "pron", "common", 5.0),
    BilingualEntry("什么", "what", "pron", "common", 6.0),
    BilingualEntry("哪里", "where", "pron", "common", 4.0),
    BilingualEntry("怎么", "how", "pron", "common", 5.0),
    BilingualEntry("为什么", "why", "pron", "common", 4.0),
]

COMMON_MODALS = [
    BilingualEntry("可以", "can", "modal", "common", 8.0),
    BilingualEntry("能", "can", "modal", "common", 8.0),
    BilingualEntry("应该", "should", "modal", "common", 6.0),
    BilingualEntry("必须", "must", "modal", "common", 5.0),
    BilingualEntry("会", "will", "modal", "common", 9.0),
    BilingualEntry("要", "want to", "modal", "common", 9.0),
    BilingualEntry("需要", "need to", "modal", "common", 6.0),
    BilingualEntry("可能", "may", "modal", "common", 6.0),
    BilingualEntry("能够", "be able to", "modal", "common", 4.0),
    BilingualEntry("愿意", "be willing to", "modal", "common", 4.0),
    BilingualEntry("敢", "dare", "modal", "common", 3.0),
    BilingualEntry("值得", "be worth", "modal", "common", 3.0),
]

# ── News Domain: Politics ────────────────────────────────────────────────────

NEWS_POLITICS_TERMS = [
    BilingualEntry("政府", "government", "noun", "news", 9.0),
    BilingualEntry("总统", "president", "noun", "news", 7.0),
    BilingualEntry("选举", "election", "noun", "news", 6.0),
    BilingualEntry("政策", "policy", "noun", "news", 8.0),
    BilingualEntry("改革", "reform", "noun", "news", 7.0),
    BilingualEntry("议会", "parliament", "noun", "news", 5.0),
    BilingualEntry("外交", "diplomacy", "noun", "news", 6.0),
    BilingualEntry("法案", "bill", "noun", "news", 5.0),
    BilingualEntry("民主", "democracy", "noun", "news", 6.0),
    BilingualEntry("宪法", "constitution", "noun", "news", 5.0),
    BilingualEntry("官员", "official", "noun", "news", 7.0),
    BilingualEntry("部长", "minister", "noun", "news", 6.0),
    BilingualEntry("国会", "congress", "noun", "news", 5.0),
    BilingualEntry("投票", "vote", "noun", "news", 6.0),
    BilingualEntry("候选人", "candidate", "noun", "news", 5.0),
    BilingualEntry("反对党", "opposition party", "noun", "news", 4.0),
    BilingualEntry("执政党", "ruling party", "noun", "news", 4.0),
    BilingualEntry("抗议", "protest", "noun", "news", 5.0),
    BilingualEntry("制裁", "sanction", "noun", "news", 5.0),
    BilingualEntry("谈判", "negotiation", "noun", "news", 6.0),
    BilingualEntry("协议", "agreement", "noun", "news", 7.0),
    BilingualEntry("声明", "statement", "noun", "news", 6.0),
    BilingualEntry("新闻发布会", "press conference", "noun", "news", 5.0),
    BilingualEntry("峰会", "summit", "noun", "news", 5.0),
    BilingualEntry("领导人", "leader", "noun", "news", 7.0),
    BilingualEntry("主权", "sovereignty", "noun", "news", 4.0),
    BilingualEntry("领土", "territory", "noun", "news", 4.0),
    BilingualEntry("联合国", "United Nations", "noun", "news", 5.0),
    BilingualEntry("联盟", "alliance", "noun", "news", 5.0),
    BilingualEntry("战略", "strategy", "noun", "news", 6.0),
    BilingualEntry("局势", "situation", "noun", "news", 6.0),
    BilingualEntry("冲突", "conflict", "noun", "news", 6.0),
    BilingualEntry("和平", "peace", "noun", "news", 6.0),
    BilingualEntry("安全", "security", "noun", "news", 7.0),
    BilingualEntry("人权", "human rights", "noun", "news", 5.0),
    BilingualEntry("自由", "freedom", "noun", "news", 5.0),
    BilingualEntry("法律", "law", "noun", "news", 6.0),
    BilingualEntry("法院", "court", "noun", "news", 5.0),
    BilingualEntry("司法", "justice", "noun", "news", 4.0),
    BilingualEntry("腐败", "corruption", "noun", "news", 4.0),
    BilingualEntry("透明度", "transparency", "noun", "news", 3.0),
    BilingualEntry("权力", "power", "noun", "news", 6.0),
    BilingualEntry("责任", "responsibility", "noun", "news", 5.0),
    BilingualEntry("公民", "citizen", "noun", "news", 5.0),
    BilingualEntry("社会", "society", "noun", "news", 7.0),
    BilingualEntry("报道", "report", "noun", "news", 7.0),
    BilingualEntry("记者", "journalist", "noun", "news", 6.0),
    BilingualEntry("媒体", "media", "noun", "news", 6.0),
    BilingualEntry("舆论", "public opinion", "noun", "news", 5.0),
]

# ── News Domain: Economy ─────────────────────────────────────────────────────

NEWS_ECONOMY_TERMS = [
    BilingualEntry("经济", "economy", "noun", "news", 10.0),
    BilingualEntry("增长", "growth", "noun", "news", 9.0),
    BilingualEntry("市场", "market", "noun", "news", 9.0),
    BilingualEntry("贸易", "trade", "noun", "news", 8.0),
    BilingualEntry("投资", "investment", "noun", "news", 7.0),
    BilingualEntry("股票", "stock", "noun", "news", 6.0),
    BilingualEntry("金融", "finance", "noun", "news", 6.0),
    BilingualEntry("银行", "bank", "noun", "news", 7.0),
    BilingualEntry("利率", "interest rate", "noun", "news", 6.0),
    BilingualEntry("通货膨胀", "inflation", "noun", "news", 5.0),
    BilingualEntry("预算", "budget", "noun", "news", 5.0),
    BilingualEntry("赤字", "deficit", "noun", "news", 4.0),
    BilingualEntry("债务", "debt", "noun", "news", 5.0),
    BilingualEntry("税收", "tax", "noun", "news", 6.0),
    BilingualEntry("收入", "income", "noun", "news", 6.0),
    BilingualEntry("消费", "consumption", "noun", "news", 6.0),
    BilingualEntry("出口", "export", "noun", "news", 6.0),
    BilingualEntry("进口", "import", "noun", "news", 6.0),
    BilingualEntry("产业", "industry", "noun", "news", 7.0),
    BilingualEntry("制造业", "manufacturing", "noun", "news", 5.0),
    BilingualEntry("服务业", "service sector", "noun", "news", 5.0),
    BilingualEntry("企业", "enterprise", "noun", "news", 7.0),
    BilingualEntry("公司", "company", "noun", "news", 8.0),
    BilingualEntry("创业", "startup", "noun", "news", 4.0),
    BilingualEntry("就业", "employment", "noun", "news", 6.0),
    BilingualEntry("失业率", "unemployment rate", "noun", "news", 5.0),
    BilingualEntry("工资", "wage", "noun", "news", 5.0),
    BilingualEntry("价格", "price", "noun", "news", 7.0),
    BilingualEntry("成本", "cost", "noun", "news", 6.0),
    BilingualEntry("利润", "profit", "noun", "news", 6.0),
    BilingualEntry("亏损", "loss", "noun", "news", 5.0),
    BilingualEntry("资产", "asset", "noun", "news", 5.0),
    BilingualEntry("资源", "resource", "noun", "news", 6.0),
    BilingualEntry("能源", "energy", "noun", "news", 6.0),
    BilingualEntry("石油", "oil", "noun", "news", 5.0),
    BilingualEntry("房地产", "real estate", "noun", "news", 5.0),
    BilingualEntry("基础设施", "infrastructure", "noun", "news", 5.0),
    BilingualEntry("供应链", "supply chain", "noun", "news", 4.0),
    BilingualEntry("消费者", "consumer", "noun", "news", 5.0),
    BilingualEntry("竞争", "competition", "noun", "news", 5.0),
    BilingualEntry("垄断", "monopoly", "noun", "news", 3.0),
    BilingualEntry("补贴", "subsidy", "noun", "news", 4.0),
    BilingualEntry("关税", "tariff", "noun", "news", 4.0),
    BilingualEntry("国内生产总值", "GDP", "noun", "news", 5.0),
    BilingualEntry("外汇", "foreign exchange", "noun", "news", 4.0),
    BilingualEntry("汇率", "exchange rate", "noun", "news", 5.0),
    BilingualEntry("基金", "fund", "noun", "news", 5.0),
    BilingualEntry("指数", "index", "noun", "news", 5.0),
    BilingualEntry("衰退", "recession", "noun", "news", 4.0),
    BilingualEntry("复苏", "recovery", "noun", "news", 4.0),
]

# ── News Domain: Technology & Science ────────────────────────────────────────

NEWS_TECH_TERMS = [
    BilingualEntry("技术", "technology", "noun", "news", 9.0),
    BilingualEntry("科技", "science and technology", "noun", "news", 8.0),
    BilingualEntry("互联网", "internet", "noun", "news", 7.0),
    BilingualEntry("人工智能", "artificial intelligence", "noun", "news", 7.0),
    BilingualEntry("数据", "data", "noun", "news", 8.0),
    BilingualEntry("算法", "algorithm", "noun", "news", 6.0),
    BilingualEntry("软件", "software", "noun", "news", 6.0),
    BilingualEntry("硬件", "hardware", "noun", "news", 5.0),
    BilingualEntry("网络", "network", "noun", "news", 7.0),
    BilingualEntry("平台", "platform", "noun", "news", 6.0),
    BilingualEntry("应用程序", "application", "noun", "news", 5.0),
    BilingualEntry("系统", "system", "noun", "news", 7.0),
    BilingualEntry("机器人", "robot", "noun", "news", 5.0),
    BilingualEntry("自动化", "automation", "noun", "news", 4.0),
    BilingualEntry("数字化", "digitalization", "noun", "news", 4.0),
    BilingualEntry("云计算", "cloud computing", "noun", "news", 5.0),
    BilingualEntry("大数据", "big data", "noun", "news", 5.0),
    BilingualEntry("区块链", "blockchain", "noun", "news", 4.0),
    BilingualEntry("网络安全", "cybersecurity", "noun", "news", 5.0),
    BilingualEntry("隐私", "privacy", "noun", "news", 5.0),
    BilingualEntry("加密", "encryption", "noun", "news", 4.0),
    BilingualEntry("传感器", "sensor", "noun", "news", 4.0),
    BilingualEntry("芯片", "chip", "noun", "news", 5.0),
    BilingualEntry("半导体", "semiconductor", "noun", "news", 4.0),
    BilingualEntry("卫星", "satellite", "noun", "news", 4.0),
    BilingualEntry("航天", "aerospace", "noun", "news", 4.0),
    BilingualEntry("生物技术", "biotechnology", "noun", "news", 4.0),
    BilingualEntry("基因", "gene", "noun", "news", 4.0),
    BilingualEntry("疫苗", "vaccine", "noun", "news", 5.0),
    BilingualEntry("药物", "drug", "noun", "news", 5.0),
    BilingualEntry("医疗", "medical care", "noun", "news", 6.0),
    BilingualEntry("诊断", "diagnosis", "noun", "news", 4.0),
    BilingualEntry("治疗", "treatment", "noun", "news", 5.0),
    BilingualEntry("研究", "research", "noun", "news", 8.0),
    BilingualEntry("科学家", "scientist", "noun", "news", 6.0),
    BilingualEntry("实验", "experiment", "noun", "news", 5.0),
    BilingualEntry("发现", "discovery", "noun", "news", 6.0),
    BilingualEntry("创新", "innovation", "noun", "news", 7.0),
    BilingualEntry("专利", "patent", "noun", "news", 4.0),
    BilingualEntry("工程师", "engineer", "noun", "news", 5.0),
    BilingualEntry("开发", "development", "noun", "news", 7.0),
    BilingualEntry("发布", "release", "noun", "news", 5.0),
    BilingualEntry("版本", "version", "noun", "news", 4.0),
    BilingualEntry("用户", "user", "noun", "news", 6.0),
    BilingualEntry("设备", "device", "noun", "news", 5.0),
    BilingualEntry("电池", "battery", "noun", "news", 4.0),
    BilingualEntry("新能源", "renewable energy", "noun", "news", 5.0),
    BilingualEntry("太阳能", "solar energy", "noun", "news", 4.0),
    BilingualEntry("电动车", "electric vehicle", "noun", "news", 4.0),
    BilingualEntry("环保", "environmental protection", "noun", "news", 5.0),
    BilingualEntry("气候", "climate", "noun", "news", 5.0),
]

# ── News Domain: Shared News Verbs ───────────────────────────────────────────

NEWS_VERBS = [
    BilingualEntry("宣布", "announced", "verb", "news", 8.0),
    BilingualEntry("表示", "stated", "verb", "news", 9.0),
    BilingualEntry("报道", "reported", "verb", "news", 8.0),
    BilingualEntry("指出", "pointed out", "verb", "news", 7.0),
    BilingualEntry("强调", "emphasized", "verb", "news", 6.0),
    BilingualEntry("警告", "warned", "verb", "news", 5.0),
    BilingualEntry("呼吁", "called for", "verb", "news", 5.0),
    BilingualEntry("承诺", "promised", "verb", "news", 5.0),
    BilingualEntry("否认", "denied", "verb", "news", 4.0),
    BilingualEntry("证实", "confirmed", "verb", "news", 5.0),
    BilingualEntry("透露", "revealed", "verb", "news", 5.0),
    BilingualEntry("签署", "signed", "verb", "news", 5.0),
    BilingualEntry("批准", "approved", "verb", "news", 5.0),
    BilingualEntry("拒绝", "rejected", "verb", "news", 4.0),
    BilingualEntry("支持", "supported", "verb", "news", 6.0),
    BilingualEntry("反对", "opposed", "verb", "news", 5.0),
    BilingualEntry("讨论", "discussed", "verb", "news", 6.0),
    BilingualEntry("分析", "analyzed", "verb", "news", 5.0),
    BilingualEntry("调查", "investigated", "verb", "news", 5.0),
    BilingualEntry("发布", "released", "verb", "news", 6.0),
    BilingualEntry("提出", "proposed", "verb", "news", 6.0),
    BilingualEntry("推动", "promoted", "verb", "news", 5.0),
    BilingualEntry("实施", "implemented", "verb", "news", 5.0),
    BilingualEntry("加强", "strengthened", "verb", "news", 5.0),
    BilingualEntry("减少", "reduced", "verb", "news", 5.0),
    BilingualEntry("增加", "increased", "verb", "news", 5.0),
    BilingualEntry("下降", "declined", "verb", "news", 5.0),
    BilingualEntry("上升", "rose", "verb", "news", 5.0),
    BilingualEntry("达到", "reached", "verb", "news", 6.0),
    BilingualEntry("超过", "exceeded", "verb", "news", 5.0),
    BilingualEntry("预计", "is expected to", "verb", "news", 6.0),
    BilingualEntry("估计", "estimated", "verb", "news", 5.0),
    BilingualEntry("显示", "showed", "verb", "news", 7.0),
    BilingualEntry("表明", "indicated", "verb", "news", 6.0),
    BilingualEntry("反映", "reflected", "verb", "news", 5.0),
    BilingualEntry("引起", "caused", "verb", "news", 5.0),
    BilingualEntry("导致", "led to", "verb", "news", 6.0),
    BilingualEntry("影响", "affected", "verb", "news", 6.0),
    BilingualEntry("面临", "faced", "verb", "news", 5.0),
    BilingualEntry("遭遇", "encountered", "verb", "news", 4.0),
    BilingualEntry("发生", "occurred", "verb", "news", 6.0),
    BilingualEntry("爆发", "broke out", "verb", "news", 4.0),
    BilingualEntry("引发", "triggered", "verb", "news", 4.0),
    BilingualEntry("缓解", "eased", "verb", "news", 4.0),
    BilingualEntry("改善", "improved", "verb", "news", 5.0),
]

NEWS_ADJECTIVES = [
    BilingualEntry("重要的", "important", "adj", "news", 7.0),
    BilingualEntry("重大的", "significant", "adj", "news", 6.0),
    BilingualEntry("关键的", "critical", "adj", "news", 5.0),
    BilingualEntry("严重的", "serious", "adj", "news", 5.0),
    BilingualEntry("巨大的", "huge", "adj", "news", 6.0),
    BilingualEntry("迅速的", "rapid", "adj", "news", 5.0),
    BilingualEntry("持续的", "sustained", "adj", "news", 5.0),
    BilingualEntry("稳定的", "stable", "adj", "news", 5.0),
    BilingualEntry("积极的", "positive", "adj", "news", 5.0),
    BilingualEntry("消极的", "negative", "adj", "news", 4.0),
    BilingualEntry("复杂的", "complex", "adj", "news", 5.0),
    BilingualEntry("全面的", "comprehensive", "adj", "news", 4.0),
    BilingualEntry("有效的", "effective", "adj", "news", 5.0),
    BilingualEntry("广泛的", "widespread", "adj", "news", 5.0),
    BilingualEntry("激烈的", "fierce", "adj", "news", 4.0),
    BilingualEntry("密切的", "close", "adj", "news", 4.0),
    BilingualEntry("直接的", "direct", "adj", "news", 4.0),
    BilingualEntry("正式的", "formal", "adj", "news", 4.0),
    BilingualEntry("临时的", "temporary", "adj", "news", 3.0),
    BilingualEntry("长期的", "long-term", "adj", "news", 5.0),
    BilingualEntry("短期的", "short-term", "adj", "news", 4.0),
    BilingualEntry("初步的", "preliminary", "adj", "news", 3.0),
    BilingualEntry("最终的", "final", "adj", "news", 4.0),
    BilingualEntry("最新的", "latest", "adj", "news", 6.0),
    BilingualEntry("全球的", "global", "adj", "news", 6.0),
    BilingualEntry("国内的", "domestic", "adj", "news", 5.0),
    BilingualEntry("国际的", "international", "adj", "news", 6.0),
    BilingualEntry("经济的", "economic", "adj", "news", 7.0),
    BilingualEntry("政治的", "political", "adj", "news", 6.0),
    BilingualEntry("技术的", "technical", "adj", "news", 5.0),
    BilingualEntry("社会的", "social", "adj", "news", 5.0),
    BilingualEntry("文化的", "cultural", "adj", "news", 4.0),
    BilingualEntry("环境的", "environmental", "adj", "news", 5.0),
    BilingualEntry("军事的", "military", "adj", "news", 4.0),
    BilingualEntry("合法的", "legal", "adj", "news", 4.0),
]

# ── News Time & Place ────────────────────────────────────────────────────────

NEWS_TIME_EXPRESSIONS = [
    BilingualEntry("今天", "today", "time", "news", 7.0),
    BilingualEntry("昨天", "yesterday", "time", "news", 6.0),
    BilingualEntry("明天", "tomorrow", "time", "news", 5.0),
    BilingualEntry("上周", "last week", "time", "news", 5.0),
    BilingualEntry("本周", "this week", "time", "news", 5.0),
    BilingualEntry("下周", "next week", "time", "news", 4.0),
    BilingualEntry("上个月", "last month", "time", "news", 4.0),
    BilingualEntry("本月", "this month", "time", "news", 4.0),
    BilingualEntry("今年", "this year", "time", "news", 6.0),
    BilingualEntry("去年", "last year", "time", "news", 5.0),
    BilingualEntry("明年", "next year", "time", "news", 4.0),
    BilingualEntry("最近", "recently", "time", "news", 7.0),
    BilingualEntry("日前", "recently", "time", "news", 6.0),
    BilingualEntry("据报道", "according to reports", "time", "news", 6.0),
    BilingualEntry("当地时间", "local time", "time", "news", 5.0),
    BilingualEntry("上午", "morning", "time", "news", 5.0),
    BilingualEntry("下午", "afternoon", "time", "news", 5.0),
    BilingualEntry("晚上", "evening", "time", "news", 4.0),
    BilingualEntry("周一", "Monday", "time", "news", 4.0),
    BilingualEntry("周二", "Tuesday", "time", "news", 4.0),
    BilingualEntry("周三", "Wednesday", "time", "news", 4.0),
    BilingualEntry("周五", "Friday", "time", "news", 4.0),
    BilingualEntry("周末", "weekend", "time", "news", 4.0),
    BilingualEntry("年初", "early this year", "time", "news", 3.0),
    BilingualEntry("年底", "end of the year", "time", "news", 3.0),
]

NEWS_PLACES = [
    BilingualEntry("北京", "Beijing", "place", "news", 7.0),
    BilingualEntry("上海", "Shanghai", "place", "news", 6.0),
    BilingualEntry("纽约", "New York", "place", "news", 5.0),
    BilingualEntry("伦敦", "London", "place", "news", 5.0),
    BilingualEntry("东京", "Tokyo", "place", "news", 5.0),
    BilingualEntry("巴黎", "Paris", "place", "news", 4.0),
    BilingualEntry("华盛顿", "Washington", "place", "news", 5.0),
    BilingualEntry("莫斯科", "Moscow", "place", "news", 4.0),
    BilingualEntry("柏林", "Berlin", "place", "news", 4.0),
    BilingualEntry("首尔", "Seoul", "place", "news", 4.0),
    BilingualEntry("新加坡", "Singapore", "place", "news", 4.0),
    BilingualEntry("香港", "Hong Kong", "place", "news", 4.0),
    BilingualEntry("台湾", "Taiwan", "place", "news", 4.0),
    BilingualEntry("印度", "India", "place", "news", 4.0),
    BilingualEntry("欧盟", "the European Union", "place", "news", 5.0),
    BilingualEntry("中东", "the Middle East", "place", "news", 4.0),
    BilingualEntry("非洲", "Africa", "place", "news", 4.0),
    BilingualEntry("亚洲", "Asia", "place", "news", 4.0),
    BilingualEntry("欧洲", "Europe", "place", "news", 5.0),
    BilingualEntry("北美", "North America", "place", "news", 4.0),
    BilingualEntry("总部", "headquarters", "place", "news", 4.0),
    BilingualEntry("首都", "capital", "place", "news", 4.0),
    BilingualEntry("城市", "city", "place", "news", 5.0),
    BilingualEntry("地区", "region", "place", "news", 6.0),
    BilingualEntry("国家", "country", "place", "news", 7.0),
    BilingualEntry("世界", "world", "place", "news", 7.0),
]

# ── Literature Domain: Emotions ──────────────────────────────────────────────

LIT_EMOTION_TERMS = [
    BilingualEntry("爱", "love", "noun", "lit", 8.0),
    BilingualEntry("悲伤", "sorrow", "noun", "lit", 6.0),
    BilingualEntry("快乐", "joy", "noun", "lit", 7.0),
    BilingualEntry("孤独", "loneliness", "noun", "lit", 5.0),
    BilingualEntry("恐惧", "fear", "noun", "lit", 5.0),
    BilingualEntry("希望", "hope", "noun", "lit", 7.0),
    BilingualEntry("绝望", "despair", "noun", "lit", 4.0),
    BilingualEntry("愤怒", "anger", "noun", "lit", 5.0),
    BilingualEntry("幸福", "happiness", "noun", "lit", 7.0),
    BilingualEntry("痛苦", "pain", "noun", "lit", 6.0),
    BilingualEntry("思念", "longing", "noun", "lit", 5.0),
    BilingualEntry("感激", "gratitude", "noun", "lit", 4.0),
    BilingualEntry("遗憾", "regret", "noun", "lit", 4.0),
    BilingualEntry("温柔", "tenderness", "noun", "lit", 5.0),
    BilingualEntry("热情", "passion", "noun", "lit", 5.0),
    BilingualEntry("同情", "compassion", "noun", "lit", 4.0),
    BilingualEntry("自豪", "pride", "noun", "lit", 4.0),
    BilingualEntry("羞耻", "shame", "noun", "lit", 3.0),
    BilingualEntry("嫉妒", "jealousy", "noun", "lit", 3.0),
    BilingualEntry("勇气", "courage", "noun", "lit", 5.0),
    BilingualEntry("安慰", "comfort", "noun", "lit", 4.0),
    BilingualEntry("心灵", "soul", "noun", "lit", 5.0),
    BilingualEntry("眼泪", "tears", "noun", "lit", 5.0),
    BilingualEntry("微笑", "smile", "noun", "lit", 6.0),
    BilingualEntry("梦想", "dream", "noun", "lit", 6.0),
    BilingualEntry("记忆", "memory", "noun", "lit", 6.0),
    BilingualEntry("命运", "fate", "noun", "lit", 4.0),
    BilingualEntry("生命", "life", "noun", "lit", 7.0),
    BilingualEntry("死亡", "death", "noun", "lit", 5.0),
    BilingualEntry("灵魂", "soul", "noun", "lit", 4.0),
]

# ── Literature Domain: Nature ────────────────────────────────────────────────

LIT_NATURE_TERMS = [
    BilingualEntry("天空", "sky", "noun", "lit", 7.0),
    BilingualEntry("大海", "sea", "noun", "lit", 6.0),
    BilingualEntry("太阳", "sun", "noun", "lit", 7.0),
    BilingualEntry("月亮", "moon", "noun", "lit", 6.0),
    BilingualEntry("星星", "stars", "noun", "lit", 6.0),
    BilingualEntry("云", "cloud", "noun", "lit", 5.0),
    BilingualEntry("风", "wind", "noun", "lit", 6.0),
    BilingualEntry("雨", "rain", "noun", "lit", 6.0),
    BilingualEntry("雪", "snow", "noun", "lit", 5.0),
    BilingualEntry("山", "mountain", "noun", "lit", 6.0),
    BilingualEntry("河流", "river", "noun", "lit", 5.0),
    BilingualEntry("森林", "forest", "noun", "lit", 5.0),
    BilingualEntry("花", "flower", "noun", "lit", 6.0),
    BilingualEntry("树", "tree", "noun", "lit", 5.0),
    BilingualEntry("叶子", "leaf", "noun", "lit", 4.0),
    BilingualEntry("鸟", "bird", "noun", "lit", 5.0),
    BilingualEntry("鱼", "fish", "noun", "lit", 4.0),
    BilingualEntry("蝴蝶", "butterfly", "noun", "lit", 3.0),
    BilingualEntry("春天", "spring", "noun", "lit", 6.0),
    BilingualEntry("夏天", "summer", "noun", "lit", 5.0),
    BilingualEntry("秋天", "autumn", "noun", "lit", 5.0),
    BilingualEntry("冬天", "winter", "noun", "lit", 5.0),
    BilingualEntry("清晨", "early morning", "noun", "lit", 5.0),
    BilingualEntry("黄昏", "dusk", "noun", "lit", 4.0),
    BilingualEntry("夜晚", "night", "noun", "lit", 5.0),
    BilingualEntry("黎明", "dawn", "noun", "lit", 4.0),
    BilingualEntry("阳光", "sunlight", "noun", "lit", 5.0),
    BilingualEntry("月光", "moonlight", "noun", "lit", 4.0),
    BilingualEntry("彩虹", "rainbow", "noun", "lit", 3.0),
    BilingualEntry("露水", "dew", "noun", "lit", 3.0),
    BilingualEntry("雾", "fog", "noun", "lit", 3.0),
    BilingualEntry("雷声", "thunder", "noun", "lit", 3.0),
    BilingualEntry("闪电", "lightning", "noun", "lit", 3.0),
    BilingualEntry("波浪", "wave", "noun", "lit", 4.0),
    BilingualEntry("沙滩", "beach", "noun", "lit", 4.0),
    BilingualEntry("岛屿", "island", "noun", "lit", 4.0),
    BilingualEntry("山谷", "valley", "noun", "lit", 4.0),
    BilingualEntry("田野", "field", "noun", "lit", 4.0),
    BilingualEntry("花园", "garden", "noun", "lit", 5.0),
    BilingualEntry("草原", "grassland", "noun", "lit", 3.0),
]

# ── Literature Domain: Daily Life ────────────────────────────────────────────

LIT_DAILY_TERMS = [
    BilingualEntry("家", "home", "noun", "lit", 8.0),
    BilingualEntry("门", "door", "noun", "lit", 5.0),
    BilingualEntry("窗户", "window", "noun", "lit", 5.0),
    BilingualEntry("灯", "lamp", "noun", "lit", 4.0),
    BilingualEntry("桌子", "table", "noun", "lit", 5.0),
    BilingualEntry("椅子", "chair", "noun", "lit", 4.0),
    BilingualEntry("床", "bed", "noun", "lit", 5.0),
    BilingualEntry("书", "book", "noun", "lit", 7.0),
    BilingualEntry("信", "letter", "noun", "lit", 4.0),
    BilingualEntry("照片", "photograph", "noun", "lit", 4.0),
    BilingualEntry("画", "painting", "noun", "lit", 4.0),
    BilingualEntry("音乐", "music", "noun", "lit", 6.0),
    BilingualEntry("歌", "song", "noun", "lit", 5.0),
    BilingualEntry("茶", "tea", "noun", "lit", 5.0),
    BilingualEntry("酒", "wine", "noun", "lit", 4.0),
    BilingualEntry("饭", "meal", "noun", "lit", 5.0),
    BilingualEntry("面包", "bread", "noun", "lit", 3.0),
    BilingualEntry("路", "road", "noun", "lit", 6.0),
    BilingualEntry("桥", "bridge", "noun", "lit", 4.0),
    BilingualEntry("车站", "station", "noun", "lit", 4.0),
    BilingualEntry("火车", "train", "noun", "lit", 4.0),
    BilingualEntry("街道", "street", "noun", "lit", 5.0),
    BilingualEntry("商店", "shop", "noun", "lit", 4.0),
    BilingualEntry("医院", "hospital", "noun", "lit", 4.0),
    BilingualEntry("学校", "school", "noun", "lit", 5.0),
    BilingualEntry("教堂", "church", "noun", "lit", 3.0),
    BilingualEntry("市场", "market", "noun", "lit", 4.0),
    BilingualEntry("衣服", "clothes", "noun", "lit", 4.0),
    BilingualEntry("鞋子", "shoes", "noun", "lit", 3.0),
    BilingualEntry("帽子", "hat", "noun", "lit", 3.0),
    BilingualEntry("戒指", "ring", "noun", "lit", 3.0),
    BilingualEntry("礼物", "gift", "noun", "lit", 4.0),
    BilingualEntry("故事", "story", "noun", "lit", 6.0),
    BilingualEntry("诗歌", "poem", "noun", "lit", 4.0),
    BilingualEntry("孩子", "child", "noun", "lit", 6.0),
    BilingualEntry("母亲", "mother", "noun", "lit", 6.0),
    BilingualEntry("父亲", "father", "noun", "lit", 5.0),
    BilingualEntry("朋友", "friend", "noun", "lit", 7.0),
    BilingualEntry("邻居", "neighbor", "noun", "lit", 3.0),
    BilingualEntry("老人", "old person", "noun", "lit", 4.0),
    BilingualEntry("年轻人", "young person", "noun", "lit", 4.0),
    BilingualEntry("陌生人", "stranger", "noun", "lit", 3.0),
    BilingualEntry("旅行者", "traveler", "noun", "lit", 3.0),
]

# ── Literature Domain: Verbs ─────────────────────────────────────────────────

LIT_VERBS = [
    BilingualEntry("爱", "love", "verb", "lit", 7.0),
    BilingualEntry("恨", "hate", "verb", "lit", 4.0),
    BilingualEntry("喜欢", "like", "verb", "lit", 7.0),
    BilingualEntry("想", "think", "verb", "lit", 8.0),
    BilingualEntry("知道", "know", "verb", "lit", 7.0),
    BilingualEntry("相信", "believe", "verb", "lit", 6.0),
    BilingualEntry("忘记", "forget", "verb", "lit", 5.0),
    BilingualEntry("记得", "remember", "verb", "lit", 5.0),
    BilingualEntry("等待", "wait", "verb", "lit", 6.0),
    BilingualEntry("寻找", "search for", "verb", "lit", 5.0),
    BilingualEntry("发现", "discover", "verb", "lit", 5.0),
    BilingualEntry("失去", "lose", "verb", "lit", 5.0),
    BilingualEntry("离开", "leave", "verb", "lit", 6.0),
    BilingualEntry("回来", "come back", "verb", "lit", 5.0),
    BilingualEntry("到达", "arrive", "verb", "lit", 5.0),
    BilingualEntry("走过", "walk past", "verb", "lit", 4.0),
    BilingualEntry("站着", "stand", "verb", "lit", 5.0),
    BilingualEntry("坐着", "sit", "verb", "lit", 5.0),
    BilingualEntry("躺着", "lie down", "verb", "lit", 4.0),
    BilingualEntry("睡着", "fall asleep", "verb", "lit", 4.0),
    BilingualEntry("醒来", "wake up", "verb", "lit", 4.0),
    BilingualEntry("哭泣", "cry", "verb", "lit", 4.0),
    BilingualEntry("笑", "laugh", "verb", "lit", 5.0),
    BilingualEntry("说话", "speak", "verb", "lit", 6.0),
    BilingualEntry("沉默", "remain silent", "verb", "lit", 4.0),
    BilingualEntry("看着", "look at", "verb", "lit", 6.0),
    BilingualEntry("听着", "listen to", "verb", "lit", 5.0),
    BilingualEntry("闻着", "smell", "verb", "lit", 3.0),
    BilingualEntry("感受", "feel", "verb", "lit", 6.0),
    BilingualEntry("拥抱", "embrace", "verb", "lit", 4.0),
    BilingualEntry("亲吻", "kiss", "verb", "lit", 3.0),
    BilingualEntry("握住", "hold", "verb", "lit", 4.0),
    BilingualEntry("放下", "put down", "verb", "lit", 4.0),
    BilingualEntry("拿起", "pick up", "verb", "lit", 4.0),
    BilingualEntry("打开", "open", "verb", "lit", 5.0),
    BilingualEntry("关上", "close", "verb", "lit", 4.0),
    BilingualEntry("走进", "walk into", "verb", "lit", 4.0),
    BilingualEntry("走出", "walk out of", "verb", "lit", 4.0),
    BilingualEntry("穿过", "cross", "verb", "lit", 4.0),
    BilingualEntry("爬上", "climb up", "verb", "lit", 3.0),
    BilingualEntry("跳下", "jump down", "verb", "lit", 3.0),
    BilingualEntry("飞翔", "fly", "verb", "lit", 3.0),
    BilingualEntry("游泳", "swim", "verb", "lit", 3.0),
    BilingualEntry("唱歌", "sing", "verb", "lit", 4.0),
    BilingualEntry("跳舞", "dance", "verb", "lit", 3.0),
    BilingualEntry("画画", "paint", "verb", "lit", 3.0),
    BilingualEntry("写作", "write", "verb", "lit", 4.0),
    BilingualEntry("阅读", "read", "verb", "lit", 5.0),
    BilingualEntry("做梦", "dream", "verb", "lit", 4.0),
    BilingualEntry("思考", "ponder", "verb", "lit", 4.0),
    BilingualEntry("改变", "change", "verb", "lit", 5.0),
    BilingualEntry("成长", "grow", "verb", "lit", 4.0),
    BilingualEntry("衰老", "age", "verb", "lit", 3.0),
]

# ── Literature Domain: Adjectives ────────────────────────────────────────────

LIT_ADJECTIVES = [
    BilingualEntry("美丽的", "beautiful", "adj", "lit", 7.0),
    BilingualEntry("温柔的", "gentle", "adj", "lit", 5.0),
    BilingualEntry("善良的", "kind", "adj", "lit", 5.0),
    BilingualEntry("勇敢的", "brave", "adj", "lit", 4.0),
    BilingualEntry("聪明的", "wise", "adj", "lit", 4.0),
    BilingualEntry("愚蠢的", "foolish", "adj", "lit", 3.0),
    BilingualEntry("诚实的", "honest", "adj", "lit", 4.0),
    BilingualEntry("虚伪的", "hypocritical", "adj", "lit", 2.0),
    BilingualEntry("孤独的", "lonely", "adj", "lit", 5.0),
    BilingualEntry("快乐的", "happy", "adj", "lit", 6.0),
    BilingualEntry("悲伤的", "sad", "adj", "lit", 5.0),
    BilingualEntry("安静的", "quiet", "adj", "lit", 6.0),
    BilingualEntry("喧闹的", "noisy", "adj", "lit", 3.0),
    BilingualEntry("明亮的", "bright", "adj", "lit", 5.0),
    BilingualEntry("黑暗的", "dark", "adj", "lit", 5.0),
    BilingualEntry("温暖的", "warm", "adj", "lit", 6.0),
    BilingualEntry("寒冷的", "cold", "adj", "lit", 5.0),
    BilingualEntry("新鲜的", "fresh", "adj", "lit", 4.0),
    BilingualEntry("陈旧的", "old", "adj", "lit", 4.0),
    BilingualEntry("柔软的", "soft", "adj", "lit", 4.0),
    BilingualEntry("坚硬的", "hard", "adj", "lit", 3.0),
    BilingualEntry("深沉的", "deep", "adj", "lit", 4.0),
    BilingualEntry("浅薄的", "shallow", "adj", "lit", 2.0),
    BilingualEntry("遥远的", "distant", "adj", "lit", 5.0),
    BilingualEntry("靠近的", "close", "adj", "lit", 4.0),
    BilingualEntry("熟悉的", "familiar", "adj", "lit", 4.0),
    BilingualEntry("陌生的", "unfamiliar", "adj", "lit", 4.0),
    BilingualEntry("自由的", "free", "adj", "lit", 5.0),
    BilingualEntry("束缚的", "bound", "adj", "lit", 2.0),
    BilingualEntry("强大的", "powerful", "adj", "lit", 4.0),
    BilingualEntry("脆弱的", "fragile", "adj", "lit", 3.0),
    BilingualEntry("神秘的", "mysterious", "adj", "lit", 4.0),
    BilingualEntry("平凡的", "ordinary", "adj", "lit", 3.0),
    BilingualEntry("奇妙的", "wonderful", "adj", "lit", 4.0),
    BilingualEntry("可怕的", "terrible", "adj", "lit", 3.0),
    BilingualEntry("优雅的", "elegant", "adj", "lit", 4.0),
    BilingualEntry("粗糙的", "rough", "adj", "lit", 3.0),
    BilingualEntry("清晰的", "clear", "adj", "lit", 4.0),
    BilingualEntry("模糊的", "blurry", "adj", "lit", 3.0),
    BilingualEntry("永恒的", "eternal", "adj", "lit", 3.0),
    BilingualEntry("短暂的", "fleeting", "adj", "lit", 3.0),
    BilingualEntry("甜蜜的", "sweet", "adj", "lit", 4.0),
    BilingualEntry("苦涩的", "bitter", "adj", "lit", 3.0),
]


# ══════════════════════════════════════════════════════════════════════════════
# Noun Subcategorization (for grammatically valid slot assignment)
# ══════════════════════════════════════════════════════════════════════════════

# Entries suitable as subjects (agents: people, organizations, countries)
NEWS_SUBJECT_NOUNS = [
    BilingualEntry("政府", "government", "subj", "news", 9.0),
    BilingualEntry("总统", "president", "subj", "news", 7.0),
    BilingualEntry("官员", "official", "subj", "news", 7.0),
    BilingualEntry("部长", "minister", "subj", "news", 6.0),
    BilingualEntry("领导人", "leader", "subj", "news", 7.0),
    BilingualEntry("国会", "congress", "subj", "news", 5.0),
    BilingualEntry("议会", "parliament", "subj", "news", 5.0),
    BilingualEntry("反对党", "opposition party", "subj", "news", 4.0),
    BilingualEntry("执政党", "ruling party", "subj", "news", 4.0),
    BilingualEntry("记者", "journalist", "subj", "news", 6.0),
    BilingualEntry("媒体", "media", "subj", "news", 6.0),
    BilingualEntry("科学家", "scientist", "subj", "news", 6.0),
    BilingualEntry("工程师", "engineer", "subj", "news", 5.0),
    BilingualEntry("公司", "company", "subj", "news", 8.0),
    BilingualEntry("企业", "enterprise", "subj", "news", 7.0),
    BilingualEntry("银行", "bank", "subj", "news", 7.0),
    BilingualEntry("消费者", "consumer", "subj", "news", 5.0),
    BilingualEntry("投资者", "investor", "subj", "news", 5.0),
    BilingualEntry("联合国", "United Nations", "subj", "news", 5.0),
    BilingualEntry("联盟", "alliance", "subj", "news", 5.0),
    BilingualEntry("军队", "military", "subj", "news", 4.0),
    BilingualEntry("法院", "court", "subj", "news", 5.0),
    BilingualEntry("专家", "expert", "subj", "news", 6.0),
    BilingualEntry("分析师", "analyst", "subj", "news", 4.0),
    BilingualEntry("监管机构", "regulator", "subj", "news", 4.0),
    BilingualEntry("央行", "central bank", "subj", "news", 4.0),
    BilingualEntry("委员会", "committee", "subj", "news", 5.0),
    BilingualEntry("代表团", "delegation", "subj", "news", 3.0),
]

# News non-subject nouns (abstract concepts, phenomena, measure terms)
NEWS_OBJECT_NOUNS = [
    BilingualEntry("经济", "economy", "obj", "news", 10.0),
    BilingualEntry("政策", "policy", "obj", "news", 8.0),
    BilingualEntry("改革", "reform", "obj", "news", 7.0),
    BilingualEntry("法案", "bill", "obj", "news", 5.0),
    BilingualEntry("协议", "agreement", "obj", "news", 7.0),
    BilingualEntry("声明", "statement", "obj", "news", 6.0),
    BilingualEntry("战略", "strategy", "obj", "news", 6.0),
    BilingualEntry("制裁", "sanction", "obj", "news", 5.0),
    BilingualEntry("谈判", "negotiation", "obj", "news", 6.0),
    BilingualEntry("民主", "democracy", "obj", "news", 5.0),
    BilingualEntry("宪法", "constitution", "obj", "news", 4.0),
    BilingualEntry("选举", "election", "obj", "news", 6.0),
    BilingualEntry("投票", "vote", "obj", "news", 6.0),
    BilingualEntry("和平", "peace", "obj", "news", 5.0),
    BilingualEntry("安全", "security", "obj", "news", 7.0),
    BilingualEntry("人权", "human rights", "obj", "news", 5.0),
    BilingualEntry("自由", "freedom", "obj", "news", 5.0),
    BilingualEntry("法律", "law", "obj", "news", 6.0),
    BilingualEntry("腐败", "corruption", "obj", "news", 4.0),
    BilingualEntry("透明度", "transparency", "obj", "news", 3.0),
    BilingualEntry("贸易", "trade", "obj", "news", 8.0),
    BilingualEntry("投资", "investment", "obj", "news", 7.0),
    BilingualEntry("增长", "growth", "obj", "news", 8.0),
    BilingualEntry("市场", "market", "obj", "news", 8.0),
    BilingualEntry("金融", "finance", "obj", "news", 6.0),
    BilingualEntry("预算", "budget", "obj", "news", 5.0),
    BilingualEntry("债务", "debt", "obj", "news", 5.0),
    BilingualEntry("税收", "tax", "obj", "news", 6.0),
    BilingualEntry("收入", "income", "obj", "news", 6.0),
    BilingualEntry("就业", "employment", "obj", "news", 6.0),
    BilingualEntry("通货膨胀", "inflation", "obj", "news", 5.0),
    BilingualEntry("利率", "interest rate", "obj", "news", 5.0),
    BilingualEntry("竞争", "competition", "obj", "news", 5.0),
    BilingualEntry("补贴", "subsidy", "obj", "news", 4.0),
    BilingualEntry("关税", "tariff", "obj", "news", 4.0),
    BilingualEntry("资源", "resource", "obj", "news", 6.0),
    BilingualEntry("能源", "energy", "obj", "news", 6.0),
    BilingualEntry("基础设施", "infrastructure", "obj", "news", 5.0),
    BilingualEntry("供应链", "supply chain", "obj", "news", 4.0),
    BilingualEntry("技术", "technology", "obj", "news", 8.0),
    BilingualEntry("创新", "innovation", "obj", "news", 6.0),
    BilingualEntry("数据", "data", "obj", "news", 7.0),
    BilingualEntry("人工智能", "artificial intelligence", "obj", "news", 6.0),
    BilingualEntry("研究", "research", "obj", "news", 7.0),
    BilingualEntry("开发", "development", "obj", "news", 7.0),
    BilingualEntry("网络安全", "cybersecurity", "obj", "news", 4.0),
    BilingualEntry("隐私", "privacy", "obj", "news", 5.0),
    BilingualEntry("疫苗", "vaccine", "obj", "news", 5.0),
    BilingualEntry("药物", "drug", "obj", "news", 5.0),
    BilingualEntry("治疗", "treatment", "obj", "news", 5.0),
    BilingualEntry("环保", "environmental protection", "obj", "news", 5.0),
    BilingualEntry("气候", "climate", "obj", "news", 5.0),
    BilingualEntry("新能源", "renewable energy", "obj", "news", 4.0),
    BilingualEntry("出口", "export", "obj", "news", 6.0),
    BilingualEntry("进口", "import", "obj", "news", 6.0),
    BilingualEntry("价格", "price", "obj", "news", 7.0),
    BilingualEntry("利润", "profit", "obj", "news", 5.0),
    BilingualEntry("资产", "asset", "obj", "news", 5.0),
    BilingualEntry("基金", "fund", "obj", "news", 5.0),
    BilingualEntry("网络", "network", "obj", "news", 6.0),
    BilingualEntry("系统", "system", "obj", "news", 7.0),
]

# Literature: subjects (people, personified nature)
LIT_SUBJECT_NOUNS = [
    BilingualEntry("孩子", "child", "subj", "lit", 7.0),
    BilingualEntry("母亲", "mother", "subj", "lit", 6.0),
    BilingualEntry("父亲", "father", "subj", "lit", 5.0),
    BilingualEntry("朋友", "friend", "subj", "lit", 7.0),
    BilingualEntry("邻居", "neighbor", "subj", "lit", 3.0),
    BilingualEntry("老人", "old man", "subj", "lit", 5.0),
    BilingualEntry("年轻人", "young man", "subj", "lit", 4.0),
    BilingualEntry("陌生人", "stranger", "subj", "lit", 4.0),
    BilingualEntry("旅行者", "traveler", "subj", "lit", 3.0),
    BilingualEntry("女孩", "girl", "subj", "lit", 5.0),
    BilingualEntry("男孩", "boy", "subj", "lit", 4.0),
    BilingualEntry("女人", "woman", "subj", "lit", 5.0),
    BilingualEntry("男人", "man", "subj", "lit", 5.0),
    BilingualEntry("诗人", "poet", "subj", "lit", 3.0),
    BilingualEntry("画家", "painter", "subj", "lit", 3.0),
    BilingualEntry("音乐家", "musician", "subj", "lit", 3.0),
    BilingualEntry("作家", "writer", "subj", "lit", 4.0),
    BilingualEntry("太阳", "sun", "subj", "lit", 6.0),
    BilingualEntry("月亮", "moon", "subj", "lit", 5.0),
    BilingualEntry("风", "wind", "subj", "lit", 5.0),
    BilingualEntry("鸟", "bird", "subj", "lit", 4.0),
    BilingualEntry("蝴蝶", "butterfly", "subj", "lit", 3.0),
    BilingualEntry("雨", "rain", "subj", "lit", 4.0),
    BilingualEntry("雪", "snow", "subj", "lit", 3.0),
]

# Literature: objects (emotions, concrete things, natural phenomena)
LIT_OBJECT_NOUNS = [
    BilingualEntry("爱", "love", "obj", "lit", 8.0),
    BilingualEntry("悲伤", "sorrow", "obj", "lit", 6.0),
    BilingualEntry("快乐", "joy", "obj", "lit", 7.0),
    BilingualEntry("孤独", "loneliness", "obj", "lit", 5.0),
    BilingualEntry("恐惧", "fear", "obj", "lit", 5.0),
    BilingualEntry("希望", "hope", "obj", "lit", 7.0),
    BilingualEntry("绝望", "despair", "obj", "lit", 4.0),
    BilingualEntry("愤怒", "anger", "obj", "lit", 5.0),
    BilingualEntry("幸福", "happiness", "obj", "lit", 7.0),
    BilingualEntry("痛苦", "pain", "obj", "lit", 6.0),
    BilingualEntry("思念", "longing", "obj", "lit", 5.0),
    BilingualEntry("勇气", "courage", "obj", "lit", 5.0),
    BilingualEntry("安慰", "comfort", "obj", "lit", 4.0),
    BilingualEntry("眼泪", "tears", "obj", "lit", 5.0),
    BilingualEntry("微笑", "smile", "obj", "lit", 6.0),
    BilingualEntry("梦想", "dream", "obj", "lit", 6.0),
    BilingualEntry("记忆", "memory", "obj", "lit", 6.0),
    BilingualEntry("命运", "fate", "obj", "lit", 4.0),
    BilingualEntry("生命", "life", "obj", "lit", 7.0),
    BilingualEntry("死亡", "death", "obj", "lit", 5.0),
    BilingualEntry("天空", "sky", "obj", "lit", 6.0),
    BilingualEntry("大海", "sea", "obj", "lit", 5.0),
    BilingualEntry("星星", "stars", "obj", "lit", 5.0),
    BilingualEntry("花", "flower", "obj", "lit", 6.0),
    BilingualEntry("树", "tree", "obj", "lit", 5.0),
    BilingualEntry("叶子", "leaf", "obj", "lit", 4.0),
    BilingualEntry("花园", "garden", "obj", "lit", 5.0),
    BilingualEntry("书", "book", "obj", "lit", 7.0),
    BilingualEntry("信", "letter", "obj", "lit", 4.0),
    BilingualEntry("照片", "photograph", "obj", "lit", 4.0),
    BilingualEntry("音乐", "music", "obj", "lit", 6.0),
    BilingualEntry("歌", "song", "obj", "lit", 5.0),
    BilingualEntry("茶", "tea", "obj", "lit", 4.0),
    BilingualEntry("酒", "wine", "obj", "lit", 4.0),
    BilingualEntry("饭", "meal", "obj", "lit", 4.0),
    BilingualEntry("礼物", "gift", "obj", "lit", 4.0),
    BilingualEntry("故事", "story", "obj", "lit", 6.0),
    BilingualEntry("诗歌", "poem", "obj", "lit", 4.0),
    BilingualEntry("戒指", "ring", "obj", "lit", 3.0),
    BilingualEntry("河流", "river", "obj", "lit", 5.0),
    BilingualEntry("森林", "forest", "obj", "lit", 5.0),
    BilingualEntry("山谷", "valley", "obj", "lit", 4.0),
    BilingualEntry("田野", "field", "obj", "lit", 4.0),
    BilingualEntry("岛屿", "island", "obj", "lit", 4.0),
    BilingualEntry("波浪", "wave", "obj", "lit", 4.0),
    BilingualEntry("彩虹", "rainbow", "obj", "lit", 3.0),
    BilingualEntry("阳光", "sunlight", "obj", "lit", 5.0),
    BilingualEntry("月光", "moonlight", "obj", "lit", 4.0),
    BilingualEntry("路", "road", "obj", "lit", 5.0),
    BilingualEntry("桥", "bridge", "obj", "lit", 4.0),
    BilingualEntry("街道", "street", "obj", "lit", 5.0),
]

# ══════════════════════════════════════════════════════════════════════════════
# Lexicon Assembly
# ══════════════════════════════════════════════════════════════════════════════

ALL_LEXICONS = (
    COMMON_PARTICLES + COMMON_ADVERBS + COMMON_MEASURE +
    COMMON_PRONOUNS + COMMON_MODALS +
    NEWS_SUBJECT_NOUNS + NEWS_OBJECT_NOUNS +
    NEWS_POLITICS_TERMS + NEWS_ECONOMY_TERMS + NEWS_TECH_TERMS +
    NEWS_VERBS + NEWS_ADJECTIVES + NEWS_TIME_EXPRESSIONS + NEWS_PLACES +
    LIT_SUBJECT_NOUNS + LIT_OBJECT_NOUNS +
    LIT_EMOTION_TERMS + LIT_NATURE_TERMS + LIT_DAILY_TERMS +
    LIT_VERBS + LIT_ADJECTIVES
)


def _by_pos(entries: List[BilingualEntry], pos: str) -> List[BilingualEntry]:
    return [e for e in entries if e.pos == pos]


def _by_domain(entries: List[BilingualEntry], domain: str) -> List[BilingualEntry]:
    return [e for e in entries if e.domain == domain]


def _by_pos_domain(entries: List[BilingualEntry], pos: str, domain: str) -> List[BilingualEntry]:
    return [e for e in entries if e.pos == pos and e.domain == domain]


# Pre-computed lookups (subcategorized)
LEX_NEWS_SUBJ = NEWS_SUBJECT_NOUNS
LEX_NEWS_OBJ = NEWS_OBJECT_NOUNS
LEX_LIT_SUBJ = LIT_SUBJECT_NOUNS
LEX_LIT_OBJ = LIT_OBJECT_NOUNS

LEX_PRON = _by_pos(ALL_LEXICONS, "pron")
LEX_MODAL = _by_pos(ALL_LEXICONS, "modal")
LEX_CONJ = _by_pos(ALL_LEXICONS, "conj")
LEX_ADV = _by_pos(ALL_LEXICONS, "adv")

LEX_NEWS_VERB = _by_pos_domain(ALL_LEXICONS, "verb", "news")
LEX_NEWS_ADJ = _by_pos_domain(ALL_LEXICONS, "adj", "news")
LEX_NEWS_TIME = [e for e in ALL_LEXICONS if e.pos == "time" and e.domain == "news"]
LEX_NEWS_PLACE = [e for e in ALL_LEXICONS if e.pos == "place" and e.domain == "news"]

LEX_LIT_VERB = _by_pos_domain(ALL_LEXICONS, "verb", "lit")
LEX_LIT_ADJ = _by_pos_domain(ALL_LEXICONS, "adj", "lit")


# ══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ══════════════════════════════════════════════════════════════════════════════

def weighted_choice(rng: random.Random, entries: List[BilingualEntry]) -> BilingualEntry:
    """Pick an entry with probability proportional to its freq_weight."""
    if not entries:
        raise ValueError("Empty entry list")
    total = sum(e.freq_weight for e in entries)
    r = rng.uniform(0, total)
    cumulative = 0.0
    for e in entries:
        cumulative += e.freq_weight
        if r <= cumulative:
            return e
    return entries[-1]


def pick_n_weighted(rng: random.Random, entries: List[BilingualEntry], n: int) -> List[BilingualEntry]:
    """Pick n distinct entries with weighted probability."""
    if n >= len(entries):
        return list(entries)
    pool = list(entries)
    chosen = []
    for _ in range(n):
        e = weighted_choice(rng, pool)
        chosen.append(e)
        pool.remove(e)
    return chosen


def zipf_weight(rank: int, total: int) -> float:
    """Return approximate Zipf weight for rank k out of total N."""
    # Zipf: f(k) ∝ 1/k roughly
    return 1.0 / (rank + 1)


def apply_zipf_distribution(entries: List[BilingualEntry]) -> List[BilingualEntry]:
    """Re-weight entries using a Zipf-like distribution based on sort order."""
    sorted_entries = sorted(entries, key=lambda e: e.zh)  # stable order
    result = []
    for i, entry in enumerate(sorted_entries):
        weight = 1.0 / (i + 1)  # Zipf: 1/rank
        # Apply as a multiplier capped between 0.3 and 3.0
        multiplier = max(0.3, min(3.0, weight * len(sorted_entries) * 0.3))
        result.append(BilingualEntry(
            zh=entry.zh, en=entry.en, pos=entry.pos,
            domain=entry.domain,
            freq_weight=round(entry.freq_weight * multiplier, 2)
        ))
    return result


# Apply Zipf to all sub-lists
LEX_NEWS_SUBJ = apply_zipf_distribution(LEX_NEWS_SUBJ)
LEX_NEWS_OBJ = apply_zipf_distribution(LEX_NEWS_OBJ)
LEX_LIT_SUBJ = apply_zipf_distribution(LEX_LIT_SUBJ)
LEX_LIT_OBJ = apply_zipf_distribution(LEX_LIT_OBJ)
LEX_NEWS_VERB = apply_zipf_distribution(LEX_NEWS_VERB)
LEX_NEWS_ADJ = apply_zipf_distribution(LEX_NEWS_ADJ)
LEX_LIT_VERB = apply_zipf_distribution(LEX_LIT_VERB)
LEX_LIT_ADJ = apply_zipf_distribution(LEX_LIT_ADJ)
LEX_NEWS_PLACE = apply_zipf_distribution(LEX_NEWS_PLACE)
LEX_NEWS_TIME = apply_zipf_distribution(LEX_NEWS_TIME)
LEX_ADV = apply_zipf_distribution(LEX_ADV)
LEX_PRON = apply_zipf_distribution(LEX_PRON)
LEX_CONJ = apply_zipf_distribution(LEX_CONJ)
LEX_MODAL = apply_zipf_distribution(LEX_MODAL)


# ══════════════════════════════════════════════════════════════════════════════
# Template System
# ══════════════════════════════════════════════════════════════════════════════

# A template is defined as a pair:
#   (zh_structure, en_structure)
# Each structure is a list of "slots".
# A slot is either:
#   - A string literal (fixed text)
#   - A function (slot selector): called as fn(rng, prev_entries) → BilingualEntry

# For simplicity, we define slot types as string tags and a dispatch table.

class TemplateSlot:
    """Descriptor for a slot in a template that resolves to a bilingual entry."""
    def __init__(self, slot_type: str):
        self.slot_type = slot_type

    def resolve(self, rng: random.Random, prev_entries: list) -> BilingualEntry:
        raise NotImplementedError


class FixedSlot(TemplateSlot):
    """A fixed text that requires no selection."""
    def __init__(self, zh_text: str, en_text: str):
        super().__init__("fixed")
        self.entry = BilingualEntry(zh_text, en_text, "fixed", "common", 1.0)

    def resolve(self, rng, prev_entries):
        return self.entry


class PoolSlot(TemplateSlot):
    """A slot that draws from a pool of entries."""
    def __init__(self, slot_type: str, pool: List[BilingualEntry]):
        super().__init__(slot_type)
        self.pool = pool

    def resolve(self, rng, prev_entries):
        return weighted_choice(rng, self.pool)


# We'll use a simpler approach: template definitions as tuples of
# (zh_slots, en_slots, slot_resolvers) where resolvers is a dict
# mapping slot name → callable or list.

def make_resolver(pool: List[BilingualEntry]):
    """Create a resolver function for a pool."""
    def resolve(rng: random.Random) -> BilingualEntry:
        return weighted_choice(rng, pool)
    return resolve


def make_optional_resolver(pool: List[BilingualEntry], prob: float = 0.5):
    """Create a resolver that sometimes returns empty."""
    def resolve(rng: random.Random) -> BilingualEntry:
        if rng.random() < prob:
            return weighted_choice(rng, pool)
        return BilingualEntry("", "", "null", "common", 0.0)
    return resolve


# ── Template Definitions ─────────────────────────────────────────────────────

# Each template: (zh_parts, en_parts, resolvers, domain)
# parts are lists of: "text" for fixed text, or "tag" for substitutable slots
# resolvers: {tag: resolver_fn}
# domain: "news" | "lit" | "both"

@dataclass
class Template:
    zh_parts: List[str]
    en_parts: List[str]
    resolvers: Dict[str, callable]
    domain: str  # "news", "lit", "both"
    weight: float = 1.0

    def generate(self, rng: random.Random) -> Tuple[str, str]:
        """Generate a (zh_sentence, en_sentence) pair.

        Tags (like {S}, {V}) are resolved ONCE per generation call.
        The same bilingual entry is used for both the ZH and EN slots,
        ensuring proper cross-lingual alignment.
        """
        # Phase 1: Resolve all unique tags once
        resolved: Dict[str, BilingualEntry] = {}
        for part in self.zh_parts + self.en_parts:
            if part.startswith("{") and part.endswith("}"):
                tag = part[1:-1]
                if tag not in resolved:
                    resolver = self.resolvers.get(tag)
                    if resolver:
                        resolved[tag] = resolver(rng)
                    else:
                        resolved[tag] = BilingualEntry(tag, tag, "unknown", "common", 0.0)

        # Phase 2: Build ZH sentence
        zh_words = []
        for zp in self.zh_parts:
            if zp.startswith("{") and zp.endswith("}"):
                entry = resolved[zp[1:-1]]
                if entry.zh:
                    zh_words.append(entry.zh)
            else:
                zh_words.append(zp)

        # Phase 3: Build EN sentence
        en_words = []
        for ep in self.en_parts:
            if ep.startswith("{") and ep.endswith("}"):
                entry = resolved[ep[1:-1]]
                if entry.en:
                    en_words.append(entry.en)
            else:
                en_words.append(ep)

        # Clean up
        zh_sent = "".join(zh_words).strip()
        en_sent = " ".join(w for w in en_words if w).strip()
        en_sent = _normalize_en(en_sent)
        return zh_sent, en_sent


def _normalize_en(text: str) -> str:
    """Clean up English spacing and punctuation."""
    import re
    # Remove space before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Remove double spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ── Resolver Factories ───────────────────────────────────────────────────────

def r_pron(rng): return weighted_choice(rng, LEX_PRON)
def r_modal(rng): return weighted_choice(rng, LEX_MODAL)
def r_adv(rng): return weighted_choice(rng, LEX_ADV)
def r_conj(rng): return weighted_choice(rng, LEX_CONJ)
def r_news_verb(rng): return weighted_choice(rng, LEX_NEWS_VERB)
def r_news_adj(rng): return weighted_choice(rng, LEX_NEWS_ADJ)
def r_news_time(rng): return weighted_choice(rng, LEX_NEWS_TIME)
def r_news_place(rng): return weighted_choice(rng, LEX_NEWS_PLACE)
def r_news_subj(rng): return weighted_choice(rng, LEX_NEWS_SUBJ)
def r_news_obj(rng): return weighted_choice(rng, LEX_NEWS_OBJ)
def r_lit_verb(rng): return weighted_choice(rng, LEX_LIT_VERB)
def r_lit_adj(rng): return weighted_choice(rng, LEX_LIT_ADJ)
def r_lit_subj(rng): return weighted_choice(rng, LEX_LIT_SUBJ)
def r_lit_obj(rng): return weighted_choice(rng, LEX_LIT_OBJ)
def r_lit_noun(rng): return weighted_choice(rng, LEX_LIT_SUBJ + LEX_LIT_OBJ)
def r_news_noun(rng): return weighted_choice(rng, LEX_NEWS_SUBJ + LEX_NEWS_OBJ)


# ── NEWS Templates ───────────────────────────────────────────────────────────

# Template notation:
#   {S} = subject (pronoun or noun)
#   {V} = verb
#   {O} = object (noun)
#   {A} = adjective
#   {T} = time expression
#   {P} = place
#   {M} = modal
#   {C} = conjunction
#   {D} = adverb

NEWS_TEMPLATES = [
    # 1. Simple reporting: "Subject verbed object"
    Template(
        ["{S}", "{V}", "{O}", "。"],
        ["{S}", "{V}", "{O}"],
        {"S": r_news_subj, "V": r_news_verb, "O": r_news_obj},
        "news", 10.0
    ),
    # 2. With time: "Time, subject verbed object"
    Template(
        ["{T}", "，", "{S}", "{V}", "{O}", "。"],
        ["{T}", ",", "{S}", "{V}", "{O}"],
        {"T": r_news_time, "S": r_news_subj, "V": r_news_verb, "O": r_news_obj},
        "news", 8.0
    ),
    # 3. With place: "In place, subject verbed object"
    Template(
        ["在", "{P}", "，", "{S}", "{V}", "{O}", "。"],
        ["In", "{P}", ",", "{S}", "{V}", "{O}"],
        {"P": r_news_place, "S": r_news_subj, "V": r_news_verb, "O": r_news_obj},
        "news", 6.0
    ),
    # 4. With adjective: "Adjective subject verbed object"
    Template(
        ["{A}", "的", "{S}", "{V}", "{O}", "。"],
        ["The", "{A}", "{S}", "{V}", "{O}"],
        {"A": r_news_adj, "S": r_news_subj, "V": r_news_verb, "O": r_news_obj},
        "news", 5.0
    ),
    # 5. Negation: "Subject did not verb object"
    Template(
        ["{S}", "没有", "{V}", "{O}", "。"],
        ["{S}", "did not", "{V}", "{O}"],
        {"S": r_news_subj, "V": r_news_verb, "O": r_news_obj},
        "news", 4.0
    ),
    # 6. Modal: "Subject may verb object"
    Template(
        ["{S}", "{M}", "{V}", "{O}", "。"],
        ["{S}", "{M}", "{V}", "{O}"],
        {"S": r_news_subj, "M": r_modal, "V": r_news_verb, "O": r_news_obj},
        "news", 5.0
    ),
    # 7. Compound: "Subject1 verbed object1, and subject2 verbed object2"
    Template(
        ["{S1}", "{V1}", "{O1}", "，", "{C}", "{S2}", "{V2}", "{O2}", "。"],
        ["{S1}", "{V1}", "{O1}", ",", "{C}", "{S2}", "{V2}", "{O2}"],
        {"S1": r_news_subj, "V1": r_news_verb, "O1": r_news_obj,
         "C": r_conj, "S2": r_news_subj, "V2": r_news_verb, "O2": r_news_obj},
        "news", 4.0
    ),
    # 8. Because-therefore: "Because S1 V1 O1, S2 V2 O2"
    Template(
        ["因为", "{S1}", "{V1}", "{O1}", "，", "{S2}", "{V2}", "{O2}", "。"],
        ["Because", "{S1}", "{V1}", "{O1}", ",", "{S2}", "{V2}", "{O2}"],
        {"S1": r_news_subj, "V1": r_news_verb, "O1": r_news_obj,
         "S2": r_news_subj, "V2": r_news_verb, "O2": r_news_obj},
        "news", 4.0
    ),
    # 9. Statistics: "Data shows that S V O"
    Template(
        ["数据", "显示", "，", "{S}", "{V}", "{O}", "。"],
        ["Data", "shows", "that", "{S}", "{V}", "{O}"],
        {"S": r_news_subj, "V": r_news_verb, "O": r_news_obj},
        "news", 4.0
    ),
    # 10. Expert opinion: "S1 stated that S2 V O"
    Template(
        ["{S1}", "表示", "，", "{S2}", "{V}", "{O}", "。"],
        ["{S1}", "stated", "that", "{S2}", "{V}", "{O}"],
        {"S1": r_news_subj, "S2": r_news_subj, "V": r_news_verb, "O": r_news_obj},
        "news", 4.0
    ),
    # 11. Comparison: "S1 is more A than S2"
    Template(
        ["{S1}", "比", "{S2}", "更", "{A}", "。"],
        ["{S1}", "is", "more", "{A}", "than", "{S2}"],
        {"S1": r_news_subj, "S2": r_news_subj, "A": r_news_adj},
        "news", 4.0
    ),
    # 12. Announce: "S announced that O will be developed"
    Template(
        ["{S}", "宣布", "将", "推动", "{O}", "。"],
        ["{S}", "announced", "plans", "to", "promote", "{O}"],
        {"S": r_news_subj, "O": r_news_obj},
        "news", 3.0
    ),
    # 13. Question: "Will S V O?"
    Template(
        ["{S}", "会", "{V}", "{O}", "吗", "？"],
        ["Will", "{S}", "{V}", "{O}", "?"],
        {"S": r_news_subj, "V": r_news_verb, "O": r_news_obj},
        "news", 3.0
    ),
    # 14. Adverb + verb: "S rapidly V-ed O"
    Template(
        ["{S}", "{D}", "地", "{V}", "{O}", "。"],
        ["{S}", "{D}", "{V}", "{O}"],
        {"S": r_news_subj, "D": r_adv, "V": r_news_verb, "O": r_news_obj},
        "news", 4.0
    ),
    # 15. Multiple adjectives: "A1 and A2 S V O"
    Template(
        ["{A1}", "而", "{A2}", "的", "{S}", "{V}", "{O}", "。"],
        ["The", "{A1}", "and", "{A2}", "{S}", "{V}", "{O}"],
        {"A1": r_news_adj, "A2": r_news_adj, "S": r_news_subj,
         "V": r_news_verb, "O": r_news_obj},
        "news", 3.0
    ),
    # 16. Passive: "O was V-ed by S"
    Template(
        ["{O}", "被", "{S}", "{V}", "。"],
        ["{O}", "was", "{V}", "by", "{S}"],
        {"O": r_news_obj, "S": r_news_subj, "V": r_news_verb},
        "news", 4.0
    ),
    # 17. Trend: "O has been V-ing"
    Template(
        ["{O}", "一直", "在", "{V}", "。"],
        ["{O}", "has", "been", "{V}"],
        {"O": r_news_obj, "V": r_news_verb},
        "news", 3.0
    ),
    # 18. Breaking news: "Breaking: S V O at P"
    Template(
        ["快讯", "：", "在", "{P}", "，", "{S}", "{V}", "{O}", "。"],
        ["Breaking:", "In", "{P}", ",", "{S}", "{V}", "{O}"],
        {"P": r_news_place, "S": r_news_subj, "V": r_news_verb, "O": r_news_obj},
        "news", 3.0
    ),
    # 19. According to S1, S2 V O
    Template(
        ["据", "{S1}", "报道", "，", "{S2}", "{V}", "{O}", "。"],
        ["According", "to", "{S1}", ",", "{S2}", "{V}", "{O}"],
        {"S1": r_news_subj, "S2": r_news_subj, "V": r_news_verb, "O": r_news_obj},
        "news", 4.0
    ),
    # 20. S emphasized that O is important
    Template(
        ["{S}", "强调", "{O}", "的", "重要性", "。"],
        ["{S}", "emphasized", "the", "importance", "of", "{O}"],
        {"S": r_news_subj, "O": r_news_obj},
        "news", 3.0
    ),
    # 21. S warned about O
    Template(
        ["{S}", "对", "{O}", "发出", "警告", "。"],
        ["{S}", "warned", "about", "{O}"],
        {"S": r_news_subj, "O": r_news_obj},
        "news", 3.0
    ),
    # 22. Amid concerns over O, S V
    Template(
        ["在", "对", "{O}", "的", "担忧", "中", "，", "{S}", "{V}", "。"],
        ["Amid", "concerns", "over", "{O}", ",", "{S}", "{V}"],
        {"O": r_news_obj, "S": r_news_subj, "V": r_news_verb},
        "news", 3.0
    ),
    # 23. S called for O
    Template(
        ["{S}", "呼吁", "加强", "{O}", "。"],
        ["{S}", "called", "for", "strengthening", "{O}"],
        {"S": r_news_subj, "O": r_news_obj},
        "news", 3.0
    ),
    # 24. S rejected claims about O
    Template(
        ["{S}", "否认", "关于", "{O}", "的", "指控", "。"],
        ["{S}", "denied", "allegations", "about", "{O}"],
        {"S": r_news_subj, "O": r_news_obj},
        "news", 3.0
    ),
    # 25. S confirmed O progress
    Template(
        ["{S}", "证实", "{O}", "取得", "进展", "。"],
        ["{S}", "confirmed", "progress", "on", "{O}"],
        {"S": r_news_subj, "O": r_news_obj},
        "news", 3.0
    ),
    # 26. New: S reported that O increased
    Template(
        ["{S}", "报告", "称", "，", "{O}", "有所", "增加", "。"],
        ["{S}", "reported", "that", "{O}", "has", "increased"],
        {"S": r_news_subj, "O": r_news_obj},
        "news", 3.0
    ),
    # 27. New: S and S2 reached agreement on O
    Template(
        ["{S1}", "与", "{S2}", "就", "{O}", "达成", "协议", "。"],
        ["{S1}", "and", "{S2}", "reached", "agreement", "on", "{O}"],
        {"S1": r_news_subj, "S2": r_news_subj, "O": r_news_obj},
        "news", 3.0
    ),
    # 28. New: S proposed new O policy
    Template(
        ["{S}", "提出", "新的", "{O}", "政策", "。"],
        ["{S}", "proposed", "a", "new", "{O}", "policy"],
        {"S": r_news_subj, "O": r_news_obj},
        "news", 3.0
    ),
]

# ── LITERATURE Templates ─────────────────────────────────────────────────────

LIT_TEMPLATES = [
    # 1. Simple emotional: "S felt A"
    Template(
        ["{S}", "感到", "{A}", "。"],
        ["{S}", "felt", "{A}"],
        {"S": r_lit_noun, "A": r_lit_adj},
        "lit", 8.0
    ),
    # 2. Subject verbed object with emotion
    Template(
        ["{S}", "{V}", "{O}", "。"],
        ["{S}", "{V}", "{O}"],
        {"S": r_lit_noun, "V": r_lit_verb, "O": r_lit_noun},
        "lit", 10.0
    ),
    # 3. Time, S V O
    Template(
        ["{A}", "的", "{S}", "{V}", "{O}", "。"],
        ["The", "{A}", "{S}", "{V}", "{O}"],
        {"A": r_lit_adj, "S": r_lit_noun, "V": r_lit_verb, "O": r_lit_noun},
        "lit", 6.0
    ),
    # 4. Descriptive natural: "The A S V-ed under the O"
    Template(
        ["{A}", "的", "{S}", "在", "{O}", "下", "{V}", "。"],
        ["The", "{A}", "{S}", "{V}", "under", "the", "{O}"],
        {"A": r_lit_adj, "S": r_lit_noun, "O": r_lit_noun, "V": r_lit_verb},
        "lit", 5.0
    ),
    # 5. Sensory: "S heard/saw/felt O"
    Template(
        ["{S}", "{V}", "着", "{O}", "。"],
        ["{S}", "was", "{V}", "the", "{O}"],
        {"S": r_lit_noun, "V": r_lit_verb, "O": r_lit_noun},
        "lit", 5.0
    ),
    # 6. Memory: "S remembered the A O"
    Template(
        ["{S}", "想起了", "{A}", "的", "{O}", "。"],
        ["{S}", "remembered", "the", "{A}", "{O}"],
        {"S": r_lit_noun, "A": r_lit_adj, "O": r_lit_noun},
        "lit", 5.0
    ),
    # 7. Metaphor: "S is like A O"
    Template(
        ["{S}", "就像", "{A}", "的", "{O}", "。"],
        ["{S}", "is", "like", "a", "{A}", "{O}"],
        {"S": r_lit_noun, "A": r_lit_adj, "O": r_lit_noun},
        "lit", 4.0
    ),
    # 8. Narrative compound: "S1 V1 O1, and S2 V2 O2"
    Template(
        ["{S1}", "{V1}", "{O1}", "，", "而", "{S2}", "{V2}", "{O2}", "。"],
        ["{S1}", "{V1}", "{O1}", ",", "while", "{S2}", "{V2}", "{O2}"],
        {"S1": r_lit_noun, "V1": r_lit_verb, "O1": r_lit_noun,
         "S2": r_lit_noun, "V2": r_lit_verb, "O2": r_lit_noun},
        "lit", 4.0
    ),
    # 9. Emotive: "How A S V O!"
    Template(
        ["{S}", "多么", "{A}", "地", "{V}", "{O}", "！"],
        ["How", "{A}", "{S}", "{V}", "{O}", "!"],
        {"S": r_lit_noun, "A": r_lit_adj, "V": r_lit_verb, "O": r_lit_noun},
        "lit", 3.0
    ),
    # 10. Past reflection: "S once V-ed O"
    Template(
        ["{S}", "曾经", "{V}", "{O}", "。"],
        ["{S}", "once", "{V}", "{O}"],
        {"S": r_lit_noun, "V": r_lit_verb, "O": r_lit_noun},
        "lit", 5.0
    ),
    # 11. Nature scene: "A S shone over the O"
    Template(
        ["{A}", "的", "{S}", "照耀", "着", "{O}", "。"],
        ["The", "{A}", "{S}", "shone", "over", "the", "{O}"],
        {"A": r_lit_adj, "S": r_lit_noun, "O": r_lit_noun},
        "lit", 4.0
    ),
    # 12. Dialogue: "'O,' said S"
    Template(
        ["“", "{O}", "”", "，", "{S}", "说", "。"],
        ['"', "{O}", ',"', "{S}", "said"],
        {"O": r_lit_noun, "S": r_lit_noun},
        "lit", 3.0
    ),
    # 13. Inner monologue: "S thought about O"
    Template(
        ["{S}", "思考", "着", "{O}", "。"],
        ["{S}", "thought", "about", "{O}"],
        {"S": r_lit_noun, "O": r_lit_noun},
        "lit", 4.0
    ),
    # 14. Transition: "As time passed, S V-ed O"
    Template(
        ["随着", "时间", "流逝", "，", "{S}", "{V}", "{O}", "。"],
        ["As", "time", "passed", ",", "{S}", "{V}", "{O}"],
        {"S": r_lit_noun, "V": r_lit_verb, "O": r_lit_noun},
        "lit", 4.0
    ),
    # 15. Personification: "O smiled/sang/cried"
    Template(
        ["{O}", "{V}", "。"],
        ["The", "{O}", "{V}"],
        {"O": r_lit_noun, "V": r_lit_verb},
        "lit", 3.0
    ),
    # 16. Place description: "P was A and A"
    Template(
        ["{P}", "是", "{A1}", "而", "{A2}", "的", "。"],
        ["{P}", "was", "{A1}", "and", "{A2}"],
        {"P": r_lit_noun, "A1": r_lit_adj, "A2": r_lit_adj},
        "lit", 3.0
    ),
    # 17. Longing: "S longs for O"
    Template(
        ["{S}", "渴望", "{O}", "。"],
        ["{S}", "longs", "for", "{O}"],
        {"S": r_lit_noun, "O": r_lit_noun},
        "lit", 3.0
    ),
    # 18. Discovery: "S found O"
    Template(
        ["{S}", "发现", "了", "{O}", "。"],
        ["{S}", "found", "the", "{O}"],
        {"S": r_lit_noun, "O": r_lit_noun},
        "lit", 4.0
    ),
    # 19. Departure: "S left O behind"
    Template(
        ["{S}", "离开", "了", "{O}", "。"],
        ["{S}", "left", "the", "{O}", "behind"],
        {"S": r_lit_noun, "O": r_lit_noun},
        "lit", 4.0
    ),
    # 20. Antithesis: "S1 V1, but S2 V2"
    Template(
        ["{S1}", "{V1}", "，", "但", "{S2}", "{V2}", "。"],
        ["{S1}", "{V1}", ",", "but", "{S2}", "{V2}"],
        {"S1": r_lit_noun, "V1": r_lit_verb, "S2": r_lit_noun, "V2": r_lit_verb},
        "lit", 3.0
    ),
    # 21. A night scene: "Under the A moonlight, S V-ed"
    Template(
        ["在", "{A}", "的", "月光", "下", "，", "{S}", "{V}", "。"],
        ["Under", "the", "{A}", "moonlight", ",", "{S}", "{V}"],
        {"A": r_lit_adj, "S": r_lit_noun, "V": r_lit_verb},
        "lit", 3.0
    ),
    # 22. Whispers: "S whispered to O"
    Template(
        ["{S}", "对", "{O}", "轻声", "说", "。"],
        ["{S}", "whispered", "to", "{O}"],
        {"S": r_lit_noun, "O": r_lit_noun},
        "lit", 3.0
    ),
    # 23. Simile: "S is as A as O"
    Template(
        ["{S}", "像", "{O}", "一样", "{A}", "。"],
        ["{S}", "is", "as", "{A}", "as", "{O}"],
        {"S": r_lit_noun, "O": r_lit_noun, "A": r_lit_adj},
        "lit", 3.0
    ),
    # 24. Epiphany: "S suddenly realized O"
    Template(
        ["{S}", "突然", "意识", "到", "{O}", "。"],
        ["{S}", "suddenly", "realized", "{O}"],
        {"S": r_lit_noun, "O": r_lit_noun},
        "lit", 3.0
    ),
    # 25. Cycle of life: "O was born, lived, and died"
    Template(
        ["{O}", "诞生", "，", "活着", "，", "然后", "消逝", "。"],
        ["{O}", "was", "born", ",", "lived", ",", "and", "faded"],
        {"O": r_lit_noun},
        "lit", 2.0
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# Corpus Generator
# ══════════════════════════════════════════════════════════════════════════════

class CorpusGenerator:
    """Main corpus generation engine."""

    def __init__(self,
                 size: int = 10000,
                 seed: int = 42,
                 news_ratio: float = 0.6,
                 output_dir: str = "data/generated"):
        self.size = size
        self.seed = seed
        self.news_ratio = news_ratio
        self.output_dir = Path(output_dir)
        self.rng = random.Random(seed)

        # Template pools
        self.news_templates = NEWS_TEMPLATES
        self.lit_templates = LIT_TEMPLATES

        # Track generated pairs for dedup
        self.generated_zh: set = set()

        # Statistics
        self.stats = {
            "total_pairs": 0,
            "news_count": 0,
            "lit_count": 0,
            "train_count": 0,
            "test_count": 0,
            "zh_vocab": set(),
            "en_vocab": set(),
            "zh_word_counts": Counter(),
            "en_word_counts": Counter(),
            "zh_sentence_lengths": [],
            "en_sentence_lengths": [],
            "template_usage": Counter(),
        }

    def _pick_template(self, genre: str) -> Template:
        """Pick a template with weighted probability for the given genre."""
        if genre == "news":
            pool = self.news_templates
        else:
            pool = self.lit_templates
        total = sum(t.weight for t in pool)
        r = self.rng.uniform(0, total)
        cumulative = 0.0
        for t in pool:
            cumulative += t.weight
            if r <= cumulative:
                return t
        return pool[-1]

    def _generate_one(self, genre: str) -> Tuple[str, str]:
        """Generate a single sentence pair."""
        max_attempts = 50
        for _ in range(max_attempts):
            template = self._pick_template(genre)
            zh, en = template.generate(self.rng)

            # Skip empty or degenerate sentences
            if not zh or not en:
                continue
            if len(zh) < 3 or len(en.split()) < 2:
                continue

            # Deduplicate on Chinese (strict) — regenerate if seen
            if zh in self.generated_zh:
                continue

            self.generated_zh.add(zh)
            self.stats["template_usage"][template.zh_parts[0]] += 1
            return zh, en

        # Fallback: modify slightly
        zh, en = template.generate(self.rng)
        suffix = str(self.rng.randint(1, 9999))
        return zh + suffix, en + " " + suffix

    def generate(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Generate the full corpus. Returns (train_zh, train_en, test_zh, test_en)."""
        test_size = max(100, self.size // 100)  # ~1% for test
        train_size = self.size

        zh_pairs = []
        en_pairs = []

        # Generate training set
        news_target = int(train_size * self.news_ratio)
        lit_target = train_size - news_target

        print(f"Generating {train_size} sentence pairs...")
        print(f"  News (target):     {news_target}")
        print(f"  Literature (target): {lit_target}")

        news_gen = 0
        lit_gen = 0

        # Interleave news and lit generation for variety
        while news_gen < news_target or lit_gen < lit_target:
            if news_gen < news_target and (lit_gen >= lit_target or self.rng.random() < self.news_ratio):
                zh, en = self._generate_one("news")
                news_gen += 1
            else:
                zh, en = self._generate_one("lit")
                lit_gen += 1
            zh_pairs.append(zh)
            en_pairs.append(en)

        self.stats["news_count"] = news_gen
        self.stats["lit_count"] = lit_gen

        # Generate test set: half news, half lit, ensuring vocabulary overlap
        test_zh = []
        test_en = []
        test_news = test_size // 2
        test_lit = test_size - test_news
        for i in range(test_news):
            zh, en = self._generate_one("news")
            test_zh.append(zh)
            test_en.append(en)
        for i in range(test_lit):
            zh, en = self._generate_one("lit")
            test_zh.append(zh)
            test_en.append(en)

        # Compute statistics
        self._compute_stats(zh_pairs, en_pairs, test_zh, test_en)

        return zh_pairs, en_pairs, test_zh, test_en

    def _compute_stats(self, train_zh, train_en, test_zh, test_en):
        """Compute corpus statistics."""
        all_zh = train_zh + test_zh
        all_en = train_en + test_en

        for line in all_zh:
            # Chinese: count characters as rough word proxy (raw, no segmentation)
            chars = list(line.replace(" ", ""))
            self.stats["zh_word_counts"].update(chars)
            self.stats["zh_vocab"].update(chars)
            self.stats["zh_sentence_lengths"].append(len(chars))

        for line in all_en:
            words = line.split()
            self.stats["en_word_counts"].update(words)
            self.stats["en_vocab"].update(words)
            self.stats["en_sentence_lengths"].append(len(words))

        self.stats["total_pairs"] = len(train_zh) + len(test_zh)
        self.stats["train_count"] = len(train_zh)
        self.stats["test_count"] = len(test_zh)

    def write_output(self, train_zh, train_en, test_zh, test_en):
        """Write corpus files and statistics."""
        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write sentence files
        files = {
            "train.zh": train_zh,
            "train.en": train_en,
            "test.zh": test_zh,
            "test.en": test_en,
        }

        for fname, lines in files.items():
            fpath = out_dir / fname
            fpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"  Wrote {len(lines)} lines → {fpath}")

        # Write statistics
        stats_data = {
            "total_pairs": self.stats["total_pairs"],
            "train_pairs": self.stats["train_count"],
            "test_pairs": self.stats["test_count"],
            "news_count": self.stats["news_count"],
            "lit_count": self.stats["lit_count"],
            "zh_vocab_size": len(self.stats["zh_vocab"]),
            "en_vocab_size": len(self.stats["en_vocab"]),
            "zh_vocab_size_unique_chars": len(self.stats["zh_vocab"]),
            "en_vocab_size_unique_words": len(self.stats["en_vocab"]),
            "zh_avg_sent_len": (sum(self.stats["zh_sentence_lengths"]) /
                                max(1, len(self.stats["zh_sentence_lengths"]))),
            "en_avg_sent_len": (sum(self.stats["en_sentence_lengths"]) /
                                max(1, len(self.stats["en_sentence_lengths"]))),
            "zh_min_sent_len": min(self.stats["zh_sentence_lengths"]) if self.stats["zh_sentence_lengths"] else 0,
            "zh_max_sent_len": max(self.stats["zh_sentence_lengths"]) if self.stats["zh_sentence_lengths"] else 0,
            "en_min_sent_len": min(self.stats["en_sentence_lengths"]) if self.stats["en_sentence_lengths"] else 0,
            "en_max_sent_len": max(self.stats["en_sentence_lengths"]) if self.stats["en_sentence_lengths"] else 0,
            "zh_top_words": self.stats["zh_word_counts"].most_common(50),
            "en_top_words": self.stats["en_word_counts"].most_common(50),
            "news_ratio": self.news_ratio,
            "seed": self.seed,
            "generated_at": str(self.stats.get("_timestamp", "")),
        }
        stats_path = out_dir / "corpus_stats.json"
        stats_path.write_text(
            json.dumps(stats_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"  Wrote statistics → {stats_path}")

        return stats_data


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Chinese-English parallel corpus for SMT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --size 10000 --seed 42
  %(prog)s --size 5000 --news-ratio 0.5 --output-dir my_corpus
  %(prog)s --size 20000 --seed 123 --output-dir data/large_corpus
        """
    )
    parser.add_argument("--size", type=int, default=10000,
                        help="Number of training sentence pairs (default: 10000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output-dir", type=str, default="data/generated",
                        help="Output directory (default: data/generated)")
    parser.add_argument("--news-ratio", type=float, default=0.6,
                        help="Ratio of news sentences (default: 0.6)")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of sample pairs to print (default: 5)")
    args = parser.parse_args()

    # Resolve output dir relative to script location or cwd
    script_dir = Path(__file__).resolve().parent
    smt_root = script_dir.parent  # smt_model/

    if not Path(args.output_dir).is_absolute():
        output_dir = smt_root / args.output_dir
    else:
        output_dir = Path(args.output_dir)

    print("=" * 70)
    print("  Synthetic Chinese-English Parallel Corpus Generator")
    print("=" * 70)
    print(f"  Training pairs:  {args.size}")
    print(f"  Test pairs:      {max(100, args.size // 100)}")
    print(f"  News ratio:      {args.news_ratio}")
    print(f"  Seed:            {args.seed}")
    print(f"  Output dir:      {output_dir}")
    print(f"  {'─' * 60}")

    generator = CorpusGenerator(
        size=args.size,
        seed=args.seed,
        news_ratio=args.news_ratio,
        output_dir=str(output_dir),
    )

    train_zh, train_en, test_zh, test_en = generator.generate()
    stats = generator.write_output(train_zh, train_en, test_zh, test_en)

    print(f"\n  {'─' * 60}")
    print(f"  Corpus Statistics:")
    print(f"  {'─' * 60}")
    print(f"  Total pairs:     {stats['total_pairs']}")
    print(f"  Training pairs:  {stats['train_pairs']}")
    print(f"  Test pairs:      {stats['test_pairs']}")
    print(f"  News sentences:  {stats['news_count']}")
    print(f"  Literature:      {stats['lit_count']}")
    print(f"  ZH unique chars: {stats['zh_vocab_size']}")
    print(f"  EN unique words: {stats['en_vocab_size']}")
    print(f"  ZH avg length:   {stats['zh_avg_sent_len']:.1f} chars")
    print(f"  EN avg length:   {stats['en_avg_sent_len']:.1f} words")
    print(f"  ZH length range: {stats['zh_min_sent_len']}–{stats['zh_max_sent_len']}")
    print(f"  EN length range: {stats['en_min_sent_len']}–{stats['en_max_sent_len']}")

    # Print sample pairs
    print(f"\n  {'─' * 60}")
    print(f"  Sample Sentence Pairs:")
    print(f"  {'─' * 60}")
    indices = generator.rng.sample(
        range(min(len(train_zh), len(train_en))),
        min(args.samples, len(train_zh))
    )
    for idx in indices:
        print(f"\n  [{idx}] ZH: {train_zh[idx]}")
        print(f"  [{idx}] EN: {train_en[idx]}")

    print(f"\n  {'─' * 60}")
    print(f"  ✅ Corpus generation complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Files: train.zh, train.en, test.zh, test.en, corpus_stats.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
