# 跨架构机器翻译文本统计比较 — 论文资料汇集

> **实验协议**: 统计课_跨架构文本统计比较实验协议.md (2026-06-03, v1.0)  
> **状态**: 方法学 + 文献对比 完成 | 实验待执行 | 结果待补充  
> **最后更新**: 2026-06-07

---

## 目录结构

| 文件 | 内容 | 行数 | 大小 |
|:-----|:-----|:----:|:----:|
| `01_methodology.md` | SMT 系统方法论: 架构、模型演进(5轮)、7个Bug修复、效率发现、关键设计决策、协议合规评估 | 546 | 39.9KB |
| `02_literature_comparison.md` | 文献对比: Moses/cdec/Joshua/Phrasal/NiuTrans 系统比较、中文SMT专项、跨架构文本对比文献、研究定位、BibTeX引用 | 265 | 32.6KB |
| `03_experiment_design.md` | 实验设计: 研究问题、2×2×2因子设计、功效分析、源文本语料库、翻译系统规格、特征提取流水线、统计分析计划、预注册、时间线 | 411 | 24.8KB |

**合计**: 1,222 行 | 104KB

---

## 论文各节映射 (IMRaD)

| 论文章节 | 对应文件 | 完成度 |
|:---------|:---------|:------:|
| **引言** (Introduction) | `03_experiment_design.md` §3.1 + `02_literature_comparison.md` §2.3 | ✅ |
| **方法** (Methods) — SMT系统 | `01_methodology.md` | ✅ |
| **方法** (Methods) — 实验设计 | `03_experiment_design.md` | ✅ |
| **方法** (Methods) — 文献定位 | `02_literature_comparison.md` | ✅ |
| **结果** (Results) | ⏳ 待实验执行后补充 | ❌ |
| **讨论** (Discussion) | ⏳ 待结果 | ❌ |
| **参考文献** | `02_literature_comparison.md` §2.5 | ✅ (15+条目) |

---

## 核心发现摘要

### SMT 系统构建
- **5轮迭代**: 合成模板(BLEU=0) → IBM2单向(乱码) → sym gdfa(⭐⭐⭐) → 213K(数据非瓶颈) → fast_align(短语7.6×但质量未提升)
- **7个Bug修复**: B1(词法下溢) → B7(中文清洗)
- **最佳模型**: `smt_zh2en_sym` 8,705短语 + 3-gram LM + MERT网格搜索
- **核心发现**: IBM2+gdfa的8.7K高质量短语 > fast_align的65.9K噪声短语

### 实验设计
- **2×2×2完全析因**: 架构(SMT/LLM) × 方向(ZH↔EN) × 体裁(新闻/文学)
- **160篇分析单元**: 80源文本 × 2翻译架构
- **4个维度**: 词汇丰富度(STTR/MTLD/HD-D) + 句长分布(KS检验) + 情感(XLM-RoBERTa) + 风格计量学(POS熵)
- **功效**: d=0.8, α=0.05, power=0.80, n≥21/组 ✅

### 文献定位
- 与Moses/cdec/Joshua/Phrasal/NiuTrans完整对比
- 与Zhu 2024, Gude 2025, Reinhart 2025等跨架构文本研究对标
- 创新点: 从零构建完整SMT管道、受控2×2×2因子设计、纯SMT数据(无LLM污染)

---

## 源文件索引 (项目根目录)

| 文件 | 用途 |
|:-----|:----|
| `统计课_跨架构文本统计比较实验协议.md` | 实验协议定稿 (Oracle审核) |
| `context_session.md` | 项目上下文 (模型演进、服务器、Bug) |
| `FINAL_REPORT.md` | SMT实施最终报告 (模型清单、Bug详情、部署) |
| `extension_roadmap.md` | 扩展路线图 (ROI分析、优先级) |
| `research_findings.md` | 外部调研 (Moses/cdec/Joshua等文献) |
| `critical_review.md` | 批判审查 (协议合规、Bug、就绪评估) |
| `progress.md` | 进度日志 |

---

## 下一步

1. [ ] 完成特征提取流水线实现 (§4 of protocol)
2. [ ] 执行SMT批量翻译 (80源文本 × 2方向)
3. [ ] 执行LLM批量翻译 (80源文本 × 2方向)
4. [ ] 运行统计分析 (4项假设检验 + SVM分类)
5. [ ] 撰写结果与讨论章节
6. [ ] 预注册 (OSF/AsPredicted)
