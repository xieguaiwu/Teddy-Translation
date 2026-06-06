# Context Session — SMT 传统机器学习翻译模型

> 最后更新: 2026-06-06 22:00 CST
> 项目: 跨架构机器翻译文本统计比较实验协议
> 状态: ✅ 全部完成 — sym 模型 + MERT 调优 + fast_align 就绪

---

## 一、项目概述

构建传统短语级 SMT 作为实验对照组基线，与 LLM 输出进行统计特征对比。
维度：词汇丰富度 / 句长分布 / 情感倾向 / 风格计量学。

---

## 二、最终交付模型

| 模型 | 路径 | 短语 | LM | 对齐 | MERT |
|:----|:----|:----|:---|:----|:----|
| ZH→EN sym | `model/smt_zh2en_sym/` | 8,705 | 3-gram 75MB | IBM2+gdfa | lm=0.5,trans=0.5,wp=-0.5 |
| EN→ZH sym | `model/smt_en2zh_sym/` | 8,729 | 3-gram 118MB | IBM2+gdfa | 默认权重 |

训练数据：WMT news-commentary v12，50K 句（从 213K 清洗后子集），jieba 中文分词 + Moses 英文分词。

### 翻译质量基线

```
ZH→EN News (⭐⭐⭐):
  源: 美国总统今日宣布了一项新的经济政策，旨在促进就业增长。
  译: President announced New economy policy , aimed promoting employment growth . has

EN→ZH News (⭐⭐⭐):
  源: The president announced a major infrastructure plan to boost the economy.
  译: 。提振经济基础设施计划向一个重大该总统宣布

ZH→EN Lit (⭐):    文学词汇 OOV，输出碎片化
EN→ZH Lit (⭐⭐):  碎片化但非空
```

---

## 三、模型演进历史（5 轮迭代）

| 版本 | 数据 | 短语 | 对齐 | 结果 |
|:-----|:-----|:-----|:-----|:-----|
| v1 (smt_zh2en) | 10K 合成模板 | 9,753 | 单向 IBM2 | BLEU=0, 87% OOV |
| v2 (v3) | 50K WMT | 29,856 | 单向 IBM2 | 词序混乱，印尼语残留 |
| **v3 (sym)** | **50K WMT** | **8,705** | **IBM2+gdfa** | **当前最佳，半可理解** |
| v4 (213k) | 213K WMT | 9,087 | IBM2+gdfa | +5% 短语，质量不变 → 数据不是瓶颈 |
| v5 (fa) | 50K WMT | 65,909 | fast_align HMM | 短语 7.5× 但噪声大 → 更多≠更好 |
| v5b (fa_v2) | 50K WMT | 65,913 | fast_align+IBM2词法 | 与 v5 相同质量 |

**核心发现：IBM2+gdfa 的 8.7K 高质量短语 > fast_align 的 65.9K 噪声短语**

---

## 四、服务器

| 服务器 | 地址 | 密码 | 状态 |
|:------|:-----|:----|:-----|
| **S1 (24212)** | `root@223.109.239.36:24212` | `ohyaegh9` | ✅ 在线，fast_align 已编译 |

其他服务器 (24520, 20248, 33748) 当前不可达。

---

## 五、6 个已修复 Bug

| # | 文件:行 | 缺陷 | 修复 |
|:--|:--|:-----|:----|
| B1 | phrase_table.py:137 | `_LEX_EPSILON=1e-10` 致 log-space 下溢 | → 1e-7 |
| B2 | language_model.py:203 | 并行 n-gram 计数 order 3-5 相同 | 强制顺序 |
| B3 | language_model.py:440+492 | JSON 键 `str(tuple)` + ast.literal_eval 44s | → `json.dumps(list(k))` |
| B4 | config.yaml + retrain_v3.py | LM order 5 致 300MB | → 3 |
| B5 | decoder.py:335 | 重组 key 仅 coverage | → (coverage, last_N_words) |
| B6 | clean_wmt.py | WMT 数据含印尼语词 | langdetect 过滤 |
| B7 | batch_v2.py | clean_zh pv=cu 在 if 内吞掉字符 | pv=cu 移到 if 外 |

---

## 六、关键技术决策

| 决策 | 原因 |
|:----|:----|
| grow-diag-final-and | 单向 IBM2 太弱，对称化减少 70% 短语但质量提升 |
| 3-gram LM | 5-gram 在 50K 句上过拟合（300MB），3-gram 足够 |
| prune_threshold=0 | 剪枝需 O(n_grams) 概率计算，213K 句时不可行 |
| 不实现 IBM3/4 | fast_align C++ 是更优选择（已编译就绪），从零实现需 20h |
| 不实现 MERT Och 算法 | F1 仅 0.31→0.34，当前模型质量下权重调优收益极低 |
| fast_align 短语 > IBM2 但质量更差 | 词法加权依赖 IBM2 t_table，无法从 fast_align 提取 |

---

## 七、翻译输出

```
output/smt_sym_v2/       ← 最终版 (sym+MERT, 80篇, 全部非空)
├── zh_news/  (20 ZH→EN)  ⭐⭐⭐ 新闻域可用
├── zh_lit/   (20 ZH→EN)  ⭐    文学域 OOV
├── en_news/  (20 EN→ZH)  ⭐⭐⭐ 新闻域可用
└── en_lit/   (20 EN→ZH)  ⭐⭐  碎片化

output/smt_sym/           ← 旧版 (sym, 80篇, 含空输出 bug)
output/smt_fa/            ← fast_align 版
```

---

## 八、已实现但未使用的组件

| 组件 | 文件 | 状态 |
|:--|:--|:--|
| n-best 解码 | decoder.py `decode_nbest()` | ✅ 就绪，MERT 用 |
| Och MERT | scripts/mert_tune.py | ✅ 就绪，F1 提升仅 0.03 |
| fast_align C++ | /tmp/fast_align/build/ | ✅ 编译完成 |
| fast_align Python 包装 | smt/align_fast.py | ✅ 就绪 |
| fast_align 训练管道 | scripts/train_fastalign.py | ✅ 就绪（v2 含 IBM2 词法） |
| 213K 全量训练 | scripts/train_symmetrized.py | ✅ 已验证（质量不变） |
| BOOKS 单语提取 | scripts/extract_mono.py | ✅ 505K ZH + 253K EN 已提取 |
| LM 增强 | scripts/enhance_lm.py | ✅ 就绪 |
| 网格搜索 | scripts/grid_search_decoder.py | ✅ 就绪（BLEU=0 无意义） |

---

## 九、实验就绪评估

| 统计维度 | 新闻域 | 文学域 |
|:--|:--|:--|
| 词汇丰富度 (STTR/MTLD) | ✅ 可用 | ⚠ 稀疏 |
| 句长分布 | ✅ 可用（SMT 显著短于 LLM） | ⚠ 部分可用 |
| 情感倾向 | ✅ 可用 | ❌ 噪声大 |
| POS 熵 | ⚠ 词序混乱 | ❌ 不可靠 |

**Go/No-Go: 新闻域实验可以继续。文学域需要更多领域内训练数据或接受高噪声。**

---

## 十、关键文件索引

| 文件 | 路径 | 用途 |
|:----|:----|:----|
| 本文件 | `context_session.md` | AI 助手上下文 |
| 批判审查 | `critical_review.md` | Bug 历史 + 协议合规 |
| 扩展路线图 | `prometheus_roadmap.md` | 未来方向 |
| ZH→EN sym | `model/smt_zh2en_sym/` | 最佳 SMT 模型 |
| EN→ZH sym | `model/smt_en2zh_sym/` | 最佳 SMT 模型 |
| 翻译输出 | `output/smt_sym_v2/` | 80 篇最终译文 |
| 对称训练 | `scripts/train_symmetrized.py` | 可复现训练 |
| fast_align 训练 | `scripts/train_fastalign.py` | 高级对齐训练 |
| MERT 调优 | `scripts/mert_tune.py` | Och 算法 |
| 后处理 | `smt/postprocess.py` | 去除非目标语言 |
| 批量翻译 v2 | `scripts/batch_v2.py` | 修复 clean_zh 的版本 |
| 解码器 n-best | `smt/decoder.py` (decode_nbest) | MERT 用 |
| fast_align 包装 | `smt/align_fast.py` | Python 接口 |
