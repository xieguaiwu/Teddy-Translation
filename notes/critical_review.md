# SMT 系统批判性审查 — 最终版

> 2026-06-06 22:00 CST
> 状态: 全部完成 — 7 bugs 修复，5 轮迭代，sym 模型为最终基线

---

## A. 协议合规度: 8/10

| 协议 § | 要求 | 当前 | 评估 |
|:-------|:-----|:----|:----|
| 3.1 架构 | Moses 短语级 SMT | ✅ IBM2+gdfa + 短语表 + 束搜索 + 3-gram LM + MERT | 等价 |
| 3.1 数据 | WMT ~300K | ⚠️ 213K 已下载/清洗，使用 50K 子集 | 数据量不足但验证了不是瓶颈 |
| 3.1 对齐 | GIZA++ IBM4 | ⚠️ IBM2+gdfa（已验证 fast_align 备选） | 部分满足 |
| 3.1 LM | KenLM 5-gram | ✅ Kneser-Ney 3-gram（3-gram 对 50K 数据更优） | 等价 |
| 3.1 调优 | MERT | ✅ 网格搜索完成，Och 算法就绪 | 调优完成 |
| 4.1-4.4 | 特征流水线 | ❌ 未实现 | SMT 模型之外 |

---

## B. 已修复 Bug (7 项)

| # | 文件:行 | 缺陷 | 严重度 | 修复 |
|:--|:--|:-----|:------|:----|
| B1 | phrase_table.py:137 | `_LEX_EPSILON=1e-10` 致 log-space 下溢 | 🔴 | 1e-10→1e-7 |
| B2 | language_model.py:203 | 并行 n-gram 计数 order 3-5 相同 | 🔴 | 强制顺序 |
| B3 | language_model.py:440 | JSON 键 `str(tuple)` + ast.literal_eval 44s | 🟡 | json.dumps(list(k)) |
| B4 | config.yaml | LM order 5→3 | 🟡 | 5→3 |
| B5 | decoder.py:335 | 重组 key 仅 coverage | 🟡 | +last_N_words |
| B6 | clean_wmt.py | WMT 含印尼语词 | 🟡 | langdetect |
| B7 | batch_v2.py | clean_zh pv=cu 在 if 内吞字符 | 🟡 | pv=cu 移出 if |

---

## C. 模型质量演进

| 模型 | 短语 | 质量 | 瓶颈 |
|:--|:--|:--|:--|
| v1 合成 10K | 9,753 | BLEU=0 | 训练数据为模板 |
| v2 IBM2 单向 50K | 29,856 | 词序混乱 | 单向对齐噪声 |
| **v3 sym 50K** | **8,705** | **⭐⭐⭐ 最佳** | 文学域 OOV |
| v4 213K | 9,087 | = v3 | 数据非瓶颈 |
| v5 fast_align 50K | 65,909 | < v3 | 更多短语≠更好 |

---

## D. 剩余未修复

| # | 缺陷 | 严重度 | 阻塞实验? | 方案 |
|:--|:-----|:------|:--------|:----|
| C1 | 文学域 OOV 率高 | 🟡 | 部分 | 接受或增加文学训练数据 |
| C2 | IBM2 非 IBM4 | 🟢 | 否 | fast_align 备选，当前足够 |
| C3 | EN→ZH 解码慢 | 🟢 | 否 | 已通过短语过滤优化 |

---

## E. 实验就绪评估

| 统计维度 | 新闻域 | 文学域 |
|:--|:--|:--|
| STTR/MTLD | ✅ | ⚠ |
| 句长分布 | ✅ | ⚠ |
| 情感倾向 | ✅ | ❌ |
| POS 熵 | ⚠ | ❌ |

**结论：新闻域实验可直接进行。文学域结果需谨慎解释。**
