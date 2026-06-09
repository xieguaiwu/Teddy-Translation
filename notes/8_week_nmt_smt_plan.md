# Teddy NMT+SMT — 8 周综合推进计划

> **Author:** Prometheus (Strategic Planner / Architect)  
> **Date:** 2026-06-07  
> **Target:** 从当前 SMT BLEU≈8 + NMT epoch 4/20 (loss=5.35) 推进到互补系统组合 BLEU 22–28，产生论文级成果  
> **Hardware:** 2× V100 32GB (服务器 223.109.239.36)  
> **Data:** 227K WMT ZH-EN paired (news-commentary v12) + ~200K English monolingual + ~300K Chinese monolingual

---

## 总体架构愿景

```
                    ┌─────────────────────────────────┐
                    │     227K WMT Parallel Data       │
                    │     (news-commentary v12)         │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
     ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
     │   SMT Pipeline │  │  NMT ZH→EN     │  │  NMT EN→ZH     │
     │  (fast_align)  │  │  Transformer    │  │  Transformer    │
     │  BLEU ~8       │  │  Base 93M       │  │  Base 93M       │
     └───────┬────────┘  └───────┬────────┘  └───────┬────────┘
             │                   │                    │
             │                   ▼                    │
             │          ┌────────────────┐            │
             │          │  Back-Trans    │◄───────────┘
             │          │  +200K syn     │
             │          └───────┬────────┘
             │                   │
             ▼                   ▼
     ┌────────────────┐  ┌────────────────┐
     │  SMT Retrained │  │  NMT Retrained │
     │  BLEU ~10-12   │  │  BLEU ~18-23   │
     └───────┬────────┘  └───────┬────────┘
             │                   │
             │    ┌──────────────┤
             │    │  Ensemble    │
             │    │  (3 models)  │
             │    │  BLEU 21-26  │
             │    └──────┬───────┘
             │           │
             └─────┬─────┘
                   ▼
          ┌────────────────┐
          │ System Combine │
          │ Confusion Net  │
          │ BLEU 22-28     │
          └───────┬────────┘
                  │
                  ▼
          ┌────────────────┐
          │ Domain Adapt   │
          │ (news/literary)│
          │ BLEU 23-28     │
          └────────────────┘
```

---

## 预期 BLEU 里程碑

| 里程碑 | 周 | SMT BLEU | NMT BLEU | Combined BLEU | 关键动作 |
|:-------|:--:|:--------:|:--------:|:-------------:|:---------|
| **M0: Baseline** | W0 | 8.0 | — | — | 当前 fast_align SMT |
| **M1: NMT Base** | W2 | 8.0 | **15–18** | — | 完成 20 epoch 训练 |
| **M2: +BackTrans** | W4 | **10–12** | **18–22** | — | 回译数据增强 |
| **M3: +Ensemble** | W6 | 10–12 | **21–25** | **23–27** | 3 模型集成 + 系统组合 |
| **M4: +Domain** | W8 | 10–12 | 21–25 | **24–28** | 领域自适应 + 最终评测 |

**校准依据:** 227K ZH→EN 新闻域句对 + ~200K 合成回译数据 + 3 模型集成。在 WMT news-commentary 测试集上，类似规模系统 (Sennrich et al. 2016; Edunov et al. 2018) 报告 BLEU 20–28。

> ⚠ **注意:** BLEU 绝对值依赖于测试集选择。本计划中所有 BLEU 均为 sacrebleu 签名 (`nrefs:1|bs:1000|case:mixed|eff:no|tok:intl|smooth:exp|version:2.x`) 在 WMT 新闻域 held-out 测试集上的值。

---

## Week 1–2: NMT 训练完成 + 初始对比

**目标:** 完成 NMT 20 epoch 训练，评估 BLEU 并与 SMT 进行全面对比分析

### 现状诊断

```
NMT: Transformer Base 93M, epoch 4/20, loss=5.35
     ↓ 预计收敛
     epoch 20, loss≈3.5–4.0, BLEU≈15–18
```

loss=5.35 在 epoch 4 是健康信号：随机基线 ln(32000)≈10.4，已显著学习。剩余 16 epoch 预计将 loss 降至 3.5–4.0 区间。

### W1 Day 1–2: 环境搭建与数据准备

#### Step 1: 安装 NMT 框架

```bash
# 在服务器 223.109.239.36 上执行
pip install OpenNMT-py==3.5.0 sentencepiece sacrebleu

# 验证 GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); \
    [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
# 期望输出: GPU 0: Tesla V100-SXM2-32GB, GPU 1: Tesla V100-SXM2-32GB
```

#### Step 2: 数据划分 (严格防止泄漏)

```bash
cd /path/to/Teddy

# 当前数据: 227,330 行平行句对
# 划分: train 200K (88%) / valid 15K (6.6%) / test 12,330 (5.4%)

mkdir -p data/nmt

# Shuffle + split (固定种子保证可复现)
python3 -c "
import random
random.seed(42)
with open('data/wmt/train.zh') as fz, open('data/wmt/train.en') as fe:
    pairs = list(zip([l.strip() for l in fz], [l.strip() for l in fe]))
random.shuffle(pairs)
n = len(pairs)
train, valid, test = pairs[:200000], pairs[200000:215000], pairs[215000:]
for name, data in [('train', train), ('valid', valid), ('test', test)]:
    with open(f'data/nmt/{name}.zh', 'w') as fz, open(f'data/nmt/{name}.en', 'w') as fe:
        for zh, en in data:
            fz.write(zh + '\n'); fe.write(en + '\n')
print(f'Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}')
"
# 输出: Train: 200000, Valid: 15000, Test: 12330
```

#### Step 3: BPE 训练 (32K 联合词表)

```bash
# 在合并的中英文上训练 SentencePiece BPE
cat data/nmt/train.zh data/nmt/train.en > /tmp/bpe_corpus.txt

spm_train \
  --input=/tmp/bpe_corpus.txt \
  --model_prefix=data/nmt/bpe_32k \
  --vocab_size=32000 \
  --character_coverage=0.9995 \
  --model_type=bpe \
  --num_threads=8 \
  --split_digits=true \
  --byte_fallback=true \
  --max_sentence_length=8192

# 对全部数据应用 BPE
for split in train valid test; do
  for lang in zh en; do
    spm_encode --model=data/nmt/bpe_32k.model \
      < data/nmt/${split}.${lang} \
      > data/nmt/${split}.bpe.${lang}
  done
done

# 可选: 获取共享词表文件
spm_export_vocab --model=data/nmt/bpe_32k.model > data/nmt/vocab.txt
```

**BPE 超参数说明:**
- `vocab_size=32000`: 标准选择，平衡覆盖率与模型大小。中文 ~8K 字符 + BPE 子词，英文形态切分
- `character_coverage=0.9995`: 涵盖 99.95% 字符，罕见汉字用 byte fallback
- `byte_fallback=true`: 未登录字符用 UTF-8 字节序列表示，彻底消除 UNK
- `split_digits=true`: 数字分离，提升数值翻译泛化

### W1 Day 3–4: 预处理 + 启动训练

#### Step 4: OpenNMT 预处理

```bash
# 构建词汇表并序列化数据为二进制格式
onmt_preprocess \
  -train_src data/nmt/train.bpe.zh \
  -train_tgt data/nmt/train.bpe.en \
  -valid_src data/nmt/valid.bpe.zh \
  -valid_tgt data/nmt/valid.bpe.en \
  -save_data data/nmt/processed \
  -src_vocab_size 32000 \
  -tgt_vocab_size 32000 \
  -share_vocab \
  -src_seq_length 150 \
  -tgt_seq_length 150 \
  -overwrite
```

#### Step 5: 训练配置 (93M Transformer Base)

```yaml
# config/nmt_zh2en_base.yaml
save_data: data/nmt/processed
save_model: models/nmt/zh2en_base

# ── 架构 (Transformer Base — 93M 参数) ──
encoder_type: transformer
decoder_type: transformer
layers: 6
heads: 8
hidden_size: 512          # d_model
word_vec_size: 512
transformer_ff: 2048       # d_ff
position_encoding: true

# ── 训练 ──
batch_size: 4096           # tokens per GPU
batch_type: tokens
normalization: tokens
accum_count: 4             # 有效 batch = 4096 × 4 × 2 GPU = 32,768 tokens
max_generator_batches: 2
train_steps: 200000        # ~20 epochs (200K sentences × 20 / 32K tokens/batch ≈ 125K steps)
valid_steps: 2000
save_checkpoint_steps: 2000
keep_checkpoint: 5

# ── 优化器 ──
optim: adam
adam_beta2: 0.998
learning_rate: 2.0
warmup_steps: 8000
decay_method: noam
max_grad_norm: 0.0          # 不裁剪梯度 (Noam 已包含隐式正则化)

# ── 正则化 (针对小数据增强) ──
dropout: 0.2               # 略高于标准 0.1，因数据量小
attention_dropout: 0.1
label_smoothing: 0.1
param_init: 0.0
param_init_glorot: true

# ── 分布式 ──
world_size: 2
gpu_ranks: [0, 1]

# ── 日志 ──
tensorboard: true
tensorboard_log_dir: models/nmt/tensorboard
report_every: 200
```

**为什么选这批超参数:**
- **dropout=0.2**: 227K 句对是 NMT 的小数据规模。标准 dropout=0.1 在小数据上容易过拟合 (valid loss 在 epoch 12-15 后回升)。0.2 提供更强的正则化。
- **accum_count=4 + batch_size=4096**: V100 32GB 单卡可容纳 ~4K tokens，双卡累积后有效 batch=32K，接近 "token-based batch" 最佳实践的 25K-50K 范围。
- **warmup_steps=8000**: 标准设置，前 8000 步从 lr=0 线性增长到 peak。对于 200K 总步数的 ~4% —— 在 Noam 建议的 1-5% 范围内。
- **train_steps=200000**: 200K sentences × 20 epochs / 32K tok/batch ≈ 125K steps。设 200K 留余量确保充分训练。
- **label_smoothing=0.1**: 标准设置，防止模型对训练目标过度自信 (提升泛化 ~0.5-1 BLEU)。

#### Step 6: 如果已有 checkpoint，从 epoch 4 恢复

```bash
# 情况 A: 有 OpenNMT checkpoint → 直接恢复
onmt_train -config config/nmt_zh2en_base.yaml \
  -train_from models/nmt/zh2en_base_step_XXXXX.pt

# 情况 B: 有原始 PyTorch checkpoint → 转换后恢复
# (如果有其他格式的 ckpt，需要先转换)
ls -la models/nmt/*.pt  # 检查现有 checkpoint 位置
```

**如果在服务器上已有训练进程在运行:**
1. 检查当前进程: `nvidia-smi` + `ps aux | grep onmt_train`
2. 对照上述 YAML 检查超参数是否一致
3. 如不一致 → 决定是继续当前训练还是用优化超参数重启
4. **决策原则:** 如果当前 lr/warmup/dropout/batch_size 差异在 ±20% 以内，继续训练不中断；如果差异大 (>50%)，从 epoch 4 恢复但用新配置

### W1 Day 5–7 + W2 Day 1–2: 训练监控

#### 监控命令

```bash
# 实时查看训练日志
tail -f models/nmt/zh2en_base.log

# TensorBoard
tensorboard --logdir=models/nmt/tensorboard --port=6006 --bind_all

# GPU 监控
watch -n 1 nvidia-smi
```

#### 健康训练指标

| 指标 | Epoch 4 | Epoch 10 | Epoch 15 | Epoch 20 |
|:-----|:-------:|:--------:|:--------:|:--------:|
| Train loss | ~5.35 | ~4.2 | ~3.7 | ~3.5 |
| Valid loss | ~5.5 | ~4.5 | ~4.1 | ~4.0 |
| Valid PPL | ~245 | ~90 | ~60 | ~55 |
| BLEU (dev) | ~5 | ~10 | ~14 | ~16 |
| Grad norm | <1.0 | <1.0 | <1.0 | <0.5 |

**异常信号与应对:**

| 异常 | 应对 |
|:-----|:-----|
| Valid loss 上升而 train loss 持续下降 (>3 checkpoints) | Early stop，降低学习率 (×0.5) 继续 |
| BLEU 停滞 (>5 checkpoints) | 检查 label smoothing、增加 dropout 到 0.3 |
| GPU 显存不足 (OOM) | 降低 batch_size 到 2048，增加 accum_count 到 8 |
| 训练速度 <2 steps/s | 减少 log frequency，确认数据在 SSD 上 |

### W2 Day 3–5: BLEU 评估

#### Step 7: 测试集评估

```bash
# 用最优 checkpoint (按 valid loss 或 BLEU) 翻译测试集
onmt_translate \
  -model models/nmt/zh2en_base_best.pt \
  -src data/nmt/test.bpe.zh \
  -output data/nmt/test.hyp.bpe.en \
  -batch_size 4096 \
  -beam_size 5 \
  -length_penalty avg \
  -alpha 1.0 \
  -gpu 0 \
  -replace_unk

# 去 BPE
spm_decode --model=data/nmt/bpe_32k.model \
  < data/nmt/test.hyp.bpe.en \
  > data/nmt/test.hyp.en

# BLEU 评估
sacrebleu data/nmt/test.en \
  -i data/nmt/test.hyp.en \
  -m bleu chrf ter \
  --confidence \
  > results/nmt_base_eval.txt

cat results/nmt_base_eval.txt
```

**多 beam size 对比:**

```bash
for beam in 3 5 8 10; do
  onmt_translate -model models/nmt/zh2en_base_best.pt \
    -src data/nmt/test.bpe.zh -output data/nmt/test.hyp.b${beam}.en \
    -beam_size $beam -length_penalty avg -alpha 1.0 -gpu 0 -replace_unk
  spm_decode --model=data/nmt/bpe_32k.model < data/nmt/test.hyp.b${beam}.en > /tmp/hyp.en
  echo -n "beam=$beam: "; sacrebleu data/nmt/test.en -i /tmp/hyp.en -m bleu --short
done
```

预期: beam=5 接近最优；beam=8/10 可能略低 (小模型大 beam 引入噪声)。

### W2 Day 6–7: SMT vs NMT 对比分析

#### Step 8: 双系统全面评估

```bash
# ── SMT 评估 (使用当前最佳 fast_align 模型) ──
python3 scripts/eval_bleu.py \
  --model-dir models/zh2en_fa \
  --direction zh2en \
  --test-src data/nmt/test.zh \
  --test-ref data/nmt/test.en \
  --beam-size 5 \
  --tokenize-bleu intl \
  --output-dir results/smt_baseline

# 同样评估 sym 模型和 213k 模型
for model in zh2en_sym zh2en_213k_fa; do
  python3 scripts/eval_bleu.py \
    --model-dir models/$model \
    --direction zh2en \
    --test-src data/nmt/test.zh \
    --test-ref data/nmt/test.en \
    --output-dir results/smt_$(basename $model)
done
```

#### Step 9: 互补性分析矩阵

```bash
python3 scripts/compare_smt_nmt.py \
  --smt-hyp results/smt_baseline/hyp_zh2en.txt \
  --nmt-hyp data/nmt/test.hyp.en \
  --ref data/nmt/test.en \
  --src data/nmt/test.zh \
  --output results/comparison_matrix.json
```

**分析维度:**
- 句子级 BLEU 分布对比 (箱线图)
- 句子长度比 (hyp_len/ref_len) 分布 — SMT 通常更短
- 词汇重叠率 (SMT ∩ NMT ∩ REF) — 识别互补覆盖
- OOV 率对比 — SMT 受限于短语表，NMT 受限于 BPE
- N-gram 多样性 (distinct-1/2/3) — 测量词汇丰富度

**预期发现 (驱动后续互补策略):**

| 维度 | SMT 特征 | NMT 特征 | 互补机会 |
|:-----|:---------|:---------|:---------|
| 短句 (<10词) | 更好 (短语精确匹配) | 好 | NMT 主导 |
| 长句 (>30词) | 差 (搜索困难) | 更好 (全局注意力) | NMT 主导 |
| 罕见词 | 差 (OOV，短语表未覆盖) | 较好 (BPE 子词) | NMT 主导 |
| 高频短语 | 好 (短语表精确保留) | 好 | 相当 |
| 词序 | 偏离 (距离失真) | 更接近原文 | NMT 主导 |
| 领域术语 | 准确 (短语表记忆) | 可能泛化 | **SMT 可补充** |
| 输出流畅度 | 较差 (n-gram LM) | 好 (神经 LM) | NMT 主导 |

**核心洞察:** NMT 在大多数维度上优于 SMT，但 SMT 在精确短语记忆上有独特优势。这决定了后续的互补策略 — **以 NMT 为主干，SMT 做词典覆盖补充**。

---

## Week 3–4: 回译数据增强

**目标:** 利用单语数据通过回译生成合成平行语料，增强 SMT 和 NMT 的训练数据

### 原理

```
单语英文 (200K sentences)
    ↓ EN→ZH NMT 翻译
合成中文 (200K sentences)
    ↓ 配对: (合成中文, 原始英文)
合成 ZH→EN 平行语料
    ↓ 与原 227K 真实语料合并
~427K 增强训练集 → 重新训练 SMT + NMT
```

### W3 Day 1–3: 训练反向 NMT (EN→ZH)

```bash
# ── EN→ZH 训练配置 (与 ZH→EN 对称) ──
cp config/nmt_zh2en_base.yaml config/nmt_en2zh_base.yaml

# 修改方向: -train_src data/nmt/train.bpe.en -train_tgt data/nmt/train.bpe.zh
# save_model: models/nmt/en2zh_base

onmt_train -config config/nmt_en2zh_base.yaml

# 训练时间: ~17-25h (与正向模型相同)
# 目标 BLEU (EN→ZH): 10-14 (中文 BLEU 天然低于英文 BLEU)
```

**EN→ZH 特殊考量:**
- 中文输出用 BPE 解码后无需额外分词即可评估 (sacrebleu `--tokenize zh`)
- BLEU 预期低于 ZH→EN: 中文是目标语言时，评估更严格 (字符级重叠 vs 词级)
- 回译质量容忍度较高: 合成中文不需要完美，只要保留足够语义信号

### W3 Day 4–5: 生成合成数据

#### Step 1: 准备英文单语数据

```bash
# 使用 data/mono/en.txt (~29MB, ~200K+ sentences)
# 清洗过滤
python3 -c "
import re
with open('data/mono/en.txt') as f:
    lines = [l.strip() for l in f if l.strip()]
# 过滤: 长度 5-100 词, 英文占比 >80%, 去重
import hashlib
seen = set()
clean = []
for line in lines:
    if len(line.split()) < 5 or len(line.split()) > 100:
        continue
    h = hashlib.md5(line.encode()).hexdigest()
    if h in seen:
        continue
    seen.add(h)
    clean.append(line)
print(f'Cleaned: {len(clean)} sentences (from {len(lines)})')
with open('data/mono/en_clean.txt', 'w') as f:
    f.write('\n'.join(clean))
"
```

#### Step 2: 回译生成

```bash
# BPE 编码英文单语
spm_encode --model=data/nmt/bpe_32k.model \
  < data/mono/en_clean.txt \
  > data/mono/en_clean.bpe.en

# 用 EN→ZH 模型翻译 (采样模式增加多样性)
onmt_translate \
  -model models/nmt/en2zh_base_best.pt \
  -src data/mono/en_clean.bpe.en \
  -output data/mono/backtrans.bpe.zh \
  -batch_size 4096 \
  -beam_size 1 \
  -random_sampling_topk 10 \
  -random_sampling_temp 0.8 \
  -gpu 0

# 去 BPE
spm_decode --model=data/nmt/bpe_32k.model \
  < data/mono/backtrans.bpe.zh \
  > data/mono/backtrans.zh

# 配对: (合成中文, 原始英文) 
paste data/mono/backtrans.zh data/mono/en_clean.txt \
  > data/nmt/backtrans_pairs.tsv
```

**为什么用采样而非 beam search:**
- Beam search 产生"过于干净"的合成数据，缺乏多样性
- `random_sampling_topk=10` + `temp=0.8` 引入可控随机性，生成更多样的中文表达
- 更接近人类写作的多样性 → 训练出的模型更鲁棒
- 参考 Edunov et al. (2018): 采样回译比 beam 回译提升 ~1-2 BLEU

#### Step 3: 合成数据质量过滤

```bash
python3 scripts/filter_backtrans.py \
  --input data/nmt/backtrans_pairs.tsv \
  --output data/nmt/backtrans_filtered.tsv \
  --src-lang zh --tgt-lang en \
  --min-len 3 --max-len 150 \
  --len-ratio-min 0.3 --len-ratio-max 3.0 \
  --keep-top 200000
```

过滤条件:
- 源/目标长度 3–150 tokens
- 长度比 0.3–3.0 (过滤明显不对齐)
- 保留质量最高的 top-200K 对 (如需更多, 可保留全部)

### W4 Day 1–2: 构建增强数据集

```bash
# 合并真实 + 合成数据
cat data/nmt/train.zh <(cut -f1 data/nmt/backtrans_filtered.tsv) > data/nmt/train_aug.zh
cat data/nmt/train.en <(cut -f2 data/nmt/backtrans_filtered.tsv) > data/nmt/train_aug.en

# 验证
wc -l data/nmt/train_aug.*
# 预期: ~400K 行 (~200K real + ~200K synthetic)

# BPE 编码
for lang in zh en; do
  spm_encode --model=data/nmt/bpe_32k.model \
    < data/nmt/train_aug.${lang} \
    > data/nmt/train_aug.bpe.${lang}
done

# 预处理
onmt_preprocess \
  -train_src data/nmt/train_aug.bpe.zh \
  -train_tgt data/nmt/train_aug.bpe.en \
  -valid_src data/nmt/valid.bpe.zh \
  -valid_tgt data/nmt/valid.bpe.en \
  -save_data data/nmt/processed_aug \
  -src_vocab_size 32000 -tgt_vocab_size 32000 -share_vocab \
  -src_seq_length 150 -tgt_seq_length 150 -overwrite
```

### W4 Day 3–7: 重训双系统

#### NMT 重训 (增强数据)

```bash
# 基于增强数据训练新模型
# 注意: 从零开始 (不从 prev checkpoint) — 数据分布变了
onmt_train -config config/nmt_zh2en_base.yaml \
  -data data/nmt/processed_aug \
  -save_model models/nmt/zh2en_aug \
  -train_steps 300000  # ~15% 更多步数 (数据多了 2×)
```

**增强训练注意事项:**
- 合成数据质量低于真实数据 → 可能需要调高 dropout 到 0.25
- 可选: 区分真实/合成数据源，给合成数据较低采样权重 (但实现复杂，非必需)
- 预期收敛略慢但最终 BLEU 更高 (+3–6 点)

#### SMT 重训 (增强数据)

```bash
# SMT 从增强数据重建
# 注意: SMT 对数据量敏感，200K 合成数据可显著降低 OOV 率

python3 scripts/train_fastalign.py \
  --direction zh2en \
  --train-src data/nmt/train_aug.zh \
  --train-tgt data/nmt/train_aug.en \
  --output-dir models/smt_zh2en_aug \
  --max-sentences 400000

# 评估
python3 scripts/eval_bleu.py \
  --model-dir models/smt_zh2en_aug \
  --direction zh2en \
  --test-src data/nmt/test.zh \
  --test-ref data/nmt/test.en \
  --output-dir results/smt_aug
```

**SMT 预期提升:**
- OOV 率: 3.8% → <1.5% (合成数据覆盖更多词汇)
- 短语表: 65.9K → ~80-120K 条目
- BLEU: 8 → 10-12 (SMT 对数据量的边际收益递减，但回译数据中的高频模式仍有价值)

### W4 产出

```
results/
├── smt_aug/
│   └── eval/
│       └── hyp_zh2en.txt         # SMT 增强译本
├── nmt_aug_eval.txt              # NMT 增强 BLEU
└── smt_aug_eval.txt              # SMT 增强 BLEU
```

---

## Week 5–6: 集成与系统组合

**目标:** 训练多模型集成 → NMT 集成 BLEU 最大化 → SMT+NMT 系统组合

### W5 Day 1–5: 多模型训练 (不同种子)

```bash
# 训练 3 个独立 NMT 模型，仅随机种子不同
for seed in 42 123 456; do
  onmt_train -config config/nmt_zh2en_base.yaml \
    -data data/nmt/processed_aug \
    -save_model models/nmt/zh2en_aug_seed${seed} \
    -seed ${seed} \
    -train_steps 300000
done
```

**为什么 3 个模型:**
- 2 个模型集成: +1-2 BLEU
- 3 个模型集成: +2-3.5 BLEU
- 5+ 模型集成: 边际收益递减，+0.3-0.5/模型
- **3 个模型是性价比最优点** (训练时间 vs 质量提升)

**备选方案 (如果GPU时间不够):**
- 使用同一模型的不同 checkpoint (epoch 18, 19, 20) 做 checkpoint averaging
- 只需 1 次训练，取最后 3-5 个 checkpoint 平均
- BLEU 提升 ~1-2 点 (不如多种子但省时)

### W5 Day 6–7: 集成评估

#### 方法 A: Checkpoint 平均 (最简单)

```bash
# 平均最后 5 个 checkpoint 的权重
onmt_average_models \
  -models models/nmt/zh2en_aug_step_290000.pt \
          models/nmt/zh2en_aug_step_292000.pt \
          models/nmt/zh2en_aug_step_294000.pt \
          models/nmt/zh2en_aug_step_296000.pt \
          models/nmt/zh2en_aug_step_298000.pt \
  -output models/nmt/zh2en_aug_avg5.pt

# 评估
onmt_translate -model models/nmt/zh2en_aug_avg5.pt \
  -src data/nmt/test.bpe.zh -output /tmp/hyp_avg5.en -beam_size 5 -gpu 0
spm_decode --model=data/nmt/bpe_32k.model < /tmp/hyp_avg5.en > /tmp/hyp_avg5.txt
sacrebleu data/nmt/test.en -i /tmp/hyp_avg5.txt -m bleu
```

#### 方法 B: 多模型 Beam 联合解码

```bash
# OpenNMT 支持多模型联合解码 (ensemble decoding)
onmt_translate \
  -model models/nmt/zh2en_aug_seed42_best.pt \
         models/nmt/zh2en_aug_seed123_best.pt \
         models/nmt/zh2en_aug_seed456_best.pt \
  -src data/nmt/test.bpe.zh \
  -output data/nmt/test.hyp_ens3.bpe.en \
  -batch_size 2048 \
  -beam_size 5 \
  -length_penalty avg \
  -alpha 1.0 \
  -gpu 0

spm_decode --model=data/nmt/bpe_32k.model \
  < data/nmt/test.hyp_ens3.bpe.en > data/nmt/test.hyp_ens3.en

sacrebleu data/nmt/test.en -i data/nmt/test.hyp_ens3.en \
  -m bleu chrf ter --confidence > results/nmt_ens3_eval.txt
```

#### 集成策略对比

```bash
# 全面对比所有集成策略
python3 scripts/compare_ensemble.py \
  --single models/nmt/zh2en_aug_seed42 \
  --single models/nmt/zh2en_aug_seed123 \
  --single models/nmt/zh2en_aug_seed456 \
  --avg5 models/nmt/zh2en_aug_avg5.pt \
  --ens3 models/nmt/zh2en_aug_seed42_best.pt \
          models/nmt/zh2en_aug_seed123_best.pt \
          models/nmt/zh2en_aug_seed456_best.pt \
  --test-src data/nmt/test.bpe.zh \
  --test-ref data/nmt/test.en \
  --bpe-model data/nmt/bpe_32k.model \
  --output results/ensemble_comparison.json
```

### W6 Day 1–3: 系统组合设计

**组合策略:** 以 NMT 为主干，SMT 做特定场景的补充。

```
输入: 中文源句
    │
    ├──▶ SMT Decoder ──▶ N-best list (top 10)
    │
    └──▶ NMT Ensemble ──▶ 1-best + N-best list (top 10)
              │
              ▼
    ┌─────────────────────────┐
    │  Confusion Network      │
    │  Combination            │
    │                         │
    │  1. 对齐 SMT 和 NMT    │
    │     输出的词序列        │
    │  2. 构建混淆网络        │
    │  3. 解码最优路径        │
    │     (LM + posteriors)   │
    └───────────┬─────────────┘
                ▼
         最终输出译文
```

#### 实现: 混淆网络系统组合

```python
# scripts/system_combine.py (新文件)

"""
SMT + NMT 系统组合 via Confusion Network Decoding.

原理:
1. 收集 SMT (N-best=10) 和 NMT ensemble (N-best=10) 的所有候选
2. 通过 TER 对齐将所有候选映射到统一词序列空间
3. 构建混淆网络 (confusion network): 每个位置有若干候选词及后验概率
4. 用语言模型在混淆网络上解码最优路径

参考: Rosti et al. (2007), "Combining Outputs from Multiple MT Systems"
"""

import sys
import argparse
from collections import defaultdict
import numpy as np
from smt.language_model import KneserNeyLM

def build_confusion_network(hypotheses: list[list[str]], scores: list[float]) -> dict:
    """
    构建混淆网络。
    
    hypotheses: 所有系统输出的候选译文 (词列表)
    scores: 对应的模型分数 (用于计算后验概率)
    
    返回: {
        'positions': [position_0_words, position_1_words, ...],
        'posteriors': [position_0_probs, position_1_probs, ...],
    }
    """
    # 1. 选择"骨架" (skeleton) — 得分最高的候选
    best_idx = np.argmax(scores)
    skeleton = hypotheses[best_idx]
    
    # 2. 将其他候选通过 TER 对齐到骨架
    # 使用编辑距离 (插入/删除/替换) 建立词对应
    positions = [[w] for w in skeleton]  # 每个位置的候选词列表
    posteriors = [[1.0] for _ in skeleton]  # 后验概率
    
    # 归一化分数为后验概率 (softmax over hypotheses)
    scores = np.array(scores)
    scores = scores - np.max(scores)  # 数值稳定
    probs = np.exp(scores) / np.sum(np.exp(scores))
    
    for i, hyp in enumerate(hypotheses):
        if i == best_idx:
            continue
        # 编辑距离对齐
        alignment = _align_to_skeleton(skeleton, hyp)
        for skel_pos, hyp_word, op in alignment:
            if op == 'match' or op == 'substitute':
                positions[skel_pos].append(hyp_word)
                posteriors[skel_pos].append(probs[i])
            elif op == 'insert':
                # 插入位置: 在当前位置后添加
                positions[skel_pos].append(hyp_word)
                posteriors[skel_pos].append(probs[i] * 0.5)  # 插入惩罚
            # 'delete': 跳过
    
    # 3. 每个位置归一化后验概率
    for j in range(len(posteriors)):
        total = sum(posteriors[j])
        if total > 0:
            posteriors[j] = [p / total for p in posteriors[j]]
    
    return {
        'positions': positions,
        'posteriors': posteriors,
        'skeleton': skeleton,
    }


def decode_confusion_network(cn: dict, lm, lm_weight: float = 0.5) -> list[str]:
    """
    在混淆网络上用 LM 解码最优路径。
    简化版: 贪心选择每个位置 LM 得分最高的词。
    """
    result = []
    history = ['<s>']
    
    for pos_words, pos_probs in zip(cn['positions'], cn['posteriors']):
        best_word = None
        best_score = -float('inf')
        
        for word, cn_prob in zip(pos_words, pos_probs):
            # 组合分数: λ·log P_cn(word) + (1-λ)·log P_lm(word | history)
            lm_prob = lm.log_prob(word, tuple(history[-(lm.order-1):]))
            combined = lm_weight * np.log(cn_prob + 1e-10) + lm_prob
            
            if combined > best_score:
                best_score = combined
                best_word = word
        
        if best_word and best_word != '<eps>':
            result.append(best_word)
            history.append(best_word)
    
    return result


def _align_to_skeleton(skeleton, hyp):
    """Levenshtein 对齐，返回 [(skel_pos, hyp_word, op), ...]"""
    # 简化版使用 Python difflib
    import difflib
    matcher = difflib.SequenceMatcher(None, skeleton, hyp)
    alignment = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == 'equal':
            for k in range(i1, i2):
                alignment.append((k, skeleton[k], 'match'))
        elif op == 'replace':
            for k in range(i1, i2):
                if k - i1 < j2 - j1:
                    alignment.append((k, hyp[j1 + (k - i1)], 'substitute'))
        elif op == 'insert':
            for k in range(j1, j2):
                alignment.append((i1, hyp[k], 'insert'))
        # delete: 不添加
    return alignment


def main():
    parser = argparse.ArgumentParser(description='SMT+NMT System Combination')
    parser.add_argument('--smt-nbest', required=True, help='SMT N-best file')
    parser.add_argument('--nmt-nbest', required=True, help='NMT N-best file')
    parser.add_argument('--lm', required=True, help='Path to KneserNeyLM for decoding')
    parser.add_argument('--lm-weight', type=float, default=0.5)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    # 加载 LM
    lm = KneserNeyLM.load(args.lm)
    
    # 解析 N-best 列表
    smt_hyps, smt_scores = _load_nbest(args.smt_nbest)
    nmt_hyps, nmt_scores = _load_nbest(args.nmt_nbest)
    
    # 合并
    all_hyps = smt_hyps + nmt_hyps
    all_scores = smt_scores + nmt_scores
    
    # 构建混淆网络 + 解码
    cn = build_confusion_network(all_hyps, all_scores)
    result = decode_confusion_network(cn, lm, args.lm_weight)
    
    with open(args.output, 'w') as f:
        f.write(' '.join(result) + '\n')
    
    print(f"Combined output written to {args.output}")


def _load_nbest(path):
    """解析 N-best 文件: 每行 "score ||| hypothesis" """
    hyps, scores = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(' ||| ')
            if len(parts) >= 2:
                scores.append(float(parts[0]))
                hyps.append(parts[1].split())
    return hyps, scores


if __name__ == '__main__':
    main()
```

### W6 Day 4–5: 系统组合执行

```bash
# Step 1: 生成 SMT N-best (top 10)
python3 scripts/generate_nbest.py \
  --model-dir models/smt_zh2en_aug \
  --direction zh2en \
  --test-src data/nmt/test.zh \
  --nbest 10 \
  --output results/smt_nbest.txt

# Step 2: 生成 NMT N-best (top 10)  
# (使用 OpenNMT 的 n_best 参数)
onmt_translate \
  -model models/nmt/zh2en_aug_seed42_best.pt \
         models/nmt/zh2en_aug_seed123_best.pt \
         models/nmt/zh2en_aug_seed456_best.pt \
  -src data/nmt/test.bpe.zh \
  -output data/nmt/test.nbest.en \
  -n_best 10 \
  -beam_size 10 \
  -gpu 0

# Step 3: 系统组合
python3 scripts/system_combine.py \
  --smt-nbest results/smt_nbest.txt \
  --nmt-nbest data/nmt/test.nbest.en \
  --lm models/zh2en_fa/lm.json \
  --lm-weight 0.4 \
  --output results/combined.hyp.en

# Step 4: BLEU 评估
sacrebleu data/nmt/test.en -i results/combined.hyp.en \
  -m bleu chrf ter --confidence > results/system_combined_eval.txt
```

### W6 Day 6–7: 结果汇总

```bash
# 全面对比
python3 scripts/final_comparison.py \
  --systems \
    smt_baseline:results/smt_baseline/hyp_zh2en.txt \
    nmt_base:data/nmt/test.hyp.en \
    smt_aug:results/smt_aug/hyp_zh2en.txt \
    nmt_aug:results/nmt_aug_test.hyp.en \
    nmt_ens3:data/nmt/test.hyp_ens3.en \
    combined:results/combined.hyp.en \
  --ref data/nmt/test.en \
  --output results/final_comparison_table.json
```

**预期汇总表:**

| System | BLEU | chrF | TER | Δ vs SMT |
|:-------|:----:|:----:|:---:|:---------:|
| SMT Baseline (fast_align) | 8.0 | 35 | 85 | — |
| NMT Base (epoch 20) | 16.5 | 48 | 65 | +8.5 |
| SMT + BackTrans | 11.0 | 40 | 78 | +3.0 |
| NMT + BackTrans | 20.0 | 52 | 58 | +12.0 |
| NMT Ensemble (3 models) | 23.0 | 55 | 53 | +15.0 |
| **System Combined** | **24.5** | **57** | **50** | **+16.5** |

---

## Week 7–8: 领域自适应 + 最终评估

**目标:** 对实验协议的 80 篇源文本领域进行微调，产出论文级结果

### W7 Day 1–3: 领域数据准备

#### 实验协议数据回顾

实验需翻译 80 篇源文本 (40 中文 + 40 英文)，分两个领域:
- **新闻域** (40 篇): 20 中文新闻 + 20 英文新闻
- **文学域** (40 篇): 20 中文文学 + 20 英文文学

```bash
# 从 data/source_texts/ 收集领域数据
# 新闻: data/source_texts/zh_news/* + en_news/*
# 文学: data/source_texts/zh_lit/* + en_lit/*

# 合并为领域语料
mkdir -p data/domain

# 新闻域 (用于领域自适应训练)
python3 scripts/build_domain_corpus.py \
  --news-zh data/source_texts/zh_news/ \
  --news-en data/source_texts/en_news/ \
  --literary-zh data/source_texts/zh_lit/ \
  --literary-en data/source_texts/en_lit/ \
  --output data/domain/

# 产出:
#   data/domain/news.zh, data/domain/news.en     (~40篇, 约 15K-25K tokens)
#   data/domain/literary.zh, data/domain/literary.en
```

**⚠ 领域数据极少:** 40 篇文本 (~15K-25K tokens) 对 NMT 微调来说极小。直接微调可能导致灾难性遗忘 (catastrophic forgetting)。采用策略:

#### 领域自适应策略: 数据选择 + 轻量微调

```
策略 A (推荐): 领域数据选择 → 从大语料中筛选领域相关句对 → 增强微调
策略 B (备选): 轻量微调 → 极低学习率 + 混合通用数据
```

**策略 A: 领域数据选择**

```bash
# 从 227K WMT 数据中筛选与新闻/文学域最相似的句对
# 使用 LASER / LaBSE 多语言句子嵌入 + 余弦相似度

python3 scripts/domain_data_selection.py \
  --candidate data/wmt/train.zh data/wmt/train.en \
  --domain data/domain/news.zh data/domain/news.en \
  --method labse \
  --top-k 20000 \
  --output data/domain/news_selected.tsv

# 同样为文学域筛选
python3 scripts/domain_data_selection.py \
  --candidate data/wmt/train.zh data/wmt/train.en \
  --domain data/domain/literary.zh data/domain/literary.en \
  --method labse \
  --top-k 20000 \
  --output data/domain/literary_selected.tsv
```

**策略 B (如果无 LaBSE/GPU 做嵌入):** 使用 n-gram 重叠 + TF-IDF 快速筛选。

### W7 Day 4–5: 领域微调

```bash
# ── 新闻域微调 ──
# 混合: 选定新闻域数据 (20K) + 原始通用数据 (40K) + 实验源文本 (40篇)
cat <(cut -f1 data/domain/news_selected.tsv) data/domain/news.zh > data/domain/news_finetune.zh
cat <(cut -f2 data/domain/news_selected.tsv) data/domain/news.en > data/domain/news_finetune.en

# BPE + 预处理
spm_encode --model=data/nmt/bpe_32k.model < data/domain/news_finetune.zh > data/domain/news.bpe.zh
spm_encode --model=data/nmt/bpe_32k.model < data/domain/news_finetune.en > data/domain/news.bpe.en

onmt_preprocess \
  -train_src data/domain/news.bpe.zh -train_tgt data/domain/news.bpe.en \
  -valid_src data/nmt/valid.bpe.zh -valid_tgt data/nmt/valid.bpe.en \
  -save_data data/domain/news_processed -overwrite

# 从最佳集成模型开始微调 (极低学习率)
onmt_train \
  -config config/nmt_zh2en_base.yaml \
  -data data/domain/news_processed \
  -save_model models/nmt/zh2en_news_ft \
  -train_from models/nmt/zh2en_aug_avg5.pt \
  -learning_rate 0.0005 \
  -warmup_steps 500 \
  -train_steps 5000 \
  -dropout 0.1 \
  -reset_optim all

# ── 文学域微调 (同理) ──
onmt_train \
  -config config/nmt_zh2en_base.yaml \
  -data data/domain/literary_processed \
  -save_model models/nmt/zh2en_lit_ft \
  -train_from models/nmt/zh2en_aug_avg5.pt \
  -learning_rate 0.0005 \
  -warmup_steps 500 \
  -train_steps 5000 \
  -dropout 0.2  # 文学域数据更少，加强正则化
```

**关键微调超参数:**
- `learning_rate=0.0005`: 通用训练 (2.0) 的 1/4000。防止灾难性遗忘。
- `warmup_steps=500`: 更短的预热，因为从已收敛点开始。
- `train_steps=5000`: 小数据只需少量步数 (~5-10 epochs on 20K pairs)。
- `reset_optim=all`: 重置优化器状态 (Adam 动量等)，因为数据分布变了。

### W7 Day 6–7: 生成实验用 160 篇译文

```bash
# ── 为实验协议生成全部译文 ──
mkdir -p output/experiment_translations/

# 1. NMT 系统 (所有 80 篇源文本)
for domain in news literary; do
  for direction in zh2en en2zh; do
    # 选择对应的微调模型
    model=models/nmt/${direction}_${domain}_ft_best.pt
    src=data/domain/${domain}.${direction%%2*}  # zh or en
    
    onmt_translate -model $model -src $src \
      -output output/experiment_translations/nmt_${domain}_${direction}.txt \
      -beam_size 5 -gpu 0
  done
done

# 2. SMT 系统 (使用最佳 fast_align 模型)
for domain in news literary; do
  python3 scripts/eval_bleu.py \
    --model-dir models/zh2en_fa \
    --direction zh2en \
    --test-src data/domain/${domain}.zh \
    --test-ref data/domain/${domain}.en \
    --output-dir output/experiment_translations/smt_${domain}_zh2en
done

# 3. 系统组合 (SMT+NMT) — 对每个源句子
python3 scripts/experiment_combine.py \
  --smt-nbest output/experiment_translations/smt_nbest/ \
  --nmt-nbest output/experiment_translations/nmt_nbest/ \
  --lm models/zh2en_fa/lm.json \
  --output output/experiment_translations/combined/
```

### W8 Day 1–3: 全面评估

```bash
# ── 1. 自动化指标评估 ──
# 对新闻域测试集 (held-out WMT)
python3 scripts/comprehensive_eval.py \
  --systems \
    smt_base:results/smt_baseline/hyp_zh2en.txt \
    nmt_base:data/nmt/test.hyp.en \
    nmt_aug:results/nmt_aug_test.hyp.en \
    nmt_ens3:data/nmt/test.hyp_ens3.en \
    combined:results/combined.hyp.en \
    nmt_news_ft:output/experiment_translations/nmt_news_zh2en.txt \
    nmt_lit_ft:output/experiment_translations/nmt_lit_zh2en.txt \
  --ref data/nmt/test.en \
  --metrics bleu chrf ter comet \
  --output results/full_eval_table.json

# ── 2. 统计显著性检验 ──
# Bootstrap 重采样 (1000 samples) 计算 BLEU 差异的置信区间
python3 scripts/significance_test.py \
  --baseline results/smt_baseline/hyp_zh2en.txt \
  --systems \
    nmt_base:data/nmt/test.hyp.en \
    nmt_ens3:data/nmt/test.hyp_ens3.en \
    combined:results/combined.hyp.en \
  --ref data/nmt/test.en \
  --n-bootstrap 1000 \
  --output results/significance.json
```

**预期显著性结果:**
- NMT vs SMT: p < 0.001 (极其显著)
- Ensemble vs Single NMT: p < 0.01
- Combined vs Ensemble: p < 0.05 (边界显著，取决于实现质量)

### W8 Day 4–5: 消融实验

```bash
# ── 消融分析: 量化每个改进的贡献 ──
# 
# 基线: SMT fast_align (BLEU 8)
# + 回译数据: SMT BLEU 10-12 (+2-4)
# + Transformer Base: NMT BLEU 16 (+8)
# + 回译数据: NMT BLEU 20 (+4)
# + 3 模型集成: BLEU 23 (+3)
# + 系统组合: BLEU 24.5 (+1.5)
# + 领域微调: BLEU 25 (+0.5, 通用测试集上; 领域测试上 +3-5)

python3 scripts/ablation_study.py \
  --output results/ablation_table.tex \
  --output results/ablation_figure.png
```

### W8 Day 6–7: 论文级产出

#### 产出清单

```
results/
├── full_eval_table.json          # 全系统评估指标
├── ablation_table.tex            # 消融实验 LaTeX 表
├── ablation_figure.png           # 消融贡献条形图
├── significance.json             # Bootstrap 显著性检验
├── comparison_matrix.json        # SMT vs NMT 互补分析
├── domain_results/               # 领域特定结果
│   ├── news_bleu.json
│   └── literary_bleu.json
├── human_eval_sample.md          # 50 句人工抽检对比
└── paper_figures/                # 论文图表
    ├── bleu_progression.png      # BLEU 进展曲线
    ├── smt_vs_nmt_scatter.png    # 句子级 BLEU 散点图
    └── ensemble_diversity.png    # 集成多样性分析
```

#### 论文关键表格: 主结果

| System | BLEU | chrF | TER | Params | Speed (sent/s) |
|:-------|:----:|:----:|:---:|:------:|:--------------:|
| SMT (fast_align) | 8.0 | 35.0 | 85.2 | — | 10 |
| SMT + BackTrans | 11.0 | 40.0 | 78.0 | — | 10 |
| NMT Base | 16.5 | 48.0 | 65.0 | 93M | 25 |
| NMT + BackTrans | 20.0 | 52.0 | 58.0 | 93M | 25 |
| NMT Ensemble (3) | 23.0 | 55.0 | 53.0 | 279M | 8 |
| **System Combined** | **24.5** | **57.0** | **50.0** | 279M+ | **5** |
| + Domain Adapt (news) | 25.0 | 58.0 | 48.0 | 279M+ | 5 |

#### 论文关键图表: 消融瀑布

```
BLEU Progression
24 ┤                                          ┌── Combined 24.5
22 ┤                                     ┌────┘
20 ┤                               ┌─────┘  (+3.0) Ensemble
18 ┤                          ┌────┘
16 ┤                     ┌────┘  (+4.0) +BackTrans
14 ┤                    │
12 ┤               ┌────┘
10 ┤          ┌────┘  (+8.0) NMT Base
 8 ┤─────SMT──┘
 6 ┤
   └──────────────────────────────────────────
    SMT    NMT    +BT    +Ens   +Comb  +Domain
```

---

## 风险矩阵与缓解

| 风险 | 概率 | 影响 | 缓解 |
|:-----|:----:|:----:|:-----|
| **GPU OOM** (V100 32GB 不足以训练 Transformer) | 低 | 高 | Batch size 降至 2048, accum_count 增至 8; 使用 FP16 混合精度 |
| **训练不收敛** (loss plateau 在 epoch 10+) | 中 | 高 | 调整 lr schedule; 降低 label smoothing; 检查数据质量 |
| **回译质量差** (EN→ZH 模型 BLEU < 8) | 中 | 中 | 使用更简单的复制策略 (copy monolingual EN to both sides); 或跳过回译直接集成 |
| **SMT BLEU 天花板** (增强后仍 < 12) | 中高 | 低 | SMT 作为补充组件，BLEU 低不影响最终组合; 强调其短语精确性优势 |
| **领域微调过拟合** (极小数据导致性能崩溃) | 中 | 中 | 使用极低 LR + early stopping + 混合通用数据; 不过度追求域内 BLEU |
| **训练时间超预期** (20 epochs × 3 seeds = 150+ GPU hours) | 中 | 中 | 用 checkpoint averaging 替代多 seed; 使用更小的 Transformer (4 layers, 512d) |
| **服务器网络/稳定性问题** | 低 | 中 | 保存 checkpoint 每 2000 steps; 从最新 ckpt 恢复 |

---

## 命令速查表

```bash
# ── 快速评估 BLEU ──
sacrebleu ref.txt -i hyp.txt -m bleu chrf --confidence

# ── 查看 GPU 状态 ──
nvidia-smi && watch -n 1 nvidia-smi

# ── 从 checkpoint 恢复训练 ──
onmt_train -config config.yaml -train_from models/xxx_step_N.pt

# ── 仅翻译 (不训练) ──
onmt_translate -model model.pt -src test.bpe.zh -output hyp.bpe.en -gpu 0

# ── 平均多个 checkpoint ──
onmt_average_models -models m1.pt m2.pt m3.pt -output avg.pt

# ── BPE 编解码 ──
spm_encode --model=bpe.model < raw.txt > bpe.txt
spm_decode --model=bpe.model < bpe.txt > raw.txt

# ── 检查训练日志 ──
grep "Step\|Validation\|BLEU\|Perplexity" models/nmt/zh2en_base.log | tail -20

# ── SMT 评估 ──
python3 scripts/eval_bleu.py --model-dir models/zh2en_fa \
  --direction zh2en --test-src test.zh --test-ref test.en \
  --beam-size 5 --output-dir results/eval
```

---

## 备选/降级路径

### 路径 A: 快速通道 (GPU 时间有限)

如果训练时间不足，跳过 3 模型集成:

```
W1-2: NMT Base (1 model) → BLEU 16
W3-4: 回译 + 重训 (1 model) → BLEU 20
W5-6: Checkpoint 平均 (替代多 seed) → BLEU 21.5
W7-8: 系统组合 + 领域 → BLEU 23
```

节省 ~60 GPU hours (省去 2 个额外 seed 训练)，BLEU 仅损失 ~1.5 点。

### 路径 B: 最小可行 (最终期限紧迫)

如果只剩 4 周:

```
W1-2: 完成 NMT 20 epoch + 评估 → BLEU 16
W3: 简单 SMT+NMT 组合 (不增强) → BLEU 18
W4: 领域微调 + 论文产出 → BLEU 19
```

### 路径 C: SMT 优先 (GPU 不可用)

如果 V100 不可用:

```
W1-2: SMT 213K 扩展 + MERT 调优 → BLEU 12
W3-4: SMT 短语表改进 + 词汇化重排序 → BLEU 14
W5-6: NiuTrans.SMT 评估 → BLEU 18-22
W7-8: 最终评估 + 论文 → BLEU 20
```

---

## 附录: 关键文件清单

```
Teddy/
├── data/
│   ├── nmt/                       # NMT 处理后数据
│   │   ├── train.bpe.{zh,en}      # BPE 编码训练集
│   │   ├── valid.bpe.{zh,en}      # 验证集
│   │   ├── test.bpe.{zh,en}       # 测试集
│   │   ├── bpe_32k.{model,vocab}  # SentencePiece 模型
│   │   ├── processed*             # OpenNMT 预处理数据
│   │   └── train_aug.*            # 增强训练数据
│   ├── domain/                    # 领域数据
│   └── mono/                      # 单语数据
├── models/
│   ├── nmt/                       # NMT checkpoints
│   │   ├── zh2en_base_*.pt        # 基础模型
│   │   ├── zh2en_aug_*.pt         # 增强模型
│   │   ├── en2zh_base_*.pt        # 反向模型
│   │   ├── zh2en_news_ft_*.pt     # 新闻微调
│   │   └── zh2en_lit_ft_*.pt      # 文学微调
│   └── smt_zh2en_aug/             # 增强 SMT
├── config/
│   ├── nmt_zh2en_base.yaml        # NMT 训练配置
│   └── nmt_en2zh_base.yaml        # 反向训练配置
├── scripts/
│   ├── system_combine.py          # 系统组合脚本 (新)
│   ├── compare_smt_nmt.py         # 对比分析脚本 (新)
│   ├── comprehensive_eval.py      # 综合评估 (新)
│   ├── ablation_study.py          # 消融研究 (新)
│   ├── domain_data_selection.py   # 领域数据选择 (新)
│   ├── experiment_combine.py      # 实验批量组合 (新)
│   ├── filter_backtrans.py        # 回译数据过滤 (新)
│   └── generate_nbest.py          # N-best 生成 (新)
├── results/                       # 所有评估结果
└── output/
    └── experiment_translations/   # 160 篇实验译文
```

---

## 参考文献

1. Sennrich et al. (2016). "Improving Neural Machine Translation Models with Monolingual Data." *ACL*.
2. Edunov et al. (2018). "Understanding Back-Translation at Scale." *EMNLP*.
3. Vaswani et al. (2017). "Attention Is All You Need." *NeurIPS*.
4. Rosti et al. (2007). "Combining Outputs from Multiple Machine Translation Systems." *NAACL*.
5. Koehn et al. (2007). "Moses: Open Source Toolkit for Statistical Machine Translation." *ACL*.
6. Dyer et al. (2013). "A Simple, Fast, and Effective Reparameterization of IBM Model 2." *NAACL*.
7. Ott et al. (2019). "fairseq: A Fast, Extensible Toolkit for Sequence Modeling." *NAACL*.
8. Klein et al. (2017). "OpenNMT: Open-Source Toolkit for Neural Machine Translation." *ACL*.

---

*Generated: 2026-06-07 | Prometheus — Teddy NMT+SMT Strategic Plan v1.0*
