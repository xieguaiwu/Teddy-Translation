# Research: NMT Data Quality Diagnostics — Remote Server

## Summary
Diagnostic checks were designed to assess NMT parallel data quality on server **223.109.239.36:24224** (root access). Due to the research subagent lacking a shell/SSH execution tool, the commands could not be executed remotely. This document records the complete methodology, commands, and expected diagnostic criteria. If a bash-enabled session is granted, the commands can be run immediately.

## Findings

### 1. Diagnostic Scope
The checks cover seven dimensions of NMT data quality and training status:
- **Line counts**: Verify corpus size and train/validation split ratio
- **Data preview**: Spot-check parallel alignment quality, encoding, and tokenization
- **Empty lines**: Detect corrupt or missing entries that will crash training
- **OOV / <unk> tokens**: Measure vocabulary coverage in validation data
- **Sentence length distribution**: Identify outliers (ultra-short or ultra-long) that bias training
- **GPU utilization**: Determine if a training job is actively running
- **Python processes**: Identify which training script (if any) is executing

### 2. Commands and Expected Interpretation

#### 2a. Line Counts
```bash
wc -l /home/vipuser/zh.txt /home/vipuser/en.txt /home/vipuser/nmt/val_tok.zh /home/vipuser/nmt/val_tok.en
```
**Expected**: `zh.txt` and `en.txt` should have identical line counts (parallel corpus, one sentence per line). `val_tok.zh` and `val_tok.en` should also match each other. A mismatch of >5 lines indicates alignment corruption.

#### 2b. Data Preview
```bash
head -3 /home/vipuser/zh.txt /home/vipuser/en.txt /home/vipuser/nmt/val_tok.zh /home/vipuser/nmt/val_tok.en
```
**Checks**: Are Chinese lines actually Chinese (detectable CJK characters)? Are English lines actually English (Latin script)? Are tokenized files using BPE/SPM subword units (e.g., `▁`, `@@`)? Is validation data from the same domain as training data?

#### 2c. Empty Lines
```bash
awk 'length==0{count++} END{print count}' /home/vipuser/zh.txt
```
**Expected**: `0`. Any empty line in a parallel corpus breaks sentence alignment and will cause training to crash or silently skip pairs. This same check should also be run on `en.txt`.

#### 2d. Unknown Tokens
```bash
grep -c '<unk>' /home/vipuser/nmt/val_tok.zh
```
**Expected**: `0` or very low (<1% of lines). High <unk> count means the validation data contains many OOV words not seen in the training vocabulary, indicating a domain mismatch or insufficient vocabulary size.

#### 2e. Sentence Length Distribution
```bash
awk '{print NF}' /home/vipuser/zh.txt | sort -n | \
  awk 'BEGIN{min=9999;max=0}{if(NR==1)min=$1;if(NR==int(NR/2))med=$1; \
  if(NR==int(NR*0.9))p90=$1;if(NR==int(NR*0.99))p99=$1;max=$1} \
  END{print "min="min" med="med" p90="p90" p99="p99" max="max}'
```
**What to look for**:
- **min=0 or med < 5**: Many very short sentences — may indicate empty lines or tokenization problems
- **p90 > 200**: Long sentences will consume disproportionate GPU memory and may need to be truncated
- **p99/max >> p90**: Extreme outliers should be inspected; consider discarding sentences > 250 tokens
- **High ratio of max/med** (>20x): Unbalanced length distribution causes padding waste in batching

This should be run on both `zh.txt` and `en.txt` (and separately on `val_tok.zh` and `val_tok.en`).

#### 2f. GPU Status
```bash
nvidia-smi
```
**Checks**: Are any GPUs active? If utilization >0% and a training process is consuming GPU memory, NMT training is likely running. If GPUs show 0% utilization and no processes, training may have crashed or not yet started.

#### 2g. Python Processes
```bash
ps aux | grep python
```
**Identifies**: Active training scripts (e.g., `fairseq-train`, `onmt_train`, `train.py`), data loading processes, or any zombie/defunct processes that need cleanup. The absence of Python processes suggests training hasn't started or has already completed/crashed.

### 3. Execution Status
- **Remote host**: `223.109.239.36:24224`
- **Credentials**: Available (root / quah9Moh)
- **Authentication method**: `sshpass` (password-based SSH)
- **Executed**: ❌ Not executed — research subagent lacks bash/SSH tool

## Sources
- **Kept**: N/A — no external sources were consulted
- **Dropped**: N/A

## Gaps
1. **Actual server data**: All seven diagnostics need to be run on the remote server to obtain real metrics. The output will reveal specific issues with corpus size, alignment quality, vocabulary coverage, length distribution balance, and training status.
2. **en.txt empty-line and <unk> checks**: The task only specified empty-line check for zh.txt and <unk> check for val_tok.zh. For completeness, identical checks should also run on the English side.
3. **Tokenization scheme detection**: The head output will reveal whether BPE (`@@`) or SentencePiece (`▁`) tokenization was used, which informs vocabulary size and model selection decisions.

## Recommended Next Steps
1. **Grant bash/SSH tool** to this subagent so the diagnostics can be executed directly
2. **OR run the sshpass one-liner manually**, paste output back, and this research.md will be updated with findings
3. After diagnostics, if quality issues are found (empty lines, high <unk>, length imbalance), corrective actions include:
   - Remove empty lines: `awk 'length>0' zh.txt > zh_clean.txt`
   - Filter long sentences: `awk -v max=250 '{if(NF<=max)print}' zh.txt > zh_filtered.txt`
   - Rebuild BPE vocabulary with larger vocab size if <unk> > 1%
   - Re-align parallel corpus if line counts mismatch

## Supervisor coordination
`contact_supervisor` was attempted 3 times (all returned "No reply within 10 minutes"). `intercom` was attempted 5+ times against sessions `fe4f2c64` and `1c90d20d` (both show in `list` but return "Session not found"). The diagnostic commands are fully prepared and ready for execution once shell access is available.
