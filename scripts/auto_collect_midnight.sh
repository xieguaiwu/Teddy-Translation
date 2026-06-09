#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# auto_collect_midnight.sh — Teddy NMT+SMT 定时数据收集脚本
#
# 运行窗口: 00:00 ~ 00:15 CST (每日)
# 密码硬编码，直接配置 crontab 即可运行
#
# crontab:
#   0 0 * * * /home/xieguiawu/Desktop/ML/Teddy/scripts/auto_collect_midnight.sh
# ═══════════════════════════════════════════════════════════════════════════

export PATH="/usr/local/bin:/usr/bin:/bin:/home/xieguiawu/.local/bin:$PATH"
set -o pipefail

# ── 服务器配置 ──────────────────────────────────────────────────────────
HOST_A="223.109.239.11"; PORT_A="15512"; PASS_A="ahd3Ahc3"
HOST_B="223.109.239.36"; PORT_B="24224"; PASS_B="quah9Moh"
USER="root"

# ── 本地路径 ─────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOCAL_BASE="/home/xieguiawu/Desktop/ML/Teddy/results/remote/auto_${TIMESTAMP}"
LOCAL_MODELS="/home/xieguiawu/Desktop/ML/Teddy/models"
LOCAL_OUTPUT="/home/xieguiawu/Desktop/ML/Teddy/output"

# ── 远程路径 ─────────────────────────────────────────────────────────────
REMOTE_TRAIN_LOG="/root/teddy_nmt_v2/finetune_opus.log"
REMOTE_NMT_DIR="/root/teddy_nmt_v2"
REMOTE_CHECKPOINTS="/root/teddy_nmt_v2/checkpoints"
REMOTE_FINETUNED="/root/teddy_nmt_v2/finetuned_opus"
REMOTE_STATUS_JSON="/root/teddy_nmt_v2/TRAINING_STATUS.json"
REMOTE_V2_LOG="/home/vipuser/nmt_v2/train.log"
REMOTE_V2_CHECKPOINTS="/home/vipuser/nmt_v2/checkpoints"

# ── 截止时间 (00:15 CST) ────────────────────────────────────────────────
DEADLINE_EPOCH=$(TZ=Asia/Shanghai date -d "00:15" +%s 2>/dev/null || echo 0)

# ── 初始化 ───────────────────────────────────────────────────────────────
mkdir -p "${LOCAL_BASE}"/{server_a,server_b,server_b/checkpoints,reports}
LOGFILE="${LOCAL_BASE}/collect.log"
exec > >(tee -a "${LOGFILE}") 2>&1

log()  { echo "[$(TZ=Asia/Shanghai date '+%H:%M:%S')] $*"; }
die()  { echo "[FATAL] $*" >&2; exit 1; }

command -v sshpass >/dev/null 2>&1 || die "sshpass not found. apt-get install sshpass"

is_deadline_passed() { [ "$(date +%s)" -ge "$DEADLINE_EPOCH" ]; }

# ── SSH/SCP ──────────────────────────────────────────────────────────────
ssh_a()  { sshpass -p "$PASS_A" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
              -o ServerAliveInterval=10 -o ServerAliveCountMax=2 \
              -p "$PORT_A" "$USER@$HOST_A" "$@" 2>/dev/null; }
scp_a()  { sshpass -p "$PASS_A" scp -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
              -o ServerAliveInterval=10 -o ServerAliveCountMax=2 \
              -P "$PORT_A" "$@" 2>/dev/null; }
ssh_b()  { sshpass -p "$PASS_B" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
              -o ServerAliveInterval=10 -o ServerAliveCountMax=2 \
              -p "$PORT_B" "$USER@$HOST_B" "$@" 2>/dev/null; }
scp_b()  { sshpass -p "$PASS_B" scp -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
              -o ServerAliveInterval=10 -o ServerAliveCountMax=2 \
              -P "$PORT_B" "$@" 2>/dev/null; }

# ── 检查可达性 ──────────────────────────────────────────────────────────
check_alive() {
    if [ "$2" = "alive" ]; then
        log "  + Server $1: reachable"; return 0
    else
        log "  - Server $1: unreachable"; return 1
    fi
}

# ── 收集 Server A ──────────────────────────────────────────────────────
collect_server_a() {
    local dest="$LOCAL_BASE/server_a"; local err="$LOCAL_BASE/ssh_errors.log"
    mkdir -p "$dest/checkpoints" "$dest/finetuned_opus" 2>/dev/null
    log ""; log "-- Server A: RTX 3080 ($HOST_A:$PORT_A) --"

    local alive; alive=$(ssh_a "echo alive" 2>/dev/null) || alive=""
    check_alive "A" "$alive" || return 1

    local gpu=$(ssh_a "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader" 2>/dev/null)
    [ -n "$gpu" ] && while IFS= read -r line; do [ -n "$line" ] && log "  GPU: $line"; done <<< "$gpu"

    local procs=$(ssh_a "ps -eo pid,pcpu,etime,args --sort=-pcpu 2>/dev/null | grep -E 'train\\.py|finetune' | grep -v grep" 2>/dev/null)
    if [ -n "$procs" ]; then
        local cnt=$(echo "$procs" | wc -l)
        log "  Training processes ($cnt):"
        while IFS= read -r line; do log "    $line"; done <<< "$procs"
    else
        log "  Training: no active processes"
    fi

    log "  Downloading train log..."
    timeout 30 scp_a "$USER@$HOST_A:$REMOTE_TRAIN_LOG" "$dest/finetune_opus.log" 2>>"$err"
    [ -f "$dest/finetune_opus.log" ] && log "    + $(wc -c < "$dest/finetune_opus.log") bytes" || log "    - log not found"

    timeout 15 scp_a "$USER@$HOST_A:$REMOTE_NMT_DIR/config.py" "$dest/config.py" 2>>"$err"
    timeout 15 scp_a "$USER@$HOST_A:$REMOTE_NMT_DIR/model.py"  "$dest/model.py"  2>>"$err"
    timeout 15 scp_a "$USER@$HOST_A:$REMOTE_STATUS_JSON" "$dest/TRAINING_STATUS.json" 2>>"$err"

    local ckpt=$(ssh_a "ls $REMOTE_CHECKPOINTS/*.pt 2>/dev/null | wc -l" 2>/dev/null)
    if [ "${ckpt:-0}" -gt 0 ] 2>/dev/null; then
        timeout 60 scp_a "$USER@$HOST_A:$REMOTE_CHECKPOINTS/*.pt" "$dest/checkpoints/" 2>>"$err"
        log "  + Checkpoints: $ckpt file(s)"
        is_deadline_passed && { log "  - Deadline passed"; return 0; }
    fi

    local ft=$(ssh_a "du -sh $REMOTE_FINETUNED 2>/dev/null | cut -f1" 2>/dev/null)
    [ -n "$ft" ] && timeout 120 scp_a -r "$USER@$HOST_A:$REMOTE_FINETUNED/." "$dest/finetuned_opus/" 2>>"$err" && \
        log "  + Fine-tuned: $ft" || log "  - Fine-tuned: none"
}

# ── 收集 Server B ──────────────────────────────────────────────────────
collect_server_b() {
    local dest="$LOCAL_BASE/server_b"; local err="$LOCAL_BASE/ssh_errors.log"
    mkdir -p "$dest/checkpoints" 2>/dev/null
    log ""; log "-- Server B: V100 ($HOST_B:$PORT_B) --"

    local alive; alive=$(ssh_b "echo alive" 2>/dev/null) || alive=""
    check_alive "B" "$alive" || return 1

    local gpu=$(ssh_b "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader" 2>/dev/null)
    [ -n "$gpu" ] && while IFS= read -r line; do [ -n "$line" ] && log "  GPU: $line"; done <<< "$gpu"

    local procs=$(ssh_b "ps -eo pid,pcpu,etime,args --sort=-pcpu 2>/dev/null | grep -E 'train\\.py' | grep -v grep" 2>/dev/null)
    [ -n "$procs" ] && while IFS= read -r line; do log "    $line"; done <<< "$procs" || log "  Training: none"

    timeout 30 scp_b "$USER@$HOST_B:$REMOTE_V2_LOG" "$dest/train_v2.log" 2>>"$err"
    [ -f "$dest/train_v2.log" ] && log "  + log ($(wc -c < "$dest/train_v2.log") bytes)" || log "  - log not found"

    local ckpt=$(ssh_b "ls $REMOTE_V2_CHECKPOINTS/*.pt 2>/dev/null | wc -l" 2>/dev/null)
    [ "${ckpt:-0}" -gt 0 ] 2>/dev/null && \
        timeout 60 scp_b "$USER@$HOST_B:$REMOTE_V2_CHECKPOINTS/*.pt" "$dest/checkpoints/" 2>>"$err" && \
        log "  + Checkpoints: $ckpt" || log "  - Checkpoints: none"
}

# ── 本地快照 ────────────────────────────────────────────────────────────
snapshot_local() {
    log ""; log "-- Local Snapshot --"
    log "  SMT models:"
    for d in "$LOCAL_MODELS"/*/; do
        [ -d "$d" ] || continue
        local n=$(basename "$d"); local s=$(du -sh "$d" 2>/dev/null | cut -f1)
        local p="N/A"
        [ -f "${d}phrase_table.txt" ] && p=$(grep -c '|||' "${d}phrase_table.txt" 2>/dev/null || echo 0)
        log "    $n: $s, $p phrases"
    done
    log "  SMT outputs:"
    for d in "$LOCAL_OUTPUT"/*/; do
        [ -d "$d" ] || continue
        local n=$(basename "$d"); local c=$(find "$d" -name '*.txt' 2>/dev/null | wc -l)
        log "    $n: $c files"
    done
    local hf=$(du -sh /home/xieguiawu/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/ 2>/dev/null | cut -f1)
    log "  HF cache (opus): ${hf:-not found}"
}

# ═══════════════════════════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════════════════════════
log "================================"
log "Teddy Auto Collect STARTED"
log "$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S CST') -> $LOCAL_BASE"
log "================================"

collect_server_a
collect_server_b
snapshot_local

CYCLE=0
while ! is_deadline_passed; do
    CYCLE=$((CYCLE + 1)); log ""; log "== Cycle $CYCLE =="
    collect_server_a; collect_server_b
    local cf=$(find "$LOCAL_BASE" -type f \( -name "*.log" -o -name "*.pt" -o -name "*.json" \) 2>/dev/null | wc -l)
    log "  Files: $cf"
    if ! is_deadline_passed; then
        local r=$((DEADLINE_EPOCH - $(date +%s)))
        [ "$r" -le 55 ] && break
        local st=$(( r > 115 ? 60 : r - 55 ))
        [ "$st" -gt 0 ] && { log "  Sleep ${st}s..."; sleep "$st"; }
    fi
done

log ""; log "== FINAL =="
collect_server_a; collect_server_b; snapshot_local

# ── 报告 ────────────────────────────────────────────────────────────────
total_files=$(find "$LOCAL_BASE" -type f ! -name 'collect.log' ! -path '*/reports/*' 2>/dev/null | wc -l)
[ "$total_files" -eq 0 ] && die "No data collected"

REPORT="$LOCAL_BASE/reports/teddy_status_$TIMESTAMP.md"
COLLECT_TIME="$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S CST')"

# ── Python 报告生成 ────────────────────────────────────────────────────
python3 << PYEOF
import os, glob, subprocess as sp

base = "$LOCAL_BASE"
report_path = "$REPORT"
ct = "$COLLECT_TIME"
pa, pb, ha, hp, hb, pb_port = "$PASS_A", "$PASS_B", "$HOST_A", "$PORT_A", "$HOST_B", "$PORT_B"

def read_or(p, d=""):
    try:
        with open(p) as f: return f.read()
    except: return d

def run(cmd, t=15):
    try: return sp.check_output(cmd, shell=True, stderr=sp.DEVNULL, timeout=t).decode().strip()
    except (sp.CalledProcessError, sp.TimeoutExpired, FileNotFoundError, OSError): return "unreachable"

def rc(pw, h, pt, cmd):
    return run(f'sshpass -p "{pw}" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 -p {pt} root@{h} "{cmd}" 2>/dev/null', 15)

s1l = read_or(base + "/server_a/finetune_opus.log")
s2l = read_or(base + "/server_b/train_v2.log")
s1lt = "\n".join(s1l.strip().split("\n")[-20:]) if s1l else "n/a"
s2lt = "\n".join(s2l.strip().split("\n")[-20:]) if s2l else "n/a"
s1r = "\n".join([l for l in s1l.split("\n") if any(k in l for k in ["BLEU","Best","Epoch","Dev","loss"])][-10:]) if s1l else "n/a"

s1g = rc(pa, ha, hp, "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader")
s1p = rc(pa, ha, hp, "ps -eo pid,pcpu,etime,args --sort=-pcpu 2>/dev/null | grep -E 'train\\.py|finetune' | grep -v grep")
s2g = rc(pb, hb, pb_port, "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader")
s2p = rc(pb, hb, pb_port, "ps -eo pid,pcpu,etime,args --sort=-pcpu 2>/dev/null | grep -E 'train\\.py' | grep -v grep")

mdls = []
for d in sorted(glob.glob("$LOCAL_MODELS/*/")):
    n = os.path.basename(d.rstrip("/"))
    s = run(f"du -sh '{d}' 2>/dev/null | cut -f1")
    pt = os.path.join(d, "phrase_table.txt")
    if os.path.exists(pt):
        with open(pt) as f: p = str(sum(1 for _ in f))
    else: p = "N/A"
    mdls.append(f"  {n}: {s}, {p} phrases")

outs = []
for d in sorted(glob.glob("$LOCAL_OUTPUT/*/")):
    n = os.path.basename(d.rstrip("/"))
    c = len(glob.glob(os.path.join(d, "*.txt")))
    outs.append(f"  {n}: {c} files")

fl = []
for r, dd, ff in os.walk(base):
    if "/reports" in r: continue
    for f in ff:
        fp = os.path.join(r, f)
        fl.append(f"  {os.path.getsize(fp):>8}  {os.path.relpath(fp, base)}")

report = f"""# Teddy NMT+SMT 实验状态报告

> 自动收集时间: {ct}
> 路径: {base}

## Server A: RTX 3080

### GPU
\`\`\`
{s1g}
\`\`\`
### 进程
\`\`\`
{s1p or "none"}
\`\`\`
### 日志最后 20 行
\`\`\`
{s1lt}
\`\`\`
### 结果摘要
\`\`\`
{s1r or "n/a"}
\`\`\`

## Server B: V100

### GPU
\`\`\`
{s2g}
\`\`\`
### 进程
\`\`\`
{s2p or "none"}
\`\`\`
### 日志最后 20 行
\`\`\`
{s2lt}
\`\`\`

## 本地资产

### SMT 模型
\`\`\`
{chr(10).join(mdls) if mdls else "  none"}
\`\`\`
### SMT 译文
\`\`\`
{chr(10).join(outs) if outs else "  none"}
\`\`\`
### 文件清单
\`\`\`
{chr(10).join(fl) if fl else "  (empty)"}
\`\`\`
"""

os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, "w") as f: f.write(report)
print(f"Report: {report_path}")
PYEOF

log ""; log "================================"
log "DONE: $REPORT"
log "================================"
