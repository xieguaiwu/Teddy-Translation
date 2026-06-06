#!/bin/bash
# SMT Training Progress Checker
# Usage: bash scripts/check_progress.sh

S1="sshpass -p 'The9phae' ssh -o StrictHostKeyChecking=no root@223.109.239.36 -p 24520"
S2="sshpass -p 'iewoh9vu' ssh -o StrictHostKeyChecking=no root@223.109.239.32 -p 20248"

echo "╔══════════════════════════════════════════════════════╗"
echo "║          SMT Training Progress Monitor              ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Server 1 - ZH→EN
echo "━━━ Server 1: ZH→EN 200K ━━━"
eval "$S1" '
  echo "Process: $(ps aux | grep train_200k | grep -v grep | awk "{printf \"PID=%s CPU=%s%% MEM=%s%% TIME=%s\", \$2, \$3, \$4, \$10}" 2>/dev/null || echo "NOT RUNNING")"
  echo "Last log lines:"
  tail -3 /root/smt_model/logs/train_zh2en_200k.log 2>/dev/null
  echo "Models: $(ls /root/smt_model/model/ | grep zh2en_200k | tr \"\n\" \" \" 2>/dev/null)"
  free -h | head -2
'

echo ""
echo "━━━ Server 2: EN→ZH 200K ━━━"
eval "$S2" '
  echo "Process: $(ps aux | grep train_200k | grep -v grep | awk "{printf \"PID=%s CPU=%s%% MEM=%s%% TIME=%s\", \$2, \$3, \$4, \$10}" 2>/dev/null || echo "NOT RUNNING")"
  echo "Last log lines:"
  tail -3 /root/smt_model/logs/train_en2zh_200k_v2.log 2>/dev/null
  echo "Models: $(ls /root/smt_model/model/ | grep en2zh_200k | tr \"\n\" \" \" 2>/dev/null)"
  free -h | head -2
'

echo ""
echo "━━━ Time estimate ───────────────────────────────────"
echo "Each batch (10K sentences): ~15-20 min for IBM2 + ~3-5 min for phrase table + ~2 min for LM"
echo "20 batches total → estimated ~7-9 hours per direction"
echo "Start: ~19:50 → Estimated finish: ~03:00-05:00"
echo "────────────────────────────────────────────────────"
