#!/usr/bin/env bash
# chip_wedge_monitor.sh — READ-ONLY fast chip-wedge detector for B2 (petalinux2).
set -u
SSH_HOST="${SSH_HOST:-147.46.117.99}"; SSH_PORT="${SSH_PORT:-1326}"; SSH_USER="${SSH_USER:-root}"
HB_PATH="${HB_PATH:-/var/volatile/imcflow_chip_heartbeat.txt}"
DUMP_DIR="${DUMP_DIR:-/var/volatile/debug_nodes}"
STALL_SECS=15; INTERVAL=3; SAMPLE_IDX=0; LOGFILE=""
while [[ $# -gt 0 ]]; do case "$1" in
  --stall) STALL_SECS="$2"; shift 2;; --interval) INTERVAL="$2"; shift 2;;
  --sample) SAMPLE_IDX="$2"; shift 2;; --log) LOGFILE="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 2;; esac; done
ts(){ date '+%Y-%m-%d %H:%M:%S'; }
log(){ local l="[$(ts)] $*"; echo "$l"; [[ -n "$LOGFILE" ]] && echo "$l" >>"$LOGFILE"; }
rssh(){ timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=6 -o ServerAliveInterval=3 \
        -o ServerAliveCountMax=1 -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "$@" 2>/dev/null; }
probe(){ rssh "
  echo -n alive';';
  if [ -f '$HB_PATH' ]; then hb=\$(cat '$HB_PATH' 2>/dev/null);
    e=\$(echo \"\$hb\"|sed -n 's/.*epoch=\([0-9]*\).*/\1/p');
    it=\$(echo \"\$hb\"|sed -n 's/.*sample_iter=\([0-9]*\/[0-9]*\).*/\1/p');
    echo -n \"\${e:--}';'\${it:--}';\"; else echo -n \"-';'-';\"; fi;
  d='$DUMP_DIR/sample_$SAMPLE_IDX';
  if [ -d \"\$d\" ]; then
    mx=\$(ls -1 \"\$d\" 2>/dev/null|sed -n 's/^\([0-9]\{1,\}\)_.*/\1/p'|sort -n|tail -1);
    cnt=\$(ls -1 \"\$d\" 2>/dev/null|wc -l); echo \"\${mx:--}';'\${cnt:--}\";
  else echo \"-';'-\"; fi"; }
log "=== chip_wedge_monitor start ==="
log "host=$SSH_USER@$SSH_HOST:$SSH_PORT hb=$HB_PATH dump=$DUMP_DIR/sample_$SAMPLE_IDX stall=${STALL_SECS}s int=${INTERVAL}s"
[[ -z "$(rssh 'echo alive')" ]] && { log "SSH DEAD at startup — board unreachable (reboot?)."; exit 42; }
last_sig=""; last_change=$(date +%s); last_node="-"; last_iter="-"; fail=0
while true; do
  out="$(probe)"; now=$(date +%s)
  if [[ -z "$out" || "$out" != alive* ]]; then
    fail=$((fail+1)); log "ssh probe failed (streak=$fail)"
    if [[ $fail -ge 2 ]]; then
      log "*** WEDGE: SSH DEAD (SoC frozen). ***"
      log "*** LAST heartbeat iter=$last_iter, LAST completed node index=$last_node (wedge is at NEXT node). ***"
      log "*** Board needs human reboot; clear /tmp/imcflow_user.lock afterwards. ***"; exit 42
    fi; sleep "$INTERVAL"; continue
  fi
  fail=0; sig="${out#alive;}"; IFS=';' read -r e it mx cnt <<<"$sig"
  [[ "$it" != "-" ]] && last_iter="$it"; [[ "$mx" != "-" ]] && last_node="$mx"
  if [[ "$sig" != "$last_sig" ]]; then last_sig="$sig"; last_change=$now
    log "progress: hb_iter=$it node_max=$mx node_count=$cnt (advancing)"
  else st=$((now-last_change))
    if [[ $st -ge $STALL_SECS ]]; then
      log "*** WEDGE SUSPECTED: no progress for ${st}s (>=${STALL_SECS}s), ssh still alive. ***"
      log "*** STALLED at hb_iter=$it, LAST completed node index=$mx (count=$cnt). Wedge at NEXT node. Watch for SSH death. ***"
      last_change=$now
    fi
  fi
  sleep "$INTERVAL"
done
