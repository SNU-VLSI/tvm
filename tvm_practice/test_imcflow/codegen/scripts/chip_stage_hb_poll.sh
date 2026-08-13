#!/usr/bin/env bash
# Poll /var/volatile/imcflow_stage_hb.txt on the B2 board every second, recording the
# LAST stage line seen on the HOST side (survives a SoC hard-wedge). Companion to
# chip_wedge_monitor.sh for the IMCFLOW_STAGE_HB kernel-entry localizer.
# Usage: chip_stage_hb_poll.sh --log <abs_path> [--int 1]
set -u
HOST="root@147.46.117.99"; PORT=1326; HBFILE="/var/volatile/imcflow_stage_hb.txt"
LOG="/tmp/chip_stage_hb.log"; INT=1
while [ $# -gt 0 ]; do case "$1" in
  --log) LOG="$2"; shift 2;; --int) INT="$2"; shift 2;; *) shift;; esac; done
echo "[$(date '+%F %T')] === stage_hb_poll start (file=$HBFILE) ===" | tee -a "$LOG"
last=""
while true; do
  cur=$(timeout 6 ssh -o BatchMode=yes -o ConnectTimeout=5 -p "$PORT" "$HOST" "tail -1 $HBFILE 2>/dev/null" 2>/dev/null)
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date '+%F %T')] SSH-FAIL (board busy/wedged) — last stage seen: '${last:-<none>}'" | tee -a "$LOG"
  elif [ -n "$cur" ] && [ "$cur" != "$last" ]; then
    echo "[$(date '+%F %T')] STAGE: $cur" | tee -a "$LOG"; last="$cur"
  fi
  sleep "$INT"
done
