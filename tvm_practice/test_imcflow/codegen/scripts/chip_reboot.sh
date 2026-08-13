#!/usr/bin/env bash
# B2 (petalinux2, 147.46.117.99) 원격 재부팅: measurement2의 UART(/dev/ttyUSB8) 경유.
# 콘솔 프롬프트 살아있으면 login/reboot, 죽어있으면 sysrq BREAK+b 폴백.
# 사용: scripts/chip_reboot.sh   (부팅 복귀까지 대기, 최대 ~5분)
set -u
echo "[chip_reboot] UART reboot via measurement2:/dev/ttyUSB8 ..."
timeout 60 ssh -o BatchMode=yes measurement2 'python3 ~/chip_uart_reboot.py' 2>&1 | sed 's/^/[uart] /'
echo "[chip_reboot] waiting for SSH to come back ..."
for i in $(seq 1 30); do
  if timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no -p 1326 root@147.46.117.99 'echo READY' 2>/dev/null | grep -q READY; then
    echo "[chip_reboot] BOARD BACK (attempt $i)"
    timeout 12 ssh -o BatchMode=yes -p 1326 root@147.46.117.99 'uptime; rm -f /tmp/imcflow_user.lock && echo LOCK_CLEARED'
    exit 0
  fi
  sleep 10
done
echo "[chip_reboot] STILL DOWN after ~5min — sysrq도 안 먹는 hard wedge일 수 있음 (물리 전원 필요)"
exit 1
