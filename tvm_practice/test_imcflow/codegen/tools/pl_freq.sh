#!/bin/bash
# PL (accelerator) clock control on petalinux2 via CRL_APB PL0_REF_CTRL (0xFF5E00C0).
#
#   tools/pl_freq.sh get           # decode current divisors + effective MHz
#   tools/pl_freq.sh set <MHz>     # e.g. 100, 50, 25 (DIV0 = round(1499.85/MHz), DIV1 kept)
#
# Register layout (Zynq US+ CRL_APB): [24] CLKACT, [21:16] DIVISOR1, [13:8] DIVISOR0,
# [2:0] SRCSEL (0 = IOPLL @ 1499.85 MHz on this board).
# Change the divisor only while the accelerator is IDLE (no SET_RUN in flight).
set -e
SSH_DEST=${PLFREQ_SSH:-"-o BatchMode=yes -p 1326 root@147.46.117.99"}
REG=0xFF5E00C0
PARENT_HZ=1499850000

board() { ssh $SSH_DEST "$@"; }

decode() {
  local v=$1
  local div0=$(( (v >> 8) & 0x3F ))
  local div1=$(( (v >> 16) & 0x3F ))
  local hz=$(( PARENT_HZ / (div0 * div1) ))
  echo "raw=$(printf 0x%08X $v) DIV0=$div0 DIV1=$div1 -> $((hz/1000000)).$(( (hz/10000)%100 )) MHz"
}

case "$1" in
  get)
    v=$(board "devmem $REG"); v=$((v))
    decode $v
    # NOTE: clk_summary shows the kernel CCF's CACHED rate -- devmem writes bypass
    # CCF, so after `set` it goes stale. Trust the register decode; verify
    # physically (idle current drops, kernel pulse width scales 1/f).
    board "grep -E ' pl0_ref ' /sys/kernel/debug/clk/clk_summary" | awk '{print "clk_summary(CCF cache):", $5/1000000, "MHz"}'
    ;;
  set)
    mhz=$2
    [ -n "$mhz" ] || { echo "usage: pl_freq.sh set <MHz>"; exit 1; }
    div0=$(( (PARENT_HZ + mhz*500000) / (mhz*1000000) ))   # round
    [ $div0 -ge 1 ] && [ $div0 -le 63 ] || { echo "DIV0 $div0 out of range"; exit 1; }
    v=$(board "devmem $REG"); v=$((v))
    new=$(( (v & ~0x3F00) | (div0 << 8) ))
    echo "before: $(decode $v)"
    board "devmem $REG 32 $(printf 0x%08X $new)"
    rb=$(board "devmem $REG"); rb=$((rb))
    echo "after : $(decode $rb)"
    [ $rb -eq $new ] || { echo "READBACK MISMATCH"; exit 1; }
    ;;
  *) echo "usage: pl_freq.sh {get|set <MHz>}"; exit 1;;
esac
