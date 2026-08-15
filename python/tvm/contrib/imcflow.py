# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from typing import Tuple, List, Dict, Union
from enum import Enum
from collections import defaultdict, UserDict
import copy

import re
import math
import os
import json

import tvm
from tvm import relay
from tvm.relay.op.contrib.imcflow import CustomIDToNode

SMALL_DEBUG = 0

BIG_IMEM = os.getenv("IMCFLOW_BIG_IMEM", "0") == "1"


# ---------------------------------------------------------------------------
# Master BUGFIX knob (IMCFLOW_BUGFIX = on | off, default off)
# ---------------------------------------------------------------------------
# ONE knob that switches the whole imcflow codegen flow between two behaviors:
#
#   * knob=off (DEFAULT): current-HEAD behavior = the 934 + P0-P3 + P4 NoC-sync
#     code path (deadlock-free codegen that passes the BUGFIX-off RTL). This is
#     the chip_acc_measure behavior we ship today. The shared RTL runner builds
#     without the BUGFIX_* RTL defines.
#
#   * knob=on: fall back to the merge-base a8af0e4cf behavior (the "BUGFIX-on
#     golden") -- the 934/P0-P4 rendezvous + barrier sync emission is turned OFF
#     and codegen reproduces a8af byte-for-byte. The shared RTL runner adds the
#     BUGFIX_* RTL defines.
#
# `bugfix_off_mode() == True`  => knob is OFF => emit the 934 + P4 NoC sync
#                                 (the default BUGFIX-off RTL co-sim path).
# `bugfix_off_mode() == False` => knob is ON  => a8af fallback (no new sync).
#
# DEFAULT is OFF, matching the Makefile. Set IMCFLOW_BUGFIX=on explicitly to
# reproduce the a8af behavior and compile RTL with the BUGFIX_* defines. See
# DWCONV_SYNC_GRANULARITY_DESIGN.md.
#
# Mirror of the existing _is_multl_swfix_enabled() precedent in
# relay/backend/contrib/imcflow/imce_codeblock.py. Import this helper at every
# sync call site instead of scattering raw os.environ.get(...) reads.
def get_imcflow_bugfix_mode() -> str:
  """Return the validated ``IMCFLOW_BUGFIX`` mode (default: ``off``)."""
  mode = os.environ.get("IMCFLOW_BUGFIX", "off").strip().lower()
  if mode not in ("on", "off"):
    raise ValueError(
      f"IMCFLOW_BUGFIX must be 'on' or 'off', got {os.environ.get('IMCFLOW_BUGFIX')!r}"
    )
  return mode


def bugfix_off_mode() -> bool:
  """Return True when IMCFLOW_BUGFIX=off -> emit 934+P4 NoC sync.

  Unset defaults to ``off``. Return False for explicit ``on`` -> reproduce a8af
  (no new sync emission), i.e. pristine chip_acc_measure behavior.
  """
  return get_imcflow_bugfix_mode() == "off"


# IMCE-packing lever (IMCFLOW_PACK_BN_MINMAX): fold a conv's consumer-side
# BN (imcflow.fused_batch_norm) and min_max_quantize into the qconv composite
# so they render as same-IMCE post_ops instead of a dedicated preop-minmax /
# vecops IMCE (see BN_MINMAX_PACKING_HANDOFF.md). Gates ONLY the two
# make_postop_pattern_start_with chains in relay/op/contrib/imcflow.py.
# merge_composite_ops runs AFTER split_conv_to_atomic, so the chain matches
# per-atom convs: unsplit convs and the IC-split psum-merge atom (BN after the
# complete psum add) fuse; OC-split BN sits after concat (a pattern terminal)
# and is never absorbed. Default OFF -> byte-identical to stock. Defined in
# relay/op/contrib/imcflow.py (this module imports it there) because the
# pattern tables are built at that module's import time.
from tvm.relay.op.contrib.imcflow import pack_bn_minmax_mode  # noqa: F401


def feed_spread_n() -> int:
  """Max-throughput lever: spread the conv activation feed across N RECV FIFOs.

  The conv activation feed (inode->imce data input for imcflow_qconv/qdwconv)
  is normally pinned to RECV fifo 0 (a single depth-2 FIFO). The INODE send-fifo
  then PUSH_STALLs because the IMCE drains one word per LOAD_LB through that one
  depth-2 fifo -> the ~3cyc/word pacing that dominates the per-STEP feed tail.

  When IMCFLOW_FEED_SPREAD=N (N in 2..8, default 4 when set to a non-numeric
  truthy value), the 4 bitplanes of each pixel are round-robined across RECV
  fifos 0..N-1 instead of all landing in fifo 0. This is HARDWARE-CORRECT because
  the linebuffer assembles bitplanes purely by LOAD_LB *issue order* (a modN
  input-handshake counter in addr_shfl_gen.sv), NOT by source fifo; the packet
  carries the fifo_id (fifo_block.sv push_id) so a SEND(...,fid=N) lands in RECV
  fifo N and a subsequent LOAD_LB(N) pops it; and backpressure is strictly
  per-fifo, so up to N depth-2 fifos (2*N words) can be resident at once,
  relieving the depth-2 drain. The global SETFLAG/STANDBY window (a separate flag
  register file, imce_ctrl.sv) is orthogonal to fifo selection, so the per-pixel
  window still bounds the whole 4-bitplane burst.

  Returns the spread width N (>=2) when enabled, else 0 (feature off ->
  byte-identical to the pinned-fifo-0 behavior). Default OFF: unset env var
  keeps SINGLE / non-QUADRU paths byte-identical to stock.
  """
  raw = os.environ.get("IMCFLOW_FEED_SPREAD", "")
  if not raw:
    return 0
  raw = raw.strip().lower()
  if raw in ("0", "off", "false", "no"):
    return 0
  try:
    n = int(raw)
  except ValueError:
    # any other truthy value -> default spread of 4 (the 4-bitplane option (a))
    n = 4
  if n <= 1:
    return 0
  if n > 8:
    n = 8  # HW has only 8 RECV FIFOs
  return n


def drop_psum_send() -> bool:
  """Max-throughput lever (IMCFLOW_DROP_PSUM): DON'T-CARE output mode. Drop the
  per-pixel psum drain (GET_CREG + IMCE_SEND) after each STEP so the NEXT pixel's
  LOAD_LB / OUTPUT_HS can issue earlier. Only legal when conv correctness is
  irrelevant (garbage output). Default OFF -> byte-identical to stock.

  RTL caveat (CHIP / BUGFIX-off build, imcu_ctrl.sv:69
  `core_rx.ready = core_ready && core_tx.ready`): on the TAPED-OUT chip the
  BUGFIX_STEP macro is NOT defined, so the crossbar input fetch is gated on the
  post_imcu OUTPUT side being ready. The out_fifo is drained by the psum NoC
  SEND being consumed (an inode RECV), NOT by OP_STEP. Dropping ALL psum SENDs
  therefore leaves the depth-32 out_fifo undrained; once it fills (~a handful of
  STEPs) core_tx.ready deasserts, the feed stalls, the array stops converting,
  and DDA/ADC current goes flat. RTL-proven (2026-08-15, one_1x1_quant bugfix-off,
  SRAM_BACKDOOR=0): DROP_PSUM=1 keep=0 X-fatals on imce_intf_tx after 4 STEP;
  no-drop completes 64/64 STEP+ADC. So on chip a bounded keep (drop_psum_keep_every,
  K <= 32 fifo depth) is REQUIRED, not optional. The earlier "OP_STEP drains the
  out_fifo" reasoning holds ONLY under the BUGFIX_STEP build
  (`core_rx.ready = core_ready && !core_tx.valid`), which the chip lacks."""
  return os.environ.get("IMCFLOW_DROP_PSUM", "").strip().lower() in ("1", "on", "true", "yes")


def step_freerun_n() -> int:
  """IMCFLOW_STEP_FREERUN (power-measurement, DON'T-CARE output): repeat the fed
  conv STEP body this many EXTRA times so the crossbar free-runs for a bounded,
  wall-clock-sized burst that a bench DMM (e.g. Keysight 34410A) can integrate.
  0/unset -> byte-identical. N>=1 -> the whole conv body runs (1+N) times.

  MUST be a FED repeat (a LOAD_LB per STEP), NOT a bare STEP-repeat: a stale
  linebuffer supplies only a finite set of pad windows then STALLs (all_recived=1,
  addr_shfl_gen.sv:178/236), and under QUADRU the first zero-feed STEP WEDGEs (the
  reason K=1 was chosen). Repeating the fed K=1 body keeps every STEP fed ->
  QUADRU-safe. Emitted via SimpleFor -> clang -force-hardware-loops -> a real imem
  backward-branch loop (hw_loop.sv, 6 nested levels x 14-bit = 16384 each), so
  millions of STEPs cost NO extra imem. Both the imce LOAD_LB+STEP loop AND the
  matching inode activation-feed loop must scale by the SAME N (producer/consumer
  packet balance on the depth-2 NoC fifo). Pair with IMCFLOW_DROP_PSUM=1 so there
  is no psum/output back-channel to balance. Keep N FINITE so STATE returns to IDLE
  and the chip lock releases. At 100MHz, ~22 cyc/STEP -> ~4.5M STEPs ~= 1 s."""
  try:
    return max(0, int(os.environ.get("IMCFLOW_STEP_FREERUN", "0")))
  except ValueError:
    return 0


def step_freerun_factors(reps: int):
  """Split a STEP_FREERUN repeat count into nested-loop factors each <= 16384, the
  IMCE hardware-loop counter limit (IMCE_LOOP_LEN=14 bits -> 2^14). A single
  SimpleFor(reps) with reps>16384 makes clang -force-hardware-loops emit a counter
  that overflows -> clang fatal error. So emit nested SimpleFors whose product
  >= reps. Returns a list of factors (outer..inner). For reps<=16384 -> [reps].
  For larger, [ceil(reps/16384), 16384] (product may slightly exceed reps; the
  extra STEPs only lengthen the power-measurement burst, which is harmless)."""
  LIMIT = 16384
  if reps <= LIMIT:
    return [reps]
  outer = -(-reps // LIMIT)  # ceil
  # keep nesting if outer itself exceeds the limit (reps up to 16384^2 = 268M)
  if outer <= LIMIT:
    return [outer, LIMIT]
  return step_freerun_factors(outer) + [LIMIT]


def step_freerun_wall_sec() -> int:
  """Wall-clock ceiling (seconds) for the STEP_FREERUN completion wait
  (IMCFLOW_STEP_FREERUN_WALL, default 3). Raise it when the free-run loop needs
  longer than 3s to reach OP_HALT so the host blocks until the array is genuinely
  IDLE and reports the EXACT held-busy time (=> exact silicon cyc/STEP), instead of
  bailing early as 'measurement incomplete'. Only consulted when STEP_FREERUN>0."""
  try:
    return max(1, int(os.environ.get("IMCFLOW_STEP_FREERUN_WALL", "3")))
  except ValueError:
    return 3


def step_freerun_hold_sec() -> int:
  """IMCFLOW_STEP_FREERUN_HOLD_SEC (0=off, default). When >0, the freerun wait
  BUSY-HOLDS the host for exactly this many seconds after SET_RUN, IGNORING STATE,
  so a bench DMM integrates a deterministic array-active window and the outer eval
  loop cannot re-fire the kernel mid-measurement. STATE=0x1(S_RUN) alone does NOT
  prove the crossbar is converting (an inode can sit in S_RUN while STANDBY-stalled,
  no STEP retiring, no ADC current); the fixed hold gives a clean window and the DMM
  (DDA/ADC rail) is the discriminator -- DDA rises => STEPs retiring, DDA flat =>
  rendezvous slipped. Pair with a HUGE IMCFLOW_STEP_FREERUN so the fed STEP loop
  out-lasts the hold. 0 -> byte-identical (original busy-first wait)."""
  try:
    return max(0, int(os.environ.get("IMCFLOW_STEP_FREERUN_HOLD_SEC", "0")))
  except ValueError:
    return 0


def drop_output_readback() -> bool:
  """Design A host-side half of IMCFLOW_DROP_PSUM (DON'T-CARE output). When the
  psum SEND (imce) and INODE_RECV collector loop (inode) are dropped, the func_out0
  region (inode_3_0's local NoC-node data mem @ 0x31080) is never written and its
  port never returns to a host-readable idle -> a host MMIO read of it HANGS THE
  BUS on silicon. So under DROP_PSUM the host must ALSO skip the func_out* read-back
  loop (generateFromNpuTransferCode). Gated by the same env so the three drops stay
  a matched set; default OFF -> byte-identical (host still reads real output)."""
  return drop_psum_send()


def drop_psum_keep_every() -> int:
  """When IMCFLOW_DROP_PSUM is on, still emit ONE psum drain (GET_CREG + IMCE_SEND
  on imce, and the matching INODE_RECV) every K pixels so the depth-32 post_imcu
  out_fifo is drained at least every K STEPs and core_tx.ready never sticks low
  (see drop_psum_send). REQUIRED on the taped-out chip (BUGFIX-off), where the
  out_fifo is NOT drained by OP_STEP. K MUST be <= 32 (out_fifo depth; the safe
  bound is fifo_depth minus in-flight packets, so K<=8..16 has margin). K=0
  (default) -> drop every SEND: only safe when the whole kernel's STEP count is
  bounded < fifo depth (it is NOT for a normal conv / free-run) -> will wedge on
  chip. K>=1 -> keep 1 drain per K pixels. Both imce (keep_psum_pixel) and inode
  (matching RECV keep) use the SAME K so producer/consumer stay balanced."""
  raw = os.environ.get("IMCFLOW_DROP_PSUM_KEEP", "").strip()
  if not raw:
    return 0
  try:
    return max(0, int(raw))
  except ValueError:
    return 0


# NOTE: K-keep is implemented STRUCTURALLY (nested counted loops that split the pixel
# loop into keep/skip tiers -- imce_codeblock ConvBlock._build_structure and inode
# RecvBlock._build_tiled), NOT with a runtime `if (ctr % K)`: the IMCE and INODE LLVM
# targets only support counted hardware loops (-force-hardware-loops), so a
# data-dependent branch (`br_cc`) or `/K` (`sra`) is un-selectable ("Cannot select").


def serialize_imcu_load() -> bool:
  """Silicon-SAFE lever (IMCFLOW_SERIALIZE_IMCU, default OFF).

  ResNet8 region3 wedges the real B2 chip (SoC SSH-dead at region3 kernel entry,
  before the first input transfer) while passing the BUGFIX-off RTL. Root cause
  localized by the subset ladder: subset21 (only inode_2_0 streams WR_IMCU, even
  though it streams 2 bursts back-to-back) PASSES; subset22 (all 4 inodes stream
  WR_IMCU concurrently after the shared 255-barrier) WEDGES. So the trigger is
  4-inode-CONCURRENT IMCU weight-load bursts hammering the NoC / IMCU write path
  at silicon speed. The idealized RTL timing tolerates it; the silicon does not.

  When ON, codegen.py.initialize() serializes the IMCU-write phase: it inserts a
  255 barrier (SyncAllINodes, the proven inter-inode primitive) BEFORE each
  inode's WriteIMCUBlock in a fixed inode order, so at any instant only one inode
  streams a WR_IMCU burst instead of all four at once. Turns concurrent bursts
  into sequential ones -> removes the silicon NoC/IMCU write-contention wedge.

  Default OFF -> byte-identical to stock (no new blocks emitted). Independent of
  IMCFLOW_BUGFIX; safe to set alongside IMCFLOW_BUGFIX=off.

  ROUND-2 UPDATE (chip result): the inter-inode serialization got the chip through
  region1+region2 but region3 STILL wedged at kernel entry (same signature as
  stock). So inter-inode concurrency is NOT the region3 mechanism. The residual
  differentiator is inode_3_0's TWO consecutive WR_IMCU bursts (512 words
  back-to-back). See imcu_intra_drain_nops() (IMCFLOW_IMCU_INTRA_DRAIN) -- the
  preferred region3 fix -- which drains between an inode's back-to-back bursts
  WITHOUT any inter-inode handshake (so it cannot introduce the rendezvous
  deadlock that the barrier variant caused in region2). This IMCFLOW_SERIALIZE_IMCU
  barrier variant is kept switchable so drain+barrier can be combined if drain
  alone is insufficient."""
  return os.environ.get("IMCFLOW_SERIALIZE_IMCU", "").strip().lower() in (
      "1", "on", "true", "yes")


def imcu_intra_drain_nops() -> int:
  """Silicon-SAFE lever (IMCFLOW_IMCU_INTRA_DRAIN, integer nop count, default 0=OFF).

  ResNet8 region3 wedges the real B2 chip at region3 kernel entry (INIT phase,
  during the IMCU weight-load) while the BUGFIX-off RTL passes. Chip-ladder +
  round-2 evidence narrows the trigger to a SINGLE inode streaming TWO consecutive
  256-word WR_IMCU bursts back-to-back (inode_3_0 in region3: -> imce col 2 then
  col 4). Inter-inode serialization (IMCFLOW_SERIALIZE_IMCU) did NOT fix it;
  neither concurrency alone (region2 = 4x single burst PASS) nor a lone double
  burst under light init (subset21 PASS) wedges -- it is the back-to-back double
  burst under region3's heavy 4-inode init that overruns the IMCU write path on
  silicon.

  When >0, WriteIMCUBlock._build() inserts a NOP-delay loop of this many nops
  BETWEEN consecutive WR_IMCU burst loops within one inode (i.e. only when an
  inode has >=2 weight blocks), so the first burst fully commits to the IMCU
  before the second starts. The NOP-delay (NopLoopBlock) is the same proven,
  self-contained timing primitive used for the single_qconv nop_delay -- it is
  purely local to the inode (no NoC handshake), so unlike the 255-barrier variant
  it cannot deadlock a rendezvous.

  Value = nop count. A 256-word burst issues in ~256 inode cycles; default
  suggestion 256 gives a full-burst drain margin. 0 (default) -> byte-identical to
  stock (no nops emitted). Independent of IMCFLOW_BUGFIX and IMCFLOW_SERIALIZE_IMCU
  (both can be combined)."""
  raw = os.environ.get("IMCFLOW_IMCU_INTRA_DRAIN", "").strip()
  if not raw:
    return 0
  try:
    return max(0, int(raw))
  except ValueError:
    return 0


def mmio_block_barrier_usec() -> int:
  """Silicon-SAFE HOST-SIDE lever (IMCFLOW_MMIO_BARRIER, default -1 = OFF).

  ROOT CAUSE (round-4 localizer result): ResNet8 region3 wedges the real B2 chip
  NOT in the inode program but on the HOST side, while streaming the region's
  ~61 back-to-back block transfers (inode/imce imem+policy + const weights) into
  the accelerator as tight non-volatile MMIO stores `npu_pointer[..]=..`. The
  IMCFLOW_STAGE_HB localizer PROVED this: stock codegen (all inode/imce blobs
  byte-identical, all four codegen-sync fix attempts irrelevant) + ONLY host-side
  per-block fsync heartbeats -> subset22 PASSES region3 end-to-end. The fsync's
  syscall (kernel entry drains the CPU store buffer + yields) between block
  transfers is the accidental fix; without it the un-ordered/buffered MMIO store
  stream overruns the SoC<->accelerator bus and wedges (region3 = largest xfer).

  When >= 0, generateToNpuTransferCode() emits a `__sync_synchronize()` full
  memory barrier AFTER each block's transfer loop (drains the store buffer, no
  syscall), plus a `usleep(<value>)` when value > 0 (matches the fsync's
  CPU-yield / real-time drain if the bare barrier is insufficient). value == 0 ->
  barrier only. value < 0 (default, env unset) -> emit nothing -> byte-identical
  to stock. Host-side ONLY: accelerator (inode/imce) blobs are untouched, so no
  RTL rerun is required. Independent of all codegen-sync levers."""
  raw = os.environ.get("IMCFLOW_MMIO_BARRIER", "").strip()
  if not raw:
    return -1
  try:
    return int(raw)
  except ValueError:
    return -1


def multiblock_fusedadd_bare() -> bool:
  """Silicon-deadlock lever (IMCFLOW_MULTIBLOCK_FUSEDADD_BARE, default OFF).

  Removes the inode->imce data-input rendezvous (the Fix-D merged
  SETFLAG(1);STANDBY(inode,1);...;SETFLAG(0) consumer window AND the matching
  inode-side 4-phase _get_presend_sync_code_str per-word handshake) for a
  MULTI-BLOCK (num_blocks > 1) 2-inode fused-add consumer -- e.g. ResNet8
  region3 imce_1_1 (lhs from inode_0_0, rhs from inode_1_0, num_blocks==2).

  WHY: that consumer wraps EACH block's RECV pair in its own window (Fix E
  per-block re-emission), so the node's SINGLE flag register (imce_ctrl.sv)
  toggles 1->0->1->0 TWICE per loop iteration while TWO independent producer
  inodes each re-arm a 4-phase STANDBY/SETFLAG gate PER WORD on that same flag.
  On the BUGFIX-off *simv* the fixed NoC latency keeps the two producers'
  arrivals ordered so the level handshake resolves; on real silicon the two
  producers race and the consumer's flag edge (1->0->1) can be collapsed/missed
  by a producer polling at silicon speed -> a STANDBY(6,0)/STANDBY(6,1) never
  fires -> wedge at region3 entry (localized: chip-v1-region3-fusedadd-wedge).

  Correctness after removal relies on FIFO backpressure (DESIGN baseline: the
  standalone RecvSendWrapper is bare RECV/compute/SEND). Each producer->consumer
  edge is a SINGLE dedicated RECV fifo (lhs=fifo2, rhs=fifo3) with SEND count ==
  RECV count (same tile loop bound), so the depth-2 backpressured fifo already
  guarantees lossless, in-order delivery -- the rendezvous was added only to
  satisfy simv pacing. region2 imce_1_3 (the other Fix-D fused-add) is
  num_blocks==1 (single window per iter, no mid-iteration toggle) so it is
  EXCLUDED and unaffected -> its RTL sync is preserved.

  Default OFF -> byte-identical to the P4 output (RTL/everything unchanged);
  opt in only for the silicon fused-add probe.
  """
  return os.environ.get(
      "IMCFLOW_MULTIBLOCK_FUSEDADD_BARE", "").strip().lower() in (
      "1", "on", "true", "yes")


# Silicon-SAFE fused-add rendezvous: the monotonic phase-token base value. Both
# the consumer window (imce_codeblock RecvSendWrapper) and the producer pre-send
# (inode_codeblock SendBlock) MUST use this identical base so their STANDBY/
# SETFLAG values match. Base 3 skips the reserved consumer-flag values on the
# fused-add node: 0 (idle), 1 (legacy input invite), 2 (OUTPUT multicast barrier
# that downstream imce receivers STANDBY on). block b -> {base+2b, base+2b+1};
# num_blocks<=8 -> max base+2*7+1 = 18 < 255 (SYNC_REG_WIDTH=8).
SAFE_TOKEN_BASE = 3


def multiblock_fusedadd_safe() -> bool:
  """Silicon-SAFE fused-add rendezvous REDESIGN (IMCFLOW_MULTIBLOCK_FUSEDADD_SAFE,
  default OFF). Replaces -- does NOT remove -- the 2-inode fused-add consumer's
  rendezvous with a monotonic phase-token, fully-interlocked, order-independent
  handshake that survives BOTH the BUGFIX-off simv AND real silicon.

  Target: the 2-producer fused-add consumer (ResNet8 region3: lhs from inode_0_0
  fifo2, rhs from inode_1_0 fifo3). In v1-multicore subset the consumer is
  imce_1_1 emitted via the standalone-VecBlock (call_created_loop) path with BARE
  RECV/SEND (no pacing at all); in the full-region3 handcraft topology it is
  imce_0_2 with the old per-word 1->0->1->0 4-phase toggle. BOTH are unsafe on
  silicon (fsim-proven edge-collapse: a producer drives the consumer's single
  8-bit level flag 1->0 ~167us before the back-pressured consumer arms its
  matching STANDBY -> awaited value gone -> wedge state 0x1).

  REDESIGN (per outer iter, per block b in 0..num_blocks-1):
    consumer C:  SETFLAG(2b+1); STANDBY(P0,2b+1); STANDBY(P1,2b+1);
                 SETFLAG(2b+2); RECV(lhs_b); RECV(rhs_b)
    producer Pk: STANDBY(C,2b+1); SETFLAG(2b+1); STANDBY(C,2b+2); SEND(word_b)
  Every flag VALUE is unique within the live window (no repeated 1/0 toggle ->
  no edge aliasing). Every write is INTERLOCKED: C writes 2b+2 only after it saw
  BOTH producers' 2b+1, then BLOCKS on RECV (which cannot complete until each Pk
  SENDs, i.e. observed 2b+2) so C never overwrites FC past 2b+2 before both
  producers consumed it -> no lost wakeup, ORDER-INDEPENDENT (either producer may
  arrive first; its level simply waits). See DESIGN_region3_fusedadd_redesign.md.

  Token width: values 2b+1,2b+2 with num_blocks<=8 -> max 16, well under the
  8-bit (0..255) SYNC_REG_WIDTH; reused each outer iteration (uniqueness only
  needed within the interlock-bounded one-iteration window).

  Mutually exclusive with the (dead-end) _BARE lever. Default OFF -> baseline
  P0-P4 / standalone path emitted byte-identically; opt in only for the silicon
  fused-add probe after the RTL regression passes region3.
  """
  return os.environ.get(
      "IMCFLOW_MULTIBLOCK_FUSEDADD_SAFE", "").strip().lower() in (
      "1", "on", "true", "yes")


def feed_prefetch_n() -> int:
  """Max-throughput lever (IMCFLOW_FEED_PREFETCH): reserved gate for extending
  the feed spread to 8 fifos AND pre-sending P pixels ahead so the next pixel's
  bitplanes are resident during the current compute. Returns P (pixels ahead),
  0 = off. Spread width itself is still IMCFLOW_FEED_SPREAD."""
  raw = os.environ.get("IMCFLOW_FEED_PREFETCH", "").strip()
  if not raw:
    return 0
  try:
    return max(0, int(raw))
  except ValueError:
    return 0


def overflow_sw_default_on() -> bool:
  """Default for the SEPARATE IMCFLOW_BUGFIX_OVERFLOW_SW knob, coupled to the
  master knob: the BUGFIX-off RTL lacks the HW overflow fix, so codegen should
  SW-compensate when the master knob is off. The explicit
  IMCFLOW_BUGFIX_OVERFLOW_SW env var still overrides this default (see
  _is_multl_swfix_enabled)."""
  return bugfix_off_mode()

class NodeID(Enum):
  inode_0_0 = 0
  imce_0_1 = 1
  imce_0_2 = 2
  imce_0_3 = 3
  imce_0_4 = 4
  inode_1_0 = 5
  imce_1_1 = 6
  imce_1_2 = 7
  imce_1_3 = 8
  imce_1_4 = 9
  inode_2_0 = 10
  imce_2_1 = 11
  imce_2_2 = 12
  imce_2_3 = 13
  imce_2_4 = 14
  inode_3_0 = 15
  imce_3_1 = 16
  imce_3_2 = 17
  imce_3_3 = 18
  imce_3_4 = 19

  @staticmethod
  def from_coord(x: int, y: int) -> 'NodeID':
    """Returns the NodeID corresponding to a 2D coordinate."""
    value = x * ImcflowDeviceConfig.NODE_COL_NUM + y
    for node in NodeID:
      if node.value == value:
        return node
    raise ValueError(f"No Node found for coordinate ({x}, {y})")

  @staticmethod
  def from_inode_coord(x: int) -> 'NodeID':
    assert x >= 0 and x < ImcflowDeviceConfig.INODE_NUM, "inode coord is out of range"
    return NodeID(ImcflowDeviceConfig.NODE_COL_NUM*x)

  @staticmethod
  def from_imce_coord(x: int, y: Union[None | int] = None) -> 'NodeID':
    if y is None:
      ImceHeight = x//ImcflowDeviceConfig.IMCE_W_NUM
      ImceWidth = x % ImcflowDeviceConfig.IMCE_W_NUM
      return NodeID(ImcflowDeviceConfig.NODE_COL_NUM*ImceHeight + (ImceWidth+1))
    else:
      return NodeID(ImcflowDeviceConfig.NODE_COL_NUM*x + (y+1))

  @staticmethod
  def inodes() -> List['NodeID']:
    """Returns a list of all inode nodes."""
    return [node for node in NodeID if node.is_inode()]

  @staticmethod
  def imces() -> List['NodeID']:
    """Returns a list of all imce nodes."""
    return [node for node in NodeID if node.is_imce()]

  def is_inode(self) -> bool:
    return self.value % ImcflowDeviceConfig.NODE_COL_NUM == 0

  def is_imce(self) -> bool:
    return not self.is_inode()

  def to_coord(self, *args) -> Union[tuple, int]:
    """Converts this node to its 2D coordinate."""
    coord = divmod(self.value, ImcflowDeviceConfig.NODE_COL_NUM)
    if len(args) == 1 and args[0] == 0:
      return coord[0]
    elif len(args) == 1 and args[0] == 1:
      return coord[1]
    elif len(args) == 0:
      return coord
    else:
      raise ValueError("Invalid number of arguments")

  def slaves(self) -> List['NodeID']:
    """Returns a list of imces that are slaved to this inode."""
    assert self.is_inode(), "Only inode nodes have slaves"
    return [NodeID(self.value + i) for i in range(1, ImcflowDeviceConfig.NODE_COL_NUM)]

  def master(self) -> 'NodeID':
    """Returns the inode that is master to this imce."""
    assert self.is_imce(), "Only imce nodes have master"
    return NodeID(self.value - self.value % ImcflowDeviceConfig.NODE_COL_NUM)


class TensorID:
  _instances = {}

  def __new__(cls, graph_node_id: Union[int, Tuple], tensor_type: str):
    key = (graph_node_id, tensor_type)
    valid_pattern = r"^(data|odata|weight|scale|bias|fused_scale|fused_bias|lhs|rhs|min|max|threshold|zero|config|var|func_out.*)$"
    if not re.match(valid_pattern, tensor_type):
      print(f"Invalid tensor type: {tensor_type}")
    if key not in cls._instances:
      instance = super(TensorID, cls).__new__(cls)
      cls._instances[(graph_node_id, tensor_type)] = instance
      instance.graph_node_id = graph_node_id
      instance.tensor_type = tensor_type

    return cls._instances[key]

  def inner_gid_match(self, graph_node_id: Union[int, Tuple]):
    import tvm
    if isinstance(self.graph_node_id, (int, tvm.tir.expr.IntImm)):
      return self.graph_node_id == graph_node_id
    if isinstance(self.graph_node_id, tuple):
      return graph_node_id == self.graph_node_id[1]
    print("Error in inner_gid_match")
    return False

  def __str__(self):
    return f"TensorID({self.graph_node_id}, {self.tensor_type})"

  def __repr__(self):
    return self.__str__()

  def __reduce__(self):
    """Support for pickling singleton instances"""
    return (self.__class__, (self.graph_node_id, self.tensor_type))


class TensorEdge:
  _instances = {}

  def __new__(cls, src_id: TensorID, dst_id: TensorID, split_idx: Union[None, int] = None):
    key = (src_id, dst_id, split_idx)
    if key not in cls._instances:
      instance = super(TensorEdge, cls).__new__(cls)
      cls._instances[(src_id, dst_id, split_idx)] = instance
      instance.src_id = src_id
      instance.dst_id = dst_id
      instance.split_idx = split_idx

    return cls._instances[key]

  def src_inner_gid_match(self, graph_node_id: Union[int, Tuple]):
    return self.src_id.inner_gid_match(graph_node_id)

  def dst_inner_gid_match(self, graph_node_id: Union[int, Tuple]):
    return self.dst_id.inner_gid_match(graph_node_id)
  
  def simple_name(self):
    format_gid = lambda gid: f"{gid}" if not isinstance(gid, Tuple) else f"{gid[0]}_{gid[1]}"

    name=f"s{format_gid(self.src_id.graph_node_id)}_d{format_gid(self.dst_id.graph_node_id)}{self.dst_id.tensor_type}"
    if self.split_idx is not None:
      name += f"_split{self.split_idx}"
    name = name.replace("-", "m")
    return name

  def __str__(self):
    if self.split_idx is None:
      return f"TensorEdge(({self.src_id.graph_node_id}, {self.src_id.tensor_type}), ({self.dst_id.graph_node_id}, {self.dst_id.tensor_type}))"
    else:
      return f"TensorEdge(({self.src_id.graph_node_id}, {self.src_id.tensor_type}), ({self.dst_id.graph_node_id}, {self.dst_id.tensor_type}), {self.split_idx})"

  def __repr__(self):
    return self.__str__()

  def __reduce__(self):
    """Support for pickling singleton instances"""
    return (self.__class__, (self.src_id, self.dst_id, self.split_idx))


class FunctionInfo:
  """Information about an IMCFlow function"""
  def __init__(self, func_node, tiling_factor=1):
    self.func_node = func_node          # relay.Function object
    self.tiling_factor = tiling_factor  # int: tiling factor for memory optimization
    self.const_name_map = {} # node id : name

  def __repr__(self):
    return f"FunctionInfo(tiling_factor={self.tiling_factor})"

class BlockTileInfo:
  """Tiling information for a data block"""
  def __init__(self):
    self.height_base_coords = [] # for tiling
    self.height_sizes       = [] # for tiling
    self.pkt_cnts           = [] # for tiling
    self.c_var_offsets      = [] # for tiling
    self.c_var_sizes        = [] # for tiling
  
  def set_info(self, height_base_coords: List[int]=None, 
                     height_sizes: List[int]=None, 
                     pkt_cnts: List[int]=None, 
                     c_var_offsets: List[int]=None,
                     c_var_sizes: List[int]=None):
    if height_base_coords is not None:
      self.height_base_coords = height_base_coords
    if height_sizes is not None:
      self.height_sizes = height_sizes
    if pkt_cnts is not None:
      self.pkt_cnts = pkt_cnts
    if c_var_offsets is not None:
      self.c_var_offsets = c_var_offsets
    if c_var_sizes is not None:
      self.c_var_sizes = c_var_sizes


class DataBlock:
  def __init__(self, id: Union[str, TensorEdge, List[TensorEdge]], size: int):
    """
    id can be either:
      str, used for "policy", "imem", etc
      TensorEdge, corresponding to a single edge (one dst)
      List[TensorEdge], corresponding to multiple edges (1< dst)
    kept internally as edges
    """
    self.edges = [id] if not isinstance(id, List) else id
    self.size = size
    self.offset = -1  # offset in the region
    self.base_address = -1  # base address in the device memory
    self.region_base_address = -1  # base address of the containing region
    self.tiling_info = None
  
  @property
  def id(self):
    if len(self.edges) == 1:
      return self.edges[0]
    else:
      return self.edges

  def set_size(self, size: int):
    if isinstance(size, float):
      assert size % 1 == 0, "Size should be an integer"
    self.size = int(size)

  def set_offset(self, offset: int):
    if isinstance(offset, float):
      assert offset % 1 == 0, "Offset should be an integer"
    self.offset = int(offset)

  def set_base_address(self, address: int, region_base_address: int = None):
    if isinstance(address, float):
      assert address % 1 == 0, "Address should be an integer"
    self.base_address = int(address)
    if region_base_address is not None:
      self.region_base_address = int(region_base_address)

  @property
  def rel_address(self) -> int:
    """Relative address within the containing region."""
    region_base = getattr(self, 'region_base_address', -1)
    if region_base == -1:
      return -1
    return self.base_address - region_base

  def __str__(self):
    return f"DataBlock({self.id}, size={self.size}, rel={self.rel_address}, addr={self.base_address})"

  def __repr__(self):
    return self.__str__()


class MemoryRegionEntry:
  def __init__(self, name: str, size: int):
    self.name = name
    self.size = size
    self.blocks = {}  # {data_block_name : DataBlock}
    self.base_address = -1  # offset in the device memory
    self._last_offset = 0  # last offset in the region

  def get_data_block_by_edge(self, edge):
    """
    Gets data_block by edge.
    """
    for block in self.blocks.values():
      if edge in block.edges:
        return block
    return None
  
  def _already_exists(self, block: DataBlock):
    if block.id in [x.id for x in self.blocks.values()]:
      print(f"Trying allocate {block} but skipped (already exists)")
      return True

    # extend existing Datablock's id when same src_id found (same data, different destinations)
    if isinstance(block.id, TensorEdge) and block.id.split_idx is None:
      for existing_block in self.blocks.values():
        # Check if any edge has the same src_id
        for edge in existing_block.edges:
          if isinstance(edge, TensorEdge) and edge.src_id == block.id.src_id:
            existing_block.edges.append(block.id)
            print(f"Extended existing block with {block.id}, now block.id = {existing_block.id}")
            return True
    return False

  def allocate(self, block: DataBlock) -> bool:
    """
    Allocate a data block in the region sequentially, assuming they are not deallocated
    returns True (when allocated, or already allocated) or False (exceeds region size)
    """
    if self._already_exists(block):
      return True

    print(f"Trying allocate {block}")
    # find first 32B aligned free offset
    aligned_offset = math.ceil((self.base_address + self._last_offset) / 32) * 32 - self.base_address
    if block.size + aligned_offset <= self.size:
      block.set_offset(aligned_offset)
      block.set_base_address(aligned_offset + self.base_address, region_base_address=self.base_address)
      self._last_offset = aligned_offset + block.size
      self.blocks[block.id] = block
      return True
    else:
      print(f"Data block size exceeds region size. {block.size} + {aligned_offset} > {self.size}")
      return False

  def set_base_address(self, address: int):
    self.base_address = int(address)

  def __str__(self):
    if not self.blocks:
      return f"MemoryRegionEntry({self.name}, {self.size}, {self.base_address}, blocks=[])"
    blocks_str = ",\n      ".join(str(block) for block in self.blocks.values())
    return (f"MemoryRegionEntry({self.name}, {self.size}, {self.base_address}, "
            f"blocks=[\n      {blocks_str}\n    ])")

  def __repr__(self):
    return self.__str__()


class MemoryRegion(UserDict):
  """
    MemoryRegion is a dict of MemoryRegionEntry accessible using key (idx).
    Automatically creates new MemoryRegionEntry instances from the template when accessing non-existent keys.
  """
  def __init__(self, region: MemoryRegionEntry):
    super().__init__()
    self._template_region = region

  @property
  def blocks(self):
    """Aggregate all blocks from all entries. Returns dict {block_id: DataBlock}"""
    aggregated = {}
    for idx, region in self.data.items():
      if region.blocks:  # Only include entries with blocks
        aggregated.update(region.blocks)
    return aggregated
  
  def allocate(self, block: DataBlock, phase: str):
    """Allocate block by finding first region of the phase with available space."""
    i = 0
    while not self[(phase, i)].allocate(block):
      raise RuntimeError(f"Failed to allocate block {block} in region {self._template_region.name} for phase {phase}")
      i += 1


  def __getitem__(self, key) -> 'MemoryRegionEntry':
    """auto-create from template"""
    if key not in self.data:
      return self.__missing__(key)
    return self.data[key]

  def __missing__(self, key) -> 'MemoryRegionEntry':
    """
    Called when a key is not found. Creates a new MemoryRegionEntry from the template.
    """
    # Create a deep copy of the template region
    new_region = copy.deepcopy(self._template_region)
    self[key] = new_region
    return new_region

  def __str__(self):
    if not self.data:
      return f"MemoryRegion('{self._template_region.name}', empty)"

    lines = [f"MemoryRegion('{self._template_region.name}', ["]
    for idx, region in self.data.items():
      lines.append(f"  {idx}: {region},")
    lines.append("])")
    return "\n".join(lines)

  def __repr__(self):
    return self.__str__()



class FuncMemoryLayout(UserDict):
  def __init__(self, *regions: MemoryRegionEntry):
    super().__init__()
    _last_end_address = 0

    for region in regions:
      self.data[region.name] = MemoryRegion(region)
      region.set_base_address(_last_end_address)
      _last_end_address += region.size

  @property
  def blocks(self):
    """Aggregate all blocks from all regions and entries. Returns dict {block_id: DataBlock}"""
    aggregated = {}
    for region_name, region_dict in self.data.items():
      region_blocks = region_dict.blocks
      if region_blocks:  # Only include regions with blocks
        aggregated.update(region_blocks)
    return aggregated

  def __getitem__(self, key: str) -> 'MemoryRegion':
    """Get a memory region by name (e.g., 'inode_0_0_data')"""
    return self.data[key]

  def get_data_block_by_edge(self, edge: Union[str, TensorEdge]):
    """
    Gets data_block by id from aggregated blocks
    """
    for block in self.blocks.values():
      if edge in block.edges:
        return block
    return None
  
  def get_data_block_by_id(self, block_id: Union[str, TensorEdge]):
    """
    Gets data_block by id from aggregated blocks
    """
    for block in self.blocks.values():
      if block_id == block.id:
        return block
    return None

  def __str__(self):
    regions_str = ",\n  ".join(str(region) for region in self.data.values())
    return f"FuncMemoryLayout(regions=[\n  {regions_str}\n])"

  def __repr__(self):
    return self.__str__()

class MemoryLayout(UserDict):
  """
    MemoryLayout is a dict of FuncMemoryLayout accessible using key (func_name).
    Automatically creates new FuncMemoryLayout instances from the template when accessing non-existent keys.
  """
  def __init__(self, *regions: MemoryRegionEntry):
    super().__init__()
    self._layout = FuncMemoryLayout(*regions)

  @property
  def blocks(self):
    """Aggregate all blocks from all functions. Returns dict {func_name: {region_name: {idx: {block_id: DataBlock}}}}"""
    aggregated = {}
    for func_name, layout in self.data.items():
      layout_blocks = layout.blocks
      if layout_blocks:  # Only include functions with blocks
        aggregated[func_name] = layout_blocks
    return aggregated

  def __getitem__(self, key: str) -> 'FuncMemoryLayout':
    """auto-create from template"""
    if key not in self.data:
      return self.__missing__(key)
    return self.data[key]

  def __missing__(self, key: str) -> 'FuncMemoryLayout':
    """
    Called when a key is not found. Creates a new FuncMemoryLayout from the template.
    """
    # Create a deep copy of the template layout
    new_layout = copy.deepcopy(self._layout)
    self[key] = new_layout
    return new_layout

  def __str__(self):
    if not self.data:
      return "MemoryLayout(empty)"

    lines = ["MemoryLayout("]
    for func_name, layout in self.data.items():
      lines.append(f"  {func_name!r}: {layout},")
    lines.append(")")
    return "\n".join(lines)

  def __repr__(self):
    return self.__str__()


class RouterEntry:
  def __init__(self, router_id: int, address: int, data: Dict):
    self.router_id = router_id
    self.address = address
    self.data = data

  def __str__(self):
    return f"RouterEntry({self.router_id}, {self.address}, {self.data})"

  def __repr__(self):
    return self.__str__()

  def __eq__(self, other):
    is_equal = self.router_id == other.router_id and self.address == other.address
    if is_equal:
      assert self.data == other.data, f"Data mismatch: {self.data} != {other.data} in same router entry"
    return is_equal


class EdgeInfo:
  """ stores the list of router entries and memory block info for a data movement (edge). """

  def __init__(self, policy_info: List[RouterEntry], data_block: Union[DataBlock, None] = None):
    self.policy_info = policy_info
    self.data_block = data_block

  def set_policy_info(self, policy_info: List[RouterEntry]):
    self.policy_info = policy_info

  def append_policy_info(self, entry: RouterEntry):
    self.policy_info.append(entry)

  def set_data_block(self, data_block: DataBlock):
    self.data_block = data_block


class InstEdgeInfo(EdgeInfo):
  def __str__(self):
    policy_info_str = ", ".join(str(entry) for entry in self.policy_info)
    return f"InstEdgeInfo([{policy_info_str}], {self.data_block})"

  def __repr__(self):
    return self.__str__()


class TensorEdgeInfo(EdgeInfo):
  LOCAL_FIFO = -2
  def __init__(self, policy_info: List[RouterEntry] = None, data_block: Union[DataBlock, None] = None, fifo_id: int = -1, block_tiling_info: BlockTileInfo = None, producer_sync_granularity: int = None):
    super().__init__(policy_info, data_block)
    self.fifo_id = fifo_id
    self.owner = None
    self.block_tiling_info = block_tiling_info
    self.producer_sync_granularity = producer_sync_granularity  # Number of SENDs per STANDBY from producer IMCE
    # --- Sync-granularity contract (DWCONV_SYNC_GRANULARITY_DESIGN.md) ---
    # PnR fills these once; inode/imce codeblocks READ them instead of each
    # recomputing packet/handshake granularity locally. All default to None
    # (== "not yet contracted"), so codegen that has not been migrated to read
    # them behaves exactly as before. See design doc sections 2.1 / 2.3.
    #   channels_per_issue     : channels the consumer op processes per issue
    #                            (conv-fed=64, dwconv/pool/quant-fed=16). HW:
    #                            post_imcu NumChannels / vpu NumBlocks.
    #   fill_order             : axis order the consumer fills data in
    #                            (e.g. ["ch_pass","h","w","bitplane"] for a
    #                            linebuffer LOAD_LB consumer; None == flat RF).
    #   producer_send_per_sync : producer SENDs emitted per SETFLAG/STANDBY pair
    #                            (== the old producer_sync_granularity meaning;
    #                            1 == per-packet handshake, the current default).
    #   consumer_recv_per_sync : consumer RECV/LOAD_LB per receiver-side window.
    #   needs_flag_rendezvous  : True iff a same-fifo matched SEND/RECV count
    #                            does NOT already order this edge (cyclic /
    #                            compute-gating / transitive). If False, raw
    #                            SEND/RECV + fifo backpressure suffices.
    self.channels_per_issue = None
    self.fill_order = None
    self.producer_send_per_sync = None
    self.consumer_recv_per_sync = None
    self.needs_flag_rendezvous = None
    # --- Max-throughput feed-spread (IMCFLOW_FEED_SPREAD) ---
    # When >0, the conv activation feed round-robins its per-packet fifo_id
    # across fifo_id .. fifo_id+spread_fifo_n-1 (mod the 8 HW RECV fifos)
    # instead of pinning every packet to `fifo_id`. Set ONLY on the inode->imce
    # data-input edge of an imcflow_qconv/qdwconv, ONLY when the env flag is on
    # (feed_spread_n() > 0). Default 0 -> spread_fifo_id() == fifo_id for every
    # packet -> byte-identical to the pinned behavior. The route (policy_info)
    # is unchanged; only the terminal fifo_id operand of SEND/LOAD_LB rotates
    # (HW dispatches to RECV fifo N by the packet's fifo_id field).
    self.spread_fifo_n = 0

  def effective_spread_n(self, repeat: int) -> int:
    """Clamp the requested spread width to the largest divisor of `repeat`
    (the per-pixel bitplane count) that is <= spread_fifo_n. This guarantees
    the fifo pattern repeats exactly every `repeat` packets, so the IMCE
    LOAD_LB (which rotates on the per-pixel bitplane index b in 0..repeat-1)
    and the INODE SEND (which rotates on the FLAT packet index k = pixel*repeat
    + b) select the SAME fifo for the same word: with repeat % n == 0 we have
    (pixel*repeat + b) % n == b % n. Returns 1 (== no spread) when disabled."""
    n = self.spread_fifo_n
    if n <= 1 or self.fifo_id < 0 or repeat <= 1:
      return 1
    n = min(n, repeat, 8)
    # largest divisor of `repeat` that is <= n
    for d in range(n, 1, -1):
      if repeat % d == 0:
        return d
    return 1

  def prefetch_group(self, repeat: int):
    """Max-throughput lever (IMCFLOW_FEED_PREFETCH): return (pixels_per_group,
    width) when prefetch is active for THIS edge, else None. Prefetch spreads a
    P-pixel flat window (P*repeat words) across width = P*repeat fifos (<=8) so
    the next pixel's bitplanes are resident in distinct RECV fifos during the
    current compute. Requires spread enabled on the edge, repeat==4 (conv
    bitplanes), P*repeat<=8. Both LOAD_LB (unrolled P pixels x repeat bitplanes)
    and INODE SEND (unrolled P*repeat words) index the flat word k and pick fifo
    k%width, so they agree. Returns None (-> fall back to effective_spread_n)
    when off / not applicable."""
    from tvm.contrib.imcflow import feed_prefetch_n
    p = feed_prefetch_n()
    if p < 2 or self.spread_fifo_n <= 1 or self.fifo_id != 0 or repeat <= 1:
      return None
    width = p * repeat
    if width > 8:
      return None  # HW has only 8 RECV fifos; P*repeat must fit
    return (p, width)

  def spread_fifo_id(self, packet_index: int, repeat: int) -> int:
    """Terminal RECV fifo_id for the packet at `packet_index` (may be a flat
    stream index or a per-pixel bitplane index -- see effective_spread_n for
    why both agree). With spread OFF this is always `self.fifo_id`
    (byte-identical). With spread ON, round-robin across
    fifo_id .. fifo_id+eff_n-1, wrapping within the 8 HW RECV fifos."""
    eff = self.effective_spread_n(repeat)
    if eff <= 1:
      return self.fifo_id
    return (self.fifo_id + (int(packet_index) % eff)) % 8

  def set_sync_contract(self, channels_per_issue=None, fill_order=None,
                        producer_send_per_sync=None, consumer_recv_per_sync=None,
                        needs_flag_rendezvous=None):
    """Populate the sync-granularity contract (called once from PnR /
    policy_table_builder). Only overwrites fields explicitly provided."""
    if channels_per_issue is not None:
      self.channels_per_issue = channels_per_issue
    if fill_order is not None:
      self.fill_order = fill_order
    if producer_send_per_sync is not None:
      self.producer_send_per_sync = producer_send_per_sync
    if consumer_recv_per_sync is not None:
      self.consumer_recv_per_sync = consumer_recv_per_sync
    if needs_flag_rendezvous is not None:
      self.needs_flag_rendezvous = needs_flag_rendezvous

  def set_fifo_id(self, fifo_id):
    self.fifo_id = fifo_id

  @property
  def node_info_str(self):
    src_node = self.policy_info[0].router_id.name
    dst_nodes = []
    for router_entry in self.policy_info:
      if router_entry.data["Local"]["enable"]:
        dst_nodes.append(router_entry.router_id.name)
    return f"{src_node} -> {', '.join(dst_nodes)}"

  def set_tiling_info(self, height_base_coords: List[int], height_sizes: List[int], pkt_cnts: List[int]):
    self.height_base_coords = height_base_coords
    self.height_sizes = height_sizes
    self.pkt_cnts = pkt_cnts

  def set_block_tiling_info(self, block_tiling_info: BlockTileInfo):
    self.block_tiling_info = block_tiling_info

  def get_height_base_coords(self):
    if self.block_tiling_info:
      return self.block_tiling_info.height_base_coords
    return getattr(self, 'height_base_coords', [])

  def get_height_sizes(self):
    if self.block_tiling_info:
      return self.block_tiling_info.height_sizes
    return getattr(self, 'height_sizes', [])

  def get_pkt_cnts(self):
    if self.block_tiling_info:
      return self.block_tiling_info.pkt_cnts
    return getattr(self, 'pkt_cnts', [])

  def get_c_input_var_offsets(self):
    if self.block_tiling_info:
      return self.block_tiling_info.c_input_var_offsets
    return []

  def get_c_input_var_sizes(self):
    if self.block_tiling_info:
      return self.block_tiling_info.c_input_var_sizes
    return []

  def __str__(self):
    # policy_info_str = ", ".join(str(entry) for entry in self.policy_info) if self.policy_info else "[]"
    policy_info_str = "router entries:\n"
    for entry in self.policy_info:
      policy_info_str += f"  {str(entry)}"
    
    data = "TensorEdgeInfo(\n"
    data += f"  {policy_info_str}\n"
    data += f"  data_block={self.data_block}\n"
    data += f"  fifo_id={self.fifo_id}\n"
    data += ")"

    return data

  def __repr__(self):
    return self.__str__()


class CodegenContext:
  """Singleton context for codegen that tracks the current function being processed"""

  def __new__(cls):
    if not hasattr(cls, "instance"):
      cls.instance = super(CodegenContext, cls).__new__(cls)
      cls.instance._initialize()
    return cls.instance

  def _initialize(self):
    self._func_name = None

  def set_func_name(self, func_name: str):
    """Set the current function name being processed"""
    self._func_name = func_name

  def get_func_name(self) -> str:
    """Get the current function name"""
    if self._func_name is None:
      raise RuntimeError("No function context set. Call set_func_name() first in codegen.")
    return self._func_name

  @property
  def func_name(self) -> str:
    """Property access to current function name"""
    return self.get_func_name()

  def clear(self):
    """Clear the current function context"""
    self._func_name = None


class ImcflowDeviceConfig:
  """Imcflow config class"""
  if SMALL_DEBUG:
    NODE_COL_NUM = 3
    INODE_NUM = 4
    IMCE_H_NUM = 4
    IMCE_W_NUM = 2
    IMCE_NUM = 8
  else:
    NODE_COL_NUM = 5
    INODE_NUM = 4
    IMCE_H_NUM = 4
    IMCE_W_NUM = 4
    IMCE_NUM = 16

  IMCU_ROW_NUM = 256
  INODE_MMREG_SIZE = 128
  INODE_DATA_MEM_SIZE = 65536
  INODE_MAX_TILING_SIZE = 65536 # should be smaller than INODE_DATA_MEM_SIZE  
  # INODE_DATA_MEM_SIZE = 65536
  # INODE_DATA_MEM_SIZE = 131072

  if not BIG_IMEM:
    INODE_INST_MEM_SIZE = 1024
  else:
    INODE_INST_MEM_SIZE = 2048

  if not BIG_IMEM:
    IMCE_INST_MEM_SIZE = 1024
  else:
    IMCE_INST_MEM_SIZE = 2048

  # IMCFLOW_ADDR_SIZE = 266368 # 128 + 4*(65536+1024) == 260.125KB
  IMCFLOW_ADDR_SIZE = 128 + INODE_NUM * (INODE_DATA_MEM_SIZE + INODE_INST_MEM_SIZE) # 270464 for BIG_IMEM, 266368 for normal
  HOST_OS = os.getenv("IMCFLOW_HOST_OS", "linux")
  HOST_ISA = os.getenv("IMCFLOW_HOST_ISA", "arm")

  INODE0_IMEM_BASE_ADDR = 128
  INODE0_DMEM_BASE_ADDR = INODE0_IMEM_BASE_ADDR + INODE_INST_MEM_SIZE

  INODE1_IMEM_BASE_ADDR = INODE0_DMEM_BASE_ADDR + INODE_DATA_MEM_SIZE
  INODE1_DMEM_BASE_ADDR = INODE1_IMEM_BASE_ADDR + INODE_INST_MEM_SIZE

  INODE2_IMEM_BASE_ADDR = INODE1_DMEM_BASE_ADDR + INODE_DATA_MEM_SIZE
  INODE2_DMEM_BASE_ADDR = INODE2_IMEM_BASE_ADDR + INODE_INST_MEM_SIZE

  INODE3_IMEM_BASE_ADDR = INODE2_DMEM_BASE_ADDR + INODE_DATA_MEM_SIZE # 0x30c80
  INODE3_DMEM_BASE_ADDR = INODE3_IMEM_BASE_ADDR + INODE_INST_MEM_SIZE

  SUPPORTED_OPS = ["nn.imcflow_qconv", "nn.imcflow_qdwconv", "nn.bias_add", "imcflow.fused_batch_norm",
                   "nn.relu", "add", "split", "concatenate", "qnn.imcflow_min_max_quantize",
                   "qnn.imcflow_nu_quantize", "divide", "imcflow_packing", "imcflow_unpacking",
                   "nn.conv2d", "nn.batch_norm","multiply"]
  NO_COST_OPS = ["split", "concatenate", "imcflow_packing", "imcflow_unpacking"]
  QAUNT_OPS = ["qnn.imcflow_min_max_quantize", "qnn.imcflow_nu_quantize"]

  def __new__(cls):
    if not hasattr(cls, "instance"):
      cls.instance = super(ImcflowDeviceConfig, cls).__new__(cls)
      cls.instance._initialize()
    return cls.instance

  def _initialize(self):
    self.HWNodeMap = {}
    self.TensorIDtoEdge = {}
    self.TensorEdgetoInfo = {}
    self.TensorEdgeList = []
    self.TensorEdgeListDict = {}
    self.PolicyTableDict = {}
    self.InstEdgeInfoDict = {}
    # Flag to control whether to use .patched.cpp files during compilation
    # When True, codegen will look for {base}.patched.cpp before falling back to {base}.cpp
    self.use_patched_cpp = False
    self.single_qconv = False
    self.MemLayout = MemoryLayout(
        MemoryRegionEntry("state_regs", ImcflowDeviceConfig.INODE_MMREG_SIZE),
        MemoryRegionEntry("inode_0_0_inst", ImcflowDeviceConfig.INODE_INST_MEM_SIZE),
        MemoryRegionEntry("inode_0_0_data", ImcflowDeviceConfig.INODE_DATA_MEM_SIZE),
        MemoryRegionEntry("inode_1_0_inst", ImcflowDeviceConfig.INODE_INST_MEM_SIZE),
        MemoryRegionEntry("inode_1_0_data", ImcflowDeviceConfig.INODE_DATA_MEM_SIZE),
        MemoryRegionEntry("inode_2_0_inst", ImcflowDeviceConfig.INODE_INST_MEM_SIZE),
        MemoryRegionEntry("inode_2_0_data", ImcflowDeviceConfig.INODE_DATA_MEM_SIZE),
        MemoryRegionEntry("inode_3_0_inst", ImcflowDeviceConfig.INODE_INST_MEM_SIZE),
        MemoryRegionEntry("inode_3_0_data", ImcflowDeviceConfig.INODE_DATA_MEM_SIZE),
    )
    self.ActiveIMCEPerFunc = {}
    self.NoCPaths = {}
    self.DataBlocks = {}
    self.ImcflowFuncMap = {}  # {func_name: FunctionInfo}
    self.use_def_chain = {}   # {func_name: use-def mapping}
    self.LayoutMap={}
    self.FIFOConflictTable = {}
    self.NoCDeadlockTable = {}
    self.SplitInfo = {} # func_name : {split_node_id : split_info}. split_info = {'is_multi_cast':bool, channels : int, num_splits : int}
    # Atomic-qconv split metadata, keyed by a hash of the weight Constant bytes.
    # Populated by split_conv_to_atomic; consumed by psum_mapping to project
    # atomic qconvs back onto their original conv's (OC, IC) grid.
    # Each value: dict(orig_conv_id, orig_conv_name, oc_id, ic_id, oc_block,
    #                  ic_block, total_oc, total_ic, kernel)
    self.AtomicSplitInfo = {}
    # Same metadata as AtomicSplitInfo, but keyed by the imcflow function
    # name that wraps each atomic qconv (one qconv per function in
    # single_qconv / v2 mode). Populated by remap_atomic_split_info_by_func()
    # right after qconv_only_partition, BEFORE layout legalization rewrites
    # the weight Constants. Function names are stable through layout
    # legalization and PnR, so this is the source psum_mapping consumes.
    self.AtomicSplitInfoByFunc = {}
    # weight-bytes-hash -> original conv weight Var name. Captured by
    # capture_orig_conv_names() right before bind_params_by_name converts
    # weight Vars to Constants; consumed by split_conv_to_atomic to replace
    # synthetic 'conv_N' with the model-side weight name (e.g. the relay-level
    # name assigned when building the model).
    self.OrigConvNameMap = {}
    # Column disable config
    self.ColumnDisableMap = {}   # imce_linear_id (0-15) -> list[int] of disabled column indices
    self.NumDisableColumns = 0   # default: no columns disabled
    self.ActiveIMCESet = None    # None = all IMCE_NUM cores active; otherwise set[int] of allowed linear IDs

  def clear(self):
    self._initialize()

  def load_column_disable_config(self, filepath, num_disable_columns=8):
    """Load per-IMCU disabled column indices from JSON file.

    Schema (CIM training disabled.json):

        {
          "num_disable": <int>,
          "per_core": true,
          "cores": [
            {"h_id": <int>, "w_id": <int>, "disabled": [<col>, ...], "active": [<col>, ...]},
            ...
          ]
        }

    h_id is the row (0..IMCE_H_NUM-1), w_id is the IMCE column (1..IMCE_W_NUM,
    where 0 is the INODE column). linear_id = h_id * IMCE_W_NUM + (w_id - 1).

    The ``cores`` list may cover all IMCEs or only a subset; if a subset, the
    listed cores become the active set for PnR (see ``ActiveIMCESet``).
    """
    with open(filepath, 'r') as f:
      data = json.load(f)

    if "cores" not in data:
      raise ValueError(
        f"{filepath}: column-disable JSON must have a 'cores' list "
        f"({{num_disable, per_core, cores: [{{h_id, w_id, disabled, active}}, ...]}}). "
        f"The legacy linear-id keyed format is no longer supported."
      )

    self.ColumnDisableMap = {}
    self.ActiveIMCESet = None
    self.NumDisableColumns = data.get("num_disable", num_disable_columns)
    for core in data["cores"]:
      h_id = core["h_id"]
      w_id = core["w_id"]
      imce_id = h_id * self.IMCE_W_NUM + (w_id - 1)
      indices = core["disabled"]
      assert 0 <= imce_id < self.IMCE_NUM, \
          f"IMCE ({h_id},{w_id}) -> linear {imce_id} out of range [0, {self.IMCE_NUM})"
      assert len(indices) == self.NumDisableColumns, \
          f"IMCE {imce_id}: expected {self.NumDisableColumns} disabled columns, got {len(indices)}"
      for idx in indices:
        assert 0 <= idx < 64, f"Column index {idx} out of range [0, 64)"
      self.ColumnDisableMap[imce_id] = sorted(indices)

    # If the JSON only mentions a subset of cores, treat that as the active core set
    # for PnR. When all IMCE_NUM cores are listed, leave ActiveIMCESet=None to preserve
    # the legacy "use all cores" behavior.
    if 0 < len(self.ColumnDisableMap) < self.IMCE_NUM:
      self.ActiveIMCESet = set(self.ColumnDisableMap.keys())

    print(f"[ColumnDisable] Loaded {len(self.ColumnDisableMap)} IMCEs, "
          f"{self.NumDisableColumns} disabled cols each from {os.path.basename(filepath)}")
    if self.ActiveIMCESet is not None:
      print(f"[ColumnDisable] Active IMCE subset: {sorted(self.ActiveIMCESet)}")

  def validate_noise_layout(self, noise_layout_json_path):
    """Fail-fast if a noise layout JSON's disabled set disagrees with the
    column-disable config loaded by ``load_column_disable_config``.

    ``noise_layout_json_path`` is the path to ``concat_per_core.json`` (the
    imce_map noise layout). For every IMCE present in BOTH configs, this
    asserts that ``set(column_disable_config[linear]) == set(noise_layout[linear])``.

    The column-disable JSON may list a subset of cores (PnR's ActiveIMCESet);
    only the listed cores are required to match the noise layout — extra cores
    in the noise layout for inactive IMCEs are tolerated.
    """
    if not self.ColumnDisableMap:
      raise RuntimeError(
        f"validate_noise_layout('{noise_layout_json_path}') called before "
        f"load_column_disable_config — nothing to compare against."
      )

    with open(noise_layout_json_path, 'r') as f:
      layout = json.load(f)

    disabled_by_core = layout.get("disabled_by_core")
    if not disabled_by_core:
      raise RuntimeError(
        f"Noise layout '{noise_layout_json_path}': missing 'disabled_by_core' "
        f"(cannot cross-validate against column-disable config)."
      )

    layout_disabled = {}   # linear_id -> set(int)
    for core_key, indices in disabled_by_core.items():
      h_str, w_str = str(core_key).split("_")
      h_id = int(h_str)
      w_id = int(w_str)
      linear = h_id * self.IMCE_W_NUM + (w_id - 1)
      layout_disabled[linear] = set(int(i) for i in indices)

    # Per-core diff
    mismatches = []
    cfg_n = layout.get("num_disable")
    if cfg_n is not None and int(cfg_n) != self.NumDisableColumns:
      mismatches.append(
        f"num_disable: column_disable_config={self.NumDisableColumns} "
        f"vs noise_layout={cfg_n}"
      )

    for linear, cfg_disabled in self.ColumnDisableMap.items():
      cfg_set = set(int(i) for i in cfg_disabled)
      lay_set = layout_disabled.get(linear)
      if lay_set is None:
        mismatches.append(
          f"IMCE linear={linear}: present in column-disable config but "
          f"absent from noise_layout"
        )
        continue
      if cfg_set != lay_set:
        only_cfg = sorted(cfg_set - lay_set)
        only_lay = sorted(lay_set - cfg_set)
        mismatches.append(
          f"IMCE linear={linear}: disabled sets differ "
          f"(only in column-disable: {only_cfg}; only in noise_layout: {only_lay})"
        )

    if mismatches:
      detail = "\n  - " + "\n  - ".join(mismatches)
      raise RuntimeError(
        f"Noise layout '{noise_layout_json_path}' does not match the column-"
        f"disable config:{detail}\n"
        f"Weight loading and noise sampling would use different enabled column "
        f"sets — outputs would be silently wrong. Regenerate one of the configs "
        f"so the disabled sets agree per IMCE."
      )

    print(f"[ColumnDisable] noise layout '{os.path.basename(noise_layout_json_path)}' "
          f"matches column-disable config across {len(self.ColumnDisableMap)} active IMCEs.")

  def get_valid_columns(self, imce_linear_id):
    """Returns sorted list of valid column indices for the given IMCE."""
    disabled = set(self.ColumnDisableMap.get(imce_linear_id, []))
    return sorted(set(range(64)) - disabled)

  def get_effective_oc(self):
    """Returns effective output channels per IMCU: 64 - NumDisableColumns."""
    return 64 - self.NumDisableColumns

  def get_active_imce_ids(self):
    """Returns sorted list of IMCE linear IDs available for placement.

    When ActiveIMCESet is None (no JSON or full-16 JSON), returns 0..IMCE_NUM-1.
    Otherwise returns only the cores listed in the column-disable JSON.
    """
    if self.ActiveIMCESet is None:
      return list(range(self.IMCE_NUM))
    return sorted(self.ActiveIMCESet)

  @ staticmethod
  def is_supported_kernel(KH, KW):
    return (KH, KW) in {(1, 1), (3, 3), (5, 5), (7, 7)}

  def add_hw_node(self, graph_node_id: Union[int, Tuple], hwnode_id: int):
    self.HWNodeMap[graph_node_id] = hwnode_id

  def get_hw_node(self, graph_node_id: Union[int, Tuple], tuple_idx=None):
    out = None
    if graph_node_id in self.HWNodeMap:
      out = self.HWNodeMap[graph_node_id]
    elif isinstance(graph_node_id, Tuple):
      out = self.HWNodeMap.get(graph_node_id[1], None)

    if tuple_idx is not None:
      return out[tuple_idx] if out is not None else None
    else:
      return out
  
  def is_in_hw_node(self, graph_node_id: Union[int, Tuple], tuple_idx=None):
    return self.get_hw_node(graph_node_id, tuple_idx) is not None
  
  def add_tensor_edge(self, tensor_id: TensorID, tensor_edge: TensorEdge):
    self.TensorIDtoEdge[tensor_id] = tensor_edge

  def get_tensor_edge(self, tensor_id: TensorID):
    return self.TensorIDtoEdge.get(tensor_id, None)

  def get_tensor_edges_from_graph_node_id(self, graph_node_id: Union[int, Tuple], dir="inout"):
    edges = []
    for tensor_edge in self.TensorEdgeList:
      src_gid = tensor_edge.src_id.graph_node_id
      dst_gid = tensor_edge.dst_id.graph_node_id
      getInnerID = lambda gid: gid[1] if isinstance(gid, Tuple) else gid
      if dir == "inout":
        if getInnerID(src_gid) == getInnerID(graph_node_id) or getInnerID(dst_gid) == getInnerID(graph_node_id):
          edges.append(tensor_edge)
      elif dir == "in":
        if getInnerID(dst_gid) == getInnerID(graph_node_id):
          edges.append(tensor_edge)
      elif dir == "out":
        if getInnerID(src_gid) == getInnerID(graph_node_id):
          edges.append(tensor_edge)
      else:
        raise ValueError("Invalid direction")
    return edges

  def add_tensor_edge_info(self, tensor_edge: TensorEdge, tensor_edge_info: TensorEdgeInfo):
    tensor_edge_info.owner = tensor_edge
    self.TensorEdgetoInfo[tensor_edge] = tensor_edge_info

  def get_tensor_edge_info(self, tensor_edge: TensorEdge):
    return self.TensorEdgetoInfo.get(tensor_edge, None)

  def get_tensor_edge_info_with_id_dir(self, tensor_id: TensorID, dir: str) -> List[TensorEdgeInfo]:
    edge_infos = []
    if dir == "in":
      for edge in self.TensorEdgetoInfo.keys():
        if edge.dst_id == tensor_id:
          edge_infos.append(self.TensorEdgetoInfo[edge])
      return edge_infos
    elif dir == "out":
      for edge in self.TensorEdgetoInfo.keys():
        if edge.src_id == tensor_id:
          edge_infos.append(self.TensorEdgetoInfo[edge])
      return edge_infos
    else:
      raise ValueError("Invalid direction")

  def get_tensor_ids_from_graph_node_id(self, graph_node_id: Union[int, Tuple]):
    tids = []
    for tid in self.TensorIDtoEdge.keys():
      if tid.graph_node_id == graph_node_id:
        tids.append(tid)
    return tids

  def add_inst_edge_info(self, func_name: str, imce_id: NodeID, inst_edge_info: InstEdgeInfo):
    assert imce_id.is_imce(), "Only imce nodes have inst edge info"
    self.InstEdgeInfoDict.setdefault(func_name, {})[imce_id] = inst_edge_info

  def get_inst_edge_info(self, func_name: str, imce_id: NodeID):
    assert imce_id.is_imce(), "Only imce nodes have inst edge info"
    return self.InstEdgeInfoDict.get(func_name, {}).get(imce_id, None)

  @property
  def CurrFuncMemLayout(self):
    """
    Get the memory layout for the current function in CodegenContext.

    This is a convenience property that allows accessing the current function's
    memory layout without explicitly passing func_name around:
    """
    return self.MemLayout[CodegenContext().func_name]

  def format_policy_table(self):
    """
    Format the PolicyTableDict into a hierarchical string representation.

    Returns:
      str: Formatted policy table with function names, node IDs, and compact policy entries.

    Example format:
      tvmgen_default_tvmgen_default_imcflow_main_47_round_imcflow_region1_main_0
        <NodeID.inode_0_0: 0>
          0:  L:(0, 0, 0), N:(1, 31), E:(0, 50), S:(1, 61), W:(0, 37)
          1:  L:(0, 0, 0), N:(1, 31), E:(0, 50), S:(1, 61), W:(0, 37)
    """
    lines = []

    for func_name, nodes in self.PolicyTableDict.items():
      lines.append(f"{func_name}")

      for node_id, policies in nodes.items():
        lines.append(f"  {node_id}")

        for idx, policy in enumerate(policies):
          # Extract direction info
          local = policy.get('Local', {})
          north = policy.get('North', {})
          east = policy.get('East', {})
          south = policy.get('South', {})
          west = policy.get('West', {})

          # Format Local: (enable, addr, chunk_index)
          local_str = ""
          if local:
            enable = int(local.get('enable', False))
            addr = local.get('addr', 0)
            chunk_index = local.get('chunk_index', 0)
            ksel = local.get('ksel', 0)
            local_str = f"L:({enable}, {addr}, {ksel}, {chunk_index})"

          # Format other directions: (enable, addr)
          def format_dir(dir_dict, name):
            if dir_dict:
              enable = int(dir_dict.get('enable', False))
              addr = dir_dict.get('addr', 0)
              return f"{name}:({enable}, {addr})"
            return None

          parts = []
          if local_str:
            parts.append(local_str)
          for dir_data, dir_name in [(north, 'N'), (east, 'E'), (south, 'S'), (west, 'W')]:
            formatted = format_dir(dir_data, dir_name)
            if formatted:
              parts.append(formatted)

          lines.append(f"    {idx}:  {', '.join(parts)}")

      lines.append("")  # Empty line between functions

    return "\n".join(lines)


  def update_compiled_blocks(self, func_name):
    from tvm.relay.backend.contrib.imcflow import transform as imcflow_transform
    compiled_blocks, compiled_per_tile_blocks = [], []

    for memory_region in ImcflowDeviceConfig().MemLayout[func_name].values():
      for block_name, block in memory_region.blocks.items():
        # get compiled data blocks
        if isinstance(block_name, str):
          # Filter blocks ending with _cnt_base_addr into compiled_per_tile_blocks
          if block_name.endswith("_cnt_base_addr"):
            compiled_per_tile_blocks.append(block)
          else:
            compiled_blocks.append(block)

    if func_name not in self.DataBlocks:
      self.DataBlocks[func_name] = {}

    self.DataBlocks[func_name]["compiled"] = compiled_blocks
    self.DataBlocks[func_name]["compiled_per_tile"] = compiled_per_tile_blocks

  def update_data_blocks(self, func_name, input_node_ids=None, output_node_id=None, const_node_ids=None):
    from tvm.relay.backend.contrib.imcflow import transform as imcflow_transform
    input_data_blocks, output_data_blocks, const_data_blocks = [], [], []

    for memory_region in ImcflowDeviceConfig().MemLayout[func_name].values():
      for block_name, block in memory_region.blocks.items():
        # get input & output data blocks
        if isinstance(block_name, TensorEdge):
          if isinstance(block_name.src_id.graph_node_id, Tuple):
            src_gid = block_name.src_id.graph_node_id[1]
          else:
            src_gid = block_name.src_id.graph_node_id

          is_input_block = False
          is_const_block = False
          is_output_block = False
          # get input data blocks
          if any([input_node_id == imcflow_transform.getInnerNodeID(src_gid) for input_node_id in input_node_ids]):
            input_data_blocks.append(block)
            is_input_block = True

          # get const data blocks
          if any([const_node_id == imcflow_transform.getInnerNodeID(src_gid) for const_node_id in const_node_ids]):
            const_data_blocks.append(block)
            is_const_block = True

          # get output data blocks
          if isinstance(block_name.dst_id.graph_node_id, Tuple):
            dst_gid = block_name.dst_id.graph_node_id[1]
          else:
            dst_gid = block_name.dst_id.graph_node_id
          if output_node_id == imcflow_transform.getInnerNodeID(dst_gid):
            output_data_blocks.append(block)
            is_output_block = True

          if sum([is_input_block, is_const_block, is_output_block]) == 0:
            print(f"Warning: DataBlock {block} is neither input, output, nor const block for function {func_name}")
            # TODO: add the exception again after dealing with is_input_block with split node
            # raise ValueError("DataBlock type identification error")
          elif sum([is_input_block, is_const_block, is_output_block]) > 1:
            print(f"Warning: DataBlock {block} is multiple types of blocks for function {func_name}")
            raise ValueError("DataBlock type identification error")

    if func_name not in self.DataBlocks:
      self.DataBlocks[func_name] = {}

    self.DataBlocks[func_name]["input"] = input_data_blocks
    self.DataBlocks[func_name]["output"] = output_data_blocks
    self.DataBlocks[func_name]["const"] = const_data_blocks

  def save_state(self, filepath: str):
    """
    Save the DevConfig state to a file for later restoration.
    This allows rebuild_modified_cpp to work without re-running transform passes.

    Args:
        filepath: Path to save the serialized state
    """
    import pickle
    from tvm.relay.op.contrib.imcflow import (
        HashToCustomID, CustomIDToName, CustomIDToNode, CustomIDInFunc
    )

    state = {
        'HWNodeMap': self.HWNodeMap,
        'TensorIDtoEdge': self.TensorIDtoEdge,
        'TensorEdgetoInfo': self.TensorEdgetoInfo,
        'TensorEdgeList': self.TensorEdgeList,
        'TensorEdgeListDict': self.TensorEdgeListDict,
        'PolicyTableDict': self.PolicyTableDict,
        'InstEdgeInfoDict': self.InstEdgeInfoDict,
        'MemLayout': self.MemLayout,
        'ActiveIMCEPerFunc': self.ActiveIMCEPerFunc,
        'NoCPaths': self.NoCPaths,
        'DataBlocks': self.DataBlocks,
        'ImcflowFuncMap': self.ImcflowFuncMap,
        'use_def_chain': self.use_def_chain,
        'LayoutMap': self.LayoutMap,
        'FIFOConflictTable': self.FIFOConflictTable,
        'NoCDeadlockTable': self.NoCDeadlockTable,
        'AtomicSplitInfo': self.AtomicSplitInfo,
        'AtomicSplitInfoByFunc': self.AtomicSplitInfoByFunc,
        'OrigConvNameMap': self.OrigConvNameMap,
        # Save singleton state for rebuild_modified_cpp
        'HashToCustomID': dict(HashToCustomID()),
        'CustomIDToName': dict(CustomIDToName()),
        'CustomIDToNode': dict(CustomIDToNode()),
        'CustomIDInFunc': dict(CustomIDInFunc()),
    }

    with open(filepath, 'wb') as f:
      pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"DevConfig state saved to: {filepath}")

  def load_state(self, filepath: str):
    """
    Load the DevConfig state from a file.
    This restores the state created during transform passes for rebuild_modified_cpp.

    Args:
        filepath: Path to the serialized state file
    """
    import pickle
    from tvm.relay.op.contrib.imcflow import (
        HashToCustomID, CustomIDToName, CustomIDToNode, CustomIDInFunc
    )

    if not os.path.exists(filepath):
      raise FileNotFoundError(f"DevConfig state file not found: {filepath}")

    with open(filepath, 'rb') as f:
      state = pickle.load(f)

    # Restore all state
    self.HWNodeMap = state['HWNodeMap']
    self.TensorIDtoEdge = state['TensorIDtoEdge']
    self.TensorEdgetoInfo = state['TensorEdgetoInfo']
    self.TensorEdgeList = state['TensorEdgeList']
    self.TensorEdgeListDict = state['TensorEdgeListDict']
    self.PolicyTableDict = state['PolicyTableDict']
    self.InstEdgeInfoDict = state['InstEdgeInfoDict']
    self.MemLayout = state['MemLayout']
    self.ActiveIMCEPerFunc = state['ActiveIMCEPerFunc']
    self.NoCPaths = state['NoCPaths']
    self.DataBlocks = state['DataBlocks']
    self.ImcflowFuncMap = state['ImcflowFuncMap']
    self.use_def_chain = state['use_def_chain']
    self.LayoutMap = state['LayoutMap']
    self.FIFOConflictTable = state['FIFOConflictTable']
    self.NoCDeadlockTable = state['NoCDeadlockTable']
    self.AtomicSplitInfo = state.get('AtomicSplitInfo', {})
    self.AtomicSplitInfoByFunc = state.get('AtomicSplitInfoByFunc', {})
    self.OrigConvNameMap = state.get('OrigConvNameMap', {})

    # Restore singleton state
    # Must use clear() + update() since these are singleton instances
    HashToCustomID().clear()
    HashToCustomID().update(state.get('HashToCustomID', {}))

    CustomIDToName().clear()
    CustomIDToName().update(state.get('CustomIDToName', {}))

    CustomIDToNode().clear()
    CustomIDToNode().update(state.get('CustomIDToNode', {}))

    CustomIDInFunc().clear()
    CustomIDInFunc().update(state.get('CustomIDInFunc', {}))

    print(f"DevConfig state loaded from: {filepath}")

  def update_datablocks_state(self, filepath: str):
    """
    Update only the DataBlocks field in the saved DevConfig state.
    This is called after constructDataBlockDict to add DataBlock categorization
    without saving the allocation information from CodegenSuite.

    Args:
        filepath: Path to the serialized state file
    """
    import pickle

    if not os.path.exists(filepath):
      raise FileNotFoundError(f"DevConfig state file not found: {filepath}")

    # Load existing state
    with open(filepath, 'rb') as f:
      state = pickle.load(f)

    # Update only DataBlocks field
    state['DataBlocks'] = self.DataBlocks

    # Re-save the updated state
    with open(filepath, 'wb') as f:
      pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"DevConfig DataBlocks updated in: {filepath}")
