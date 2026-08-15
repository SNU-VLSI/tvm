from typing import List
import tvm
from tvm import relay
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.contrib.imcflow import DataBlock
from tvm.relay.ty import TensorType, TupleType
from . import transform as imcflow_transform
from tvm.contrib.imcflow import ImcflowDeviceConfig, TensorID, DataBlock, TensorEdge
from tvm.contrib.imcflow import mmio_block_barrier_usec
from tvm.relay.op.contrib.imcflow import CustomIDToNode
from tvm.relay.expr import (Var, Constant)
from tvm.runtime import String
import math
import os
from enum import Enum

if (os.getenv("IMCFLOW_HOST_OS") == "baremetal"):
  print("IMCFLOW_HOST_OS: baremetal")
  IMCFLOW_ADDR = 0x80000000
  IMCFLOW_LEN = DevConfig.IMCFLOW_ADDR_SIZE
  INT_ACK_GEN_ADDR = 0
  INT_ACK_GEN_LEN = 0
  big_imem = os.getenv("IMCFLOW_BIG_IMEM", "").lower() in ("1", "true", "yes")
  if big_imem:
    RESET_GEN_ADDR = 0x80000000 + 270464 + 4
  else:
    RESET_GEN_ADDR = 0x80000000 + 266368 + 4
  MEASURE_POWER = False
elif (os.getenv("IMCFLOW_HOST_OS") == "linux"):
  print("IMCFLOW_HOST_OS: linux")
  IMCFLOW_ADDR = os.environ["IMCFLOW_ADDR"]
  IMCFLOW_LEN = os.environ["IMCFLOW_LEN"]
  INT_ACK_GEN_ADDR = os.environ["INT_ACK_GEN_ADDR"]
  INT_ACK_GEN_LEN = os.environ["INT_ACK_GEN_LEN"]
  RESET_GEN_ADDR = 0xa0130000
else:
  raise ValueError(f"Unsupported IMCFLOW_HOST_OS: {os.getenv('IMCFLOW_HOST_OS')}")

# --- DMM in-kernel power measurement (IMCFLOW_MEASURE_POWER, default OFF) ---
# Ported from origin/power. When OFF (default) NO dmm_* code is emitted so the
# generated C is byte-identical to stock. When ON (linux only) the kernel brackets
# EXACTLY its own SET_RUN compute invoke with a non-blocking dmm_start_current_now()
# right before SET_RUN and a blocking dmm_get_result_now()x N + dmm_close() right
# after wait_for_freerun_done()/wait_imcflow_interrupt() returns. Because the kernel
# itself brackets the array-active window there is no host<->DMM clock-sync problem.
MEASURE_POWER = os.getenv("IMCFLOW_MEASURE_POWER", "0").lower() in ("1", "true", "yes")

class PowerMeasurePhase(Enum):
  MODEL = 0    # one Start at region1 entry .. one End at last-region exit (whole model)
  REGION = 1   # one Start/End pair per region kernel, bracketing its single SET_RUN exec
  TILE = 2     # one Start/End pair per tile invoke

# Which granularity to bracket. REGION is the right choice for the STEP_FREERUN
# single-conv power sweep: each region kernel has exactly ONE SET_RUN compute invoke
# (our tree's generateInvokeCode; the program-load uses SET_PROGRAM_CODE via
# generatePolicyUpdateCode and is intentionally NOT bracketed), so REGION yields
# exactly ONE dmm_start/dmm_get pair around the real compute.
POWER_MEASURE_PHASE = PowerMeasurePhase.REGION

# Per-DMM config. DMM_NAMES selects which supply rails to sample. For the imcflow
# board: VDD = digital core, DDA = analog/ADC rail (rises only when STEPs actually
# retire -> the discriminator for "crossbar converting"), DDC = crossbar rail.
#
# KNOBS (must be tuned so the DMM window ~= the compute window):
#   nplc          integration time per sample in power-line-cycles. 0.001 = ~fastest
#                 (~a few us/sample on a 34410A-class DMM).
#   interval_s    inter-sample interval; -1 == MIN (back-to-back, fastest sampling).
#   sample_count  how many samples dmm_get_result_now() BLOCKS for. dmm_start is
#                 NON-blocking; dmm_get returns only after this many samples land.
#                 So (sample_count * per_sample_time) must be >= the compute window,
#                 otherwise dmm_get returns before compute finishes and you clip the
#                 window; if it is much larger, dmm_get blocks after compute ends and
#                 you integrate idle current. origin/power default = 50000 samples,
#                 which at nplc=0.001 spans ~a large fraction of a 1s STEP_FREERUN
#                 burst. With STEP_FREERUN_HOLD_SEC the compute busy-holds a fixed
#                 wall time, so size sample_count to match that hold (raise it if the
#                 hold is long; lower it if the fed conv is short).
#   curr_range    ammeter range in Amps (0.1 A here).
#   reset         1 -> reset the DMM before this measurement.
if POWER_MEASURE_PHASE in (PowerMeasurePhase.REGION, PowerMeasurePhase.TILE,
                           PowerMeasurePhase.MODEL):
  DMM_NAMES       = ["VDD", "DDA", "DDC"]
  NPLCs           = [0.001, 0.001, 0.001]
  INTERVALs       = [-1, -1, -1]
  SAMPLE_COUNTs   = [50000, 50000, 50000]
  CURR_RANGEs     = [0.1, 0.1, 0.1]
  RESETs          = [1, 1, 1]
  OFNAME_POSTFIXs = ["vdd", "dda", "ddc"]
else:
  raise ValueError(f"Unsupported POWER_MEASURE_PHASE: {POWER_MEASURE_PHASE}")

# Device paths
IMCFLOW_DEVICE = "/dev/uio5"
INT_ACK_GEN_DEVICE = "/dev/uio4"
# Code generation constants
CONST_TAGS = ["weight", "bias", "fused_scale", "fused_bias", "min", "max", "threshold", "scale", "config"]
ALIGNMENT_BYTES = 32

# Polling configuration
USE_POLLING = True
MAX_POLL_COUNT = 20000  # Maximum polling iterations before auto-termination

def getInnerNodeID(graph_node_id):
  if isinstance(graph_node_id, tuple):
    return graph_node_id[1]
  else:
    return graph_node_id

def align_to_n_bytes(size, n_bytes):
  if (size % n_bytes) != 0:
    size = (size // n_bytes + 1) * n_bytes
  return size


def dtype_to_cpp(dtype: str) -> str:
  mapping = {
      "float64": "double",
      "float32": "float",
      "float": "float",
      "uint32": "uint32_t",
      "int32": "int32_t",
      "uint16": "uint16_t",
      "int16": "int16_t",
      "uint8": "uint8_t",
      "int8": "int8_t",
  }

  # if dtype not in mapping: print(dtype)
  return mapping.get(dtype, "unknown_type")


class CodeWriter:
  def __init__(self, indent_str="  "):
    self.lines = []
    self.indent_str = indent_str
    self.indent_level = 0

  def nextIndent(self):
    self.indent_level += 1

  def prevIndent(self):
    self.indent_level -= 1

  def write(self, line=""):
    for line_ in line.split("\n"):
      if len(line_) > 0:
        self.lines.append(
            f"{self.indent_str * self.indent_level}{line_}")

  def get_code(self):
    return "\n".join(self.lines)

  def __str__(self):
    return self.get_code()

  def __add__(self, other):
    if isinstance(other, CodeWriter):
      # Apply current indentation level to incoming lines
      if self.indent_level > 0:
        indent_prefix = self.indent_str * self.indent_level
        indented_lines = [indent_prefix + line for line in other.lines]
        self.lines.extend(indented_lines)
      else:
        self.lines.extend(other.lines)
      return self
    elif isinstance(other, str):
      self.write(other)
      return self

def makeBaseAddrName(block):
  if isinstance(block.id, str):
    return f"{block.id.upper()}_BASE_ADDR"

  # Get first TensorEdge from edges list
  if isinstance(block.id, TensorEdge) or isinstance(block.id, List):
    edge = block.edges[0]  # Use first edge for naming
    graph_node_id = imcflow_transform.getInnerNodeID(edge.src_id.graph_node_id)
    node_id_str = str(graph_node_id).replace("-", "m")
    if edge.src_id.tensor_type in CONST_TAGS:
      return f"{edge.src_id.tensor_type.upper()}_{node_id_str}_BASE_ADDR"
    else:
      return f"{edge.dst_id.tensor_type.upper()}_{node_id_str}_BASE_ADDR"

  raise ValueError("Wrong data block type!")

def makeConstArrayDecl(func, func_name, target_func):
  params = {}
  class ConstantCollector(tvm.relay.ExprVisitor):
      def __init__(self):
          super().__init__()
          self.cnt=0

      def visit_constant(self, const):
          # constant 이름 생성 (symbol 기반)
          node_id = getInnerNodeID(imcflow_transform.getNodeID(const))
          if node_id is None:
            node_id = hash(const.__repr__()) # use a fallback hash if node_id is not found
          name = f"imcflow_{func_name}_const_{self.cnt}"
          params[String(name)] = const.data
          DevConfig().ImcflowFuncMap[func_name].const_name_map[node_id] = String(name)
          self.cnt += 1
          super().visit_constant(const)

  collector = ConstantCollector()
  collector.visit(target_func)

  code = CodeWriter()
  for const_name, array in params.items():
    dtype = dtype_to_cpp(array.dtype)
    shape = array.shape
    size = 1
    for dim in shape:
      size *= dim
    code += f"static const {dtype} {const_name}[] __attribute__((aligned(16))) = {{"
    array_values = array.asnumpy().flatten()
    for i, val in enumerate(array_values):
      if i % 16 == 0:
        code += "\n  "
      code += f"{val}, "
    code += "\n};\n\n"

  return code


def getConstantIdx(func, node_id):
  """
  Get the index of the constant node in the function by its inner node ID.
  parameters:
    func    : relay.Function
    node_id : inner node ID of the constant node
  """
  node_id_to_constant_id = {}

  class _Visitor(tvm.relay.ExprVisitor):
    def __init__(self):
      super().__init__()
      self.Cnt = 0

    def visit_constant(self, const):
      node_id = getInnerNodeID(imcflow_transform.getNodeID(const))
      node_id_to_constant_id[node_id] = self.Cnt
      self.Cnt = self.Cnt + 1
      super().visit_constant(const)

  _Visitor().visit(func)
  return node_id_to_constant_id[node_id]


def getCInputVarName(func_name, data_block):
  node_map = CustomIDToNode()

  # Get first edge from edges list (handles both single TensorEdge and List[TensorEdge])
  assert data_block.edges, "data_block must have at least one TensorEdge to get C input var name"

  edge = data_block.edges[0]
  graph_node_inner_id = imcflow_transform.getInnerNodeID(edge.src_id.graph_node_id)

  node_type = node_map[graph_node_inner_id]
  if isinstance(node_type, Var):
    return node_type.name_hint
  elif isinstance(node_type, Constant):
    data_type = dtype_to_cpp(node_type.checked_type.dtype)
    node_id = getInnerNodeID(imcflow_transform.getNodeID(node_type))
    if node_id is None:
      node_id = hash(node_type.__repr__())  # use a fallback hash if node_id is not found
    const_name = DevConfig().ImcflowFuncMap[func_name].const_name_map[node_id]
    return f"(({data_type}*)({const_name}))"
  else:
    raise ValueError(f"Invalid node_type!: {node_type}")

def getObjectFileName(data_block, func_name=None):
  assert isinstance(data_block.id, str), "data_block.id must be string to get object file name"
  if func_name:
    return f"_binary_{func_name}_{data_block.id}_bin"
  else:
    return f"_binary_{data_block.id}_bin"


class KernelCodeGenerator:
  """Code generator for IMCFlow kernel functions."""

  def __init__(self, func_name, func, os="linux"):
    """
    Initialize the kernel code generator.

    Parameters:
      func_name: Name of the function to generate code for
      func: Relay function (wrap function)
      os: Target OS ("linux" or "baremetal")
    """
    self.func_name = func_name
    self.func = func
    self.os = os

    # Get target function info
    self.target_func_info = DevConfig().ImcflowFuncMap.get(func_name, None)
    if self.target_func_info is None:
      raise ValueError(f"Function {func_name} not found in ImcflowFuncMap")
    self.target_func = self.target_func_info.func_node

    # Get data blocks
    self.compiled_blocks = ImcflowDeviceConfig().DataBlocks[func_name]["compiled"]
    self.compiled_per_tile_blocks = ImcflowDeviceConfig().DataBlocks[func_name]["compiled_per_tile"]
    self.const_blocks = ImcflowDeviceConfig().DataBlocks[func_name]["const"]
    self.input_blocks = ImcflowDeviceConfig().DataBlocks[func_name]["input"]
    self.output_blocks = ImcflowDeviceConfig().DataBlocks[func_name]["output"]

    # Initialize base address macros
    self.base_address_macros = {
        "IMCFLOW_ADDR": IMCFLOW_ADDR,
        "IMCFLOW_LEN": IMCFLOW_LEN,
        "INT_ACK_GEN_ADDR": INT_ACK_GEN_ADDR,
        "RESET_GEN_ADDR": RESET_GEN_ADDR,
        "INT_ACK_GEN_LEN": INT_ACK_GEN_LEN,
        "IMCFLOW_DEVICE": f'"{IMCFLOW_DEVICE}"',
        "INT_ACK_GEN_DEVICE": f'"{INT_ACK_GEN_DEVICE}"',
        "SET_IDLE_CODE": 0,
        "SET_RUN_CODE": 1,
        "SET_PROGRAM_CODE": 2,
        "STATE_REG_IDX": 0,
        "PC_REG_IDX": 2,
        "INTR_DONE_REG_IDX": 7,
        "INODE_PC_START_P1_ENUM_VAL": 0,
        "INODE_PC_START_EXTERN_ENUM_VAL": 1,
        "INODE_PC_START_P0_ENUM_VAL": 2,
        "INODE_NUM": ImcflowDeviceConfig().INODE_NUM,
    }

    # Get input/output node types
    self.input_nodes = [n for n in imcflow_transform.getInputNodesOfFunc(func)]
    self.input_node_types = [n.checked_type.dtype for n in self.input_nodes]
    self.output_node = imcflow_transform.getOutputNodeOfFunc(func)
    self.output_node_types = self._node_types_to_list(self.output_node)

  def _node_types_to_list(self, node):
    """Extract dtype(s) from a node's checked_type into a list."""
    checked_type = node.checked_type
    if isinstance(checked_type, TupleType):
      return [field.dtype for field in checked_type.fields]
    elif isinstance(checked_type, TensorType):
      return [checked_type.dtype]
    else:
      raise TypeError(f"unsupported node type {checked_type.__class__}")

  def generateRetryMacros(self):
    """Generate retry control macros."""
    return ("""
#ifndef RETRY_DISABLE
#ifndef MAX_RETRY_COUNT
#define MAX_RETRY_COUNT 3
#endif
#endif
""")

  def generateRetryCheck(self, location_label):
    """Generate retry check code after a wait call.
    On failure: cleanup device pointers, increment retry count, continue loop.
    With RETRY_DISABLE: exit(1) on failure (original behavior).
    """
    code = CodeWriter()
    code += f"#ifndef RETRY_DISABLE\n"
    code += f"if (_wait_rc != 0) {{\n"
    code.nextIndent()
    code += f'fprintf(stderr, "[RETRY] Timeout at {location_label}, attempt %d/%d\\n", _retry_count+1, MAX_RETRY_COUNT);\n'
    if self.os == "linux":
      # Drain + ack any pending interrupt BEFORE tearing down / re-arming. On the
      # timeout path the success-path acks (generate_ack + INTR_DONE) were skipped,
      # so a genuinely-fired-but-late edge would otherwise leave the UIO count and
      # the IP interrupt asserted; the next attempt's enable_imcflow_interrupt
      # would then stack on stale pending state and desync across kernels. Clear
      # the IP-level interrupt (int_ack_gen + INTR_DONE reg) while the mmaps are
      # still valid, i.e. before generateDevicePointerCleanup() munmaps them.
      code += "generate_ack(int_ack_gen_pointer);\n"
      code += "npu_pointer[INTR_DONE_REG_IDX] = 1;\n"
    code += self.generateDevicePointerCleanup()
    code += f"_retry_count++;\n"
    code += f"continue;\n"
    code.prevIndent()
    code += f"}}\n"
    code += f"#else\n"
    code += f"if (_wait_rc != 0) {{\n"
    code += f'  fprintf(stderr, "[TIMEOUT] {location_label} failed (retry disabled)\\n");\n'
    code += self.generateDevicePointerCleanup()
    code += f"  g_imcflow_kernel_failed = 1;\n"
    code += f"  return;\n"
    code += f"}}\n"
    code += f"#endif\n"
    return code

  def generateHeader(self):
    """Generate C header includes."""
    # Stage-heartbeat instrumentation (IMCFLOW_STAGE_HB, default OFF). When ON,
    # IMCFLOW_STAGE_HB(msg) writes an unbuffered stderr line AND appends+fsyncs a
    # line to /var/volatile/imcflow_stage_hb.txt on the board, so a host-side
    # wedge monitor SSH-polling that file can record the LAST stage reached even
    # after the SoC hard-wedges (stderr may be lost at the freeze). Localizes the
    # region3 kernel-entry wedge (reset / warmup / compiled-block / const-block /
    # policy-update / invoke). Host-side prints ONLY -- emits no accelerator
    # (inode/imce) code, so region cpp/bin blobs stay byte-identical. When OFF the
    # macro expands to nothing -> byte-identical to stock.
    stage_hb_macro = ("""
// --- IMCFLOW_STAGE_HB instrumentation (opt-in) ---
#define IMCFLOW_STAGE_HB(msg) do { \\
  fprintf(stderr, "[STAGE] %s\\n", (msg)); fflush(stderr); \\
  int _hbfd = open("/var/volatile/imcflow_stage_hb.txt", O_WRONLY|O_CREAT|O_APPEND, 0644); \\
  if (_hbfd >= 0) { \\
    char _hbbuf[256]; int _n = snprintf(_hbbuf, sizeof(_hbbuf), "%s\\n", (msg)); \\
    if (_n > 0) { ssize_t _w = write(_hbfd, _hbbuf, (size_t)_n); (void)_w; } \\
    fsync(_hbfd); close(_hbfd); \\
  } \\
} while (0)
""") if os.environ.get("IMCFLOW_STAGE_HB", "") not in ("", "0") else ""
    return ("""
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/wait.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>
#include <dlpack/dlpack.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <unistd.h>
""" + stage_hb_macro + ('#include "dmm_measure.h"\n' if (self.os == "linux" and MEASURE_POWER) else "") + """
// Global failure flag: set by kernel on timeout, checked by host loop
extern volatile int g_imcflow_kernel_failed;
""")

  def generateInterruptUtilities(self):
    """Generate interrupt handling utility functions."""
    return ("""
static inline void enable_imcflow_interrupt(int fd)
{
  uint32_t info = 1;
  ssize_t nb = write(fd, &info, sizeof(info));
  if (nb != (ssize_t)sizeof(info)) {
    perror("write failed");
    close(fd);
    exit(1);
  }
}

static inline int wait_imcflow_interrupt(int fd, volatile uint32_t* npu_pointer)
{
  uint32_t info;
  fd_set readfds;
  struct timeval timeout;

  // Defense against a lost/already-latched interrupt edge: the STATE register is
  // the ground truth for completion, the UIO interrupt is only a wake hint. If
  // the op already reached IDLE (edge fired before we armed/waited, or was never
  // delivered), return immediately instead of blocking on an edge that will
  // never come. This is the primary fix for the ~sample-46 chip wedge: a single
  // missed UIO edge previously hung forever with no status cross-check.
  if (npu_pointer[STATE_REG_IDX] == SET_IDLE_CODE) {
    return 0;
  }

  FD_ZERO(&readfds);
  FD_SET(fd, &readfds);

  timeout.tv_sec = 1;
  timeout.tv_usec = 0;

  int ret = select(fd + 1, &readfds, NULL, NULL, &timeout);
  if (ret == 0) {
    // Interrupt did not arrive within 1s. Do NOT declare failure yet — the edge
    // may have been missed while the compute actually finished. Fall back to
    // polling the STATE register (bounded, MAX_POLL_COUNT) so a lost edge cannot
    // wedge the run. Only if the array is genuinely not IDLE do we return -1.
    fprintf(stderr, "WARN: Interrupt timeout (1s) - falling back to STATE-register poll\\n");
    return wait_for_idle(npu_pointer);
  } else if (ret < 0) {
    perror("select failed");
    // select error is not proof the op failed; cross-check the STATE register.
    return wait_for_idle(npu_pointer);
  }

  ssize_t nb = read(fd, &info, sizeof(info));
  if (nb != (ssize_t)sizeof(info)) {
    perror("read interrupt failed");
    return wait_for_idle(npu_pointer);
  }
  return 0;
}

static inline void generate_ack(uint32_t* int_ack_gen)
{
  int_ack_gen[0] = 0b1;
}
""")

  def generatePollingUtilities(self):
    """Generate polling utility functions for non-interrupt based synchronization."""
    return ("""
// Poll until ImcFlow returns to IDLE state
#define POLL_LOG_INTERVAL 1000
#define MAX_POLL_COUNT 20000
static int wait_for_idle(volatile uint32_t* npu_pointer) {
  uint32_t poll_count = 0;
  uint32_t state;

  fprintf(stderr,"[POLLING] Waiting for ImcFlow to return to IDLE state...\\n");

  while (1) {
    state = npu_pointer[STATE_REG_IDX];

    if (state == SET_IDLE_CODE) {
      fprintf(stderr,"[POLLING] Operation complete! (polled %u times)\\n", poll_count);
      return 0;
    }

    poll_count++;

    // Check for timeout
    if (poll_count >= MAX_POLL_COUNT) {
      fprintf(stderr,"[POLLING ERROR] Timeout after %u polls (state: 0x%x)\\n", poll_count, state);
      fprintf(stderr,"[POLLING ERROR] ImcFlow hardware appears to be stuck.\\n");
      return -1;
    }

    // Log progress every 1000 polls for debugging
    if (poll_count % POLL_LOG_INTERVAL == 0) {
      fprintf(stderr,"[POLLING] Still waiting... (poll count: %u, current state: 0x%x)\\n",
             poll_count, state);
      // [PROBE] region3-entry wedge localizer: dump all 8 ctrl_regs each interval.
      // STATE=0x1 busy + INTR_ID/INTR_DONE unchanged across polls => wedge is BEFORE
      // any inode's launch-end INTRT, i.e. stuck in imem/IMCU load + 255-barrier
      // (Phase A), not the Phase-B fused-add SEND/RECV. fflush: SoC goes SSH-dead
      // seconds after wedge, so unbuffered output is essential.
      fprintf(stderr,"[PROBE] ctrl_reg: STATE=0x%x CMD=0x%x INODE_PC[0..3]=0x%x 0x%x 0x%x 0x%x INTR_ID=0x%x INTR_DONE=0x%x\\n",
             npu_pointer[0],npu_pointer[1],npu_pointer[2],npu_pointer[3],
             npu_pointer[4],npu_pointer[5],npu_pointer[6],npu_pointer[7]);
      fflush(stderr);
    }
  }
}
""")

  def emitReset(self):
    """Generate code to reset the NPU state."""
    return f"reset_gen_pointer[0] = 1;\n"

  def emitWarmup(self):
    """Generate code to warm up the NPU state."""
    return ("""
// Warmup: clear timing data and run warmup routine
{
  int ret;

  // Step 1: Clear timing data
  ret = system("make -C /home/root/imcflow/xilinx/petalinux-csrc clear_time > /dev/null 2>&1");
  if (ret == -1) {
    fprintf(stderr, "Error: make clear_time failed to execute: %s\\n", strerror(errno));
  } else if (WIFEXITED(ret) && WEXITSTATUS(ret) != 0) {
    fprintf(stderr, "Error: make clear_time exited with status %d\\n", WEXITSTATUS(ret));
  } else if (WIFSIGNALED(ret)) {
    fprintf(stderr, "Error: make clear_time killed by signal %d\\n", WTERMSIG(ret));
  }

  // Step 2: Execute warmup target
  ret = system("make -C /home/root/imcflow/xilinx/petalinux-csrc warmup > /dev/null 2>&1");
  if (ret == -1) {
    fprintf(stderr, "Error: make warmup failed to execute: %s\\n", strerror(errno));
  } else if (WIFEXITED(ret) && WEXITSTATUS(ret) != 0) {
    fprintf(stderr, "Error: make warmup exited with status %d\\n", WEXITSTATUS(ret));
  } else if (WIFSIGNALED(ret)) {
    fprintf(stderr, "Error: make warmup killed by signal %d\\n", WTERMSIG(ret));
  }
}
    """)

  def generatePowerMeasureStart(self, t_idx=None):
    """Generate C code to start DMM current measurement (NON-blocking).

    Ported verbatim from origin/power. Emits a dmm_config_t array and a single
    dmm_start_current_now() call that returns immediately; the DMM then samples in
    the background while the accelerator runs the SET_RUN compute. Paired with
    generatePowerMeasureEnd() which blocks on dmm_get_result_now(). Only ever
    called when self.os == 'linux' and MEASURE_POWER is set."""
    assert self.os == "linux" and MEASURE_POWER, "Power measurement code should only be generated for Linux with MEASURE_POWER enabled"

    n_dmms = len(DMM_NAMES)
    code = CodeWriter()

    if t_idx is not None:
      code += f"// --- Power measurement start (tile {t_idx}) ---\n"
    else:
      code += f"// --- Power measurement start ---\n"

    code += f"fprintf(stderr, \"[DMM] starting measurement...\\n\");\n"

    code += "{\n"
    code.nextIndent()
    code += f"dmm_config_t dmm_cfgs[{n_dmms}] = {{\n"
    code.nextIndent()

    for i in range(n_dmms):
      interval_val = -1 if INTERVALs[i] == "MIN" else INTERVALs[i]

      if t_idx is not None:
        ofname = f"{self.func_name}_tile{t_idx}_{OFNAME_POSTFIXs[i]}.txt"
        server_ofname = f"{self.func_name}_tile{t_idx}_{OFNAME_POSTFIXs[i]}_server.txt"
      else:
        ofname = f"{self.func_name}_{OFNAME_POSTFIXs[i]}.txt"
        server_ofname = f"{self.func_name}_{OFNAME_POSTFIXs[i]}_server.txt"

      trailing = "," if i < n_dmms - 1 else ""
      code += "{\n"
      code.nextIndent()
      code += f".name = \"{DMM_NAMES[i]}\",\n"
      code += f".nplc = {NPLCs[i]},\n"
      code += f".interval_s = {interval_val},\n"
      code += f".sample_count = {SAMPLE_COUNTs[i]},\n"
      code += f".curr_range = {CURR_RANGEs[i]},\n"
      code += f".reset = {RESETs[i]},\n"
      code += f".ofname = \"{ofname}\",\n"
      code += f".server_ofname = \"{server_ofname}\",\n"
      code.prevIndent()
      code += "}" + trailing + "\n"
    code.prevIndent()
    code += "};\n"
    code += f"if (dmm_start_current_now({n_dmms}, dmm_cfgs) != 0) {{\n"
    code += f"  fprintf(stderr, \"ERROR: dmm_start_current_now failed: %s\\n\", dmm_last_error());\n"
    code += f"  exit(1);\n"
    code += f"}}\n"

    if t_idx is not None:
      code += f"fprintf(stderr, \"[DMM] measurement started (tile {t_idx})\\n\");\n"
    else:
      code += f"fprintf(stderr, \"[DMM] measurement started \\n\");\n"

    code.prevIndent()
    code += "}\n"
    return code

  def generatePowerMeasureEnd(self, t_idx=None):
    """Generate C code to wait for DMM results and close (BLOCKING).

    Ported verbatim from origin/power. Calls dmm_get_result_now() once per DMM
    (each blocks until its configured sample_count samples arrive), prints the
    per-rail average current, then dmm_close(). Placed immediately after the
    SET_RUN compute wait returns so the sampled window spans exactly the compute."""
    assert self.os == "linux" and MEASURE_POWER, "Power measurement code should only be generated for Linux with MEASURE_POWER enabled"

    n_dmms = len(DMM_NAMES)
    code = CodeWriter()
    if t_idx is not None:
      code += f"// --- Power measurement end (tile {t_idx}) ---\n"
    else:
      code += f"// --- Power measurement end ---\n"
    if t_idx is not None:
      code += f"fprintf(stderr, \"[DMM] getting measurement result... (tile {t_idx})\\n\");\n"
    else:
      code += f"fprintf(stderr, \"[DMM] getting measurement result... \\n\");\n"
    code += "{\n"
    code.nextIndent()
    code += f"char dmm_name[64];\n"
    code += f"double dmm_avg;\n"
    code += f"int dmm_count;\n"
    code += f"for (int dmm_i = 0; dmm_i < {n_dmms}; dmm_i++) {{\n"
    code.nextIndent()
    code += f"int rc = dmm_get_result_now(dmm_name, sizeof(dmm_name), &dmm_avg, &dmm_count);\n"
    code += f"if (rc == -2) {{\n"
    code += f"  fprintf(stderr, \"DMM ERROR [%s]: %s\\n\", dmm_name, dmm_last_error());\n"
    code += f"  continue;\n"
    code += f"}} else if (rc != 0) {{\n"
    code += f"  fprintf(stderr, \"ERROR: dmm_get_result_now failed: %s\\n\", dmm_last_error());\n"
    code += f"  dmm_close();\n"
    code += f"  exit(1);\n"
    code += f"}}\n"
    code += f"fprintf(stderr, \"[DMM] [%s] avg = %.9g A  (%d samples)\\n\", dmm_name, dmm_avg, dmm_count);\n"
    code.prevIndent()
    code += f"}}\n"
    code += f"dmm_close();\n"
    if t_idx is not None:
      code += f"fprintf(stderr, \"[DMM] measurement ended (tile {t_idx})\\n\");\n"
    else:
      code += f"fprintf(stderr, \"[DMM] measurement ended \\n\");\n"
    code.prevIndent()
    code += "}\n"
    return code

  def generateDevicePointerSetup(self):
    """Generate device pointer setup code based on OS."""
    if self.os == "linux":
      return ("""
  int npu_fd = open(IMCFLOW_DEVICE, O_RDWR);
  if (npu_fd < 0) {
    perror("npu UIO cannot be opened");
    exit(1);
  }

  int int_ack_gen_fd = open(INT_ACK_GEN_DEVICE, O_RDWR);
  if (int_ack_gen_fd < 0) {
    perror("interrupt ack gen UIO cannot be opened");
    close(npu_fd);
    exit(1);
  }

  size_t npu_len = (size_t) IMCFLOW_LEN;
  uint32_t *npu_pointer = (uint32_t *) mmap(NULL, npu_len, PROT_WRITE | PROT_READ, MAP_SHARED, npu_fd, 0);
  if (npu_pointer == MAP_FAILED) {
    perror("npu_pointer mmap error");
    close(npu_fd);
    close(int_ack_gen_fd);
    exit(1);
  }

  size_t int_ack_gen_len = (size_t)INT_ACK_GEN_LEN;
  uint32_t *int_ack_gen_pointer = (uint32_t*) mmap(NULL, int_ack_gen_len, PROT_WRITE | PROT_READ, MAP_SHARED, int_ack_gen_fd, 0);
  if (int_ack_gen_pointer == MAP_FAILED) {
    perror("int_ack_gen_pointer mmap error");
    munmap(npu_pointer, npu_len);
    close(npu_fd);
    close(int_ack_gen_fd);
    exit(1);
  }

  int reset_gen_fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (reset_gen_fd < 0) {
    perror("Cannot open /dev/mem for reset generator");
    munmap(npu_pointer, npu_len);
    close(npu_fd);
    munmap(int_ack_gen_pointer, int_ack_gen_len);
    close(int_ack_gen_fd);
    exit(1);
  }

  off_t reset_gen_page_offset = RESET_GEN_ADDR & ~(sysconf(_SC_PAGE_SIZE) - 1);
  size_t reset_gen_map_size = sysconf(_SC_PAGE_SIZE);
  void* reset_gen_map_base = mmap(NULL, reset_gen_map_size, PROT_READ | PROT_WRITE, MAP_SHARED, reset_gen_fd, reset_gen_page_offset);
  if (reset_gen_map_base == MAP_FAILED) {
    perror("reset_gen mmap error");
    munmap(npu_pointer, npu_len);
    close(npu_fd);
    munmap(int_ack_gen_pointer, int_ack_gen_len);
    close(int_ack_gen_fd);
    close(reset_gen_fd);
    exit(1);
  }
  volatile uint32_t* reset_gen_pointer = (volatile uint32_t*)((char*)reset_gen_map_base + (RESET_GEN_ADDR - reset_gen_page_offset));
  """)
    elif self.os == "baremetal":
      return (f"""
    uint32_t* npu_pointer = (uint32_t*)IMCFLOW_ADDR;
    uint32_t* int_ack_gen_pointer = (uint32_t*)INT_ACK_GEN_ADDR;
    uint32_t* reset_gen_pointer = (uint32_t*)RESET_GEN_ADDR;
""")
    else:
      raise ValueError("Unsupported OS type for device pointer setup!")

  def generatePolicyUpdateCode(self):
    """Generate policy update code."""
    out = [
      "// Set the inode pc to 0 and run.",
      "for(int i=0; i<INODE_NUM; i++) {",
      "  npu_pointer[(PC_REG_IDX + i)] = (INODE_PC_START_EXTERN_ENUM_VAL << 30 + 0);",
      "}",
      "enable_imcflow_interrupt(npu_fd);" if self.os == "linux" else "",
      " npu_pointer[STATE_REG_IDX] = SET_PROGRAM_CODE;",
      "int _wait_rc = wait_imcflow_interrupt(npu_fd, npu_pointer);" if self.os == "linux" else ("int _wait_rc = wait_for_idle(npu_pointer);" if USE_POLLING else "int _wait_rc = 0;"),
      "generate_ack(int_ack_gen_pointer);" if self.os == "linux" else "",
      "npu_pointer[INTR_DONE_REG_IDX] = 1;",
    ]
    return "\n".join(out) + "\n"

  def generateInvokeCode(self):
    """Generate NPU invoke code."""
    out = [
      "for(int i=0; i<INODE_NUM; i++) {",
      "  npu_pointer[(PC_REG_IDX + i)] = (INODE_PC_START_P1_ENUM_VAL << 30 + 0);",
      "}",
      "enable_imcflow_interrupt(npu_fd);" if self.os == "linux" else "",
      "npu_pointer[STATE_REG_IDX] = SET_RUN_CODE;",
      "_wait_rc = wait_imcflow_interrupt(npu_fd, npu_pointer);" if self.os == "linux" else ("_wait_rc = wait_for_idle(npu_pointer);" if USE_POLLING else "_wait_rc = 0;"),
      "generate_ack(int_ack_gen_pointer);" if self.os == "linux" else "",
        "npu_pointer[INTR_DONE_REG_IDX] = 1;"
    ]
    return "\n".join(out) + "\n"

  def generateDevicePointerCleanup(self):
    """Generate device pointer cleanup code."""
    if self.os == "linux":
      return ("""
  // Cleanup device pointer
  munmap(npu_pointer, npu_len);
  close(npu_fd);
  munmap(int_ack_gen_pointer, int_ack_gen_len);
  close(int_ack_gen_fd);
  munmap(reset_gen_map_base, reset_gen_map_size);
  close(reset_gen_fd);
  """)
    else:
      return ""

  def _get_transfer_loop_params(self, block, tile_idx):
    """
    Calculate loop parameters for data transfer.
    Returns (loop_start, loop_end).
    """
    if block.tiling_info is not None:
      # Tiled transfer
      tiling_info = block.tiling_info
      loop_start = tiling_info.c_var_offsets[tile_idx]
      loop_end = loop_start + tiling_info.c_var_sizes[tile_idx]
    else:
      # Regular transfer
      size = align_to_n_bytes(block.size, ALIGNMENT_BYTES)
      loop_start = 0
      loop_end = math.ceil(size/4)
    return loop_start, loop_end

  def generateToNpuTransferCode(self, blocks, tile_idx=None):
    """Generate code to transfer data to NPU memory."""

    def _appendLoopForObjectFileTransfer(code, block, base_address_name, func_name, tile_idx=None):
      # Binary object file transfer
      var_prefix = getObjectFileName(block, func_name)
      src_var = f"{var_prefix}_start"
      if tile_idx is None:
        loop_end = f"(size_t)({var_prefix}_end-{var_prefix}_start)"
        code += f"for(int i=0; i<{loop_end}; i++){{\n"
        code += f"  npu_pointer[({base_address_name} / 4) + i] = ((uint32_t*){src_var})[i];\n"
        code += f"}}\n"
      else:
        code += f"  npu_pointer[({base_address_name} / 4)] = ((uint32_t*){src_var})[{tile_idx}];\n"
        code += f"for(int i=1; i<8; i++){{\n"
        code += f"  npu_pointer[({base_address_name} / 4) + i] = 0;\n"
        code += f"}}\n"
      return code

    def _appendLoopForCVarTransfer(code, block, base_address_name, func_name, tile_idx=None):
      # C Var transfer
      loop_start, loop_end = self._get_transfer_loop_params(block, tile_idx)
      src_var = getCInputVarName(func_name, block)

      code += f"for(int i=0; i<{loop_end-loop_start}; i++){{\n"
      code += f"  npu_pointer[({base_address_name} / 4) + i] = ((uint32_t*){src_var})[i + {loop_start//4}];\n"
      code += f"}}\n"
      return code

    _hb_on = os.environ.get("IMCFLOW_STAGE_HB", "") not in ("", "0")
    # Host-side MMIO write-ordering barrier between block transfers (root-cause fix
    # for the region3 chip wedge; see imcflow.mmio_block_barrier_usec). -1 == OFF ->
    # emit nothing -> byte-identical. Accelerator blobs untouched (host-side only).
    _mmio_barrier = mmio_block_barrier_usec()
    code = CodeWriter()
    code += "// Transfer data into NPU memory\n"
    for block in blocks:
      base_address = block.base_address
      base_address_name = makeBaseAddrName(block)
      self.base_address_macros.update({base_address_name: base_address})

      # Per-block stage heartbeat (opt-in) so a mid-transfer SoC wedge localizes to
      # the exact block whose MMIO write hung. Host-side print only.
      if _hb_on:
        try:
          _blk_name = str(block.id)
        except Exception:
          _blk_name = base_address_name
        _blk_name = _blk_name.replace('"', "'")[:120]
        code += f"IMCFLOW_STAGE_HB(\"{self.func_name}: xfer block {base_address_name} ({_blk_name}) sz={block.size}\");\n"

      # Add tiling comment if applicable
      if block.tiling_info is not None:
        code += f"// Transfer data [TILE:{tile_idx}]\n"
        code += f"fprintf(stderr,\"Transferring input block to NPU [TILE:{tile_idx}]\\n\");\n"

      # Determine source variable and loop parameters based on block type
      if isinstance(block.id, str):
        code = _appendLoopForObjectFileTransfer(code, block, base_address_name, self.func_name, tile_idx)
      else:
        code = _appendLoopForCVarTransfer(code, block, base_address_name, self.func_name, tile_idx)

      # MMIO write-ordering barrier AFTER this block's stores (drains the CPU store
      # buffer so the accelerator sees complete, in-order blocks; fixes the region3
      # host-side MMIO overrun wedge). usleep adds a real-time drain if needed.
      if _mmio_barrier >= 0:
        code += "__sync_synchronize();\n"
        if _mmio_barrier > 0:
          code += f"usleep({_mmio_barrier});\n"

    return code

  def generateFromNpuTransferCode(self, blocks, tile_idx=None):
    """Generate code to transfer data from NPU memory."""
    code = CodeWriter()
    code += "// Transfer data from NPU memory\n"
    for block in blocks:
      assert "func_out" in block.id.dst_id.tensor_type, "output data block must have 'func_out' in tensor_type"
      idx = block.id.dst_id.tensor_type.replace("func_out", "")
      base_address = block.base_address
      base_address_name = makeBaseAddrName(block)
      self.base_address_macros.update({base_address_name: base_address})

      # Add tiling comment if applicable
      if block.tiling_info is not None:
        code += f"// Transfer data [TILE:{tile_idx}]\n"
        code += f"fprintf(stderr,\"Transferring output block to out{idx} [TILE:{tile_idx}]\\n\");\n"

      # Get loop parameters
      loop_start, loop_end = self._get_transfer_loop_params(block, tile_idx)

      # Generate loop code
      code += f"for(int i=0; i<{loop_end-loop_start}; i++){{\n"
      code += f"  ((uint32_t*)out{idx})[i + {loop_start//4}] = npu_pointer[({base_address_name} / 4) + i];\n"
      code += f"}}\n"
    return code

  def generateBaseAddrMacros(self):
    """Generate base address macro definitions."""
    code = CodeWriter()
    for key, value in self.base_address_macros.items():
      if isinstance(value, int):
        code += f"#define {key} 0x{value:X}\n"
      else:
        code += f"#define {key} {value}\n"
    code += "\n"
    return code

  def generateExternLink(self):
    """Generate extern C linkage declarations for compiled blocks."""
    code = CodeWriter()
    code += 'extern "C" { \n'
    for block in (self.compiled_blocks + self.compiled_per_tile_blocks):
      if isinstance(block.id, str):
        filename = f"_binary_{self.func_name}_{block.id}_bin"
        code += f'  extern const int32_t {filename}_start[];\n'
        code += f'  extern const int32_t {filename}_end[];\n'
    code += '}\n'
    return code

  def generatePackedFuncWrapper(self):
    """Generate PackedFunc wrapper for CRT."""
    code = CodeWriter()
    code += "#ifdef __cplusplus\n"
    code += "extern \"C\"\n"
    code += "#endif\n"
    code += f"TVM_DLL int32_t {self.func_name}(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {{\n"
    code.nextIndent()
    code += "(void)resource_handle;\n"
    code += "if (num_args < 2) return -1;\n"

    num_in_args = len(self.input_node_types)
    num_out_args = len(self.output_node_types)

    # get input and output data pointers
    for idx in range(num_in_args):
      code += f"void* _in{idx} = (((TVMValue*)args)[{idx}].v_handle);\n"
      code += f"DLTensor* in{idx} = (DLTensor*)_in{idx};\n"
    for idx in range(num_out_args):
      code += f"void* _out{idx} = (((TVMValue*)args)[{idx + num_in_args}].v_handle);\n"
      code += f"DLTensor* out{idx} = (DLTensor*)_out{idx};\n"

    # call kernel function
    args_list = []
    for idx, i_type in enumerate(self.input_node_types):
      args_list.append(f"({dtype_to_cpp(i_type)}*)in{idx}->data")
    for idx, o_type in enumerate(self.output_node_types):
      args_list.append(f"({dtype_to_cpp(o_type)}*)out{idx}->data")
    code += f"{self.func_name}_kernel({', '.join(args_list)});\n"
    code += "if (g_imcflow_kernel_failed) return -1;\n"

    code += "(void)out_ret_value;\n"
    code += "if (out_ret_tcode) { *out_ret_tcode = kTVMArgInt; }\n"
    code += "return 0;\n"
    code.prevIndent()
    code += "}\n"
    return code

  def makeKernelDef(self):
    """Generate the complete kernel definition."""
    # Build function prototype
    proto_list = []
    for i, param in enumerate(self.func.params):
      impl_param = self.target_func.params[i]
      param_name = impl_param.name_hint if impl_param.name_hint else f"arg{i}"
      dtype = "float32"
      if hasattr(param, "checked_type") and isinstance(param.checked_type, TensorType):
        dtype = param.checked_type.dtype
        cpp_type = dtype_to_cpp(dtype)
        proto_list.append(f"{cpp_type}* {param_name}")

    for idx, node_type in enumerate(self.output_node_types):
      proto_list.append(f"{dtype_to_cpp(node_type)}* out{idx}")

    args_proto_type = ", ".join(proto_list)

    code = CodeWriter()
    code += self.generateHeader()
    code += self.generateRetryMacros()
    code += self.generateExternLink()
    code += makeConstArrayDecl(self.func, self.func_name, self.target_func)
    # Emit polling utilities (wait_for_idle) BEFORE interrupt utilities:
    # wait_imcflow_interrupt now calls wait_for_idle as its STATE-register
    # fallback, so the polling helper must be defined first. Emit it
    # unconditionally (not gated on USE_POLLING) since the linux/chip interrupt
    # path always needs it as the lost-edge safety net.
    code += self.generatePollingUtilities()
    code += self.generateInterruptUtilities()

    # Kernel function prototype and definition (C)
    code += f"void {self.func_name}_kernel({args_proto_type}) {{\n"
    code.nextIndent()
    code += f"fprintf(stderr,\"{self.func_name}_kernel called\\n\");\n"

    # Early exit if a previous kernel already failed
    code += "if (g_imcflow_kernel_failed) return;\n"

    # Retry loop start
    code += "#ifndef RETRY_DISABLE\n"
    code += "int _retry_count = 0;\n"
    code += "do {\n"
    code.nextIndent()
    code += "if (_retry_count > 0) {\n"
    code += f"  fprintf(stderr, \"[RETRY] {self.func_name}_kernel retry attempt %d/%d\\n\", _retry_count, MAX_RETRY_COUNT);\n"
    code += "}\n"
    code += "#endif\n"

    # Stage-heartbeat markers (IMCFLOW_STAGE_HB, default OFF -> emit nothing ->
    # byte-identical). Host-side only; localizes the region3 kernel-entry wedge.
    _hb_on = os.environ.get("IMCFLOW_STAGE_HB", "") not in ("", "0")
    def _hb(stage):
      return f"IMCFLOW_STAGE_HB(\"{self.func_name}: {stage}\");\n" if _hb_on else ""

    code += _hb("before device_pointer_setup")
    code += self.generateDevicePointerSetup()
    code += _hb("before reset")
    code += self.emitReset()
    code += _hb("after reset")
    if self.os == "linux":
      # Per-kernel warmup gate. emitWarmup() forks `make clear_time && make warmup`
      # (a full accelerator hard-reset + 16 warmup binaries) on EVERY kernel call.
      # warmup is GLOBAL (identical make target, not per-kernel state), so once per
      # run suffices; run_dataset_eval.sh already issues a single pre-run warmup.
      # Set IMCFLOW_NO_PERKERNEL_WARMUP=1 to drop the per-kernel warmup and measure
      # its cost (this re-applies the reverted 9857698bf, but as an opt-in switch so
      # it can be A/B'd against the warmup baseline for accuracy/wedge regressions).
      if os.environ.get("IMCFLOW_NO_PERKERNEL_WARMUP", "") not in ("", "0"):
        pass  # per-kernel warmup intentionally skipped
      else:
        code += _hb("before warmup")
        code += self.emitWarmup()
        code += _hb("after warmup")
    code += _hb("before compiled_blocks transfer")
    code += self.generateToNpuTransferCode(self.compiled_blocks) # inode instrunction + policy
    code += _hb("after compiled_blocks transfer / before const_blocks transfer")
    code += self.generateToNpuTransferCode(self.const_blocks) # constant
    code += _hb("after const_blocks transfer / before policy_update")
    code += self.generatePolicyUpdateCode() # start from pc 0, up to halt
    code += _hb("after policy_update")
    code += self.generateRetryCheck("policy_update")

    # Power measurement (IMCFLOW_MEASURE_POWER, default OFF -> nothing emitted ->
    # byte-identical). CRITICAL PLACEMENT: bracket ONLY the SET_RUN compute invoke
    # below -- NOT generatePolicyUpdateCode() above (that is the SET_PROGRAM_CODE
    # program/policy load, not the array compute). generateInvokeCode() emits the
    # single SET_RUN exec (busy-holds for the whole STEP_FREERUN burst when active),
    # so dmm_start() fires immediately before SET_RUN and dmm_get()/dmm_close() fire
    # immediately after the compute wait returns -> the DMM window == the compute.
    # NOTE: the FIRST invoke below (right after policy_update, BEFORE any input
    # transfer) runs with the array effectively idle -- STATE stays 0x0 and NO real
    # crossbar compute happens (chip-observed: bracketing here gave DDA==idle). The
    # REAL array-active compute is the per-tile invoke inside the tile loop, which
    # fires AFTER generateToNpuTransferCode(input_blocks). So power measurement must
    # bracket the TILE invoke, not this one.
    power_measure_active = self.os == "linux" and MEASURE_POWER

    code += _hb("before invoke")
    code += self.generateInvokeCode() # proceed up to halt (pre-input SET_RUN)
    code += _hb("after invoke / before poll")

    code += self.generateRetryCheck("invoke")

    # kernel tiling factor
    tile_factor = self.target_func_info.tiling_factor
    code += f"// Tiled execution with factor {tile_factor}\n"
    code += f"fprintf(stderr,\"Starting tiled execution with factor {tile_factor}\\n\");\n"
    for t_idx in range(tile_factor):
      code += f"fprintf(stderr,\"-- Tiled execution: TILE {t_idx} / {tile_factor} --\\n\");\n"
      code += self.generateToNpuTransferCode(self.compiled_per_tile_blocks, t_idx) # per-tile: cnt_base_addr
      code += self.generateToNpuTransferCode(self.input_blocks, t_idx) # input
      # Power measurement (IMCFLOW_MEASURE_POWER, default OFF -> byte-identical):
      # bracket the ACTUAL array-active compute -- this per-tile invoke fires with
      # input present so the crossbar really runs. dmm_start non-blocking right
      # before SET_RUN; dmm_get/close blocking right after the compute wait returns
      # -> the DMM window == the real compute window.
      if power_measure_active:
        code += self.generatePowerMeasureStart(t_idx)
      code += self.generateInvokeCode() # end of exec (SET_RUN compute, array active)
      if power_measure_active:
        code += self.generatePowerMeasureEnd(t_idx)
      code += self.generateRetryCheck(f"tile_{t_idx}_invoke")
      code += self.generateFromNpuTransferCode(self.output_blocks, t_idx) # output

    # Retry loop end + cleanup
    code += self.generateDevicePointerCleanup()
    code += "#ifndef RETRY_DISABLE\n"
    code += "break; // success\n"
    code.prevIndent()
    code += "} while (_retry_count <= MAX_RETRY_COUNT);\n"
    code += "if (_retry_count > MAX_RETRY_COUNT) {\n"
    code += '  fprintf(stderr, "[RETRY] Exhausted %d retries.\\n", MAX_RETRY_COUNT);\n'
    code += "  g_imcflow_kernel_failed = 1;\n"
    code += "  return;\n"
    code += "}\n"
    code += "#endif\n"

    code.prevIndent()
    code += '}\n'

    code += self.generatePackedFuncWrapper()
    code = self.generateBaseAddrMacros() + code

    return code


def makeKernelStartCode(func_name, func, os="linux"):
  """Generate kernel start code using KernelCodeGenerator class."""
  generator = KernelCodeGenerator(func_name, func, os)
  code = generator.makeKernelDef()
  return str(code)

def generate_invoke_code_for_subgraphs(mod):
  invoke_code_map = {}
  for func_name_var in mod.functions:
    func = mod[func_name_var.name_hint]
    if func.attrs and func.attrs.get("Compiler") == "imcflow":
      func_name = func_name_var.name_hint
      code = makeKernelStartCode(func_name, func)
      invoke_code_map[func_name] = code

  for fn, code in invoke_code_map.items():
    with open(f"{fn}.cc", "w") as f:
      f.write(code)

  return invoke_code_map

@tvm._ffi.register_func("relay.ext.imcflow.constant_updater")
def imcflow_constant_updater(expr, symbol):
    """
    Relay function에서 constant를 추출하여 반환
    """
    return dict()

@tvm._ffi.register_func("relay.ext.imcflow")
def imcflow_external_codegen(func: relay.Function):
  # Obtain the function name (global symbol) assigned by the partitioning pass
  func_name = func.attrs["global_symbol"] if hasattr(
      func, "attrs") and "global_symbol" in func.attrs else "imcflow_subgraph"

  # const_vars = list(DevConfig().ImcflowFuncMap[func_name].const_name_map.values())

  # Reuse existing kernel code generator
  code = makeKernelStartCode(func_name, func, DevConfig.HOST_OS)

  # Wrap as a CSourceModule so TVM can compile/link it with the rest of the MLF
  # Note: returning a CSourceModule is the standard for BYOC Python codegen.
  return tvm.runtime._ffi_api.CSourceModuleCreate(code, "cc", [String(func_name)], None)