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
import json
import re


def _env_flag(name, default=False):
  value = os.getenv(name)
  if value is None:
    return default
  normalized = value.strip().lower()
  if normalized in ("1", "true", "yes", "on"):
    return True
  if normalized in ("0", "false", "no", "off", ""):
    return False
  raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _env_int(name, default, minimum=None):
  value = int(os.getenv(name, str(default)))
  if minimum is not None and value < minimum:
    raise ValueError(f"{name} must be >= {minimum}, got {value}")
  return value


def _env_float(name, default, minimum=None):
  value = float(os.getenv(name, str(default)))
  if minimum is not None and value < minimum:
    raise ValueError(f"{name} must be >= {minimum}, got {value}")
  return value


def _power_function_order_key(func_name):
  """Return the Relay graph order encoded in an IMCFLOW function name.

  ``ImcflowFuncMap`` is populated from IRModule iteration and is not an
  execution-order contract.  The final ``main_<custom_id>`` component is the
  graph node id assigned while traversing the Relay program, so it provides a
  stable boundary for MODEL-scoped measurement.
  """
  match = re.search(r"_main_(\d+)$", func_name)
  if match:
    return (0, int(match.group(1)), func_name)
  # Keep code generation usable for hand-written names while making their
  # ordering deterministic.
  return (1, 0, func_name)


POWER_MEASURE_ENABLED = False
POWER_MEASURE_SCOPE = "REGION"
POWER_MEASURE_MODE = "now"
POWER_DMM_NAME = "DMM_GPIB3"
POWER_DMM_NPLC = 0.001
POWER_DMM_INTERVAL_S = -1.0
POWER_DMM_SAMPLE_COUNT = 50000
POWER_DMM_CURRENT_RANGE = 0.1
POWER_DMM_RESET = True
POWER_DMM_START_TIMEOUT_S = 30
POWER_DMM_RESULT_TIMEOUT_S = 300
POWER_SERVER_OUTPUT_PREFIX = "/tmp/imcflow_power"
REGION_TIMING_ENABLED = _env_flag("IMCFLOW_REGION_TIMING", False)

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
elif (os.getenv("IMCFLOW_HOST_OS") == "linux"):
  print("IMCFLOW_HOST_OS: linux")
  IMCFLOW_ADDR = os.environ["IMCFLOW_ADDR"]
  IMCFLOW_LEN = os.environ["IMCFLOW_LEN"]
  INT_ACK_GEN_ADDR = os.environ["INT_ACK_GEN_ADDR"]
  INT_ACK_GEN_LEN = os.environ["INT_ACK_GEN_LEN"]
  RESET_GEN_ADDR = 0xa0130000 
  POWER_MEASURE_ENABLED = _env_flag("IMCFLOW_MEASURE_POWER", False)
  if POWER_MEASURE_ENABLED:
    POWER_MEASURE_SCOPE = os.getenv("IMCFLOW_POWER_SCOPE", "REGION").strip().upper()
    if POWER_MEASURE_SCOPE not in ("MODEL", "REGION", "TILE"):
      raise ValueError(
          "IMCFLOW_POWER_SCOPE must be MODEL, REGION, or TILE, "
          f"got {POWER_MEASURE_SCOPE!r}")
    POWER_MEASURE_MODE = os.getenv("IMCFLOW_POWER_MODE", "now").strip().lower()
    if POWER_MEASURE_MODE not in ("now", "wait"):
      raise ValueError(
          f"IMCFLOW_POWER_MODE must be now or wait, got {POWER_MEASURE_MODE!r}")
    POWER_DMM_NAME = os.getenv("IMCFLOW_POWER_DMM_NAME", "DMM_GPIB3").strip()
    if not POWER_DMM_NAME:
      raise ValueError("IMCFLOW_POWER_DMM_NAME must not be empty")
    POWER_DMM_NPLC = _env_float("IMCFLOW_POWER_NPLC", 0.001, 0.0)
    POWER_DMM_INTERVAL_S = _env_float("IMCFLOW_POWER_INTERVAL_S", -1.0)
    POWER_DMM_SAMPLE_COUNT = _env_int("IMCFLOW_POWER_SAMPLE_COUNT", 50000, 1)
    POWER_DMM_CURRENT_RANGE = _env_float("IMCFLOW_POWER_CURRENT_RANGE", 0.1)
    POWER_DMM_RESET = _env_flag("IMCFLOW_POWER_RESET", True)
    POWER_DMM_START_TIMEOUT_S = _env_int(
        "IMCFLOW_POWER_START_TIMEOUT_S", 30, 0)
    POWER_DMM_RESULT_TIMEOUT_S = _env_int(
        "IMCFLOW_POWER_RESULT_TIMEOUT_S", 300, 0)
    POWER_SERVER_OUTPUT_PREFIX = os.getenv(
        "IMCFLOW_POWER_SERVER_OUTPUT_PREFIX", "/tmp/imcflow_power").strip()
    if not POWER_SERVER_OUTPUT_PREFIX:
      raise ValueError("IMCFLOW_POWER_SERVER_OUTPUT_PREFIX must not be empty")
else:
  raise ValueError(f"Unsupported IMCFLOW_HOST_OS: {os.getenv('IMCFLOW_HOST_OS')}")

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

  def generateRetryCheck(self, location_label, power_session_active=False):
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
    if power_session_active:
      code += "dmm_close();\n"
    code += self.generateDevicePointerCleanup()
    code += f"_retry_count++;\n"
    code += f"continue;\n"
    code.prevIndent()
    code += f"}}\n"
    code += f"#else\n"
    code += f"if (_wait_rc != 0) {{\n"
    code += f'  fprintf(stderr, "[TIMEOUT] {location_label} failed (retry disabled)\\n");\n'
    if power_session_active:
      code += "  dmm_close();\n"
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
    code = ("""
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
""" + stage_hb_macro + """
// Global failure flag: set by kernel on timeout, checked by host loop
extern volatile int g_imcflow_kernel_failed;
""")
    if self.os == "linux" and POWER_MEASURE_ENABLED:
      code += '#include "dmm_measure.h"\n'
    if REGION_TIMING_ENABLED:
      code += "#include <time.h>\n"
    return code

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

  def _power_server_ofname(self, scope, tile_idx=None):
    suffix = self.func_name
    if scope == "MODEL":
      suffix = "model"
    elif tile_idx is not None:
      suffix = f"{suffix}_tile{tile_idx}"
    return f"{POWER_SERVER_OUTPUT_PREFIX}_{suffix}.txt"

  def generatePowerMeasureStart(self, scope, tile_idx=None):
    """Start one legacy START/STARTED DMM session."""
    assert self.os == "linux" and POWER_MEASURE_ENABLED
    server_ofname = json.dumps(self._power_server_ofname(scope, tile_idx))
    dmm_name = json.dumps(POWER_DMM_NAME)
    start_func = (
        "dmm_start_current_now" if POWER_MEASURE_MODE == "now"
        else "dmm_start_current")
    label = scope.lower()
    if tile_idx is not None:
      label += f" tile {tile_idx}"

    code = CodeWriter()
    code += f"// Legacy power measurement begin ({label})\n"
    code += "{\n"
    code.nextIndent()
    code += "dmm_config_t _power_dmm_cfg = {\n"
    code.nextIndent()
    code += f".name = {dmm_name},\n"
    code += f".nplc = {POWER_DMM_NPLC!r},\n"
    code += f".interval_s = {POWER_DMM_INTERVAL_S!r},\n"
    code += f".sample_count = {POWER_DMM_SAMPLE_COUNT},\n"
    code += f".curr_range = {POWER_DMM_CURRENT_RANGE!r},\n"
    code += f".reset = {1 if POWER_DMM_RESET else 0},\n"
    code += ".ofname = NULL,\n"
    code += f".server_ofname = {server_ofname},\n"
    code.prevIndent()
    code += "};\n"
    code += (
        f"dmm_set_timeouts({POWER_DMM_START_TIMEOUT_S}, "
        f"{POWER_DMM_RESULT_TIMEOUT_S});\n")
    code += f"if ({start_func}(1, &_power_dmm_cfg) != 0) {{\n"
    code.nextIndent()
    code += (
        f'fprintf(stderr, "[POWER] {label} begin failed: %s\\n", '
        "dmm_last_error());\n")
    code += "dmm_close();\n"
    code += self.generateDevicePointerCleanup()
    code += "g_imcflow_kernel_failed = 1;\n"
    code += "return;\n"
    code.prevIndent()
    code += "}\n"
    code.prevIndent()
    code += "}\n"
    return code

  def generatePowerMeasureEnd(self, scope, tile_idx=None):
    """Send legacy GO, receive RESULT, and close the DMM session."""
    assert self.os == "linux" and POWER_MEASURE_ENABLED
    result_func = (
        "dmm_get_result_now" if POWER_MEASURE_MODE == "now"
        else "dmm_wait_result")
    label = scope.lower()
    if tile_idx is not None:
      label += f" tile {tile_idx}"

    code = CodeWriter()
    code += f"// Legacy power measurement end ({label})\n"
    code += "{\n"
    code.nextIndent()
    code += "char _power_dmm_name[64];\n"
    code += "double _power_dmm_avg = 0.0;\n"
    code += "int _power_dmm_count = 0;\n"
    code += (
        f"int _power_dmm_rc = {result_func}(_power_dmm_name, "
        "sizeof(_power_dmm_name), &_power_dmm_avg, &_power_dmm_count);\n")
    code += "if (_power_dmm_rc != 0) {\n"
    code.nextIndent()
    code += (
        f'fprintf(stderr, "[POWER] {label} result failed: %s\\n", '
        "dmm_last_error());\n")
    code += "dmm_close();\n"
    code += self.generateDevicePointerCleanup()
    code += "g_imcflow_kernel_failed = 1;\n"
    code += "return;\n"
    code.prevIndent()
    code += "}\n"
    code += (
        f'fprintf(stderr, "[POWER] {label}: %s avg=%.9g A samples=%d\\n", '
        "_power_dmm_name, _power_dmm_avg, _power_dmm_count);\n")
    code += "dmm_close();\n"
    code.prevIndent()
    code += "}\n"
    return code

  def generatePowerRegionTag(self, region_number, boundary):
    """Record a region boundary inside one MODEL-scoped DMM trace.

    Even IDs identify region starts and the following odd IDs identify ends:
    region N start = 2*N, region N end = 2*N+1.
    """
    assert boundary in ("start", "end")
    tag_id = 2 * region_number + (1 if boundary == "end" else 0)
    return (
        f"// MODEL trace: region {region_number} {boundary} "
        f"(tag {tag_id})\n"
        f"(void)set_tag({tag_id});\n")

  def generateRegionTimingStart(self, region_number):
    """Start opt-in elapsed timing at the MODEL tag-start boundary."""
    return (
        f"// Region {region_number} timing start (MODEL tag boundary)\n"
        "struct timespec _imcflow_region_time_start;\n"
        "clock_gettime(CLOCK_MONOTONIC, &_imcflow_region_time_start);\n")

  def generateRegionTimingEnd(self, region_number):
    """Report elapsed timing at the MODEL tag-end boundary."""
    return (
        f"// Region {region_number} timing end (MODEL tag boundary)\n"
        "struct timespec _imcflow_region_time_end;\n"
        "clock_gettime(CLOCK_MONOTONIC, &_imcflow_region_time_end);\n"
        "unsigned long long _imcflow_region_elapsed_ns =\n"
        "    ((unsigned long long)_imcflow_region_time_end.tv_sec * 1000000000ull +\n"
        "     (unsigned long long)_imcflow_region_time_end.tv_nsec) -\n"
        "    ((unsigned long long)_imcflow_region_time_start.tv_sec * 1000000000ull +\n"
        "     (unsigned long long)_imcflow_region_time_start.tv_nsec);\n"
        f'fprintf(stderr, "[REGION_TIMING] region={region_number} '
        f'function={self.func_name} elapsed_ns=%llu elapsed_ms=%.6f\\n",\n'
        "        _imcflow_region_elapsed_ns,\n"
        "        (double)_imcflow_region_elapsed_ns / 1000000.0);\n")

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

    power_measure_active = self.os == "linux" and POWER_MEASURE_ENABLED
    power_func_names = sorted(
        DevConfig().ImcflowFuncMap.keys(), key=_power_function_order_key)
    first_power_func = bool(power_func_names) and self.func_name == power_func_names[0]
    last_power_func = bool(power_func_names) and self.func_name == power_func_names[-1]
    region_match = re.search(r"_region(\d+)_", self.func_name)
    region_number = (
        int(region_match.group(1)) if region_match
        else (power_func_names.index(self.func_name) + 1
              if self.func_name in power_func_names else 1))

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
    if power_measure_active:
      if POWER_MEASURE_SCOPE == "MODEL" and first_power_func:
        code += self.generatePowerMeasureStart("MODEL")
      elif POWER_MEASURE_SCOPE == "REGION":
        code += self.generatePowerMeasureStart("REGION")
    if REGION_TIMING_ENABLED:
      code += self.generateRegionTimingStart(region_number)
    if power_measure_active:
      if POWER_MEASURE_SCOPE == "MODEL":
        code += self.generatePowerRegionTag(region_number, "start")
    code += _hb("before compiled_blocks transfer")
    code += self.generateToNpuTransferCode(self.compiled_blocks) # inode instrunction + policy
    code += _hb("after compiled_blocks transfer / before const_blocks transfer")
    code += self.generateToNpuTransferCode(self.const_blocks) # constant
    code += _hb("after const_blocks transfer / before policy_update")
    code += self.generatePolicyUpdateCode() # start from pc 0, up to halt
    code += _hb("after policy_update")
    code += self.generateRetryCheck(
        "policy_update",
        power_session_active=(
            power_measure_active and POWER_MEASURE_SCOPE == "REGION"))
    code += _hb("before invoke")
    code += self.generateInvokeCode() # proceed up to halt
    code += _hb("after invoke / before poll")
    code += self.generateRetryCheck(
        "invoke",
        power_session_active=(
            power_measure_active and POWER_MEASURE_SCOPE == "REGION"))

    # kernel tiling factor
    tile_factor = self.target_func_info.tiling_factor
    code += f"// Tiled execution with factor {tile_factor}\n"
    code += f"fprintf(stderr,\"Starting tiled execution with factor {tile_factor}\\n\");\n"
    for t_idx in range(tile_factor):
      code += f"fprintf(stderr,\"-- Tiled execution: TILE {t_idx} / {tile_factor} --\\n\");\n"
      code += self.generateToNpuTransferCode(self.compiled_per_tile_blocks, t_idx) # per-tile: cnt_base_addr
      code += self.generateToNpuTransferCode(self.input_blocks, t_idx) # input
      if power_measure_active and POWER_MEASURE_SCOPE == "TILE":
        code += self.generatePowerMeasureStart("TILE", t_idx)
      code += self.generateInvokeCode() # end of exec
      code += self.generateRetryCheck(
          f"tile_{t_idx}_invoke",
          power_session_active=(
              power_measure_active and POWER_MEASURE_SCOPE == "TILE"))
      if power_measure_active and POWER_MEASURE_SCOPE == "TILE":
        code += self.generatePowerMeasureEnd("TILE", t_idx)
      code += self.generateFromNpuTransferCode(self.output_blocks, t_idx) # output

    if power_measure_active:
      if POWER_MEASURE_SCOPE == "MODEL":
        code += self.generatePowerRegionTag(region_number, "end")
    if REGION_TIMING_ENABLED:
      code += self.generateRegionTimingEnd(region_number)
    if power_measure_active:
      if POWER_MEASURE_SCOPE == "REGION":
        code += self.generatePowerMeasureEnd("REGION")
      elif POWER_MEASURE_SCOPE == "MODEL" and last_power_func:
        code += self.generatePowerMeasureEnd("MODEL")

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
