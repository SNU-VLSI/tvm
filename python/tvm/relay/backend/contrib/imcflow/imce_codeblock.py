from abc import *
from typing import *
from copy import copy
import math
from pprint import pprint
from tvm.relay.op.contrib.imcflow import CustomIDToNode
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.contrib.imcflow import NodeID, TensorID, TensorEdge, TensorEdgeInfo
from tvm.relay.op.op_attrs import Conv2DAttrs
from tvm.relay.backend.contrib.imcflow.conv_util import ConvUtil
from tvm.relay.backend.contrib.imcflow.codeblock import *
from tvm.relay.backend.contrib.imcflow.layout import apply_layout_to_type
from tvm.relay.backend.contrib.imcflow.transform_utils import getInnerNodeID, getOuterNodeID, get_type, getNodeID
from tvm.relay.dataflow_pattern import *
from textwrap import indent
import logging
import pdb
from dataclasses import dataclass

if TYPE_CHECKING:
  from .builder_context import BuilderContext

ConstPat = is_constant()

# for debugging
send_num_map = {}
recv_num_map = {}
@dataclass
class RecvSendNum:
  dir       : str  = ""  # "recv" or "send"
  total_num : int  = 1

  def set_iter(self, iter_num):
    self.total_num = self.total_num * iter_num
  
  def __int__(self):
    try:
      self.total_num = int(self.total_num)
    except:
      self.total_num = self.total_num.value
    return self.total_num
  
  def __eq__(self, other):
    if not isinstance(other, RecvSendNum):
      return False
    return (self.dir == other.dir and
            self.total_num == other.total_num)

def add_to_map(edge, count, is_send=True):
  outer_loop_count = 1 if len(SimpleFor.count_stack) == 0 else int(math.prod(SimpleFor.count_stack))
  target_map = send_num_map if is_send else recv_num_map
  old_count = count.total_num
  count.set_iter(outer_loop_count)
  if edge in target_map:
    target_map[edge].total_num += count.total_num
  else:
    target_map[edge] = count
  print(f"[recv send map] {'send_map' if is_send else 'recv_map'} | {edge} | {target_map[edge].total_num} | {old_count}*{outer_loop_count}={count.total_num}")


class ImceCodeBlock(CodeBlock):
  def __init__(self, annotation: str = ""):
    super().__init__(annotation)

  @abstractmethod
  def _render(self) -> str:
    pass


class ImceCallCodeBlock(ImceCodeBlock):
  num_in_edges = None

  def __init__(self, call: 'BuilderContext', annotation: str = ""):
    super().__init__(annotation)
    self.call = call
    self.in_edges = call.get_input_edges()
    self.out_edges = call.get_output_edges()
    self.prev_op = None
    if self.num_in_edges is not None:
      assert len(self.in_edges) == self.num_in_edges
  
  def get_graph_node_id(self) -> NodeID:
    return getNodeID(self.call.call)

  def _make_unique_input_var_for_post_op(self, edge, i=None):
    if self.prev_op and (edge in self.prev_op.out_edges):
      edge = self.prev_op
    if i is not None:
      return UniqueVar((edge, i))
    else:
      return UniqueVar(edge)

  @property
  def num_blocks(self) -> int:
    return 4 if self.prev_op else 1

  @property
  def num_out_blocks(self) -> int:
    return 4 if self.prev_op else 1

  def __repr__(self):
    return f"{self.__class__.__name__}(gid: {self.call.get_gid()})"


class LoadLBBlock(ImceCodeBlock):
  """ Code block for receiving data from given fifo id to the line buffer """

  def __init__(self, count: int, repeat: int, edge: TensorEdge, edge_info: TensorEdgeInfo, annotation: str = ""):
    super().__init__(annotation)
    self.count = count
    self.repeat = repeat
    self.edge = edge
    self.edge_info = edge_info

    body = SequentialBlock()
    load_fifo_id = self.edge_info.fifo_id
    annotation = f"{self.edge}, {self.edge_info.node_info_str}"
    for _ in range(self.repeat):
      body.add(TextBlock(f"__builtin_IMCE_LOAD_LB({load_fifo_id}); // {annotation}"))
    
    self.body = SimpleFor(self.count, body, "load_block")

  def _render(self) -> str:
    add_to_map(self.edge, RecvSendNum("recv", self.count * self.repeat), is_send=False)
    return self.body.render()


class RecvConstBlock(ImceCodeBlock):
  """ Code block for receiving constant from given fifo id into a variable """
  # FIXME: Add support for initializing QREGs to zero
  num_in_edges = 1

  class ConstType(Enum):
    MIN = "MIN"
    MAX = "MAX"
    CONFIG = "CONFIG"
    SCAN = "SCAN"
    NORMAL = "NORMAL"

  def __init__(self, in_edge: TensorEdge, type : ConstType, annotation: str = ""):
    super().__init__(annotation)
    self.in_edge = in_edge
    self.type = type
    self.recv_map = {}

    te_info = DevConfig().get_tensor_edge_info_with_id_dir(
        self.in_edge.dst_id, "in")  # a hack to get the tensor edge info
    assert te_info, "Tensor edge info not found"

    assert len(te_info) == 1, "Multiple tensor edge infos found for the given edge"
    te_info = te_info[0]

    size = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(self.in_edge).size
    base_addr = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(
        self.in_edge).base_address
    assert base_addr % 32 == 0, "Base address must be a multiple of 32"

    self.te_info = te_info
    self.recv_count = math.ceil(size / 32.0)  # recv operates on 32-byte word

  def _render(self) -> str:
    owner_edge = self.te_info.owner
    add_to_map(owner_edge, RecvSendNum("recv",  self.recv_count), is_send=False)
    code = TextBlock("")
    for i in range(self.recv_count):
      var = UniqueVar((self.in_edge, i))
      var.set_static()
      if self.type == RecvConstBlock.ConstType.MIN:
        code += f"__builtin_IMCE_RECV_MIN({self.te_info.fifo_id});"
      elif self.type == RecvConstBlock.ConstType.MAX:
        code += f"__builtin_IMCE_RECV_MAX({self.te_info.fifo_id});"
      elif self.type == RecvConstBlock.ConstType.CONFIG:
        code += f"__builtin_IMCE_RECV_CFG({self.te_info.fifo_id});"
      elif self.type == RecvConstBlock.ConstType.SCAN:
        code += f"{var}_{i} = __builtin_IMCE_RECV({self.te_info.fifo_id});"
        code += f"{var}_{i} = __builtin_IMCE_SCAN_RW({var}_{i});"
      elif self.type == RecvConstBlock.ConstType.NORMAL:
        code += f"{var} = __builtin_IMCE_RECV({self.te_info.fifo_id});"
      else:
        raise ValueError(f"Unknown ConstType: {self.type}")
    return code.render()


class VecBlock(ImceCallCodeBlock):
  """
  VecBlock is base class for implementing R,I type vector operations.
  Only generates computation. RECV/SEND handled by wrapper or ConvBlock.
  """
  num_in_edges = 2

  def __init__(self, call: 'BuilderContext', annotation: str = ""):
    """ Code block for vector operations """
    super().__init__(call, annotation)
    self.op_name = self._op_name()
    self.imm_value = self._get_imm_value()

  @abstractmethod
  def _get_imm_value(self) -> int:
    pass

  @abstractmethod
  def _op_name(self) -> str:
    pass

  def _render(self) -> str:
    """Generate only computation, no RECV/SEND."""
    code = TextBlock("")

    for i in range(self.num_blocks):
      # put a tuple of (tensor edge, block index) as the key, giving a unique variable name
      var_ins = [self._make_unique_input_var_for_post_op(
          edge, i) for edge in self.in_edges]
      var_o = UniqueVar((self, i))
      var_in_str = ", ".join([f"{var_i}" for var_i in var_ins])
      # e.g. __builtin_IMCE_ADD(a, b, 15);
      code += f"{var_o} = __builtin_IMCE_{self.op_name}({var_in_str}, {self.imm_value});"

    return code.render()


class AddBlock(VecBlock):
  def _get_imm_value(self) -> int:
    return 15  # src_mask

  def _op_name(self) -> str:
    return "ADD"


class DivBlock(VecBlock):
  def _get_imm_value(self) -> int:
    return 15  # src_mask

  def _op_name(self) -> str:
    return "DIV"


class MultlBlock(VecBlock):
  def _get_imm_value(self) -> int:
    return 15  # src_mask

  def _op_name(self) -> str:
    return "MULTL"


class MulthBlock(VecBlock):
  def _get_imm_value(self) -> int:
    return 15  # src_mask

  def _op_name(self) -> str:
    return "MULTH"


class ReLUBlock(VecBlock):
  num_in_edges = 1

  def _get_imm_value(self) -> int:
    return 0  # immediate value for MAXI

  def _op_name(self) -> str:
    return "MAXI"


class MinmaxQuantBlock(ImceCallCodeBlock):
  """
  MinmaxQuantBlock for min/max quantization operations.
  Only generates computation. RECV/SEND handled by wrapper or ConvBlock.
  """
  num_in_edges = 3

  def __init__(self, call: 'BuilderContext', o_split_idx: int, annotation: str = ""):
    """ Code block for min/max quantization """
    super().__init__(call, annotation)
    self._num_blocks = 4
    self.o_split_idx = o_split_idx

  @property
  def num_blocks(self) -> int:
    return self._num_blocks

  @property
  def num_out_blocks(self) -> int:
    return 4  # FIXED in MinmaxQuantBlock

  def _render(self) -> str:
    """Generate only computation, no RECV/SEND."""
    src_mask = 15
    data_edge = next(
        edge for edge in self.in_edges if edge.dst_id.tensor_type == "data")

    code = TextBlock("")

    # arg = CustomIDToNode()[getInnerNodeID(data_edge.src_id.graph_node_id)]
    # arg_shape = get_type(call.module, arg).shape
    # arg_layout = DevConfig().LayoutMap[arg]

    for i in range(self.num_blocks):
      var_i = self._make_unique_input_var_for_post_op(data_edge, i)
      qreg_start_idx = i + 4 * self.o_split_idx
      # min max quantization does not require 
      code += f"__builtin_IMCE_MM_QUANT({var_i}, 0, {src_mask}, {qreg_start_idx});"

    # NOTE: currently, it is not possible to have consequtive 4*(MM_QUANT -> QREG)s.
    # Instead of the below code,
    #  __builtin_IMCE_MM_QUANT(var267, 0, 15, 0);
    #  var274 = __builtin_IMCE_GET_QREG(0);
    #  __builtin_IMCE_MM_QUANT(var269, 0, 15, 1);
    #  var275 = __builtin_IMCE_GET_QREG(1);
    #  __builtin_IMCE_MM_QUANT(var271, 0, 15, 2);
    #  var276 = __builtin_IMCE_GET_QREG(2);
    #  __builtin_IMCE_MM_QUANT(var273, 0, 15, 3);
    #  var277 = __builtin_IMCE_GET_QREG(3);
    # We put MM_QUANTs block first, then GET_QREGS.
    # Otherwise, it results in llvm artifact of moving qregs into vector registers.
    # e.g. vaddi %v3 %qreg2 0
    for i in range(self.num_out_blocks):
      # Get QREG result for this block
      var_o = UniqueVar((self, i))
      code += f"{var_o} = __builtin_IMCE_GET_QREG({i});"

    return code.render()


class ConcatBlock(ImceCallCodeBlock):
  min_in_edges = 2

  """
  Code block for concatenating multiple tensors
  FIXME: needs to look upon, since concat can happen not only in bitplanes...
         concat is occured in channel axis. it means store data in register file.
         We don't need to OR at this situation.
  """

  def __init__(self, call: 'BuilderContext', annotation: str = ""):
    """ Code block for min/max quantization """
    super().__init__(call, annotation)
    assert len(
        self.in_edges) >= self.min_in_edges, "At least two input edges are required"

  def _render(self) -> str:
    num_bitplanes = 4
    src_mask = 15

    code = TextBlock("")

    external_in_edges = [
        e for e in self.in_edges if e in DevConfig().TensorEdgetoInfo]
    internal_in_edge = (set(self.in_edges) - set(external_in_edges)).pop()

    for i in range(num_bitplanes):
      var_i = UniqueVar((internal_in_edge, i))
      var_o = UniqueVar((self, i))
      for ext_edge in external_in_edges:
        var_e = UniqueVar((ext_edge, i))
        fifo_id = DevConfig().get_tensor_edge_info(ext_edge).fifo_id

        code += f"{var_e} = __builtin_IMCE_RECV({fifo_id});"
        code += f"{var_o} = __builtin_IMCE_OR({var_i}, {var_e}, {src_mask});"

    return code.render()


class SplitBlock(ImceCallCodeBlock):
  num_in_edges = 1

  """ Code block for splitting a tensor into multiple tensors """

  def __init__(self, call: 'BuilderContext', annotation: str = ""):
    super().__init__(call, annotation)
    first_policies = [DevConfig().get_tensor_edge_info(
        out_edge).policy_info[0] for out_edge in self.out_edges]
    fifo_ids = [DevConfig().get_tensor_edge_info(
        out_edge).fifo_id for out_edge in self.out_edges]
    assert all(policy == first_policies[0]
               for policy in first_policies), "All output edges must have the same first policy info"
    assert all(fid == fifo_ids[0]
               for fid in fifo_ids), "All output edges must have the same fifo id"

  def _render(self) -> str:
    return ""


class ConvBlock(ImceCallCodeBlock):
  """ Code block for receiving conv input data from given fifo id """
  num_in_edges = 3

  def __init__(self, call: 'BuilderContext', shapes: dict, conv_attrs: Conv2DAttrs,
               post_ops: List[ImceCallCodeBlock] = None, annotation: str = ""):
    super().__init__(call, annotation)
    self.conv = ConvUtil(shapes["data"][2], shapes["data"][3],
                         conv_attrs.padding[0], conv_attrs.strides[0],
                         conv_attrs.kernel_size[0], conv_attrs.kernel_size[1])
    self.post_ops = post_ops if post_ops is not None else []
    self.total_in_read_counts = self.conv.get_total_input_read_counts()
    self.origin_hw = shapes["data"][2] * shapes["data"][3]  # H * W
    self.remain = self.origin_hw - self.total_in_read_counts
    
    # Link post-ops
    prev = self
    for op in self.post_ops:
      op.prev_op = prev
      prev = op
      
    self.body = self._build_structure()

  @property
  def num_blocks(self) -> int:
    return 4  # FIXED in ConvBlock

  @property
  def num_out_blocks(self) -> int:
    return 4  # FIXED in ConvBlock

  def _build_loop_body(self, recv_count: int) -> CodeBlock:

    load_info = []
    for edge in self.in_edges:
      te_infos = DevConfig().get_tensor_edge_info_with_id_dir(edge.dst_id, "in")
      assert len(te_infos) == 1, "more than one te_info found!"
      te_info = te_infos[0]
      try:
        arg_id = edge.src_id.graph_node_id
        if ConstPat.match(CustomIDToNode()[arg_id]):
          continue
      except KeyError:
        # If the node is not found in CustomIDToNode, treat it as non-constant
        pass
      load_info.append({"edge": edge, "te_info": te_info})

    assert len(load_info) == 1, "there should be exactly one load edge"
    load_edge = load_info[0]["edge"]
    load_edge_info = load_info[0]["te_info"]
    self.load_edge_info = load_edge_info

    comp = SequentialBlock()
    comp.add(LoadLBBlock(recv_count, self.num_blocks, load_edge, load_edge_info))
    comp.add(TextBlock("__builtin_IMCE_STEP();\n"))

    creg_code = TextBlock("")
    for i in range(self.num_blocks):
      var_o = UniqueVar((self, i))
      creg_code += f"{var_o} = __builtin_IMCE_GET_CREG((short){i});"
    comp.add(creg_code)

    for op in self.post_ops:
      comp.add(op)

    if self.post_ops:
      all_in_edges = copy(self.in_edges)
      all_out_edges = copy(self.out_edges)
      for op in self.post_ops:
        all_in_edges += op.in_edges
        all_out_edges += op.out_edges
      recv_edges = list(set(all_in_edges) - set(all_out_edges) - set(self.out_edges) - set([load_edge]))
      send_edges = list(set(all_out_edges) - set(all_in_edges))
      last_out_edges = self.post_ops[-1].out_edges
      assert (set(send_edges) == set(last_out_edges)), "currently doesn't support middle op SEND"
      send_block = self.post_ops[-1]

      print(f"[ConvBlock] with post ops : recv_edges: {recv_edges}, send_edges: {send_edges}, send_block: {type(send_block).__name__}")
    else:
      recv_edges = self.in_edges
      send_edges = self.out_edges
      send_block = self

    return RecvSendWrapper(comp, self.num_blocks, self.num_out_blocks, send_block, recv_edges, send_edges)

  def _build_structure(self) -> CodeBlock:
    """
    row pattern example:
    [
      {'count': 1, 'pattern': [
        {'count': 1, 'pattern': 6}, {'count': 2, 'pattern': 1}, {'count': 1, 'pattern': 0}]
      }, 
      {'count': 2, 'pattern': [
        {'count': 1, 'pattern': 2}, {'count': 2, 'pattern': 1}, {'count': 1, 'pattern': 0}
      ]}, 
      {'count': 1, 'pattern': [
        {'count': 4, 'pattern': 0}
      ]}
    ]
    """
    row_pattern = self.conv.extract_2d_pattern()
    pprint(f"[ConvBlock] row pattern for node {getNodeID(self.call.call)}:")
    pprint(row_pattern)
    root = SequentialBlock()
    for idx, row_pat in enumerate(row_pattern):
      # row_pat["count"] : number of rows that share the same pattern
      # row_pat["pattern"] : pattern for a row. list of {count, pattern}. pattern is the read count for a output pixel
      
      outer_body = SequentialBlock()
      tag = self.annotation + f"_row_group{idx}"
      
      for inner_idx, pat in enumerate(row_pat["pattern"]):
         inner_loop = SimpleFor(pat["count"], 
                                self._build_loop_body(pat["pattern"]), 
                                f"{tag}_col_group{inner_idx}")
         outer_body.add(inner_loop)

      outer_loop = SimpleFor(row_pat["count"], outer_body, f"{tag}_outer_loop(iterate row offset)")
      root.add(outer_loop)
    
    # read remaining pixels if any
    if self.remain > 0:
      print(f"[ConvBlock] node {getNodeID(self.call.call)} has remaining pixels to read: {self.remain}")
      tail_body = TextBlock(f"__builtin_IMCE_RECV({self.load_edge_info.fifo_id});")
      tail_loop = SimpleFor(self.remain, tail_body, f"{self.annotation}_tail_loop")
      root.add(tail_loop)

    return root

  def _render(self) -> str:
    return self.body.render()


class BatchNormBlock(ImceCallCodeBlock):
  """
  BatchNormBlock for batch normalization operations.
  Only generates computation. RECV/SEND handled by wrapper or ConvBlock.
  """
  num_in_edges = 3

  def __init__(self, call: 'BuilderContext', annotation: str = ""):
    """ Code block for batch normalization """
    super().__init__(call, annotation)

  def _render(self) -> str:
    """Generate only computation, no RECV/SEND."""
    code = TextBlock("")

    # Identify edges by tensor type
    for edge in self.in_edges:
      if edge.dst_id.tensor_type == "fused_scale":
        scale_edge = edge
      elif edge.dst_id.tensor_type == "fused_bias":
        bias_edge = edge
      elif edge.dst_id.tensor_type == "data":
        data_edge = edge

    print("[BatchNormBlock] num blocks:", self.num_blocks)
    for i in range(self.num_blocks):
      var_data = self._make_unique_input_var_for_post_op(data_edge, i)
      var_scale = UniqueVar((scale_edge, i))
      var_bias = UniqueVar((bias_edge, i))
      var_o = UniqueVar((self, i))

      # e.g. __builtin_IMCE_MULTL(data, scale, 15);
      code += f"{var_o} = __builtin_IMCE_MULTL({var_data}, {var_scale}, 15);"
      # e.g. __builtin_IMCE_ADD(out, bias, 15);
      code += f"{var_o} = __builtin_IMCE_ADD({var_o}, {var_bias}, 15);"

    return code.render()


class RecvSendWrapper(ImceCodeBlock):
  """
  Wrapper that adds RECV and SEND operations around a computation block.
  """

  def __init__(self, body: CodeBlock, num_blocks: int, num_out_blocks: int, send_block: ImceCodeBlock,
               in_edges: List[TensorEdge], out_edges: List[TensorEdge], annotation: str = ""):
    """Wrap a computation block with RECV/SEND operations.

    Args:
        body: The inner CodeBlock (usually a SequentialBlock or ImceCallCodeBlock)
        in_edges:
        out_edges:
        annotation: Optional annotation string
    """
    super().__init__(annotation)
    self.body = body
    self.num_blocks = num_blocks
    self.num_out_blocks = num_out_blocks
    self.in_edges = in_edges
    self.out_edges = out_edges
    self.send_block = send_block
    self.send_map = {}
    self.recv_map = {}
  
  @classmethod
  def from_codeblock(cls, codeblock: ImceCallCodeBlock, annotation: str=""):
    # Instead of calling content(), we wrap the codeblock itself
    body = codeblock 
    send_block = codeblock
    in_edges = codeblock.in_edges
    out_edges = codeblock.out_edges
    num_blocks = codeblock.num_blocks
    num_out_blocks = codeblock.num_out_blocks

    return cls(body, num_blocks, num_out_blocks, send_block, in_edges, out_edges, annotation)

  def _render(self) -> str:
    """Generate RECV -> body -> SEND."""
    code = TextBlock("")

    # --- 1. Generate RECVs ---
    if self.in_edges:
      for i in range(self.num_blocks):
        for edge in self.in_edges:
          te_infos = DevConfig().get_tensor_edge_info_with_id_dir(edge.dst_id, "in")
          assert len(te_infos) == 1, "more than one te_info found!"
          te_info = te_infos[0]

          try:
            arg_id = edge.src_id.graph_node_id
            if ConstPat.match(CustomIDToNode()[arg_id]):
              continue
          except KeyError:
            pass
          var_i = UniqueVar((edge, i))
          if not te_info or var_i.static:
            continue
          if te_info.fifo_id == 0:
            continue
          annotation = f"{edge}, {te_info.node_info_str}"
          code += f"{var_i} = __builtin_IMCE_RECV({te_info.fifo_id}); // {annotation}"
          owner_edge = te_info.owner
          add_to_map(owner_edge, RecvSendNum("recv", 1), is_send=False)
  
    # --- 2. Generate Body ---
    # Here we call content() on the child block(s)
    code += self.body

    # --- 3. Generate SENDs ---
    if self.out_edges:
      out_edge_src_ids = {edge.src_id for edge in self.out_edges}
      assert len(out_edge_src_ids) == 1, "out_edge_src_ids should have only one element"

      src_id = out_edge_src_ids.pop()
      te_out_infos = DevConfig().get_tensor_edge_info_with_id_dir(
          src_id, "out")

      output_edges=self.out_edges
      if not te_out_infos:
        dst_node = CustomIDToNode()[getInnerNodeID(self.out_edges[0].dst_id.graph_node_id)]
        if not dst_node.op.name == "split":
          print(f"Warning: no tensor edge info found for src_id {src_id}, dst_node op: {dst_node.op.name}")
        else:
          print(f"Info: no tensor edge info found for src_id {src_id}, dst_node is split, checking its output edges")
          split_node_graph_id = self.out_edges[0].dst_id.graph_node_id
          target_tensor_id = TensorID(split_node_graph_id, "odata")
          te_out_infos = DevConfig().get_tensor_edge_info_with_id_dir(target_tensor_id, "out")
          output_edges = [edge for edge in DevConfig().TensorEdgetoInfo.keys() if getInnerNodeID(edge.src_id.graph_node_id) == getInnerNodeID(target_tensor_id.graph_node_id)]
          print(f"Info: got {len(te_out_infos)} tensor edge infos from split dst edge")

      if te_out_infos:
        addresses = {info.policy_info[0].address for info in te_out_infos}
        if len(addresses) == 1:
          fifo_ids = {info.fifo_id for info in te_out_infos}
          assert len(fifo_ids) == 1, "When merging same-address outputs, fifo_id must be identical"
          te_out_infos = [te_out_infos[0]]
      
      for i in range(self.num_out_blocks):
        for te_out_info in te_out_infos:
          var_o = UniqueVar((self.send_block, i))
          if te_out_info:
            annotation = f"{','.join(map(str, self.out_edges))}, {te_out_info.node_info_str}"
            code += f"__builtin_IMCE_SEND({te_out_info.policy_info[0].address}, {var_o}, {te_out_info.fifo_id}, 0); // {annotation}"
            for out_edge in output_edges:
              add_to_map(out_edge, RecvSendNum("send", 1), is_send=True)

    return code.render()


  def create_loop_from_call(self, call_ctx : 'BuilderContext', to_process_in_edges=None):
    """Wrap the content in a loop based on call's type_args.

    Returns a SimpleFor object.
    """

    assert len(self.out_edges) == 1, "Only single output edge is supported in create_loop_from_call"

    call = call_ctx.call
    
    # FIXME: this is a quick fix for ruling out the constant edges using to_process_in_edges
    in_edges = to_process_in_edges or self.in_edges

    data_edge = next(
        edge for edge in in_edges if edge.dst_id.tensor_type in ["data", "rhs", "lhs"])
    out_edge = self.out_edges[0]

    datablock = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(data_edge)

    # get input edge size
    args = []
    for idx, arg in enumerate(call.args):
      if ConstPat.match(arg):
        continue
      else:
        args.append(arg)

    real_ttype = None
    ch_size = None
    for arg in args:
      layout = DevConfig().LayoutMap[arg]
      vir_ttype = get_type(call_ctx.module, arg)
      ch_size = vir_ttype.shape[1]  # assuming NCHW
      real_ttype_ = apply_layout_to_type(vir_ttype, layout)
      if real_ttype and real_ttype != real_ttype_:
        raise RuntimeError("Mismatched real tensor types among input args")
      real_ttype = real_ttype_
    
    elem_count = math.prod(list(real_ttype.shape))
    if isinstance(elem_count, tvm.tir.expr.IntImm):
      elem_count = elem_count.value
    dtype = real_ttype.dtype
    if "int8" in dtype or "uint8" in dtype:
      bytes_per_elem = 1
    elif "int16" in dtype or "uint16" in dtype:
      bytes_per_elem = 2
    elif "int32" in dtype or "uint32" in dtype or "float32" in dtype:
      bytes_per_elem = 4
    else:
      raise RuntimeError(f"Unsupported dtype {dtype} in create_loop_from_call")
    in_total_bytes = elem_count * bytes_per_elem

    # get output edge size
    layout = DevConfig().LayoutMap[call_ctx.call]
    real_ttype = apply_layout_to_type(get_type(call_ctx.module, call_ctx.call), layout)
    
    elem_count = math.prod(list(real_ttype.shape))
    if isinstance(elem_count, tvm.tir.expr.IntImm):
      elem_count = elem_count.value
    dtype = real_ttype.dtype
    if "int8" in dtype or "uint8" in dtype:
      bytes_per_elem = 1
    elif "int16" in dtype or "uint16" in dtype:
      bytes_per_elem = 2
    elif "int32" in dtype or "uint32" in dtype or "float32" in dtype:
      bytes_per_elem = 4
    else:
      raise RuntimeError(f"Unsupported dtype {dtype} in create_loop_from_call")
    out_total_bytes = elem_count * bytes_per_elem

    assert in_total_bytes % 32 == 0, "Input total bytes must be multiple of 32"
    assert out_total_bytes % 32 == 0, "Output total bytes must be multiple of 32"
    num_blocks=None
    num_out_blocks=None
    count=None
    if out_total_bytes >= in_total_bytes:
      ratio = float(out_total_bytes) / float(in_total_bytes)
      assert ratio.is_integer(), "Output to input byte size ratio must be integer"
      num_blocks=1
      num_out_blocks=int(ratio)
      count=in_total_bytes//32
    else:
      ratio = float(in_total_bytes) / float(out_total_bytes)
      assert ratio.is_integer(), "Input to output byte size ratio must be integer"
      num_blocks=int(ratio)
      num_out_blocks=1
      count=out_total_bytes//32
    
    # for min_max_quant, it has minimum granularity more then one.
    # It have to recv all input channels first.
    if isinstance(self.send_block, MinmaxQuantBlock):
      if num_out_blocks <= 4:
        ratio = 4 // num_out_blocks
        num_blocks = num_blocks * ratio
        count = count // ratio
        num_out_blocks = 4
      else:
        ratio = num_out_blocks // 4
        num_out_blocks = 4
        count = count * ratio
        num_blocks = num_blocks // ratio
      self.send_block._num_blocks = num_blocks

    # Create a new RecvSendWrapper that represents the inner logic
    inner = RecvSendWrapper(self.body, num_blocks, num_out_blocks, 
                            self.send_block, self.in_edges, self.out_edges, self.annotation)

    return SimpleFor(count, inner, f"call_created_loop")


class ImceCodeBlockManager(NodeCodeBlockManager):
  """A class that manages and generates code blocks for imces."""

  def __init__(self, func_name: str):
    super().__init__()
    self.func_name = func_name

  @property
  def nodes(self) -> List[NodeID]:
    return NodeID.imces()

  @property
  def target(self) -> str:
    return "imce"

  def start_block(self) -> str:
    code = (
        "#include \"../common_decl.h\"\n"
        f"void {self.func_name}() {{\n"
        "  int hid = __builtin_IMCE_GET_CORE_HID();\n"
        "  int wid = __builtin_IMCE_GET_CORE_WID();\n"
        f"{indent(UniqueVar.get_decls_str(), '  ')}\n"
    )
    return code

  def end_block(self) -> str:
    return "}\n"


"""
  short16 test_builtins(short16 a, short16 b) {
  short16 var1 = __builtin_IMCE_ADD(a, b, 15);
  short16 var2 = __builtin_IMCE_SUB(a, var1, 15);
  short16 var3 = __builtin_IMCE_AND(a, var2, 15);
  short16 var4 = __builtin_IMCE_OR(a, var3, 15);
  short16 var5 = __builtin_IMCE_XOR(a, var4, 15);
  short16 var6 = __builtin_IMCE_SRL(a, var5, 15);
  short16 var7 = __builtin_IMCE_SLL(a, var6, 15);
  short16 var8 = __builtin_IMCE_SRA(a, var7, 15);
  short16 var9 = __builtin_IMCE_MAX(a, var8, 15);
  short16 var10 = __builtin_IMCE_MIN(a, var9, 15);
  short16 var11 = __builtin_IMCE_MULTL(a, var10, 15);
  short16 var12 = __builtin_IMCE_MULTH(a, var11, 15);

  short16 var14 = __builtin_IMCE_ADDI(var12, 1);
  short16 var15 = __builtin_IMCE_SUBI(var14, 1);
  short16 var16 = __builtin_IMCE_ANDI(var15, 1);
  short16 var17 = __builtin_IMCE_ORI(var16, 1);
  short16 var18 = __builtin_IMCE_XORI(var17, 1);
  short16 var19 = __builtin_IMCE_SRLI(var18, 1);
  short16 var20 = __builtin_IMCE_SLLI(var19, 1);
  short16 var21 = __builtin_IMCE_SRAI(var20, 1);
  short16 var22 = __builtin_IMCE_MAXI(var21, 1);
  short16 var23 = __builtin_IMCE_MINI(var22, 1);
  short16 var24 = __builtin_IMCE_MULTLI(var23, 1);
  short16 var25 = __builtin_IMCE_MULTHI(var24, 1);

  short16 var26 = __builtin_IMCE_DWCONV(var25, 1, 0, 1, 1);
  __builtin_IMCE_SEND(1, var26, 2, 3);
  short16 var27 = __builtin_IMCE_RECV(0);
  short16 var_min = __builtin_IMCE_RECV_MIN(0);
  short16 var_max = __builtin_IMCE_RECV_MAX(0);
  short16 var_cfg = __builtin_IMCE_RECV_CFG(0);
  short16 var_scan0 = __builtin_IMCE_RECV_SREG0(0);
  short16 var_scan1 = __builtin_IMCE_RECV_SREG1(0);
  __builtin_IMCE_SETFLAG(1);
  __builtin_IMCE_STANDBY(1, 2);

  short16 var28 = __builtin_IMCE_MAXPOOL(1, 2, 3);
  short16 var29 = __builtin_IMCE_AVGPOOL(1, 2, 3);

  __builtin_IMCE_ADDQ(var27, var28, 1, 2);
  __builtin_IMCE_SUBQ(a, var29, 1, 2);
  __builtin_IMCE_MULTLQ(a, var29, 1, 2);
  __builtin_IMCE_MULTHQ(a, var29, 1, 2);
  __builtin_IMCE_NU_QUANT(a, var29, 1, 2);
  __builtin_IMCE_MM_QUANT(a, 0, 15, 2);
  short16 var30 = __builtin_IMCE_GET_QREG(0);
  short16 var31 = __builtin_IMCE_GET_QREG(1);
  short16 var32 = __builtin_IMCE_GET_QREG(2);
  short16 var33 = __builtin_IMCE_GET_QREG(3);
  short16 var_0 = __builtin_IMCE_ADD(var30, var31, 15);
  short16 var_1 = __builtin_IMCE_ADD(var32, var33, 15);

  __builtin_IMCE_STEP();
  __builtin_IMCE_NOP();
  __builtin_IMCE_STOP();
  short16 var34 = __builtin_IMCE_GET_CREG(0);
  short16 var35 = __builtin_IMCE_GET_CREG(1);
  short16 var36 = __builtin_IMCE_GET_CREG(2);
  short16 var37 = __builtin_IMCE_GET_CREG(3);

  short16 var_2 = __builtin_IMCE_ADD(var34, var35, 15);
  short16 var_3 = __builtin_IMCE_ADD(var36, var37, 15);

  short16 var38 = __builtin_IMCE_SCAN_RW(a);

  short16 var_4 = __builtin_IMCE_ADD(var_0, var_1, 15);
  short16 var_5 = __builtin_IMCE_ADD(var_2, var_3, 15);
  short16 var_6 = __builtin_IMCE_ADD(var_4, var_5, 15);

  __builtin_IMCE_LOAD_LB(0);
"""
