from abc import *
from typing import *
from copy import copy
import math
from tvm.relay.op.contrib.imcflow import CustomIDToNode
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.contrib.imcflow import NodeID, TensorID, TensorEdge, TensorEdgeInfo
from tvm.relay.op.op_attrs import Conv2DAttrs
from tvm.relay.backend.contrib.imcflow.conv_util import ConvUtil
from tvm.relay.backend.contrib.imcflow.codeblock import *
from tvm.relay.dataflow_pattern import *
from textwrap import indent
import logging
import pdb

if TYPE_CHECKING:
  from .builder_context import BuilderContext

ConstPat = is_constant()


class ImceCodeBlock(CodeBlock):
  def __init__(self, annotation: str = ""):
    super().__init__()
    self.annotation = annotation

  def content(self) -> CodeBlock:
    if self.annotation:
      code = TextBlock("")
      code += f"// generate: {self.annotation}"
      code += copy(self._content())
      code += f"// endgenerate: {self.annotation}"
      return code
    else:
      return self._content()

  @abstractmethod
  def _content(self) -> CodeBlock:
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

  def _content(self) -> CodeBlock:
    code = TextBlock("")
    load_fifo_id = self.edge_info.fifo_id
    annotation = f"{self.edge}, {self.edge_info.node_info_str}"
    for _ in range(self.repeat):
      code += f"__builtin_IMCE_LOAD_LB({load_fifo_id}); // {annotation}"
    return SimpleFor(self.count, code, "load_block")


class RecvConstBlock(ImceCodeBlock):
  """ Code block for receiving constant from given fifo id into a variable """
  # FIXME: Add support for initializing QREGs to zero
  num_in_edges = 1

  def __init__(self, in_edge: TensorEdge, annotation: str = ""):
    super().__init__(annotation)
    self.in_edge = in_edge

  def _content(self) -> CodeBlock:
    code = TextBlock("")
    te_info = DevConfig().get_tensor_edge_info_with_id_dir(
        self.in_edge.dst_id, "in")  # a hack to get the tensor edge info
    assert te_info, "Tensor edge info not found"

    assert len(te_info) == 1, "Multiple tensor edge infos found for the given edge"
    te_info = te_info[0]

    size = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(self.in_edge).size
    base_addr = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(
        self.in_edge).base_address
    assert base_addr % 32 == 0, "Base address must be a multiple of 32"
    recv_count = math.ceil(size / 32.0)  # recv operates on 32-byte word

    for i in range(recv_count):
      var = UniqueVar((self.in_edge, i))
      var.set_static()
      if self.annotation == "min write":
        # code += f"{var} = __builtin_IMCE_RECV_MIN({te_info.fifo_id});"
        code += f"__builtin_IMCE_RECV_MIN({te_info.fifo_id});"
      elif self.annotation == "max write":
        # code += f"{var} = __builtin_IMCE_RECV_MAX({te_info.fifo_id});"
        code += f"__builtin_IMCE_RECV_MAX({te_info.fifo_id});"
      elif self.annotation == "config write":
        # code += f"{var} = __builtin_IMCE_RECV_CFG({te_info.fifo_id});"
        code += f"__builtin_IMCE_RECV_CFG({te_info.fifo_id});"
      elif self.annotation == "scan write":
        # code += f"{var} = __builtin_IMCE_RECV_SREG{i}({te_info.fifo_id});"
        code += f"{var}_{i} = __builtin_IMCE_RECV({te_info.fifo_id});"
        code += f"{var}_{i} = __builtin_IMCE_SCAN_RW({var}_{i});"
      else:
        code += f"{var} = __builtin_IMCE_RECV({te_info.fifo_id});"
    return code


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

  def _content(self) -> CodeBlock:
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

    return code


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
    self.o_split_idx = o_split_idx

  @property
  def num_out_blocks(self) -> int:
    return 4  # FIXED in MinmaxQuantBlock

  def _content(self) -> CodeBlock:
    """Generate only computation, no RECV/SEND."""
    src_mask = 15
    data_edge = next(
        edge for edge in self.in_edges if edge.dst_id.tensor_type == "data")

    code = TextBlock("")

    for i in range(self.num_blocks):
      var_i = self._make_unique_input_var_for_post_op(data_edge, i)
      qreg_start_idx = i + 4 * self.o_split_idx
      # min max quantization does not require $rs2
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

    return code


class ConcatBlock(ImceCallCodeBlock):
  min_in_edges = 2

  """
  Code block for concatenating multiple tensors
  FIXME: needs to look upon, since concat can happen not only in bitplanes...
  """

  def __init__(self, call: 'BuilderContext', annotation: str = ""):
    """ Code block for min/max quantization """
    super().__init__(call, annotation)
    assert len(
        self.in_edges) >= self.min_in_edges, "At least two input edges are required"

  def _content(self) -> CodeBlock:
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

    return code


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

  def _content(self) -> CodeBlock:
    return TextBlock("")


class ConvBlock(ImceCallCodeBlock):
  """ Code block for receiving conv input data from given fifo id """
  num_in_edges = 3

  def __init__(self, call: 'BuilderContext', shapes: dict, conv_attrs: Conv2DAttrs,
               annotation: str = ""):
    super().__init__(call, annotation)
    self.conv = ConvUtil(shapes["data"][2], shapes["data"][3],
                         conv_attrs.padding[0], conv_attrs.strides[0],
                         conv_attrs.kernel_size[0], conv_attrs.kernel_size[1])
    self.post_op_chain = []

  def add_post_op(self, code: ImceCallCodeBlock):
    if self.post_op_chain:
      code.prev_op = self.post_op_chain[-1]
    else:
      code.prev_op = self
    self.post_op_chain.append(code)

  @property
  def num_blocks(self) -> int:
    return 4  # FIXED in ConvBlock

  @property
  def num_out_blocks(self) -> int:
    return 4  # FIXED in ConvBlock

  def _loop_body_content(self, recv_count: int) -> CodeBlock:

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

    code = TextBlock("")
    code += LoadLBBlock(recv_count, self.num_blocks, load_edge, load_edge_info)
    code += "__builtin_IMCE_STEP();\n"

    for i in range(self.num_blocks):
      var_o = UniqueVar((self, i))
      code += f"{var_o} = __builtin_IMCE_GET_CREG((short){i});"

    for op in self.post_op_chain:
      code += "\n"
      code += copy(op)

    code += "\n"

    if self.post_op_chain:
      all_in_edges = copy(self.in_edges)
      all_out_edges = copy(self.out_edges)
      for op in self.post_op_chain:
        all_in_edges += op.in_edges
        all_out_edges += op.out_edges
      recv_edges = list(set(all_in_edges) - set(all_out_edges) - set(self.out_edges) - set([load_edge]))
      send_edges = list(set(all_out_edges) - set(all_in_edges))
      last_out_edges = self.post_op_chain[-1].out_edges
      assert (set(send_edges) == set(last_out_edges)), "currently doesn't support middle op SEND"
      send_block = self.post_op_chain[-1]
    else:
      recv_edges = self.in_edges
      send_edges = self.out_edges
      send_block = self

    code = RecvSendWrapper(code, self.num_blocks, self.num_out_blocks, send_block, recv_edges, send_edges)

    return code

  def _inner_loop_content(self, loop_count: int, recv_count: int, tag="") -> CodeBlock:
    return SimpleFor(loop_count, self._loop_body_content(recv_count), f"{tag}_inner_loop(iterate col offset. load inputs)")

  def _outer_loop_content(self, loop_count: int, loop_pattern: dict, tag="") -> CodeBlock:
    code = TextBlock("")
    for idx, pat in enumerate(loop_pattern):
      code += self._inner_loop_content(pat["count"], pat["pattern"], f"{tag}_col_group{idx}")

    return SimpleFor(loop_count, code, f"{tag}_outer_loop(iterate row offset)")

  def _content(self) -> CodeBlock:
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
    code = TextBlock("")
    for idx, row_pat in enumerate(row_pattern):
      # row_pat["count"] : number of rows that share the same pattern
      # row_pat["pattern"] : pattern for a row. list of {count, pattern}. pattern is the read count for a output pixel
      code += self._outer_loop_content(row_pat["count"], row_pat["pattern"], self.annotation + f"_row_group{idx}")

    return code


class BatchNormBlock(ImceCallCodeBlock):
  """
  BatchNormBlock for batch normalization operations.
  Only generates computation. RECV/SEND handled by wrapper or ConvBlock.
  """
  num_in_edges = 3

  def __init__(self, call: 'BuilderContext', annotation: str = ""):
    """ Code block for batch normalization """
    super().__init__(call, annotation)

  def _content(self) -> CodeBlock:
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

    for i in range(self.num_blocks):
      var_data = self._make_unique_input_var_for_post_op(data_edge, i)
      var_scale = UniqueVar((scale_edge, i))
      var_bias = UniqueVar((bias_edge, i))
      var_o = UniqueVar((self, i))

      # e.g. __builtin_IMCE_MULTL(data, scale, 15);
      code += f"{var_o} = __builtin_IMCE_MULTL({var_data}, {var_scale}, 15);"
      # e.g. __builtin_IMCE_ADD(out, bias, 15);
      code += f"{var_o} = __builtin_IMCE_ADD({var_o}, {var_bias}, 15);"

    return code


class RecvSendWrapper(ImceCodeBlock):
  """
  Wrapper that adds RECV and SEND operations around a computation block.
  """

  def __init__(self, body: TextBlock, num_blocks: int, num_out_blocks: int, send_block: ImceCodeBlock,
               in_edges: List[TensorEdge], out_edges: List[TensorEdge], annotation: str = ""):
    """Wrap a computation block with RECV/SEND operations.

    Args:
        code:
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
    self._loop_wrapper = None  # Store the loop wrapper if create_loop_from_call is used

  @classmethod
  def from_codeblock(cls, codeblock: ImceCallCodeBlock, annotation: str=""):
    body = codeblock.content()  # Call the method to get the actual CodeBlock
    send_block = codeblock
    in_edges = codeblock.in_edges
    out_edges = codeblock.out_edges
    num_blocks = codeblock.num_blocks
    num_out_blocks = codeblock.num_out_blocks

    return cls(body, num_blocks, num_out_blocks, send_block, in_edges, out_edges, annotation)

  def _content(self) -> CodeBlock:
    """Generate RECV -> body -> SEND."""
    # If a loop wrapper was created via create_loop_from_call, return that instead
    if self._loop_wrapper is not None:
      return self._loop_wrapper

    code = TextBlock("")

    # Get tensor edge info for inputs and outputs
    if self.in_edges:
      # te_in_infos = [DevConfig().get_tensor_edge_info_with_id_dir(
      #     edge.dst_id, "in")[0] for edge in self.in_edges
      #     if DevConfig().get_tensor_edge_info_with_id_dir(edge.dst_id, "in") != []]
      # Generate RECV for non-constant input edges
      for i in range(self.num_blocks):
        # for edge, te_info in zip(self.in_edges, te_in_infos):
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
          var_i = UniqueVar((edge, i))
          if not te_info or var_i.static:
            continue
          if te_info.fifo_id == 0:
            continue
          annotation = f"{edge}, {te_info.node_info_str}"
          code += f"{var_i} = __builtin_IMCE_RECV({te_info.fifo_id}); // {annotation}"
  
    # Add the inner block's computation
    code += str(self.body)

    if self.out_edges:
      out_edge_src_ids = {edge.src_id for edge in self.out_edges}
      assert len(out_edge_src_ids) == 1, "out_edge_src_ids should have only one element"

      te_out_infos = DevConfig().get_tensor_edge_info_with_id_dir(
          out_edge_src_ids.pop(), "out")
      # If all policy addresses are identical, merge into a single SEND (dedupe)
      if te_out_infos:
        addresses = {info.policy_info[0].address for info in te_out_infos}
        if len(addresses) == 1:
          fifo_ids = {info.fifo_id for info in te_out_infos}
          assert len(fifo_ids) == 1, "When merging same-address outputs, fifo_id must be identical"
          te_out_infos = [te_out_infos[0]]
      
      # Generate SEND for all output edges (supports merged case above)
      for i in range(self.num_out_blocks):
        for te_out_info in te_out_infos:
          var_o = UniqueVar((self.send_block, i))
          if te_out_info:
            annotation = f"{te_out_info.node_info_str}"
            code += f"__builtin_IMCE_SEND({te_out_info.policy_info[0].address}, {var_o}, {te_out_info.fifo_id}, 0); // {annotation}"

    return code


  def create_loop_from_call(self, call_ctx, to_process_in_edges=None):
    """Wrap the content in a loop based on call's type_args.

    Returns self to allow method chaining.
    """
    # Wrap in a loop based on calls' type_args
    call = call_ctx.call
    code = self._content()  # Get the current content (RECV/SEND wrapped code)

    # FIXME: this is a quick fix for ruling out the constant edges using to_process_in_edges
    in_edges = to_process_in_edges or self.in_edges

    data_edge = next(
        edge for edge in in_edges if edge.dst_id.tensor_type in ["data", "rhs", "lhs"])

    # count = (edge_shape[0] * math.ceil(float(edge_shape[1].value)/16) * edge_shape[2] * edge_shape[3]) // self.num_blocks
    print(f"Warning: create_loop_from_call is used, double-check the loop count calculation!")
    datablock = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(data_edge)

    if datablock:
      count = math.ceil(datablock.size / 32.0)
    else:
      edge_shape = None
      for idx, arg in enumerate(call.args):
        if ConstPat.match(arg):
          continue
        else:
          if edge_shape is not None:
            assert (
                edge_shape == call.type_args[idx].shape), "all input args should have the same shape"
          else:
            edge_shape = call.type_args[idx].shape
      count = edge_shape[-2] * edge_shape[-1]

    # Store the loop-wrapped content in the _loop_wrapper attribute
    self._loop_wrapper = SimpleFor(count, code, f"call_created_loop")

    return self


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
