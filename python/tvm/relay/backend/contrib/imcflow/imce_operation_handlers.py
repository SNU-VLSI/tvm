"""Concrete operation handlers for IMCFlow IMCE code generation.

This module contains all the specific handlers for different relay operations
that generate IMCE code blocks.

All handlers receive a BuilderContext that wraps the relay.Call with helper methods.
"""

from tvm import relay
from tvm.relay import op
from tvm.relay.frontend.common import infer_shape
from tvm.relay.backend.contrib.imcflow.operation_handlers import (
    OperationHandler,
    register_operation_handler
)
from tvm.relay.backend.contrib.imcflow.transform import getNodeID, getNodeDebugID
from tvm.relay.backend.contrib.imcflow.imce_codeblock import *
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from .builder_context import BuilderContext


@register_operation_handler
class CompositeHandler(OperationHandler):
  """Handles composite function calls.

  Composite functions wrap sequences of operations and must be handled
  with highest priority to maintain correct visitation context.
  """

  @property
  def priority(self) -> int:
    return 0  # Highest priority

  def can_handle(self, call: relay.Call) -> bool:
    return (isinstance(call.op, relay.Function) and
            "Composite" in call.op.attrs)

  def handle(self, call: 'BuilderContext') -> None:
    # Set composite context on both BuilderContext and builder
    composite_id = getNodeID(call.call)
    call.curr_composite_id = composite_id
    self.builder.curr_composite_id = composite_id
    
    # Initialize stack and pending info
    call.post_op_stack = []
    call.conv_pending_info = {}

    # Also set on builder so recursive visits see them
    self.builder.post_op_stack = call.post_op_stack
    self.builder.conv_pending_info = call.conv_pending_info

    # Visit the body of the composite function using self.builder
    self.builder.visit(call.call.op.body)

    # Assemble ConvBlock or DWConvBlock if pending info exists
    if call.conv_pending_info:
        info = call.conv_pending_info
        conv_ctx = info['ctx']
        shapes = info['shapes']
        attrs = info['attrs']
        hid = info['hid']
        annotation = info['annotation']
        is_depthwise = info.get('is_depthwise', False)

        if is_depthwise:
            weight_edge = info.get('weight_edge')
            block = DWConvBlock(conv_ctx, shapes, attrs, weight_edge=weight_edge, post_ops=call.post_op_stack, annotation=annotation)
        else:
            block = ConvBlock(conv_ctx, shapes, attrs, post_ops=call.post_op_stack, annotation=annotation)
        call.codeblocks.append(hid, block, CodePhase.EXEC)

    # Clear composite context on both BuilderContext and builder
    call.curr_composite_id = None
    self.builder.curr_composite_id = None
    call.post_op_stack = []
    call.conv_pending_info = None
    self.builder.post_op_stack = None
    self.builder.conv_pending_info = None

    # Visit arguments
    # for a in call.call.args:
    #   self.builder.visit(a)


@register_operation_handler
class ConvHandler(OperationHandler):
  """Handles nn.imcflow_qconv operations.

  Generates ConvBlock for convolution execution and LoadLBBlock for
  weight initialization.
  """

  uid=0

  @property
  def priority(self) -> int:
    return 10

  def can_handle(self, call: relay.Call) -> bool:
    return call.op == op.get("nn.imcflow_qconv")

  def handle(self, call: 'BuilderContext') -> None:
    # Get hid and shapes
    hid = call.get_hid()
    graph_node_id = getNodeID(call.call)
    shapes = call.get_arg_shape_dict()
    shapes["output"] = infer_shape(call.call)

    # scan reg
    # TODO: add scan reg code block
    # edge = call.get_tensor_edge_from_tag("scan")
    # block = RecvConstBlock(edge, f"scan write")
    # self.codeblocks.append(hid, block, CodePhase.INIT)

    # config reg
    # TODO: add config reg code block
    edge = call.get_tensor_edge_from_tag("config")
    block = RecvConstBlock(edge, RecvConstBlock.ConstType.CONFIG, f"{edge}, config write")
    call.codeblocks.append(hid, block, CodePhase.INIT)
    # add constedge info to codeblock info
    IMCECodeBlockInfo().append_const_edge_info(edge, hid)

    # write weights => on INODE

    if call.curr_composite_id:
        # Defer ConvBlock creation if inside composite
        call.conv_pending_info.update({
            'ctx': call,
            'shapes': shapes,
            'attrs': call.call.attrs,
            'hid': hid,
            'annotation': f"conv exec{ConvHandler.uid}",
            "is_depthwise": False
        })
    else:
        # Standalone Conv
        block = ConvBlock(call, shapes, call.call.attrs, post_ops=[], annotation=f"conv exec{ConvHandler.uid}")
        call.codeblocks.append(hid, block, CodePhase.EXEC)
        
    ConvHandler.uid += 1


@register_operation_handler
class DwConvHandler(OperationHandler):
  """Handles nn.imcflow_qdwconv (depthwise convolution) operations.

  Generates DWConvBlock for depthwise convolution execution.
  Uses OP_DWCONV instruction which operates on BSHR input and register weights.

  Key differences from standard convolution:
  - Uses BSHR (bit shift register) for input data
  - Weight stored in registers with layout ic16_pad16_kh3_kw3
  - Only 3x3 kernel supported
  - Uses bshr_sel for input channel group selection
  """

  uid = 0

  @property
  def priority(self) -> int:
    return 10

  def can_handle(self, call: relay.Call) -> bool:
    return call.op == op.get("nn.imcflow_qdwconv")

  def handle(self, call: 'BuilderContext') -> None:
    # Get hid and shapes
    hid = call.get_hid()
    graph_node_id = getNodeID(call.call)
    shapes = call.get_arg_shape_dict()
    shapes["output"] = infer_shape(call.call)

    # config reg
    config_edge = call.get_tensor_edge_from_tag("config")
    block = RecvConstBlock(config_edge, RecvConstBlock.ConstType.CONFIG, f"{config_edge}, config write")
    call.codeblocks.append(hid, block, CodePhase.INIT)
    # add constedge info to codeblock info
    IMCECodeBlockInfo().append_const_edge_info(config_edge, hid)

    # DW conv weight: received via FIFO (not loaded to IMCU like standard conv)
    weight_edge = call.get_tensor_edge_from_tag("weight")
    # Register weight edge so INODE knows to SEND it
    IMCECodeBlockInfo().append_const_edge_info(weight_edge, hid)

    if call.curr_composite_id:
        # Defer DWConvBlock creation if inside composite
        call.conv_pending_info.update({
            'ctx': call,
            'shapes': shapes,
            'attrs': call.call.attrs,
            'hid': hid,
            'annotation': f"dwconv exec{DwConvHandler.uid}",
            'is_depthwise': True,
            'weight_edge': weight_edge,
        })
    else:
        # Standalone DWConv
        block = DWConvBlock(call, shapes, call.call.attrs, weight_edge=weight_edge, post_ops=[], annotation=f"dwconv exec{DwConvHandler.uid}")
        call.codeblocks.append(hid, block, CodePhase.EXEC)

    DwConvHandler.uid += 1


@register_operation_handler
class AddHandler(OperationHandler):
  """Handles add operations.

  Adds as a post-op to the current convolution block when inside a composite.
  """

  @property
  def priority(self) -> int:
    return 10

  def can_handle(self, call: relay.Call) -> bool:
    return call.op == op.get("add")

  def handle(self, call: 'BuilderContext') -> None:
    # assert call.curr_composite_id, "Add must be inside a composite function"
    hid = call.get_hid()

    to_process_in_edges = []
    for in_edge in call.get_input_edges():
      arg_gid = in_edge.src_id.graph_node_id
      arg_node = CustomIDToNode()[arg_gid]
      if ConstPat.match(arg_node):
        block = RecvConstBlock(in_edge, RecvConstBlock.ConstType.NORMAL, f"add const")
        call.codeblocks.append(hid, block, CodePhase.INIT)
        # add constedge info to codeblock info
        IMCECodeBlockInfo().append_const_edge_info(in_edge, hid)
      else:
        to_process_in_edges.append(in_edge)
      
    block = AddBlock(call, "add")
    if call.curr_composite_id is not None:
      call.post_op_stack.append(block)
    else:
      wrapped_block = RecvSendWrapper.from_codeblock(block, "add standalone").create_loop_from_call(call, to_process_in_edges)
      call.codeblocks.append(hid, wrapped_block, CodePhase.EXEC)


@register_operation_handler
class MultHandler(OperationHandler):
  """Handles multiply operations.
  FIXME: uses MULTL for now

  Adds as a post-op to the current convolution block when inside a composite.
  """

  @property
  def priority(self) -> int:
    return 10

  def can_handle(self, call: relay.Call) -> bool:
    return call.op == op.get("multiply")

  def handle(self, call: 'BuilderContext') -> None:
    hid = call.get_hid()

    to_process_in_edges = []
    for in_edge in call.get_input_edges():
      arg_gid = in_edge.src_id.graph_node_id
      if ConstPat.match(CustomIDToNode()[arg_gid]):
        block = RecvConstBlock(in_edge, RecvConstBlock.ConstType.NORMAL, f"mult const")
        call.codeblocks.append(hid, block, CodePhase.INIT)
        # add constedge info to codeblock info
        IMCECodeBlockInfo().append_const_edge_info(in_edge, hid)
      else:
        to_process_in_edges.append(in_edge)

    block = MultlBlock(call, "multl")
    if call.curr_composite_id is not None:
      call.post_op_stack.append(block)
    else:
      wrapped_block = RecvSendWrapper.from_codeblock(block, "multiply standalone", builder=call.builder).create_loop_from_call(call, to_process_in_edges)
      call.codeblocks.append(hid, wrapped_block, CodePhase.EXEC)


@register_operation_handler
class DivideHandler(OperationHandler):
  """Handles divide operations.

  Adds as a post-op to the current convolution block when inside a composite.
  """

  @property
  def priority(self) -> int:
    return 10

  def can_handle(self, call: relay.Call) -> bool:
    return call.op == op.get("divide")

  def handle(self, call: 'BuilderContext') -> None:
    # TODO: divide block should be replaced later
    # assert call.curr_composite_id, "Divide must be inside a composite function"
    hid= call.get_hid()

    to_process_in_edges = []
    for in_edge in call.get_input_edges():
      arg_gid = in_edge.src_id.graph_node_id
      if ConstPat.match(CustomIDToNode()[arg_gid]):
        block = RecvConstBlock(in_edge, RecvConstBlock.ConstType.NORMAL, f"div const")
        call.codeblocks.append(hid, block, CodePhase.INIT)
        # add constedge info to codeblock info
        IMCECodeBlockInfo().append_const_edge_info(in_edge, hid)
      else:
        to_process_in_edges.append(in_edge)
    block = DivBlock(call, "div")

    if call.curr_composite_id is not None:
      call.post_op_stack.append(block)
    else:
      wrapped_block = RecvSendWrapper.from_codeblock(block, "divide standalone").create_loop_from_call(call, to_process_in_edges)
      call.codeblocks.append(hid, wrapped_block, CodePhase.EXEC)

@register_operation_handler
class ConcatHandler(OperationHandler):
  """Handles concatenate operations.

  Adds as a post-op to the convolution block for the target hardware node.
  """

  @property
  def priority(self) -> int:
    return 10

  def can_handle(self, call: relay.Call) -> bool:
    return call.op == op.get("concatenate")

  def handle(self, call: 'BuilderContext') -> None:
    block = ConcatBlock(call, "concat")

    if call.curr_composite_id is not None:
      call.post_op_stack.append(block)
    else:
      # Standalone concat needs RECV/SEND wrapper
      wrapped_block = RecvSendWrapper.from_codeblock(block, "concat_standalone", builder=call.builder).create_loop_from_call(call)
      call.codeblocks.append(call.get_hid(), wrapped_block, CodePhase.EXEC)


@register_operation_handler
class SplitHandler(OperationHandler):
  """Handles split operations.

  Adds as a post-op to the convolution block for IMCE hardware nodes.
  """

  @property
  def priority(self) -> int:
    return 10

  def can_handle(self, call: relay.Call) -> bool:
    return call.op == op.get("split")

  def handle(self, call: 'BuilderContext') -> None:
    return
    hid = call.get_hid()
    if hid.is_imce():
      block = SplitBlock(call.get_input_edge(), call.get_output_edges(), "split")
      call.post_op_stack.append(block)


@register_operation_handler
class MinMaxQuantizeHandler(OperationHandler):
  """Handles qnn.imcflow_min_max_quantize operations.

  Generates RecvConstBlock for min/max parameters and MinmaxQuantBlock
  as a post-op to the current convolution OR as a standalone operation
  depending on the self.curr_composite_id
  """

  @property
  def priority(self) -> int:
    return 10

  def can_handle(self, call: relay.Call) -> bool:
    return call.op == op.get("qnn.imcflow_min_max_quantize")

  def consumer_is_non_multicast_split(self, call:'BuilderContext') -> tuple[bool, int, int]:
    out_edges = call.get_output_edges()
    assert len(out_edges) == 1, "Only one output edge is expected"
    out_edge = out_edges[0]
    dst_gid = out_edge.dst_id.graph_node_id
    dst_node = CustomIDToNode()[getInnerNodeID(dst_gid)]
    if isinstance(dst_node, relay.Call) and dst_node.op.name == "split":
      split_info = DevConfig().SplitInfo[call.func_name][getNodeID(dst_node)]
      is_multicast = split_info["is_multi_cast"]
      channels = split_info["channels"]
      num_splits = split_info["num_splits"]
      return (not is_multicast), channels, num_splits
    else:
      return False, None, None

  def handle(self, call: 'BuilderContext') -> None:
    print(f"[IMCE CODE BUILDER] handle MinMaxQuantize: {getNodeID(call.call)} {getNodeDebugID(call.call)}")
    hid = call.get_hid()

    # Generate RecvConst blocks for min/max parameters
    for tag in ("min", "max"):
      edge = call.get_tensor_edge_from_tag(tag)
      # TODO: inode code block needs to put appropriate address for min/max reg.
      # TODO: two ways to set min/max reg. RecvConst vs. ADDI
      const_type = RecvConstBlock.ConstType.MIN if tag == "min" else RecvConstBlock.ConstType.MAX
      block = RecvConstBlock(edge, const_type, f"{edge}, {tag} write")
      call.codeblocks.append(hid, block, CodePhase.INIT)
      # add constedge info to codeblock info
      IMCECodeBlockInfo().append_const_edge_info(edge, hid)

    # TODO: add reset qreg code block
    # _edge = TensorEdge(TensorID(-1, "zero"), TensorID(getNodeID(call), "data"))
    # block = RecvConstBlock(_edge, f"qreg reset")
    # call.codeblocks.append(hid, block, CodePhase.INIT)

    is_non_multicast_split, channels, num_splits = self.consumer_is_non_multicast_split(call)

    # set o_split_idx to 0 when last_tuple_idx is None
    block = MinmaxQuantBlock(call, call.last_tuple_idx or 0, "min_max_quantize")
    if call.curr_composite_id is not None:
      call.post_op_stack.append(block)
    elif not is_non_multicast_split:
      # Standalone minmax quantize needs RECV/SEND wrapper
      wrapped_block = RecvSendWrapper.from_codeblock(block, "min_max_quantize_standalone", builder=call.builder).create_loop_from_call(call)
      call.codeblocks.append(hid, wrapped_block, CodePhase.EXEC)
    else:
      call.codeblocks.append(hid, block.build_mmquant_for_dwconv(), CodePhase.EXEC)


@register_operation_handler
class ReLUHandler(OperationHandler):
  """Handles nn.relu operations.

  Generates a standalone ReLUBlock in the EXEC phase.
  """

  @property
  def priority(self) -> int:
    return 10

  def can_handle(self, call: relay.Call) -> bool:
    return call.op == op.get("nn.relu")

  def handle(self, call: 'BuilderContext') -> None:
    hid = call.get_hid()

    # Create ReLU computation block
    block = ReLUBlock(call, "relu")

    # Wrap with RECV/SEND if standalone, or add as post-op if in composite
    if call.curr_composite_id is not None:
      call.post_op_stack.append(block)
    else:
      # Standalone ReLU needs RECV/SEND wrapper
      wrapped_block = RecvSendWrapper.from_codeblock(block, "relu_standalone", builder=call.builder).create_loop_from_call(call)
      call.codeblocks.append(hid, wrapped_block, CodePhase.EXEC)


@register_operation_handler
class BiasAddHandler(OperationHandler):
  """Handles nn.bias_add operations (currently disabled).

  Generates RecvConstBlock for bias and AddBlock as a post-op.
  """

  @property
  def priority(self) -> int:
    return 10

  def can_handle(self, call: relay.Call) -> bool:
    # Currently disabled - return False to skip
    return False
    # Uncomment to enable:
    # return call.op == op.get("nn.bias_add")

  def handle(self, call: 'BuilderContext') -> None:
    assert call.curr_composite_id, \
        f"BiasAdd must be inside a composite function, got gid: {call.get_gid()}"
    hid = call.get_hid()

    bias_edge = call.get_tensor_edge_from_tag("bias")
    block = RecvConstBlock(bias_edge, RecvConstBlock.ConstType.NORMAL, "bias write")
    call.codeblocks.append(hid, block, CodePhase.INIT)

    # add constedge info to codeblock info
    IMCECodeBlockInfo().append_const_edge_info(bias_edge, hid)

    block = AddBlock(call, "add_bias")
    call.post_op_stack.append(block)


@register_operation_handler
class BatchNormHandler(OperationHandler):
  """Handles imcflow.fused_batch_norm operations (currently disabled).

  Generates RecvConstBlock for scale/bias and VecBlock/AddBlock as post-ops.
  """

  @property
  def priority(self) -> int:
    return 10

  def can_handle(self, call: relay.Call) -> bool:
    return call.op == op.get("imcflow.fused_batch_norm")

  def handle(self, call: 'BuilderContext') -> None:
    print(f"[IMCE CODE BUILDER] handle BatchNorm: {getNodeID(call.call)} {getNodeDebugID(call.call)}")
    hid = call.get_hid()
    scale_edge = call.get_tensor_edge_from_tag("fused_scale")
    bias_edge = call.get_tensor_edge_from_tag("fused_bias")

    print(f"[IMCE CODE BUILDER] BatchNormHandler: append recv const block hid={hid}, scale_edge={scale_edge}, bias_edge={bias_edge}")
    block = RecvConstBlock(scale_edge, RecvConstBlock.ConstType.NORMAL, f"{scale_edge}, fused_scale write")
    call.codeblocks.append(hid, block, CodePhase.INIT)
    # add constedge info to codeblock info
    IMCECodeBlockInfo().append_const_edge_info(scale_edge, hid)
    block = RecvConstBlock(bias_edge, RecvConstBlock.ConstType.NORMAL, f"{bias_edge}, fused_bias write")
    call.codeblocks.append(hid, block, CodePhase.INIT)
    # add constedge info to codeblock info
    IMCECodeBlockInfo().append_const_edge_info(bias_edge, hid)

    block = BatchNormBlock(call, "batch_norm")
    if call.curr_composite_id:
      # TODO: how to scale?
      call.post_op_stack.append(block)
    else:
      wrapped_block = RecvSendWrapper.from_codeblock(block, "bn_standalone", builder=call.builder).create_loop_from_call(call)
      call.codeblocks.append(hid, wrapped_block, CodePhase.EXEC)

@register_operation_handler
class NuQuantizeHandler(OperationHandler):
  """Handles qnn.imcflow_nu_quantize operations (currently disabled).

  This is a placeholder handler that does nothing.
  """

  @property
  def priority(self) -> int:
    return 10

  def can_handle(self, call: relay.Call) -> bool:
    return call.op == op.get("qnn.imcflow_nu_quantize")

  def handle(self, call: 'BuilderContext') -> None:
    # Currently does nothing
    pass

class IMCECodeBlockInfo:
  """Holds ordered constant receive edges per IMCE core.

  Structure:
    self.imce_const_edges[hid] = [TensorEdge, ...] in arrival order.
  """
  def __new__(cls):
    if not hasattr(cls, "instance"):
      cls.instance = super(IMCECodeBlockInfo, cls).__new__(cls)
      cls.instance._initialize()
    return cls.instance

  def _initialize(self):
    super().__init__()
    self.imce_const_edges = {
      hid: [] for hid in NodeID.imces()
    }  # key: hid value: list[TensorEdge]
  
  def clear(self):
    self._initialize()

  def append_const_edge_info(self, edge: TensorEdge, hid: NodeID):
    self.imce_const_edges[hid].append(edge)