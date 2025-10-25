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
from tvm.relay.backend.contrib.imcflow.transform import getNodeID
from tvm.relay.backend.contrib.imcflow.imce_codeblock import *
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig


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

    def handle(self, call) -> None:
        # Set composite context
        call.curr_composite_id = getNodeID(call.call)
        # Visit the body of the composite function using self.builder
        self.builder.visit(call.call.op.body)
        # Clear composite context
        call.curr_composite_id = None
        # Visit arguments
        for a in call.call.args:
            self.builder.visit(a)


@register_operation_handler
class ConvHandler(OperationHandler):
    """Handles nn.imcflow_qconv operations.

    Generates ConvBlock for convolution execution and LoadLBBlock for
    weight initialization.
    """

    @property
    def priority(self) -> int:
        return 10

    def can_handle(self, call: relay.Call) -> bool:
        return call.op == op.get("nn.imcflow_qconv")

    def handle(self, call) -> None:
        # Get argument dictionaries and shapes
        args = call.get_arg_dict()
        shapes = call.get_arg_shape_dict()
        shapes["output"] = infer_shape(call.call)

        # Get input/output edges
        in_edges = call.get_input_edges()
        out_edge = call.get_output_edge()

        # Find weight edge
        w_edge = None
        for edge in in_edges:
            if edge.src_id.tensor_type == "weight":
                w_edge = edge
                break

        w_info = DevConfig().get_tensor_edge_info(w_edge)
        w_tid = w_edge.src_id
        hid = call.get_hid()

        # scan reg
        # TODO: add scan reg code block

        # config reg
        # TODO: add config reg code block

        # write weights using recv
        size = DevConfig().MemLayout.get_data_block_by_id(w_tid).size
        # TODO: change to write weight block
        block = LoadLBBlock(size, 1, w_info.fifo_id, "weight write")
        call.codeblocks.append(hid, block, CodePhase.INIT)

        block = ConvBlock(in_edges, out_edge, shapes, call.call.attrs, "conv exec")
        # FIXME: this assumes that convblock is called first... we don't want that
        call.curr_conv_block = block
        call.codeblocks.append(hid, block, CodePhase.EXEC)


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

    def handle(self, call) -> None:
        assert call.curr_composite_id, "Add must be inside a composite function"

        in_edges = call.get_input_edges()
        out_edge = call.get_output_edge()

        block = AddBlock(in_edges, out_edge, "add")
        call.curr_conv_block.add_post_op(block)


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

    def handle(self, call) -> None:
        # TODO: divide block should be replaced later
        assert call.curr_composite_id, "Divide must be inside a composite function"

        in_edges = call.get_input_edges()
        out_edge = call.get_output_edge()
        block = DivBlock(in_edges, out_edge, "div")
        call.curr_conv_block.add_post_op(block)


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

    def handle(self, call) -> None:
        hid = call.get_hid()
        conv_block = call.get_conv_block_by_hid(hid)

        in_edges = call.get_input_edges()
        out_edge = call.get_output_edge()

        block = ConcatBlock(in_edges, out_edge, "concat")
        conv_block.add_post_op(block)


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

    def handle(self, call) -> None:
        hid = call.get_hid()
        if hid.is_imce():
            conv_block = call.get_conv_block_by_hid(hid)

            in_edge = call.get_input_edge()
            out_edges = call.get_output_edges()

            block = SplitBlock(in_edge, out_edges, "split")
            conv_block.add_post_op(block)


@register_operation_handler
class MinMaxQuantizeHandler(OperationHandler):
    """Handles qnn.imcflow_min_max_quantize operations.

    Generates RecvConstBlock for min/max parameters and MinmaxQuantBlock
    as a post-op to the current convolution.
    """

    @property
    def priority(self) -> int:
        return 10

    def can_handle(self, call: relay.Call) -> bool:
        return call.op == op.get("qnn.imcflow_min_max_quantize")

    def handle(self, call) -> None:
        assert call.curr_composite_id, \
            f"MinMaxQuantize must be inside a composite function, got gid: {call.get_gid()}"
        hid = call.get_hid()

        # Generate RecvConst blocks for min/max parameters
        for tag in ("min", "max"):
            edge = call.get_tensor_edge_from_tag(tag)
            # TODO: inode code block needs to put appropriate address for min/max reg.
            # TODO: two ways to set min/max reg. RecvConst vs. ADDI
            block = RecvConstBlock(edge, f"{tag} write")
            call.codeblocks.append(hid, block, CodePhase.INIT)

        # TODO: add reset qreg code block
        # _edge = TensorEdge(TensorID(-1, "zero"), TensorID(getNodeID(call), "data"))
        # block = RecvConstBlock(_edge, f"qreg reset")
        # call.codeblocks.append(hid, block, CodePhase.INIT)

        in_edges = call.get_input_edges()
        out_edge = call.get_output_edge()
        # set o_split_idx to 0 when last_tuple_idx is None
        block = MinmaxQuantBlock(in_edges, out_edge, call.last_tuple_idx or 0, "min_max_quantize")
        call.curr_conv_block.add_post_op(block)


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

    def handle(self, call) -> None:
        hid = call.get_hid()
        in_edges = call.get_input_edges()
        out_edge = call.get_output_edge()
        block = ReLUBlock(in_edges, out_edge, "relu")
        call.codeblocks.append(hid, block, CodePhase.EXEC)
        # call.curr_conv_block.add_post_op(block)


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

    def handle(self, call) -> None:
        assert call.curr_composite_id, \
            f"BiasAdd must be inside a composite function, got gid: {call.get_gid()}"
        hid = call.get_hid()

        bias_edge = call.get_tensor_edge_from_tag("bias")
        block = RecvConstBlock(bias_edge, "bias write")
        call.codeblocks.append(hid, block, CodePhase.INIT)

        in_edges = call.get_input_edges()
        out_edge = call.get_output_edge()
        block = AddBlock(in_edges, out_edge, "add_bias")
        call.curr_conv_block.add_post_op(block)


@register_operation_handler
class BatchNormHandler(OperationHandler):
    """Handles imcflow.fused_batch_norm operations (currently disabled).

    Generates RecvConstBlock for scale/bias and VecBlock/AddBlock as post-ops.
    """

    @property
    def priority(self) -> int:
        return 10

    def can_handle(self, call: relay.Call) -> bool:
        # Currently disabled - return False to skip
        return False
        # Uncomment to enable:
        # return call.op == op.get("imcflow.fused_batch_norm")

    def handle(self, call) -> None:
        assert call.curr_composite_id, \
            f"BatchNorm must be inside a composite function, got gid: {call.get_gid()}"
        hid = call.get_hid()
        scale_edge = call.get_tensor_edge_from_tag("fused_scale")
        bias_edge = call.get_tensor_edge_from_tag("fused_bias")

        block = RecvConstBlock(scale_edge, "fused_scale write")
        call.codeblocks.append(hid, block, CodePhase.INIT)
        block = RecvConstBlock(bias_edge, "fused_bias write")
        call.codeblocks.append(hid, block, CodePhase.INIT)

        in_edges = call.get_input_edges()
        out_edge = call.get_output_edge()
        # TODO: how to scale?
        block = VecBlock(in_edges, out_edge, "batch_norm_scale")
        call.curr_conv_block.add_post_op(block)
        block = AddBlock(in_edges, out_edge, "batch_norm_bias")
        call.curr_conv_block.add_post_op(block)


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

    def handle(self, call) -> None:
        # Currently does nothing
        pass
