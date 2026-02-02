"""
Op-specific tensor tagging logic for IMCFLOW backend.

This module defines how tensor inputs/outputs are tagged (odata, scale, weight, etc.)
for each supported operation, enabling proper classification as data tensors or
group constants.
"""

from tvm import relay
from tvm.relay.expr import Constant, Var, Tuple, TupleGetItem

# ============================================================
# Tagging Table: Op별 입력 태그 정의
# ============================================================

# Standard ops: list of (src_tag, dst_tag) tuples, indexed by arg position
# Element-wise ops with potential constant inputs have special handling
OP_INPUT_TAGS = {
    # Split/Concat: single data input
    "split": [("odata", "data")],
    "concatenate": [("odata", "data")],

    # Conv ops
    "nn.conv2d": [("odata", "data"), ("weight", "weight")],
    "nn.bias_add": [("odata", "data"), ("bias", "bias")],
    "nn.relu": [("odata", "data")],

    # IMCFlow custom ops
    "nn.imcflow_qconv": [("odata", "data"), ("weight", "weight"), ("config", "config")],
    "nn.imcflow_qdwconv": [("odata", "data"), ("weight", "weight"), ("config", "config")],
    "imcflow.fused_batch_norm": [("odata", "data"), ("fused_scale", "fused_scale"), ("fused_bias", "fused_bias")],

    # Quantization ops
    "qnn.imcflow_min_max_quantize": [("odata", "data"), ("min", "min"), ("max", "max")],
    "qnn.imcflow_nu_quantize": [("odata", "data"), ("threshold", "threshold")],
}

# Element-wise binary ops that need constant detection
ELEMENTWISE_BINARY_OPS = {"add", "multiply", "divide"}


class OpTagger:
    """Op별 tagging 로직을 처리하는 클래스"""

    @staticmethod
    def is_elementwise_binary_op(op_name: str) -> bool:
        """Element-wise binary op인지 확인 (상수 입력 감지 필요)"""
        return op_name in ELEMENTWISE_BINARY_OPS

    @staticmethod
    def get_elementwise_tags(call, op_name: str):
        """
        Element-wise op의 태그 결정.
        상수 입력이 있으면 scale로, 아니면 odata로 태깅.

        Returns:
            List of (arg_index, src_tag, dst_tag, needs_split_idx)
            needs_split_idx: True if this arg needs split index (odata), False otherwise
        """
        # 상수 입력 감지
        if isinstance(call.args[0], Constant):
            scale_idx, data_idx = 0, 1
        elif isinstance(call.args[1], Constant):
            scale_idx, data_idx = 1, 0
        else:
            scale_idx, data_idx = None, None

        if scale_idx is not None:
            # 하나가 상수인 경우: data는 odata, 상수는 scale
            return [
                (data_idx, "odata", "lhs", True),
                (scale_idx, "scale", "rhs", False),
            ]
        else:
            # 둘 다 데이터인 경우
            return [
                (0, "odata", "lhs", True),
                (1, "odata", "rhs", True),
            ]

    @staticmethod
    def get_standard_tags(op_name: str):
        """
        일반 op의 태그 반환.

        Returns:
            List of (arg_index, src_tag, dst_tag, needs_split_idx)
        """
        tags = OP_INPUT_TAGS.get(op_name, [])
        result = []
        for i, (src_tag, dst_tag) in enumerate(tags):
            # odata needs split index, others don't
            needs_split_idx = (src_tag == "odata")
            result.append((i, src_tag, dst_tag, needs_split_idx))
        return result

    @staticmethod
    def get_input_tags(call, op_name: str):
        """
        Op에 맞는 입력 태그를 반환하는 통합 메서드.

        Returns:
            List of (arg_index, src_tag, dst_tag, needs_split_idx)
        """
        if OpTagger.is_elementwise_binary_op(op_name):
            return OpTagger.get_elementwise_tags(call, op_name)
        elif op_name in OP_INPUT_TAGS:
            return OpTagger.get_standard_tags(op_name)
        else:
            return None  # Unknown op
