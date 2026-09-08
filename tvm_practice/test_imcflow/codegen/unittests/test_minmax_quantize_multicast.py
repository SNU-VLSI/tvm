"""The handler and renderer must agree on ordinary multicast vs DW splitting."""

from types import SimpleNamespace

import pytest
from tvm import relay
from tvm.contrib.imcflow import TensorEdge, TensorID
from tvm.relay.backend.contrib.imcflow import imce_codeblock as blocks
from tvm.relay.backend.contrib.imcflow.imce_operation_handlers import MinMaxQuantizeHandler


@pytest.fixture
def classify(monkeypatch):
    monkeypatch.setenv("IMCFLOW_BUGFIX", "on")
    data = relay.var("data", shape=(1, 64, 4, 4), dtype="int16")
    nodes = {
        1: relay.Call(relay.op.get("nn.imcflow_qconv"), [data]),
        2: relay.Call(relay.op.get("nn.imcflow_qconv"), [data]),
        3: relay.split(data, 2, axis=1).tuple_value,
        4: relay.split(data, 2, axis=1).tuple_value,
    }
    info = {"func": {3: {"is_multi_cast": False, "channels": 64, "num_splits": 2}}}
    monkeypatch.setattr(blocks, "CustomIDToNode", lambda: nodes)
    monkeypatch.setattr(blocks, "getNodeID", lambda node: next(k for k, v in nodes.items() if v.same_as(node)))
    monkeypatch.setattr(blocks, "DevConfig", lambda: SimpleNamespace(SplitInfo=info))

    def check(destinations, expected=None, error=None, sources=None):
        sources = sources or [10] * len(destinations)
        edges = [TensorEdge(TensorID(src, "odata"), TensorID(dst, "data"))
                 for src, dst in zip(sources, destinations)]
        call = SimpleNamespace(func_name="func", get_output_edges=lambda: edges)
        block = SimpleNamespace(out_edges=edges, call=call)
        methods = (
            lambda: MinMaxQuantizeHandler.consumer_is_non_multicast_split(None, call),
            lambda: blocks.MinmaxQuantBlock.consumer_is_non_multicast_split(block),
        )
        for method in methods:
            if error:
                with pytest.raises(error):
                    method()
            else:
                assert method() == expected
    return check, info


@pytest.mark.parametrize("destinations", [[1], [1, 2], [(9, 1), 2]])
def test_ordinary_consumers(classify, destinations):
    check, _ = classify
    check(destinations, (False, None, None))


@pytest.mark.parametrize("multicast", [False, True])
def test_single_split_preserves_metadata(classify, multicast):
    check, info = classify
    info["func"][3]["is_multi_cast"] = multicast
    check([3], (not multicast, 64, 2))


@pytest.mark.parametrize("destinations,sources", [([], None), ([1, 2], [10, 11]),
                                                        ([1, 3], None), ([3, 4], None)])
def test_unsupported_outputs(classify, destinations, sources):
    check, _ = classify
    check(destinations, error=ValueError, sources=sources)


def test_missing_split_metadata_is_not_ordinary_multicast(classify):
    check, info = classify
    info["func"].clear()
    check([3], error=KeyError)
