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
"""Abstract class used by :py:class:`tvm.contrib.relay_viz.RelayVisualizer`."""
import abc
from typing import (
    Dict,
    Union,
    Tuple,
    List,
)

import tvm
from tvm import relay

UNKNOWN_TYPE = "unknown"

from tvm.relay.op.contrib.imcflow import HashToCustomID

def addCustomID(node, node_detail):
  id_dict = HashToCustomID()
  if int(hash(node)) in id_dict:
    if isinstance(node_detail, str):
      node_detail += f"\nCustomID : {id_dict[int(hash(node))]}"
    elif isinstance(node_detail, list):
      node_detail.append(f"CustomID : {id_dict[int(hash(node))]}")
    else:
      raise ValueError("node_detail must be either str or list for custom ID")
  return node_detail


class VizNode:
    """VizNode carry node information for `VizGraph` interface.

    Parameters
    ----------
    node_id: str
        Unique identifier for this node.
    node_type: str
        Type of this node.
    node_detail: str
        Any supplement for this node such as attributes.
    """

    def __init__(self, node_id: str, node_type: str, node_detail: str):
        self._id = node_id
        self._type = node_type
        self._detail = node_detail

    @property
    def identity(self) -> str:
        return self._id

    @property
    def type_name(self) -> str:
        return self._type

    @property
    def detail(self) -> str:
        return self._detail

    def __repr__(self) -> str:
        detail = self._detail.replace("\n", ", ")
        return f"VizNode(identity: {self._id}, type_name: {self._type}, detail: {detail}"


class VizEdge:
    """VizEdge connect two `VizNode`.

    Parameters
    ----------
    start_node: str
        The identifier of the node starting the edge.
    end_node: str
        The identifier of the node ending the edge.
    """

    def __init__(self, start_node: str, end_node: str):
        self._start_node = start_node
        self._end_node = end_node

    @property
    def start(self) -> str:
        return self._start_node

    @property
    def end(self) -> str:
        return self._end_node


class VizParser(abc.ABC):
    """VizParser parses out a VizNode and VizEdges from a `relay.Expr`."""

    @abc.abstractmethod
    def get_node_edges(
        self,
        node: relay.Expr,
        relay_param: Dict[str, tvm.runtime.NDArray],
        node_to_id: Dict[relay.Expr, str],
    ) -> Tuple[Union[VizNode, None], List[VizEdge]]:
        """Get VizNode and VizEdges for a `relay.Expr`.

        Parameters
        ----------
        node : relay.Expr
            relay.Expr which will be parsed and generate a node and edges.

        relay_param: Dict[str, tvm.runtime.NDArray]
            relay parameters dictionary.

        node_to_id : Dict[relay.Expr, str]
            This is a mapping from relay.Expr to a unique id, generated by `RelayVisualizer`.

        Returns
        -------
        rv1 : Union[VizNode, None]
            VizNode represent the relay.Expr. If the relay.Expr is not intended to introduce a node
            to the graph, return None.

        rv2 : List[VizEdge]
            a list of VizEdges to describe the connectivity of the relay.Expr.
            Can be empty list to indicate no connectivity.
        """


class VizGraph(abc.ABC):
    """Abstract class for graph, which is composed of nodes and edges."""

    @abc.abstractmethod
    def node(self, viz_node: VizNode) -> None:
        """Add a node to the underlying graph.
        Nodes in a Relay IR Module are expected to be added in the post-order.

        Parameters
        ----------
        viz_node : VizNode
            A `VizNode` instance.
        """

    @abc.abstractmethod
    def edge(self, viz_edge: VizEdge) -> None:
        """Add an edge to the underlying graph.

        Parameters
        ----------
        viz_edge : VizEdge
            A `VizEdge` instance.
        """


class DefaultVizParser(VizParser):
    """DefaultVizParser provde a set of logics to parse a various relay types.
    These logics are inspired and heavily based on
    `visualize` function in https://tvm.apache.org/2020/07/14/bert-pytorch-tvm
    """

    def get_node_edges(
        self,
        node: relay.Expr,
        relay_param: Dict[str, tvm.runtime.NDArray],
        node_to_id: Dict[relay.Expr, str],
    ) -> Tuple[Union[VizNode, None], List[VizEdge]]:
        if isinstance(node, relay.Function):
            return self._function(node, node_to_id)
        if isinstance(node, relay.expr.Call):
            return self._call(node, node_to_id)
        if isinstance(node, relay.expr.Var):
            return self._var(node, relay_param, node_to_id)
        if isinstance(node, relay.expr.Tuple):
            return self._tuple(node, node_to_id)
        if isinstance(node, relay.expr.TupleGetItem):
            return self._tuple_get_item(node, node_to_id)
        if isinstance(node, relay.expr.Constant):
            return self._constant(node, node_to_id)
        # GlobalVar possibly mean another global relay function,
        # which is expected to in "Graph" level, not in "Node" level.
        if isinstance(node, (relay.expr.GlobalVar, tvm.ir.Op)):
            return None, []

        viz_node = VizNode(node_to_id[node], UNKNOWN_TYPE, f"don't know how to parse {type(node)}")
        viz_edges = []
        return viz_node, viz_edges

    def _var(
        self,
        node: relay.Expr,
        relay_param: Dict[str, tvm.runtime.NDArray],
        node_to_id: Dict[relay.Expr, str],
    ) -> Tuple[Union[VizNode, None], List[VizEdge]]:
        """Render rule for a relay var node"""

        node_id = node_to_id[node]
        name_hint = node.name_hint
        node_detail = f"name_hint: {name_hint}"
        node_type = "Var(Param)" if name_hint in relay_param else "Var(Input)"

        if node.type_annotation is not None:
            if hasattr(node.type_annotation, "shape"):
                shape = tuple(map(int, node.type_annotation.shape))
                dtype = node.type_annotation.dtype
                node_detail = f"{node_detail}\nshape: {shape}\ndtype: {dtype}"
            else:
                node_detail = f"{node_detail}\ntype_annotation: {node.type_annotation}"

        node_detail = addCustomID(node, node_detail)

        # only node
        viz_node = VizNode(node_id, node_type, node_detail)
        viz_edges = []
        return viz_node, viz_edges

    def _function(
        self,
        node: relay.Expr,
        node_to_id: Dict[relay.Expr, str],
    ) -> Tuple[Union[VizNode, None], List[VizEdge]]:
        """Render rule for a relay function node"""
        func_attrs = node.attrs
        node_detail = [f"{k}: {func_attrs.get_str(k)}" for k in func_attrs.keys()]
        # "Composite" might from relay.transform.MergeComposite
        if "Composite" in func_attrs.keys():
            name = func_attrs["Composite"]
        else:
            name = ""

        node_detail = addCustomID(node, node_detail)
        node_id = node_to_id[node]

        # Body -> FunctionNode
        viz_node = VizNode(node_id, f"Func {name}", "\n".join(node_detail))
        viz_edges = [VizEdge(node_to_id[node.body], node_id)]
        return viz_node, viz_edges

    def _call(
        self,
        node: relay.Expr,
        node_to_id: Dict[relay.Expr, str],
    ) -> Tuple[Union[VizNode, None], List[VizEdge]]:
        """Render rule for a relay call node"""
        node_id = node_to_id[node]
        op_name = UNKNOWN_TYPE
        node_detail = []
        if isinstance(node.op, tvm.ir.Op):
            op_name = node.op.name
            if node.attrs:
                node_detail = [f"{k}: {node.attrs.get_str(k)}" for k in node.attrs.keys()]
        elif isinstance(node.op, relay.Function):
            func_attrs = node.op.attrs
            op_name = "Anonymous Func"
            node_detail = [f"{k}: {func_attrs.get_str(k)}" for k in func_attrs.keys()]
            # "Composite" might from relay.transform.MergeComposite
            if "Composite" in func_attrs.keys():
                op_name = func_attrs["Composite"]
        elif isinstance(node.op, relay.GlobalVar):
            op_name = "GlobalVar"
            node_detail = [f"GlobalVar.name_hint: {node.op.name_hint}"]
        else:
            op_name = str(type(node.op)).split(".")[-1].split("'")[0]

        node_detail = addCustomID(node, node_detail)

        # Arguments -> CallNode
        viz_node = VizNode(node_id, f"Call {op_name}", "\n".join(node_detail))
        args = [node_to_id[arg] for arg in node.args]
        viz_edges = [VizEdge(arg, node_id) for arg in args]
        return viz_node, viz_edges

    def _tuple(
        self,
        node: relay.Expr,
        node_to_id: Dict[relay.Expr, str],
    ) -> Tuple[Union[VizNode, None], List[VizEdge]]:
        node_id = node_to_id[node]

        # Fields -> TupleNode
        viz_node = VizNode(node_id, "Tuple", "")
        viz_edges = [VizEdge(node_to_id[field], node_id) for field in node.fields]
        return viz_node, viz_edges

    def _tuple_get_item(
        self,
        node: relay.Expr,
        node_to_id: Dict[relay.Expr, str],
    ) -> Tuple[Union[VizNode, None], List[VizEdge]]:
        node_id = node_to_id[node]

        # Tuple -> TupleGetItemNode
        viz_node = VizNode(node_id, "TupleGetItem", f"idx: {node.index}")
        viz_edges = [VizEdge(node_to_id[node.tuple_value], node_id)]
        return viz_node, viz_edges

    def _constant(
        self,
        node: relay.Expr,
        node_to_id: Dict[relay.Expr, str],
    ) -> Tuple[Union[VizNode, None], List[VizEdge]]:
        node_id = node_to_id[node]
        node_detail = f"shape: {node.data.shape}, dtype: {node.data.dtype}"

        node_detail = addCustomID(node, node_detail)

        # only node
        viz_node = VizNode(node_id, "Const", node_detail)
        viz_edges = []
        return viz_node, viz_edges


class Plotter(abc.ABC):
    """Plotter can render a collection of Graph interfaces to a file."""

    @abc.abstractmethod
    def create_graph(self, name: str) -> VizGraph:
        """Create a VizGraph

        Parameters
        ----------
        name : str
            the name of the graph

        Return
        ------
        rv1: an instance of class inheriting from VizGraph interface.
        """

    @abc.abstractmethod
    def render(self, filename: str) -> None:
        """Render the graph as a file.

        Parameters
        ----------
        filename : str
            see the definition of implemented class.
        """
