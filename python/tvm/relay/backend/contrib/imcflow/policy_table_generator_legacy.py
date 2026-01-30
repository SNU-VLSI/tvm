"""
Legacy Policy Table Generator (XY/YX Routing)

This is the original PolicyTableGenerator that uses XY/YX routing.
Kept for reference and backward compatibility testing.

For new code, use the 3-phase pipeline:
- Phase 1: Router (MCFRouter or other)
- Phase 2: PathTreeBuilder
- Phase 3: PolicyTableGenerator (from policy_table_generator.py)
"""

import tvm
from tvm import relay
from tvm.contrib.imcflow import (
    ImcflowDeviceConfig,
    TensorEdge,
    NodeID,
    TensorEdgeInfo,
    InstEdgeInfo,
    RouterEntry,
    DataBlock,
)
from tvm.relay.op.contrib.imcflow import CustomIDToName, CustomIDToNode


def getInnerNodeID(node):
    """Extract inner node ID from potentially nested tuple."""
    if isinstance(node, tuple):
        return node[1]
    else:
        return node


class PolicyTableGeneratorLegacy:
    """
    Original Policy Table Generator using XY/YX routing.

    This implementation uses a greedy approach with XY routing (horizontal first,
    then vertical), falling back to YX routing if capacity is exceeded.
    """

    def __init__(self, NoCPaths):
        self.NoCPaths = NoCPaths
        self.PolicyTable_2D = {}

    def run(self, mod):
        class _PolicyTableGenerator(tvm.relay.ExprVisitor):
            def __init__(self, NoCPaths):
                super().__init__()
                self.NoCPaths = NoCPaths
                self.router_entry_list_temp = {}
                self.Policytable = []
                self.explored_router_list = {}

                # Dictionary to store initial addresses for each source-index pair
                self.start_addr_dict = {}  # {(source, data type): start_address}

                self.table_capacity = 32
                self.InSubFunction = False
                self.SubFunctionMapping = None
                self.SubFunctionNodeID = None
                self.VarProperties = {}

            def generate_policy_table(self, func_name):
                # Initialize policy tables for all nodes using NodeID as keys
                # Each policy table starts with an all-zeros entry at address 0
                zero_entry = {"Local": {"enable": False, "chunk_index": 0, "addr": 0, "ksel": 0}, \
                              "North": {"enable": False, "addr": 0}, \
                              "East": {"enable": False, "addr": 0},  \
                              "South": {"enable": False, "addr": 0}, \
                              "West": {"enable": False, "addr": 0}}
                policy_tables = {node_id: [zero_entry.copy()] for node_id in NodeID}

                def get_direction(source_coord, dest_coord):
                    if source_coord[1] < dest_coord[1]:
                        return "East"
                    elif source_coord[1] > dest_coord[1]:
                        return "West"
                    elif source_coord[0] < dest_coord[0]:
                        return "South"
                    elif source_coord[0] > dest_coord[0]:
                        return "North"
                    return None

                def check_path_capacity(path_coords, explored_router_list):
                    """Check if all nodes in the path have available capacity"""
                    for coord in path_coords:
                        node = NodeID.from_coord(coord[0],coord[1])
                        if len(policy_tables[node]) >= self.table_capacity:
                            if explored_router_list is not None and coord in explored_router_list:
                                # This is only for multicast case, allow reusing existing path
                                # For single path case, this won't be triggered as the explored_router_list is None
                                continue
                            else:
                                return False
                    return True

                def get_path_coords(source_coord, dest_coord, is_xy_routing=True, explored_router_list=None):
                    """Get list of coordinates for the path"""
                    path_coords = []
                    current_coord = source_coord

                    if is_xy_routing:
                        # Move horizontally first (X)
                        while current_coord[1] != dest_coord[1]:
                            next_coord = (current_coord[0],
                                        current_coord[1] + (1 if current_coord[1] < dest_coord[1] else -1))
                            path_coords.append(next_coord)
                            current_coord = next_coord

                        # Then vertically (Y)
                        while current_coord[0] != dest_coord[0]:
                            next_coord = (current_coord[0] + (1 if current_coord[0] < dest_coord[0] else -1),
                                        current_coord[1])
                            path_coords.append(next_coord)
                            current_coord = next_coord
                    else:
                        # Move vertically first (Y)
                        while current_coord[0] != dest_coord[0]:
                            next_coord = (current_coord[0] + (1 if current_coord[0] < dest_coord[0] else -1),
                                        current_coord[1])
                            path_coords.append(next_coord)
                            current_coord = next_coord

                        # Then horizontally (X)
                        while current_coord[1] != dest_coord[1]:
                            next_coord = (current_coord[0],
                                        current_coord[1] + (1 if current_coord[1] < dest_coord[1] else -1))
                            path_coords.append(next_coord)
                            current_coord = next_coord

                    # check policy table's capacity along the designated routing path
                    if not check_path_capacity(path_coords, explored_router_list):
                        # If X-Y fails, try Y-X routing
                        path_coords = get_path_coords(source_coord, dest_coord, False, explored_router_list)
                        if not check_path_capacity(path_coords, explored_router_list):
                            raise ValueError("Routing failed for both X-Y and Y-X!")

                    #TODO: there may be cases that X-Y and Y-X both fails!!!!!

                    return path_coords

                def handle_single_path(edge, mapping_info, init_addr_save=True, router_entry_list=None):
                    """Append new entries to policy tables for a single destination"""
                    source_node = mapping_info[0]
                    dest_node = mapping_info[1]
                    dest_index = mapping_info[2] or 0 # split index for destination node
                    if mapping_info[2] is not None:
                        dst_graph_node = CustomIDToNode()[getInnerNodeID(edge.dst_id.graph_node_id)]
                        kernel_size = dst_graph_node.attrs['kernel_size'][0].value
                        ksel = kernel_size
                        if ksel not in [1, 2, 3, 5, 7]: raise ValueError("Unsupported kernel size for split index calculation.")
                    else:
                        ksel = 0

                    if isinstance(edge, NodeID):
                        src_node_data = f"instruction_{edge.name}"
                    else:
                        src_node_data = edge.src_id.graph_node_id

                    source_coord = NodeID.to_coord(source_node)
                    dest_coord = NodeID.to_coord(dest_node)
                    entry_addr = len(policy_tables[source_node])

                    if router_entry_list is None: # initial handling
                        router_entry_list= []
                        if source_coord == dest_coord: # if same node, return
                            return
                        # check if there's previous path with same source and same source tensor id, which means multicast(i.e. split operation)
                        elif any(k[0] == source_node and k[2] == src_node_data for k in self.start_addr_dict.keys()):
                            handle_multicast(edge, mapping_info)
                            return
                        else:
                            self.start_addr_dict[(source_node, dest_node, src_node_data)] = entry_addr # each source can have several tensor type

                    # Try X-Y routing first
                    path_coords = get_path_coords(source_coord, dest_coord, True)
                    if (source_node, dest_node, src_node_data) not in self.explored_router_list:
                        self.explored_router_list[(source_node, dest_node, src_node_data)] = path_coords
                    else:
                        self.explored_router_list[(source_node, dest_node, src_node_data)].extend(path_coords)

                    current_coord = source_coord
                    current_node = source_node
                    # Apply the successful path to tables
                    for next_coord in path_coords:
                        direction = get_direction(current_coord, next_coord)
                        next_node = NodeID.from_coord(next_coord[0], next_coord[1])

                        #append entry to router's policy table
                        entry = {"Local": {"enable": False, "chunk_index": 0, "addr": 0, "ksel":ksel}, \
                            "North": {"enable": False, "addr": 0}, \
                            "East": {"enable": False, "addr": 0},  \
                            "South": {"enable": False, "addr": 0}, \
                            "West": {"enable": False, "addr": 0}}

                        target_addr = len(policy_tables[next_node])
                        entry[direction]["addr"] = target_addr
                        entry[direction]["enable"] = True
                        policy_tables[current_node].append(entry)

                        #create RouterEntry and append to router_entry_list
                        router_entry_list.append((current_node, len(policy_tables[current_node])-1))

                        #switch to next node
                        current_coord = next_coord
                        current_node = NodeID.from_coord(current_coord[0], current_coord[1])

                    # insert entry for destination node
                    entry = {"Local": {"enable": True, "chunk_index": dest_index, "addr": 0, "ksel":ksel}, \
                        "North": {"enable": False, "addr": 0}, \
                        "East": {"enable": False, "addr": 0},  \
                        "South": {"enable": False, "addr": 0}, \
                        "West": {"enable": False, "addr": 0}}

                    policy_tables[dest_node].append(entry)

                    #create RouterEntry and append to RouterEntry_list
                    router_entry_list.append((dest_node, len(policy_tables[dest_node])-1))

                    # temporary saving. Final saving is done after whole paths finish.
                    self.router_entry_list_temp[edge] = router_entry_list

                def handle_multicast(edge, mapping_info):
                    """Handle multiple destinations with potential path sharing"""
                    source_node = mapping_info[0]
                    dest_node = mapping_info[1]
                    dest_index = mapping_info[2] or 0 # split index for destination node
                    if mapping_info[2] is not None:
                        dst_graph_node = CustomIDToNode()[getInnerNodeID(edge.dst_id.graph_node_id)]
                        kernel_size = dst_graph_node.attrs['kernel_size'][0].value
                        ksel = kernel_size
                        if ksel not in [1, 2, 3, 5, 7]: raise ValueError("Unsupported kernel size for split index calculation.")
                    else:
                        ksel = 0

                    if isinstance(edge, NodeID):
                        src_node_data = f"instruction_{edge.name}"
                    else:
                        src_node_data = edge.src_id.graph_node_id
                    router_entry_list= []

                    if source_node == dest_node: # if same node, return
                        return

                    # Follow existing path and modify at divergence point
                    previous_path_key = None
                    for k in self.start_addr_dict.keys():
                        if k[0] == source_node and k[2] == src_node_data:
                            previous_path_key = k
                            break
                    if previous_path_key is None:
                        raise ValueError("No previous path found for multicast handling.")

                    entry_addr = self.start_addr_dict[previous_path_key]
                    current_node = source_node
                    current_coord = NodeID.to_coord(current_node)
                    dest_coord = NodeID.to_coord(dest_node)
                    next_coord = None

                    while current_coord != dest_coord:
                        entry = policy_tables[current_node][entry_addr] # current policy table entry

                        # Find which direction to go next.
                        path_coords = get_path_coords(current_coord, dest_coord, self.explored_router_list[previous_path_key])
                        next_coord = path_coords[0]
                        next_node = NodeID.from_coord(next_coord[0],next_coord[1])
                        direction = get_direction(current_coord, next_coord)

                        # If direction is different from previous path, diverge!
                        if entry[direction]["enable"] is False:
                            # modify entry
                            target_addr = len(policy_tables[next_node])
                            policy_tables[current_node][entry_addr][direction]["addr"] = target_addr
                            policy_tables[current_node][entry_addr][direction]["enable"] = True

                            #create RouterEntry and append to router_entry_list
                            router_entry_list.append((current_node, entry_addr))

                            # diverge into new path
                            new_mapping = (next_node, mapping_info[1], mapping_info[2])
                            handle_single_path(edge, new_mapping, init_addr_save=False, router_entry_list=router_entry_list)
                            break
                        else:
                            # create RouterEntry and append to router_entry_list
                            router_entry_list.append((current_node, entry_addr))

                            # keep following the previous path
                            current_coord = next_coord
                            current_node = next_node
                            entry_addr = entry[direction]["addr"]

                            if current_node == dest_node: # if same node, return
                                policy_tables[dest_node][entry_addr]["Local"]["enable"] = True
                                policy_tables[dest_node][entry_addr]["Local"]["chunk_index"] = dest_index
                                policy_tables[dest_node][entry_addr]["Local"]["ksel"] = ksel
                                # create RouterEntry and append to router_entry_list
                                router_entry_list.append((current_node, entry_addr))
                                # temporary saving. Final saving is done after whole paths finish.
                                self.router_entry_list_temp[edge] = router_entry_list
                                break

                # Main logic
                for edge, mapping_info in self.NoCPaths.items():
                    handle_single_path(edge, mapping_info)

                self.Policytable = policy_tables
                ImcflowDeviceConfig().PolicyTableDict[func_name] = policy_tables

            def add_EdgeInfo(self, func_name):
                # after policy table entry generation finished, add to TensorEdgeToInfo
                fifo_id_cnt = {node_id: 2 for node_id in NodeID}
                ID_dict = CustomIDToName()
                for edge, mapping_info in self.NoCPaths.items():
                    # if tensoredge, save to TensorEdgetoInfo
                    dest_node = mapping_info[1]
                    router_entry_list=[]
                    if edge in self.router_entry_list_temp:
                        for entry_tuple in self.router_entry_list_temp[edge]:
                            router_entry_list.append(RouterEntry(entry_tuple[0], entry_tuple[1], self.Policytable[entry_tuple[0]][entry_tuple[1]]))

                        if isinstance(edge, TensorEdge): # TensorEdge
                            if edge.src_id.tensor_type in ["odata", "var"]:
                                # get src node name from CustomIDToName
                                dst_node_name = ID_dict[getInnerNodeID(edge.dst_id.graph_node_id)]

                                if dst_node_name in ["nn.imcflow_qconv", "nn.imcflow_qdwconv"]:
                                    edgeinfo = TensorEdgeInfo(router_entry_list, None, 0)
                                    ImcflowDeviceConfig().add_tensor_edge_info(edge, edgeinfo)
                                else:
                                    edgeinfo = TensorEdgeInfo(router_entry_list, None, fifo_id_cnt[dest_node])
                                    ImcflowDeviceConfig().add_tensor_edge_info(edge, edgeinfo)

                                    fifo_id_cnt[dest_node] = fifo_id_cnt[dest_node] + 1
                                    if fifo_id_cnt[dest_node] >= 8:
                                        raise ValueError("FIFO ID cannot be over 7!")

                            elif edge.src_id.tensor_type in ["weight", "bias", "fused_scale", "fused_bias", "min", "max", "threshold", "scale", "config"]:
                                edgeinfo = TensorEdgeInfo(router_entry_list, None, 1)
                                ImcflowDeviceConfig().add_tensor_edge_info(edge, edgeinfo)
                            else:
                                raise ValueError("Wrong tensor type!")

                        else: # Instruction edge
                            edgeinfo = InstEdgeInfo(router_entry_list, None)
                            ImcflowDeviceConfig().add_inst_edge_info(func_name, edge, edgeinfo)
                    else: # src hw node and dst hw node is equal. it is local edge
                        edgeinfo = TensorEdgeInfo([], None, TensorEdgeInfo.LOCAL_FIFO)
                        ImcflowDeviceConfig().add_tensor_edge_info(edge, edgeinfo)

            def allocate(self, func_name):
                # Allocate memory for policy tables
                for node_id, policy_table in self.Policytable.items():
                    if len(policy_table) == 0:
                        continue
                    mem_size = len(policy_table) * 32
                    mem_block = DataBlock(f"{node_id.name}_policy", mem_size)
                    inode_id = node_id.master() if node_id.is_imce() else node_id
                    ImcflowDeviceConfig().MemLayout[func_name][f"{inode_id.name}_data"].allocate(mem_block, phase="init")

            def update_device_config(self, func_name):
                # traverse input function by visit() to make PathDict and generate policy table for it
                self.generate_policy_table(func_name)
                self.add_EdgeInfo(func_name)
                self.allocate(func_name)
                return self.Policytable

        # Returns list of (GlobalVar, Function) pairs sorted alphabetically by function name
        for gv, func in mod.functions.items():
            if isinstance(func, relay.Function) and hasattr(func.attrs, "Compiler") and func.attrs["Compiler"]=="imcflow":
                self.PolicyTable_2D[gv.name_hint] = _PolicyTableGenerator(self.NoCPaths[gv.name_hint]).update_device_config(gv.name_hint)
                for x in self.PolicyTable_2D[gv.name_hint]:
                    print(x)

        return mod
