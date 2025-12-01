
def constructNoCPathDict(mod):
  """
  Make NoC path dict from tensor edge list.
  Plus, add instruction path from inode to imce nodes.
  """

  HwMapping = ImcflowDeviceConfig().HWNodeMap
  NocPaths = ImcflowDeviceConfig().NoCPaths
  IMCECOL = ImcflowDeviceConfig.IMCE_W_NUM
  for func_name_var, func in mod.functions.items():
    if func_name_var.name_hint == "main": continue
    elif func.attrs["Compiler"]=="imcflow":
      NocPaths[func_name_var.name_hint] = {}
      # tensor edge to path entry
      TensorEdgeList_ = ImcflowDeviceConfig().TensorEdgeListDict[func_name_var.name_hint]
      for tensor_edge in TensorEdgeList_:
        SrcTensorID = tensor_edge.src_id
        DstTensorID = tensor_edge.dst_id
        SplitIdx = tensor_edge.split_idx
        SrcGraphNode = CustomIDToNode()[getInnerNodeID(SrcTensorID.graph_node_id)]
        DstGraphNode = CustomIDToNode()[getInnerNodeID(DstTensorID.graph_node_id)]
        NocPaths[func_name_var.name_hint][tensor_edge] = (
          (HwMapping[getInnerNodeID(SrcTensorID.graph_node_id)], HwMapping[getInnerNodeID(DstTensorID.graph_node_id)], SplitIdx)
        )
        # if isinstance(SrcGraphNode, (Var, Constant)):
        #   # else, map src node into inode
        #   SrcHwNodeID = HwMapping[getOuterNodeID(SrcTensorID.graph_node_id)]
        #   DstHwNodeID = HwMapping[getOuterNodeID(DstTensorID.graph_node_id)]
        #   NocPaths[func_name_var.name_hint][tensor_edge] = (
        #     (SrcHwNodeID, DstHwNodeID, SplitIdx)
        #   )
        #   # # if "inode" not in DstHwNodeID:
        #   # if not DstHwNodeID.is_inode():
        #   #   InodeID = NodeID.from_inode_coord(NodeID.to_coord(DstHwNodeID)[0])
        #   #   NocPaths[func_name_var.name_hint][tensor_edge] = (
        #   #     (InodeID, DstHwNodeID, SplitIdx)
        #   #   )
        #   #   HwMapping[SrcTensorID.graph_node_id] = InodeID
        # elif hasattr(DstGraphNode, "attrs") and hasattr(DstGraphNode.attrs, "Compiler") and DstGraphNode.attrs["Compiler"] == "imcflow" :
        #   # if this tensoredge is the final edge directly connected to host (= if destination is function)
        #   SrcHwNodeID = HwMapping[getOuterNodeID(SrcTensorID.graph_node_id)]
        #   DstHwNodeID = HwMapping[getOuterNodeID(DstTensorID.graph_node_id)]
        #   NocPaths[func_name_var.name_hint][tensor_edge] = (
        #     (SrcHwNodeID, DstHwNodeID, SplitIdx)
        #   )
        #   # InodeID = NodeID.from_inode_coord(NodeID.to_coord(SrcHwNodeID)[0])
        #   # NocPaths[func_name_var.name_hint][tensor_edge] = (
        #   #   (HwMapping[getOuterNodeID(SrcTensorID.graph_node_id)], InodeID, SplitIdx)
        #   # )
        #   # HwMapping[DstTensorID.graph_node_id] = InodeID
        # else:
        #   NocPaths[func_name_var.name_hint][tensor_edge] = (
        #     (HwMapping[getOuterNodeID(SrcTensorID.graph_node_id)], HwMapping[getOuterNodeID(DstTensorID.graph_node_id)], SplitIdx)
        #   )

      # instruction path
      for DstHwNodeID in NodeID.imces():
        InodeID = NodeID.from_inode_coord(NodeID.to_coord(DstHwNodeID)[0])
        NocPaths[func_name_var.name_hint][DstHwNodeID] = (
          (InodeID, DstHwNodeID, None)
        )

@relay.transform.function_pass(opt_level=0)
class PolicyTableGenerator:
    def __init__(self, NoCPaths):
      self.NoCPaths = NoCPaths
      self.PolicyTable_2D = {}

    def transform_function(self, func, mod, ctx):
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

        def generate_policy_table(self):
            # Initialize policy tables for all nodes using NodeID as keys
            policy_tables = {node_id: [] for node_id in NodeID}

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
                dest_index = mapping_info[2]
                if isinstance(edge, NodeID):
                  dest_node_data_type = f"instruction_{edge.name}"
                else:
                  dest_node_data_type = edge.dst_id.tensor_type

                source_coord = NodeID.to_coord(source_node)
                dest_coord = NodeID.to_coord(dest_node)
                entry_addr = len(policy_tables[source_node])

                if router_entry_list is None: # initial handling
                    router_entry_list= []
                    if source_coord == dest_coord: # if same node, return
                        return
                    # check if there's previous path with same source and same tensor type, which means multicast
                    elif (source_node, dest_node_data_type) in self.start_addr_dict:
                        handle_multicast(edge, mapping_info)
                        return
                    else:
                        self.start_addr_dict[(source_node, dest_node_data_type)] = entry_addr # each source can have several tensor type

                # Try X-Y routing first
                path_coords = get_path_coords(source_coord, dest_coord, True)
                if (source_node, dest_node_data_type) not in self.explored_router_list:
                    self.explored_router_list[(source_node, dest_node_data_type)] = path_coords
                else:
                    self.explored_router_list[(source_node, dest_node_data_type)].extend(path_coords)

                current_coord = source_coord
                current_node = source_node
                # Apply the successful path to tables
                for next_coord in path_coords:
                    direction = get_direction(current_coord, next_coord)
                    next_node = NodeID.from_coord(next_coord[0], next_coord[1])

                    #append entry to router's policy table
                    entry = {"Local": {"enable": False, "chunk_index": 0, "addr": 0}, \
                      "North": {"enable": False, "addr": 0}, \
                      "South": {"enable": False, "addr": 0}, \
                      "East": {"enable": False, "addr": 0},  \
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
                entry = {"Local": {"enable": True, "chunk_index": dest_index, "addr": 0}, \
                  "North": {"enable": False, "addr": 0}, \
                  "South": {"enable": False, "addr": 0}, \
                  "East": {"enable": False, "addr": 0},  \
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
                # dest_index = mapping_info[2]
                if isinstance(edge, NodeID):
                  dest_node_data_type = f"instruction_{edge.name}"
                else:
                  dest_node_data_type = edge.dst_id.tensor_type
                router_entry_list= []

                if source_node == dest_node: # if same node, return
                    return

                # Follow existing path and modify at divergence point
                entry_addr = self.start_addr_dict[(source_node, dest_node_data_type)]
                current_node = source_node
                current_coord = NodeID.to_coord(current_node)
                dest_coord = NodeID.to_coord(dest_node)
                next_coord = None

                while current_coord != dest_coord:
                    entry = policy_tables[current_node][entry_addr] # current policy table entry

                    # Find which direction to go next.
                    path_coords = get_path_coords(current_coord, dest_coord, self.explored_router_list[(source_node, dest_node_data_type)])
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
                            # create RouterEntry and append to router_entry_list
                            router_entry_list.append((current_node, entry_addr))
                            # temporary saving. Final saving is done after whole paths finish.
                            self.router_entry_list_temp[edge] = router_entry_list
                            break

            # Main logic
            for edge, mapping_info in self.NoCPaths.items():
                handle_single_path(edge, mapping_info)

            self.Policytable = policy_tables
            ImcflowDeviceConfig().PolicyTableDict = policy_tables

        def add_EdgeInfo(self):
            # def get_meminfo(edge):
            #     if isinstance(edge.src_id, tuple):
            #         id = edge.src_id[1]
            #     else:
            #         id = edge.src_id

            #     size = self.DataBlockDict[id]["size"]
            #     offset = self.DataBlockDict[id]["offset"]
            #     base_address = self.DataBlockDict[id]["base_address"]
            #     meminfo = DataBlock(id, size)

            #     meminfo.set_offset(offset)
            #     meminfo.set_base_address(base_address)

            #     return meminfo

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
                      # find mem_info
                      # meminfo = get_meminfo(edge) # decided to erase MemoryBlock in EdgeInfo

                      # FIFO ID assign
                      # 0: conv input
                      # 1: const (including weight)
                      # 2~6: rest
                      # edgeinfo = ImcflowDeviceConfig().get_tensor_edge_info(edge)
                      # edgeinfo.set_policy_info(router_entry_list)

                      if edge.src_id.tensor_type in ["odata", "var"]:
                        # get src node name from CustomIDToName
                        dst_node_name = ID_dict[getInnerNodeID(edge.dst_id.graph_node_id)]

                        if dst_node_name == "nn.imcflow_qconv":
                          # if src is input of qconv, FIFO ID = 0
                          # edgeinfo.set_fifo_id(0)
                          edgeinfo = TensorEdgeInfo(router_entry_list, None, 0)
                          ImcflowDeviceConfig().add_tensor_edge_info(edge, edgeinfo)
                        else:
                          # if not, FIFO ID = 2~6
                          # edgeinfo.set_fifo_id(fifo_id_cnt[dest_node])
                          edgeinfo = TensorEdgeInfo(router_entry_list, None, fifo_id_cnt[dest_node])
                          ImcflowDeviceConfig().add_tensor_edge_info(edge, edgeinfo)

                          fifo_id_cnt[dest_node] = fifo_id_cnt[dest_node] + 1
                          if fifo_id_cnt[dest_node] >= 8:
                            raise ValueError("FIFO ID cannot be over 7!")

                      elif edge.src_id.tensor_type in ["odata", "weight", "bias", "fused_scale", "fused_bias", "min", "max", "threshold", "scale", "config"]:
                        # if const, FIFO ID = 1
                        edgeinfo = TensorEdgeInfo(router_entry_list, None, 1)
                        ImcflowDeviceConfig().add_tensor_edge_info(edge, edgeinfo)
                      else:
                        raise ValueError("Wrong tensor type!")

                  else: # Instruction edge
                      # meminfo = get_meminfo(edge) # decided to erase MemoryBlock in EdgeInfo
                      edgeinfo = InstEdgeInfo(router_entry_list, None)
                      ImcflowDeviceConfig().add_inst_edge_info(edge, edgeinfo)

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
            self.generate_policy_table()
            self.add_EdgeInfo()
            self.allocate(func_name)
            return self.Policytable

      # Returns list of (GlobalVar, Function) pairs sorted alphabetically by function name
      for gv, func in mod.functions.items():
        if isinstance(func, relay.Function) and hasattr(func.attrs, "Compiler") and func.attrs["Compiler"]=="imcflow":
          self.PolicyTable_2D[gv.name_hint] = _PolicyTableGenerator(self.NoCPaths[gv.name_hint]).update_device_config(gv.name_hint)
          for x in self.PolicyTable_2D[gv.name_hint]:
            print(x)

      return func

class TensorPathVisualizer:
    """
    Visualizes tensor routing paths in the 2D mesh NoC topology.
    
    For each imcflow function, generates an image showing:
    - 2D mesh grid with inodes and imces as labeled squares
    - Tensor paths as colored lines between nodes
    - Each tensor gets a unique color
    """
    
    def __init__(self, output_dir="noc_visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Import visualization libraries
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.collections import LineCollection
            self.plt = plt
            self.patches = patches
            self.LineCollection = LineCollection
        except ImportError:
            raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
    
    def visualize_all_functions(self, mod):
        """
        Generate visualizations for all imcflow functions in the module.
        
        Parameters
        ----------
        mod : tvm.IRModule
            The module containing imcflow functions
        """
        for gv, func in mod.functions.items():
            if isinstance(func, relay.Function) and hasattr(func.attrs, "Compiler") and func.attrs["Compiler"] == "imcflow":
                func_name = gv.name_hint
                debug_print(f"Generating visualization for function: {func_name}")
                self.visualize_function(func_name)
    
    def visualize_function(self, func_name):
        """
        Generate visualization for a single imcflow function.
        Creates separate images for each tensor type (odata, weight, bias, etc.)
        
        Parameters
        ----------
        func_name : str
            Name of the imcflow function
        """
        # Get NoC paths for this function
        if func_name not in ImcflowDeviceConfig().NoCPaths:
            debug_print(f"No NoC paths found for function {func_name}")
            return
        
        noc_paths = ImcflowDeviceConfig().NoCPaths[func_name]
        tensor_edge_list = ImcflowDeviceConfig().TensorEdgeListDict.get(func_name, [])
        
        # Create subdirectory for this function
        func_output_dir = os.path.join(self.output_dir, func_name)
        os.makedirs(func_output_dir, exist_ok=True)
        
        # Group NoC paths by tensor type
        paths_by_type = {}
        for edge, mapping_info in noc_paths.items():
            if isinstance(edge, TensorEdge):
                tensor_type = edge.src_id.tensor_type
                if tensor_type not in paths_by_type:
                    paths_by_type[tensor_type] = {}
                paths_by_type[tensor_type][edge] = mapping_info
        
        # Create a visualization for each tensor type
        if not paths_by_type:
            debug_print(f"No tensor edges found for function {func_name}")
            return
        
        # Create individual visualizations for each tensor type
        for tensor_type, type_paths in sorted(paths_by_type.items()):
            debug_print(f"  Creating visualization for {tensor_type}: {len(type_paths)} paths")
            
            # Create the visualization
            fig, ax = self._create_mesh_grid(title=f"{func_name} - {tensor_type} Paths")
            
            # Draw tensor paths for this type only
            self._draw_tensor_paths(ax, type_paths, tensor_edge_list)
            
            # Save the figure
            output_path = os.path.join(func_output_dir, f"{tensor_type}.png")
            self.plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.plt.close(fig)
            
            debug_print(f"    Saved: {output_path}")
        
        # Also create an overview image with all tensor types
        debug_print(f"  Creating overview with all {len(paths_by_type)} tensor types")
        fig, ax = self._create_mesh_grid(title=f"{func_name} - All Tensor Paths (Overview)")
        
        # Collect all tensor edges
        all_type_paths = {}
        for type_paths in paths_by_type.values():
            all_type_paths.update(type_paths)
        
        self._draw_tensor_paths(ax, all_type_paths, tensor_edge_list)
        
        overview_path = os.path.join(func_output_dir, "00_overview_all_types.png")
        self.plt.savefig(overview_path, dpi=300, bbox_inches='tight')
        self.plt.close(fig)
        
        debug_print(f"    Saved: {overview_path}")
        debug_print(f"Completed visualization for {func_name}: {len(paths_by_type)} tensor types")
    
    def _create_mesh_grid(self, title="NoC Tensor Routing Paths"):
        """
        Create the 2D mesh grid with nodes.
        
        Parameters
        ----------
        title : str, optional
            Title for the visualization
        
        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        # Grid dimensions
        rows = ImcflowDeviceConfig.INODE_NUM  # 4 rows
        cols = ImcflowDeviceConfig.NODE_COL_NUM  # 5 columns (1 inode + 4 imces)
        
        # Node size and spacing
        node_size = 1.0
        spacing = 0.5
        
        # Calculate figure size
        fig_width = cols * (node_size + spacing) + spacing
        fig_height = rows * (node_size + spacing) + spacing
        
        fig, ax = self.plt.subplots(figsize=(fig_width * 2, fig_height * 2))
        
        # Draw each node
        for node_id in NodeID:
            coord = NodeID.to_coord(node_id)
            row, col = coord
            
            # Calculate position (flip y-axis so row 0 is at top)
            x = col * (node_size + spacing) + spacing
            y = (rows - 1 - row) * (node_size + spacing) + spacing
            
            # Determine node color
            if node_id.is_inode():
                color = 'lightblue'
                edgecolor = 'darkblue'
            else:
                color = 'lightgreen'
                edgecolor = 'darkgreen'
            
            # Draw node as rectangle
            rect = self.patches.Rectangle((x, y), node_size, node_size, 
                                         linewidth=2, edgecolor=edgecolor, 
                                         facecolor=color, alpha=0.7)
            ax.add_patch(rect)
            
            # Add node label
            ax.text(x + node_size/2, y + node_size/2, node_id.name,
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Set axis properties
        ax.set_xlim(0, fig_width)
        ax.set_ylim(0, fig_height)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        return fig, ax
    
    def _draw_tensor_paths(self, ax, noc_paths, tensor_edge_list):
        """
        Draw tensor paths on the mesh grid.
        
        Parameters
        ----------
        ax : matplotlib axis
            The axis to draw on
        noc_paths : dict
            Dictionary mapping edges to (source_node, dest_node, dest_index) tuples
        tensor_edge_list : list
            List of TensorEdge objects for this function
        """
        # Get unique colors for each tensor edge
        num_tensor_edges = len([e for e in noc_paths.keys() if isinstance(e, TensorEdge)])
        colors = self._generate_colors(num_tensor_edges)
        
        # Node size and spacing (must match _create_mesh_grid)
        node_size = 1.0
        spacing = 0.5
        rows = ImcflowDeviceConfig.INODE_NUM
        
        # Track which tensor edges we've drawn
        tensor_edge_idx = 0
        legend_entries = []
        
        # Track edge segments to add offsets for overlapping paths
        segment_usage = {}  # ((x1,y1), (x2,y2)) -> count
        
        for edge, mapping_info in noc_paths.items():
            # Only visualize TensorEdge (not instruction edges)
            if not isinstance(edge, TensorEdge):
                continue
            
            source_node = mapping_info[0]
            dest_node = mapping_info[1]
            
            # Get color for this edge
            color = colors[tensor_edge_idx % len(colors)]
            tensor_edge_idx += 1
            
            # Get the full path by looking up router entries
            if edge in ImcflowDeviceConfig().TensorEdgetoInfo:
                edge_info = ImcflowDeviceConfig().TensorEdgetoInfo[edge]
                if edge_info.policy_info:
                    # Extract path from router entries
                    path_nodes = [entry.router_id for entry in edge_info.policy_info]
                    
                    # Convert path to coordinates and draw
                    path_coords = []
                    for node_id in path_nodes:
                        coord = NodeID.to_coord(node_id)
                        row, col = coord
                        x = col * (node_size + spacing) + spacing + node_size/2
                        y = (rows - 1 - row) * (node_size + spacing) + spacing + node_size/2
                        path_coords.append((x, y))
                    
                    # Draw the path with offsets to avoid overlap
                    if len(path_coords) > 1:
                        offset_coords = []
                        for i, (x, y) in enumerate(path_coords):
                            if i > 0:
                                # Calculate offset based on segment usage
                                prev_pt = path_coords[i-1]
                                curr_pt = (x, y)
                                segment = (prev_pt, curr_pt)
                                segment_rev = (curr_pt, prev_pt)
                                
                                # Count usage (consider both directions as same segment)
                                if segment in segment_usage:
                                    offset_idx = segment_usage[segment]
                                    segment_usage[segment] += 1
                                elif segment_rev in segment_usage:
                                    offset_idx = segment_usage[segment_rev]
                                    segment_usage[segment_rev] += 1
                                else:
                                    offset_idx = 0
                                    segment_usage[segment] = 1
                                
                                # Apply perpendicular offset
                                dx = curr_pt[0] - prev_pt[0]
                                dy = curr_pt[1] - prev_pt[1]
                                length = (dx**2 + dy**2)**0.5
                                if length > 0:
                                    # Perpendicular direction
                                    perp_x = -dy / length
                                    perp_y = dx / length
                                    # Offset amount (alternate positive/negative)
                                    offset_amount = 0.08 * (offset_idx + 1) * (1 if offset_idx % 2 == 0 else -1)
                                    x_offset = x + perp_x * offset_amount
                                    y_offset = y + perp_y * offset_amount
                                    offset_coords.append((x_offset, y_offset))
                                else:
                                    offset_coords.append((x, y))
                            else:
                                offset_coords.append((x, y))
                        
                        xs, ys = zip(*offset_coords)
                        line = ax.plot(xs, ys, color=color, linewidth=2.5, alpha=0.8, 
                                      marker='o', markersize=5, markeredgecolor='white', 
                                      markeredgewidth=0.5, zorder=10)
                        
                        # Add arrow at the end
                        if len(offset_coords) >= 2:
                            dx = offset_coords[-1][0] - offset_coords[-2][0]
                            dy = offset_coords[-1][1] - offset_coords[-2][1]
                            length = (dx**2 + dy**2)**0.5
                            if length > 0:
                                ax.arrow(offset_coords[-2][0], offset_coords[-2][1], dx*0.6, dy*0.6,
                                       head_width=0.2, head_length=0.15, fc=color, ec=color, 
                                       alpha=0.9, linewidth=1.5, zorder=11)
                        
                        # Create legend entry with NodeID and CustomID information
                        src_node_name = source_node.name
                        dst_node_name = dest_node.name
                        tensor_type = edge.src_id.tensor_type
                        
                        # Get custom IDs from the tensor edge
                        src_custom_id = edge.src_id.graph_node_id
                        dst_custom_id = edge.dst_id.graph_node_id
                        
                        # Format custom IDs (handle tuples for composite functions)
                        src_id_str = f"{src_custom_id[1]}" if isinstance(src_custom_id, tuple) else f"{src_custom_id}"
                        dst_id_str = f"{dst_custom_id[1]}" if isinstance(dst_custom_id, tuple) else f"{dst_custom_id}"
                        
                        tensor_label = f"{src_node_name} → {dst_node_name} | ID:{src_id_str}→{dst_id_str} ({tensor_type})"
                        if edge.split_idx is not None:
                            tensor_label += f"[{edge.split_idx}]"
                        legend_entries.append((line[0], tensor_label))
            else:
                # Fallback: draw direct line from source to dest
                src_coord = NodeID.to_coord(source_node)
                dst_coord = NodeID.to_coord(dest_node)
                
                src_x = src_coord[1] * (node_size + spacing) + spacing + node_size/2
                src_y = (rows - 1 - src_coord[0]) * (node_size + spacing) + spacing + node_size/2
                dst_x = dst_coord[1] * (node_size + spacing) + spacing + node_size/2
                dst_y = (rows - 1 - dst_coord[0]) * (node_size + spacing) + spacing + node_size/2
                
                line = ax.plot([src_x, dst_x], [src_y, dst_y], 
                             color=color, linewidth=2.5, alpha=0.8, marker='o', 
                             markersize=5, markeredgecolor='white', markeredgewidth=0.5, zorder=10)
                
                # Add arrow
                dx = dst_x - src_x
                dy = dst_y - src_y
                length = (dx**2 + dy**2)**0.5
                if length > 0:
                    ax.arrow(src_x, src_y, dx*0.6, dy*0.6,
                           head_width=0.2, head_length=0.15, fc=color, ec=color, 
                           alpha=0.9, linewidth=1.5, zorder=11)
                
                # Create legend entry with NodeID and CustomID information
                src_node_name = source_node.name
                dst_node_name = dest_node.name
                tensor_type = edge.src_id.tensor_type
                
                # Get custom IDs from the tensor edge
                src_custom_id = edge.src_id.graph_node_id
                dst_custom_id = edge.dst_id.graph_node_id
                
                # Format custom IDs (handle tuples for composite functions)
                src_id_str = f"{src_custom_id[1]}" if isinstance(src_custom_id, tuple) else f"{src_custom_id}"
                dst_id_str = f"{dst_custom_id[1]}" if isinstance(dst_custom_id, tuple) else f"{dst_custom_id}"
                
                tensor_label = f"{src_node_name} → {dst_node_name} | ID:{src_id_str}→{dst_id_str} ({tensor_type})"
                if edge.split_idx is not None:
                    tensor_label += f"[{edge.split_idx}]"
                legend_entries.append((line[0], tensor_label))
        
        # Add legend if there are paths
        if legend_entries:
            lines, labels = zip(*legend_entries)
            ax.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.02, 1), 
                     fontsize=7, framealpha=0.95, edgecolor='gray', fancybox=True)
    
    def _generate_colors(self, n):
        """
        Generate n distinct colors for visualization.
        
        Parameters
        ----------
        n : int
            Number of colors to generate
            
        Returns
        -------
        list of color strings
        """
        if n == 0:
            return []
        
        # Use a colormap to generate distinct colors
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        if n <= 10:
            # Use tab10 for small number of colors
            cmap = cm.get_cmap('tab10')
            return [cmap(i) for i in range(n)]
        elif n <= 20:
            # Use tab20 for medium number
            cmap = cm.get_cmap('tab20')
            return [cmap(i) for i in range(n)]
        else:
            # Use hsv for large number
            cmap = cm.get_cmap('hsv')
            return [cmap(i/n) for i in range(n)]

def generateNoCVisualizations(mod, output_dir="noc_visualizations"):
    """
    Generate NoC path visualizations for all imcflow functions in the module.
    
    This function should be called after PolicyTableGenerator has run and
    populated ImcflowDeviceConfig with NoC paths and tensor edge information.
    
    For each imcflow function, creates:
    - A subdirectory named after the function
    - Separate images for each tensor type (odata.png, weight.png, bias.png, etc.)
    - An overview image showing all tensor types together (00_overview_all_types.png)
    
    Parameters
    ----------
    mod : tvm.IRModule
        The module containing imcflow functions
    output_dir : str, optional
        Base directory to save visualization images (default: "noc_visualizations")
    
    Output Structure
    ----------------
    noc_visualizations/
        function_name_1/
            00_overview_all_types.png
            odata.png
            weight.png
            bias.png
            ...
        function_name_2/
            00_overview_all_types.png
            odata.png
            ...
    
    Example
    -------
    >>> # After running PolicyTableGenerator
    >>> generateNoCVisualizations(mod, "my_visualizations")
    """
    visualizer = TensorPathVisualizer(output_dir=output_dir)
    visualizer.visualize_all_functions(mod)
    debug_print(f"NoC visualizations saved to: {output_dir}")
