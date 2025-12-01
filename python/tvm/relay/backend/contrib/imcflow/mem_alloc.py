
class MemoryAllocator:
    """
    Allocate memory block to var, constant, function output.
    Target Operators:
      conv2d, bias_add, batch_norm, relu, add and fused versions
      split, concat
    
    Assumption:
      no edge from inode to inode directly
    """
    def run_(self, func, func_name, ttype_map):
      class _MemoryAllocator(tvm.relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.TensorEdgeList = ImcflowDeviceConfig().TensorEdgeList
            # self.DataBlockDict ={edge: DataBlock(edge.dst_id, None) for edge in self.TensorEdgeList}
            self.DataBlockDict ={}

            self.imce_index = ImcflowDeviceConfig.IMCE_NUM - 1
            self.inode_index = ImcflowDeviceConfig.INODE_NUM - 1

            self.id_dict = HashToCustomID()
            self.name_dict = CustomIDToName()
            self.data = CustomIDToNode()
            self.hwnodemap = ImcflowDeviceConfig().HWNodeMap

            self.func_name = None
            self.ttype_map = None

        def traverse_func(self, func, func_name, ttype_map):
            self.func_name = func_name
            self.ttype_map = ttype_map
            self.visit(func)
            self.allocate()
            return self.DataBlockDict

        def visit_function(self, fn):
          super().visit_function(fn)

          if hasattr(fn.attrs, "Compiler") and fn.attrs["Compiler"]=="imcflow":
            edges = self.find_in_edge_from_list(fn)
            for edge in edges:
              self.add_to_block_dict(edge, fn)
        
        def visit_var(self, var):
          super().visit_var(var)
          edges = self.find_out_edge_from_list(var)
          for edge in edges:
            self.add_to_block_dict(edge, var)
        
        def visit_constant(self, const):
          super().visit_constant(const)
          edges = self.find_out_edge_from_list(const)
          for edge in edges:
            self.add_to_block_dict(edge, const)
          
        def add_to_block_dict(self, edge, node):
            size = self.get_size(edge, node)
            if size > 0:
              datablock = DataBlock(edge, None)
              datablock.set_size(size)
              self.DataBlockDict[edge] = datablock
            else:
              raise ValueError("edge has zero size.")

        def visit_tuple_getitem(self, op):
          super().visit_tuple_getitem(op)

        def visit_tuple(self, op):
          super().visit_tuple(op)

        def is_inode_in_edge(self, edge):
          dst_hw_node_id = None
          src_hw_node_id = None
          is_inode = False
          inode_tensorid = None

          #dst id
          if getInnerNodeID(edge.dst_id.graph_node_id) in self.hwnodemap:
            dst_hw_node_id = self.hwnodemap[getInnerNodeID(edge.dst_id.graph_node_id)]
            if dst_hw_node_id.name.startswith("inode"):
              # determine whether inode is included in the edge and which id it is.
              is_inode = True
              inode_tensorid = edge.dst_id

          #src id
          if getInnerNodeID(edge.src_id.graph_node_id) in self.hwnodemap:
            src_hw_node_id = self.hwnodemap[getInnerNodeID(edge.src_id.graph_node_id)]
            if src_hw_node_id.name.startswith("inode"):
              # determine whether inode is included in the edge and which id it is.
              is_inode = True
              inode_tensorid = edge.src_id

          return is_inode, inode_tensorid
        
        def find_out_edge_from_list(self, call, to_only_inode=False):
          tensor_edge_list = self.TensorEdgeList
          graph_node_id = getNodeID(call)

          def matches_node_id(node_id):
            if isinstance(node_id, (int, tvm.tir.expr.IntImm)):
              return node_id == graph_node_id
            elif isinstance(node_id, tuple):
              return graph_node_id in node_id
            return False

          edges = []
          for edge in tensor_edge_list:
            if matches_node_id(getInnerNodeID(edge.src_id.graph_node_id)) and (not to_only_inode or self.is_inode_in_edge(edge)[0]):
              edges.append(edge)

          return edges

        def find_in_edge_from_list(self, call, from_only_inode=False):
          tensor_edge_list = self.TensorEdgeList
          graph_node_id = getNodeID(call)

          def matches_node_id(node_id):
            if isinstance(node_id, (int, tvm.tir.expr.IntImm)):
              return node_id == graph_node_id
            elif isinstance(node_id, tuple):
              return graph_node_id in node_id
            return False

          edges = []
          for edge in tensor_edge_list:
            if matches_node_id(getInnerNodeID(edge.dst_id.graph_node_id)) and (not from_only_inode or self.is_inode_in_edge(edge)[0]):
              edges.append(edge)

          return edges

        def allocate(self):
          """
          Two-phase memory allocation:
          Phase 1: Collect information about input/output tensors per inode
          Phase 2: Calculate tiling factor and perform actual allocation
          """
          # Phase 1: Collect information
          # {inode_name: {'input': [], 'output': [], 'weight': [], 'other': []}}
          inode_tensors = {}
          
          for edge, mem_block in self.DataBlockDict.items():
            if mem_block.size is None:
              raise ValueError("Memory size cannot be none.")

            _, inode_tensorid = self.is_inode_in_edge(edge)
            hw_node_id = self.hwnodemap[getInnerNodeID(inode_tensorid.graph_node_id)]
            inode_name = hw_node_id.name  # ex) inode_3
            
            if inode_name not in inode_tensors:
              inode_tensors[inode_name] = {
                'input': [],
                'output': [],
                'weight': [],
                'other': []
              }
            
            # Classify tensor type
            tensor_type = inode_tensorid.tensor_type
            
            if tensor_type == "weight":
              inode_tensors[inode_name]['weight'].append((edge, mem_block, inode_tensorid))
            elif tensor_type == "data" or tensor_type == "odata" or tensor_type == "func_out" or tensor_type == "var":
              # Check if this is function input or output
              src_node = self.data.get(getInnerNodeID(edge.src_id.graph_node_id))
              dst_node = self.data.get(getInnerNodeID(edge.dst_id.graph_node_id))
              
              if isinstance(src_node, relay.Var):
                # Function input
                inode_tensors[inode_name]['input'].append((edge, mem_block, inode_tensorid))
              elif isinstance(dst_node, relay.Function):
                # Function output
                inode_tensors[inode_name]['output'].append((edge, mem_block, inode_tensorid))
              else:
                # Intermediate tensor
                inode_tensors[inode_name]['other'].append((edge, mem_block, inode_tensorid))
            else:
              # Other types (bias, min, max, etc.)
              inode_tensors[inode_name]['other'].append((edge, mem_block, inode_tensorid))
          
          # Phase 2: Calculate tiling factor for this function
          tiling_factor = 1
          
          for inode_name, tensors in inode_tensors.items():
            # Calculate total size of input/output tensors for this inode
            input_output_total = 0
            
            for edge, mem_block, _ in tensors['input']:
              input_output_total += mem_block.size
            
            for edge, mem_block, _ in tensors['output']:
              input_output_total += mem_block.size
            
            # Check if tiling is needed
            if input_output_total > ImcflowDeviceConfig.INODE_DATA_MEM_SIZE:
              required_factor = math.ceil(input_output_total / ImcflowDeviceConfig.INODE_DATA_MEM_SIZE)
              tiling_factor = max(tiling_factor, required_factor)
              debug_print(f"  [{self.func_name}] {inode_name}: input/output total = {input_output_total} bytes")
              debug_print(f"    > Memory capacity = {ImcflowDeviceConfig.INODE_DATA_MEM_SIZE} bytes")
              debug_print(f"    > Required tiling factor = {required_factor}")
          
          # Store tiling factor in FunctionInfo
          func_info = ImcflowDeviceConfig().ImcflowFuncMap[self.func_name]
          func_info.tiling_factor = tiling_factor
          
          if tiling_factor > 1:
            debug_print(f"  [{self.func_name}] Tiling factor = {tiling_factor}")
          
          # Phase 3: Perform actual allocation with tiling
          for inode_name, tensors in inode_tensors.items():
            # Allocate weight tensors (no tiling, allow overlap)
            for edge, mem_block, inode_tensorid in tensors['weight']:
              ImcflowDeviceConfig().add_tensor_edge_info(edge, TensorEdgeInfo(data_block=mem_block))
              ImcflowDeviceConfig().MemLayout[self.func_name][f"{inode_name}_data"].allocate(mem_block, phase="init")
            
            # Allocate input tensors (with tiling if needed)
            for edge, mem_block, inode_tensorid in tensors['input']:
              if tiling_factor > 1:
                # Apply tiling: divide size by tiling factor
                # This represents height-wise tiling (axis=2)
                tiled_size = math.ceil(mem_block.size / tiling_factor)
                mem_block.set_size(tiled_size)
                debug_print(f"    Input tensor tiled: {mem_block.size} -> {tiled_size} bytes")
              
              ImcflowDeviceConfig().add_tensor_edge_info(edge, TensorEdgeInfo(data_block=mem_block))
              ImcflowDeviceConfig().MemLayout[self.func_name][f"{inode_name}_data"].allocate(mem_block, phase="exec")
            
            # Allocate output tensors (with tiling if needed)
            for edge, mem_block, inode_tensorid in tensors['output']:
              if tiling_factor > 1:
                # Apply tiling: divide size by tiling factor
                tiled_size = math.ceil(mem_block.size / tiling_factor)
                mem_block.set_size(tiled_size)
                debug_print(f"    Output tensor tiled: {mem_block.size} -> {tiled_size} bytes")
              
              ImcflowDeviceConfig().add_tensor_edge_info(edge, TensorEdgeInfo(data_block=mem_block))
              ImcflowDeviceConfig().MemLayout[self.func_name][f"{inode_name}_data"].allocate(mem_block, phase="exec")
            
            # Allocate other tensors (no tiling)
            for edge, mem_block, inode_tensorid in tensors['other']:
              ImcflowDeviceConfig().add_tensor_edge_info(edge, TensorEdgeInfo(data_block=mem_block))
              ImcflowDeviceConfig().MemLayout[self.func_name][f"{inode_name}_data"].allocate(mem_block, phase="init")

          return

        def get_arg_idx(self, edge, call):
          # edge is input edge to call node
          # find arg index from call by comparing edge's tensorid
          idx = None
          shape = None
          arg_dtype = None
          for i, arg in enumerate(call.args):
            # Determine the source ID based on the type of `arg`
            if isinstance(arg, TupleGetItem):
                src_id = getNodeID(arg.tuple_value)
            else:
                src_id = getNodeID(arg)

            dst_id = getNodeID(call)

            # Check if `src_id` matches the source node in `edge`
            if isinstance(edge.src_id.graph_node_id, tuple):
              if src_id in edge.src_id.graph_node_id:
                idx = i
                shape = call.type_args[idx].shape
                arg_dtype = call.type_args[idx].dtype
            else:
              if src_id == edge.src_id.graph_node_id:
                idx = i
                shape = call.type_args[idx].shape
                arg_dtype = call.type_args[idx].dtype

            # Check if `dst_id` matches the source node in `edge`
            # this is only for the case where src node is Var node, because customID of Var node in subfunction is not the same one in tensoredge.
            if isinstance(edge.dst_id.graph_node_id, tuple):
              if dst_id in edge.dst_id.graph_node_id and isinstance(arg, Var):
                idx = i
                shape = call.type_args[idx].shape
                arg_dtype = call.type_args[idx].dtype

          return idx, shape, arg_dtype

        def get_op_from_id(self, node_id):
            if isinstance(node_id, (int, tvm.tir.expr.IntImm)):
                return self.name_dict[node_id]
            elif isinstance(node_id, tuple):
                return self.name_dict[node_id[1]]
            else:
              raise ValueError("CustomIDToName does not have this node id.")

        def get_size(self, edge, call):
            size = None
            arg_shape = None
            arg_dtype = None

            if isinstance(call, Function): # output edge of function
              size = None
              arg_node = call.body
              if isinstance(arg_node, Tuple):
                # find field of current edge
                target_idx = -1
                for i, field in enumerate(arg_node.fields):
                  if isinstance(edge.src_id.graph_node_id, tuple):
                    if getNodeID(field) in edge.src_id.graph_node_id:
                      target_idx = i
                      break
                      # arg_node = field
                      # func_ret_type = call.ret_type.fields[i]
                  else:
                    if getNodeID(field) == edge.src_id.graph_node_id:
                      target_idx = i
                      break
                      # arg_node = field
                      # func_ret_type = call.ret_type.fields[i]
                assert target_idx != -1, "Cannot find target field index in function return tuple."
                arg_ttype = self.ttype_map[self.func_name][target_idx]
              else:
                arg_ttype = self.ttype_map[self.func_name]
              arg_shape = arg_ttype[0]
              arg_dtype = arg_ttype[1]
            else:
              src_op = self.get_op_from_id(edge.src_id.graph_node_id)

              #find which argument index this edge correspond to find corresponding shape by type_args.shape
              src_node = self.data[getInnerNodeID(edge.src_id.graph_node_id)]
              if isinstance(src_node, relay.Var):
                arg_ttype = self.ttype_map[src_node.name_hint]
                arg_shape, arg_dtype = arg_ttype[0], arg_ttype[1]
              elif isinstance(src_node, relay.Constant):
                arg_shape, arg_dtype = list(src_node.data.shape), str(src_node.data.dtype)
                # _, arg_shape, arg_dtype = self.get_arg_idx(edge, call)
              else:
                raise ValueError("Source node is neither Var nor Constant.")

              # if src_op == "Op(split)":
              #   # when first node of subgraph is split, memoryblock is already allocated by (src: var -> dst: split) case.
              #   arg_shape = -1
              #   raise ValueError("Split operator output edge should not be allocated here.")

            # calculate size for inode memory allocation
            # if arg_shape == -1:
            #   size = -1
            # else:
            size = math.prod(arg_shape)
            if arg_dtype == "int32" or arg_dtype == "uint32":
              size = size * 32 // 8 # dtype is int32 and unit is byte
            elif arg_dtype == "int16" or arg_dtype == "uint16":
              size = size * 16 // 8 # dtype is int16 and unit is byte
            elif arg_dtype == "int8" or arg_dtype == "uint8":
              size = size * 8 // 8 # dtype is int8 and unit is byte
            elif arg_dtype == "uint4":
              size = size * 4 // 8 # dtype is int4 and unit is byte
            else:
              #sanity check
              raise ValueError(f"Unsupported dtype {arg_dtype} in function return type.")

            if size is None:
              raise ValueError("Size cannot be none.")

            return size

      _MemoryAllocator().traverse_func(func, func_name, ttype_map)
      return func

    def run(self, mod, ttype_map):
      imcflow_func_map = ImcflowDeviceConfig().ImcflowFuncMap
      for gv, func in mod.functions.items():
        if isinstance(func, relay.Function) and hasattr(func.attrs, "Compiler") and func.attrs["Compiler"]=="imcflow":
          func_info = imcflow_func_map[gv.name_hint]
          self.run_(func_info.func_node, gv.name_hint, ttype_map[gv.name_hint])


def constructDataBlockDict(mod):
  imcflow_func_map = ImcflowDeviceConfig().ImcflowFuncMap
  for func_name_var, func in mod.functions.items():
    if func_name_var.name_hint == "main": continue
    elif isinstance(func, relay.Function) and hasattr(func.attrs, "Compiler") and func.attrs["Compiler"]=="imcflow" :
      target_func = imcflow_func_map[func_name_var.name_hint]
      input_node_ids = [getNodeID(n) for n in getInputNodesOfFunc(target_func.func_node)]
      output_node_id = getNodeID(target_func.func_node)
      const_node_ids = [getNodeID(n) for n in getConstNodesOfFunc(target_func.func_node)]
      ImcflowDeviceConfig().get_data_block_dict(target_func, func_name_var.name_hint, input_node_ids, output_node_id, const_node_ids)