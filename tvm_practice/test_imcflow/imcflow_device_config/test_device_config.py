from tvm.contrib.imcflow import TensorID, TensorEdge, MultiCastTensorEdge, RouterEntry, TensorEdgeInfo, DataBlock, MemoryRegion, MemoryLayout, ImcflowDeviceConfig

def test_imcflow_classes():
    # Test TensorID
    tensor_id1 = TensorID(1, "idata")
    tensor_id2 = TensorID(2, "odata")
    print(tensor_id1)
    print(tensor_id2)

    # Test TensorEdge
    tensor_edge = TensorEdge(tensor_id1, tensor_id2, split_idx=0)
    print(tensor_edge)

    # Test MultiCastTensorEdge
    multi_cast_edge = MultiCastTensorEdge(tensor_id1, [tensor_id2], [None])
    print(multi_cast_edge)

    # Test RouterEntry
    router_entry = RouterEntry(0, 100, {"key": "value"})
    print(router_entry)

    # Test TensorEdgeInfo
    data_block = DataBlock("block1", 1024)
    data_block.set_offset(100)
    data_block.set_base_address(200)
    edge_info = TensorEdgeInfo([router_entry], data_block, fifo_id=1)
    print(edge_info)

    new_router_entry = RouterEntry(1, 101, {"another_key": "another_value"})
    edge_info.append_policy_info(new_router_entry)
    print(edge_info)

    # Test DataBlock
    data_block2 = DataBlock("block2", 2048)
    print(data_block2)

    # Test MemoryRegion
    mem_region = MemoryRegion("region1", 4096)
    mem_region.allocate(data_block)
    mem_region.allocate(data_block2)
    print(mem_region)

    # Test MemoryLayout
    region2 = MemoryRegion("region2", 8192)
    layout = MemoryLayout(mem_region, region2)
    print(layout)

    # Test ImcflowDeviceConfig
    config = ImcflowDeviceConfig()
    print(config.mem_layout)
    print(config.is_supported_kernel(3, 3))
    print(config.is_supported_kernel(2, 2))

# Run the test
test_imcflow_classes()
