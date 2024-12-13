import tvm.testing
from tvm.contrib.imcflow import TensorID, TensorEdge, MultiCastTensorEdge, RouterEntry, TensorEdgeInfo, DataBlock, MemoryRegion, MemoryLayout, ImcflowDeviceConfig

def test_tensor_id():
    tensor_id1 = TensorID(1, "idata")
    tensor_id2 = TensorID(2, "odata")
    assert str(tensor_id1) == "TensorID(1, idata)"
    assert str(tensor_id2) == "TensorID(2, odata)"

def test_tensor_edge():
    tensor_id1 = TensorID(1, "idata")
    tensor_id2 = TensorID(2, "odata")
    tensor_edge = TensorEdge(tensor_id1, tensor_id2, split_idx=0)
    assert str(tensor_edge) == "TensorEdge(TensorID(1, idata), TensorID(2, odata), 0)"

def test_multicast_tensor_edge():
    tensor_id1 = TensorID(1, "idata")
    tensor_id2 = TensorID(2, "odata")
    tensor_id3 = TensorID(3, "weight")
    multi_cast_edge = MultiCastTensorEdge(tensor_id1, [tensor_id2, tensor_id3], [None, 1])
    assert str(multi_cast_edge) == "MultiCastTensorEdge(TensorID(1, idata), [TensorID(2, odata), TensorID(3, weight)], [None, 1])"

def test_router_entry():
    router_entry = RouterEntry(0, 100, {"Local": True, "North": None, "South": None, "East": None, "West": None})
    assert str(router_entry) == "RouterEntry(0, 100, {'Local': True, 'North': None, 'South': None, 'East': None, 'West': None})"

def test_tensor_edge_info():
    mem_info = DataBlock("block1", 1024)
    mem_info.set_offset(100)
    mem_info.set_base_address(200)
    router_entry = RouterEntry(0, 100, {"Local": True, "North": None, "South": None, "East": None, "West": None})
    edge_info = TensorEdgeInfo([router_entry], mem_info, fifo_id=1)
    assert str(edge_info) == "TensorEdgeInfo([RouterEntry(0, 100, {'Local': True, 'North': None, 'South': None, 'East': None, 'West': None})], DataBlock(block1, 1024, 200), 1)"

    new_router_entry = RouterEntry(1, 101, {"Local": None, "North": None, "South": None, "East": None, "West": None})
    edge_info.append_policy_info(new_router_entry)
    assert len(edge_info.policy_info) == 2

def test_data_block():
    data_block = DataBlock("block1", 1024)
    data_block.set_offset(100)
    data_block.set_base_address(200)
    assert str(data_block) == "DataBlock(block1, 1024, 200)"

def test_memory_region():
    mem_region = MemoryRegion("region1", 4096)
    data_block = DataBlock("block1", 1024)
    data_block2 = DataBlock("block2", 2048)
    mem_region.allocate(data_block)
    mem_region.allocate(data_block2)
    assert str(mem_region).startswith("MemoryRegion(region1, 4096, -1, blocks=[")

def test_memory_layout():
    region1 = MemoryRegion("region1", 4096)
    region2 = MemoryRegion("region2", 8192)
    layout = MemoryLayout(region1, region2)
    assert "region1" in str(layout)
    assert "region2" in str(layout)

def test_imcflow_device_config_singleton():
    config1 = ImcflowDeviceConfig()
    config2 = ImcflowDeviceConfig()
    assert config1 is config2  # Ensure singleton behavior

def test_imcflow_device_config_methods():
    config = ImcflowDeviceConfig()

    # Test add_hw_node and get_hw_node
    config.add_hw_node(1, 100)
    config.add_hw_node(2, 200)
    assert config.get_hw_node(1) == 100
    assert config.get_hw_node(2) == 200

    # Test add_tensor_edge and get_tensor_edge
    tensor_id = TensorID(1, "idata")
    tensor_edge = TensorEdge(tensor_id, TensorID(2, "odata"))
    config.add_tensor_edge(tensor_id, tensor_edge)
    assert config.get_tensor_edge(tensor_id) == tensor_edge

    # Test add_tensor_edge_info and get_tensor_edge_info
    data_block = DataBlock("block1", 1024)
    tensor_edge_info = TensorEdgeInfo([], data_block, fifo_id=1)
    config.add_tensor_edge_info(tensor_edge, tensor_edge_info)
    assert config.get_tensor_edge_info(tensor_edge) == tensor_edge_info

if __name__ == "__main__":
    tvm.testing.main()