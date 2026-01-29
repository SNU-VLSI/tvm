# Joint PnR ILP Code Flow Review

## 1. Overview

Joint PnR (Place and Route) ILP은 그래프 노드를 IMCE/INODE에 배치하고, 데이터 흐름 경로를 동시에 최적화하는 시스템입니다.

```
Driver (imcflow_compiler_driver.py)
    │
    ├── Step 16a: constructTensorEdgeList()
    │
    ├── Step 17: run_joint_pnr_and_update_config()
    │       │
    │       ├── GraphExtractor.extract()  ← 노드 추출
    │       │       ├── _visit() ← relay 표현식 순회
    │       │       └── _extract_commodities_from_tensor_edge_list() ← TensorEdgeList 사용
    │       │
    │       ├── JointPnRILP.solve() ← ILP 최적화
    │       │
    │       └── update_hw_node_map() ← HWNodeMap 업데이트
    │
    ├── Step 19: construct_noc_paths_from_pnr_results()
    │
    └── Step 20+: MemoryAllocator 등
```

---

## 2. ID 시스템

### 2.1 Node ID 종류

| ID 종류 | 설명 | 예시 |
|---------|------|------|
| `custom_id` | `annotateCustomId()`에서 할당된 고유 ID | `40`, `41`, `42` |
| `graph_node_id` | TensorEdgeList에서 사용하는 ID | 일반: `40`, Composite: `(41, 33)` |
| `outer_id` | Composite의 외부 ID (custom_id) | `(41, 33)` → `41` |
| `inner_id` | Composite의 내부 ID | `(41, 33)` → `33` |

### 2.2 Composite 노드 구조

```
Composite Call (custom_id=41)
    │
    └── Inner Function Body
            └── Inner Call (inner_id=33)
```

**TensorEdgeList에서의 표현:**
```python
# Composite 내부에서 외부로 나가는 edge
TensorEdge(
    src_id=TensorID(graph_node_id=(41, 33), tensor_type='odata'),
    dst_id=TensorID(graph_node_id=43, tensor_type='func_out0'),
    split_idx=0
)
```

### 2.3 ID 변환 함수

```python
# transform.py:160-170
def getInnerNodeID(node):
    """(41, 33) → 33, 40 → 40"""
    if isinstance(node, tuple):
        return node[1]
    return node

def getOuterNodeID(node):
    """(41, 33) → 41, 40 → 40"""
    if isinstance(node, tuple):
        return node[0]
    return node
```

---

## 3. HWNodeMap 구조

### 3.1 Joint PnR 이후 HWNodeMap 내용

```python
HWNodeMap = {
    # IMCE 매핑 (graph_node_id → NodeID)
    40: NodeID("imce_0_1"),      # 일반 노드
    41: NodeID("imce_1_2"),      # Composite 노드 (outer_id로 저장)
    42: NodeID("imce_2_0"),

    # INODE 매핑 (var/funcout node_id → NodeID)
    38: NodeID("inode_0_0"),     # input var
    43: NodeID("inode_3_0"),     # func_out
}
```

**중요:** Joint PnR은 **outer_id (custom_id)** 를 키로 사용합니다.

### 3.2 update_hw_node_map() 코드

```python
# joint_pnr_ilp.py:1563-1594
def update_hw_node_map(results: Dict[str, JointPnRResult], hw_node_map: Dict):
    for func_name, result in results.items():
        # IMCE 매핑
        for graph_node_id, coord in result.mapping.items():
            node_id = NodeID.from_imce_coord(coord.row, coord.col - 1)
            hw_node_map[graph_node_id] = node_id  # ← graph_node_id는 outer_id

        # INODE 매핑 (input vars)
        if result.var_to_inode:
            for var_node_id, coord in result.var_to_inode.items():
                node_id = NodeID.from_inode_coord(coord.row)
                hw_node_map[var_node_id] = node_id

        # INODE 매핑 (funcout)
        if result.funcout_to_inode:
            for funcout_node_id, coord in result.funcout_to_inode.items():
                node_id = NodeID.from_inode_coord(coord.row)
                hw_node_map[funcout_node_id] = node_id
```

---

## 4. HWNodeMap Lookup 문제

### 4.1 문제 상황

**TensorEdgeList의 graph_node_id:**
- Composite: `(41, 33)` (tuple)
- 일반: `40` (int)

**HWNodeMap의 키:**
- Joint PnR: `41` (outer_id만 저장)

### 4.2 Lookup 방식 비교

| 함수 | 기존 코드 | 변경 후 |
|------|-----------|---------|
| `is_inode_in_edge()` | `getInnerNodeID()` | `getOuterNodeID()` ✅ |
| `allocate()` | `getInnerNodeID()` | `getOuterNodeID()` ✅ |

### 4.3 변경된 코드

```python
# transform.py:4444-4472 (is_inode_in_edge)
def is_inode_in_edge(self, edge):
    # 변경 전: getInnerNodeID(edge.dst_id.graph_node_id)
    # 변경 후: getOuterNodeID(edge.dst_id.graph_node_id)
    dst_key = getOuterNodeID(edge.dst_id.graph_node_id)
    if dst_key in self.hwnodemap:
        dst_hw_node_id = self.hwnodemap[dst_key]
        ...

# transform.py:5117-5119 (allocate)
_, inode_tensorid = self.is_inode_in_edge(edge)
# 변경 전: getInnerNodeID(inode_tensorid.graph_node_id)
# 변경 후: getOuterNodeID(inode_tensorid.graph_node_id)
hw_node_id = self.hwnodemap[getOuterNodeID(inode_tensorid.graph_node_id)]
```

---

## 5. GraphExtractor 노드 추출

### 5.1 _get_node_id() 함수

```python
# joint_pnr_ilp.py:622-639
def _get_node_id(self, expr) -> Any:
    """Get unique node ID for an expression.

    For Call/Function nodes with custom_id attr, use custom_id.
    This matches TensorEdgeList IDs.
    """
    if isinstance(expr, (relay.Call, relay.Function)):
        if hasattr(expr, 'attrs') and expr.attrs:
            try:
                custom_id = getattr(expr.attrs, "custom_id", None)
                if custom_id is not None:
                    return int(custom_id)
            except (AttributeError, TypeError):
                pass
    return hash(expr)
```

### 5.2 노드 타입

```python
class NodeType(Enum):
    CALL = "call"           # 일반 연산 노드
    SPLIT = "split"         # Split 연산
    CONCAT = "concat"       # Concat 연산
    VAR = "var"             # 입력 변수
    CONST = "const"         # 상수
    FUNC_OUT = "func_out"   # 함수 출력
```

### 5.3 INODE 할당

```python
# _visit_var(): input var → INODE 할당
def _visit_var(self, var, in_composite):
    if in_composite:
        return
    node_id = self._get_node_id(var)
    ...
    if node_id not in self.var_to_inode:
        self.var_to_inode[node_id] = next(self.var_inode_iter)

# _visit_function(): func_out → INODE 할당
def _visit_function(self, fn, in_composite, composite_node_id):
    node_id = self._get_node_id(fn)
    if not in_composite:
        ...
        if node_id not in self.funcout_to_inode:
            self.funcout_to_inode[node_id] = next(self.funcout_inode_iter)
```

---

## 6. Commodity 추출 (TensorEdgeList 기반)

### 6.1 _extract_commodities_from_tensor_edge_list()

```python
# joint_pnr_ilp.py
def _extract_commodities_from_tensor_edge_list(self, tensor_edge_list):
    for edge in tensor_edge_list:
        src_graph_id = edge.src_id.graph_node_id
        dst_graph_id = edge.dst_id.graph_node_id

        # Composite 노드는 outer_id만 사용
        src_node_id = getOuterNodeID(src_graph_id)
        dst_node_id = getOuterNodeID(dst_graph_id)

        # Composite 내부 edge는 skip
        if src_node_id == dst_node_id:
            continue

        # 노드 lookup
        src_node = self.nodes.get(src_node_id)
        dst_node = self.nodes.get(dst_node_id)

        if src_node is None or dst_node is None:
            continue

        self.add_commodity(
            src_node_id, dst_node_id,
            src_node.node_type, dst_node.node_type,
            tensor_type=edge.src_id.tensor_type,
            split_idx=edge.split_idx,
            metadata=edge
        )
```

---

## 7. NoCPaths 구성

### 7.1 construct_noc_paths_from_pnr_results()

```python
# joint_pnr_ilp.py:1866-1945
def construct_noc_paths_from_pnr_results(
    pnr_results: Dict[str, JointPnRResult],
    tensor_edge_list_dict: Dict[str, List],
) -> Dict[str, Dict]:
    """Construct NoCPaths dict from Joint PnR results."""

    for func_name, pnr_result in pnr_results.items():
        noc_paths[func_name] = {}
        tensor_edge_list = tensor_edge_list_dict.get(func_name, [])

        for tensor_edge in tensor_edge_list:
            # outer_id로 변환하여 mapping에서 조회
            src_graph_id = getOuterNodeID(tensor_edge.src_id.graph_node_id)
            dst_graph_id = getOuterNodeID(tensor_edge.dst_id.graph_node_id)

            src_coord = pnr_result.mapping.get(src_graph_id)
            dst_coord = pnr_result.mapping.get(dst_graph_id)

            # Coord → NodeID 변환
            src_hwnode = coord_to_node_id(src_coord)
            dst_hwnode = coord_to_node_id(dst_coord)

            noc_paths[func_name][tensor_edge] = (src_hwnode, dst_hwnode, split_idx)
```

---

## 8. 데이터 흐름 요약

```
1. annotateCustomId()
   └── 각 Call/Function에 custom_id 할당

2. constructTensorEdgeList()
   └── TensorEdge 생성 (graph_node_id = custom_id 또는 (outer_id, inner_id))

3. GraphExtractor.extract()
   ├── _visit(): relay 순회하여 노드 추출 (key = custom_id)
   └── _extract_commodities_from_tensor_edge_list(): TensorEdgeList로 commodity 생성

4. JointPnRILP.solve()
   └── ILP 최적화 → mapping (graph_node_id → Coord)

5. update_hw_node_map()
   └── HWNodeMap[graph_node_id] = NodeID  (key는 outer_id/custom_id)

6. construct_noc_paths_from_pnr_results()
   └── TensorEdge → (src_hwnode, dst_hwnode, split_idx)

7. MemoryAllocator.allocate()
   └── is_inode_in_edge()로 INODE edge 찾기
       └── getOuterNodeID()로 HWNodeMap lookup
```

---

## 9. 잠재적 문제점

### 9.1 Var 노드의 custom_id

**문제:** `relay.Var`는 `attrs`가 없을 수 있음

```python
# _get_node_id()에서 Var 처리
if isinstance(expr, (relay.Call, relay.Function)):  # Var는 포함 안됨!
    ...
return hash(expr)  # Var는 hash로 fallback
```

**결과:**
- GraphExtractor의 var_node_id = `hash(var)`
- TensorEdgeList의 graph_node_id = `custom_id` (다를 수 있음)

### 9.2 확인 필요 사항

1. **Var에 custom_id가 할당되는지?**
   - `annotateCustomId()`가 Var도 처리하는지 확인

2. **TensorEdgeList에서 Var의 graph_node_id는?**
   - Var edge의 src_id.graph_node_id 값 확인

3. **is_inode_in_edge()에서 lookup 실패 시:**
   - `inode_tensorid`가 None이 됨
   - `allocate()`에서 AttributeError 발생

---

## 10. 디버깅 제안

### 10.1 HWNodeMap 키 확인

```python
# 디버깅 코드 추가
print("=== HWNodeMap keys ===")
for k, v in hw_node_map.items():
    print(f"  {k} ({type(k).__name__}): {v}")
```

### 10.2 TensorEdgeList graph_node_id 확인

```python
# 디버깅 코드 추가
print("=== TensorEdgeList graph_node_ids ===")
for edge in tensor_edge_list:
    print(f"  src: {edge.src_id.graph_node_id}, dst: {edge.dst_id.graph_node_id}")
```

### 10.3 is_inode_in_edge() 결과 확인

```python
# allocate() 내부
for edge, mem_block in self.DataBlockDict.items():
    is_inode, inode_tensorid = self.is_inode_in_edge(edge)
    print(f"Edge: {edge}")
    print(f"  is_inode: {is_inode}, inode_tensorid: {inode_tensorid}")
    if inode_tensorid is None:
        print(f"  WARNING: inode_tensorid is None!")
        print(f"  src_key: {getOuterNodeID(edge.src_id.graph_node_id)}")
        print(f"  dst_key: {getOuterNodeID(edge.dst_id.graph_node_id)}")
        print(f"  src in hwnodemap: {getOuterNodeID(edge.src_id.graph_node_id) in self.hwnodemap}")
        print(f"  dst in hwnodemap: {getOuterNodeID(edge.dst_id.graph_node_id) in self.hwnodemap}")
```
