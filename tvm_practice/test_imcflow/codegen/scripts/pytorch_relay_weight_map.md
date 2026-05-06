# PyTorch ↔ Relay weight 이름 매핑 (`PsumResNet8_cifar`)

`tvm_practice/models/resnet8_subset_models.py`의 relay 그래프와
`CIM/.../model.txt`의 PyTorch `nn.Module` (PsumResNet8_cifar) 사이의 conv
1:1 매핑. training 코드에서 PyTorch conv 인스턴스를 `psum_imcu_column_map.npz`
의 `conv/<relay_name>/...` 테이블과 연결할 때 사용한다.

## 매핑 표

| PyTorch dotted name | `TPsumQConv.layer_idx` | shape (OC×IC) | kernel | relay weight |
|---|---|---|---|---|
| `conv1.1`               | (none, FP conv) | 3→16    | 3×3 | `weight1`   |
| `layer1.0.conv1`        | 0  | 16→16   | 3×3 | `weight2_1` |
| `layer1.0.conv2`        | 1  | 16→16   | 3×3 | `weight2_2` |
| `layer2.0.conv1`        | 2  | 16→32   | 3×3 | `weight3_1` |
| `layer2.0.conv2`        | 3  | 32→32   | 3×3 | `weight3_2` |
| `layer2.0.downsample.1` | 4  | 16→32   | 1×1 | `weight3_0` |
| `layer3.0.conv1`        | 5  | 32→64   | 3×3 | `weight4_1` |
| `layer3.0.conv2`        | 6  | 64→64   | 3×3 | `weight4_2` |
| `layer3.0.downsample.1` | 7  | 32→64   | 1×1 | `weight4_0` |

총 9개 conv (FP `conv1.1` + 8 `TPsumQConv`).

## 제공 API

`tvm_practice/models/resnet8_subset_models.py`에 module-level로 expose됨:

| 이름 | 타입 | 의미 |
|---|---|---|
| `PYTORCH_TO_RELAY_WEIGHT_NAME` | `Dict[str, str]` | dotted name → relay name |
| `RELAY_TO_PYTORCH_WEIGHT_NAME` | `Dict[str, str]` | 역방향 |
| `LAYER_IDX_TO_RELAY_WEIGHT_NAME` | `Dict[int, str]` | `TPsumQConv.layer_idx` → relay name (FP `conv1.1`은 layer_idx 없으므로 빠짐) |
| `relay_weight_name_for_pytorch_module(name)` | callable | conv가 아니면 `None` |
| `relay_weight_name_for_layer_idx(idx)` | callable | 없으면 `None` |

## 사용 시나리오

### 1. training의 forward 훅에서 IMCE/column 조회

`TPsumQConv` 인스턴스마다 forward hook을 걸어 그 layer가 어느 IMCE에서 도는지
확인할 때.

```python
import numpy as np
from tvm_practice.models.resnet8_subset_models import (
    relay_weight_name_for_layer_idx,
)

psum = np.load("eval_dir/<model>_evl.linux/psum_imcu_column_map.npz",
               allow_pickle=True)

def hook(module, inputs, output):
    relay_name = relay_weight_name_for_layer_idx(module.layer_idx)
    imce = psum[f"conv/{relay_name}/imce_linear"]   # [OC, IC]
    col  = psum[f"conv/{relay_name}/column"]        # [OC, IC]
    # imce.shape == (module.out_channels, module.in_channels)
    # 여기서 logging / 분석 / IMCU별 통계 등...

for m in model.modules():
    if hasattr(m, "layer_idx"):
        m.register_forward_hook(hook)
```

### 2. `named_modules()` 기반 매핑

forward hook 대신 한 번에 매핑 테이블만 만들고 싶을 때.

```python
from tvm_practice.models.resnet8_subset_models import (
    relay_weight_name_for_pytorch_module,
)

mapping = {}
for name, module in model.named_modules():
    relay_name = relay_weight_name_for_pytorch_module(name)
    if relay_name is None:
        continue   # BatchNorm, ReLU, LSQ 등 conv가 아닌 모듈
    mapping[name] = {
        "module":     module,
        "relay_name": relay_name,
        "imce":       psum[f"conv/{relay_name}/imce_linear"],
        "column":     psum[f"conv/{relay_name}/column"],
    }
```

### 3. relay → PyTorch 역방향

dump를 보다가 어떤 relay weight가 어떤 PyTorch conv인지 알고 싶을 때.

```python
from tvm_practice.models.resnet8_subset_models import RELAY_TO_PYTORCH_WEIGHT_NAME

for relay_name in sorted({k.split("/")[1] for k in psum.files
                          if k.startswith("conv/")}):
    print(relay_name, "->", RELAY_TO_PYTORCH_WEIGHT_NAME.get(relay_name, "<unknown>"))
```

### 4. `[N, OC, IC, OH, OW]`로 broadcast해서 forward 결과와 직접 비교

mapping이 N/OH/OW에 invariant이므로 broadcast로 5-D 텐서 만들 수 있다.

```python
relay_name = relay_weight_name_for_pytorch_module("layer3.0.conv2")  # 'weight4_2'
imce = psum[f"conv/{relay_name}/imce_linear"]   # [64, 64]

# 활성화 텐서 [N, OC=64, OH, OW] 와 같은 (oc, ic) 그리드를 만들고 싶다면:
N, OH, OW = 1, 8, 8
imce_5d = np.broadcast_to(
    imce[None, :, :, None, None], (N, 64, 64, OH, OW))
# psum[n, oc, ic, oh, ow]가 imce_5d[n, oc, ic, oh, ow]번 IMCE에서 계산됨
```

## 주의사항

1. **`conv1.1`은 IMCFlow에 매핑되지 않는다.** floating-point CPU conv이므로
   `psum_imcu_column_map.npz`에 `conv/weight1/...` 그룹이 없을 수도 있다
   (relay 그래프에서 `nn.imcflow_qconv`로 변환되는 conv만 dump 대상).
2. **다른 모델 변형에는 적용되지 않는다.** 위 매핑은 `PsumResNet8_cifar`
   기준. `resnet8_subset06_pretrained_orig` 등 다른 변형이 추가되면
   별도의 매핑 dict를 정의해야 한다.
3. **PyTorch `state_dict` 키가 wrapper로 인해 다를 수 있다.** training 측
   checkpoint가 `block_int16.conv1.weight` 같은 wrapper prefix를 쓰는 경우가
   있다 (`getModel_from_pretrained_weight`의 weight loader 참조). 위 매핑은
   `nn.Module.named_modules()` 기준이므로, state_dict 키와는 다를 수 있음에
   유의.
4. **`layer_idx` 기반 lookup은 TPsumQConv 전용.** FP `conv1.1`은 `layer_idx`
   attribute가 없으므로 `LAYER_IDX_TO_RELAY_WEIGHT_NAME`에서 빠져 있다.
