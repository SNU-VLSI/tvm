# `psum_imcu_column_map.npz` 형식

`single_qconv` (v1) 또는 `--driver-v2` (v2) 모드로 컴파일한 결과의
**psum tensor → IMCU/column 매핑**을 담은 npz 파일.

qconv input `[N, IC, IH, IW]`, output `[N, OC, OH, OW]` 일 때 partial sum은
개념적으로 `[N, OC, IC, OH, OW]`. 매핑은 `N`, `OH`, `OW`에 대해 invariant
이므로 npz는 `(oc, ic)` projection만 저장한다.

## 생성 방법

### 1. 컴파일 시 자동 dump (env-gated)
```bash
IMCFLOW_DUMP_PSUM_MAP=1 python main.py --model <name> --single-qconv  # v1
IMCFLOW_DUMP_PSUM_MAP=1 python main.py --model <name> --driver-v2     # v2
```
출력 위치: `{eval_dir}/psum_imcu_column_map.npz`

### 2. 컴파일 끝난 후 후처리로 dump (CLI)
```bash
python scripts/dump_psum_mapping.py --eval-dir eval_dir/<model>_evl.linux/
```

### 3. 코드에서 직접 호출 (util import)
```python
from tvm.relay.backend.contrib.imcflow.psum_mapping import (
    dump_psum_mapping,           # online: mod 메모리에 있을 때
    dump_psum_mapping_offline,   # offline: eval_dir만 있을 때
)
dump_psum_mapping(mod, "/path/out.npz")
dump_psum_mapping_offline(eval_dir, "/path/out.npz")
```

## NPZ 키 — 세 가지 그룹

`atomic qconv`는 `split_conv_to_atomic` 후 한 imcflow function 안에 들어 있는
실제 IMCU 단위 conv. 원본 큰 conv 하나가 OC/IC 차원에서 여러 atomic qconv로
분할될 수 있다.

레코드 수 `N` = 모듈 안 atomic qconv 총 개수.
원본 conv 수 `M` ≤ `N` (split된 경우 한 원본이 여러 atomic을 만든다).

### A. 평면 배열 (length = `N`)

| 키 | dtype | 의미 |
|---|---|---|
| `func_names` | `object[N]` (str) | atomic qconv를 감싸는 imcflow function 이름 (`tvmgen_default_imcflow_main_<idx>`) |
| `qconv_op` | `object[N]` (str) | `"nn.imcflow_qconv"` 또는 `"nn.imcflow_qdwconv"` |
| `is_depthwise` | `bool[N]` | depthwise 여부 |
| `custom_id` | `int32[N]` | qconv의 custom_id (`DevConfig.HWNodeMap` 키) |
| `imce_linear_id` | `int32[N]` | 칩 내 IMCE 0..15 (= `imce_row * IMCE_W_NUM + imce_col_in_imce`) |
| `imce_row` | `int32[N]` | IMCE row 좌표 0..3 |
| `imce_col_in_imce` | `int32[N]` | row 내 IMCE column 0..3 (PnR Coord의 col은 1..4지만 여기선 0-base) |
| `oc_size` | `int32[N]` | atomic qconv가 처리하는 OC 크기 (block 크기 또는 마지막 partial) |
| `ic_size` | `int32[N]` | atomic qconv가 처리하는 IC 크기 |
| `kernel_h` / `kernel_w` | `int32[N]` | KH, KW |
| `oc_id` | `int32[N]` | 원본 conv 안에서 OC 차원 split block index |
| `ic_id` | `int32[N]` | 원본 conv 안에서 IC 차원 split block index |
| `oc_block` | `int32[N]` | OC split block size (보통 `effective_oc=64`; 마지막 block은 더 작을 수 있음) |
| `ic_block` | `int32[N]` | IC split block size (3x3 conv면 `floor(256/9)=28`, 1x1 conv면 256) |
| `total_oc` | `int32[N]` | 원본 conv의 전체 OC |
| `total_ic` | `int32[N]` | 원본 conv의 전체 IC |
| `orig_conv` | `object[N]` (str) | 원본 conv weight Var의 이름 (모델 정의에서 부여한 이름; 못 찾으면 synthetic `conv_<id>`) |
| `orig_conv_id` | `int32[N]` | `split_conv_to_atomic` 내부 카운터 (모듈 순회 순서, 0-base) |
| `weight_var` | `object[N]` (str) | atomic qconv 자신의 weight 식별자 (post-bind면 `<const>`) |
| `metadata` | `object[1]` (str) | `repr({imce_h_num, imce_w_num, imce_num, single_qconv})` |

### B. 함수별 ragged 배열 (length 가변)

| 키 패턴 | shape | 의미 |
|---|---|---|
| `valid_cols/<func_name>` | `int32[oc_size]` | 각 oc-local index `oc_local ∈ [0, oc_size)` 가 매핑되는 IMCU column index `0..63` |

- column-disable이 없을 때: `[0, 1, ..., oc_size-1]` (항등 매핑).
- column-disable 있을 때 (`--driver-v2 --column-disable-config ...`): disabled 빠진 sorted set의 앞쪽 `oc_size`개. 즉 `valid_cols/<f>[oc_local]`이 곧 IMCU 안에서 그 weight column이 적재되는 실제 column index.

### C. 원본 conv별 2-D 매핑 (M개 그룹)

각 원본 conv `<orig>`마다 다음 4개 배열이 생긴다. shape는 모두 `[total_oc, total_ic]`.

| 키 | dtype | 의미 |
|---|---|---|
| `conv/<orig>/imce_linear` | `int32[OC, IC]` | psum `[oc, ic]` 가 계산되는 IMCE 0..15 |
| `conv/<orig>/imce_row` | `int32[OC, IC]` | 그 IMCE의 row |
| `conv/<orig>/imce_col` | `int32[OC, IC]` | 그 IMCE의 row 내 column index 0..3 |
| `conv/<orig>/column` | `int32[OC, IC]` | IMCU 안에서 그 weight가 적재된 column 0..63 |

값 `-1`은 "이 `(oc, ic)` 슬롯에 atomic qconv가 없음"을 의미한다. 일반
(non-depthwise) conv는 `split_conv_to_atomic`이 모든 `(oc_id, ic_id)`
조합을 다 만들고 마지막 partial block까지 채우므로 `-1`이 남지 않는다.
`-1`이 나오는 실제 케이스:

1. **Depthwise conv** — `total_ic = total_oc = groups` 인데 각 atomic qconv는
   자기 OC slice의 `IC=1`만 처리한다. 따라서 `imce_linear[oc, ic]`는 `ic`가
   그 `oc`가 속한 block 범위 안일 때만 채워지고 (diagonal block), off-diagonal
   영역은 `-1`이다. 의미: "이 `(oc, ic)` 조합은 depthwise 정의상 0".
2. **`column`만 `-1`** — `valid_cols`가 `oc_size`보다 짧을 때. column-disable
   설정에 모순이 없으면 일어나지 않는다.

> **그룹이 만들어지지 않는 경우**: 어떤 conv에 대해 `oc_id`/`ic_id` 또는
> `total_oc`/`total_ic` 메타데이터가 누락된 경우 (일부 빌드에서 `<const>` 만
> 들어왔을 때) 해당 conv의 `conv/<orig>/*` 그룹은 생략된다. 평면 배열은
> 항상 정확하다.

## 좌표계 한 줄 정리

| 좌표 | 무엇 | 범위 |
|---|---|---|
| `imce_row`, `imce_col_in_imce`, `imce_linear_id` | 칩 안에서 **어느 IMCE 코어**인지 | row 0..3, col 0..3 / linear 0..15 |
| `column` (또는 `valid_cols/<f>[oc_local]`) | 그 IMCE 안에서 **어느 IMCU column**인지 | 0..63 |

## 사용 예시

### 한 conv의 psum tensor 매핑 보기
```python
import numpy as np
d = np.load("psum_imcu_column_map.npz", allow_pickle=True)

# 어떤 원본 conv가 있는지
groups = sorted({k.split("/")[1] for k in d.files if k.startswith("conv/")})
print(groups)  # ['weight2_1', 'weight2_2', 'weight3_0', ...]

orig = "weight4_2"
imce = d[f"conv/{orig}/imce_linear"]   # [OC, IC]
col  = d[f"conv/{orig}/column"]        # [OC, IC]
print(imce.shape, "distinct imce =", sorted(set(imce.flatten().tolist())))
```

### `[N, OC, IC, OH, OW]`로 broadcast
```python
imce = d["conv/weight4_2/imce_linear"]      # [64, 64]
col  = d["conv/weight4_2/column"]            # [64, 64]
N, OH, OW = 1, 32, 32
imce_5d = np.broadcast_to(imce[None, :, :, None, None], (N, 64, 64, OH, OW))
col_5d  = np.broadcast_to(col[None,  :, :, None, None], (N, 64, 64, OH, OW))
# psum[n, oc, ic, oh, ow] 는 imce_5d[...]의 IMCE에서, col_5d[...]의 column에서 계산
```

### 함수 단위로 lookup
```python
n = len(d["func_names"])
for i in range(n):
    print(d["func_names"][i],
          d["orig_conv"][i],
          f"imce={d['imce_linear_id'][i]}",
          f"valid_cols={d['valid_cols/' + str(d['func_names'][i])][:8]}")
```

## 한계와 주의사항

1. **시간축 직렬화**: single_qconv 모드는 atomic qconv를 시간축으로 직렬 실행한다. 같은 IMCE가 여러 conv의 매핑에 반복 등장할 수 있다 (공간적으론 같은 코어, 시간적으론 다른 round).
2. **N, OH, OW invariance**: 매핑은 batch / spatial 위치에 독립이다. 2-D 테이블만 저장하고 broadcast로 쓰는 이유.
3. **column-disable**: column-disable 없으면 `column = oc_local`. 있으면 `valid_cols`가 비연속.
4. **`orig_conv` 이름 매칭**: bind_params 직전에 weight Var 이름을 weight bytes hash로 캡처한다 (`capture_orig_conv_names`). 모델 빌더가 weight Var에 의미있는 이름을 부여하면 그대로 보존되고, 못 잡으면 `conv_<orig_conv_id>` 형태로 fallback.
5. **`<const>`**: weight가 처음부터 Constant로 들어오면 (드문 케이스) 평면 배열의 `weight_var`는 `<const>`로 표시되지만 `orig_conv`는 hash lookup으로 정상 복원된다.
