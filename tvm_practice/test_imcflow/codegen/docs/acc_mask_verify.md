# Problem
- acc_mask는 연산 정확도에 큰 영향을 준다.
- acc_mask가 예상대로 동작하는지 검증해본적이 없다.
- 이번 검증의 범위는 `AccMask.BM_0000`만이다.
- 모든 input case를 acc-mode skip path로 들어가도록 만들어서, chip `res`와 reference `ref`가 exact match하는지 확인한다.
  
# Initial Plan
- acc_mask를 항상 `AccMask.BM_0000`으로 set한다.
- input bitplane의 popcount를 acc-mode skip 조건에 맞게 작게 정한다. 그리고 input active channel 조합을 random sampling한다.
- weight는 random sampling한다.
- output 결과를 sampling하고 ref value와 eval value를 비교한다.
- 모든 case가 acc-mode skip path로 동작하도록 만들기 때문에 error는 항상 정확히 0이어야 한다.
- imcflow repo의 `xilinx/measurement`에서 사용한 code들을 재사용해서 측정한다.

## Plan 검토

방향은 맞지만, 구현할 때 아래 조건을 명확히 해야 한다.

1. threshold는 `8 이하`가 아니라 **`8 미만`** 이다.
   - simulator/deploy 기준 condition은 `popcount < 8`.
   - popcount가 정확히 8이면 skip이 아니라 ADC quant/noise path로 가기 때문에 이번 검증 대상에서 제외한다.
   - 따라서 zero-error를 기대하려면 popcount는 `0..7`이어야 한다.

2. `acc_mask` polarity를 명확히 해야 한다.
   - `acc_mask` bit clear: 해당 `abit`는 acc-mode 대상. `popcount < 8`이면 ADC quant/noise를 skip한다.
   - `acc_mask` bit set: 해당 `abit`는 항상 ADC quant/noise path를 탄다.
   - 따라서 `AccMask.BM_0000`은 모든 activation bit에서 skip 가능.
   - 이번 실험에서는 `BM_1111`이나 single-bit mask를 쓰지 않는다.

3. random input을 그대로 쓰면 모든 bitplane의 popcount를 통제하기 어렵다.
   - uint4 input을 random으로 만들면 abit별 popcount가 섞이고, 어떤 bit는 popcount>=8이 될 수 있다.
   - 따라서 uint4 value를 직접 random sampling하지 않는다.
   - 4개 bitplane을 각각 독립적으로 생성하고, 마지막에 bitwise sum으로 합친다.
   - 이렇게 하면 모든 bitplane의 popcount가 `0..7`임을 완벽하게 보장할 수 있다.

4. "error가 정확히 0"이라는 기대는 skip path에 대해서만 성립한다.
   - `acc_mask` bit clear + 모든 abit의 popcount<8이면 chip result와 reference가 exact match해야 한다.
   - 이번 test matrix는 모든 case가 이 조건을 만족하도록 구성한다.
   - 따라서 any nonzero diff는 fail로 본다.

# Implementation Plan
## 구현 목표

새 스크립트를 TVM 쪽에 추가한다.

```text
tvm_practice/test_imcflow/codegen/scripts/verify_acc_mask_skip.py
```

역할:

- synthetic input/weight `.npz` 생성
- `/root/project/imcflow/xilinx/measurement`의 `test_conv`/`BulkTuneExecutor` 재사용
- 여러 bitplane-popcount profile을 `AccMask.BM_0000`으로 chip에서 실행
- `ref`와 `res`를 비교해서 acc-mode skip이 exact zero-error인지 검증

## Test Matrix

가장 중요한 matrix는 bitplane별 popcount profile이다. 모든 profile에서 각 abit popcount가 7 이하여야 한다.

```text
popcount profiles:
  uniform    : [1, 1, 1, 1], [2, 2, 2, 2], [4, 4, 4, 4], [7, 7, 7, 7]
  mixed      : [1, 2, 4, 7], [7, 4, 2, 1]
  sparse     : [0, 1, 0, 7], [3, 0, 5, 0]
acc_mask      : BM_0000 only
repeat        : 기본 20~100
```

기대값:

| 조건 | 기대 |
| --- | --- |
| `BM_0000`, `popcount < 8` | skip path. `res-ref == 0` expected |

이번 실험에서는 positive control을 하지 않는다. 목적은 "BM_0000에서 acc-mode로 들어간 case가 정확히 0 error를 내는지"만 확인하는 것이다.

주의: 사용자가 말한 "8 이하"는 의도상 "acc-mode로 들어가도록 작은 popcount"이지만, 현재 simulator/deploy 구현은 `popcount < 8`이다. 그러므로 구현에서는 popcount 8을 넣지 않는다.

## Synthetic Input/Weight 생성

처음에는 conv shape을 작고 단순하게 고정한다.

```text
input  : IC=16, IH=8, IW=8
weight : OC=64, IC=16, KH=1, KW=1
stride : 1
pad    : 0
output : OC=64, OH=8, OW=8
```

input 생성:

```text
1. input 전체를 0으로 초기화
2. abit=0..3 각각에 대해 독립적인 active channel mask를 생성한다
3. 각 bitplane의 active channel 수는 profile[abit]이며, 항상 0..7이다
4. 선택된 channel의 모든 spatial 위치에 해당 bit value `(1 << abit)`를 더한다
5. 4개 bitplane을 합친 uint4 input을 저장한다
```

이렇게 하면 모든 output spatial location에서 각 bitplane의 popcount가 profile과 정확히 일치한다.

예:

```text
profile = [1, 2, 4, 7]

abit0 active channels = 1개  -> input += 1
abit1 active channels = 2개  -> input += 2
abit2 active channels = 4개  -> input += 4
abit3 active channels = 7개  -> input += 8
```

같은 channel이 여러 bitplane에서 active일 수 있다. 그 경우 input value는 bitwise OR/sum으로 합쳐진다. 예를 들어 channel 3이 abit0과 abit2에서 active이면 해당 channel value는 `0b0101 = 5`가 된다. 그래도 각 bitplane popcount는 독립적으로 유지된다.

weight 생성:

```text
1. weight 전체를 0으로 초기화
2. target physical column 몇 개만 사용하거나, OC=64 전체에 random signed int4 weight를 채운다
3. 처음 검증은 해석을 쉽게 하기 위해 OC=64 전체 random weight 사용
4. 필요하면 target column 하나만 nonzero인 isolated mode도 추가
```

random signed int4 weight 범위:

```text
-8 .. 7
```

주의:

- popcount=0이면 output은 항상 0이라 검증력이 약하다. smoke/baseline으로만 선택적으로 사용한다.
- main test는 profile 안의 각 bitplane popcount가 1,2,4,7 같은 값을 갖도록 구성한다.
- popcount=8은 이번 실험에서 제외한다. 현재 코드 기준 `popcount < 8`만 skip이기 때문이다.

## Measurement Args 생성

measurement의 `common/test_conv.py`가 이미 `i_npz`, `w_npz`, `acc_mask`를 받는다. 따라서 custom generator가 만든 `.npz` path를 args dict에 넣으면 된다.

args dict 예:

```python
{
    "IH": 8,
    "IW": 8,
    "IC": 16,
    "OC": 64,
    "kernel": 1,
    "stride": 1,
    "padding": 0,
    "h_id": 0,
    "w_id": 1,
    "i_npz": ".../case_input.npz",
    "w_npz": ".../case_weight.npz",
    "scan_val": "0x0a",
    "adcmode": ADCMode.SIX,
    "vmode": VMode.FULL,
    "multmode_set": MultModeSet.S4,
    "acc_mask": AccMask.BM_0000,
    "runsim": 0,
    "test_num": case_id,
    "file_postfix": case_name,
}
```

실행은 `measure_weight3_0_synthetic_noise.py`와 같은 방식으로 한다.

```text
BulkTuneExecutor(args_list, dda, ddc, ddl, ddf, ps_manager, ssh_client, bft_client)
executor.execute()
executor.rerun() for repeats
executor.get_reference_and_result()
```

## Result 분석

각 case에 대해 `ref`, `res`를 `(OC, OH, OW)`로 복원하고 다음 metric을 저장한다.

```text
diff = res - ref

n_elem
zero_rate
nonzero_count
diff_abs_max
diff_mean
diff_std
diff_min
diff_max
```

pass/fail rule:

```text
pass if nonzero_count == 0 and diff_abs_max == 0
```

모든 generated case가 expected-skip case이므로, repeat 전체에서 nonzero diff가 하나라도 나오면 fail이다.

## Output Files

기본 output directory:

```text
tvm_practice/test_imcflow/codegen/debugging/acc_mask_verify/
```

생성 파일:

```text
synthetic_npz/
  case_*.input.npz
  case_*.weight.npz

acc_mask_cases_manifest.json
acc_mask_test_args.json
acc_mask_measurements_raw.npz
acc_mask_summary.csv
```

`acc_mask_summary.csv` columns:

```text
case_id
popcount_profile
acc_mask
expected_skip
n_repeats
n_elem
zero_rate
nonzero_count
diff_abs_max
diff_mean
diff_std
diff_min
diff_max
pass
```

여기서 `acc_mask`는 항상 `BM_0000`, `expected_skip`은 항상 `True`로 기록한다.

## CLI

예상 CLI:

```bash
python3 tvm_practice/test_imcflow/codegen/scripts/verify_acc_mask_skip.py \
  --connection root@HOST:PORT \
  --board B2 \
  --scan-val 0x0a \
  --dda 1.13 --ddc 1.17 --ddl 0.006 --ddf 1.24 \
  --repeats 50 \
  --out-dir tvm_practice/test_imcflow/codegen/debugging/acc_mask_verify
```

generation-only smoke:

```bash
python3 tvm_practice/test_imcflow/codegen/scripts/verify_acc_mask_skip.py \
  --generate-only \
  --scan-val 0x0a \
  --dda 1.13 --ddc 1.17 --ddl 0.006 --ddf 1.24
```

옵션:

```text
--profiles uniform,mixed,sparse
--pairs-per-case 4
--isolated-column
--node 0,1
--generate-only
--transfer-only
--dryrun
```

## 구현 순서

1. `verify_acc_mask_skip.py` skeleton 작성
   - argument parser
   - measurement stack import helper
   - enum parsing helper

2. synthetic case generator 구현
   - bitplane별 independent controlled-popcount input 생성
   - random signed int4 weight 생성
   - `.npz` 저장
   - manifest 저장

3. local verification 추가
   - 생성된 input에서 abit=0..3 각각의 popcount가 profile과 정확히 일치하는지 assert
   - 모든 bitplane popcount가 8 미만인지 assert
   - expected skip 계산이 맞는지 manifest에 기록

4. measurement args builder 구현
   - `i_npz`, `w_npz`, `acc_mask`, mode, voltage metadata 포함
   - `acc_mask`는 항상 `AccMask.BM_0000`
   - `file_postfix`로 case별 generated test가 섞이지 않게 함

5. BulkTuneExecutor integration
   - `execute()`
   - repeat loop에서 `rerun()`
   - `get_reference_and_result()`로 raw result collect

6. result decode 구현
   - flat int16 result를 256-bit word로 repack
   - `transform_conv_output_to_3d(res_words, (OC, OH, OW))`
   - diff summary 계산

7. pass/fail report 출력
   - nonzero diff가 하나라도 있으면 바로 fail로 highlight
   - profile별 zero-rate table 출력

## 성공 기준

최소 성공 조건:

```text
BM_0000 + all bitplane popcount <= 7:
  all profile에서 diff_abs_max == 0
  all repeat에서 nonzero_count == 0
```

이 기준을 통과하면 `AccMask.BM_0000`에서 acc-mode skip path로 들어간 input은 chip에서도 exact raw accumulation으로 처리된다고 볼 수 있다.
