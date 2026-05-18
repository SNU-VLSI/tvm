# Fine Noise Model 분석 계획

downsample 쪽, 특히 `weight3_0`의 `in_range`가 유독 작다. 현재 CSV noise model 자체가 실제 inference에서 쓰이는 bitline-level condition을 잘 대표하는지 확인해야 한다.

핵심 질문은 다음이다.

- CSV가 측정한 `(pseudo_ch, wbit, adc_code/ref)` noise 분포가 실제 ResNet inference의 fine-psum 조건에서도 맞는가?
- 현재 bitline planner의 `upper_n` pattern이 너무 제한적인 input/weight 조합만 보고 있는 것은 아닌가?
- mismatch가 CSV 분포 문제인지, skip/acc-mode 문제인지, 혹은 chip의 systematic offset/correlation 문제인지 분리할 수 있는가?

여기서 관심있는 psum은 최종 qconv output psum이 아니라, IMC array에서 한 번의 bitline use로 만들어지는 fine psum이다. 즉 CSV lookup 단위인:

```text
(pseudo_ch, wbit, adc_code/ref)
```

에 대응하는 raw popcount/ADC code를 말한다.

## 현재 문제

현재 noise modeling은 bitline planner 등을 사용한다. 이때 input bitplane은 all-ones이고, weight는 IMCE의 위쪽 row부터 1을 채우는 `upper_n` pattern이다.

예를 들어 bitline ref value가 8일 때의 noise를 분석하고 싶으면:

```text
input bitplane = all ones
weight bitplane = upper 8 rows만 1, 나머지는 0
```

으로 만든다.

clip option을 쓰면 현재 modeling하는 bitline을 제외한 나머지를 max value로 만들고 후처리할 수도 있다. 그래도 핵심은 동일하다. 현재 protocol은 ref value `X`를 만드는 수많은 input/weight 조합 중 하나인 `all-ones input + upper_n weight`만 측정한다.

하지만 실제 inference의 fine psum은 다음처럼 훨씬 다양한 방식으로 만들어진다.

```text
raw_psum = sum_c input_bit[c, oh, ow] * weight_bit[oc, c, kh, kw]
```

따라서 같은 raw psum `X`라도 아래 조합들이 모두 가능하다.

- input active channel이 위쪽 row에 몰린 경우
- input active channel이 흩어진 경우
- weight active channel이 위쪽 row에 몰린 경우
- input/weight active row의 intersection만 `X`인 경우
- popcount는 크지만 target wbit intersection은 작은 경우

현재 CSV가 이런 조합 차이에 둔감하다는 가정이 깔려 있다. 이 가정이 깨지면 `ref=8` CSV 분포가 실제 `ref=8` inference noise를 대표하지 못한다.

## 관찰된 현상

### `weight3_0` mapping

`tvmgen_default_imcflow_main_15`는 실제 ResNet8의 `weight3_0`에 해당한다.

```text
orig_conv  = weight3_0
PyTorch    = layer2.0.downsample.1 / layer2.block_int16.downsample.1
shape      = 1x1, stride=2, 16 -> 32 channels
core       = imce_h=2, imce_w=1
```

즉 residual downsample branch의 1x1 qconv이다.

### Downsample의 `in_range`가 낮음

`diagnose_noise_per_qconv.py` 기준으로 `weight3_0`, `weight4_0` 같은 1x1 downsample layer에서 `in_range`가 특히 낮다. 이는 단순히 stride=2 때문이라기보다는, 1x1 downsample의 input/weight pattern이 CSV 측정 protocol과 더 다르게 생겼을 가능성이 크다.

이전 측정에서 `weight3_0`은 acc-mode skip 영향도 컸다.

```text
weight3_0 skip_aff ~= 96.8%
strict in_range ~= 9.0%
no-skip shifted in_range ~= 40.7%
```

`--acc-mask 15`로 보면 `in_range`가 올라가지만, 이것은 너무 당연한 결과다. skip을 끄면 clean reference와 predicted range 자체가 바뀌고, range도 넓어진다. 따라서 `acc-mask=15` 결과만으로 chip에서 skip이 잘못 동작한다고 결론낼 수는 없다.

### pysim은 `in_range=100%`

pysim 비교에서는 항상 `in_range=100%`였다. 이는 적어도 다음 lookup pipeline은 크게 틀리지 않았다는 뜻이다.

- ADC row index 계산
- `PSTEP` scale
- `wbit`별 signed scale
- `pseudo_ch` mapping
- popcount<8 skip 처리
- CSV min/max range 계산

따라서 현재 문제는 "CSV lookup 구현 버그"보다는 "CSV로 모델링한 분포가 실제 chip/inference condition을 충분히 대표하지 못함" 쪽에 무게가 있다.

## 수식 정리

### Fine raw psum

`weight3_0`은 1x1 conv라 spatial kernel mixing이 없다.

```text
raw_psum[abit, wbit, oc, oh, ow]
  = sum_ic act_bit[abit, ic, ih, iw] * weight_bit[wbit, oc, ic, 0, 0]
```

stride=2이므로:

```text
ih = oh * 2
iw = ow * 2
```

### ADC code

현재 deploy/pysim 쪽 rounding은 다음으로 맞추고 있다.

```text
adc_code = clamp(round(raw_psum / PSTEP + 0.01), 0, 63)
PSTEP = 2.0
```

CSV row는:

```text
row = wbit * n_refs + adc_code
```

### Output noise scale

CSV의 `diff_bin`은 ADC code noise 단위다. qconv output level로 변환하려면 다음 scale을 곱한다.

```text
output_noise = diff_bin * PSTEP * W_SCALE[wbit] * 2^abit
W_SCALE = [1, 2, 4, -8]
```

따라서 같은 CSV diff noise라도 `abit`, `wbit`에 따라 output noise 크기와 부호가 달라진다.

### Skip condition

acc-mode skip 조건은 activation bitplane의 spatial popcount 기준이다.

```text
skip = (acc_mask bit for abit is unset) and (popcount(input_bitplane) < 8)
```

skip이면 ADC quantization과 noise injection을 생략하고 raw conv result를 사용한다. 이 경우 CSV noise target이 아니다. 따라서 CSV 분포 검증은 우선 `skip=False` tuple에 집중해야 한다.

## 검증 전략

목표는 점점 low-level로 내려가며 원인을 분리하는 것이다.

### 1. 실제 ResNet `weight3_0`의 fine psum 분포 분석

스크립트:

```bash
python3 tvm_practice/test_imcflow/codegen/scripts/analyze_weight3_0_ref_psum.py \
  --n-samples 200
```

출력:

```text
debugging/noise_lowlevel/weight3_0/
  weight3_0_ref_psum_by_tuple.csv
  weight3_0_ref_psum_by_adc.csv
  weight3_0_candidate_tuples.json
```

집계 key:

```text
orig_conv, func,
core_h, core_w,
valid_col, pseudo_ch, oc_local,
abit, wbit,
raw_psum, adc_code,
skip
```

이 단계에서 봐야 할 것:

- 실제 inference가 어떤 `adc_code`에 몰려 있는지
- skip/non-skip 비율이 어떤지
- 특정 `pseudo_ch`, `valid_col`, `wbit`에 bias가 있는지
- CSV support가 너무 좁은 row를 실제 inference가 자주 쓰는지

### 2. Candidate tuple 선정

우선은 `skip=False` tuple만 target으로 한다. skip=True는 ADC/noise path를 타지 않으므로 CSV noise 분포 검증 대상이 아니다.

candidate JSON에는 다음 정보가 들어간다.

```text
valid_col
pseudo_ch
core_h, core_w
abit, wbit
raw_psum
adc_code
target_popcount
csv_diff_mean/std/min/max
output_scale
csv_hist
```

선정 기준:

- non-skip count가 큰 tuple
- 모든 `wbit`을 어느 정도 커버
- low ADC code 영역 우선 포함
- `weight3_0`에서 실제 자주 등장하는 physical column 우선

### 3. 같은 ref psum을 만드는 synthetic input/weight pair 생성

스크립트:

```bash
python3 tvm_practice/test_imcflow/codegen/scripts/measure_weight3_0_synthetic_noise.py \
  --generate-only \
  --candidates tvm_practice/test_imcflow/codegen/debugging/noise_lowlevel/weight3_0/weight3_0_candidate_tuples.json \
  --limit 1 \
  --pairs-per-candidate 2 \
  --scan-val 0x0a \
  --dda 1.13 --ddc 1.17 --ddl 0.006 --ddf 1.24
```

synthetic conv shape:

```text
input  = (IC=16, IH=32, IW=32)
weight = (OC=64, IC=16, KH=1, KW=1)
stride = 2
padding = 0
output = (OC=64, OH=16, OW=16)
```

생성 방식:

```text
1. input은 전부 0으로 초기화
2. target abit만 켜진 active_input channel을 target_popcount개 선택
3. weight는 전부 0으로 초기화
4. target valid_col에만 nonzero weight를 넣음
5. active_input 중 raw_psum개 channel만 target wbit weight를 켬
```

결과적으로 target column에서:

```text
raw_psum == candidate.raw_psum
adc_code == candidate.adc_code
popcount >= 8
skip == False
```

가 된다.

`raw_psum < 8`이어도 `target_popcount >= 8`로 만들 수 있다. input bit가 켜진 channel은 8개 이상 두고, target weight bit가 켜진 channel만 `raw_psum`개로 제한하면 된다.

생성 파일:

```text
synthetic_npz/
  cand000_pair000_input.npz
  cand000_pair000_weight.npz
synthetic_cases_manifest.json
```

### 4. Chip에서 synthetic case 반복 측정

실제 측정:

```bash
python3 tvm_practice/test_imcflow/codegen/scripts/measure_weight3_0_synthetic_noise.py \
  --candidates tvm_practice/test_imcflow/codegen/debugging/noise_lowlevel/weight3_0/weight3_0_candidate_tuples.json \
  --connection root@HOST:PORT \
  --board B2 \
  --scan-val 0x0a \
  --dda 1.13 --ddc 1.17 --ddl 0.006 --ddf 1.24 \
  --repeats 100
```

measurement reuse 방식:

- `i_npz`, `w_npz`에 synthetic input/weight path를 넣은 `test_conv` args dict 생성
- `/root/project/imcflow/xilinx/measurement`의 `BulkTuneExecutor`에 args list 전달
- `BulkExecutor`가 missing generated test를 찾고, 없으면 `common.test_maker.run_test_convs()`로 생성
- generated test files를 board로 upload
- remote에서 `test_imcflow.out` 실행
- result file을 다시 local로 download
- 반복 측정은 첫 `execute()` 이후 `rerun()` 사용

결과:

```text
synthetic_test_args.json
synthetic_measurements_raw.npz
synthetic_vs_csv_summary.csv
```

비교 방식:

```text
chip_diff = res - ref
diff_bin = chip_diff / output_scale
```

그리고 CSV의 `csv_diff_mean/std/min/max`와 비교한다.

주요 metric:

```text
diff_bin_mean
diff_bin_std
csv_diff_mean
csv_diff_std
csv_range_hit_pct
mean_shift_diff_bin
std_ratio_diff_bin
```

## 해석 기준

### Case A: synthetic 측정이 CSV와 잘 맞음

이 경우 CSV row 자체는 맞다. 그러면 ResNet inference mismatch는 다음 쪽을 의심해야 한다.

- 여러 `(abit,wbit)` contribution을 합칠 때의 correlation
- 같은 output element 안에서 noise independence 가정 실패
- non-target columns 또는 disabled columns와의 coupling
- clean reference mismatch
- full network dump path의 postprocess/wrap/scale 문제

### Case B: synthetic 측정도 CSV와 안 맞음

이 경우 CSV measurement protocol의 representativeness 문제가 크다.

가능한 원인:

- `upper_n` row placement가 실제 active row placement를 대표하지 못함
- 같은 raw psum이라도 active row 위치에 따라 bitline noise가 달라짐
- input all-ones 조건과 실제 sparse activation 조건이 다름
- physical column/pseudo channel mapping별 offset이 CSV에 충분히 반영되지 않음
- scan/voltage/mode 조건이 CSV 생성 조건과 다름

이 경우 다음 단계는 같은 `(pseudo_ch,wbit,adc_code)`에 대해 여러 input/weight pair를 랜덤 생성해서 분포를 넓게 샘플링하는 것이다.

### Case C: mean은 맞고 std/tail만 안 맞음

CSV의 평균 calibration은 괜찮지만 tail/support가 부족한 상황이다.

대응:

- CSV histogram support 확장
- per-row empirical tail 보정
- variance inflation factor 도입
- row placement별 mixture model 도입

### Case D: std는 맞고 mean shift만 큼

systematic offset 문제다.

대응:

- `(pseudo_ch,wbit,adc_code)`별 mean offset 보정
- physical column별 bias table
- voltage/scan별 offset calibration

## 다음 구현 TODO

1. `synthetic_vs_csv_summary.csv`를 여러 run에서 merge하는 비교 스크립트 추가
2. 같은 candidate에 대해 `pairs-per-candidate`를 크게 늘려 row-placement sensitivity 측정
3. `upper_n` pair와 random pair를 같은 `(raw_psum, adc_code)`로 나란히 측정
4. `skip=True` 전용 실험 추가: chip이 정말 skip path에서 noise를 inject하지 않는지 확인
5. `weight4_0`에도 같은 pipeline 확장
6. `vmode`, `scan_val`, voltage가 CSV 생성 조건과 완전히 같은지 metadata로 강제 기록

## 현재 결론

현재까지의 evidence는 다음과 같다.

- pysim은 `in_range=100%`라 CSV lookup implementation은 대체로 맞다.
- chip에서는 `weight3_0`/`weight4_0` downsample에서 mismatch가 크다.
- `acc-mask=15`로 `in_range`가 오르는 것은 expected behavior라 원인 증명으로 쓰면 안 된다.
- 가장 유력한 다음 검증은 `weight3_0`의 실제 fine psum tuple을 뽑고, 같은 tuple을 synthetic input/weight로 chip에서 반복 측정해 CSV row와 직접 비교하는 것이다.

즉 문제를 full qconv output에서 바로 보지 말고:

```text
ResNet output mismatch
  -> weight3_0 atomic mismatch
  -> output element별 observed noise
  -> (abit,wbit,pseudo_ch,adc_code)별 fine noise
  -> 같은 fine psum을 만드는 synthetic chip measurement
```

순서로 내려가며 분해한다.
