# `diagnose_noise_per_qconv.py` 코드 리뷰 가이드

Per-qconv local 분석으로 chip이 실제 발생시킨 noise가 noise CSV의 예측 분포와 얼마나 일치하는지 측정하는 스크립트. `--compare pysim` 옵션으로 py_runner(=같은 CSV를 직접 사용해 noise를 inject하는 simulator) dump에 대해 같은 분석을 돌려서 lookup pipeline의 self-consistency를 sanity-check할 수 있음.

## 변경/신규 파일

| 파일 | 종류 | 변경 의도 |
|---|---|---|
| `scripts/diagnose_noise_per_qconv.py` | 신규 | 본 분석 스크립트 |
| `tvm_practice/models/resnet8_subset_models.py:317` | 수정 | checkpoint path를 `2026-May-14-16-04-08` (재생성된 imcflow ckpt)로 교체. TVM compile이 사용하는 weight·adjust_factors source. |

리뷰 시 위 2개 파일만 보면 됩니다. 분석 스크립트가 읽어들이는 외부 파일들 (psum_imcu_column_map.npz, concat_per_core.json, noise CSV, chip dump npy, 체크포인트)은 다른 pipeline의 결과물이라 본 리뷰 범위 밖.

---

## 데이터 흐름 (high-level)

```
                        ┌──────────────────────────────────────────────┐
chip dump (per sample,  │  observed_noise =                            │
 per qconv atomic)      │      chip_output                             │
  - input npy  ────────►│      − noise_free_qconv(chip_input,         │
  - output npy ────────►│                          weight_tile,        │
                        │                          kH,stride,pad)      │
checkpoint (imcflow)    │                                               │
  - weight tensors ────►│                                               │
                        │                                               │
psum_imcu_column_map    │                                               │
  - atomic mapping ────►│  predicted_E/Var/range =                     │
  - valid_cols        ─►│      look up noise CSV at                    │
                        │      (pseudo_ch, wbit, adc_code) for          │
concat_per_core.json    │      each (abit,wbit) iteration's psum_adc   │
  - core→pseudo_ch  ───►│      and sum weighted contributions          │
                        │                                               │
noise CSV (per-ch)      │                                               │
  - prob table       ──►│  compare observed vs predicted               │
                        └──────────────────────────────────────────────┘
                                          │
                                          ▼
                          per-atomic / per-orig_conv summary
                            (obs_mean, obs_std, bias,
                             in_range%, |z|>3%, chi2/n)
```

핵심 아이디어: noise-free 함수를 **그 atomic만 isolated** 하게 chip의 실제 input dump 위에 돌려 reference를 만든다. (full sim을 noise=0으로 돌리면 cumulative noise propagation이 없어져서 각 layer의 input 자체가 chip과 달라지므로 incorrect.)

---

## `diagnose_noise_per_qconv.py` 섹션별 설명

### 상수 (line 46~55)

```
ARRAY_SIZE = 256, PRANGE = 2 (HALF), PBITS = 6
WBITS = ABITS = 4
PBOUND = 64, PSTEP = 2.0, NUM_LEVELS = 64
W_SCALE = (1, 2, 4, -8)  # per wbit (sign bit = -8)
```

- HALF mode 가정 (`--vmode HALF`가 default). FULL/QRTR 사용한다면 PRANGE만 바꾸면 PSTEP 자동 재계산.
- `W_SCALE`은 4-bit 2's complement: bit 0/1/2가 양수 1·2·4, bit 3(sign)이 -8.
- **PSTEP은 noise contribution scaling에도 필요**: `inject_noise`는 `pp_noisy = (adc_code + diff_bin) * pp_scale`로 noise를 추가하므로 int16 output 단위의 noise contribution = `diff_bin * PSTEP * w_scale * (1<<abit)`.

### `CONV_PARAMS` 매핑 (line 52~61)

원본 conv name → (kernel_h, stride, padding, checkpoint key). 하드코딩. 모델 (resnet8_subset31)이 바뀌면 여기 수정 필요. resnet8_subset_models.py의 `getModel_()` 정의와 일치해야 함.

리뷰 포인트: weight3_0/weight4_0이 1x1 stride=2 padding=0 downsample. 다른 8개는 3x3.

### `load_atomic_info` (line 81~104)

`psum_imcu_column_map.npz`의 flat 배열을 atomic당 dict로 변환. 각 dict에 atomic의 모든 metadata + `valid_cols` (해당 atomic의 64-col 슬롯 중 실제 사용된 column index 리스트, 길이 = oc_size). `imce_w_1based`는 npz의 `imce_col_in_imce` (0-base) + 1 — concat_per_core.json이 core key를 "h_w"의 w로 1-base 표현하기 때문.

### `build_qconv_to_input_map` (line 107~123)

DEBUG_EXE chip dump dir에서 file prefix 번호 순서를 walk하면서, 각 `imcflow_main_X.npy`에 대해 **직전의 `fused_qnn_imcflow_min_max_quantize*.npy`** 파일명을 찾아 매핑. chip이 IMCE에 던지기 직전의 post-quantize uint8 입력이 이 파일.

리뷰 포인트: 이 heuristic이 안 맞는 경우가 있는지. 현재는 동일 conv의 모든 atomic이 같은 quantize 출력을 공유하는 구조를 가정. (예: weight4_2 6개 atomic이 quantize_3 하나를 공유.)

### `load_pseudo_ch_map` (line 126~140)

`concat_per_core.json["pseudo_ch_to_orig"]`를 역방향 dict로: `(core_h, core_w_1based, orig_ch_0..63) → pseudo_ch_0..511`. pseudo_ch가 noise CSV column index.

### `load_noise_csv` (line 143~204)

CSV를 pandas로 2-row header 형식 (`diff_bin`, `channel`)로 읽어 `(C, R, K)` numpy array로 변환. C=512 pseudo_ch, R=256 (= 4 wpatterns × 64 refs), K=29 diff_bin values.

핵심 invariant 검증:
- row 순서가 `(wpattern_0, ref_0..63), (wpattern_1, ref_0..63), ...` 패턴 — 이 가정이 깨지면 `_sample_noise_per_ch`의 row 인덱스 공식 `wbit*n_refs + adc_code`가 안 맞으므로 assertion으로 검사.
- wpattern은 4개 (0001/0010/0100/1000) → WBITS=4와 일치.

이후 per-row mean/variance/min/max 사전계산:
- `E[c, r] = Σ_k probs[c,r,k] * diff_bins[k]`
- `Var[c, r] = Σ_k probs[c,r,k] * diff_bins[k]² − E[c,r]²`
- `diff_min/diff_max[c, r]`: probability > 1e-12인 bin들의 min/max diff_bin

리뷰 포인트: `EPS = 1e-12`로 prob>0 판정. CSV가 매우 작은 확률(1e-15 같은)이 있다면 cutoff 조정 필요.

### `noise_free_qconv` (line 217~283) — 가장 중요

**Chip ISA를 모사하는 bit-serial qconv. `PsumConv._forward_hw` (CIM/deploy/deploy_modules.py:280-452)의 noise-free 버전.**

```
1. act bit-decompose: act_bp[abit] = (input_uint8 >> abit) & 1  # (1, IC, H, W) {0,1}
2. weight 2's complement bit-decompose:
     w_uint = weight & 0xF   # signed [-8,7] → unsigned [0,15]
     w_bp[wbit] = (w_uint >> wbit) & 1
3. for abit in 0..3:
     # popcount<8 short-circuit eligibility per abit
     acc_mode = (acc_mask & (1<<abit)) == 0
     if acc_mode:
       popcount = conv2d(act_bp[abit], ones_kernel)   # spatial bit-sum
       skip_mask = popcount < 8        # (1,1,OH,OW) bool
     wbs = 0
     for wbit in 0..3:
       psum = conv2d(act_bp[abit], w_bp[wbit])   # int count
       adc_code = round(psum/PSTEP + 0.01).clamp(0, 63)
       psum_adc = adc_code * PSTEP               # 0, 2, 4, ..., 126
       if skip_mask: psum_final = where(skip, psum, psum_adc)  # popcount<8 → raw int
       else:         psum_final = psum_adc
       wbs += psum_final * W_SCALE[wbit]
     scaled = (wbs << abit)
     out += int16_wrap(scaled)
   out = int16_wrap(out)
```

리뷰 포인트:
- **popcount<8 path가 핵심**: input bit-plane의 spatial popcount가 8 미만이면 chip은 ADC quantize도 안 하고 noise도 안 inject 함 (deploy_modules.py:357-371, 393, 405-414). raw integer값을 그대로 사용. 이걸 빠뜨리면 chip-비교 시 부정확한 noise-free reference가 됨.
- skip_mask는 (ABITS, OH, OW) shape으로 반환 — predicted noise 계산 시 skip된 element는 contribution=0으로 처리.
- ADC 양자화의 `+0.01` rounding bias는 `_qconv_single_group`과 acim.py에서 따옴 (banker's rounding 0.5 경계 회피).
- int16 wrap을 `((x + 32768) & 0xFFFF) - 32768`로 구현. PyTorch의 `.to(int16)`가 saturating일 가능성이 있어 explicit 2's-complement wrap 사용.
- 반환값에 `adc_codes_all` shape `(ABITS, WBITS, OC, OH, OW)` 포함 — 각 (abit, wbit) iteration의 ADC 출력 코드. 이걸 predicted noise lookup에서 row index로 사용.

### `compute_predicted_stats` (line 288~383)

각 출력 element별 predicted E[noise], Var[noise], range_min, range_max, mode_total 계산.

- CSV row index: `row = wbit * n_refs + adc_code` (← noise.py:_sample_noise_per_ch:957-1014의 공식과 동일).
- `ch_b`로 OC 차원별 pseudo_ch 인덱싱.
- **각 (abit, wbit) noise scale = `PSTEP * (1 << abit) * W_SCALE[wbit]` (signed)**. PSTEP 인자를 빼면 noise std가 PSTEP=2배만큼 underestimate됨.
- skip_mask로 popcount<8 element를 0으로 mask (chip/pysim 모두 그 위치에서 noise 안 inject).
- 16개 (abit, wbit) contribution 합산:
  - `E_total = Σ E_lk * scale`
  - `Var_total = Σ Var_lk * scale²`  (independence 가정)
  - `mode_total = Σ mode_lk * scale`  (greedy 모드 deterministic 예측값)
  - range_max/min은 signed scale 부호에 따라 dmax/dmin을 swap해서 sum (monotonicity).

리뷰 포인트 (중요):
- **PSTEP 인자가 필수**: `inject_noise`에서 noise는 `(adc_code + diff_bin) * pp_scale`로 더해지고 `pp_scale = PSTEP`. 한번에 다 빼먹기 쉬운 부분이라 명시적으로 documenting.
- **skip_mask 적용**: popcount<8 element에서 contribution=0. chip/pysim과 일관됨.
- **Independence 가정**: 16개 noise sample이 독립이라고 가정하고 variance를 sum. 실제 chip에서는 같은 cycle 내 cross-coupling이 있을 가능성 있음. pysim에는 independence가 정확히 성립하므로 pysim chi2/n=1.00이 정상.
- **int16 wrap 무시**: PsumConv는 (weight_bit_sum << abit)에 int16 wrap을 함. predicted는 wrap 없이 직접 더하는 first-order 근사. 출력이 int16 범위를 크게 안 벗어나는 경우엔 OK이지만, 큰 값에서 mismatch 가능. (코드 docstring에 명시.)
- range_min/range_max는 worst-case 합산 — 즉 모든 contribution이 동시에 max diff_bin을 hit한다는 가정. pysim에서는 이론적으로 항상 100% (CSV 분포에서 sample되므로 worst-case 안에 들어옴) — 실제로도 100% 측정됨. chip에서 100% 미만이면 model gap signal.

### `process_dump_source` (line 389~459)

dump_dir당 한 번 호출. 한 dump source (chip 또는 pysim)에 대해 sample×atomic 전체를 처리.

per (sample, atomic):
1. `qconv_to_input` map으로 input dump 찾기, `np.load`.
2. IC slice (`ic_id * ic_block : ic_id * ic_block + ic_size`) — 같은 conv의 atomic이 input 공유하지만 IC 차원에서 다른 슬라이스를 사용.
3. weight tile: checkpoint의 full weight에서 `[oc_lo:oc_hi, ic_lo:ic_hi]` slice.
4. `noise_free_qconv` 호출 → `clean_out`, `adc_codes`, `skip_mask`.
5. dump (1,1,OH,OW,64) 에서 `valid_cols`만 select하고 transpose해 `(oc_size, OH, OW)` 형태로 정렬 — clean과 channel-by-channel 매칭됨.
6. `observed = dump_selected - clean`.
7. `compute_predicted_stats` 호출 → `E_pred, Var_pred, range_min, range_max, mode_pred`.
8. `in_range`: observed가 [range_min, range_max] 안에 들어가는지 element-wise.
9. per-atomic 누적.

### `print_summaries` (line 462~531)

label (chip/pysim)별로 per-atomic / per-orig_conv 표 출력.

- Per-atomic summary: obs (mean/std), pred (E/sd), mode%, inrng%, |z|>3%.
- Per-orig_conv summary: atomic들 합쳐서 같은 지표 + bias (= obs - pred_E의 평균) + chi2/n.
- `z = (observed - E_pred) / sqrt(max(Var_pred, VAR_FLOOR))`. `VAR_FLOOR=1.0`은 pred_Var=0 element (모든 contribution이 deterministic CSV row를 hit)에서 z가 발산하는 걸 막기 위함.
- `chi2/n = mean(z²)`. ≈ 1이면 perfect fit, 클수록 model이 variance/bias를 underestimate.
- `mode%` = `(observed == predicted_mode_sum).mean()`. py_runner가 greedy mode일 때만 의미 있음 (deterministic이라 ~100% 기대). alias/sample 모드면 random이라 ~0%가 정상.

### main (line 534~622)

1. args 파싱 + 데이터 로드 (atomic, layout, CSV, ckpt, weights).
2. atomic별 pseudo_chs 계산.
3. `sources`: `[(label, dump_dir), ...]` — `--compare` 옵션에 따라 chip/pysim/both 결정.
4. 각 source에 대해 `process_dump_source` 호출.
5. 각 source에 대해 `print_summaries` 호출 (chip / pysim 각각 별도 표).
6. `--output-dir` 지정 시 source별 atomic당 npz 저장 (`label__func.npz` 형식).

### Output 저장 (line 604~622, optional)

`--output-dir`이 지정되면 source × atomic당 1개 npz 저장 (`{label}__{func}.npz`):
- `observed`, `pred_E`, `pred_Var`, `pred_mode`, `pred_range_min/max`, `in_range`, `skip_mask`: shape `(n_samples, ...)`
- `pseudo_chs`, `valid_cols`, `imce_h/w`, `orig_conv`, `sample_start`, `n_samples`: metadata

추가 분석 (per pseudo_ch histogram, per (wbit, adc_code) bucket 분석 등)에 사용.

---

## `resnet8_subset_models.py:317` 변경

기존:
```python
VMode.HALF: '...2026-May-13-10-11-49/imcflow/2026-May-13-19-05-07/checkpoint.pth.tar',
```

변경:
```python
VMode.HALF: '...2026-May-13-10-11-49/imcflow/2026-May-14-16-04-08/checkpoint.pth.tar',
```

이유: CIM repo pull 후 imcflow ckpt를 재생성해서 `2026-May-14-16-04-08/`에 새로 저장. inference.py가 사용하는 `--adj_factor` opt_factors.json (같은 디렉토리에 저장됨)과 TVM이 load하는 `checkpoint['adjust_factors']`가 일치해야 하므로 같은 디렉토리를 가리키게 함.

검증: state_dict 110 keys 구조가 이전 (May-13-19-05-07) ckpt와 동일 → TVM의 weight loader (`_get_tensor_from_checkpoint`)가 그대로 동작. adjust_factors 값도 이전 ckpt와 1e-6 이내 일치 (regenerate 결과가 deterministic하게 같은 값).

---

## 리뷰 시 의심해볼 만한 곳

### 1. `noise_free_qconv`의 chip ISA 일치성
- PsumConv._forward_hw와 동일한 결과를 내야 함 (popcount<8 path 포함). pysim 비교에서 chi2/n=1.00 나오는 것으로 큰 부분은 검증됨 (PSTEP, popcount, row index, mapping, bit decomp 다 통과).
- 단 PsumConv 자체가 진짜 chip ISA와 일치하는지는 별도 검증 필요. chip이 noise=0으로 돌릴 수 있는 mode가 있다면 atomic 단위로 bit-for-bit 비교 가능.
- `+0.01` rounding bias가 chip ADC와 정확히 일치하는가? +0.01이 아니라 다른 값이거나, 사실 fair-round일 가능성도 검증 필요.

### 2. `build_qconv_to_input_map`의 fragility
- "직전 quantize"가 input이라는 heuristic. compile 결과의 graph ordering이 바뀌면 깨질 수 있음. atomic별로 어느 quantize에 의존하는지 별도 metadata가 있으면 더 robust할 것.

### 3. Predicted noise model의 independence 가정
- 16개 (abit, wbit) noise sample이 모두 독립이라는 가정. 실제 chip에서 같은 IMCE column의 noise는 한 cycle 내에서 correlated 될 수 있음 (PVT 변동, supply ripple 등). 만약 그렇다면 predicted Var는 실제 변동성을 underestimate.
- 또한 같은 IMCE의 인접 column들끼리 cross-coupling이 있을 수 있음 — 이건 per-channel 분포로는 capture 안 됨.

### 4. int16 wrap이 predicted 식에서 빠진 점
- PsumConv는 (weight_bit_sum * (1<<abit))을 int16 wrap. observed 쪽 (chip - clean)은 wrap이 다 들어간 결과. predicted 쪽은 wrap 없이 분산을 sum.
- 출력 값이 int16 범위 (-32768..32767) 안에 있으면 wrap이 일어나지 않으므로 OK. 평소 atomic raw psum 범위는 ±4000 정도로 보였음 → 안전.

### 5. `weight4_0`/`weight3_0` (1x1 downsample) 의 큰 mismatch
- chi2/n이 1254/776로 가장 큼. in_range%도 가장 낮음 (33%/42%).
- 1x1 conv는 IC가 적고 spatial mixing이 없어 패턴이 단순. CSV가 측정될 때의 test pattern과 실제 inference activation pattern 사이의 mismatch가 클 수 있음.
- 또는 1x1 atomic의 weight tile/배치가 다른 conv와 다르게 처리될 가능성도 있음 (downsample path 특수성). compile 단계에서 별도 처리 path를 가는지 확인하면 좋음.

### 6. systematic negative bias (-50 ~ -170 per layer)
- CSV는 zero-mean noise (또는 거의 0)를 model하는데 observed bias가 음수 일관. ADC offset, kT/C noise, leakage, bitline droop 등 mean이 아닌 source가 chip에 있는데 CSV measurement protocol에서 빠졌을 가능성.
- 이건 noise model의 fundamental gap이라 NAT 학습 모델로 회피하기 어려움 — chip-side에서 calibration이 필요할 수도.

---

## 실행 방법 및 검증

```bash
cd /root/project/tvm/tvm_practice/test_imcflow/codegen

# Smoke test (2 samples, chip only, ~10 sec)
python scripts/diagnose_noise_per_qconv.py --n-samples 2

# Sanity check on pysim (py_runner dump이 CSV로 noise inject한 결과와 비교)
python scripts/diagnose_noise_per_qconv.py --n-samples 1 --compare pysim --verbose

# 둘 다 비교
python scripts/diagnose_noise_per_qconv.py --n-samples 2 --compare both

# Full run (200 samples, raw 저장)
python scripts/diagnose_noise_per_qconv.py --n-samples 200 --output-dir runs/full_200

# Layer drill-down
python scripts/diagnose_noise_per_qconv.py --n-samples 50 --layers weight4_0,weight3_0
```

### pysim sanity check 기대치

py_runner는 CSV를 직접 사용해 noise를 inject하므로, 우리 lookup pipeline이 옳다면:
- `in_range% = 100.00%` (CSV worst-case bound이 항상 capture)
- `chi2/n ≈ 1.00` (predicted Var이 observed variance와 일치)
- `bias ≈ 0` (pred_E와 obs_mean이 일치)
- `|z|>3% ≈ 0.27%` (N(0,1)의 tail에 가깝게)
- `mode%`: py_runner의 `--noise-mode greedy`면 ~100%, `alias`/default sample이면 random이라 ~0-10%

실제 smoke test 결과 (`--compare pysim --n-samples 1`, weight2_1 atomic_0):
```
weight2_1   16384  41.10  411.69  0.04  3.00%  100.00%  0.39%  1.00
```
모두 기대치 정확히 일치 — lookup pipeline (PSTEP, popcount<8, row formula, channel mapping)이 검증됨.

### chip 결과 해석

chip은 CSV가 모델링하지 않는 noise source (ADC offset, kT/C, leakage, bitline droop, cross-coupling)도 포함하므로 mismatch가 정상. 실제 측정:
- `bias`: -20 ~ -160 (CSV는 zero-mean인데 chip은 systematic negative bias)
- `chi2/n`: 12 ~ 613 (chip의 실제 variance가 CSV 예측보다 10-600배 큼)
- `in_range%`: 50% ~ 95% (CSV의 worst-case bound조차 50-95%만 capture)
- 1x1 downsample conv (weight3_0/weight4_0)이 가장 큰 mismatch.
