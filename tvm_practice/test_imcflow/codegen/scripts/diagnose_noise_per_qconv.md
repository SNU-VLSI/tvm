# `diagnose_noise_per_qconv.py` 코드 리뷰 가이드

Per-qconv local 분석으로 chip이 실제 발생시킨 noise가 noise CSV의 예측 분포와 얼마나 일치하는지 측정하는 스크립트. `--compare pysim` 옵션으로 py_runner(=같은 CSV를 직접 사용해 noise를 inject하는 simulator) dump에 대해 같은 분석을 돌려서 lookup pipeline의 self-consistency를 sanity-check할 수 있음.

## 현재 실행 예

```bash
python3 scripts/diagnose_noise_per_qconv.py \
    --noise-dir /root/project/CIM/noise/noise_df/B2_out_refine/N32 \
    --noise-csv B2_noise_matrix_per_ch_concat__alpha0.5_w5_T2.0_exclude_ref0-16.csv \
    --ckpt-alias tmp01_refine_ndis32 \
    --ckpt-board B2 \
    --ckpt-vmode half \
    --n-samples 100 \
    --skip-diagnose
```

- `--noise-dir`: noise CSV와 `concat_per_core.json`이 있는 디렉토리.
- `--noise-csv`: 실제 lookup할 per-channel noise CSV.
- `--ckpt-alias`: chip debug dump를 만든 checkpoint와 반드시 맞춰야 함. mismatch가 있으면 `chip - clean`이 순수 noise가 아니라 weight/model mismatch까지 포함함.
- `--skip-diagnose`: 현재 skip-aware range와 no-skip 가정 range를 같이 출력.

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

실제 구현에서는 `observed_noise = signed_int16(chip_output - noise_free_output)`으로 int16 wrap을 반영한다.

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
- **predicted range는 선형 합산**: `noise_free_qconv`와 `observed` 계산은 int16 wrap을 반영하지만, predicted noise range는 contribution을 선형 합산한다. noise contribution 자체가 wrap 경계를 넘을 정도로 크면 mismatch 가능성이 있다.
- range_min/range_max는 worst-case 합산 — 즉 모든 contribution이 동시에 max diff_bin을 hit한다는 가정. pysim에서는 이론적으로 항상 100% (CSV 분포에서 sample되므로 worst-case 안에 들어옴) — 실제로도 100% 측정됨. chip에서 100% 미만이면 model gap signal.

### `process_dump_source` (line 389~459)

dump_dir당 한 번 호출. 한 dump source (chip 또는 pysim)에 대해 sample×atomic 전체를 처리.

per (sample, atomic):
1. `qconv_to_input` map으로 input dump 찾기, `np.load`.
2. IC slice (`ic_id * ic_block : ic_id * ic_block + ic_size`) — 같은 conv의 atomic이 input 공유하지만 IC 차원에서 다른 슬라이스를 사용.
3. weight tile: checkpoint의 full weight에서 `[oc_lo:oc_hi, ic_lo:ic_hi]` slice.
4. `noise_free_qconv` 호출 → `clean_out`, `adc_codes`, `skip_mask`.
5. dump (1,1,OH,OW,64) 에서 `valid_cols`만 select하고 transpose해 `(oc_size, OH, OW)` 형태로 정렬 — clean과 channel-by-channel 매칭됨.
6. `observed = signed_int16(dump_selected - clean)`.
7. `compute_predicted_stats` 호출 → `E_pred, Var_pred, range_min, range_max, mode_pred`.
8. `in_range`: observed가 [range_min, range_max] 안에 들어가는지 element-wise.
9. per-atomic 누적.

### 출력 표 컬럼 의미

`diagnose_noise_per_qconv.py`는 source별로 `[chip]`, `[pysim]` 표를 따로 출력한다. 기본 `--compare chip`에서는 chip 표만 나온다.

#### Per-atomic summary

각 atomic qconv 함수 단위의 통계다. `tvmgen_default_imcflow_main_15`처럼 TVM codegen 함수 하나가 한 행이다.

| column | 의미 |
|---|---|
| `func` | TVM codegen된 atomic qconv 함수명. |
| `orig` | 원래 ResNet layer 이름. 예: `weight3_0`은 `layer2.block_int16.downsample.1.weight`에 해당하는 downsample 1x1 conv. |
| `core` | 해당 atomic이 배치된 IMCE core `(h_id, w_id)`. `w_id`는 `concat_per_core.json`과 맞춘 1-based 값. |
| `n_elem` | 비교한 output element 수. 대략 `n_samples * oc_size * OH * OW`. |
| `obs_mean` | 관측 noise 평균. `observed = signed_int16(chip_dump - noise_free_qconv)`의 평균. |
| `obs_std` | 관측 noise 표준편차. |
| `pred_E` | CSV noise model로 예측한 noise 평균 `E_pred`의 평균. 각 element에서 16개 `(abit,wbit)` contribution을 합산한 값. |
| `pred_sd` | CSV noise model로 예측한 표준편차 `sqrt(Var_pred)`의 평균. |
| `mode%` | `observed == round(predicted_mode_sum)` 비율. py_runner greedy noise sanity check에서는 높아야 하지만, chip/random noise에서는 높을 필요가 없다. |
| `inrng%` | `observed`가 CSV support 기반 predicted range `[pred_range_min, pred_range_max]` 안에 들어간 비율. 핵심 hit-rate 지표. |
| `|z|>3%` | `z = (observed - pred_E) / sqrt(max(pred_Var, 1.0))`가 3 sigma 밖인 비율. model variance/bias mismatch 지표. |

해석:
- `inrng%`가 낮으면 CSV의 min/max support가 실제 chip noise를 감싸지 못한다.
- `bias = obs_mean - pred_E`가 큰데 `inrng%`는 높을 수 있다. 이 경우 support는 충분하지만 평균 모델이 밀려있다는 뜻이다.
- `obs_std`가 `pred_sd`보다 훨씬 크고 `|z|>3%`가 높으면 CSV variance가 실제 chip 변동성을 과소평가한다.

##### `mode%` 상세 해석

`mode%`는 CSV noise model에서 **가장 가능성이 높은 단일 noise 값**과 관측 noise가 정확히 일치했는지 보는 지표다.

```text
mode% = mean(observed == round(predicted_mode_sum))
```

`predicted_mode_sum`은 각 `(abit, wbit)` contribution마다 CSV에서 probability가 가장 큰 `diff_bin`을 고르고, `PSTEP * W_SCALE[wbit] * (1 << abit)` scale을 반영해 모두 더한 값이다.

예를 들어 어떤 CSV row가 다음 분포를 가진다면:

```text
diff_bin: -10  -6  -2   2
prob:     0.4 0.3 0.2 0.1
```

해당 contribution의 mode는 `-10`이다. 이런 mode contribution 16개를 합산한 값이 `predicted_mode_sum`이다.

해석:
- `pysim --noise-mode greedy`에서는 mode noise를 deterministic하게 inject하므로 `mode%`가 거의 100%여야 한다. 이 경우 `mode%`는 lookup/mapping sanity check다.
- 실제 chip이나 random noise sampling에서는 `mode%`가 낮아도 이상하지 않다. 분포에서 sample된 값이 항상 mode일 필요는 없기 때문이다.
- chip 분석에서 `mode%`는 핵심 지표라기보다 “noise가 mode 근처에 얼마나 몰려 있는지” 보는 보조 지표다.
- 분포가 넓거나 multi-modal이면 model이 맞아도 `mode%`는 낮을 수 있다.

##### `|z|>3%` 상세 해석

`|z|>3%`는 관측 noise가 CSV가 예측한 평균/분산 기준으로 얼마나 극단적인지 보는 지표다.

```text
z = (observed - pred_E) / sqrt(max(pred_Var, 1.0))
|z|>3% = mean(abs(z) > 3)
```

여기서:
- `observed`: chip에서 실제 관측한 noise
- `pred_E`: CSV model이 예측한 평균 noise
- `pred_Var`: CSV model이 예측한 noise variance
- `sqrt(pred_Var)`: 예측 표준편차

예:

```text
observed = -120
pred_E   = -100
pred_sd  = 10
z        = (-120 - -100) / 10 = -2
```

이 값은 2 sigma 차이라 `|z|>3` outlier가 아니다.

반면:

```text
observed = -150
pred_E   = -100
pred_sd  = 10
z        = -5
```

이 값은 5 sigma라 outlier로 카운트된다.

해석:
- model이 평균과 분산까지 잘 맞고 noise가 대략 정규분포와 비슷하면 `|z|>3%`는 약 `0.27%` 근처가 된다.
- `|z|>3%`가 높으면 보통 `pred_E`가 실제 평균에서 밀려 있거나, `pred_Var`가 실제보다 너무 작다는 뜻이다.
- `inrng%`가 높아도 `|z|>3%`가 높을 수 있다. 이 경우 CSV의 min/max support는 넓어서 range 안에는 들어오지만, 확률질량 중심이나 분산이 실제와 안 맞는다는 뜻이다.

요약:

```text
mode%
  "가장 가능성 높은 단일 예측값과 정확히 맞았나?"
  greedy pysim sanity check에 특히 유용.

|z|>3%
  "예측 평균/분산 기준으로 너무 극단적인 관측값이 얼마나 많나?"
  chip에서 model bias/variance mismatch를 보는 데 유용.
```

#### Per-orig-conv summary

같은 원본 layer에 속한 atomic들을 합쳐서 집계한 표다.

| column | 의미 |
|---|---|
| `orig_conv` | 원래 ResNet layer 이름. |
| `n_elem` | 해당 layer 전체에서 비교한 output element 수. |
| `obs_mean` | layer 전체 관측 noise 평균. |
| `obs_std` | layer 전체 관측 noise 표준편차. |
| `bias` | `(observed - pred_E)`의 평균. 양수면 chip noise가 모델 평균보다 크고, 음수면 더 작다. |
| `mode%` | layer 전체 mode prediction exact match 비율. chip 분석에서는 보조 지표. |
| `inrng%` | layer 전체 predicted range hit-rate. |
| `|z|>3%` | layer 전체 3-sigma outlier 비율. |
| `chi2/n` | `mean(z^2)`. 대략 1이면 mean/variance가 잘 맞고, 클수록 bias 또는 variance underestimate가 있다는 뜻. |

`chi2/n`은 `inrng%`와 보는 관점이 다르다.
- `inrng%`: support가 실제 noise를 감싸는지 보는 min/max 지표.
- `chi2/n`: 평균과 분산까지 확률적으로 맞는지 보는 지표.

#### Skip/no-skip diagnostic per-atomic

`--skip-diagnose`를 켰을 때만 나온다. acc-mode skip path가 in_range에 미치는 영향을 보기 위한 비교표다.

| column | 의미 |
|---|---|
| `func` | atomic qconv 함수명. |
| `orig` | 원래 ResNet layer 이름. |
| `n_elem` | 비교 element 수. |
| `skip_aff%` | 해당 output element가 하나 이상의 activation bitplane에서 `popcount < 8` skip 영향을 받은 비율. |
| `strict%` | 현재 설정의 skip-aware model로 계산한 `inrng%`. 기본 summary의 `inrng%`와 같은 의미. |
| `noskip%` | `acc_mask=15`처럼 no-skip이라고 가정해 range를 계산한 뒤, clean output 차이만큼 shift해서 비교한 hit-rate. |
| `both%` | strict와 noskip 가정 모두에서 in-range인 비율. |
| `neither%` | strict와 noskip 가정 모두에서 out-of-range인 비율. skip modeling으로도 설명되지 않는 mismatch 후보. |
| `delta_mu` | `clean_no_skip - clean_skip` 평균. skip path가 noise-free output을 평균적으로 얼마나 이동시키는지. |
| `delta_sd` | `clean_no_skip - clean_skip` 표준편차. |

해석:
- `skip_aff%`가 높고 `strict% > noskip%`면 현재 skip-aware modeling이 도움이 된다는 뜻.
- `noskip% > strict%`면 실제 chip 동작이 no-skip 쪽에 더 가까울 가능성을 의심한다.
- `neither%`가 높으면 skip 여부만으로는 설명이 안 되고, CSV 분포 자체나 checkpoint/input/mapping mismatch를 봐야 한다.

#### Skip/no-skip diagnostic per-orig-conv

위 per-atomic skip/no-skip 지표를 원본 layer 단위로 합친 표다. 컬럼 의미는 per-atomic과 동일하다.

### `print_summaries` (line 462~531)

label (chip/pysim)별로 per-atomic / per-orig_conv 표 출력.

- Per-atomic summary: obs (mean/std), pred (E/sd), mode%, inrng%, |z|>3%.
- Per-orig_conv summary: atomic들 합쳐서 같은 지표 + bias (= obs - pred_E의 평균) + chi2/n.
- `z = (observed - E_pred) / sqrt(max(Var_pred, VAR_FLOOR))`. `VAR_FLOOR=1.0`은 pred_Var=0 element (모든 contribution이 deterministic CSV row를 hit)에서 z가 발산하는 걸 막기 위함.
- `chi2/n = mean(z²)`. ≈ 1이면 perfect fit, 클수록 model이 variance/bias를 underestimate.
- `mode%` = `(observed == predicted_mode_sum).mean()`. py_runner가 greedy mode일 때만 의미 있음 (deterministic이라 ~100% 기대). alias/sample 모드면 random이라 ~0%가 정상.

### main (line 534~622)

1. args 파싱 + 데이터 로드 (atomic, layout, CSV, ckpt, weights). `--ckpt-path`가 있으면 우선 사용하고, 없으면 `--ckpt-alias`를 `/root/project/CIM/checkpoints/<board>_<vmode>.json`에서 resolve한다.
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

### 4. int16 wrap 주의점
- `noise_free_qconv`는 per-abit contribution과 final output에 int16 wrap을 적용한다.
- `observed = signed_int16(chip_dump - clean)`으로 계산한다. 단순 int32 subtraction을 쓰면 wrap 경계 근처에서 작은 noise가 `+/-65536` outlier처럼 보인다.
- predicted range는 noise contribution을 선형 합산한다. noise 자체가 int16 wrap 경계를 넘을 정도로 크면 first-order 근사가 깨질 수 있지만, 일반적인 atomic noise 규모에서는 문제가 작다.

### 5. checkpoint mismatch
- chip debug dump를 만든 binary/checkpoint와 diagnose가 사용하는 checkpoint가 다르면 `observed = chip - clean`이 noise가 아니라 model mismatch까지 포함한다.
- 현재는 `--ckpt-alias tmp01_refine_ndis32`처럼 명시해서 맞추는 것을 권장한다.
- downsample은 weight 변화나 input quantization 차이에 민감해서 checkpoint mismatch가 특히 크게 보일 수 있다.

### 6. systematic bias
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
