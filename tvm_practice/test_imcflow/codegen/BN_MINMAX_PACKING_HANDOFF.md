# Handoff: qconv IMCE에 BN/minmax(+add) postop packing 확장 탐구

2026-08-13 작성. region mapping 시각화(`_rtllog/region_layer_map_*.png`)에서 확인된 사실:
ResNet8은 active IMCE 29개 중 ~14개, DS-CNN은 24개 중 ~10개가 **conv 본체가 아닌
standalone bn / minmax / add / vecops 전용**으로 소모된다. 이들을 qconv IMCE 안으로
pack하면 IMCE 회수 → region당 더 많은 레이어 탑재(pipeline 깊이↑) 또는 더 큰 모델 매핑이
가능해진다. 이 문서는 다음 세션이 이어받기 위한 현황·코드 지도·탐구 방향이다.
**"불가능"으로 못박힌 결론은 없다** — 아래는 현재 코드가 그렇게 하지 않는 이유이지,
원리적 불가능의 증명이 아니다.

## 현황 (근거: file:line)

- postop fusion 패턴 자체는 존재: `imcflow.qconv2d-with-postop` = qconv + {bias_add,
  add, relu, divide, multiply}의 ≤9-op 선형 체인 (`python/tvm/relay/op/contrib/imcflow.py:651-733`).
- **BN·minmax는 패턴에서 의도적으로 제외**되어 있다: `imcflow.py:699-702`에 주석 처리로
  제거 + 사유 주석("BN is separated from Conv composite and integrated with min_max_quant
  as imcflow.bn-minmax"). `transform.py:1059-1061`도 동일 사유: "BN은 channel-wise라
  concat 후 적용" → `post_op_candidates = [bias_add, relu]`.
- 근본 동기 = **OC ≤ 64 어레이 상한** (`imce_codeblock.py:1177` `assert out_channels <= 64`):
  OC가 크면 conv가 atom-split되고, 현 설계는 BN을 atom concat 이후 별도 노드로 뺐다.
- 대신 BN+minmax를 **다음 conv의 producer-side preop**(`imcflow.preop-minmax`,
  `imcflow.py:570-587,725`)으로 묶는 설계.
- residual 2-입력 add/vecops는 converging 패턴(`imcflow.py:527-609`)으로 별도 IMCE.
- **자원(imem/FIFO)은 blocker가 아님이 확인됨**: ConvBlock은 same-IMCE post_ops 렌더링을
  이미 지원 (`imce_codeblock.py:1164-1403`).
- minmax는 2-pass 아님: min/max가 **상수 피연산자**인 스트리밍 clip-quantize
  (`qnn.py:1323`, `imcflow.py:450 is_constant`) → 스트리밍 fusion에 의미론적 장벽 없음.

## 핵심 탐구 방향 (사용자 통찰 포함)

1. **★OC-split atom 내부 per-atom BN**: BN은 채널별 affine이므로, conv가 **out-channel
   방향으로 split**된 경우 각 atom IMCE는 자기 담당 채널의 BN 파라미터 부분집합만 받아
   **concat 전에** 자기 출력에 BN을 적용할 수 있다(채널 분할과 채널별 연산은 교환 가능).
   현 코드의 "BN은 concat 후" 규칙은 이 경우 과잉 보수적일 수 있다 — 최우선 검증 대상.
   - 확인 포인트: split이 실제로 OC-wise인지 IC-wise인지 케이스 분류
     (`transform.py:1208` in_ch_limit/out_ch_limit, `split_conv_to_atomic`
     `transform.py:1303-1315`, split/concat 재작성 `transform.py:1351-1438`).
   - **IC-split이면 다름**: atom들이 같은 out-channel의 partial sum을 나눠 만들므로
     psum 합산 완료 전 BN 불가 → 이 경우 BN을 psum-merge하는 IMCE에 fuse하는 게 목표
     (별도 IMCE가 아니라 merge 노드의 postop으로).
2. **unsplit conv에 BN(+minmax) postop 재융합** (low-risk 첫 단계):
   `make_postop_pattern_start_with`(`imcflow.py:700`)에 BN/minmax를 되살리되 no-split
   조건으로 가드 + `post_op_candidates`(`transform.py:1062`) 확장. 예상 회수: region당
   1~3 IMCE (예: ResNet8 b1.c2의 bn 전용 imce_3_1).
3. **preop-minmax의 재배치**: producer-side로 뺀 bn+minmax를 consumer conv IMCE의
   진입 단계로 옮길 수 있는지 (RECV 직후 vec op). NoC edge로 min/max/scale/bias 상수만
   옮기면 되므로 데이터 이동은 오히려 줄 수 있음.
4. residual add: 2번째 피연산자의 RECV를 conv IMCE에 추가하는 형태의 fuse —
   converging RECV의 flag/rendezvous 설계 필요 (send_recv_sync 계열, 난이도 높음).

## 검증 규율 (이 repo의 확립된 절차)

- 새 동작은 **env lever, default OFF** + OFF 시 생성물 byte-identical 확인.
- **RTL(BUGFIX-off simv, 무수정) 무회귀 먼저** → 그 다음 chip. chip 실행 처방:
  per-kernel warmup ON + `IMCFLOW_MMIO_BARRIER=100~500` (CLAUDE.md / memory 참조).
- 수치 등가 검증: `--ref-models transformed` CPU golden과 비교 (v1 a8af RTL은 golden과
  bit-exact임이 확인되어 있음 — packing 후에도 유지되어야 함).
- 시각화 재생성으로 회수 효과 확인: `_rtllog/plot_region_layer_map.py`
  (+ `_rtllog/region_layer_map.json` 재추출 필요).

## 참고 데이터

- region별 layer/IMCE/NoC 매핑: `_rtllog/region_layer_map.json`
- 표: ResNet8 layer_imce_counts = b1.c1 2 / b1.c2 3 / b1.res 1 / b2.c1 4 / b2.c2 3 /
  b2.down 3 / b3.c1 5 / b3.c2 4 / b3.down 3 / b3.res 1 (합 29).
  DS-CNN = pw1..pw4 각 6 (합 24; dwconv는 이 아티팩트에서 IMCE 상주 — imce.cpp에
  `__builtin_IMCE_DWCONV` region1 672/region2 224개 실존).
