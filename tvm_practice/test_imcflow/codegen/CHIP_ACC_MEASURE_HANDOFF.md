# Handoff: VGG-11 / ResNet-50 컴파일 → IMC bitcell utilization 실측

2026-08-24 작성. 브랜치 `chip_acc_measure`, 커밋 `6f3b10410` (모델 정의) + `6125e9b42`
(ResNet-50 매핑 경계 수정). Push 안 됨. 이 문서는 다음 세션이 컴파일·디버깅을
이어받기 위한 현황·실행법·예상 blocker·측정 목표다.

## 왜 하는가 (배경)

PIMCA/Verma/DIANA/imcflow 4칩 bitcell utilization 비교 연구
(`/root/project/pim_soc_comparison/analysis/REPORT.md`, 재현 스크립트 `model.py`).
ResNet8/KWS는 imcflow **실제 codegen 산출물** 기반으로 계산했지만(ops 3.42%/3.12%),
Verma 논문의 두 검증 모델(VGG-11, ResNet-50)에 대한 imcflow 수치는 **가상 매핑**이다:

- VGG-11: U = 60.8% (layer당 atom ≥10이라 스케줄 방식에 둔감)
- ResNet-50: layer-순차 하한 **26.6%** ~ region packing 상한 **66.3%** (cap=16 atoms),
  현실적 NoC 제약(cap=8) 가정 시 **34.2%** — 실제 값은 컴파일해서 region이 실제로
  어떻게 잡히는지 봐야 결정됨 (REPORT §10).

목표: 실제 컴파일 → region/atom 매핑 dump 추출 → 가상 수치를 실측으로 교체.

## 현재 상태 (완료된 것)

- `tvm_practice/models/vgg11_cifar_imcflow.py` — Verma JSSC'22 Table V 사양
  (conv3×3: 128ch×4 @32², pool, 256ch×2 @16², pool, 256ch×2 @8² → dense 1024/1024/10).
- `tvm_practice/models/resnet50_imagenet_imcflow.py` — torchvision 토폴로지, 224×224.
  stem(conv7×7 s2)+maxpool은 CPU float → 첫 매핑 conv는 56×56.
- `test.py` MODEL_REGISTRY 등록: `vgg11_cifar_rnd`, `vgg11_cifar_rnd_small`(8×8),
  `resnet50_imagenet_rnd`, `resnet50_imagenet_rnd_small`(56×56).
- 패턴: `resnet8_cifar.py`/`ds_cnn_imcflow.py`와 동일 (`imcflow_min_max_quantize →
  imcflow_qconv2d → imcflow_batch_norm`, `(IRModule, params_dict)` 반환).
- **Weight = random int8** (`resnet8_cifar._rand_tensor` 범위 재사용). imcflow-quantized
  checkpoint 없음 + 이 환경에 torch/torchvision 미설치. utilization 측정에는 무영향,
  **정확도(acc) 측정에는 추후 실제 checkpoint 필요**.
- Smoke 통과: 두 모델 relay build + InferType OK (출력 (1,10)/(1,1000), qconv 7/52개).
  전체 codegen/PnR은 **아직 안 돌림**.

## 실행법

```bash
cd /root/project/tvm/tvm_practice/test_imcflow/codegen   # direnv 환경 필요
python main.py --list-models                              # 4개 키 확인
python main.py --model vgg11_cifar_rnd_small              # 가장 싼 smoke
python main.py --model vgg11_cifar_rnd --stop-at transform  # partition까지만
python main.py --model vgg11_cifar_rnd                    # VGG 전체 (1차 목표)
python main.py --model resnet50_imagenet_rnd_small        # R50 축소판
python main.py --model resnet50_imagenet_rnd              # R50 전체 (blocker 예상)
```

환경 주의: tvm이 있는 인터프리터는 direnv가 잡아주는 `/root/anaconda3/bin/python3.11`
(`python`). 맨 `python3`은 의존성 없음. `models/__init__.py`가 `deep_autoencoder`를
eager import(tensorflow 필요)하므로 순정 env에서 단독 import 시 우회 필요.

## 예상 blocker (file:line)

1. **3×3 IC=512의 IC-chain 19 > 16 IMCEs** — `atom_IC = floor(256/9) = 28`
   (`transform.py:1189`, cost `transform.py:3515-3521`), ResNet-50 layer4의 3×3 512ch가
   `ceil(512/28)=19` atom. psum chain이 16 IMCE mesh를 초과 → PnR(`joint_pnr_ilp.py`)
   실패 예상. 해결 후보: IC를 2-pass로 나누고 partial-sum을 메모리에 spill(현
   codegen에 없는 기능), 또는 512ch 3×3만 CPU fallback.
2. **ConfigData assert H,W ≤ 128** (`acim_util.py:224`). 현 구성은 stem+maxpool을
   CPU에 둬서 매핑 conv가 전부 56/28/14/7이라 발동 안 함. 경계를 112×112까지
   앞당기는 것도 합법(≤128). `_cfg()`는 clamp가 아니라 assert로 강제해 둠
   (`6125e9b42`) — 경계를 잘못 당기면 조용한 오매핑 대신 즉시 실패.
3. **대형 dense는 CPU행** (VGG 16384→1024, R50 2048→1000) — resnet8/ds_cnn과 동일
   정책이라 blocker는 아니지만 host binary 쪽 메모리/시간 확인 필요.
4. VGG L2-L8은 atom 10~40개/layer라 **한 layer도 단일 region에 안 들어감** →
   region 분할이 어떻게 잡히는지 자체가 관찰 대상 (ResNet8은 region당 2~6 atoms였음).

## 컴파일 성공 후 추출할 것 (utilization 실측 절차)

eval_dir 산출물에서 (ResNet8 때 사용한 파일과 동일):
`15_with_mappings.txt`(atom→IMCE, IC/OC split), `active_imce_list.txt`,
`func_map.txt`(region 수), `noc_visualizations/region_layer_map.json`(region별
layer/IMCE, psum edges).

이를 `/root/project/pim_soc_comparison/analysis/model.py`의 `imcflow_report()`
형식(region = (이름, [Layer…]) 목록)에 넣으면 ops/spatial U가 나온다. 비교 기준:
region당 실효 atom 수가 cap=8 근처면 R50 ≈ 34%, cap=16 근처면 ≈ 66% (REPORT §10).
BN/minmax/add 전용 IMCE가 region 자리를 얼마나 먹는지는
`BN_MINMAX_PACKING_HANDOFF.md`의 관찰(ResNet8: active 29 중 ~14가 standalone
postop)과 교차 확인 — postop packing이 되면 region당 conv atom이 늘어 U 상승 여지.

## 다음 단계 체크리스트

- [x] `vgg11_cifar_rnd --stop-at transform`로 partition 결과 확인 (2026-08-24 통과)
- [x] VGG-11 utilization 실측 → REPORT §10 갱신 완료 (**U = 56.25%**, 가상 60.8% 대체)
- [x] VGG-11 codegen 완주 (**BUGFIX-on(chip 모델) CODEGEN_OK**, 272개 IMCE imem
  오브젝트 + CPU relay.build까지 완료, eval_dir `vgg11_cifar_rnd_evl.baremetal`).
  codegen 단계 추가 fix 3건:
  1. `imce_codeblock.py` VecBlock `_build_var_ins`: composite 내부 단항 op(standalone
     bn→relu vecops)의 operand가 빈 문자열로 방출되던 것(`MAXI(, 0)`) → 내부
     producer/prev_op 폴백 추가.
  2. `imce_operation_handlers.py` + `imce_codeblock.py`의
     `consumer_is_non_multicast_split` 2곳: 출력 엣지 1개 assert → multicast(OC-split
     공유 minmax)면 False 반환으로 완화.
  3. `imce_codeblock.py` `_render_a8af` SEND 병합: policy 없는 local edge가 섞인
     multicast에서 IndexError → routed(policy 보유) edge만으로 주소 dedupe.
- [ ] **BUGFIX-off codegen은 sync 검증 FAIL** — OC-split multicast 엣지(uuid=10,
  producer 0 vs consumer 1055, region1 10개 엣지)에 rendezvous 미방출 →
  lost-wakeup deadlock 검출. BUGFIX-off RTL cosim을 하려면 send_recv_sync가
  OC-split/multicast 토폴로지를 지원해야 함(별도 작업). 급하면
  `IMCFLOW_SKIP_SYNC_ASSERT=1` 우회 가능. ※utilization 측정에는 불필요.
- [ ] R50: layer4 3×3 blocker 재현 → spill vs CPU-fallback 결정
  → **모델 레벨 IC-split(psum add)로 해소**(resnet50_imagenet_imcflow.py `_split_plan`).
  transform은 PnR 직전까지 도달하나 **PnR Infeasible 5연속** (아래 진행로그 참조).

## 2026-08-25~27 진행 기록: ResNet-50 bring-up (PnR Infeasible 반복 중)

**컴파일러 fix (libtvm 리빌드 필요)**: merge_compiler_regions.cc에 3건 —
RegionMerger 재진입 가드(무한재귀 16.5M프레임 방지), null-region skip,
glue(Tuple/TGI) 너머 parent region을 merge-restriction으로 등록.
**python fix**: CPU split layout rule NCHW 전용(packed 텐서에 raw sections 방지),
partitionRound에 `IMCFLOW_REGION_CAP`(packing 상한) / getCost에
`IMCFLOW_NOCOST_OP_COST`(concat/split cost) env 노브(기본값=기존과 동일).
**모델**: `_split_plan`(OC/IC-split, concat-tree 반영 budget), 블록/chunk마다
host cast 경계(int16→fp32→int16; 통짜 그래프의 merger/partitioner 폭주 방지),
심층그래프는 `_chip_acc_logs/run_deep_model.py`(512MB 스레드스택)로 실행.

**PnR Infeasible 5연속 시도** (각 2~3h, 로그 `_chip_acc_logs/r50small_transform_try*.log`):
| try | 조치 | 실패 region | 구성 |
|---|---|---|---|
| 1 | (기본) | main_110 r3 | 8atoms+2bn+**region내 concat트리3** |
| 2 | chunk마다 host concat | main_362 r1 | **13atoms**+concat7 |
| 3 | chunk budget에 concat-tree 반영(≤14) | main_187 r3 | 8+bn+split+concat=11 (r2엔 **8+8=16 packing**) |
| 4 | REGION_CAP=13 | main_287 r1 | 9+2bn+mm+**concat4**=16 placeable |
| 5 | NOCOST_OP_COST=1 | main_153 r1 | 8+2bn+mm+concat3=**14 placeable** |

**try5 region 크기 분포**(placeable 노드 기준): 11개=83, 13개=49, 14개=11,
**16~18개=12** — cost 회계를 정직하게 해도 supernode 내부 등 미계상 노드가
남아 partition이 PnR 한계(≈13)를 넘는 region을 계속 만든다. 반복당 2~3h가
드는 full-rerun 방식은 비효율 → 다음 단계는 (a) 실패 region만 PnR 단독
재실행하는 fast-repro 하네스, (b) "14 placeable이 왜 infeasible인가"의 ILP
제약 수준 root-cause, (c) partition cost를 placeable 기준으로 전면 정직화 중 택일.
- [ ] (acc 목표 시) 실제 quantized checkpoint 학습/변환 경로 마련 (random weight 대체)

## 2026-08-24 진행 기록: VGG-11 transform 전체 통과 + utilization 실측

`vgg11_cifar_rnd` transform(partition→PnR→NoC) **PASS**. blocker 5개를 순차 해결:

1. **"Cost of node is too high"** (transform.py:4024, cost=ceil(IC/28)×ceil(OC/64)>16):
   모델 레벨 OC-split 도입 — `models/vgg11_cifar_imcflow.py:_oc_chunks()`가 chunk당
   cost ≤ **13**이 되게 OC를 분할(qconv→bn→relu per chunk 후 concat, real_model.py
   관용구). budget 16이 아니라 13인 이유: 15-atom chunk + postop은 **PnR ILP
   Infeasible** (16/16 IMCE 포화 + 라우팅 제약).
2. **`nn.max_pool2d` CPU layout rule 없음** (layout.py:932) → CPU_REQUIRED_OP_LAYOUTS에
   NCHW 항목 추가 (avg_pool2d와 동일).
3. **blocked↔QCONV_INPUT 변환 없음** → `_convert_layout`에 NCHW 경유 bridge 2방향 추가.
4. **concat layout rule 오선택**: OC-split chunk region 출력들이 (NCHW, NHWC64C) 혼합
   tuple로 concat에 들어올 때 `_select_target_rule`의 all-blocked 검사에 안 걸려
   폴백 `[QCONV_INPUT]`(첫 규칙) 선택 → nn.bitpack 오삽입 → type mismatch. 혼합
   tuple(blocked+NCHW)도 NCHW 옵션을 선호하도록 조건 확장.
5. **빈 identity region 함수** (body==param; host concat 전용 NO_COST region 주변에
   PartitionGraph가 생성) → PnR에 배치 대상이 없어 NoC 경로 구성이
   `src_graph_id not found` KeyError. `prune_identity_region_funcs()` 신설,
   driver의 partitionRound 직후 호출.

**실측 매핑**: 17 regions × 정확히 10 conv atoms(+BN/minmax 1-3, 총 170 atoms).
cross-layer packing 없음(10+10>16). **U = 56.25%** (= 10/16 IMCE × atom 채움 90%),
`analysis/model.py:imcflow_vgg11_regions_measured()`로 재현. REPORT §10 갱신됨.
R50 시사점: 실제 region cap은 ≈10-13 atoms → R50 실측 예상은 34% 근방(cap=8 가정치).

### 왜 VGG에서 multi-layer region packing이 안 되는가 (구조 분석, 2026-08-24)

- IC-chain(psum 인접 연쇄)은 region 내에 통째로 있어야 함(IC 분할=psum spill,
  codegen 미지원) → 최소 배치 단위 = chain × 64-OC group.
- 3×3의 atom_IC=floor(256/9)=28이 2^n 채널과 어긋나는 것이 근본 원인:
  IC 64/128/256/512 → chain 3/5/10/19. **256-IC 층은 최소 단위가 10 atoms**라
  10+10>16 → 어떤 조합도 불가, 10/16=62.5% 점유(×채움90%=56.25%)가 구조적 상한.
- 128-IC 층은 이론상 10+5=15 조합이 가능하나 **15 atoms+postop = PnR ILP
  infeasible** (실측; 그래서 chunk budget 13). IMCFLOW_PACK_BN_MINMAX로 postop
  흡수 시 15-atom region 성립 여지(128-IC 단계 한정, 이득 제한적).
- region 16/16을 채우는 채널 폭은 112/224/448(chain 4/8/16)뿐 — 보편 모델 아님.
  1×1 conv는 atom_IC=256이라 완벽 정렬(R50의 1×1들이 atom 100% 채움인 이유).
- 표준 VGG-16/19는 512ch 3×3(chain 19>16) 때문에 아예 컴파일 불가 층 포함
  (R50 layer4와 동일 blocker).
