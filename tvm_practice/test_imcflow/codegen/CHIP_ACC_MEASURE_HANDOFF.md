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

- [ ] `vgg11_cifar_rnd --stop-at transform`로 partition 결과 확인 (region 개수/구성)
- [ ] VGG-11 전체 컴파일 → utilization 실측 → REPORT §10 갱신
- [ ] R50: layer4 3×3 blocker 재현 → spill vs CPU-fallback 결정
- [ ] (acc 목표 시) 실제 quantized checkpoint 학습/변환 경로 마련 (random weight 대체)
