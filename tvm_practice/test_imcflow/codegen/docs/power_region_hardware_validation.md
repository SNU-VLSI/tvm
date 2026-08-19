# Power region hardware validation

검증일: 2026-08-18

## GPIB3 live count capability

`meas-2`의 `imcflow` conda environment와 `DMM_GPIB3`를 사용했다. 5,000 sample,
100 us interval acquisition을 GET으로 시작한 뒤 acquisition 중 `DATA:POIN?`를
10회 호출했다.

- 관측 count 시작: `3072, 3584, 5000, ...`
- count 단조 증가: 통과
- polling 중 acquisition 유지: 통과
- 종료 후 raw metadata CSV: 5,000 samples
- DMM buffer drain: 5,000 samples
- raw/buffer count 일치: 통과
- GET 호출 bracket: 103,544,749 ns
- metadata mode 복원 및 DMM 내부 임시 파일 삭제: 통과

따라서 GPIB3에서는 `DATA:POIN?`를 power-region minimum sample 판단의
authoritative live count로 사용할 수 있다. 장비가 count를 512-sample 단위처럼
묶어서 갱신할 수 있으므로 최소 sample에는 overshoot가 생길 수 있다.

## Board TCP/C macro smoke test

기존 port 9910의 protocol-v4 daemon은 변경하지 않았다. 새 코드를 `meas-2`의
임시 port 9911에서 실행하고, `petalinux`의 `/tmp`에서 `dmm_measure.c`,
`power_region.c`와 standalone smoke program을 native GCC로 빌드했다.

- board → measurement server protocol-v5 handshake: 통과
- outer region 중 nested BEGIN: `POWER_REGION_ERR_NESTED` (`-3`)
- nested body 미실행 및 outer region 정상 END: 통과
- 다음 sequential region 시작: 통과
- `r0001_outer`: `complete`, progress query 1회
- `r0002_loop_body`: `complete`, body iteration 1회, progress query 2회
- raw/NPZ/tag/summary artifact 생성: 통과

검증용 v5 daemon은 실행 후 종료했다. 운영 중인 v4 daemon과 repository checkout은
변경하지 않았다.
