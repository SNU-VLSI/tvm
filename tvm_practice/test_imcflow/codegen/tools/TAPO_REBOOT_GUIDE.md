# Tapo 스마트플러그로 B2(petalinux2) 보드 자동 재부팅 — 세팅 가이드

wedge(SoC 행, SSH 불통) 시 사람이 폰으로 플러그를 껐다 켜는 대신,
`tools/tapo_ctl.py`가 자동으로 전원을 재인가하도록 하는 세팅입니다.

## ⚠️ 절대 규칙

- **fpga1 플러그는 절대 조작 금지** (파손 chip1 보드). 스크립트에
  allowlist(`fpga2`만 허용) + fpga1 명시적 차단이 하드코딩되어 있음 —
  이 안전핀을 제거하지 말 것.
- 계정 크리덴셜은 **파일로만** 보관 (chmod 600). 셸 히스토리/로그/커밋에
  절대 남기지 말 것. 스크립트는 600이 아니면 실행을 거부함.

## 왜 "owner 계정"이 필요한가

Tapo 클라우드 원격제어(passthrough)는 기기가 그 계정에 **등록(owner)**
되어 있어야 동작합니다. 공유(shared, role=1)로 받은 계정은 기기목록엔
보이지만 제어 요청이 `-20571 Device is offline`으로 거부됩니다 (2026-09
실측). 협업자 계정에 fpga2 플러그가 **직접 등록**되어 있으면 됩니다.

## 세팅 (5분)

1. 협업자의 Tapo 계정 크리덴셜로 크리덴셜 파일 생성 (이 서버의 root 홈):

   ```bash
   cat > ~/.tapo_creds.json <<'EOF'
   {"email": "collaborator@example.com",
    "password": "********",
    "plug_ip": "192.168.0.84",
    "device_alias": "fpga2"}
   EOF
   chmod 600 ~/.tapo_creds.json
   ```

   - `plug_ip`: 플러그의 랜 IP (Tapo 앱 > 기기 > 설정에서 확인).
     이 서버에서 그 서브넷이 안 닿으면 지워도 됨(클라우드 경로만 사용).
   - `device_alias`: 앱에 표시되는 기기 이름. 반드시 `fpga2`.

2. 동작 확인 (반드시 status부터! on/off는 보드 전원을 실제로 끊음):

   ```bash
   cd <codegen>; python3 tools/tapo_ctl.py status
   ```

   - `[cloud] fpga2: status=1 ... role=0` → owner 등록 OK, 제어 가능.
   - `role=1` 에러 → 그 계정엔 공유만 된 것. owner 계정 필요.
   - 로컬 경로(`[local] ...`)가 나오면 KLAP 직결이 된 것 (가장 확실).

3. (보드가 놀고 있을 때) 실제 재부팅 1회 테스트:

   ```bash
   python3 tools/tapo_ctl.py reboot     # off → 5초 → on
   # 이후 보드 부팅 ~2-4분, ssh -p 1326 root@147.46.117.99 로 확인
   ```

## 네트워크가 안 닿을 때 (릴레이)

이 dev 서버에서 플러그 서브넷(192.168.0.x)이 안 닿고 클라우드도 막히면,
플러그와 같은 망에 있는 아무 리눅스 호스트를 릴레이로 사용:

```bash
# 릴레이 호스트에도 이 스크립트와 ~/.tapo_creds.json(chmod 600)을 복사한 뒤
export TAPO_RELAY="user@relay-host"
python3 tools/tapo_ctl.py reboot   # ssh로 릴레이에서 실행됨
```

## 스윕 자동복구 연동

`experiments/shmoo_uv_cells.py`는 WEDGE 감지 시 기본은 정지(사람 재부팅
대기)이지만, 아래처럼 켜면 자동으로 전원 재인가 → SSH 대기(최대 7분) →
정규화(clk 100MHz → program_scan_reg → warmup → V1 preset → fingerprint
46±mA/149len 게이트) → 다음 셀 계속:

```bash
export TAPO_AUTO_REBOOT=1
PYTHONPATH=/root/project/tvm/3rdparty/measurement_utils \
  python3 -u experiments/shmoo_uv_cells.py
```

fingerprint까지 통과 못 하면 그때는 정지하고 사람을 부릅니다.
재부팅 직후 첫 1-2회 커널 실행은 cold-boot transient로 실패할 수 있어
fingerprint는 4회까지 재시도하게 되어 있음.

## 명령 요약

| 명령 | 동작 |
|---|---|
| `tapo_ctl.py status` | 플러그 상태 조회 (안전) |
| `tapo_ctl.py off` / `on` | 전원 차단/인가 (보드 강제단전 주의) |
| `tapo_ctl.py reboot` | off → 5초 → on |

제어 경로: ① 로컬 KLAP(`tapo` 라이브러리, plug_ip 필요) → ② 클라우드
passthrough(owner 계정 필요) 순서로 시도, 둘 다 실패하면 원인별 에러 출력.
