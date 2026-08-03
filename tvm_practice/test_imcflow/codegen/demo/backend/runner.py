"""실행 소스 추상화: mock(canned txt 재생) | chip(실제 SSH).

두 소스 모두 동일한 계약을 따른다:
    iter_lines(cfg) -> Iterator[str]   # 칩 stdout 과 같은 포맷의 라인들을 순서대로 내보냄

이 라인들을 parser.ChipStreamParser 에 먹이면 mock/chip 무관하게 같은 이벤트가 나온다.
그래서 프로파일 전환은 이 파일의 분기 하나뿐이고 파서/뷰어는 그대로다.

- mock: fixtures/ 의 canned txt 를 라인별로 읽되, per-sample 경계마다 delay_ms 만큼 sleep 해
        "실시간처럼" 재생. (실제 데모에서는 sleep 없음.)
- chip: run_dataset_eval.sh 스텝6의 REMOTE_CMD 를 그대로 복제해 ssh 로 직접 실행하고,
        그 stdout 을 line-buffered 로 흘린다. 노트북에서 실연결 검증 완료(2026-07-27).
"""

import os
import time
import shlex
import subprocess
from typing import Iterator


def _mock_lines(cfg: dict) -> Iterator[str]:
    demo_root = cfg["_demo_root"]
    fixture = os.path.join(demo_root, cfg["mock"]["fixture"])
    delay = cfg["mock"].get("delay_ms", 300) / 1000.0

    with open(fixture, "r") as f:
        for line in f:
            yield line
            # per-sample 경계 = "Predicted: ... Result:" 라인. 여기서 한 박자 쉰다.
            if "Predicted:" in line and "Result:" in line:
                if delay > 0:
                    time.sleep(delay)


def _pgrep_pat(cfg: dict) -> str:
    """pgrep/pkill 용 패턴.

    첫 글자를 [] 로 감싸는 고전적 트릭. 이걸 안 하면 패턴을 담은 원격 셸 자신의
    커맨드라인이 -f 매칭에 걸려, pgrep 은 항상 자기를 세고 pkill 은 자기(=래퍼 셸)를
    죽인다. 래퍼가 죽으면 커맨드 꼬리의 clear_time+warmup 이 실행되지 않는다.
    """
    exe = cfg["chip"]["dataset_exec"]
    return f"[{exe[0]}]{exe[1:]}"


def _build_remote_cmd(cfg: dict) -> str:
    """run_dataset_eval.sh 스텝6의 REMOTE_CMD 를 그대로 복제.

    검증된 옵션 전부 보존: taskset CPU 핀, tmpfs 덤프/heartbeat 경로, 실행 후 clear_time+warmup.
    """
    c = cfg["chip"]
    n = cfg["run"]["num_samples"]
    exec_path = f"{c['binary_dir']}/build/{c['dataset_exec']}"
    warmup_dir = c.get("warmup_dir", "/home/root/imcflow/xilinx/petalinux-csrc")
    # ⚠️ 추론 "전" warmup 필수: IMC 아날로그 코어는 warmup 으로 깨워야 정상 연산한다.
    # 안 하면 (칩 리부트/미warm 상태에서) 에러 없이 전부 class 0 만 예측한다(조용한 실패).
    # run_dataset_eval.sh 는 warmup 을 추론 "후"에만 돌려 *다음* 실행을 준비하므로,
    # 여기서 추론 앞에 clear_time+warmup 을 한 번 더 넣어 노트북 단독 실행도 안전하게 만든다.
    pre = f"cd {warmup_dir} && make clear_time && make warmup > /dev/null 2>&1"
    return (
        f"{pre}; "
        f"cd {c['remote_base_path']} && "
        f"IMCFLOW_DEBUG_DUMP_DIR={c['debug_dump_dir']} "
        f"IMCFLOW_HEARTBEAT_PATH={c['heartbeat_path']} "
        f"taskset -c {c['eval_cpu']} {exec_path} "
        f"{c['graph_path']} {c['params_path']} "
        f"{c['images_path']} {c['labels_path']} "
        f"{n} {c['remote_result_path']}; "
        f"cd {warmup_dir} && make clear_time && make warmup > /dev/null 2>&1"
    )


def _ssh_argv(cfg: dict) -> list:
    """scan_steps.sh 의 scan_ssh_once 조립을 그대로 미러한 ssh 인자 앞부분."""
    c = cfg["chip"]
    argv = ["ssh"]
    if c.get("auth_method", "key") == "key":
        argv += ["-o", "BatchMode=yes"]
    return argv + ["-p", str(c["ssh_port"]), f"{c['ssh_user']}@{c['ssh_host']}"]


def kill_remote(cfg: dict, timeout: float = 30.0) -> bool:
    """원격에 남은 추론 프로세스를 정리한다.

    로컬 ssh 클라이언트를 terminate 해도 원격 프로세스는 살아남을 수 있다(그대로 두면
    다음 실행과 칩에서 경합 → SoC wedge). stop 경로에서 반드시 같이 호출할 것.
    """
    # [e]xecute... 패턴이 아니면 pkill 이 자기를 담은 래퍼 셸까지 죽여,
    # 커맨드 꼬리의 clear_time+warmup 이 실행되지 않는다.
    argv = _ssh_argv(cfg) + ["-o", "ConnectTimeout=10", f'pkill -f "{_pgrep_pat(cfg)}"']
    try:
        # pkill 은 죽일 게 없으면 exit 1 — 실패가 아니므로 returncode 는 보지 않는다.
        subprocess.run(argv, timeout=timeout, capture_output=True)
        return True
    except Exception:
        return False


def _chip_lines(cfg: dict, on_proc=None) -> Iterator[str]:
    """실제 칩 SSH 실행. stdout 을 line-buffered 로 흘린다.

    on_proc: Popen 을 넘겨받을 콜백. 호출자(app._RunHub)가 이 핸들로 중간 정지를 건다.
             핸들을 밖으로 안 빼면 제너레이터 안에 갇혀 stop 을 걸 수 없다.
    """
    remote_cmd = _build_remote_cmd(cfg)
    ssh_argv = _ssh_argv(cfg) + [remote_cmd]

    proc = subprocess.Popen(
        ssh_argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,  # line-buffered
        text=True,
    )
    if on_proc is not None:
        on_proc(proc)
    try:
        for line in proc.stdout:  # 도착하는 대로 (sleep 없음 — 칩 실제 추론 속도 그대로)
            yield line
    finally:
        proc.stdout.close()
        proc.wait()


def iter_lines(cfg: dict, on_proc=None) -> Iterator[str]:
    profile = cfg.get("profile", "mock")
    if profile == "mock":
        return _mock_lines(cfg)     # mock 은 프로세스가 없어 on_proc 무의미 (루프에서 정지)
    elif profile == "chip":
        return _chip_lines(cfg, on_proc=on_proc)
    raise ValueError(f"unknown profile: {profile!r} (expected 'mock' or 'chip')")


def remote_cmd_preview(cfg: dict) -> str:
    """디버그/가이드용: chip 모드에서 실제로 나갈 ssh 커맨드 문자열."""
    c = cfg["chip"]
    remote_cmd = _build_remote_cmd(cfg)
    batch = "-o BatchMode=yes " if c.get("auth_method", "key") == "key" else ""
    return (
        f"ssh {batch}-p {c['ssh_port']} {c['ssh_user']}@{c['ssh_host']} "
        f"{shlex.quote(remote_cmd)}"
    )
