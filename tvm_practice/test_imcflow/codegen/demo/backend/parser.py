"""칩 stdout 라인을 한 줄씩 먹여 per-sample 이벤트를 뽑는 스트리밍 파서.

칩 바이너리(execute_graph_for_dataset)가 stdout 으로 내보내는 라인 형태:

    [Sample 0] Scores: [-4.9182, -4.3116, ..., -6.6785]
             Predicted: 3, Label: 3, Result: O
    Progress: 1/500, Correct: 1/1, Failed: 0, Accuracy: 100.00%

- `[Sample N] Scores: [...]` + 다음 줄 `Predicted: p, Label: l, Result: O/X` 가 한 샘플.
  (canned txt / mock 재생에서는 이 두 라인이 그대로 있고, 실제 chip 은 바이너리 소수정으로
   Scores 라인을 stdout 에도 흘리게 한 뒤 동일 포맷으로 온다. README 참조.)
- `Progress:` 는 칩 자체 집계 — 완료 신호 + 최종 크로스체크로만 쓴다 (화면 메인 아님).

이 파서는 상태를 최소로 들고(직전 [Sample] 라인 대기), 라인 하나가 샘플을 완성할 때마다
dict 이벤트를 yield 한다. softmax/top-5/running-accuracy 같은 파생 계산은 하지 않는다
(그건 뷰어/프론트 몫). 파서는 raw logit 과 pred/label/correct 만 넘긴다.
"""

import re
from typing import Optional

_SAMPLE_RE = re.compile(r"^\[Sample (\d+)\]\s*Scores:\s*\[([^\]]*)\]")
_PRED_RE = re.compile(r"Predicted:\s*(-?\d+),\s*Label:\s*(-?\d+),\s*Result:\s*(\w+)")
_PROGRESS_RE = re.compile(
    r"^Progress:\s*(\d+)/(\d+),\s*Correct:\s*(\d+)/(\d+),\s*Failed:\s*(\d+),\s*Accuracy:\s*([\d.]+)%"
)
_FAILED_RE = re.compile(r"^\[Sample (\d+)\]\s*FAILED")


class ChipStreamParser:
    """라인 스트림 -> 이벤트 스트림.

    feed(line) -> event dict | None
      event["type"] in {"sample", "progress", "failed"}
    """

    def __init__(self):
        self._pending_sample: Optional[dict] = None

    def feed(self, line: str) -> Optional[dict]:
        line = line.rstrip("\n")

        # 실패 샘플: "[Sample N] FAILED (timeout)"
        m = _FAILED_RE.match(line)
        if m:
            self._pending_sample = None
            return {"type": "failed", "idx": int(m.group(1))}

        # "[Sample N] Scores: [...]" — 샘플 시작. Predicted 라인을 기다린다.
        m = _SAMPLE_RE.match(line)
        if m:
            idx = int(m.group(1))
            scores = [float(x) for x in m.group(2).split(",") if x.strip()]
            self._pending_sample = {"idx": idx, "scores": scores}
            return None

        # "Predicted: p, Label: l, Result: O/X" — 대기 중인 샘플을 완성.
        m = _PRED_RE.search(line)
        if m and self._pending_sample is not None:
            ev = self._pending_sample
            self._pending_sample = None
            ev["type"] = "sample"
            ev["predicted"] = int(m.group(1))
            ev["label"] = int(m.group(2))
            ev["correct"] = m.group(3).upper() == "O"
            return ev

        # "Progress: ..." — 칩 집계 (크로스체크용).
        m = _PROGRESS_RE.match(line)
        if m:
            return {
                "type": "progress",
                "seen": int(m.group(1)),
                "total": int(m.group(2)),
                "correct": int(m.group(3)),
                "evaluated": int(m.group(4)),
                "failed": int(m.group(5)),
                "chip_accuracy": float(m.group(6)),
            }

        return None
