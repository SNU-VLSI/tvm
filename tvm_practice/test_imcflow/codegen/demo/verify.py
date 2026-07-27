"""SSE 스트림 자동검증 (헤드리스). 브라우저 육안 확인을 갈음한다.

파서+runner(mock)를 직접 태워:
  1. 샘플 이벤트 개수 == num_samples 인지
  2. 각 이벤트에 idx/scores(10개)/predicted/label/correct/orig_idx 가 다 있는지
  3. 뷰어와 동일한 방식(softmax->argmax)으로 계산한 pred 가 칩 predicted 와 일치하는지
  4. 뷰어 자체 집계 running accuracy 최종값 == 칩 Progress 최종 Accuracy 인지 (크로스체크)
  5. orig_idx 가 sample_map 을 통해 올바르게 매핑됐는지

delay_ms 는 0 으로 덮어 빠르게 검증한다.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

import yaml
import runner
from parser import ChipStreamParser

DEMO_ROOT = os.path.dirname(os.path.abspath(__file__))


def softmax_argmax(logits):
    m = max(logits)
    ex = [math.exp(v - m) for v in logits]
    s = sum(ex)
    probs = [v / s for v in ex]
    return probs.index(max(probs))


def main():
    with open(os.path.join(DEMO_ROOT, "config", "resnet8.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["_demo_root"] = DEMO_ROOT
    cfg["profile"] = "mock"
    cfg["mock"]["delay_ms"] = 0

    with open(os.path.join(DEMO_ROOT, cfg["sample_map"])) as f:
        sample_map = json.load(f)["staged_to_original"]

    parser = ChipStreamParser()
    samples, progress = [], []
    for line in runner.iter_lines(cfg):
        ev = parser.feed(line)
        if ev is None:
            continue
        if ev["type"] == "sample":
            ev["orig_idx"] = sample_map[ev["idx"]]
            samples.append(ev)
        elif ev["type"] == "progress":
            progress.append(ev)

    fails = []
    # mock 은 fixture txt 전체를 재생하므로 기대 개수는 fixture 의 [Sample N] 개수다.
    # (config.run.num_samples 는 chip 실행용 값이라 mock 검증 기준으로 쓰면 안 된다.)
    import re
    fixture_path = os.path.join(DEMO_ROOT, cfg["mock"]["fixture"])
    n_expected = len(re.findall(r"^\[Sample \d+\]", open(fixture_path).read(), re.M))

    # 1. 개수
    if len(samples) != n_expected:
        fails.append(f"sample count {len(samples)} != fixture samples {n_expected}")

    # 2. 스키마 + 3. softmax argmax == 칩 predicted + 5. orig_idx 매핑
    seen = correct = 0
    for i, ev in enumerate(samples):
        for k in ("idx", "scores", "predicted", "label", "correct", "orig_idx"):
            if k not in ev:
                fails.append(f"sample {i} missing key {k}")
        if len(ev.get("scores", [])) != cfg["task"]["num_classes"]:
            fails.append(f"sample {i} has {len(ev['scores'])} scores (want {cfg['task']['num_classes']})")
        if ev["idx"] != i:
            fails.append(f"sample order broken at {i}: idx={ev['idx']}")
        if softmax_argmax(ev["scores"]) != ev["predicted"]:
            fails.append(f"sample {i}: softmax argmax != chip predicted {ev['predicted']}")
        if ev["orig_idx"] != sample_map[ev["idx"]]:
            fails.append(f"sample {i}: orig_idx mismatch")
        # 뷰어 집계
        seen += 1
        if ev["correct"]:
            correct += 1

    viewer_acc = 100.0 * correct / seen if seen else 0.0

    # 4. 칩 Progress 최종값 크로스체크
    if not progress:
        fails.append("no Progress line parsed (chip cross-check impossible)")
    else:
        chip_final = progress[-1]["chip_accuracy"]
        if abs(chip_final - viewer_acc) > 0.05:
            fails.append(f"viewer acc {viewer_acc:.2f}% != chip acc {chip_final:.2f}%")

    print(f"samples parsed : {len(samples)}")
    print(f"progress lines : {len(progress)}")
    print(f"viewer running accuracy (self-tallied): {viewer_acc:.2f}%")
    if progress:
        print(f"chip Progress accuracy (cross-check)  : {progress[-1]['chip_accuracy']:.2f}%")
    print(f"first sample: idx={samples[0]['idx']} pred={samples[0]['predicted']} "
          f"label={samples[0]['label']} correct={samples[0]['correct']} orig_idx={samples[0]['orig_idx']}")

    if fails:
        print("\n❌ FAIL")
        for m in fails:
            print("  -", m)
        sys.exit(1)
    print("\n✅ PASS — all checks green")


if __name__ == "__main__":
    main()
