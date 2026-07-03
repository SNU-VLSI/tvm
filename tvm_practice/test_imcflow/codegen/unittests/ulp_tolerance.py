"""ULP-tolerant classification for the TVM <-> PyTorch IMCFlow bit-exact check.

Why this exists
---------------
The bit-exact check must hold the ON-ARRAY INTEGER path exactly -- that path is
the hardware (crossbar MVM + bit-serial ADC + int16 VPU), and TVM and PyTorch
run identical integer code, so any real integer-kernel bug must show up on every
input.

The OFF-CHIP FP stem (conv -> batch_norm -> * x_f -> trunc-to-int16) is a plain
float32 pipeline. TVM's float conv and the PyTorch/host float conv accumulate in
different orders, so they differ by a few ULP. On inputs where a stem output
lands within FP rounding noise of an integer (empirically ~0.006 for VWW, whose
stem values reach ~thousands after * x_f ~= 1858), the trunc-to-int16 cast lands
on opposite sides -> a +/-1 int16 difference. Almost all such +/-1 are absorbed
by the 4-bit activation quantizer; when one happens to sit on a quant-bin
boundary, the 1x1 pointwise MVM amplifies that single +/-1 across all output
channels at that pixel (one +/-1 -> up to out_channels elements, |delta| up to
the weight magnitude). That is a benign FP-boundary artifact, NOT an integer bug
-- on the real chip the stem is computed once, so there is no disagreement.

This means a naive "allow |delta| <= N on the final integer output" is WRONG: the
amplified output delta (e.g. 16 across 64 elements) is large even though the root
cause is a single 1-ULP FP truncation. Tolerance must be applied at the SOURCE.

Policy
------
- FP / off-chip layers (stem conv+bn, final dense): relative FP tolerance via
  ``fp_layer_ok``. Never required to be bit-exact.
- Integer / on-array layers (qconv / MVM, min-max quant, int16 BN): EXACT, with
  two acceptable escape hatches:
    (a) RE-ANCHOR (recommended, rigorous): run the PyTorch integer kernel on
        TVM's own integer input for that op and require exact equality
        (``integer_op_is_exact``). This verifies the kernel in isolation and is
        immune to benign FP-stem propagation, because both sides see the same
        integer input.
    (b) SOURCE CLASSIFICATION: when only end-to-end tensors are available, an
        integer mismatch is benign iff every differing stem-cast element is a
        +/-1 at a pre-cast FP value within ``boundary_eps`` of an integer
        (``stem_cast_is_boundary_ulp``), i.e. the divergence provably originates
        at the FP truncation boundary.

All functions are pure (NumPy in / verdict out) so they can be unit-tested
without TVM. See __main__ for the self-test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Verdicts
EXACT = "EXACT"                       # bit-identical
BENIGN_FP_TOL = "BENIGN_FP_TOL"       # FP layer within rounding tolerance
BENIGN_FP_BOUNDARY = "BENIGN_FP_BOUNDARY"  # int diff traced to FP-cast boundary ULP
REAL_MISMATCH = "REAL_MISMATCH"      # a genuine divergence -> must FAIL


@dataclass
class Verdict:
    status: str
    name: str
    n_mismatch: int
    max_abs_err: float
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.status in (EXACT, BENIGN_FP_TOL, BENIGN_FP_BOUNDARY)


def _np(a):
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(a)


def fp_layer_ok(tvm_arr, pt_arr, name="", rtol: float = 1e-3, atol: float = 2.0) -> Verdict:
    """Off-chip FP layer: pass within relative+absolute FP tolerance.

    atol defaults to 2.0 because after the int16 *x_f scaling the FP stem values
    reach thousands, where a few float32 ULP can be ~1-2 in absolute terms; rtol
    catches proportional drift. These layers are never expected bit-exact across
    two float implementations.
    """
    t = _np(tvm_arr).astype(np.float64)
    p = _np(pt_arr).astype(np.float64)
    if t.shape != p.shape:
        return Verdict(REAL_MISMATCH, name, t.size, float("inf"), f"shape {t.shape} vs {p.shape}")
    ad = np.abs(t - p)
    if np.array_equal(_np(tvm_arr), _np(pt_arr)):
        return Verdict(EXACT, name, 0, 0.0)
    ok = np.allclose(t, p, rtol=rtol, atol=atol)
    n = int((ad > 0).sum())
    return Verdict(BENIGN_FP_TOL if ok else REAL_MISMATCH, name, n, float(ad.max()),
                   f"rtol={rtol} atol={atol}")


def integer_op_is_exact(pt_kernel, tvm_input, tvm_output, name="") -> Verdict:
    """Re-anchor check (recommended): run the PyTorch integer kernel on TVM's own
    integer input and require bit-identical output.

    ``pt_kernel`` is a callable mapping the op's integer input tensor to its
    integer output (e.g. a single qconv / quant / int16-BN). Feeding TVM's input
    isolates the kernel from any upstream FP-stem divergence, so the only way this
    fails is a genuine integer-kernel discrepancy.
    """
    out = _np(pt_kernel(tvm_input))
    tvm_out = _np(tvm_output)
    if out.shape != tvm_out.shape:
        return Verdict(REAL_MISMATCH, name, tvm_out.size, float("inf"),
                       f"shape {tvm_out.shape} vs {out.shape}")
    if np.array_equal(out, tvm_out):
        return Verdict(EXACT, name, 0, 0.0, "integer kernel exact on TVM input")
    ad = np.abs(out.astype(np.float64) - tvm_out.astype(np.float64))
    return Verdict(REAL_MISMATCH, name, int((ad > 0).sum()), float(ad.max()),
                   "integer kernel differs on identical integer input")


def stem_cast_is_boundary_ulp(int_a, int_b, pre_cast_fp_a, pre_cast_fp_b=None,
                              name="", boundary_eps: float = 0.01) -> Verdict:
    """Source classification for the stem -> int16 cast.

    A stem-cast divergence is benign iff every differing element is a +/-1 whose
    pre-cast FP value (on at least one side) is within ``boundary_eps`` of an
    integer -- i.e. it is a truncation-boundary ULP, not a real difference. If
    any differing element has |delta| > 1, or sits away from an integer boundary,
    it is a REAL mismatch.
    """
    a = _np(int_a).astype(np.int64)
    b = _np(int_b).astype(np.int64)
    if a.shape != b.shape:
        return Verdict(REAL_MISMATCH, name, a.size, float("inf"), f"shape {a.shape} vs {b.shape}")
    diff = a != b
    n = int(diff.sum())
    if n == 0:
        return Verdict(EXACT, name, 0, 0.0)
    delta = np.abs(a - b)
    if delta.max() > 1:
        return Verdict(REAL_MISMATCH, name, n, float(delta.max()),
                       "stem cast differs by more than 1 LSB -> not an FP-boundary ULP")
    fa = _np(pre_cast_fp_a).astype(np.float64)
    frac_a = np.abs(fa - np.round(fa))
    near = frac_a < boundary_eps
    if pre_cast_fp_b is not None:
        fb = _np(pre_cast_fp_b).astype(np.float64)
        near = near | (np.abs(fb - np.round(fb)) < boundary_eps)
    # every divergent element must be at a near-integer boundary
    if bool(np.all(near[diff])):
        return Verdict(BENIGN_FP_BOUNDARY, name, n, 1.0,
                       f"all {n} diffs are +/-1 at FP cast boundary (<{boundary_eps})")
    n_bad = int((diff & ~near).sum())
    return Verdict(REAL_MISMATCH, name, n, 1.0,
                   f"{n_bad}/{n} +/-1 diffs are NOT at an FP cast boundary -> real")


def pw1x1_divergence_input_induced(tvm_in, pt_in, tvm_out, pt_out, name="") -> Verdict:
    """Dump-only re-anchor for a 1x1 pointwise on-array MVM.

    A 1x1 conv mixes channels WITHIN a pixel only -- output[:, :, h, w] depends
    solely on input[:, :, h, w]. Therefore a *correct* MVM kernel can never
    produce an output difference at a pixel whose entire input channel-column is
    identical on both sides. So, given the per-op integer inputs and outputs from
    the TVM and PyTorch dumps:

      - output bit-identical                              -> EXACT
      - every output-differing pixel (h,w) also has an    -> BENIGN_FP_BOUNDARY
        input difference at the same (h,w)                   (upstream FP-stem
                                                              propagation, not a
                                                              kernel bug)
      - an output diff at a pixel with an IDENTICAL input  -> REAL_MISMATCH
        column                                               (genuine MVM bug)

    This needs no model re-run: it isolates the integer kernel using only the
    dumped tensors (inputs[0] of the pw node on the TVM side, the pw 'input'
    hook on the PyTorch side). Arrays are [N, C, H, W] integer tensors.
    """
    ti = _np(tvm_in).astype(np.int64); pi = _np(pt_in).astype(np.int64)
    to = _np(tvm_out).astype(np.int64); po = _np(pt_out).astype(np.int64)
    if to.shape != po.shape:
        return Verdict(REAL_MISMATCH, name, to.size, float("inf"), f"out shape {to.shape} vs {po.shape}")
    out_diff = to != po
    n_out = int(out_diff.sum())
    if n_out == 0:
        return Verdict(EXACT, name, 0, 0.0)
    if ti.shape != pi.shape:
        return Verdict(REAL_MISMATCH, name, n_out, float("inf"),
                       f"cannot classify: in shape {ti.shape} vs {pi.shape}")
    # spatial masks over (N,H,W): channel axis = 1
    out_sp = out_diff.any(axis=1)
    in_sp = (ti != pi).any(axis=1)
    unexplained = out_sp & ~in_sp
    n_unexp = int(unexplained.sum())
    max_err = int(np.abs(to - po).max())
    if n_unexp == 0:
        return Verdict(BENIGN_FP_BOUNDARY, name, n_out, float(max_err),
                       f"all {n_out} output diffs sit at pixels with a differing input "
                       f"column -> upstream-FP propagation, MVM kernel consistent")
    return Verdict(REAL_MISMATCH, name, n_out, float(max_err),
                   f"{n_unexp} output-diff pixel(s) have an IDENTICAL input column "
                   f"-> genuine MVM kernel mismatch")


# --------------------------------------------------------------------------- #
# Self-test (pure synthetic; always runnable without TVM/CIM).
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    rng = np.random.default_rng(0)
    fails = []

    # 1) FP layer within ULP -> BENIGN_FP_TOL
    base = rng.uniform(-5000, 5000, size=(1, 8, 48, 48))
    noisy = base + rng.normal(0, 0.5, size=base.shape)  # ~ULP noise
    v = fp_layer_ok(base, noisy, "stem_fp")
    assert v.passed and v.status == BENIGN_FP_TOL, v
    # FP layer with a real gross error -> REAL
    bad = base.copy(); bad[0, 0, 0, 0] += 500.0
    v = fp_layer_ok(base, bad, "stem_fp_bad")
    assert v.status == REAL_MISMATCH, v

    # 2) stem cast: benign FP-boundary +/-1 at near-integer values
    pre = rng.uniform(-3000, 3000, size=(1, 8, 48, 48))
    ia = np.trunc(pre).astype(np.int64)
    # pick 25 positions, nudge them to sit just on an integer boundary and flip +/-1
    idx = tuple(rng.integers(0, s, size=25) for s in pre.shape)
    pre[idx] = np.round(pre[idx]) + 0.001            # within boundary_eps of an int
    ia = np.trunc(pre).astype(np.int64)
    ib = ia.copy(); ib[idx] += 1                      # the FP-order flip
    v = stem_cast_is_boundary_ulp(ia, ib, pre, name="stem_cast")
    assert v.status == BENIGN_FP_BOUNDARY, v
    # a +/-1 NOT at a boundary -> REAL
    ib2 = ia.copy(); off = tuple(int(rng.integers(0, s)) for s in pre.shape)
    pre[off] = np.round(pre[off]) + 0.4              # mid-bin, not a boundary
    ia2 = np.trunc(pre).astype(np.int64); ib2 = ia2.copy(); ib2[off] += 1
    v = stem_cast_is_boundary_ulp(ia2, ib2, pre, name="stem_cast_real")
    assert v.status == REAL_MISMATCH, v
    # a >1 LSB diff -> REAL (would indicate a true integer divergence)
    ib3 = ia.copy(); ib3[idx[0][0], idx[1][0], idx[2][0], idx[3][0]] += 5
    v = stem_cast_is_boundary_ulp(ia, ib3, pre, name="stem_cast_big")
    assert v.status == REAL_MISMATCH, v

    # 3) re-anchor: identical integer kernel on same input -> EXACT;
    #    a buggy kernel -> REAL even on identical input.
    w = rng.integers(-7, 8, size=(16, 8, 1, 1))
    xin = rng.integers(0, 16, size=(1, 8, 6, 6))
    def good_kernel(x):
        return np.tensordot(w.reshape(16, 8), x.reshape(8, -1), axes=1).reshape(16, 6, 6)
    tvm_out = good_kernel(xin)
    v = integer_op_is_exact(good_kernel, xin, tvm_out, "pw")
    assert v.status == EXACT, v
    def buggy_kernel(x):
        o = good_kernel(x); o[0, 0, 0] += 1; return o
    v = integer_op_is_exact(buggy_kernel, xin, tvm_out, "pw_bug")
    assert v.status == REAL_MISMATCH, v

    # 4) 1x1 pw divergence classification (dump-only re-anchor)
    cin, cout, H, W = 8, 16, 6, 6
    ti = rng.integers(0, 16, size=(1, cin, H, W))
    pi = ti.copy()
    W1 = rng.integers(-7, 8, size=(cout, cin))
    def mvm(x):
        return np.tensordot(W1, x.reshape(cin, -1), axes=1).reshape(1, cout, H, W)
    to = mvm(ti)
    # (a) identical input -> identical output -> EXACT
    v = pw1x1_divergence_input_induced(ti, pi, to, mvm(pi), "pw_exact")
    assert v.status == EXACT, v
    # (b) perturb input at one pixel -> output differs ONLY at that pixel -> BENIGN
    pi_b = ti.copy(); pi_b[0, 3, 2, 4] += 1
    v = pw1x1_divergence_input_induced(ti, pi_b, to, mvm(pi_b), "pw_benign")
    assert v.status == BENIGN_FP_BOUNDARY, v
    # (c) identical input but output altered at a pixel -> REAL kernel bug
    to_bug = to.copy(); to_bug[0, :, 1, 1] += 3
    v = pw1x1_divergence_input_induced(ti, pi, to, to_bug, "pw_kernelbug")
    assert v.status == REAL_MISMATCH, v

    print("ulp_tolerance self-test: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
