"""Fair head-to-head comparison: weak baseline vs optimized implementations.

Both old and new implementations are inlined in this script so the comparison
runs in the same Python session against identical inputs, with no branch
switching or import tricks.

Changes benchmarked
-------------------
A  subsystem.py  functools.reduce(np.multiply, ...) → in-place for-loop
B  convert.py    be2le_state_by_state: O(N²) loop → np.ix_ fancy index
C  convert.py    state_by_state2state_by_node: loop → vectorized stack+sum

Usage
-----
    uv run python script/bench_compare_baseline.py
"""

from __future__ import annotations

import functools
import timeit
from itertools import product as iterproduct
from math import log2

import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers (unchanged between old and new)
# ---------------------------------------------------------------------------

def _be2le(i: int, n: int) -> int:
    """Reverse the bits of i (big-endian ↔ little-endian state index)."""
    result = 0
    for _ in range(n):
        result = (result << 1) | (i & 1)
        i >>= 1
    return result


def _le_index2state(i: int, n: int) -> tuple:
    return tuple((i >> k) & 1 for k in range(n))


def _all_states(n: int):
    for i in range(2**n):
        yield tuple((i >> k) & 1 for k in range(n))


# ---------------------------------------------------------------------------
# Change B — be2le_state_by_state
# ---------------------------------------------------------------------------

def be2le_state_by_state_OLD(tpm: np.ndarray) -> np.ndarray:
    """Original O(N²) Python loop (pre-optimization)."""
    N = tpm.shape[0]
    n = int(log2(N))
    le = np.empty(tpm.shape, dtype=tpm.dtype)
    for i, j in iterproduct(range(N), repeat=2):
        le[i, j] = tpm[_be2le(i, n), _be2le(j, n)]
    return le


def be2le_state_by_state_NEW(tpm: np.ndarray) -> np.ndarray:
    """Optimized np.ix_ fancy-index version (post-optimization)."""
    N = tpm.shape[0]
    n = int(log2(N))
    idx = np.array([_be2le(i, n) for i in range(N)])
    return tpm[np.ix_(idx, idx)]


# ---------------------------------------------------------------------------
# Change C — state_by_state2state_by_node
# ---------------------------------------------------------------------------

def state_by_state2state_by_node_OLD(tpm: np.ndarray) -> np.ndarray:
    """Original double Python loop (pre-optimization)."""
    S, _ = tpm.shape
    N = int(log2(S))
    sbn_tpm = np.zeros([2] * N + [N])
    states = {i: _le_index2state(i, N) for i in range(S)}
    node_on = np.array([[states[i][n] for i in range(S)] for n in range(N)])
    on_probabilities = [tpm * node_on[n] for n in range(N)]
    for i, state in states.items():
        sbn_tpm[state] = [np.sum(on_probabilities[n][i]) for n in range(N)]
    return sbn_tpm


def state_by_state2state_by_node_NEW(tpm: np.ndarray) -> np.ndarray:
    """Optimized vectorized stack+sum version (post-optimization)."""
    S, _ = tpm.shape
    N = int(log2(S))
    node_on = np.array(list(_all_states(N)), dtype=float).T  # (N, S)
    sbn_tpm_flat = np.stack(
        [(tpm * node_on[n]).sum(axis=1) for n in range(N)], axis=1
    )  # (S, N)
    return np.ascontiguousarray(sbn_tpm_flat.reshape([2] * N + [N], order="F"))


# ---------------------------------------------------------------------------
# Change A — cause/effect repertoire in-place multiply
# ---------------------------------------------------------------------------

def _mock_single_node_repertoires(k: int, size: int) -> list[np.ndarray]:
    """Return k random positive arrays of length `size` to mock per-node repertoires."""
    rng = np.random.default_rng(0)
    return [rng.random(size) + 0.01 for _ in range(k)]


def cause_repertoire_OLD(items: list[np.ndarray], joint: np.ndarray) -> np.ndarray:
    """Original functools.reduce version (pre-optimization)."""
    joint = joint.copy()
    joint *= functools.reduce(np.multiply, items)
    return joint


def cause_repertoire_NEW(items: list[np.ndarray], joint: np.ndarray) -> np.ndarray:
    """Optimized in-place for-loop version (post-optimization)."""
    joint = joint.copy()
    for item in items:
        joint *= item
    return joint


def partitioned_repertoire_OLD(repertoires: list[np.ndarray]) -> np.ndarray:
    """Original functools.reduce version for partitioned_repertoire (pre-optimization)."""
    return functools.reduce(np.multiply, repertoires)


def partitioned_repertoire_NEW(repertoires: list[np.ndarray]) -> np.ndarray:
    """Optimized in-place version (post-optimization)."""
    result = repertoires[0].copy()
    for r in repertoires[1:]:
        result *= r
    return result


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------

def check_correctness():
    rng = np.random.default_rng(42)

    # Change B
    sbs = rng.random((256, 256))
    sbs /= sbs.sum(axis=1, keepdims=True)
    b_old = be2le_state_by_state_OLD(sbs)
    b_new = be2le_state_by_state_NEW(sbs)
    assert np.allclose(b_old, b_new), "Change B mismatch"

    # Change C
    c_old = state_by_state2state_by_node_OLD(sbs)
    c_new = state_by_state2state_by_node_NEW(sbs)
    assert np.allclose(c_old, c_new), "Change C mismatch"

    # Change A (cause)
    k = 5
    size = 32
    items = _mock_single_node_repertoires(k, size)
    joint = rng.random(size) + 0.01
    a_old = cause_repertoire_OLD(items, joint)
    a_new = cause_repertoire_NEW(items, joint)
    assert np.allclose(a_old, a_new), "Change A (cause) mismatch"

    # Change A (partitioned)
    reps = _mock_single_node_repertoires(4, size)
    p_old = partitioned_repertoire_OLD(reps)
    p_new = partitioned_repertoire_NEW(reps)
    assert np.allclose(p_old, p_new), "Change A (partitioned) mismatch"

    print("All correctness checks passed.")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    check_correctness()

    N_REPS = 300
    rng = np.random.default_rng(42)

    # Shared inputs
    sbs_8 = rng.random((256, 256))
    sbs_8 /= sbs_8.sum(axis=1, keepdims=True)

    k = 5  # mechanism size for Change A
    size = 128
    items = _mock_single_node_repertoires(k, size)
    joint = rng.random(size) + 0.01
    repertoires = _mock_single_node_repertoires(4, size)

    benchmarks = [
        (
            "A  cause_repertoire (k=5, size=128)",
            lambda: cause_repertoire_OLD(items, joint),
            lambda: cause_repertoire_NEW(items, joint),
        ),
        (
            "A  partitioned_repertoire (k=4, size=128)",
            lambda: partitioned_repertoire_OLD(repertoires),
            lambda: partitioned_repertoire_NEW(repertoires),
        ),
        (
            "B  be2le_state_by_state (N=8, 256×256)",
            lambda: be2le_state_by_state_OLD(sbs_8),
            lambda: be2le_state_by_state_NEW(sbs_8),
        ),
        (
            "C  state_by_state2state_by_node (N=8, 256×256)",
            lambda: state_by_state2state_by_node_OLD(sbs_8),
            lambda: state_by_state2state_by_node_NEW(sbs_8),
        ),
    ]

    print()
    print(f"{'Benchmark':<50}  {'Weak (ms)':>10}  {'Opt (ms)':>10}  {'Speedup':>8}")
    print("-" * 85)

    for label, old_fn, new_fn in benchmarks:
        # warm up
        old_fn(); new_fn()
        t_old = timeit.timeit(old_fn, number=N_REPS) / N_REPS * 1000
        t_new = timeit.timeit(new_fn, number=N_REPS) / N_REPS * 1000
        speedup = t_old / t_new if t_new > 0 else float("inf")
        print(f"{label:<50}  {t_old:>10.3f}  {t_new:>10.3f}  {speedup:>7.1f}×")

    print()
    print(f"n={N_REPS} repetitions, same inputs for old and new in each pair.")


if __name__ == "__main__":
    main()
