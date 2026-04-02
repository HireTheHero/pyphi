"""Benchmark script for the three high-ROI optimizations.

Run before and after applying changes to measure speedup:
    uv run python script/bench_optimizations.py

Benchmarks:
  A - cause_repertoire / effect_repertoire (subsystem.py in-place multiply)
  B - be2le_state_by_state (convert.py np.ix_ vectorization)
  C - state_by_state2state_by_node (convert.py matrix-multiply vectorization)
"""

import timeit

import numpy as np

import pyphi
from pyphi import Subsystem, examples
from pyphi.convert import be2le_state_by_state, state_by_state2state_by_node

# Suppress welcome message and progress bars during benchmarking
pyphi.config.WELCOME_OFF = True
pyphi.config.PROGRESS_BARS = False

# ---------------------------------------------------------------------------
# Change A: cause_repertoire / effect_repertoire  (7-node fig16 network)
# ---------------------------------------------------------------------------
network = examples.fig16_network()
state = (0,) * 7
subsys = Subsystem(network, state, network.node_indices)


def bench_cause():
    subsys._repertoire_cache.clear()
    subsys.cause_repertoire(network.node_indices, network.node_indices)


def bench_effect():
    subsys._repertoire_cache.clear()
    subsys.effect_repertoire(network.node_indices, network.node_indices)


# ---------------------------------------------------------------------------
# Change B: be2le_state_by_state  (8-node, 256×256 state-by-state TPM)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
sbs_tpm_8 = rng.random((256, 256))
# Make rows sum to 1 (valid stochastic matrix)
sbs_tpm_8 /= sbs_tpm_8.sum(axis=1, keepdims=True)


def bench_be2le():
    be2le_state_by_state(sbs_tpm_8)


# ---------------------------------------------------------------------------
# Change C: state_by_state2state_by_node  (8-node, same TPM)
# ---------------------------------------------------------------------------


def bench_sbs2sbn():
    state_by_state2state_by_node(sbs_tpm_8)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
REPEATS = 200

benchmarks = [
    ("cause_repertoire        (7-node fig16)", bench_cause),
    ("effect_repertoire       (7-node fig16)", bench_effect),
    ("be2le_state_by_state    (8-node, 256×256)", bench_be2le),
    ("state_by_state2state_by_node (8-node)", bench_sbs2sbn),
]

print(f"{'Function':<45}  {'ms/call':>10}  (n={REPEATS})")
print("-" * 62)
for label, fn in benchmarks:
    # One warm-up call
    fn()
    t = timeit.timeit(fn, number=REPEATS)
    print(f"{label:<45}  {t / REPEATS * 1000:>10.3f}")
