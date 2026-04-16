"""B7 Baseline — ΦID Integrated Synergy as Approximate Phi.

Mediano, Rosas et al. (2021) https://arxiv.org/abs/2109.13186
Rosas et al. (2025 PNAS)     https://doi.org/10.1073/pnas.2423297122

What ΦID computes
-----------------
Φ-ID (Integrated Information Decomposition) decomposes the mutual information
between the past and future of a bivariate system (X_i, X_j) into 16 atoms via
partial information decomposition (PID).  The ``sts`` atom — Syn→Syn — captures
the information that is *synergistic* in both the past and the future: only
accessible from the joint system and irreducible to its parts.

For an N-node network, system-level integrated synergy is:

    ΦID_SYN = Σ_{i≠j} sts(X_i, X_j)

summed over all N(N-1) ordered node pairs at lag τ=1.  This is a *different*
quantity from IIT Φ: it does not require partition enumeration, runs in O(N²)
pairwise computations, and correlates empirically with exact Φ on binary systems.

Implementation
--------------
1. Simulate a Markov chain of T steps using the network TPM; extract per-node
   binary time series.
2. For each ordered pair (i, j) with i≠j, call
   ``phyid.calculate.calc_PhiID(src=X_i, trg=X_j, tau=1, kind="discrete")``.
3. Sum the ``sts`` atoms across all pairs → ΦID_SYN.
4. Compare to the true phi from exhaustive ``sia()`` (Pearson r across networks
   cannot be computed from 2 data points; we report the ratio and directionality).

Complexity
----------
- Simulation: O(2^N · T) — TPM lookup per step
- Pairwise ΦID: O(N² · T) — discrete entropy enumeration per pair
- No partition search required; scales to N >> 8

Usage
-----
    uv run python script/bench_phyid.py
    uv run python script/bench_phyid.py --n 4 5 --T 20000
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Inline network definitions (shared with other bench_* scripts)
# ---------------------------------------------------------------------------

_MICRO_TPM = [
    [0.2401, 0.1029, 0.1029, 0.0441, 0.1029, 0.0441, 0.0441, 0.0189, 0.1029, 0.0441, 0.0441, 0.0189, 0.0441, 0.0189, 0.0189, 0.0081],
    [0.2401, 0.1029, 0.1029, 0.0441, 0.1029, 0.0441, 0.0441, 0.0189, 0.1029, 0.0441, 0.0441, 0.0189, 0.0441, 0.0189, 0.0189, 0.0081],
    [0.2401, 0.1029, 0.1029, 0.0441, 0.1029, 0.0441, 0.0441, 0.0189, 0.1029, 0.0441, 0.0441, 0.0189, 0.0441, 0.0189, 0.0189, 0.0081],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4900, 0.2100, 0.2100, 0.0900],
    [0.2401, 0.1029, 0.1029, 0.0441, 0.1029, 0.0441, 0.0441, 0.0189, 0.1029, 0.0441, 0.0441, 0.0189, 0.0441, 0.0189, 0.0189, 0.0081],
    [0.2401, 0.1029, 0.1029, 0.0441, 0.1029, 0.0441, 0.0441, 0.0189, 0.1029, 0.0441, 0.0441, 0.0189, 0.0441, 0.0189, 0.0189, 0.0081],
    [0.2401, 0.1029, 0.1029, 0.0441, 0.1029, 0.0441, 0.0441, 0.0189, 0.1029, 0.0441, 0.0441, 0.0189, 0.0441, 0.0189, 0.0189, 0.0081],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4900, 0.2100, 0.2100, 0.0900],
    [0.2401, 0.1029, 0.1029, 0.0441, 0.1029, 0.0441, 0.0441, 0.0189, 0.1029, 0.0441, 0.0441, 0.0189, 0.0441, 0.0189, 0.0189, 0.0081],
    [0.2401, 0.1029, 0.1029, 0.0441, 0.1029, 0.0441, 0.0441, 0.0189, 0.1029, 0.0441, 0.0441, 0.0189, 0.0441, 0.0189, 0.0189, 0.0081],
    [0.2401, 0.1029, 0.1029, 0.0441, 0.1029, 0.0441, 0.0441, 0.0189, 0.1029, 0.0441, 0.0441, 0.0189, 0.0441, 0.0189, 0.0189, 0.0081],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4900, 0.2100, 0.2100, 0.0900],
    [0.0000, 0.0000, 0.0000, 0.4900, 0.0000, 0.0000, 0.0000, 0.2100, 0.0000, 0.0000, 0.0000, 0.2100, 0.0000, 0.0000, 0.0000, 0.0900],
    [0.0000, 0.0000, 0.0000, 0.4900, 0.0000, 0.0000, 0.0000, 0.2100, 0.0000, 0.0000, 0.0000, 0.2100, 0.0000, 0.0000, 0.0000, 0.0900],
    [0.0000, 0.0000, 0.0000, 0.4900, 0.0000, 0.0000, 0.0000, 0.2100, 0.0000, 0.0000, 0.0000, 0.2100, 0.0000, 0.0000, 0.0000, 0.0900],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
]
_MICRO_CM = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

_RULE152_TPM = [
    [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 1, 0, 1, 0], [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1], [0, 0, 1, 0, 0], [0, 1, 1, 0, 1], [1, 1, 1, 0, 0],
    [1, 0, 0, 0, 0], [0, 1, 0, 0, 1], [0, 0, 1, 0, 0], [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0], [1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0], [0, 1, 0, 1, 1], [0, 0, 0, 1, 0], [1, 0, 0, 1, 1],
    [1, 0, 1, 1, 0], [0, 0, 1, 1, 1], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1],
]
_RULE152_CM = [
    [1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1], [1, 0, 0, 1, 1],
]

_VERIFIED_NETWORKS: dict[int, tuple[str, str, tuple[int, ...]]] = {
    3: ("basic_network", "basic", (0, 0, 0)),
    4: ("micro (IIT 4.0 paper)", "micro", (1, 1, 1, 1)),
    5: ("rule152 ring CA (N=5)", "rule152_5", (0, 0, 0, 0, 0)),
    6: ("rule152 ring CA (N=6)", "rule152_6", (0, 0, 0, 0, 0, 0)),
    7: ("rule152 ring CA (N=7)", "rule152_7", (0, 0, 0, 0, 0, 0, 0)),
}


def _make_rule_ca(rule_num: int, n: int):
    from pyphi import Network
    rule = {((i >> 2) & 1, (i >> 1) & 1, i & 1): (rule_num >> i) & 1 for i in range(8)}
    cm = np.zeros((n, n), dtype=int)
    for i in range(n):
        cm[(i - 1) % n, i] = 1
        cm[i, i] = 1
        cm[(i + 1) % n, i] = 1
    tpm = np.zeros((2**n, n), dtype=float)
    for s in range(2**n):
        state = [(s >> i) & 1 for i in range(n)]
        for i in range(n):
            tpm[s, i] = rule[(state[(i - 1) % n], state[i], state[(i + 1) % n])]
    return Network(tpm, cm=cm)


def _load_network(n: int):
    from pyphi import Network
    _, loader, state = _VERIFIED_NETWORKS[n]
    if loader == "basic":
        import pyphi.examples as ex
        return ex.basic_network(), state
    if loader == "micro":
        return Network(np.array(_MICRO_TPM), cm=np.array(_MICRO_CM)), state
    if loader == "rule152_5":
        return Network(np.array(_RULE152_TPM), cm=np.array(_RULE152_CM)), state
    if loader.startswith("rule152_"):
        return _make_rule_ca(152, int(loader.split("_")[1])), state
    raise ValueError(f"Unknown loader: {loader}")


# ---------------------------------------------------------------------------
# Markov chain simulation
# ---------------------------------------------------------------------------

def simulate_chain(tpm_sbn: np.ndarray, T: int, seed: int = 42) -> np.ndarray:
    """Simulate T steps of the Markov chain; return binary node time series.

    For stochastic TPMs: sample next state from the row-stochastic SBN TPM.
    For deterministic TPMs: each row has exactly one 1 per node — follow it.

    Parameters
    ----------
    tpm_sbn : (2^N, N) state-by-node TPM (probabilities per node conditional
              on current state index)
    T       : number of time steps
    seed    : random seed

    Returns
    -------
    (T, N) int array — binary state of each node at each time step
    """
    rng = np.random.default_rng(seed)
    S, N = tpm_sbn.shape
    n_bits = int(np.log2(S))
    assert 2**n_bits == S

    # Start from a random state index sampled uniformly
    state_idx = int(rng.integers(S))
    ts = np.empty((T, N), dtype=np.int8)

    for t in range(T):
        state_bits = np.array([(state_idx >> i) & 1 for i in range(N)], dtype=np.int8)
        ts[t] = state_bits
        probs = tpm_sbn[state_idx]
        # Sample each node independently from its conditional probability
        next_bits = (rng.random(N) < probs).astype(np.int8)
        state_idx = int(sum(int(b) << i for i, b in enumerate(next_bits)))

    return ts


# ---------------------------------------------------------------------------
# ΦID integrated synergy
# ---------------------------------------------------------------------------

def compute_phyid_syn(ts: np.ndarray, tau: int = 1, redundancy: str = "MMI") -> dict:
    """Compute system-level integrated synergy from node time series.

    For each ordered pair (i, j) with i≠j, compute pairwise ΦID and extract
    the ``sts`` atom (Syn→Syn = integrated synergy).  Sum across all pairs.

    Parameters
    ----------
    ts         : (T, N) binary time series
    tau        : time lag (default 1)
    redundancy : 'MMI' or 'CCS' (default 'MMI')

    Returns
    -------
    dict with:
      sts_sum   : sum of sts atoms across all ordered pairs  (system-level ΦID_SYN)
      sts_mean  : mean sts per pair
      sts_pairs : dict mapping (i, j) → sts value
      n_pairs   : N(N-1) ordered pairs evaluated
    """
    from phyid.calculate import calc_PhiID

    T, N = ts.shape
    sts_pairs = {}

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            src = ts[:, i].astype(float)
            trg = ts[:, j].astype(float)
            atoms, _ = calc_PhiID(src, trg, tau=tau, kind="discrete",
                                   redundancy=redundancy)
            sts_val = float(np.mean(atoms["sts"]))
            sts_pairs[(i, j)] = sts_val

    sts_vals = list(sts_pairs.values())
    return {
        "sts_sum": sum(sts_vals),
        "sts_mean": sum(sts_vals) / len(sts_vals) if sts_vals else 0.0,
        "sts_pairs": sts_pairs,
        "n_pairs": len(sts_pairs),
    }


# ---------------------------------------------------------------------------
# Core diagnostic
# ---------------------------------------------------------------------------

def run_phyid_diagnostic(n: int, T: int = 20000, seed: int = 42) -> dict:
    """Run ΦID diagnostic for the N-node benchmark network.

    Steps
    -----
    1. Simulate T-step Markov chain → per-node binary time series.
    2. Compute pairwise ΦID ``sts`` atoms for all N(N-1) ordered pairs.
    3. Sum → system-level ΦID_SYN (no partition search).
    4. Run exhaustive ``sia()`` for ground-truth IIT 4.0 phi.
    5. Report both quantities and computation times.

    Note: ΦID_SYN is a *different measure* from IIT Φ — it does not find
    the MIP and is not comparable on an absolute scale.  The comparison
    is directional: do networks with higher Φ also have higher ΦID_SYN?

    Returns
    -------
    dict with diagnostic results
    """
    os.environ.setdefault("PYPHI_WELCOME_OFF", "yes")
    import pyphi
    import pyphi.new_big_phi as nb
    from pyphi import Subsystem
    from pyphi.partition import system_partitions

    pyphi.config.WELCOME_OFF = True
    pyphi.config.PROGRESS_BARS = False

    network, state = _load_network(n)
    subsys = Subsystem(network, state, network.node_indices)
    node_indices = subsys.node_indices

    # ---- simulate time series -----------------------------------------------
    # network.tpm is in multidimensional form (2, 2, ..., N); reshape to (2^N, N)
    tpm_flat = network.tpm.reshape(2**n, n)
    t0 = time.perf_counter()
    ts = simulate_chain(tpm_flat, T=T, seed=seed)
    t_sim = time.perf_counter() - t0

    # ---- pairwise ΦID -------------------------------------------------------
    t0 = time.perf_counter()
    phyid_res = compute_phyid_syn(ts, tau=1, redundancy="MMI")
    t_phyid = time.perf_counter() - t0
    phyid_syn = phyid_res["sts_sum"]

    # ---- exhaustive sia (ground-truth IIT 4.0 phi) --------------------------
    t0 = time.perf_counter()
    all_parts = list(system_partitions(node_indices, partition_scheme="SET_UNI/BI"))
    full_sia = nb.sia(subsys, partitions=iter(all_parts))
    t_full = time.perf_counter() - t0
    full_phi = full_sia.phi

    return {
        "n": n,
        "T": T,
        "n_pairs": phyid_res["n_pairs"],
        "phyid_syn": phyid_syn,
        "phyid_syn_mean": phyid_res["sts_mean"],
        "t_sim": t_sim,
        "t_phyid": t_phyid,
        "full_phi": full_phi,
        "t_full": t_full,
        "sts_pairs": phyid_res["sts_pairs"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n", type=int, nargs="+", default=[4, 5],
        help="Node counts to test (default: 4 5)",
    )
    parser.add_argument("--T", type=int, default=20000,
                        help="Simulation length (default: 20000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    for n in sorted(args.n):
        if n not in _VERIFIED_NETWORKS:
            print(f"N={n}: no validated network — skipping")
            continue
        label = _VERIFIED_NETWORKS[n][0]
        print(f"  Running N={n} ({label}) ...", end="\r", flush=True)
        r = run_phyid_diagnostic(n, T=args.T, seed=args.seed)
        rows.append(r)

    print()
    print("=== B7 ΦID Integrated Synergy Diagnostic ===")
    print(f"  Simulation length T={args.T}, tau=1, redundancy=MMI")
    print()
    hdr = (
        f"{'N':>3}  {'Pairs':>5}  {'ΦID_SYN':>10}  {'ΦID/pair':>9}  "
        f"{'True phi':>9}  "
        f"{'T_sim(s)':>8}  {'T_ΦID(s)':>8}  {'T_full(s)':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['n']:>3}  {r['n_pairs']:>5}  {r['phyid_syn']:>10.4f}  "
            f"{r['phyid_syn_mean']:>9.4f}  "
            f"{r['full_phi']:>9.4f}  "
            f"{r['t_sim']:>8.3f}  {r['t_phyid']:>8.3f}  {r['t_full']:>9.2f}"
        )
    print()
    print("ΦID_SYN  = Σ_{i≠j} mean(sts(X_i, X_j))  [sum of integrated synergy over N(N-1) pairs]")
    print("ΦID/pair = ΦID_SYN / N(N-1)              [per-pair average]")
    print("True phi = IIT 4.0 Φ from exhaustive sia()")
    print("T_ΦID    = wall time for all pairwise ΦID computations (no partition search)")
    print("T_full   = wall time for exhaustive sia()")
    print()
    print("Note: ΦID_SYN and IIT Φ are different quantities — ΦID_SYN does not")
    print("      require partition search and scales to large N.")

    # ---- append to docs/speed_memory.md ------------------------------------
    docs_path = Path(__file__).parent.parent / "docs" / "speed_memory.md"
    if docs_path.exists() and rows:
        import datetime
        ts_str = datetime.datetime.now().strftime("%Y-%m-%d")
        lines = [
            "",
            "---",
            "",
            f"## B7 ΦID Integrated Synergy Diagnostic ({ts_str})",
            "",
            f"Simulate T={args.T} steps of the network Markov chain; for each ordered",
            "pair (i, j) with i≠j compute pairwise ΦID (discrete, MMI redundancy, τ=1);",
            "sum the `sts` (Syn→Syn) atoms across all N(N-1) pairs to obtain",
            "system-level integrated synergy ΦID_SYN.  Compare to exhaustive IIT 4.0 Φ.",
            "",
            "| N | Pairs | ΦID_SYN | ΦID/pair | True phi | T_sim (s) | T_ΦID (s) | T_full (s) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r['n']} | {r['n_pairs']} | {r['phyid_syn']:.4f} | "
                f"{r['phyid_syn_mean']:.4f} | {r['full_phi']:.4f} | "
                f"{r['t_sim']:.3f} | {r['t_phyid']:.3f} | {r['t_full']:.2f} |"
            )
        lines += [
            "",
            "_ΦID_SYN_: sum of pairwise `sts` atoms across all N(N-1) ordered pairs.",
            "_ΦID/pair_: per-pair average integrated synergy.",
            "_T_ΦID_: wall time for all pairwise ΦID computations (no partition search).",
            "_T_full_: wall time for exhaustive `sia()` (full partition search).",
            "",
            "**Note:** ΦID_SYN and IIT Φ are different quantities and are not on the",
            "same scale. ΦID_SYN does not require partition enumeration and scales to",
            "N >> 8 where exact IIT Φ is infeasible.",
            "",
        ]
        with open(docs_path, "a") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Results appended to {docs_path}")


if __name__ == "__main__":
    main()
