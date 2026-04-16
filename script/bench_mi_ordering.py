"""MI-ordered partition diagnostic for PyPhi SIA.

Tests whether ordering system partitions by mutual information (MI) between
the severed node groups concentrates the MIP search in the early ranks.

Hypothesis: cutting connections between strongly-coupled node groups reduces
phi more than cutting weakly-coupled groups.  If true, sorting partitions
descending by MI(X_from; X_to) should bring the MIP to a low rank —
enabling an early-stopping rule that avoids evaluating most partitions.

Algorithm
---------
1. Build the system (same verified non-zero phi networks as bench_scaling.py).
2. Compute the stationary distribution p(X) via power iteration on the
   state-by-state TPM.
3. For each system partition (from_nodes, to_nodes, direction) compute:
       MI(X_from; X_to) under p(X)          [connectivity-informed proxy]
4. Sort partitions descending by MI (strong coupling first).
5. Evaluate each partition in that order using evaluate_partition(), tracking
   the running minimum (normalized_phi, -phi) key used by sia().
6. Record the rank at which the true MIP is first reached.
7. Repeat with random ordering (10 trials) for comparison.

Output is a table: N, n_partitions, mi_rank, mi_rank_pct, random_rank_mean,
random_rank_pct_mean.  Results are also appended to docs/speed_memory.md.

Usage
-----
    uv run python script/bench_mi_ordering.py
    uv run python script/bench_mi_ordering.py --n 5 6
    uv run python script/bench_mi_ordering.py --n 4 5 6 --random-trials 20
"""

from __future__ import annotations

import argparse
import os
import random as stdlib_random
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Inline network definitions (mirrored from bench_scaling.py)
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
_MICRO_CM = [
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
]

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
    [1, 1, 0, 0, 1],
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [1, 0, 0, 1, 1],
]

_VERIFIED_NETWORKS: dict[int, tuple[str, str, tuple[int, ...]]] = {
    3: ("basic_network", "basic", (0, 0, 0)),
    4: ("micro (IIT 4.0 paper)", "micro", (1, 1, 1, 1)),
    5: ("rule152 ring CA (N=5)", "rule152_5", (0, 0, 0, 0, 0)),
    6: ("rule152 ring CA (N=6)", "rule152_6", (0, 0, 0, 0, 0, 0)),
    7: ("rule152 ring CA (N=7)", "rule152_7", (0, 0, 0, 0, 0, 0, 0)),
}


def _make_rule_ca(rule_num: int, n: int):
    """Generate a rule-{rule_num} ring CA network of size N."""
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
    """Load the verified non-zero phi network for N nodes."""
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
        size = int(loader.split("_")[1])
        return _make_rule_ca(152, size), state
    raise ValueError(f"Unknown loader: {loader}")


# ---------------------------------------------------------------------------
# Stationary distribution
# ---------------------------------------------------------------------------

def stationary_distribution(tpm_sbs: np.ndarray, max_iter: int = 2000, tol: float = 1e-12) -> np.ndarray:
    """Estimate stationary distribution by power iteration.

    For deterministic CAs (rows are one-hot) the Markov chain may have cycles,
    so we time-average 1000 steps after convergence has been checked.
    """
    n_states = tpm_sbs.shape[0]
    p = np.ones(n_states) / n_states
    for _ in range(max_iter):
        p_new = p @ tpm_sbs
        if np.max(np.abs(p_new - p)) < tol:
            return p_new
        p = p_new
    # If not converged (cyclic CA), return time average of last 100 steps
    p_avg = np.zeros(n_states)
    for _ in range(100):
        p = p @ tpm_sbs
        p_avg += p
    return p_avg / 100.0


# ---------------------------------------------------------------------------
# MI proxy for a partition
# ---------------------------------------------------------------------------

def pairwise_mi_matrix(n: int, stationary: np.ndarray) -> np.ndarray:
    """Precompute the N×N pairwise mutual-information matrix.

    M[i, j] = I(X_i; X_j) under the stationary distribution (in nats).
    State index s encodes node bits in little-endian order (bit k = node k).

    Parameters
    ----------
    n         : number of nodes
    stationary: 1-D array of shape (2**n,)

    Returns
    -------
    np.ndarray of shape (n, n) with non-negative entries.
    """
    n_states = 2 ** n
    M = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            # Build 2×2 joint distribution p(X_i, X_j)
            joint = np.zeros((2, 2))
            for s, p_s in enumerate(stationary):
                if p_s == 0.0:
                    continue
                xi = (s >> i) & 1
                xj = (s >> j) & 1
                joint[xi, xj] += p_s

            p_i = joint.sum(axis=1, keepdims=True)  # (2,1)
            p_j = joint.sum(axis=0, keepdims=True)  # (1,2)
            outer = p_i * p_j
            mask = (joint > 0) & (outer > 0)
            mi = float(np.sum(joint[mask] * np.log(joint[mask] / outer[mask])))
            mi = max(0.0, mi)
            M[i, j] = M[j, i] = mi

    return M


def partition_mi_score(cut_matrix: np.ndarray, pairwise_mi: np.ndarray) -> float:
    """Proxy MI score for a partition defined by its cut matrix.

    Score = sum of pairwise MI values for all severed connections.
    Higher score ↔ more information flow severed ↔ partition expected to hurt phi more.

    Parameters
    ----------
    cut_matrix : (N, N) integer array — 1 where a connection is severed
    pairwise_mi: (N, N) precomputed MI matrix from pairwise_mi_matrix()

    Returns
    -------
    float: non-negative proxy score
    """
    return float(np.sum(cut_matrix * pairwise_mi))


# ---------------------------------------------------------------------------
# Core diagnostic
# ---------------------------------------------------------------------------

def run_diagnostic(n: int, random_trials: int = 10) -> dict:
    """Run the MI-ordering diagnostic for an N-node system.

    Returns
    -------
    dict with keys:
      n, n_partitions, phi,
      mi_rank (1-based), mi_rank_pct,
      random_rank_mean, random_rank_pct_mean,
      mi_sort_time_s, eval_time_s
    """
    os.environ.setdefault("PYPHI_WELCOME_OFF", "yes")
    import pyphi
    import pyphi.new_big_phi as nb
    from pyphi import Subsystem
    from pyphi.convert import state_by_node2state_by_state
    from pyphi.partition import system_partitions

    pyphi.config.WELCOME_OFF = True
    pyphi.config.PROGRESS_BARS = False

    network, state = _load_network(n)
    subsys = Subsystem(network, state, network.node_indices)
    node_indices = subsys.node_indices

    # ---- stationary distribution ----------------------------------------
    tpm_sbn = network.tpm  # state-by-node, little-endian rows
    tpm_sbs = state_by_node2state_by_state(tpm_sbn)
    stationary = stationary_distribution(tpm_sbs)

    # ---- pairwise MI matrix (O(N² * 2^N)) --------------------------------
    pairwise_mi = pairwise_mi_matrix(n, stationary)

    # ---- generate all partitions ----------------------------------------
    all_partitions = list(system_partitions(node_indices))
    n_total = len(all_partitions)

    # ---- compute MI proxy for each partition ----------------------------
    t0 = time.perf_counter()
    mi_scores = []
    for part in all_partitions:
        # All IIT 4.0 partition types expose _cut_matrix (N×N) via GeneralKCut
        cmat = part._cut_matrix
        mi_scores.append(partition_mi_score(cmat, pairwise_mi))
    mi_sort_time = time.perf_counter() - t0

    # Sort both descending and ascending by MI
    order_desc = np.argsort(mi_scores)[::-1].tolist()   # high MI first
    order_asc  = np.argsort(mi_scores).tolist()          # low MI first

    # ---- evaluate all partitions in MI-descending order, track running min --
    system_state = nb.system_intrinsic_information(subsys)

    t0 = time.perf_counter()
    phi_keys_desc: list[tuple[float, float]] = []
    for i in order_desc:
        sia = nb.evaluate_partition(all_partitions[i], subsys, system_state)
        phi_keys_desc.append(nb.sia_minimization_key(sia))
    eval_time = time.perf_counter() - t0

    true_mip_key = min(phi_keys_desc)
    phi_at_mip = true_mip_key[0]  # normalized_phi at MIP

    def first_mip_rank(keys: list) -> int:
        """1-based rank at which the running minimum first equals the true MIP key."""
        running_min = keys[0]
        for i, key in enumerate(keys[1:], start=2):
            if key < running_min:
                running_min = key
            if running_min <= true_mip_key:
                return i
        return len(keys)

    mi_rank_desc = first_mip_rank(phi_keys_desc)

    # Ascending: reorder phi_keys to match ascending-MI order
    # phi_keys_desc[i] = key for partition order_desc[i]
    # Build a lookup: original partition index → phi_key
    orig_to_key: list[tuple] = [None] * n_total  # type: ignore[list-item]
    for rank_pos, orig_idx in enumerate(order_desc):
        orig_to_key[orig_idx] = phi_keys_desc[rank_pos]
    phi_keys_asc = [orig_to_key[i] for i in order_asc]
    mi_rank_asc = first_mip_rank(phi_keys_asc)

    # ---- random ordering baseline (multiple trials) ----------------------
    random_ranks = []
    for _ in range(random_trials):
        perm = list(range(n_total))
        stdlib_random.shuffle(perm)
        shuffled_keys = [orig_to_key[i] for i in perm]
        random_ranks.append(first_mip_rank(shuffled_keys))

    random_rank_mean = float(np.mean(random_ranks))
    random_rank_std = float(np.std(random_ranks))

    return {
        "n": n,
        "n_partitions": n_total,
        "phi": phi_at_mip,
        "mi_rank_desc": mi_rank_desc,
        "mi_rank_desc_pct": 100.0 * mi_rank_desc / n_total,
        "mi_rank_asc": mi_rank_asc,
        "mi_rank_asc_pct": 100.0 * mi_rank_asc / n_total,
        "random_rank_mean": random_rank_mean,
        "random_rank_std": random_rank_std,
        "random_rank_pct_mean": 100.0 * random_rank_mean / n_total,
        "mi_sort_time_s": mi_sort_time,
        "eval_time_s": eval_time,
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
        "--n", type=int, nargs="+", default=[5],
        help="Node counts to test (default: 5).  N=6 takes ~8 min.",
    )
    parser.add_argument(
        "--random-trials", type=int, default=10,
        help="Number of random-ordering trials for baseline (default: 10)",
    )
    args = parser.parse_args()

    print(
        f"{'N':>3}  {'Parts':>6}  {'↓MI%':>7}  {'↑MI%':>7}  "
        f"{'Rand%':>7} (±{' std':>5})  {'phi(norm)':>10}  {'Eval(s)':>8}"
    )
    print("-" * 80)

    rows = []
    for n in sorted(args.n):
        if n not in _VERIFIED_NETWORKS:
            print(f"{n:>3}  (no validated network — skipping)")
            continue
        label = _VERIFIED_NETWORKS[n][0]
        print(f"{n:>3}  {label}  ...", end="\r", flush=True)
        result = run_diagnostic(n, random_trials=args.random_trials)
        r = result
        print(
            f"{r['n']:>3}  {r['n_partitions']:>6}  "
            f"{r['mi_rank_desc_pct']:>6.1f}%  "
            f"{r['mi_rank_asc_pct']:>6.1f}%  "
            f"{r['random_rank_pct_mean']:>6.1f}% (±{r['random_rank_std']/r['n_partitions']*100:>5.1f}%)  "
            f"{r['phi']:>10.4f}  {r['eval_time_s']:>8.1f}"
        )
        rows.append(result)

    print()
    print("Columns: ↓MI% = rank% when sorted high→low MI (descending)")
    print("         ↑MI% = rank% when sorted low→high MI (ascending)")
    print("         Rand% = mean rank% under random ordering (lower = MIP found earlier)")

    # ---- append to docs/speed_memory.md ---------------------------------
    docs_path = Path(__file__).parent.parent / "docs" / "speed_memory.md"
    if docs_path.exists() and rows:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        lines = [
            "",
            "---",
            "",
            f"## MI-Ordering Diagnostic ({timestamp})",
            "",
            "Tests whether sorting partitions by **pairwise MI(X_i; X_j)** proxy",
            "(summed over severed edges) concentrates the MIP in early ranks.",
            "↓MI = high MI first; ↑MI = low MI first (ascending).  Lower rank% = MIP",
            "found earlier = greater potential speedup from early stopping.",
            "",
            "| N | Partitions | ↓MI rank% | ↑MI rank% | Rand rank% | phi (norm) |",
            "|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r['n']} | {r['n_partitions']} | "
                f"{r['mi_rank_desc_pct']:.1f}% | "
                f"{r['mi_rank_asc_pct']:.1f}% | "
                f"{r['random_rank_pct_mean']:.1f}% ± {r['random_rank_std']/r['n_partitions']*100:.1f}% | "
                f"{r['phi']:.4f} |"
            )
        lines.append("")
        lines.append(
            "_MI rank_: 1-based rank (as %) at which the MIP is first reached under "
            "each ordering.  _Rand rank_: mean rank% across "
            f"{args.random_trials} random-order trials.  phi(norm) = normalized phi "
            "= phi × normalization_factor (the quantity sia() minimizes)."
        )
        lines.append("")

        with open(docs_path, "a") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Results appended to {docs_path}")


if __name__ == "__main__":
    main()
