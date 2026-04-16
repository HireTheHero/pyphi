"""B2 Baseline — Queyranne's Algorithm for Minimum-MI Bipartition.

Queyranne (1998) finds the minimum of any symmetric submodular function over
all bipartitions of a ground set V in O(N³) oracle calls.  Mutual information
MI(X_A; X_B) is symmetric and submodular, so the algorithm is exact for the
minimum-MI bipartition.  GID (IIT 4.0) is NOT submodular, so this is a heuristic
proxy: the minimum-MI bipartition is evaluated with exact GID to produce a
candidate MIP, then compared against the true MIP from exhaustive sia().

Accuracy claim in literature (Kitazono et al. 2018): Queyranne finds the correct
GID-MIP >98% of the time even though GID is not submodular.

Algorithm sketch (Queyranne 1998 / Stoer-Wagner style)
------------------------------------------------------
For a symmetric submodular f, each "phase" builds a greedy ordering of the
current node set starting from an arbitrary root, always adding the node with
the highest f(current_A ∪ {v}) key.  The last node added ("pendant") defines
a candidate minimum cut: {pendant} vs {everything else}.  After N-1 phases the
globally minimum cut over all bipartitions has been found.

Oracle: f(S) = MI(X_S; X_{V\\S}) computed from the stationary distribution.

Complexity
----------
- Stationary distribution: O(4^N) (one power-iteration pass over 2^N × 2^N TPM)
- Per oracle call: O(2^N) (marginalise stationary dist to subset S)
- Oracle calls: O(N³) across all phases
- GID evaluations: 1 per bipartition candidate (vs. O(2^N) for exhaustive search)

Usage
-----
    uv run python script/bench_queyranne.py
    uv run python script/bench_queyranne.py --n 4 5 6
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Inline network definitions (shared with bench_scaling.py / bench_mi_ordering.py)
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
# Stationary distribution
# ---------------------------------------------------------------------------

def stationary_distribution(tpm_sbs: np.ndarray, max_iter: int = 2000, tol: float = 1e-12) -> np.ndarray:
    """Estimate stationary distribution by power iteration."""
    p = np.ones(tpm_sbs.shape[0]) / tpm_sbs.shape[0]
    for _ in range(max_iter):
        p_new = p @ tpm_sbs
        if np.max(np.abs(p_new - p)) < tol:
            return p_new
        p = p_new
    # Cyclic CA: return time-average of last 100 steps
    p_avg = np.zeros_like(p)
    for _ in range(100):
        p = p @ tpm_sbs
        p_avg += p
    return p_avg / 100.0


# ---------------------------------------------------------------------------
# MI oracle  f(S) = MI(X_S ; X_{V\S})
# ---------------------------------------------------------------------------

def _mi_oracle(S_sorted: list[int], comp_sorted: list[int], p_stat: np.ndarray) -> float:
    """Compute MI(X_S; X_complement) from the stationary distribution.

    Parameters
    ----------
    S_sorted, comp_sorted : sorted lists of node positions (bit positions in p_stat)
    p_stat : 1-D array of shape (2**N,)

    Returns
    -------
    float : MI in nats, clamped to >= 0
    """
    n_S = 2 ** len(S_sorted)
    n_C = 2 ** len(comp_sorted)
    joint = np.zeros((n_S, n_C))
    for state_idx, p in enumerate(p_stat):
        if p == 0.0:
            continue
        x_S = sum(((state_idx >> s) & 1) << i for i, s in enumerate(S_sorted))
        x_C = sum(((state_idx >> c) & 1) << i for i, c in enumerate(comp_sorted))
        joint[x_S, x_C] += p
    p_S = joint.sum(axis=1, keepdims=True)
    p_C = joint.sum(axis=0, keepdims=True)
    outer = p_S * p_C
    mask = (joint > 0) & (outer > 0)
    mi = float(np.sum(joint[mask] * np.log(joint[mask] / outer[mask])))
    return max(0.0, mi)


# ---------------------------------------------------------------------------
# Queyranne's algorithm
# ---------------------------------------------------------------------------

def queyranne_min_mi_bipartition(N: int, p_stat: np.ndarray) -> tuple[frozenset, frozenset, float]:
    """Find the bipartition (A, B) of {0..N-1} minimising MI(X_A; X_B).

    Implements Queyranne (1998) symmetric submodular function minimisation.
    Each phase builds a greedy maximum-key ordering, records the pendant node
    as a candidate cut, then contracts the pendant pair.

    Parameters
    ----------
    N       : number of nodes
    p_stat  : 1-D stationary distribution of length 2**N

    Returns
    -------
    (A, B, mi_value) where A ∪ B = {0..N-1}, A ∩ B = ∅, |A| >= 1, |B| >= 1
    """
    all_nodes = frozenset(range(N))

    # Each "contracted node" i has an associated group of original nodes.
    # alive : list of contracted-node IDs currently in the graph.
    # groups[i] : frozenset of original node indices in contracted node i.
    alive: list[int] = list(range(N))
    groups: list[frozenset] = [frozenset({i}) for i in range(N)]

    best_mi = float("inf")
    best_A: frozenset = frozenset({0})

    for _phase in range(N - 1):
        # ---- Greedy ordering to find pendant pair (s, t) ------------------
        # Start from the first alive contracted node
        start = alive[0]
        A_orig: frozenset = groups[start]   # union of original nodes in A so far
        in_A = {start}
        not_in_A = [v for v in alive if v != start]

        # key[v] = f(A_orig ∪ groups[v])  (the submodular function value)
        key: dict[int, float] = {}
        for v in not_in_A:
            cand = sorted(A_orig | groups[v])
            comp = sorted(all_nodes - (A_orig | groups[v]))
            key[v] = _mi_oracle(cand, comp, p_stat)

        s = start  # second-to-last added
        t = start  # last added (pendant)

        while not_in_A:
            # Add the most tightly connected node
            u = max(not_in_A, key=lambda v: key[v])
            s = t
            t = u
            in_A.add(u)
            not_in_A.remove(u)
            A_orig = A_orig | groups[u]

            # Update keys for remaining nodes
            for v in not_in_A:
                cand = sorted(A_orig | groups[v])
                comp = sorted(all_nodes - (A_orig | groups[v]))
                key[v] = _mi_oracle(cand, comp, p_stat)

        # ---- Pendant cut: {groups[t]} vs {everything else} ----------------
        cut_A = groups[t]
        cut_B = all_nodes - cut_A
        cand_sorted = sorted(cut_A)
        comp_sorted = sorted(cut_B)
        cut_mi = _mi_oracle(cand_sorted, comp_sorted, p_stat)

        if cut_mi < best_mi:
            best_mi = cut_mi
            best_A = cut_A

        # ---- Merge pendant t into its pair s --------------------------------
        groups[s] = groups[s] | groups[t]
        alive.remove(t)

    best_B = all_nodes - best_A
    return best_A, best_B, best_mi


# ---------------------------------------------------------------------------
# Map Queyranne bipartition → matching GeneralSetPartition objects
# ---------------------------------------------------------------------------

def partitions_for_bipartition(
    bipart_A: frozenset,
    bipart_B: frozenset,
    node_indices: tuple,
    all_partitions: list,
) -> list:
    """Return all partition objects whose node groups match bipartition (A, B).

    Handles two partition representations:

    - **GeneralSetPartition** (SET_UNI/BI): has ``set_partition`` — a list of
      lists of position indices within ``node_indices``.  We convert A/B to
      positions and match the two-group case.

    - **Cut** (DIRECTED_BI): has ``from_nodes`` / ``to_nodes`` — tuples of
      original node indices.  A Cut matches (A, B) if {from_nodes, to_nodes}
      == {A, B} as sets (either direction).

    Parameters
    ----------
    bipart_A, bipart_B : frozensets of original node indices
    node_indices       : ordered tuple of node indices for this subsystem
    all_partitions     : list from system_partitions(node_indices)

    Returns
    -------
    list of partition objects that are bipartitions of (A, B)
    """
    pos_map = {nidx: pos for pos, nidx in enumerate(node_indices)}
    A_pos = frozenset(pos_map[n] for n in bipart_A)
    B_pos = frozenset(pos_map[n] for n in bipart_B)
    target_pos = frozenset([A_pos, B_pos])

    matching = []
    for part in all_partitions:
        # --- GeneralSetPartition (SET_UNI/BI) ---
        sp = getattr(part, "set_partition", None)
        if sp is not None:
            if len(sp) == 2:
                groups = frozenset(frozenset(g) for g in sp)
                if groups == target_pos:
                    matching.append(part)
            continue

        # --- Cut (DIRECTED_BI) ---
        fn = getattr(part, "from_nodes", None)
        tn = getattr(part, "to_nodes", None)
        if fn is not None and tn is not None:
            part_A = frozenset(fn)
            part_B = frozenset(tn)
            if frozenset([part_A, part_B]) == frozenset([bipart_A, bipart_B]):
                matching.append(part)

    return matching


# ---------------------------------------------------------------------------
# Core diagnostic
# ---------------------------------------------------------------------------

def run_queyranne_diagnostic(n: int) -> dict:
    """Run Queyranne diagnostic for the N-node benchmark network.

    Two modes are tested in a single pass:

    Mode A — BI-only (bipartitions only, SYSTEM_PARTITION_TYPE='BI'):
        Queyranne's natural domain.  The MIP under BI-only is always a
        bipartition, so this tests whether Queyranne finds the correct one.

    Mode B — Full SET_UNI/BI (default IIT 4.0):
        Reports the true MIP partition type.  If the MIP is multi-part,
        Queyranne's bipartition candidate cannot equal it (fundamental limit).

    Steps
    -----
    1. Compute stationary distribution.
    2. Run Queyranne → minimum-MI bipartition (A, B).
    3. Mode A: exhaustive sia(BI-only) → best bipartition MIP.
       Check whether Queyranne predicts it.
    4. Mode B: exhaustive sia(SET_UNI/BI) → true IIT 4.0 MIP.
       Report MIP partition type (bipartition or multi-part).

    Returns
    -------
    dict with diagnostic results
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
    tpm_sbs = state_by_node2state_by_state(network.tpm)
    p_stat = stationary_distribution(tpm_sbs)

    # ---- Queyranne: minimum-MI bipartition --------------------------------
    t0 = time.perf_counter()
    q_A, q_B, q_mi = queyranne_min_mi_bipartition(n, p_stat)
    t_queyranne = time.perf_counter() - t0

    # =========================================================================
    # Mode A: BI-only (bipartitions only)
    # =========================================================================
    bi_parts = list(system_partitions(node_indices, partition_scheme="DIRECTED_BI"))
    n_bi_total = len(bi_parts)

    # Find which BI partitions match the Queyranne bipartition
    q_parts = partitions_for_bipartition(q_A, q_B, node_indices, bi_parts)
    n_q_parts = len(q_parts)

    # Exhaustive BI-only sia
    t0 = time.perf_counter()
    bi_sia = nb.sia(subsys, partitions=iter(bi_parts))
    t_bi_exhaustive = time.perf_counter() - t0
    bi_true_norm_phi = bi_sia.normalized_phi

    # MIP bipartition from exhaustive BI search
    bi_mip_part = bi_sia.partition
    bi_mip_sp = getattr(bi_mip_part, "set_partition", None)
    bi_mip_fn = getattr(bi_mip_part, "from_nodes", None)
    bi_mip_tn = getattr(bi_mip_part, "to_nodes", None)

    if bi_mip_sp is not None and len(bi_mip_sp) == 2:
        # GeneralSetPartition: compare via position sets
        pos_map = {nidx: pos for pos, nidx in enumerate(node_indices)}
        q_A_pos = frozenset(pos_map[x] for x in q_A)
        q_B_pos = frozenset(pos_map[x] for x in q_B)
        true_A_pos = frozenset(bi_mip_sp[0])
        true_B_pos = frozenset(bi_mip_sp[1])
        bi_bipart_match = frozenset([q_A_pos, q_B_pos]) == frozenset([true_A_pos, true_B_pos])
    elif bi_mip_fn is not None and bi_mip_tn is not None:
        # Cut: compare node index sets directly
        true_A = frozenset(bi_mip_fn)
        true_B = frozenset(bi_mip_tn)
        bi_bipart_match = frozenset([true_A, true_B]) == frozenset([q_A, q_B])
    else:
        bi_bipart_match = False

    # Queyranne candidate sia (BI partitions for q_A, q_B)
    if q_parts:
        q_sia = nb.sia(subsys, partitions=iter(q_parts))
        q_norm_phi = q_sia.normalized_phi
        bi_phi_match = abs(q_norm_phi - bi_true_norm_phi) < 1e-9
    else:
        q_norm_phi = None
        bi_phi_match = False

    bi_speedup = n_bi_total / max(n_q_parts, 1)

    # =========================================================================
    # Mode B: Full SET_UNI/BI
    # =========================================================================
    full_parts = list(system_partitions(node_indices, partition_scheme="SET_UNI/BI"))
    n_full_total = len(full_parts)

    t0 = time.perf_counter()
    full_sia = nb.sia(subsys, partitions=iter(full_parts))
    t_full_exhaustive = time.perf_counter() - t0
    full_true_phi = full_sia.phi
    full_true_norm_phi = full_sia.normalized_phi

    full_mip_sp = getattr(full_sia.partition, "set_partition", None)
    full_mip_nparts = len(full_mip_sp) if full_mip_sp else None
    full_mip_is_bipart = full_mip_nparts == 2

    return {
        "n": n,
        # Queyranne
        "q_mi": q_mi,
        "queyranne_time_s": t_queyranne,
        "q_A": sorted(q_A),
        "q_B": sorted(q_B),
        # Mode A (BI-only)
        "n_bi_total": n_bi_total,
        "n_q_parts": n_q_parts,
        "bi_speedup": bi_speedup,
        "bi_true_norm_phi": bi_true_norm_phi,
        "q_norm_phi": q_norm_phi,
        "bi_bipart_match": bi_bipart_match,
        "bi_phi_match": bi_phi_match,
        "t_bi_exhaustive": t_bi_exhaustive,
        # Mode B (full SET_UNI/BI)
        "n_full_total": n_full_total,
        "full_true_phi": full_true_phi,
        "full_true_norm_phi": full_true_norm_phi,
        "full_mip_nparts": full_mip_nparts,
        "full_mip_is_bipart": full_mip_is_bipart,
        "t_full_exhaustive": t_full_exhaustive,
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
    args = parser.parse_args()

    print("=== Mode A: BI-only (Queyranne's natural domain) ===")
    hdr_a = (
        f"{'N':>3}  {'BI-parts':>8}  {'Q-parts':>7}  {'Speedup':>8}  "
        f"{'BiMatch':>7}  {'PhiMatch':>8}  {'T_BI(s)':>8}  {'T_Q(s)':>7}"
    )
    print(hdr_a)
    print("-" * len(hdr_a))

    print()
    print("=== Mode B: Full SET_UNI/BI (default IIT 4.0) ===")
    hdr_b = (
        f"{'N':>3}  {'Total-parts':>11}  {'True phi':>9}  "
        f"{'MIP #parts':>10}  {'T_full(s)':>9}"
    )
    print(hdr_b)
    print("-" * len(hdr_b))

    rows = []
    for n in sorted(args.n):
        if n not in _VERIFIED_NETWORKS:
            print(f"{n:>3}  (no validated network — skipping)")
            continue
        label = _VERIFIED_NETWORKS[n][0]
        print(f"  Running N={n} ({label}) ...", end="\r", flush=True)

        r = run_queyranne_diagnostic(n)
        rows.append(r)

    # Print Mode A
    print("=== Mode A: BI-only (Queyranne's natural domain) ===")
    print(hdr_a)
    print("-" * len(hdr_a))
    for r in rows:
        print(
            f"{r['n']:>3}  {r['n_bi_total']:>8}  {r['n_q_parts']:>7}  "
            f"{r['bi_speedup']:>7.0f}×  "
            f"{'✓' if r['bi_bipart_match'] else '✗':>7}  "
            f"{'✓' if r['bi_phi_match'] else '✗':>8}  "
            f"{r['t_bi_exhaustive']:>8.2f}  {r['queyranne_time_s']:>7.3f}"
        )
    print()
    print("BiMatch  = Queyranne bipartition == BI-exhaustive MIP bipartition")
    print("PhiMatch = normalized phi from Q-candidates equals BI-exhaustive phi")

    # Print Mode B
    print()
    print("=== Mode B: Full SET_UNI/BI (default IIT 4.0) ===")
    print(hdr_b)
    print("-" * len(hdr_b))
    for r in rows:
        print(
            f"{r['n']:>3}  {r['n_full_total']:>11}  {r['full_true_phi']:>9.4f}  "
            f"{r['full_mip_nparts']:>10}  {r['t_full_exhaustive']:>9.1f}"
        )
    print()
    print("MIP #parts = number of groups in the true MIP partition")
    print("           = 2 means bipartition (Queyranne applicable)")
    print("           > 2 means multi-part (Queyranne cannot find it)")

    # ---- append to docs/speed_memory.md ---------------------------------
    docs_path = Path(__file__).parent.parent / "docs" / "speed_memory.md"
    if docs_path.exists() and rows:
        import datetime
        ts = datetime.datetime.now().strftime("%Y-%m-%d")
        lines = [
            "",
            "---",
            "",
            f"## B2 Queyranne Diagnostic ({ts})",
            "",
            "**Mode A (BI-only):** Queyranne oracle finds minimum-MI bipartition in O(N³)",
            "calls; those candidates are evaluated with exact GID.  Tests whether",
            "Queyranne identifies the correct MIP within the restricted bipartition search space.",
            "",
            "**Mode B (full SET_UNI/BI):** Reports the true IIT 4.0 MIP partition type.",
            "If MIP #parts > 2 Queyranne's bipartition-only search cannot reach it.",
            "",
            "### Mode A — BI-only",
            "",
            "| N | BI parts | Q parts | Speedup | BiMatch | PhiMatch | T_BI (s) | T_Q (s) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r['n']} | {r['n_bi_total']} | {r['n_q_parts']} | "
                f"{r['bi_speedup']:.0f}× | "
                f"{'✓' if r['bi_bipart_match'] else '✗'} | "
                f"{'✓' if r['bi_phi_match'] else '✗'} | "
                f"{r['t_bi_exhaustive']:.2f} | {r['queyranne_time_s']:.3f} |"
            )
        lines += [
            "",
            "### Mode B — Full SET_UNI/BI",
            "",
            "| N | Total parts | True phi | MIP #parts | T_full (s) |",
            "|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r['n']} | {r['n_full_total']} | {r['full_true_phi']:.4f} | "
                f"{r['full_mip_nparts']} | {r['t_full_exhaustive']:.1f} |"
            )
        lines += [
            "",
            "_MIP #parts > 2_: multi-part MIP — Queyranne bipartition search cannot reach it.",
            "_Speedup_: ratio of BI partition count to Queyranne candidate count.",
            "",
        ]

        with open(docs_path, "a") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Results appended to {docs_path}")


if __name__ == "__main__":
    main()
