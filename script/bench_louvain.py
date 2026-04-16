"""B3 Baseline — Louvain Community Detection for MIP-partition Search.

Blondel et al. (2008) https://doi.org/10.1088/1742-5468/2008/10/P10008
Nilsen et al. (2019)   https://doi.org/10.3390/e21050525

Algorithm
---------
1. Build a weighted graph on the N-node system: edge weight = MI(X_i; X_j)
   under the stationary distribution.
2. Run Louvain community detection (networkx implementation, fixed seed).
   Output: k communities C_0, …, C_{k-1}  (frozensets of original node indices).
3. Search the full SET_UNI/BI partition list for GeneralSetPartition objects
   whose set_partition position-groups match the Louvain communities exactly.
   These are the k! × 3^k direction-assignment variants of the same grouping.
4. Evaluate only those O(3^k) partitions with exact GID.  Compare result
   against the exhaustive MIP (evaluated over all partitions).

Rationale
---------
The minimum-phi partition tends to sever edges with high causal information.
Louvain maximises graph modularity — communities with high internal MI.
The MIP therefore lies near the Louvain cut boundaries.  The heuristic works
because minimising phi ≈ finding the cut that most reduces shared information
between parts, i.e., the same cut that separates strong communities.

Expected accuracy: Nilsen et al. (2019) report r ≈ 0.95 correlation between
Louvain-phi and exact phi on binary systems with N ≤ 8.

Complexity
----------
- Stationary distribution: O(4^N) — power iteration over SBS TPM
- Pairwise MI matrix: O(N² × 2^N)
- Louvain community detection: O(N log N) (small N, negligible)
- Partition matching: O(n_all_partitions × k) — linear scan of partition list
- GID evaluation: O(3^k) evaluations (vs O(Bell_N) for exhaustive)

Usage
-----
    uv run python script/bench_louvain.py
    uv run python script/bench_louvain.py --n 4 5
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Inline network definitions (shared with bench_scaling / bench_queyranne)
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
# Stationary distribution (reused from bench_queyranne.py)
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
# Pairwise MI matrix
# ---------------------------------------------------------------------------

def pairwise_mi_matrix(n: int, p_stat: np.ndarray) -> np.ndarray:
    """Compute an N×N matrix of pairwise mutual informations.

    M[i, j] = I(X_i ; X_j)  under the stationary distribution p_stat.

    Uses a 2×2 joint distribution for each pair, marginalised from p_stat.
    """
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            joint = np.zeros((2, 2))
            for s, p in enumerate(p_stat):
                if p == 0.0:
                    continue
                xi = (s >> i) & 1
                xj = (s >> j) & 1
                joint[xi, xj] += p
            pi = joint.sum(axis=1, keepdims=True)
            pj = joint.sum(axis=0, keepdims=True)
            outer = pi * pj
            mask = (joint > 0) & (outer > 0)
            mi = float(np.sum(joint[mask] * np.log2(joint[mask] / outer[mask])))
            M[i, j] = M[j, i] = max(0.0, mi)
    return M


# ---------------------------------------------------------------------------
# Louvain community detection
# ---------------------------------------------------------------------------

def louvain_communities_from_mi(mi_matrix: np.ndarray, seed: int = 42) -> list[frozenset]:
    """Run Louvain on MI-weighted graph; return list of frozensets of node indices."""
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    n = mi_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            w = float(mi_matrix[i, j])
            if w > 0.0:
                G.add_edge(i, j, weight=w)

    # louvain_communities returns a list of sets of node indices
    comms = louvain_communities(G, weight="weight", seed=seed)
    return [frozenset(c) for c in comms]


# ---------------------------------------------------------------------------
# Map Louvain communities → matching GeneralSetPartition objects
# ---------------------------------------------------------------------------

def partitions_for_communities(
    communities: list[frozenset],
    node_indices: tuple,
    all_partitions: list,
) -> list:
    """Return all GeneralSetPartition objects whose grouping matches ``communities``.

    A ``GeneralSetPartition`` records its grouping as ``set_partition`` — a list
    of lists of **position** indices within ``node_indices``.  We convert each
    Louvain community (a frozenset of original node indices) to its position
    equivalents and look for exact matches.

    Note: One community grouping maps to multiple partitions because each
    grouping has 3^k direction-assignment variants (cause / effect / bidirectional
    for each part).  All matching variants are returned so that ``sia()`` can
    pick the minimum-phi one among them.

    Parameters
    ----------
    communities    : Louvain output — list of frozensets of original node indices
    node_indices   : ordered tuple of original node indices for this subsystem
    all_partitions : full list from ``system_partitions(node_indices)``

    Returns
    -------
    list of GeneralSetPartition objects that share the same node grouping
    """
    pos_map = {nidx: pos for pos, nidx in enumerate(node_indices)}
    # Target: frozenset of frozensets of positions
    target = frozenset(
        frozenset(pos_map[ni] for ni in comm) for comm in communities
    )

    matching = []
    for part in all_partitions:
        sp = getattr(part, "set_partition", None)
        if sp is None:
            continue
        groups = frozenset(frozenset(g) for g in sp)
        if groups == target:
            matching.append(part)
    return matching


# ---------------------------------------------------------------------------
# Core diagnostic
# ---------------------------------------------------------------------------

def run_louvain_diagnostic(n: int, seed: int = 42) -> dict:
    """Run Louvain community detection diagnostic for the N-node benchmark network.

    Steps
    -----
    1. Compute stationary distribution → pairwise MI matrix.
    2. Build MI-weighted graph; run Louvain → k communities.
    3. Map communities to matching GeneralSetPartition objects (all direction
       variants of the same grouping).
    4. Run ``sia()`` over those candidates (exact GID for each).
    5. Run exhaustive ``sia()`` (full SET_UNI/BI) for ground-truth MIP.
    6. Report: #candidates, speedup, phi accuracy, MIP partition type.

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

    # ---- stationary distribution and pairwise MI ---------------------------
    tpm_sbs = state_by_node2state_by_state(network.tpm)
    p_stat = stationary_distribution(tpm_sbs)
    mi_mat = pairwise_mi_matrix(n, p_stat)

    # ---- Louvain communities -----------------------------------------------
    t0 = time.perf_counter()
    communities = louvain_communities_from_mi(mi_mat, seed=seed)
    t_louvain = time.perf_counter() - t0
    n_communities = len(communities)

    # ---- full partition list (SET_UNI/BI) ----------------------------------
    all_parts = list(system_partitions(node_indices, partition_scheme="SET_UNI/BI"))
    n_all = len(all_parts)

    # ---- candidate partitions matching Louvain communities -----------------
    matching = partitions_for_communities(communities, node_indices, all_parts)
    n_candidates = len(matching)

    # ---- Louvain candidate sia (exact GID over candidates) -----------------
    t0 = time.perf_counter()
    if matching:
        louvain_sia = nb.sia(subsys, partitions=iter(matching))
        louvain_phi = louvain_sia.phi
        louvain_norm_phi = louvain_sia.normalized_phi
        louvain_mip_sp = getattr(louvain_sia.partition, "set_partition", None)
        louvain_mip_nparts = len(louvain_mip_sp) if louvain_mip_sp is not None else None
    else:
        louvain_phi = louvain_norm_phi = None
        louvain_mip_nparts = None
    t_louvain_sia = time.perf_counter() - t0

    # ---- exhaustive sia (full SET_UNI/BI) ----------------------------------
    t0 = time.perf_counter()
    full_sia = nb.sia(subsys, partitions=iter(all_parts))
    t_full = time.perf_counter() - t0
    full_phi = full_sia.phi
    full_norm_phi = full_sia.normalized_phi
    full_mip_sp = getattr(full_sia.partition, "set_partition", None)
    full_mip_nparts = len(full_mip_sp) if full_mip_sp is not None else None

    # ---- accuracy metrics --------------------------------------------------
    phi_match = (louvain_phi is not None and abs(louvain_phi - full_phi) < 1e-9)
    # Louvain finds the MIP grouping if the community grouping matches the true MIP grouping
    if full_mip_sp is not None and matching:
        true_mip_groups = frozenset(frozenset(g) for g in full_mip_sp)
        pos_map = {nidx: pos for pos, nidx in enumerate(node_indices)}
        louvain_target = frozenset(
            frozenset(pos_map[ni] for ni in comm) for comm in communities
        )
        grouping_match = (louvain_target == true_mip_groups)
    else:
        grouping_match = False

    speedup = n_all / max(n_candidates, 1)

    return {
        "n": n,
        "n_communities": n_communities,
        "communities": [sorted(c) for c in communities],
        "t_louvain": t_louvain,
        "n_all": n_all,
        "n_candidates": n_candidates,
        "speedup": speedup,
        "louvain_phi": louvain_phi,
        "louvain_norm_phi": louvain_norm_phi,
        "louvain_mip_nparts": louvain_mip_nparts,
        "t_louvain_sia": t_louvain_sia,
        "full_phi": full_phi,
        "full_norm_phi": full_norm_phi,
        "full_mip_nparts": full_mip_nparts,
        "t_full": t_full,
        "phi_match": phi_match,
        "grouping_match": grouping_match,
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
    parser.add_argument("--seed", type=int, default=42, help="Louvain random seed")
    args = parser.parse_args()

    rows = []
    for n in sorted(args.n):
        if n not in _VERIFIED_NETWORKS:
            print(f"N={n}: no validated network — skipping")
            continue
        label = _VERIFIED_NETWORKS[n][0]
        print(f"  Running N={n} ({label}) ...", end="\r", flush=True)
        r = run_louvain_diagnostic(n, seed=args.seed)
        rows.append(r)

    # ---- Print results -------------------------------------------------------
    print()
    print("=== B3 Louvain Community Detection Diagnostic ===")
    print()
    hdr = (
        f"{'N':>3}  {'Comms':>5}  {'Cands':>5}  {'Speedup':>8}  "
        f"{'GrpMatch':>8}  {'PhiMatch':>8}  "
        f"{'True phi':>9}  {'Louv phi':>9}  "
        f"{'T_full(s)':>9}  {'T_louv(s)':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        louv_phi_str = f"{r['louvain_phi']:.4f}" if r["louvain_phi"] is not None else "N/A"
        print(
            f"{r['n']:>3}  {r['n_communities']:>5}  {r['n_candidates']:>5}  "
            f"{r['speedup']:>7.0f}×  "
            f"{'✓' if r['grouping_match'] else '✗':>8}  "
            f"{'✓' if r['phi_match'] else '✗':>8}  "
            f"{r['full_phi']:>9.4f}  {louv_phi_str:>9}  "
            f"{r['t_full']:>9.2f}  {r['t_louvain_sia']:>9.3f}"
        )
    print()
    print("Comms     = number of Louvain communities detected")
    print("Cands     = matching GeneralSetPartition objects (all direction variants)")
    print("Speedup   = n_all_partitions / n_candidates")
    print("GrpMatch  = Louvain community grouping matches true MIP grouping")
    print("PhiMatch  = phi from Louvain candidates equals exhaustive phi")
    print()
    for r in rows:
        print(f"  N={r['n']}: communities = {r['communities']}")
        print(f"         true MIP #parts = {r['full_mip_nparts']}, "
              f"Louvain MIP #parts (from candidates) = {r['louvain_mip_nparts']}")

    # ---- Append to docs/speed_memory.md ------------------------------------
    docs_path = Path(__file__).parent.parent / "docs" / "speed_memory.md"
    if docs_path.exists() and rows:
        import datetime
        ts = datetime.datetime.now().strftime("%Y-%m-%d")
        lines = [
            "",
            "---",
            "",
            f"## B3 Louvain Community Detection Diagnostic ({ts})",
            "",
            "Build a pairwise-MI-weighted graph on the N-node system; run Louvain",
            "community detection (networkx, fixed seed=42); map communities to",
            "matching `GeneralSetPartition` direction-variants; evaluate those with",
            "exact GID and compare against full exhaustive `sia()`.",
            "",
            "| N | Communities | Candidates | Speedup | GrpMatch | PhiMatch | True phi | Louv phi | T_full (s) | T_louv (s) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            louv_phi_str = f"{r['louvain_phi']:.4f}" if r["louvain_phi"] is not None else "N/A"
            lines.append(
                f"| {r['n']} | {r['n_communities']} | {r['n_candidates']} | "
                f"{r['speedup']:.0f}× | "
                f"{'✓' if r['grouping_match'] else '✗'} | "
                f"{'✓' if r['phi_match'] else '✗'} | "
                f"{r['full_phi']:.4f} | {louv_phi_str} | "
                f"{r['t_full']:.2f} | {r['t_louvain_sia']:.3f} |"
            )
        lines += [
            "",
            "_Communities_: Louvain community count (= partition arity if GrpMatch ✓).",
            "_Candidates_: GeneralSetPartition objects matching the Louvain grouping",
            "(all 3^k direction-assignment variants of the same node groups).",
            "_GrpMatch_: Louvain community grouping == true MIP node grouping.",
            "_PhiMatch_: phi computed from Louvain candidates == exhaustive phi.",
            "",
            f"Community assignments per network:",
        ]
        for r in rows:
            lines.append(f"- N={r['n']}: {r['communities']} (true MIP #parts={r['full_mip_nparts']})")
        lines.append("")

        with open(docs_path, "a") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Results appended to {docs_path}")


if __name__ == "__main__":
    main()
