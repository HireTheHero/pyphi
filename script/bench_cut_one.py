"""B4 Baseline — CUT_ONE Single-Node-Isolation Partitions (IIT 4.0).

The CUT_ONE approximation evaluates only the N partitions that isolate a single
node from all others.  In IIT 3.0 this is available via the registered scheme
``DIRECTED_BI_CUT_ONE``; in IIT 4.0 (``SET_UNI/BI``) no equivalent scheme is
registered.

This script implements B4 for IIT 4.0 by filtering the full ``SET_UNI/BI``
partition list to retain only 2-part ``GeneralSetPartition`` objects where one
group is a singleton.  Each singleton isolation has 3² = 9 direction-assignment
variants (cause/effect/bidirectional for each of the 2 parts), giving 3N total
candidates instead of Bell(N).

Refs
----
- Oizumi et al. (2014) https://doi.org/10.1371/journal.pcbi.1003588
  (original phi; single-node cut approximation)
- Mayner et al. (2018) https://arxiv.org/abs/1712.09644
  (PyPhi; ``CUT_ONE_APPROXIMATION`` for IIT 3.0)

Complexity
----------
- Partition generation: O(Bell_N) — we filter the full list (unavoidable given
  no dedicated generator); the filter itself is O(Bell_N).
- GID evaluations: 3N  (vs Bell_N exhaustive)
- For N=5: 45 candidates vs 1,061 → 24× reduction
- For N=6: 54 candidates vs 7,896 → 146× reduction

Usage
-----
    uv run python script/bench_cut_one.py
    uv run python script/bench_cut_one.py --n 4 5
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
# CUT_ONE filter for SET_UNI/BI
# ---------------------------------------------------------------------------

def cut_one_partitions(node_indices: tuple, all_partitions: list) -> list:
    """Filter ``all_partitions`` (SET_UNI/BI) to singleton-isolation 2-part cuts.

    A CUT_ONE partition isolates exactly one node from all others.  In the
    ``GeneralSetPartition`` representation this means ``set_partition`` has
    exactly 2 groups, one of which is a length-1 list.

    For N nodes there are N such groupings, each appearing in 3² = 9
    direction-assignment variants → 3N total candidates.

    Parameters
    ----------
    node_indices   : ordered tuple of original node indices for the subsystem
    all_partitions : list from ``system_partitions(node_indices, 'SET_UNI/BI')``

    Returns
    -------
    list of ``GeneralSetPartition`` objects with a singleton part
    """
    result = []
    for part in all_partitions:
        sp = getattr(part, "set_partition", None)
        if sp is None:
            continue
        if len(sp) == 2 and (len(sp[0]) == 1 or len(sp[1]) == 1):
            result.append(part)
    return result


# ---------------------------------------------------------------------------
# Core diagnostic
# ---------------------------------------------------------------------------

def run_cut_one_diagnostic(n: int) -> dict:
    """Run CUT_ONE diagnostic for the N-node benchmark network (IIT 4.0).

    Steps
    -----
    1. Generate full SET_UNI/BI partition list.
    2. Filter to singleton-isolation 2-part partitions (CUT_ONE set).
    3. Run ``sia()`` over CUT_ONE candidates (exact GID).
    4. Run exhaustive ``sia()`` over full list for ground-truth MIP.
    5. Report: #candidates, speedup, phi match, and whether CUT_ONE finds the MIP.

    Note on accuracy
    ----------------
    The CUT_ONE result gives an **upper bound** on phi: the true MIP phi is ≤ the
    minimum phi over CUT_ONE partitions, because the MIP may be a multi-part
    partition that severs more connections per unit normalization.  When the MIP
    is itself a singleton-isolation partition, CUT_ONE is exact.

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

    # ---- full partition list -----------------------------------------------
    t0 = time.perf_counter()
    all_parts = list(system_partitions(node_indices, partition_scheme="SET_UNI/BI"))
    t_gen = time.perf_counter() - t0
    n_all = len(all_parts)

    # ---- CUT_ONE filter -------------------------------------------------------
    cut_one_parts = cut_one_partitions(node_indices, all_parts)
    n_cut_one = len(cut_one_parts)
    # Expected: 3 * N — the unique() filter collapses the 3² = 9 direction combos
    # of a 2-part partition down to 3 distinct cut matrices per grouping:
    # (cut singleton→rest, cut rest→singleton, cut both directions).
    expected = 3 * n

    # ---- CUT_ONE sia -----------------------------------------------------------
    t0 = time.perf_counter()
    co_sia = nb.sia(subsys, partitions=iter(cut_one_parts))
    t_cut_one = time.perf_counter() - t0
    co_phi = co_sia.phi
    co_norm_phi = co_sia.normalized_phi
    co_mip_sp = getattr(co_sia.partition, "set_partition", None)
    co_mip_nparts = len(co_mip_sp) if co_mip_sp is not None else None

    # ---- exhaustive sia --------------------------------------------------------
    t0 = time.perf_counter()
    full_sia = nb.sia(subsys, partitions=iter(all_parts))
    t_full = time.perf_counter() - t0
    full_phi = full_sia.phi
    full_norm_phi = full_sia.normalized_phi
    full_mip_sp = getattr(full_sia.partition, "set_partition", None)
    full_mip_nparts = len(full_mip_sp) if full_mip_sp is not None else None

    # ---- accuracy --------------------------------------------------------------
    phi_match = abs(co_phi - full_phi) < 1e-9
    # CUT_ONE ⊆ all_partitions, so min normalized_phi over CUT_ONE ≥ min over all.
    # This is the correct upper-bound relationship (on normalized_phi, not phi,
    # because phi values across partitions are not directly comparable — different
    # partitions sever different numbers of connections and have different
    # normalization_factor values).
    co_is_upper_bound = co_norm_phi >= full_norm_phi - 1e-9
    speedup = n_all / max(n_cut_one, 1)

    return {
        "n": n,
        "n_all": n_all,
        "n_cut_one": n_cut_one,
        "expected_cut_one": expected,
        "speedup": speedup,
        "t_gen": t_gen,
        "co_phi": co_phi,
        "co_norm_phi": co_norm_phi,
        "co_mip_nparts": co_mip_nparts,
        "t_cut_one": t_cut_one,
        "full_phi": full_phi,
        "full_norm_phi": full_norm_phi,
        "full_mip_nparts": full_mip_nparts,
        "t_full": t_full,
        "phi_match": phi_match,
        "co_is_upper_bound": co_is_upper_bound,
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

    rows = []
    for n in sorted(args.n):
        if n not in _VERIFIED_NETWORKS:
            print(f"N={n}: no validated network — skipping")
            continue
        label = _VERIFIED_NETWORKS[n][0]
        print(f"  Running N={n} ({label}) ...", end="\r", flush=True)
        r = run_cut_one_diagnostic(n)
        rows.append(r)

    print()
    print("=== B4 CUT_ONE Diagnostic (IIT 4.0, SET_UNI/BI) ===")
    print()
    hdr = (
        f"{'N':>3}  {'All':>6}  {'CO':>4}  {'Exp':>4}  {'Speedup':>8}  "
        f"{'UpperBound':>10}  {'PhiMatch':>8}  "
        f"{'True phi':>9}  {'CO phi':>9}  "
        f"{'MIP#(true)':>10}  {'MIP#(CO)':>8}  "
        f"{'T_full(s)':>9}  {'T_CO(s)':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['n']:>3}  {r['n_all']:>6}  {r['n_cut_one']:>4}  {r['expected_cut_one']:>4}  "
            f"{r['speedup']:>7.0f}×  "
            f"{'✓' if r['co_is_upper_bound'] else '✗':>10}  "
            f"{'✓' if r['phi_match'] else '✗':>8}  "
            f"{r['full_phi']:>9.4f}  {r['co_phi']:>9.4f}  "
            f"{r['full_mip_nparts']:>10}  {r['co_mip_nparts']:>8}  "
            f"{r['t_full']:>9.2f}  {r['t_cut_one']:>7.3f}"
        )
    print()
    print("All       = total SET_UNI/BI partition count")
    print("CO        = CUT_ONE candidates (singleton-isolation 2-part partitions)")
    print("Exp       = expected 3×N candidates (after deduplication)")
    print("UpperBound= CO phi >= true phi (CUT_ONE is always an upper bound)")
    print("PhiMatch  = CO phi == exhaustive phi (CUT_ONE finds the exact MIP)")
    print("MIP#      = number of parts in the MIP partition")

    # ---- Append to docs/speed_memory.md ------------------------------------
    docs_path = Path(__file__).parent.parent / "docs" / "speed_memory.md"
    if docs_path.exists() and rows:
        import datetime
        ts = datetime.datetime.now().strftime("%Y-%m-%d")
        lines = [
            "",
            "---",
            "",
            f"## B4 CUT_ONE Diagnostic (IIT 4.0, {ts})",
            "",
            "Filter the full ``SET_UNI/BI`` partition list to 2-part",
            "``GeneralSetPartition`` objects where one group is a singleton node.",
            "Each singleton isolation has 3² = 9 direction-assignment variants",
            "(cause/effect/bidirectional for each part), giving 3N total candidates.",
            "Candidates are evaluated with exact GID; result compared against",
            "exhaustive ``sia()``.",
            "",
            "| N | All parts | CO parts | Expected | Speedup | UpperBound | PhiMatch | True phi | CO phi | MIP#(true) | MIP#(CO) | T_full (s) | T_CO (s) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r['n']} | {r['n_all']} | {r['n_cut_one']} | {r['expected_cut_one']} | "
                f"{r['speedup']:.0f}× | "
                f"{'✓' if r['co_is_upper_bound'] else '✗'} | "
                f"{'✓' if r['phi_match'] else '✗'} | "
                f"{r['full_phi']:.4f} | {r['co_phi']:.4f} | "
                f"{r['full_mip_nparts']} | {r['co_mip_nparts']} | "
                f"{r['t_full']:.2f} | {r['t_cut_one']:.3f} |"
            )
        lines += [
            "",
            "_UpperBound_: CO phi ≥ true phi — always holds (CUT_ONE is a superset of the MIP search space only when the MIP is a singleton cut).",
            "_PhiMatch_: exact MIP found within CUT_ONE candidates.",
            "_MIP#(CO)_: number of parts in the best CUT_ONE partition (always 2).",
            "",
        ]
        with open(docs_path, "a") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Results appended to {docs_path}")


if __name__ == "__main__":
    main()
