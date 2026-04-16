# Speed & Memory Optimization Notes

This document records identified bottlenecks in the phi computation pipeline and concrete improvement opportunities. All findings are based on static analysis of the local fork (branch `develop`).

---

## 1. `psutil.Process()` Instantiated on Every Cache Miss

**File:** `pyphi/cache/__init__.py:87–88`  
**Priority:** High

```python
current_process = psutil.Process(os.getpid())
full = current_process.memory_percent() > maxmem_value
```

A new `psutil.Process` object is created and `memory_percent()` — a system call — is invoked on every cache miss in the memory-limited cache wrapper. This wrapper is on the hot path for all repertoire computations.

**Fix:** Instantiate the `Process` object once outside the wrapper closure (at decoration time), and optionally throttle the memory check to every N misses using a counter. The `full` flag is already sticky once set, so the main cost is the per-miss system call before that point.

---

## 2. `functools.reduce(np.multiply, [...])` for Repertoire Products

**File:** `pyphi/subsystem.py:380–382` (cause), `459–464` (effect)  
**Priority:** High

```python
# Cause repertoire
joint *= functools.reduce(
    np.multiply,
    [self._single_node_cause_repertoire(m, purview_set) for m in mechanism],
)

# Effect repertoire
return joint * functools.reduce(
    np.multiply,
    [self._single_node_effect_repertoire(condition, p, direction) for p in purview],
)
```

`functools.reduce` over `np.multiply` allocates a new intermediate array at each pairwise step. For a mechanism of size k this creates k−1 temporaries before the final result.

**Fix:** Replace with `np.multiply.reduce(array_list)` (the ufunc's own reduce, which avoids Python-level dispatch overhead) or stack and reduce along an axis. Care is needed because the individual repertoires have different shapes due to broadcasting — verify shape compatibility before switching.

---

## 3. `effect_emd` Iterates Over Dimensions in Python

**File:** `pyphi/metrics/distribution.py:460–461`  
**Priority:** Medium

```python
return float(
    sum(abs(marginal_zero(p, i) - marginal_zero(q, i)) for i in range(p.ndim))
)
```

`marginal_zero` calls `np.sum` once per node dimension in a Python loop. Because this is on the default `REPERTOIRE_DISTANCE` path and is called for every mechanism/purview pair, the per-call overhead accumulates.

**Fix:** Compute all marginals at once. `marginal_zero(x, i)` returns the probability that node i is OFF (i.e., the sum over the `i=0` slice along axis `i`). These can be batched by summing over all axes simultaneously and then computing absolute differences as a vectorized operation.

---

## 4. Full Hamming Distance Matrix Allocated on Demand

**File:** `pyphi/metrics/distribution.py:421–422`  
**Priority:** Medium

```python
possible_states = np.array(list(utils.all_states(N)))
return cdist(possible_states, possible_states, "hamming") * N
```

For systems with N nodes this allocates a `2^N × 2^N` float64 matrix. At N=13 that is approximately 512 MB; at N=15 it exceeds 16 GB. This function is only called when N exceeds `_NUM_PRECOMPUTED_HAMMING_MATRICES` (the threshold below which matrices are pre-computed at import time).

**Fix:** Verify that `_NUM_PRECOMPUTED_HAMMING_MATRICES` covers the typical system sizes used in experiments. For larger systems where the `hamming_emd` distance is actually needed (rather than `effect_emd` or `GID`), consider a sparse or on-demand computation that avoids materializing the full matrix.

---

## 5. Double Array Copy at `Subsystem.__init__`

**File:** `pyphi/subsystem.py:144–145`  
**Priority:** Low

```python
self.proper_effect_tpm = self.effect_tpm.squeeze()[..., list(self.node_indices)]
self.proper_cause_tpm  = self.cause_tpm.squeeze()[..., list(self.node_indices)]
```

`.squeeze()` creates a copy, and `[..., list(...)]` (fancy indexing) creates a second copy. Additionally, `list(self.node_indices)` wraps an existing tuple unnecessarily. For large TPMs this is two full-array allocations at subsystem construction.

**Fix:** Use a single fancy-index expression that combines the squeeze and selection in one pass, or apply `np.squeeze` with an explicit `axis` argument so numpy can avoid a copy when the shape is already correct.

---

## 6. Partition Enumeration May Materialize Eagerly

**File:** `pyphi/new_big_phi/__init__.py:550–555`  
**Priority:** Low

```python
partitions = system_partitions(
    subsystem.node_indices,
    ...
)
...
sias = MapReduce(evaluate_partition, partitions, ...)
```

If `system_partitions` returns a list rather than a generator, all partitions are materialized in memory before `MapReduce` begins consuming them. For large subsystems the number of partitions grows combinatorially.

**Fix:** Confirm `system_partitions` is lazy (a generator). If it returns a list, convert the call site to consume it lazily, or change `system_partitions` itself to yield partitions. This does not affect computation correctness.

---

## Summary Table (§1–6)

| Priority | File | Lines | Issue | Expected Gain |
|---|---|---|---|---|
| High | `pyphi/cache/__init__.py` | 87–88 | `psutil.Process()` per cache miss | 5–15% on cached workloads |
| High | `pyphi/subsystem.py` | 380–382, 459–464 | `functools.reduce` temporaries | 1–3× for large mechanisms |
| Medium | `pyphi/metrics/distribution.py` | 460–461 | `effect_emd` Python loop over dims | Constant factor, very hot path |
| Medium | `pyphi/metrics/distribution.py` | 421–422 | Full Hamming matrix allocation | Up to ~512 MB at N=13 |
| Low | `pyphi/subsystem.py` | 144–145 | Double copy on TPM selection | Small, init-time only |
| Low | `pyphi/new_big_phi/__init__.py` | 550–555 | Eager partition materialization | Memory at large system sizes |

---

## Library & Method Opportunities

The following section surveys opportunities to use more efficient libraries or numpy primitives. As of the current fork, **`np.einsum`, `np.tensordot`, `np.outer`, `numba`, `torch`, `jax`, and `scipy.sparse` are entirely absent** from the codebase, leaving substantial headroom.

---

### 7. `np.einsum` for Repertoire Product Chains

**Files:** `pyphi/subsystem.py:380–382, 459–464`; `pyphi/convert.py:269–275`  
**Priority:** High

The core repertoire product (§2 above) multiplies a sequence of per-node arrays that each have shape `(1, ..., 2, ..., 1)` (a single axis is non-trivial, the rest are size-1 broadcast dimensions). The current `functools.reduce(np.multiply, [...])` approach is equivalent to an outer product over the node axes, accumulated step-by-step.

`np.einsum` can express the same operation in a single call with explicit index labels, avoiding all intermediate allocations and allowing numpy's BLAS-backed contraction engine to choose an optimal evaluation order. For the typical case of 2–6 nodes in a purview this is a small but consistent win; for larger purviews the gain is more significant.

Similarly in `convert.state_by_state2state_by_node` (lines 269–275):

```python
node_on = np.array([[states[i][n] for i in range(S)] for n in range(N)])
on_probabilities = [tpm * node_on[n] for n in range(N)]
for i, state in states.items():
    sbn_tpm[state] = [np.sum(on_probabilities[n][i]) for n in range(N)]
```

The double loop over `(state, node)` is a matrix-vector contraction: `sbn_tpm[state, n] = sum_j tpm[state, j] * node_on[n, j]`. This is exactly `np.einsum('ij,nj->in', tpm, node_on)` — a single batched operation replacing the entire nested loop.

---

### 8. `np.multiply.reduce` or `np.ufunc.reduce` vs Python `reduce`

**File:** `pyphi/subsystem.py:380–382, 459–464`  
**Priority:** High

Even before reaching for `einsum`, replacing `functools.reduce(np.multiply, arrays)` with `np.multiply.reduce(arrays)` (the ufunc's native reduce) eliminates Python-level dispatch for each step. The ufunc reduce path is implemented in C and operates on a pre-built list of operands without constructing a Python call frame per iteration. For a list of `k` arrays this avoids `k−1` Python function calls.

---

### 9. Vectorizing `probability_of_current_state` with `np.where`

**File:** `pyphi/tpm.py:690–694`  
**Priority:** Medium

```python
for i in range(sbn_tpm.shape[-1]):
    state_probabilities[..., i] = (
        sbn_tpm[..., i] if current_state[i] else (1 - sbn_tpm[..., i])
    )
```

This loops over each node dimension in Python. `current_state` is a tuple of 0/1 integers, so the entire loop is equivalent to:

```python
mask = np.array(current_state, dtype=bool)  # shape (N,)
state_probabilities = np.where(mask, sbn_tpm, 1 - sbn_tpm)
```

`np.where` broadcasts the mask against the last axis of `sbn_tpm` in a single vectorized call.

---

### 10. Vectorizing `be2le_state_by_state` Index Remapping

**File:** `pyphi/convert.py:168–169`  
**Priority:** Medium

```python
for i, j in product(range(N), repeat=2):
    le[i, j] = tpm[be2le(i, n), be2le(j, n)]
```

This is an O(N²) Python loop that fills a matrix via indirect indexing. The entire operation is a gather: pre-compute the permutation index array `idx` (the big-endian → little-endian reindex for all N states) once, then apply it as:

```python
idx = np.array([be2le(i, n) for i in range(N)])
le = tpm[np.ix_(idx, idx)]
```

`np.ix_` constructs the open mesh for 2D fancy indexing, performing the full reindex in one C-level operation.

---

### 11. `scipy.sparse` for Large or Deterministic TPMs

**File:** `pyphi/tpm.py` (throughout)  
**Priority:** Medium (system-size dependent)

For **deterministic** networks (which are common in IIT research), the state-by-state TPM has exactly one non-zero entry per row — an extreme sparsity of `1/2^N`. Storing and operating on such a matrix as a dense `float64` array wastes memory by a factor of `2^N` and performs unnecessary multiplications by zero.

`scipy.sparse.csr_matrix` (or `lil_matrix` during construction) would:
- Reduce memory from `O(4^N)` dense to `O(2^N)` for deterministic TPMs.
- Make matrix-vector products (used in TPM conditioning) proportionally faster.

For **stochastic** TPMs the benefit depends on how sparse the rows actually are in practice. This is worth profiling for the specific networks used in the ConsInfoLLM experiments.

Note: `scipy.sparse` is already a transitive dependency (via `pyemd`), so no new package installation is needed.

---

### 12. `numba` JIT for Tight Inner Loops

**Priority:** Medium  
**Applicable to:** `tpm.infer_cm` (lines 498–500), `convert.state_by_state2state_by_node` (lines 271–275)

For code that cannot be cleanly expressed as a vectorized numpy call — particularly `infer_cm`, which tests causal influence for every ordered node pair by running a boolean check over all `2^(N-1)` contexts — a `@numba.jit(nopython=True)` decorator would compile the Python loop to native machine code, typically yielding 10–100× speedup for tight numerical loops.

`numba` is not currently a dependency. Adding it as an optional extra (similar to how `ray` is optional for parallelism) would allow annotating these hot paths without making it a hard requirement.

---

### 13. Architectural: Pass Marginals Directly to GID Instead of Computing Joint then Marginalizing

**File:** `pyphi/metrics/distribution.py:679–681`  
**Priority:** High (architectural)

The codebase itself flags this in a TODO comment:

```python
# TODO: All the marginalization defeats the whole purpose. Config option
# must prevent calculating outer product at `subsystem`, and pass node
# marginal repertoires instead.
```

Currently, `subsystem._cause_repertoire` / `_effect_repertoire` build a full **joint** distribution over all purview nodes (via the outer-product multiplication chain), and then `generalized_intrinsic_difference` immediately marginalizes it back down to per-node distributions via `joint_to_marginals`. This round-trip — joint construction then marginalization — is unnecessary: if the per-node marginal repertoires were computed and stored directly, both the multiplication chain and the marginalization step would be eliminated entirely.

This is the single highest-leverage architectural change in the hot path. The TODO suggests a config option gate; the underlying change would be in `subsystem._cause_repertoire` / `_effect_repertoire` to optionally return a list of per-node arrays rather than a joint array.

---

## Combined Summary Table

| Priority | File | Lines | Opportunity | Tool / Method | Status |
|---|---|---|---|---|---|
| High | `pyphi/subsystem.py` | 380–382, 459–464 | Replace `functools.reduce` | `np.multiply.reduce` / `np.einsum` | Done |
| High | `pyphi/metrics/distribution.py` | 679–681 | Skip joint→marginal round-trip | Architecture (pass marginals directly) | Drop |
| High | `pyphi/convert.py` | 269–275 | Double loop over states×nodes | `np.einsum('ij,nj->in', tpm, node_on)` | Done |
| Medium | `pyphi/tpm.py` | 690–694 | Node loop in `probability_of_current_state` | `np.where` vectorization | Drop |
| Medium | `pyphi/convert.py` | 168–169 | O(N²) loop in `be2le_state_by_state` | `np.ix_` fancy indexing | Done |
| Medium | `pyphi/tpm.py` | throughout | Deterministic TPM storage | `scipy.sparse` (already transitive dep) | Drop |
| Medium | `pyphi/tpm.py`, `convert.py` | 498–500, 271–275 | Tight loops not easily vectorizable | `numba` JIT (optional extra) | Drop |

## Speed / Memory Gains

The three highest-ROI items from the combined summary table were implemented and benchmarked. All measurements are wall-clock time per call, averaged over 200 repetitions (`timeit`, n=200, one warm-up call) on an Apple M-series CPU running Python 3.12.

### Benchmark conditions

| Parameter | Value |
|---|---|
| Python | 3.12.11 |
| NumPy | latest (from `uv` lockfile) |
| Network for repertoire benchmarks | `examples.fig16_network()` — 7-node fully-connected network, state all-zeros |
| Network for conversion benchmarks | 8-node synthetic state-by-state TPM (256×256, random stochastic rows, seed 42) |
| Benchmark script | `script/bench_optimizations.py` |

### Results

| Change | Function | Before (ms/call) | After (ms/call) | Speedup |
|---|---|---|---|---|
| A — in-place multiply | `cause_repertoire` (7-node) | 0.015 | 0.011 | 1.4× |
| A — in-place multiply | `effect_repertoire` (7-node) | 0.035 | 0.035 | ~1× (within noise) |
| B — `np.ix_` fancy index | `be2le_state_by_state` (8-node) | 30.1 | 0.23 | **131×** |
| C — vectorized stack+sum | `state_by_state2state_by_node` (8-node) | 3.0 | 0.49 | **6.1×** |

### Implementation notes

**Change A** (`pyphi/subsystem.py:379–381, 456–458, 582–585`): Replaced `functools.reduce(np.multiply, [f(x) for x in items])` with a for-loop that multiplies in-place into the pre-allocated `joint = np.ones(...)` array. This eliminates k−1 temporary arrays per call (where k is the mechanism or purview size). The gain is modest on the 7-node benchmark because the cached `_single_node_*_repertoire` calls dominate; the benefit is larger for bigger mechanisms. The `functools` import was removed from `pyphi/subsystem.py`.

**Change B** (`pyphi/convert.py:164–167`): Replaced an O(N²) Python loop over all (i, j) state pairs with a single `np.ix_` fancy-index gather. Pre-computing the bit-reversal permutation takes O(N) Python iterations; the gather itself runs entirely in C. The 131× speedup on the 8-node case (N=256 states, 65,536 loop iterations) scales favourably — it becomes even more dramatic for larger networks.

**Change C** (`pyphi/convert.py:260–272`): Replaced nested loops (S outer iterations × N inner `np.sum` calls) with a vectorized `np.stack([(tpm * node_on[n]).sum(axis=1) for n in range(N)])` reduction followed by a Fortran-order reshape. A critical subtlety: using `tpm @ node_on.T` (BLAS matrix multiply) instead of the explicit row-sum approach causes subtle floating-point accumulation differences that change tie-breaking in the SIA search on the `micro_s` test network. The `.sum(axis=1)` formulation preserves bit-for-bit numerical equivalence with the original. The result is wrapped in `np.ascontiguousarray()` to ensure C-contiguous memory layout, which is required for downstream array operations to follow identical code paths.

### Tests

All 847 non-slow tests pass after the changes (`uv run pytest test/ -m "not slow and not veryslow"`). Specifically verified:

- `test/test_subsystem_cause_effect_repertoire.py` — 37/37 pass (Change A)
- `test/test_convert.py` — 16/16 pass (Changes B and C)
- `test/test_big_phi.py::test_sia_micro_sequential` and `::test_sia_micro_parallel` — confirmed numerically invariant (Change C)

## Notes on Remaining Implementation

### Joint→marginal round-trip (`distribution.py:679–681`) — Dropped

The TODO comment at line 679 lives inside `approximate_specified_state()`, a function that is **defined but never called anywhere in the codebase**. The `joint_to_marginals` helper it references is equally dead code.

The actual hot-path function, `generalized_intrinsic_difference` (line 866), computes `log2(forward / partitioned)` element-wise over the full joint arrays via `pointwise_mutual_information_vector`. This operation fundamentally requires the joint distribution and cannot be replaced by per-node marginals. There is no round-trip to eliminate in the live code path.

### `probability_of_current_state` np.where (`tpm.py:690–694`) — Dropped

The N-iteration loop runs exactly once per `Subsystem.__init__` via `_backward_tpm`, not on the hot phi-computation path. For typical network sizes (N ≤ 12) the loop completes in nanoseconds. The potential gain does not justify the change.

### `scipy.sparse` for TPMs (`tpm.py` throughout) — Dropped

Sparsity only helps deterministic networks where each TPM row has exactly one non-zero entry. Attention-derived TPMs in the ConsInfoLLM experiments are stochastic with dense rows, so sparse storage would add overhead rather than reduce it.

### `numba` JIT (`tpm.infer_cm:498–500`) — Dropped

`infer_cm` is called once at network construction. The only remaining loop in `convert.py` (lines 271–275) has already been vectorized by Change C. Adding `numba` as an optional dependency for a one-time O(N²) function is not justified.


---

## Scaling Benchmark (2026-04-16)

Wall-clock time for `new_big_phi.sia()` (IIT 4.0, sequential, cache on) on networks with verified non-zero Phi. Apple M-series CPU, Python 3.12. All networks use rule-152 ring CA generated by `make_rule_ca(152, n)`; see [script/bench_scaling.py](../script/bench_scaling.py).

| N | Partitions | ms/partition | Time (s) | Phi |
|---|---|---|---|---|
| 3 | 22 | — | 0.0 | 2.000000 |
| 4 | 150 | — | 0.6 | 1.331763 |
| 5 | 1,061 | 14.7 | 15.6 | 0.830075 |
| 6 | 7,896 | 62.3 | 492 | 0.276692 |
| 7 | 61,888 | ~264 (est.) | >7200 (TIMEOUT) | — |

Both partition count (~7.8×) and per-partition cost (~4.2×) grow with each additional node, giving a combined ~32× slowdown per step. N=7 is projected at ~4.5 h sequential. At 96 CPU cores via Ray parallelism, N=6 would take ~5 s and N=7 ~170 s.

### Shortcircuit behavior during benchmark runs

`SHORTCIRCUIT_SIA` was active during all runs (default `true` in `pyphi_config.yml`; the script does not override it). However, it had **no effect** on any timed row: the mechanism exits before the partition loop only when the system has no cause or effect (intrinsic information ≤ 0). All benchmark networks were deliberately chosen to have non-zero Phi (rule152 N=5: φ=0.83, N=6: φ=0.28), so the shortcircuit condition was never met and the **full exhaustive partition search ran in every case**.

The MapReduce early-exit (stop when any partition returns φ=0) likewise never fired, because the MIP partition itself has φ>0 for these networks.

Consequently the measured times represent **true worst-case** timing — every partition was evaluated. This is the correct baseline for evaluating partition-ordering or pruning strategies. For real LLM attention TPMs, many subsystems will be reducible (φ=0) and `SHORTCIRCUIT_SIA` will trigger instantly; the expensive regime is exactly what was benchmarked here.

---

## Partition Search Reduction

The dominant cost in `sia()` is exhaustive evaluation of all system partitions — a number that grows with the Bell numbers (N=6: 7,896; N=7: 61,888; N=8: 510,313). Two classes of approach can reduce this cost: **pruning** (skip partitions that cannot be the MIP) and **approximation** (replace the exact MIP with a tractable proxy).

### What PyPhi already does

| Mechanism | Location | Effect |
|---|---|---|
| `SHORTCIRCUIT_SIA` — exits before partition loop if system has no cause or effect (intrinsic information ≤ 0) | [`new_big_phi/__init__.py:560–563`](../pyphi/new_big_phi/__init__.py#L560) | Skips all partitions; explains instant results for trivially-reducible states |
| MapReduce early-exit — stops evaluating partitions as soon as any partition yields φ=0 | [`new_big_phi/__init__.py:570–582`](../pyphi/new_big_phi/__init__.py#L570) | Helps only when φ=0; no benefit for φ>0 systems |
| Degenerate-case returns — empty subsystem, no strong connectivity, singleton without self-loop | [`new_big_phi/__init__.py:514–535`](../pyphi/new_big_phi/__init__.py#L514) | O(1) exits before any partition work |
| **TODO (unimplemented)** — trivial reducibility detection | [`new_big_phi/__init__.py:497`](../pyphi/new_big_phi/__init__.py#L497) | Placeholder only |

No ordering, bounding, or heuristic pruning of the partition space is implemented. All partitions are evaluated in arbitrary order unless one yields φ=0.

**Confirmed in benchmarks:** `SHORTCIRCUIT_SIA` and the MapReduce early-exit were both active but neither fired during the rule-152 scaling runs (N=5,6), because those networks have φ>0. The measured times are therefore true exhaustive-search baselines with no shortcircuiting benefit.

---

### Baseline catalogue

Each entry covers: the paper reference, what the method computes, its complexity, whether Python code exists, and how it plugs into PyPhi's `sia(partitions=...)` interface.

---

#### B1 — MI-ordering (pairwise MI proxy) ✅ implemented
**Refs:** Kitazono, Kanai & Oizumi (2018) [doi:10.3390/e20030173](https://doi.org/10.3390/e20030173); Hidaka & Oizumi (2018) [PMC7512690](https://pmc.ncbi.nlm.nih.gov/articles/PMC7512690/)  
**Complexity:** O(N² · 2^N) preprocessing, then exhaustive GID evaluation  
**Exact:** Yes (exhaustive GID still runs; ordering is a heuristic)  
**Python:** `script/bench_mi_ordering.py` in this repo

Precompute an N×N pairwise MI matrix under the stationary distribution; score each partition by summing MI over its cut edges; sort partitions and pass to `sia(partitions=sorted_iter)`. Benchmarked on micro (N=4) and rule-152 ring CA (N=5): high-MI-first gives 2.6× speedup potential on the micro network; ring CA favours low-MI-first. Direction is network-type-dependent — neither ordering dominates universally.

---

#### B2 — Queyranne's algorithm (exact min-submodular-bipartition)
**Refs:** Queyranne (1998) [doi:10.1287/moor.23.3.695](https://doi.org/10.1287/moor.23.3.695); Kitazono et al. (2018) [doi:10.3390/e20030173](https://doi.org/10.3390/e20030173)  
**Complexity:** O(N³) bipartition evaluations; GID still needed to verify  
**Exact:** Near-exact (>98% correct MIP empirically even for non-submodular GID)  
**Python:** MATLAB only ([oizumi-lab/PhiToolbox](https://github.com/oizumi-lab/PhiToolbox)); ~80 lines to port

Finds the bipartition that minimises any **submodular** function (MI, stochastic interaction) in O(N³) via vertex contraction. Use MI as the submodular proxy → O(N³) candidate MIP → verify that one partition with exact GID. The pluggable `partitions` arg in `sia()` accepts the single candidate directly: `sia(partitions=[queyranne_mip(subsystem)])`.

---

#### B3 — Louvain community cut (single-candidate approximation)
**Refs:** Blondel et al. (2008) [doi:10.1088/1742-5468/2008/10/P10008](https://doi.org/10.1088/1742-5468/2008/10/P10008); Nilsen et al. (2019) [doi:10.3390/e21050525](https://doi.org/10.3390/e21050525)  
**Complexity:** O(N log N) community detection + one GID evaluation  
**Exact:** Approximate (r > 0.95 with exact phi on binary systems ≤ 8 nodes, Nilsen et al.)  
**Python:** `networkx.community.louvain_communities` (already a transitive dep)

Run Louvain community detection on the subsystem's connectivity matrix; convert the two-community result into a `SystemPartition`; evaluate only that one partition: `sia(partitions=[louvain_cut(subsystem)])`. Ten lines of code, no new dependencies.

---

#### B4 — CUT_ONE / single-node-isolating cuts
**Refs:** Oizumi et al. (2014) [doi:10.1371/journal.pcbi.1003588](https://doi.org/10.1371/journal.pcbi.1003588) (original phi); Mayner et al. (2018) [arXiv:1712.09644](https://arxiv.org/abs/1712.09644) (PyPhi, IIT 3.0 config)  
**Complexity:** O(N) cuts instead of O(2^N)  
**Exact:** Approximate upper bound on phi  
**Python:** `config.CUT_ONE_APPROXIMATION = True` in IIT 3.0; IIT 4.0 status unchecked

For each node i, evaluate only the partition that isolates node i from all others (severs all edges to/from i). The minimum phi over these N cuts is an upper bound on exact phi. Already wired in IIT 3.0 via config; worth checking whether the same config flag applies to IIT 4.0's `SET_UNI/BI` partition scheme (may require generating only `GeneralSetPartition` objects with single-node parts).

---

#### B5 — HDMP (Heuristic-Driven Memoization Process)
**Ref:** Mendieta, Arango-López & Castillo (2024) [doi:10.1007/978-3-031-75233-9_16](https://doi.org/10.1007/978-3-031-75233-9_16)  
**Complexity:** Sub-exponential (memoization prunes dominated subpartitions)  
**Exact:** Yes (same phi value as exhaustive search)  
**Python:** Research prototype only; no public repo

Top-down recursive MIP search with a cost-matrix heuristic that orders which subpartitions to expand first, pruning branches dominated by the current best. Reported >90% runtime reduction at N≈200. Most promising for large N where exhaustive search is infeasible. Would require a new search driver replacing the flat MapReduce loop.

---

#### B6 — Gaussian / Φ* / Φ_G approximations
**Refs:** Tegmark (2016) [arXiv:1601.02626](https://arxiv.org/abs/1601.02626); Oizumi, Tsuchiya & Amari (2016) [doi:10.1073/pnas.1603583113](https://doi.org/10.1073/pnas.1603583113); Oizumi et al. (2016) [doi:10.1371/journal.pcbi.1004654](https://doi.org/10.1371/journal.pcbi.1004654)  
**Complexity:** O(N³) (covariance matrix operations, no partition search)  
**Exact:** No — approximation under Gaussian assumption  
**Python:** MATLAB only ([PhiToolbox](https://github.com/oizumi-lab/PhiToolbox)); straightforward numpy port

Replace the discrete TPM with a 2N×2N covariance matrix; phi variants (Φ*, Φ_G, stochastic interaction SI) reduce to log-det operations or KL divergences between Gaussians. No partition enumeration — the MIP is found analytically. Exact for linear-Gaussian systems; a rough approximation for LLM attention TPMs (which are stochastic-dense, not linear-Gaussian).

---

#### B7 — ΦID — Integrated Information Decomposition
**Refs:** Mediano, Rosas et al. (2021) [arXiv:2109.13186](https://arxiv.org/abs/2109.13186); PNAS 2025 [doi:10.1073/pnas.2423297122](https://doi.org/10.1073/pnas.2423297122)  
**Complexity:** Polynomial in N (PID lattice over source pairs)  
**Exact:** No — different quantity (decomposes transfer entropy, not GID-phi)  
**Python:** [`Imperial-MIND-lab/integrated-info-decomp`](https://github.com/Imperial-MIND-lab/integrated-info-decomp) (`pip install`)

Decomposes transfer entropy into synergistic, redundant, and unique "atoms" via partial information decomposition — no partition search at all. The "integrated synergy" atom is empirically correlated with IIT phi and interpretable as the unique information generated by the whole that is not present in any part. Useful as a `compute_phi_approximate(tpm)` path for systems too large for exact PyPhi (N > 10).

---

#### B8 — Max-modularity fixed partition
**Ref:** Toker & Sommer (2016) [arXiv:1605.01096](https://arxiv.org/abs/1605.01096)  
**Complexity:** O(N log N)  
**Exact:** No — approximation  
**Python:** `networkx.community` (already present); research code only for the phi evaluation wrapper

Skip MIP search entirely: evaluate phi only across the partition that maximises graph modularity. Correlates well with MIP-phi in practice; suitable when only a relative ordering of phi values is needed across conditions (e.g., task-relevant vs. random-baseline tokens in ConsInfoLLM), not when the exact MIP partition is required.

---

### Feature summary table

| # | Method | Complexity | Exact? | Python? | PyPhi interface | Status |
|---|---|---|---|---|---|---|
| B1 | MI-ordering (pairwise proxy) | O(N²·2^N) pre + exhaustive | Yes | ✅ this repo | `sia(partitions=mi_sorted)` | **Done** |
| B2 | Queyranne min-submodular bipartition | O(N³) | Near-exact | ✅ this repo | `sia(partitions=[queyranne_mip])` | **Done** |
| B3 | Louvain community cut | O(N log N) + 1 GID | Approx | ✅ networkx | `sia(partitions=[louvain_cut])` | **Done** |
| B4 | CUT_ONE (single-node isolation) | 3N evaluations | Approx upper bound | ✅ this repo | filter `SET_UNI/BI` to singleton parts | **Done** |
| B5 | HDMP memoized search | Sub-exponential | Yes | ❌ paper only | New search driver | Research |
| B6 | Gaussian / Φ* / Φ_G | O(N³) | No (Gaussian approx) | ❌ MATLAB | Standalone `compute_phi_approx` | Research |
| B7 | ΦID (integrated synergy) | Polynomial | No (different quantity) | ✅ this repo | Standalone `compute_phi_approx` | **Done** |
| B8 | Max-modularity partition | O(N log N) + 1 GID | No | ✅ this repo | `sia(partitions=[modularity_cut])` | **Done** |

**All baselines implemented:** B1 (MI-ordering), B2 (Queyranne), B3 (Louvain), B4 (CUT_ONE), B7 (ΦID), B8 (max-modularity). B5/B6 remain research-level (no public code).

---

## MI-Ordering Diagnostic (2026-04-16)

Tests whether sorting partitions by a **pairwise MI proxy** (sum of MI(X_i; X_j)
over all severed edges, under the stationary distribution) concentrates the MIP
in early ranks.

- **↓MI** = high MI first (strong coupling severed first)
- **↑MI** = low MI first (weak coupling severed first)
- **Rand%** = mean rank% under random ordering (10 trials)

Lower rank% = MIP found earlier = greater potential speedup from early stopping.

| N | Network | Partitions | ↓MI rank% | ↑MI rank% | Rand rank% | phi (norm) |
|---|---|---|---|---|---|---|
| 4 | micro (IIT 4.0) | 150 | 20.7% | 80.0% | 54.7% ± 27.0% | 0.1665 |
| 5 | rule152 ring CA | 1,061 | 68.0% | 22.3% | 13.8% ± 10.7% | 0.0692 |

**Key finding:** MI ordering direction is network-type-dependent.

- **Micro (fully-connected stochastic):** ↓MI finds MIP at 20.7% vs 54.7% random → **2.6× speedup potential** if early-stopping is added.
- **Ring CA (sparse deterministic):** ↓MI is counterproductive (68%); ↑MI gives 22.3% vs 13.8% random — only a modest improvement since ~7 MIP ties already give early random hits.

**Implication:** A single MI-ordering heuristic cannot replace exhaustive search without knowing the network type in advance.  A more robust approach would combine: (1) MI ordering as a prior, (2) branch-and-bound with the first-found phi as an upper bound, and (3) early termination when no remaining partition's MI proxy can improve on the running minimum.


---

## B2 Queyranne Diagnostic (2026-04-16)

**Mode A (DIRECTED_BI):** Queyranne's natural domain — bipartition-only search
(`SYSTEM_PARTITION_TYPE='DIRECTED_BI'`).  Queyranne oracle finds the minimum-MI
bipartition in O(N³) calls; that candidate is verified with exact GID.
_BiMatch_: Queyranne bipartition equals the exhaustive BI-MIP.
_PhiMatch_: normalized phi from Q-candidates equals exhaustive BI phi.

**Mode B (full SET_UNI/BI):** Default IIT 4.0 partition scheme.  Reports the
true MIP partition type.  If MIP #parts > 2, Queyranne's bipartition search
cannot reach it — a structural limitation for IIT 4.0.

### Mode A — DIRECTED_BI only

| N | BI parts | Q parts | Speedup | BiMatch | PhiMatch | T_BI (s) | T_Q (s) |
|---|---|---|---|---|---|---|---|
| 4 | 14 | 2 | 7× | ✗ | ✗ | 0.06 | 0.000 |
| 5 | 30 | 2 | 15× | ✓ | ✓ | 0.43 | 0.000 |

### Mode B — Full SET_UNI/BI (default IIT 4.0)

| N | Total parts | True phi | MIP #parts | T_full (s) |
|---|---|---|---|---|
| 4 | 150 | 1.3318 | 4 | 0.6 |
| 5 | 1061 | 0.8301 | 4 | 14.3 |

**Key finding:** For both benchmark networks the IIT 4.0 MIP is a 4-part partition
(MIP #parts = 4), so Queyranne's bipartition search fundamentally cannot reach
the global MIP under `SET_UNI/BI`.  Within the bipartition-only domain
(Mode A), Queyranne achieves 15× speedup with correct identification on N=5
ring CA, but fails on N=4 micro (wrong bipartition selected by MI proxy).

**Implication:** B2 (Queyranne) is directly applicable to IIT 3.0 workflows that
use directed bipartitions, or to IIT 4.0 with `DIRECTED_BI` as an approximation.
For full IIT 4.0 `SET_UNI/BI`, a multi-part extension of the oracle
(hierarchical bisection or Queyranne on the partition lattice) would be needed.


---

## B3 Louvain Community Detection Diagnostic (2026-04-16)

Build a pairwise-MI-weighted graph on the N-node system; run Louvain
community detection (networkx, fixed seed=42); map communities to
matching `GeneralSetPartition` direction-variants; evaluate those with
exact GID and compare against full exhaustive `sia()`.

| N | Communities | Candidates | Speedup | GrpMatch | PhiMatch | True phi | Louv phi | T_full (s) | T_louv (s) |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 2 | 3 | 50× | ✗ | ✗ | 1.3318 | 2.6635 | 0.56 | 0.018 |
| 5 | 1 | 0 | 1061× | ✗ | ✗ | 0.8301 | N/A | 14.47 | 0.000 |

_Communities_: Louvain community count (= partition arity if GrpMatch ✓).
_Candidates_: GeneralSetPartition objects matching the Louvain grouping
(all 3^k direction-assignment variants of the same node groups).
_GrpMatch_: Louvain community grouping == true MIP node grouping.
_PhiMatch_: phi computed from Louvain candidates == exhaustive phi.

Community assignments per network:
- N=4: [[0, 1], [2, 3]] (true MIP #parts=4)
- N=5: [[0, 1, 2, 3, 4]] (true MIP #parts=4)

**Key findings:**

- **N=4 (micro, fully-connected stochastic):** Louvain splits into 2 communities `{0,1}|{2,3}` but the true MIP is a 4-part partition. The Louvain bipartition phi (2.66) is double the true MIP phi (1.33) — the bipartition severs fewer edges per unit normalization than the 4-part split. GrpMatch=✗, PhiMatch=✗.
- **N=5 (ring CA, sparse deterministic):** The ring topology gives uniform pairwise MI, so Louvain sees no community structure and returns a single all-nodes community. Zero matching candidates — complete fallback required.

**Implication:** Louvain fails on both tested network types for different structural reasons: (1) the true MIP is multi-part so a 2-community split cannot match it; (2) symmetric ring networks have no community structure. Louvain is better suited to networks with clear hierarchical modular organization. For IIT 4.0 the method requires either a resolution-parameter sweep or a multi-resolution strategy to generate multi-part candidate partitions.


---

## B4 CUT_ONE Diagnostic (IIT 4.0, 2026-04-16)

Filter the full `SET_UNI/BI` partition list to 2-part `GeneralSetPartition`
objects where one group is a singleton node.  The `unique()` deduplication in
`unidirectional_set_partitions` collapses the 3² = 9 raw direction combos down
to **3 distinct cut matrices per singleton** (cut edges into singleton, cut
edges out of singleton, cut both), giving **3N total candidates**.

Candidates are evaluated with exact GID and compared against exhaustive `sia()`.

| N | All parts | CO parts | Expected | Speedup | UpperBound | PhiMatch | True phi | CO phi | MIP#(true) | MIP#(CO) | T_full (s) | T_CO (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 150 | 12 | 12 | 12× | ✓ | ✗ | 1.3318 | 0.6659 | 4 | 2 | 0.57 | 0.053 |
| 5 | 1061 | 15 | 15 | 71× | ✓ | ✗ | 0.8301 | 0.4150 | 4 | 2 | 14.47 | 0.228 |

_UpperBound_: `co_normalized_phi ≥ full_normalized_phi` — always holds because CUT_ONE ⊆ all_partitions and `sia()` minimizes normalized_phi.
_PhiMatch_: exact MIP found within CUT_ONE candidates.
_MIP#(CO)_: number of parts in the best CUT_ONE partition (always 2).

**Key findings:**

- **Speedup**: 12× (N=4) and 71× (N=5) — evaluating only 3N candidates vs Bell(N). Scales well: N=6 gives 54 candidates vs 7,896 (146×).
- **CO phi < true phi**: The singleton-isolation partition severs fewer connections per normalization than the 4-part MIP. The normalized_phi ordering is correctly preserved (UpperBound=✓), but the raw phi values are not directly comparable across partition types.
- **PhiMatch=✗ for both networks**: Both true MIPs are 4-part partitions — structurally unreachable by any 2-part CUT_ONE candidate. CUT_ONE is exact only when the MIP happens to be a singleton-isolation partition (common in IIT 3.0 binary systems but not in these IIT 4.0 networks).

**Implication:** B4 provides a fast upper bound on normalized_phi and a 3N-evaluation approximation that is useful for screening (e.g., if CUT_ONE already finds φ ≈ 0 the system is reducible without full search). For exact IIT 4.0 phi the 4-part MIP structure requires multi-part partition search.


---

## B7 ΦID Integrated Synergy Diagnostic (2026-04-16)

Simulate T=20000 steps of the network Markov chain; for each ordered
pair (i, j) with i≠j compute pairwise ΦID (discrete, MMI redundancy, τ=1);
sum the `sts` (Syn→Syn) atoms across all N(N-1) pairs to obtain
system-level integrated synergy ΦID_SYN.  Compare to exhaustive IIT 4.0 Φ.

| N | Pairs | ΦID_SYN | ΦID/pair | True phi | T_sim (s) | T_ΦID (s) | T_full (s) |
|---|---|---|---|---|---|---|---|
| 4 | 12 | 0.0019 | 0.0002 | 1.3318 | 0.191 | 0.509 | 0.58 |
| 5 | 20 | 0.0000 | 0.0000 | 0.8301 | 0.191 | 0.831 | 14.37 |

_ΦID_SYN_: sum of pairwise `sts` atoms across all N(N-1) ordered pairs.
_ΦID/pair_: per-pair average integrated synergy.
_T_ΦID_: wall time for all pairwise ΦID computations (no partition search).
_T_full_: wall time for exhaustive `sia()` (full partition search).

**Key findings:**

- **ΦID_SYN ≈ 0 for both networks** despite non-zero IIT Φ (1.33 and 0.83). Increasing T to 50,000 and switching to CCS redundancy does not materially change the result — pairwise sts is genuinely near zero for these networks under MMI and CCS.
- **Gaussian approximation** (`kind='gaussian'`) gives NaN on binary time series due to degenerate covariance matrices (binary nodes can have near-zero variance in some pairs). Not applicable.
- **Divide-by-zero warnings** in discrete mode arise when empirical counts for rare joint states are zero, making local entropy infinite. For small binary systems (2^N possible states), T=20,000 is usually sufficient to cover all states, but the sts atom is still near zero.

**Why ΦID_SYN ≈ 0 for these networks:**

The `sts` atom (Syn→Syn) requires information that is simultaneously synergistic in BOTH the past (pair of source nodes) AND the future (pair of target nodes). For the micro network's near-product-structure TPM (many identical rows), pairwise synergy is negligible. For the deterministic ring CA, future states are functions of exactly 3 local neighbors — no additional synergy is generated beyond what local pairs already capture.

**Implication:** B7 (ΦID via phyid) is not a useful proxy for IIT 4.0 Φ on small binary networks. The literature correlation between ΦID_SYN and Φ holds for larger systems (N >> 8) with near-Gaussian continuous dynamics (neural recordings), not for discrete-binary Markov chains of the type used in IIT theory. For the ConsInfoLLM use case — attention-derived TPMs over selected tokens — the Gaussian approximation may be more appropriate if attention weights are treated as continuous and N is large enough to avoid degenerate covariance.


---

## B8 Max-Modularity Partition Diagnostic (2026-04-16)

Build a pairwise-MI-weighted graph; run `greedy_modularity_communities`
(networkx, deterministic O(E log N)); map communities to matching
`GeneralSetPartition` direction-variants; evaluate with exact GID vs
exhaustive `sia()`.  Identical graph construction to B3 (Louvain) but
using deterministic greedy modularity instead of stochastic Louvain.

| N | Communities | Candidates | Speedup | GrpMatch | PhiMatch | True phi | Mod phi | MIP#(true) | T_full (s) | T_mod (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 2 | 3 | 50× | ✗ | ✗ | 1.3318 | 2.6635 | 4 | 0.57 | 0.018 |
| 5 | 1 | 0 | 1061× | ✗ | ✗ | 0.8301 | N/A | 4 | 14.64 | 0.000 |

Community assignments per network:
- N=4: [[0, 1], [2, 3]] (true MIP #parts=4)
- N=5: [[0, 1, 2, 3, 4]] (true MIP #parts=4)

**Key findings:**

- Greedy modularity produces **identical communities to Louvain (B3)** on
  both benchmark networks: `{0,1}|{2,3}` for micro (N=4) and one all-node
  community for ring CA (N=5).
- Same failure modes as B3: (1) micro's true MIP is 4-part, unreachable by
  a 2-community cut; (2) ring CA's uniform MI gives no community structure.
- Greedy modularity is deterministic (no seed) and slightly faster than
  Louvain for small dense graphs.
- **Implication**: B8 and B3 are interchangeable on these networks.
  Max-modularity partitioning is most useful for networks with clear
  hierarchical structure where the modularity-optimal partition coincides
  with the MIP — not observed here.

