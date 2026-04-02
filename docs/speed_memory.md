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
