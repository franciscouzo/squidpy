**PR report: correct the permutation null for spatial autocorrelation**

Branch: `fix/spatial-autocorr-permutation-null`. Implementation commit: `602f5b7`.

`spatial_autocorr` previously shuffled only the rows of its spatial weights matrix when generating null scores. This changes which cells are connected and can introduce self-connections. It does not implement random assignment of expression values to fixed spatial locations. The fix shuffles the observation columns of the expression matrix while keeping the spatial graph fixed, for both Moran’s I and Geary’s C.

This follows the observation-permutation scheme in PySAL’s reference implementations of [Moran’s I](https://raw.githubusercontent.com/pysal/esda/main/esda/moran.py) and [Geary’s C](https://raw.githubusercontent.com/pysal/esda/main/esda/geary.py).

```diff
- score_perms[i, :] = func(g[idx_shuffle, :], vals)
+ score_perms[i, :] = func(g, vals[:, idx_shuffle])
```

Here, `vals` has shape `(n_features, n_observations)`. Each permutation applies the same column order to every feature. The existing independent random generator for each permutation is retained.

**1. What changes in a single permutation**

The original observations lie on the path A–B–C–D. The graph has row-normalized weights, and expression is `[0, 1, 2, 3]`. For the example order `[2, 0, 3, 1]`, the previous algorithm creates self-connections at B and C. The corrected algorithm keeps the original path and assigns expression `[2, 0, 3, 1]` to its locations.

![A single permutation: original data, previous graph-row shuffle, and corrected observation shuffle](01-permutation-mechanism.png)

*The lower panels show the actual weights matrices. Rows and columns retain the same cell order in every panel. The diagram is a single example; the following comparison includes every possible labeling.*

**2. Exact comparison on the regression-test graph**

We enumerate all `4! = 24` label permutations for the same path and expression feature. This is an exact finite distribution, with no Monte Carlo sampling error. Each permutation is equally likely. The corrected scores are checked against independent implementations of the statistical definitions.

For a graph with zero diagonal, the label-permutation expectations are `E[I] = −1/(n−1)` and `E[C] = 1`. With four observations, these are `−1/3` and `1`. The corrected distributions recover both expectations; the previous distributions do not.

![Exact before-and-after probability distributions for Moran's I and Geary's C](02-exact-null-distributions.png)

| Statistic | Expected mean | Before: mean | After: mean | Before: variance | After: variance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Moran’s I | −0.333333 | 0.000000 | −0.333333 | 0.066667 | 0.215556 |
| Geary’s C | 1.000000 | 0.600000 | 1.000000 | 0.037500 | 0.143750 |

*Variances use the full enumerated distribution, with `ddof=0`. Bars for the two algorithms are slightly offset to make overlapping masses visible. [All 24 score pairs](exact-null-scores.csv) are included.*

**3. A larger graph also shows a variance error**

The second example uses 256 cells on a 16 × 16 grid, with connections to horizontal and vertical neighbors, no self-connections, and row-normalized weights. One synthetic standard-normal expression feature is held fixed. Both algorithms receive the same 4,096 random label permutations. Expression and permutation generators are spawned independently from seed `42`.

![Before-and-after null distributions on a 256-cell spatial grid](03-grid-null-distributions.png)

| Statistic | Before: variance | After: variance | Before / after |
| --- | ---: | ---: | ---: |
| Moran’s I | 0.001134 | 0.001987 | 57.04% |
| Geary’s C | 0.001125 | 0.001989 | 56.55% |

In this example, the old null distribution has about 43% less variance. The corrected mean is approximately `−0.004160` for Moran’s I, compared with the theoretical `−1/255 ≈ −0.003922`, and `1.000279` for Geary’s C, compared with `1`. Unlike the four-cell enumeration, these histograms are Monte Carlo estimates. Their precise values are specific to this graph, expression feature, and sample of permutations.

The figures demonstrate the distribution error and its correction; they are not a general calibration study of every reported p-value. [All 4,096 score pairs](grid-null-scores.csv) and [unrounded summary statistics and dependency versions](summary.json) are included.

**Effect on users**

Correcting the permutation scores changes `var_sim`, `pval_sim`, `pval_z_sim`, and corresponding adjusted p-values when correction is enabled. Results from a fixed seed can therefore differ from earlier versions. The original observed I/C statistics and analytic normality calculations are unaffected by this change. The function documentation records the correction for version 1.8.5.

**Validation**

- The new regression test enumerates every labeling of the four-cell path for both statistics, checking every score against its independent mathematical definition and checking the exact null mean. It covers dense, CSR, and CSC expression matrices and verifies that the graph is unchanged.
- All six new regression cases failed against the previous implementation and pass with the fix.
- The relevant test files passed all 44 tests, including serial/parallel reproducibility and seed handling. Ruff lint, formatting, and the whitespace check passed for the implementation and tests.
- The figure-generation script checks every corrected score, including all 4,096 grid permutations for both statistics, against an independent calculation over the original graph’s edges.

The recorded test command was:

```bash
NUMBA_NUM_THREADS=2 python -m pytest \
  tests/graph/test_ppatterns.py tests/graph/test_utils.py -q --tb=short
```

**Reproduce or reuse the figures**

From the repository root, with this checkout installed in the active Python environment:

```bash
NUMBA_NUM_THREADS=1 python reports/spatial-autocorr-permutation-null/generate_figures.py
```

The [generator](generate_figures.py) runs the corrected `_score_helper` from the installed checkout and evaluates the removed row-shuffling expression as the baseline. It writes PNG figures for the PR, editable SVG figures, score CSVs, and `summary.json`. The weights and synthetic data are constructed locally; no external datasets are downloaded.

SVG downloads: [permutation mechanism](01-permutation-mechanism.svg), [exact distributions](02-exact-null-distributions.svg), [grid distributions](03-grid-null-distributions.svg).
