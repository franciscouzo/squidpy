"""Reproduce the PR figures with exact enumeration and a seeded grid example.

Run from an environment with this checkout of Squidpy installed:
    NUMBA_NUM_THREADS=1 python reports/spatial-autocorr-permutation-null/generate_figures.py
"""

from __future__ import annotations

import json
from importlib.metadata import version
from itertools import permutations
from pathlib import Path
from unittest.mock import Mock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as PlotPath
from numba import set_num_threads
from scanpy.metrics import gearys_c, morans_i
from scipy.sparse import csr_matrix
from scipy.stats import ks_2samp, permutation_test
from sklearn.preprocessing import normalize

from squidpy.gr._ppatterns import _score_helper

OUT = Path(__file__).resolve().parent
BEFORE = "#C45C26"
AFTER = "#2166AC"
EXPECTED = "#183E2F"
KS_RESAMPLES = 9999
MODES = {"moran": (morans_i, "Moran’s I"), "geary": (gearys_c, "Geary’s C")}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    svg_path = OUT / f"{name}.svg"
    fig.savefig(svg_path, bbox_inches="tight", metadata={"Date": None})
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


def reference_scores(graph: csr_matrix, values: np.ndarray, orders: np.ndarray, mode: str) -> np.ndarray:
    """Evaluate the statistic's definition on every relabeling, independently of Scanpy."""
    permuted = values[0, orders]
    centered = permuted - permuted.mean(axis=1, keepdims=True)
    edges = graph.tocoo()
    denominator = edges.data.sum() * np.sum(centered**2, axis=1)
    n = graph.shape[0]
    if mode == "moran":
        numerator = np.sum(edges.data * centered[:, edges.row] * centered[:, edges.col], axis=1)
        return n * numerator / denominator
    differences = permuted[:, edges.row] - permuted[:, edges.col]
    numerator = np.sum(edges.data * differences**2, axis=1)
    return (n - 1) * numerator / (2 * denominator)


def compare(
    graph: csr_matrix, values: np.ndarray, orders: np.ndarray, *, reference_orders: np.ndarray | None = None
) -> dict:
    scores = {}
    for mode, (func, _) in MODES.items():
        # This is the removed expression; use the same orders as the corrected implementation.
        before = np.array([func(graph[order, :], values)[0] for order in orders])
        rng = Mock(spec=np.random.Generator)
        rng.permutation.side_effect = [order.copy() for order in orders]
        after = _score_helper(list(range(len(orders))), mode, graph, values, rngs=[rng] * len(orders))[:, 0]
        reference = reference_scores(graph, values, orders, mode)
        np.testing.assert_allclose(after, reference, atol=1e-12)
        expected = reference if reference_orders is None else reference_scores(graph, values, reference_orders, mode)
        scores[mode] = {"before": before, "after": after, "expected": expected}
    return scores


def ks_comparison(sample: np.ndarray, expected: np.ndarray, *, rng: np.random.Generator | None = None) -> dict:
    """Two-sided KS distance, with a pooled-label permutation p-value for independent samples.

    Rounding identifies mathematical ties despite harmless floating-point evaluation differences.
    A missing RNG denotes complete enumeration: report the population distance without a p-value.
    """
    sample = np.round(sample, 12)
    expected = np.round(expected, 12)
    support, ranks = np.unique(np.concatenate([sample, expected]), return_inverse=True)
    pooled_counts = np.bincount(ranks, minlength=len(support))
    sample_ranks, expected_ranks = ranks[: len(sample)], ranks[len(sample) :]

    # Once scores have been ranked, resampling needs only counts, not another sort at every draw.
    def statistic(x: np.ndarray, y: np.ndarray) -> float:
        counts = np.bincount(x, minlength=len(support))
        gap = counts.cumsum() / len(x) - (pooled_counts - counts).cumsum() / len(y)
        return float(np.max(np.abs(gap)))

    distance = statistic(sample_ranks, expected_ranks)
    np.testing.assert_allclose(distance, ks_2samp(sample, expected, method="asymp").statistic, atol=1e-12)
    result = {"statistic": distance, "pvalue": None, "calibration": "complete enumeration; descriptive distance"}
    if rng is not None:
        test = permutation_test(
            (sample_ranks, expected_ranks),
            statistic,
            permutation_type="independent",
            vectorized=False,
            alternative="greater",
            n_resamples=KS_RESAMPLES,
            batch=64,
            rng=rng,
        )
        result.update(
            pvalue=float(test.pvalue),
            calibration="pooled-label permutation test of the two-sided KS statistic",
            n_resamples=len(test.null_distribution),
        )
    return result


def summary(scores: dict, n: int, exact: bool, *, ks_rng: np.random.Generator | None = None) -> dict:
    result = {}
    ks_rngs = iter(ks_rng.spawn(2 * len(scores))) if ks_rng is not None else None
    for mode, distributions in scores.items():
        expected_mean = -1 / (n - 1) if mode == "moran" else 1.0
        if exact:
            np.testing.assert_allclose(distributions["after"].mean(), expected_mean, atol=1e-12)
        result[mode] = {
            "expected_mean": expected_mean,
            "before": {
                "mean": float(distributions["before"].mean()),
                "variance": float(distributions["before"].var()),
            },
            "after": {
                "mean": float(distributions["after"].mean()),
                "variance": float(distributions["after"].var()),
            },
            "expected": {
                "mean": float(distributions["expected"].mean()),
                "variance": float(distributions["expected"].var()),
            },
            "ks": {
                algorithm: ks_comparison(
                    distributions[algorithm],
                    distributions["expected"],
                    rng=next(ks_rngs) if ks_rngs is not None else None,
                )
                for algorithm in ["before", "after"]
            },
            "before_after_variance_ratio": float(distributions["before"].var() / distributions["after"].var()),
        }
    return result


def export_scores(scores: dict, filename: str) -> None:
    columns = {"permutation": np.arange(len(scores["moran"]["before"]))}
    for mode, distributions in scores.items():
        for algorithm, values in distributions.items():
            columns[f"{mode}_{algorithm}"] = values
    pd.DataFrame(columns).to_csv(OUT / filename, index=False, float_format="%.12g")


def plot_mechanism(weights: np.ndarray) -> None:
    order = np.array([2, 0, 3, 1])
    original_values = np.arange(4)
    panels = [
        ("Original observations", weights, original_values),
        ("Before: shuffle rows of W", weights[order, :], original_values),
        ("After: shuffle values of x", weights, original_values[order]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.1), layout="constrained")
    fig.suptitle("A label permutation must preserve the spatial graph", fontsize=19, fontweight="bold")
    positions = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (0, 0)}
    for col, (title, matrix, values) in enumerate(panels):
        ax = axes[0, col]
        graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        nx.draw_networkx_nodes(graph, positions, ax=ax, node_size=1800, node_color="#EEF3F8", edgecolors="#56728A")
        nx.draw_networkx_labels(
            graph,
            positions,
            ax=ax,
            labels={i: f"{chr(65 + i)}\nx = {value}" for i, value in enumerate(values)},
            font_size=11,
        )
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=ax,
            node_size=1800,
            arrowsize=15,
            edgelist=[(a, b) for a, b in graph.edges if a != b],
            connectionstyle="arc3,rad=0.13",
            edge_color=BEFORE if col == 1 else AFTER,
        )
        for node in nx.nodes_with_selfloops(graph):
            x, y = positions[node]
            path = PlotPath(
                [(x + 0.20, y + 0.20), (x + 0.75, y + 0.50), (x + 0.75, y - 0.50), (x + 0.20, y - 0.20)],
                [PlotPath.MOVETO, PlotPath.CURVE4, PlotPath.CURVE4, PlotPath.CURVE4],
            )
            ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=15, color=BEFORE, linewidth=1.5))
        ax.set(xlim=(-0.65, 1.8), ylim=(-0.6, 1.65), aspect="equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=13, fontweight="bold", color=BEFORE if col == 1 else "#18354A")
        ax = axes[1, col]
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:g}",
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] >= 0.8 else "#18354A",
                    fontsize=12,
                )
        ax.set_xticks(range(4), list("ABCD"))
        ax.set_yticks(range(4), list("ABCD"))
        ax.set_xlabel("Target cell")
        ax.set_ylabel("Source cell")
        ax.set_title("W" if col != 1 else "W[permutation, :] — two self-loops", fontsize=11)
    fig.text(
        0.5,
        -0.025,
        "Example permutation: [2, 0, 3, 1] (zero-based). Node labels are expression values; arrows show nonzero weights.",
        ha="center",
        fontsize=10,
        color="#425466",
    )
    save(fig, "01-permutation-mechanism")


def plot_distributions(scores: dict, metrics: dict, *, exact: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), layout="constrained")
    title = (
        "Exact null distributions: all 24 labelings of four cells"
        if exact
        else "The variance error persists on a 16 × 16 grid"
    )
    subtitle = (
        "Expected distribution from independent mathematical definitions · exact KS distances"
        if exact
        else "4,096 tested permutations · 4,096 independent reference permutations · permutation-calibrated KS tests"
    )
    fig.suptitle(f"{title}\n{subtitle}", fontsize=14, fontweight="bold")
    for ax, (mode, (_, label)) in zip(axes, MODES.items(), strict=True):
        before, after = scores[mode]["before"], scores[mode]["after"]
        expected = scores[mode]["expected"]
        if exact:
            unique = np.unique(np.round(np.concatenate([before, after]), 12))
            width = min(np.diff(unique).min() * 0.38, 0.04)
            for values, color, offset, name in [(before, BEFORE, -0.5, "Before"), (after, AFTER, 0.5, "After")]:
                positions, counts = np.unique(np.round(values, 12), return_counts=True)
                ax.bar(positions + offset * width, counts / len(values), width=width, color=color, label=name)
            positions, counts = np.unique(np.round(expected, 12), return_counts=True)
            ax.plot(
                positions,
                counts / len(expected),
                linestyle="none",
                marker="_",
                markersize=15,
                markeredgewidth=2.4,
                color=EXPECTED,
                label="Expected (exact)",
            )
            ax.set_ylabel("Exact probability mass")
        else:
            bins = np.histogram_bin_edges(np.concatenate([before, after, expected]), bins=45)
            ax.hist(
                [before, after],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2.2,
                color=[BEFORE, AFTER],
                label=["Before", "After"],
            )
            ax.hist(
                expected,
                bins=bins,
                density=True,
                histtype="step",
                linestyle="--",
                linewidth=2.3,
                color=EXPECTED,
                label="Expected (independent MC)",
            )
            ax.set_ylabel("Estimated density")
        ax.axvline(
            metrics[mode]["expected_mean"], color="#34495E", linestyle="--", linewidth=1.5, label="Expected null mean"
        )
        ax.set_xlabel(label)
        ax.set_title(label, fontsize=14, fontweight="bold", pad=10)
        handles, labels = ax.get_legend_handles_labels()
        order = sorted(range(len(labels)), key=lambda i: ["Before", "After", "Expected"].index(labels[i].split()[0]))
        ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper right", frameon=False, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        b, a = metrics[mode]["before"], metrics[mode]["after"]
        # Avoid displaying numerical roundoff in an analytically zero mean.
        bmean = 0.0 if abs(b["mean"]) < 1e-12 else b["mean"]
        ks = metrics[mode]["ks"]
        if exact:
            ks_text = (
                f"KS before D={ks['before']['statistic']:.4f}    after D={ks['after']['statistic']:.4f}\n"
                "Complete enumeration: no sampling p-value"
            )
        else:
            ks_text = "\n".join(
                f"KS {name:<6} D={ks[name]['statistic']:.4f}   p_perm={ks[name]['pvalue']:.4g}"
                for name in ["before", "after"]
            )
        ax.text(
            0,
            -0.27,
            f"Before   mean {bmean: .5f}   variance {b['variance']:.6f}\n"
            f"After    mean {a['mean']: .5f}   variance {a['variance']:.6f}\n{ks_text}",
            transform=ax.transAxes,
            fontsize=9.5,
            fontfamily="monospace",
            color="#263746",
            linespacing=1.8,
            va="top",
        )
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, "02-exact-null-distributions" if exact else "03-grid-null-distributions")


def grid_graph(side: int) -> csr_matrix:
    rows, cols = [], []
    for r in range(side):
        for c in range(side):
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if 0 <= r + dr < side and 0 <= c + dc < side:
                    rows.append(r * side + c)
                    cols.append((r + dr) * side + c + dc)
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(side**2, side**2))
    return normalize(graph, norm="l1", axis=1)


def main() -> None:
    set_num_threads(1)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    weights = np.array([[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]])
    exact_graph = csr_matrix(weights)
    exact_values = np.array([[0.0, 1.0, 2.0, 3.0]])
    exact_orders = np.array(list(permutations(range(4))))
    exact_scores = compare(exact_graph, exact_values, exact_orders)
    exact_metrics = summary(exact_scores, n=4, exact=True)
    export_scores(exact_scores, "exact-null-scores.csv")
    plot_mechanism(weights)
    plot_distributions(exact_scores, exact_metrics, exact=True)

    graph = grid_graph(16)
    expression_rng, permutation_rng = np.random.default_rng(42).spawn(2)
    values = expression_rng.normal(size=(1, graph.shape[0]))
    orders = np.array([permutation_rng.permutation(graph.shape[0]) for _ in range(4096)])
    reference_rng = np.random.default_rng(43)
    reference_orders = np.array([reference_rng.permutation(graph.shape[0]) for _ in range(4096)])
    grid_scores = compare(graph, values, orders, reference_orders=reference_orders)
    grid_metrics = summary(grid_scores, n=graph.shape[0], exact=False, ks_rng=np.random.default_rng(44))
    export_scores(grid_scores, "grid-null-scores.csv")
    plot_distributions(grid_scores, grid_metrics, exact=False)

    result = {
        "exact": {"n_cells": 4, "n_permutations": 24, "metrics": exact_metrics},
        "grid": {
            "n_cells": graph.shape[0],
            "n_permutations": len(orders),
            "seed": 42,
            "n_reference_permutations": len(reference_orders),
            "reference_seed": 43,
            "ks_seed": 44,
            "ks_resamples": KS_RESAMPLES,
            "ks_score_rounding_decimals": 12,
            "metrics": grid_metrics,
        },
        "versions": {name: version(name) for name in ["numpy", "scipy", "pandas", "scanpy", "matplotlib", "numba"]},
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
