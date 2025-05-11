import math
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def plot_similarity_matrix(
    similarity_matrix: list[list[float]],
    similarity_type: Literal["Inner", "Reference", "Conditioned"] = "Inner",
    vmin: float | None = 0.5,
    vmax: float | None = 1.0,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots()
    cax = ax.imshow(
        similarity_matrix,
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
    )
    ax.set_title("Measure Similarity Matrix Heatmap")
    ax.set_xlabel(
        "Measure Index"
        if similarity_type == "Inner"
        else ("Reference Piece" if similarity_type == "Reference" else "Conditioned Prefix")
    )
    ax.set_ylabel(
        "Measure Index"
        if similarity_type == "Inner"
        else ("Examined Piece" if similarity_type == "Reference" else "Whole Piece")
    )
    fig.colorbar(cax, ax=ax)
    plt.show()


def plot_distribution_of_best_similarities(
    best_matches: list[tuple[float, list[int], list[int]]],
    n_bins: int | None = None,
    bin_min: float | None = None,
    bin_max: float | None = None,
) -> None:
    # Extract all best‐similarity values
    best_sims = [sim for sim, _, _ in best_matches]

    # Decide on bin edges (e.g. 20 bins between 0 and 1)
    raw_min, raw_max = min(best_sims), max(best_sims)
    factor = 20  # use 100 for two decimals, 1 for integers, etc.
    round_min = math.floor(raw_min * factor) / factor
    round_max = math.ceil(raw_max * factor) / factor

    num_bins = n_bins if n_bins else 21
    bins = np.linspace(bin_min if bin_min else round_min, bin_max if bin_max else round_max, num_bins)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(best_sims, bins=list(bins), edgecolor="black")

    # force y-ticks to integers
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # set the x‐tick *positions*
    ax.set_xticks(bins)

    # then rotate the *labels* via tick_params
    ax.tick_params(axis="x", rotation=45)

    ax.set_xlabel("Best Similarity Value")
    ax.set_ylabel("Count of Measures")
    ax.set_title("Distribution of Best Similarities")
    fig.tight_layout()
    plt.show()
    fig.tight_layout()
    plt.show()


def plot_distribution_of_best_match_measure_distances(
    best_matches: list[tuple[float, list[int], list[int]]],
    best_matches_type: Literal["first", "closest", "all"] | None = None,
) -> None:
    all_diffs = []
    for _, _, diffs in best_matches:
        all_diffs.extend(diffs)
    max_diff = max(all_diffs) if all_diffs else 0
    bins = range(0, max_diff + 2)  # one bin per integer

    plt.figure(figsize=(8, 4))
    plt.hist(all_diffs, bins=bins, align="left", edgecolor="black")
    plt.xticks(bins)
    plt.xlabel("Absolute Measure Index Difference (|j - i|)")
    plt.ylabel("Count of Matches")
    plt.title(
        f"Distribution of {best_matches_type.capitalize() + " " if best_matches_type is not None else ''}"
        "Best-Match Measure Distances"
    )
    plt.tight_layout()
    plt.show()
