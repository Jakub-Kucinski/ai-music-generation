import math
from typing import Callable, Literal, TypeVar

from pydantic import BaseModel

vectorT = TypeVar("vectorT")


class SimilarityResult(BaseModel):
    similarity_matrix: list[list[float]]
    best_matches: list[tuple[float, list[int], list[int]]]
    mean_best_similarities: float


def calculate_inner_similarity_of_music_vectors(
    vectors: list[list[vectorT]],
    similarity_function: Callable[[list[vectorT], list[vectorT]], float],
    return_best_matches: Literal["first", "closest", "all"] = "all",
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> SimilarityResult:
    """
    Given a list of feature vectors (one per measure) and a function
    that computes similarity between two vectors, returns:

      1) similarity_matrix: an NxN matrix of similarities between measures
      2) best_matches: for each measure i, a tuple
           (max_sim, best_js, diffs)
         where best_js is a list of indices j achieving max_sim
         and diffs = [abs(j - i) for j in best_js].
      3) mean_max_sim: the arithmetic mean of all the max_sim values in best_matches.

    The return_best_matches flag controls which matches are kept:
      - "all": keep all ties
      - "first": keep only the first tie
      - "closest": keep only the tie with smallest |j - i|
    """
    n = len(vectors)
    # 1) Build similarity matrix
    similarity_matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            similarity_matrix[i][j] = similarity_function(vectors[i], vectors[j])

    # 2) for each i, find the maximum sim over j != i,
    best_matches: list[tuple[float, list[int], list[int]]] = []
    for i in range(n):
        # find max similarity excluding self
        max_sim = max(similarity_matrix[i][j] for j in range(n) if j != i)

        # gather all js within tolerance of max_sim
        tied_js = [
            j
            for j in range(n)
            if j != i and math.isclose(similarity_matrix[i][j], max_sim, rel_tol=rel_tol, abs_tol=abs_tol)
        ]
        tied_diffs = [abs(j - i) for j in tied_js]

        if return_best_matches == "first":
            tied_js = tied_js[:1]
            tied_diffs = tied_diffs[:1]
        elif return_best_matches == "closest":
            # find the minimal diff
            min_diff = min(tied_diffs)
            # pick the first j among those at min_diff
            for j, d in zip(tied_js, tied_diffs):
                if d == min_diff:
                    tied_js = [j]
                    tied_diffs = [d]
                    break
        # else "all": leave tied_js and tied_diffs as-is

        best_matches.append((max_sim, tied_js, tied_diffs))

    # 3) Compute mean of all the max_sim values
    mean_max_sim = sum(match[0] for match in best_matches) / len(best_matches) if best_matches else 0.0
    similarity_result = SimilarityResult(
        similarity_matrix=similarity_matrix,
        best_matches=best_matches,
        mean_best_similarities=mean_max_sim,
    )
    return similarity_result


def calculate_reference_similarity_of_music_vectors(
    vectors1: list[list[vectorT]],
    vectors2: list[list[vectorT]],
    similarity_function: Callable[[list[vectorT], list[vectorT]], float],
    return_best_matches: Literal["first", "closest", "all"] = "all",
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> SimilarityResult:
    """
    Given two lists of feature-vectors (one per measure) from piece A and piece B,
    and a similarity function sim(u, v) -> float, returns:

      1) similarity_matrix: an MxN matrix of similarities between each measure i of piece A
         and each measure j of piece B
      2) best_matches: for each measure i in piece A, a tuple
           (max_sim, best_js, diffs)
         where
           - max_sim is the highest similarity sim(vectors1[i], vectors2[j]) over j
           - best_js is the list of all j achieving max_sim (modulated by return_best_matches)
           - diffs is [abs(j - i) for j in best_js], giving how far apart the measures
             are by index (you can drop or reinterpret this if it's not meaningful)
      3) mean_max_sim: the arithmetic mean of all the max_sim values in best_matches.

    The return_best_matches flag controls how ties are handled:
      - "all":   keep all js within tolerance of max_sim
      - "first": keep only the first such j
      - "closest": among tied js, keep the one(s) with minimal |j - i|

      # similarity_matrix[i][j] is sim between measure i of A and measure j of B
      # best_matches[i] is (max_sim, matching_js, diffs) for measure i of A
    """
    m = len(vectors1)
    n = len(vectors2)

    # 1) Build cross-similarity matrix
    similarity_matrix: list[list[float]] = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            similarity_matrix[i][j] = similarity_function(vectors1[i], vectors2[j])

    # 2) For each measure i in piece A, find best matches among piece B
    best_matches: list[tuple[float, list[int], list[int]]] = []
    for i in range(m):
        row = similarity_matrix[i]
        max_sim = max(row)

        # all js within tolerance of max_sim
        tied_js = [j for j, s in enumerate(row) if math.isclose(s, max_sim, rel_tol=rel_tol, abs_tol=abs_tol)]
        tied_diffs = [abs(j - i) for j in tied_js]

        if return_best_matches == "first" and tied_js:
            tied_js = tied_js[:1]
            tied_diffs = tied_diffs[:1]
        elif return_best_matches == "closest" and tied_js:
            min_diff = min(tied_diffs)
            for j, d in zip(tied_js, tied_diffs):
                if d == min_diff:
                    tied_js = [j]
                    tied_diffs = [d]
                    break

        best_matches.append((max_sim, tied_js, tied_diffs))

    # 3) Compute mean of all the max_sim values
    mean_max_sim = sum(match[0] for match in best_matches) / len(best_matches) if best_matches else 0.0

    similarity_result = SimilarityResult(
        similarity_matrix=similarity_matrix,
        best_matches=best_matches,
        mean_best_similarities=mean_max_sim,
    )
    return similarity_result


def calculate_conditioned_similarity_of_music_vectors(
    vectors: list[list[vectorT]],
    conditioned_n_measures: int,
    similarity_function: Callable[[list[vectorT], list[vectorT]], float],
    return_best_matches: Literal["first", "closest", "all"] = "all",
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> SimilarityResult:
    """
    Given a single list of feature-vectors (one per measure) and an integer k,
    builds similarities for **every** measure i against the **first** k measures,
    then—for each measure i >= k—finds its best match(es) among those first k.

    Returns:
      1) similarity_matrix: an Nxk matrix, where similarity_matrix[i][j] is the similarity
         between measure i and measure j (0 <= j < k).
      2) best_matches: for each i in [k, N), a tuple
           (max_sim, best_js, diffs)
         where
           - max_sim is the highest similarity over j in [0, k)
           - best_js is the list of all j achieving max_sim (modulated by return_best_matches)
           - diffs  = [abs(j - i) for j in best_js]
      3) mean_max_sim: the arithmetic mean of all the max_sim values in best_matches.

    Tie-breaking (return_best_matches):
      - "all":     keep all ties
      - "first":   keep only the first tie
      - "closest": keep the tie(s) with smallest |j - i|

    Raises:
      ValueError if conditioned_n_measures is not in 1..len(vectors).

      # similarity_matrix[i][j] compares measure i to measure j (j<8)
      # best_matches[i-8] gives best match for measure i (i>=8)
    """
    n = len(vectors)
    k = conditioned_n_measures
    if not (1 <= k <= n):
        raise ValueError(f"conditioned_n_measures must be between 1 and {n}, got {k}")

    # 1) Build N×k similarity matrix
    similarity_matrix: list[list[float]] = [
        [similarity_function(vectors[i], vectors[j]) for j in range(k)] for i in range(n)
    ]

    # 2) For each measure i >= k, find best match(es) among j < k
    best_matches: list[tuple[float, list[int], list[int]]] = []
    for i in range(k, n):
        row = similarity_matrix[i]
        max_sim = max(row)

        # find all js within tolerance of max_sim
        tied_js = [j for j, s in enumerate(row) if math.isclose(s, max_sim, rel_tol=rel_tol, abs_tol=abs_tol)]
        tied_diffs = [abs(j - i) for j in tied_js]

        if return_best_matches == "first" and tied_js:
            tied_js = tied_js[:1]
            tied_diffs = tied_diffs[:1]
        elif return_best_matches == "closest" and tied_js:
            min_diff = min(tied_diffs)
            for j, d in zip(tied_js, tied_diffs):
                if d == min_diff:
                    tied_js = [j]
                    tied_diffs = [d]
                    break
        # "all" leaves them as-is

        best_matches.append((max_sim, tied_js, tied_diffs))

    # 3) Compute mean of all the max_sim values
    mean_max_sim = sum(match[0] for match in best_matches) / len(best_matches) if best_matches else 0.0

    similarity_result = SimilarityResult(
        similarity_matrix=similarity_matrix,
        best_matches=best_matches,
        mean_best_similarities=mean_max_sim,
    )
    return similarity_result


def aggregate_similarity_results(similarity_results: list[SimilarityResult]) -> SimilarityResult:
    """
    Aggregate a list of SimilarityResult objects whose matrices may have different
    numbers of rows and columns.

    - The output matrix has size R x C, where
        R = max number of rows over all input matrices,
        C = max number of columns over all input matrices.
    - At each (i,j) we average only those inputs that have a value at (i,j).
    - best_matches is the concatenation of all input .best_matches lists.
    - mean_best_similarities is the arithmetic mean of all sim-values in that concatenation.
    """
    if not similarity_results:
        return SimilarityResult(
            similarity_matrix=[],
            best_matches=[],
            mean_best_similarities=0.0,
        )

    # 1) Find output dimensions
    max_rows = max(len(res.similarity_matrix) for res in similarity_results)
    max_cols = 0
    for res in similarity_results:
        for row in res.similarity_matrix:
            if len(row) > max_cols:
                max_cols = len(row)

    # 2) Prepare accumulators
    accum = [[0.0] * max_cols for _ in range(max_rows)]
    counts = [[0] * max_cols for _ in range(max_rows)]

    # 3) Sum up values where present
    for res in similarity_results:
        mat = res.similarity_matrix
        # enforce that each input matrix is rectangular
        if any(len(row) != len(mat[0]) for row in mat):
            raise ValueError("Each input similarity_matrix must be rectangular")
        for i, row in enumerate(mat):
            for j, val in enumerate(row):
                accum[i][j] += val
                counts[i][j] += 1

    # 4) Compute element‐wise means
    averaged: list[list[float]] = [
        [(accum[i][j] / counts[i][j]) if counts[i][j] > 0 else 0.0 for j in range(max_cols)] for i in range(max_rows)
    ]

    # 5) Concatenate all best_matches
    aggregated_best_matches: list[tuple[float, list[int], list[int]]] = []
    for res in similarity_results:
        aggregated_best_matches.extend(res.best_matches)

    # 6) Compute overall mean of all max_sim values
    if aggregated_best_matches:
        mean_max = sum(sim for sim, _, _ in aggregated_best_matches) / len(aggregated_best_matches)
    else:
        mean_max = 0.0

    return SimilarityResult(
        similarity_matrix=averaged,
        best_matches=aggregated_best_matches,
        mean_best_similarities=mean_max,
    )
