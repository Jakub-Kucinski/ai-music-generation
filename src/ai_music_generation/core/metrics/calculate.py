import math
from typing import Callable, Literal, TypeVar

vectorT = TypeVar("vectorT")


def calculate_inner_similarity_of_music_vectors(
    vectors: list[list[vectorT]],
    similarity_function: Callable[[list[vectorT], list[vectorT]], float],
    return_best_matches: Literal["first", "closest", "all"] = "all",
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> tuple[list[list[float]], list[tuple[float, list[int], list[int]]]]:
    """
    Given a list of feature vectors (one per measure) and a function
    that computes similarity between two vectors, returns:

      1) sim_matrix: an N×N matrix of similarities between measures
      2) best_matches: for each measure i, a tuple
           (max_sim, best_js, diffs)
         where best_js is a list of indices j achieving max_sim
         and diffs = [abs(j - i) for j in best_js].

    The return_best_matches flag controls which matches are kept:
      - "all": keep all ties
      - "first": keep only the first tie
      - "closest": keep only the tie with smallest |j - i|

    Regardless of the flag, each entry in best_matches is
    always (float, list[int], list[int]).

    Example usage:
    vectorizer = MidiVectorizer()
    pitches_distributions, offsets = vectorizer.midi_or_score_to_notes_and_offsets_feature_vectors(
        score
    )
    melodic_similarity_matrix, melodic_best_matches = calculate_inner_similarity_of_music_vectors(
        pitches_distributions,
        cyclic_pitch_similarity,
    )
    plot_similarity_matrix(melodic_similarity_matrix)
    rhythmic_similarity_matrix, rhythmic_best_matches = calculate_inner_similarity_of_music_vectors(
        offsets,
        rhythmic_similarity,
    )
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

    return similarity_matrix, best_matches


def calculate_cross_similarity_of_music_vectors(
    vectors1: list[list[vectorT]],
    vectors2: list[list[vectorT]],
    similarity_function: Callable[[list[vectorT], list[vectorT]], float],
    return_best_matches: Literal["first", "closest", "all"] = "all",
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> tuple[list[list[float]], list[tuple[float, list[int], list[int]]]]:
    """
    Given two lists of feature-vectors (one per measure) from piece A and piece B,
    and a similarity function sim(u, v) -> float, returns:

      1) sim_matrix: an M×N matrix of similarities between each measure i of piece A
         and each measure j of piece B
      2) best_matches: for each measure i in piece A, a tuple
           (max_sim, best_js, diffs)
         where
           - max_sim is the highest similarity sim(vectors1[i], vectors2[j]) over j
           - best_js is the list of all j achieving max_sim (modulated by return_best_matches)
           - diffs is [abs(j - i) for j in best_js], giving how far apart the measures
             are by index (you can drop or reinterpret this if it's not meaningful)

    The return_best_matches flag controls how ties are handled:
      - "all":   keep all js within tolerance of max_sim
      - "first": keep only the first such j
      - "closest": among tied js, keep the one(s) with minimal |j - i|

    Example usage:
      sim_matrix, best = calculate_cross_similarity_of_music_vectors(
          pitches_pieceA,
          pitches_pieceB,
          cyclic_pitch_similarity,
      )
      # sim_matrix[i][j] is sim between measure i of A and measure j of B
      # best[i] is (max_sim, matching_js, diffs) for measure i of A
    """
    m = len(vectors1)
    n = len(vectors2)

    # 1) Build cross-similarity matrix
    sim_matrix: list[list[float]] = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            sim_matrix[i][j] = similarity_function(vectors1[i], vectors2[j])

    # 2) For each measure i in piece A, find best matches among piece B
    best_matches: list[tuple[float, list[int], list[int]]] = []
    for i in range(m):
        row = sim_matrix[i]
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

    return sim_matrix, best_matches
