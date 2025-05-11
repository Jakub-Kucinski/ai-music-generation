from collections import Counter
from fractions import Fraction


def cyclic_pitch_similarity(
    v1: list[float],
    v2: list[float],
) -> float:
    """
    Compute the similarity between two 12-dimensional pitch-class
    distributions v1 and v2 by:
      1) For each of the 12 cyclic shifts of v2,
         • For each i in 0..11 compute Dice-style Coefficient
             term_i = 2 * min(v1[i], v2_shift[i]) / (v1[i] + v2_shift[i])
           with the convention that if v1[i] == v2_shift[i] == 0, term_i = 1.
         • Take the mean of the 12 term_i values to get sim_k.
      2) Return max(sim_k) over all 12 shifts.

    Raises:
      ValueError if either input is not length 12.
    """
    if len(v1) != 12 or len(v2) != 12:
        raise ValueError("Both vectors must be length 12")

    best_sim = 0.0
    # try all 12 cyclic rotations of v2
    for shift in range(12):
        v2_shift = v2[shift:] + v2[:shift]
        total = 0.0
        for a, b in zip(v1, v2_shift):
            if a == 0 and b == 0:
                term = 1.0
            else:
                term = 2 * min(a, b) / (a + b)
            total += term
        sim = total / 12
        if sim > best_sim:
            best_sim = sim

    return best_sim


def rhythmic_similarity(
    v1: list[float | Fraction],
    v2: list[float | Fraction],
) -> float:
    """
    Computes a Dice-style similarity between two sequences of offsets (or any values):
      1) Count occurrences of each distinct value in v1 and v2.
      2) numerator   = 2 * sum(min(count1[x], count2[x]) for x in intersection)
      3) denominator = len(v1) + len(v2)
      4) Return numerator / denominator (or 1.0 if both are empty).

    This ranges from 0 (no shared values) to 1 (identical multisets).
    """
    # Build frequency counters
    c1 = Counter(v1)
    c2 = Counter(v2)

    # Sum of shared counts
    shared = sum(min(c1[val], c2[val]) for val in c1.keys() & c2.keys())

    denom = len(v1) + len(v2)
    if denom == 0:
        return 1.0

    return 2 * shared / denom
