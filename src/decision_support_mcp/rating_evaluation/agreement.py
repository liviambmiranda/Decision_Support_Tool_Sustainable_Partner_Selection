"""Agreement and error statistics for ordinal 0-5 ratings.

Self-contained on purpose: ordinal Krippendorff's alpha is in neither numpy nor
scipy, and pulling scikit-learn in for one kappa would add a heavy runtime
dependency to a decision-support tool that does not otherwise need it. The unit
tests cross-validate every function here against the ``krippendorff`` and
``scikit-learn`` reference implementations.

Missing ratings are written as ``None`` and are handled natively by
Krippendorff's alpha. Quadratic weighted kappa has no such facility, so it drops
units pairwise and reports how many it used.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

import numpy as np

Rating = int | float | None

NOMINAL = "nominal"
ORDINAL = "ordinal"
INTERVAL = "interval"


def median_lower(values: Sequence[float]) -> float:
    """Median that resolves an even-count tie downward.

    With four LLMs the median of two adjacent middle values would fall on ``x.5``,
    which contradicts the integer rubric. Rounding down follows the rubric's own
    instruction to assign the lower rating when evidence falls between two levels.
    """

    if not values:
        raise ValueError("median_lower requires at least one value.")
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    if count % 2 == 1:
        return ordered[count // 2]
    return ordered[count // 2 - 1]


def _difference_matrix(
    domain: Sequence[float],
    marginals: np.ndarray,
    level: str,
) -> np.ndarray:
    """Squared difference function delta^2 over the value domain."""

    size = len(domain)
    delta = np.zeros((size, size), dtype=float)

    if level == NOMINAL:
        for i in range(size):
            for j in range(size):
                delta[i, j] = 0.0 if i == j else 1.0
        return delta

    if level == INTERVAL:
        for i in range(size):
            for j in range(size):
                delta[i, j] = (float(domain[i]) - float(domain[j])) ** 2
        return delta

    if level == ORDINAL:
        # Krippendorff's ordinal metric: the squared distance between two ranks
        # is measured in observed mass lying between them, so the metric adapts
        # to how the scale was actually used.
        for i in range(size):
            for j in range(size):
                low, high = (i, j) if i <= j else (j, i)
                between = float(marginals[low : high + 1].sum())
                delta[i, j] = (between - (marginals[i] + marginals[j]) / 2.0) ** 2
        return delta

    raise ValueError(f"level must be one of: {NOMINAL}, {ORDINAL}, {INTERVAL}.")


def krippendorff_alpha(
    reliability_data: Sequence[Sequence[Rating]],
    level: str = ORDINAL,
    value_domain: Sequence[float] | None = None,
) -> float:
    """Krippendorff's alpha for a units-by-raters matrix.

    ``reliability_data[u][r]`` is rater ``r``'s rating of unit ``u``, or ``None``
    if that rater did not score that unit. Units rated fewer than twice carry no
    agreement information and are dropped, per Krippendorff.

    Returns ``nan`` when alpha is undefined: fewer than two usable units, or a
    degenerate case in which every rating is identical, where expected
    disagreement is zero and the ratio is not formed.
    """

    units = [
        [float(value) for value in unit if value is not None]
        for unit in reliability_data
    ]
    units = [unit for unit in units if len(unit) >= 2]
    if len(units) < 1:
        return float("nan")

    if value_domain is None:
        domain = sorted({value for unit in units for value in unit})
    else:
        domain = sorted({float(value) for value in value_domain})
    index_of = {value: position for position, value in enumerate(domain)}
    size = len(domain)
    if size < 2:
        # Every rater used a single value: complete agreement, no variance to
        # correct for. Alpha is conventionally undefined here.
        return float("nan")

    coincidences = np.zeros((size, size), dtype=float)
    for unit in units:
        pairable = len(unit)
        if pairable < 2:
            continue
        counts = Counter(unit)
        for value_a, count_a in counts.items():
            for value_b, count_b in counts.items():
                i, j = index_of[value_a], index_of[value_b]
                pairs = count_a * (count_b - 1) if i == j else count_a * count_b
                coincidences[i, j] += pairs / (pairable - 1)

    marginals = coincidences.sum(axis=1)
    total = float(marginals.sum())
    if total < 2:
        return float("nan")

    delta = _difference_matrix(domain, marginals, level)

    observed = float((coincidences * delta).sum())
    expected = float((np.outer(marginals, marginals) * delta).sum())
    # The diagonal of the expected term uses n_c(n_c - 1), but delta is zero on
    # the diagonal for every supported level, so no correction is needed.
    if expected == 0.0:
        return float("nan")

    return 1.0 - (total - 1.0) * observed / expected


def quadratic_weighted_kappa(
    rater_a: Sequence[Rating],
    rater_b: Sequence[Rating],
    labels: Sequence[int],
) -> dict[str, Any]:
    """Quadratic weighted kappa against a fixed label set.

    ``labels`` must be passed explicitly and held constant across iterations.
    Inferring labels from the data would silently rescale the quadratic weights
    whenever a model failed to use the full 0-5 range, making kappas from
    different iterations incomparable.
    """

    if len(rater_a) != len(rater_b):
        raise ValueError("quadratic_weighted_kappa inputs must have equal length.")
    label_list = sorted({int(label) for label in labels})
    if len(label_list) < 2:
        raise ValueError("quadratic_weighted_kappa requires at least two labels.")
    index_of = {label: position for position, label in enumerate(label_list)}
    size = len(label_list)

    pairs = [
        (int(a), int(b))
        for a, b in zip(rater_a, rater_b)
        if a is not None and b is not None
    ]
    if not pairs:
        return {"kappa": float("nan"), "n_used": 0, "reason": "no_paired_ratings"}

    unknown = {value for pair in pairs for value in pair} - set(label_list)
    if unknown:
        raise ValueError(f"Ratings outside the declared label set: {sorted(unknown)}.")

    observed = np.zeros((size, size), dtype=float)
    for a, b in pairs:
        observed[index_of[a], index_of[b]] += 1.0

    total = float(len(pairs))
    row_marginal = observed.sum(axis=1)
    col_marginal = observed.sum(axis=0)
    expected = np.outer(row_marginal, col_marginal) / total

    weights = np.zeros((size, size), dtype=float)
    denominator = (size - 1) ** 2
    for i in range(size):
        for j in range(size):
            weights[i, j] = ((label_list[i] - label_list[j]) ** 2) / denominator

    # Both terms stay on the count scale: ``observed`` and ``expected`` each sum
    # to the number of paired ratings, so the ratio is scale-free.
    expected_disagreement = float((weights * expected).sum())
    if expected_disagreement == 0.0:
        # Both raters used exactly one label, and the same one. Agreement is
        # perfect but chance-corrected agreement is not defined.
        return {
            "kappa": float("nan"),
            "n_used": len(pairs),
            "reason": "zero_expected_disagreement",
        }

    observed_disagreement = float((weights * observed).sum())
    return {
        "kappa": 1.0 - observed_disagreement / expected_disagreement,
        "n_used": len(pairs),
        "reason": None,
    }


def mean_absolute_error(predicted: Sequence[Rating], reference: Sequence[Rating]) -> float:
    pairs = _paired(predicted, reference)
    if not pairs:
        return float("nan")
    return float(np.mean([abs(a - b) for a, b in pairs]))


def root_mean_squared_error(predicted: Sequence[Rating], reference: Sequence[Rating]) -> float:
    pairs = _paired(predicted, reference)
    if not pairs:
        return float("nan")
    return float(math.sqrt(np.mean([(a - b) ** 2 for a, b in pairs])))


def _paired(a: Sequence[Rating], b: Sequence[Rating]) -> list[tuple[float, float]]:
    if len(a) != len(b):
        raise ValueError("Error metrics require equal-length inputs.")
    return [
        (float(left), float(right))
        for left, right in zip(a, b)
        if left is not None and right is not None
    ]


def pearson_r(x: Sequence[Rating], y: Sequence[Rating]) -> dict[str, Any]:
    """Pearson correlation over complete pairs.

    Returns ``nan`` when either series is constant: the correlation is undefined
    there, and a constant rater is exactly the case this study keeps running into.
    """

    pairs = _paired(x, y)
    if len(pairs) < 3:
        return {"r": float("nan"), "n": len(pairs), "reason": "fewer_than_3_pairs"}
    left = np.asarray([a for a, _ in pairs], dtype=float)
    right = np.asarray([b for _, b in pairs], dtype=float)
    if left.std() == 0.0 or right.std() == 0.0:
        return {"r": float("nan"), "n": len(pairs), "reason": "constant_series"}
    return {"r": float(np.corrcoef(left, right)[0, 1]), "n": len(pairs), "reason": None}


def icc_a1(ratings: Sequence[Sequence[Rating]]) -> dict[str, Any]:
    """ICC(A,1): two-way mixed effects, absolute agreement, single measurement.

    ``ratings[u][r]`` is rater ``r``'s score for unit ``u``. This is McGraw &
    Wong's ICC(A,1), equivalently Shrout & Fleiss's ICC(2,1). Absolute agreement
    is the right variant when a systematic offset between raters counts as
    disagreement -- an LLM that rates every disclosure one point above the human
    should not score as perfectly reliable.

    Rows with any missing rating are dropped listwise, since the mean-square
    decomposition assumes a complete design.
    """

    complete = [
        [float(value) for value in row]
        for row in ratings
        if all(value is not None for value in row)
    ]
    n = len(complete)
    if n < 2:
        return {"icc": float("nan"), "n_units": n, "n_raters": 0, "reason": "fewer_than_2_units"}
    k = len(complete[0])
    if k < 2 or any(len(row) != k for row in complete):
        return {"icc": float("nan"), "n_units": n, "n_raters": k, "reason": "needs_2_or_more_raters"}

    data = np.asarray(complete, dtype=float)
    grand_mean = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    # Two-way ANOVA decomposition without interaction term.
    ss_rows = k * float(((row_means - grand_mean) ** 2).sum())
    ss_cols = n * float(((col_means - grand_mean) ** 2).sum())
    ss_total = float(((data - grand_mean) ** 2).sum())
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if denominator == 0.0:
        return {"icc": float("nan"), "n_units": n, "n_raters": k, "reason": "zero_variance"}

    return {
        "icc": float((ms_rows - ms_error) / denominator),
        "n_units": n,
        "n_raters": k,
        "reason": None,
    }


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    """Median, IQR, and range of a sample, ignoring undefined entries.

    Used to summarise a statistic computed once per scoring iteration. The median
    and IQR are reported rather than mean and standard deviation because these
    statistics are bounded and skewed.
    """

    finite = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    n_undefined = len(values) - len(finite)
    if not finite:
        return {
            "n": 0,
            "n_undefined": n_undefined,
            "median": None,
            "q1": None,
            "q3": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(finite, dtype=float)
    return {
        "n": len(finite),
        "n_undefined": n_undefined,
        "median": float(np.median(array)),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def bootstrap_ci(
    statistic,
    n_units: int,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> dict[str, float | None]:
    """Percentile bootstrap over units.

    ``statistic`` takes a sequence of unit indices and returns a float. Units are
    the resampling level because they, not the raters or the iterations, are the
    sample the study generalises over. With only a handful of units the interval
    is wide, which is the honest result rather than a defect.
    """

    if n_units < 2:
        return {"low": None, "high": None, "n_resamples": 0}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_resamples):
        indices = rng.integers(0, n_units, size=n_units)
        try:
            value = float(statistic(indices.tolist()))
        except (ValueError, ZeroDivisionError):
            continue
        if not math.isnan(value):
            values.append(value)
    if len(values) < 2:
        return {"low": None, "high": None, "n_resamples": len(values)}
    tail = (1.0 - confidence) / 2.0
    return {
        "low": float(np.percentile(values, tail * 100)),
        "high": float(np.percentile(values, (1.0 - tail) * 100)),
        "n_resamples": len(values),
    }
