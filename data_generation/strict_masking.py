from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExactMaskResult:
    train_mask: np.ndarray
    test_mask: np.ndarray
    train_rate: np.ndarray
    valid_rate: np.ndarray
    test_rate: np.ndarray
    train_valid_max_gap: float
    all_split_max_gap: float
    tries: int

    @property
    def max_gap(self) -> float:
        return self.all_split_max_gap


def exact_weighted_mask(
    probabilities: np.ndarray,
    missing_columns: np.ndarray,
    overall_missing_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample an exact number of missing logical cells without replacement.

    The requested overall rate is rounded to the nearest feasible cell count.
    Within eligible columns, logistic probabilities determine relative sampling
    weights.  Columns outside ``missing_columns`` remain fully observed.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    missing_columns = np.asarray(missing_columns, dtype=np.int64)
    n, d = probabilities.shape
    missing_count = int(round(float(overall_missing_rate) * n * d))
    eligible_count = n * missing_columns.size
    if missing_count > eligible_count:
        raise ValueError(
            f"Requested {missing_count} missing cells but only {eligible_count} are eligible"
        )

    mask = np.zeros((n, d), dtype=bool)
    if missing_count == 0:
        return mask

    weights = probabilities[:, missing_columns].reshape(-1)
    weights = np.where(np.isfinite(weights), np.clip(weights, 0.0, None), 0.0)
    if not np.any(weights > 0.0):
        weights = np.ones_like(weights)
    else:
        weights = np.maximum(weights, np.finfo(np.float64).tiny)
    selected = rng.choice(
        eligible_count,
        size=missing_count,
        replace=False,
        p=weights / weights.sum(),
    )
    row_indices = selected // missing_columns.size
    column_indices = missing_columns[selected % missing_columns.size]
    mask[row_indices, column_indices] = True
    return mask


def partition_counts_ok(
    mask: np.ndarray,
    missing_columns: np.ndarray,
    min_observed_rate: float,
    min_missing_rate: float,
    min_count: int,
) -> bool:
    n = mask.shape[0]
    min_observed = min(n, max(int(np.ceil(n * min_observed_rate)), int(min_count)))
    min_missing = min(n, max(int(np.ceil(n * min_missing_rate)), int(min_count)))
    if np.any((~mask).sum(axis=0) < min_observed):
        return False
    return not np.any(mask[:, missing_columns].sum(axis=0) < min_missing)


def feature_rate_gap(left_mask: np.ndarray, right_mask: np.ndarray) -> float:
    if left_mask.shape[1] == 0:
        return 0.0
    return float(np.max(np.abs(left_mask.mean(axis=0) - right_mask.mean(axis=0))))


def all_split_rate_gap(
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    test_mask: np.ndarray,
) -> float:
    rates = [train_mask.mean(axis=0), valid_mask.mean(axis=0), test_mask.mean(axis=0)]
    return float(
        np.max(
            np.maximum.reduce(
                [
                    np.abs(rates[0] - rates[1]),
                    np.abs(rates[0] - rates[2]),
                    np.abs(rates[1] - rates[2]),
                ]
            )
        )
    )


def sample_exact_split_masks(
    train_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    missing_columns: np.ndarray,
    overall_missing_rate: float,
    max_train_valid_gap: float,
    min_observed_rate: float,
    min_missing_rate: float,
    min_count: int,
    max_tries: int,
    rng: np.random.Generator,
) -> ExactMaskResult:
    """Create exact-rate masks without using test outcomes for model selection."""
    train_indices = np.asarray(train_indices, dtype=np.int64)
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    best_train = None
    best_valid = None
    best_gap = float("inf")
    accepted_try = max_tries

    for attempt in range(1, max_tries + 1):
        strict_train_mask = exact_weighted_mask(
            train_probabilities[train_indices], missing_columns, overall_missing_rate, rng
        )
        valid_mask = exact_weighted_mask(
            train_probabilities[valid_indices], missing_columns, overall_missing_rate, rng
        )
        counts_ok = partition_counts_ok(
            strict_train_mask,
            missing_columns,
            min_observed_rate,
            min_missing_rate,
            min_count,
        ) and partition_counts_ok(
            valid_mask,
            missing_columns,
            min_observed_rate,
            min_missing_rate,
            min_count,
        )
        gap = feature_rate_gap(strict_train_mask, valid_mask)
        score = gap + (0.0 if counts_ok else 100.0)
        if score < best_gap:
            best_gap = score
            best_train = strict_train_mask
            best_valid = valid_mask
            accepted_try = attempt
        if counts_ok and gap <= max_train_valid_gap:
            break

    assert best_train is not None and best_valid is not None

    # Test masks are generated only after the train/validation candidate is fixed.
    # Retrying here enforces basic evaluability but never optimizes test agreement.
    test_mask = None
    for _ in range(max_tries):
        candidate = exact_weighted_mask(
            test_probabilities, missing_columns, overall_missing_rate, rng
        )
        if partition_counts_ok(
            candidate,
            missing_columns,
            min_observed_rate,
            min_missing_rate,
            min_count,
        ):
            test_mask = candidate
            break
        if test_mask is None:
            test_mask = candidate
    assert test_mask is not None

    full_train_mask = np.zeros_like(train_probabilities, dtype=bool)
    full_train_mask[train_indices] = best_train
    full_train_mask[valid_indices] = best_valid
    train_rate = best_train.mean(axis=0)
    valid_rate = best_valid.mean(axis=0)
    test_rate = test_mask.mean(axis=0)
    return ExactMaskResult(
        train_mask=full_train_mask,
        test_mask=test_mask,
        train_rate=train_rate,
        valid_rate=valid_rate,
        test_rate=test_rate,
        train_valid_max_gap=feature_rate_gap(best_train, best_valid),
        all_split_max_gap=all_split_rate_gap(best_train, best_valid, test_mask),
        tries=accepted_try,
    )
