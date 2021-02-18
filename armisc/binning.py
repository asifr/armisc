from numba import jit, int64
from numba.typed import List
import numpy as np


def _find_binning_thresholds(data, max_bins=256, subsample=int(2e5)):
    if not (2 <= max_bins <= 256):
        raise ValueError(
            "max_bins={} should be no smaller than 2 "
            "and no larger than 256.".format(max_bins)
        )

    percentiles = np.linspace(0, 100, num=max_bins + 1)
    percentiles = percentiles[1:-1]
    binning_thresholds = List()
    for f_idx in range(data.shape[1]):
        col_data = np.ascontiguousarray(data[:, f_idx], dtype=np.float64)
        mask = np.isfinite(col_data)
        col_data = col_data[mask]
        distinct_values = np.unique(col_data)
        if len(distinct_values) <= max_bins:
            midpoints = distinct_values[:-1] + distinct_values[1:]
            midpoints *= 0.5
        else:
            midpoints = np.percentile(
                col_data, percentiles, interpolation="midpoint"
            ).astype(np.float64)
        binning_thresholds.append(np.unique(midpoints))
    return binning_thresholds


@jit(nopython=True)
def _map_to_bins(data, binning_thresholds, binned):
    """Bin numerical values to discrete integer-coded levels."""
    for feature_idx in range(data.shape[1]):
        _map_num_col_to_bins(
            data[:, feature_idx],
            binning_thresholds[feature_idx],
            binned[:, feature_idx],
        )


@jit(nopython=True)
def _map_num_col_to_bins(data, binning_thresholds, binned):
    for i in range(data.shape[0]):
        left, right = 0, binning_thresholds.shape[0]
        while left < right:
            middle = (right + left - 1) // 2
            if data[i] <= binning_thresholds[middle]:
                right = middle
            else:
                left = middle + 1
        binned[i] = left


def _assign_nan_to_bin(binned, X, actual_n_bins, assign_nan_to_unique_bin=False):
    mask = np.isnan(X)
    for i in range(X.shape[1]):
        binned[mask[:, i], i] = actual_n_bins[i] if assign_nan_to_unique_bin else 999
    return binned


def apply_binning(X, bin_thresholds, assign_nan_to_unique_bin=False):
    binned = np.zeros_like(X, dtype=np.uint32, order="F")
    _map_to_bins(X, bin_thresholds, binned)
    actual_n_bins = np.array(
        [thresholds.shape[0] + 1 for thresholds in bin_thresholds], dtype=np.uint32
    )
    binned = _assign_nan_to_bin(binned, X, actual_n_bins, assign_nan_to_unique_bin)
    return binned


def binning(X: np.ndarray, assign_nan_to_unique_bin: bool=False, max_bins: int=256):
    """Returns the binned design matrix, bin thresholds, and number of bins for each feature
    nan values are assigned to the largest bin, it may be desirable to assign binned values to
    a unique bin of their own.

    Args:
        X (np.ndarray): design matrix
        assign_nan_to_unique_bin (bool): should missing values be assigned to a unique bin
        max_bins (int): maximum number of bins

    Returns:
        X_binned (np.ndarray): binned design matrix
        bin_thresholds (List[np.ndarray]): list of bin thresholds
        actual_n_bins (List[int]): number of bins for each feature
    """
    bin_thresholds = _find_binning_thresholds(X, max_bins=max_bins)
    actual_n_bins = np.array(
        [thresholds.shape[0] + 1 for thresholds in bin_thresholds], dtype=np.uint32
    )
    X_binned = apply_binning(
        X, bin_thresholds, assign_nan_to_unique_bin=assign_nan_to_unique_bin
    )
    return X_binned, bin_thresholds, actual_n_bins
