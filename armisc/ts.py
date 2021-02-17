from typing import List
import numpy as np


def times_to_lags(T):
    """(N x n_step) matrix of times -> (N x n_step) matrix of lags.
    First time is assumed to be zero.
    """
    assert T.ndim == 2, "T must be an (N x n_step) matrix"
    return np.c_[np.diff(T, axis=1), np.zeros(T.shape[0])]


def lags_to_times(dT):
    """(N x n_step) matrix of lags -> (N x n_step) matrix of times
    First time is assumed to be zero.
    """
    assert dT.ndim == 2, "dT must be an (N x n_step) matrix"
    return np.c_[np.zeros(dT.shape[0]), np.cumsum(dT[:, :-1], axis=1)]


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def windows(maxlen, winsize):
    i = np.arange(maxlen)
    inds = rolling_window(i, winsize)
    seqlen = [len(i) for i in inds]
    ix = np.tril(np.arange(winsize)).astype(float)
    ix[np.triu_indices(winsize,1)] = np.nan
    ix = ix[:-1,:]
    seqlen = [len(i[np.isfinite(i)]) for i in ix] + seqlen
    inds = np.concatenate([ix,inds], 0)
    inds = [i[np.isfinite(i)].astype('int') for i in inds]
    return seqlen, inds


def get_group_inds(a: np.ndarray) -> List[np.ndarray]:
    """Get indices after grouping a sorted list of IDs
    Args:
        a (np.ndarray): sorted list of IDs
    Example:
        group_inds, group_ids, inds = get_group_inds(a)
    """
    u, inds = np.unique(a, return_index=True)
    return np.split(np.arange(len(a)), inds[1:]), u, inds


def grouped_expanding_windows(group_seq: np.ndarray, times: np.ndarray, maxt=720, maxwinsize: int=24) -> List[np.ndarray]:
    group_inds, _, _= get_group_inds(group_seq)
    segment_inds = []
    for g in group_inds:
        maxlen = len(g)
        winsize = maxwinsize if maxlen > maxwinsize else maxlen
        s, inds = windows(maxlen, winsize)
        for ind in inds:
            i = np.where((times[ind[-1]] - times[ind]) <= maxt)[0]
            segment_inds.append(g[ind[i]])
    return segment_inds


def ffill(arr: np.ndarray) -> np.ndarray:
    arr = arr.T
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx].T
    return out