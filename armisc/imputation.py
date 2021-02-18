import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.base import clone
from sklearn.linear_model import BayesianRidge


def initial_imputation(X, strategy="mean"):
    imp = X.copy()
    if strategy == "mean":
        means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        imp[inds] = np.take(means, inds[1])
    return imp


def abs_cor(X_filled):
    abs_corr_mat = np.abs(np.corrcoef(X_filled.T))
    abs_corr_mat[np.isnan(abs_corr_mat)] = 1e-6  # indefined for features with zero std
    np.clip(abs_corr_mat, 1e-6, None, out=abs_corr_mat)
    np.fill_diagonal(abs_corr_mat, 0)
    abs_corr_mat = normalize(abs_corr_mat, norm="l1", axis=0, copy=False)
    return abs_corr_mat


def fit_multiple_imputation(
    Xdf, n_nearest_features=None, base_predictor=BayesianRidge()
):
    if n_nearest_features is None:
        n_nearest_features = Xdf.shape[1] - 1

    predictors = {}

    feature_names = np.array(Xdf.columns)
    mask_missing_values = Xdf.isna()
    X_filled = initial_imputation(Xdf.values)

    # order of imputation from highest missing rate to lowest
    frac_of_missing_values = mask_missing_values.mean(axis=0).values
    missing_values_idx = np.nonzero(frac_of_missing_values)[0]
    n = len(frac_of_missing_values) - len(missing_values_idx)
    ordered_idx = np.argsort(frac_of_missing_values, kind="mergesort")[n:][::-1]
    n_features_with_missing_ = len(ordered_idx)

    # absolute correlation matrix
    abs_corr_mat = abs_cor(X_filled)

    n_samples, n_features = X_filled.shape
    for feat_idx in ordered_idx:
        # select nearest features to predict feat_idx from absolute correlation
        # limit the number of neightbors by n_nearest_features
        p = abs_corr_mat[:, feat_idx]
        neighbor_feat_idx = np.random.choice(
            np.arange(n_features), n_nearest_features, replace=False, p=p
        )

        # indices of missing rows for feat_idx
        missing_row_mask = mask_missing_values.values[:, feat_idx]

        # create a new estimator
        predictor = clone(base_predictor)

        # train predictor on observed values
        xtr = X_filled[:, neighbor_feat_idx][~missing_row_mask, :]  # observed
        ytr = X_filled[:, feat_idx][~missing_row_mask]  # observed
        xts = X_filled[:, neighbor_feat_idx][missing_row_mask, :]  # missing
        predictor.fit(xtr, ytr)

        # get posterior
        mus, sigmas = predictor.predict(xts, return_std=True)
        good_sigmas = sigmas > 0
        imputed_values = np.zeros(mus.shape, dtype=X_filled.dtype)
        imputed_values[~good_sigmas] = mus[~good_sigmas]
        imputed_values[good_sigmas] = np.random.normal(
            loc=mus[good_sigmas], scale=sigmas[good_sigmas]
        )
        imputed_values = np.clip(imputed_values, ytr.min(), ytr.max())

        # update the feature with imputed values
        X_filled[missing_row_mask, feat_idx] = imputed_values

        predictors[feat_idx] = {
            "predictor": predictor,
            "neighbor_feat_idx": neighbor_feat_idx,
            "feat_idx": feat_idx,
            "minval": ytr.min(),
            "maxval": ytr.max(),
        }
    X_filled[~mask_missing_values] = Xdf.values[~mask_missing_values]
    return X_filled, predictors


def impute(X, predictors):
    # fill missing values
    X_filled = initial_imputation(X)
    mask_missing_values = np.isnan(X)
    for predictor in predictors.values():
        clf = predictor["predictor"]
        neighbor_feat_idx = predictor["neighbor_feat_idx"]
        feat_idx = predictor["feat_idx"]
        minval = predictor["minval"]
        maxval = predictor["maxval"]

        # indices of missing rows for feat_idx
        missing_row_mask = mask_missing_values[:, feat_idx]

        mus, sigmas = clf.predict(
            X_filled[missing_row_mask, :][:, neighbor_feat_idx], return_std=True
        )
        good_sigmas = sigmas > 0
        imputed_values = np.zeros(mus.shape, dtype=X_filled.dtype)
        imputed_values[~good_sigmas] = mus[~good_sigmas]
        imputed_values[good_sigmas] = np.random.normal(
            loc=mus[good_sigmas], scale=sigmas[good_sigmas]
        )
        imputed_values = np.clip(imputed_values, minval, maxval)

        # updated missing values
        X_filled[missing_row_mask, feat_idx] = imputed_values
    return X_filled


def knnimputer(df, n_neighbors=10):
    from sklearn.impute import KNNImputer
    imp = KNNImputer(n_neighbors=n_neighbors)
    imp.fit(df)
    return imp