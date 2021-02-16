"""Variety of utility functions to make working with dataframes, 
loading files, and saving files easier.
"""

from typing import List
import os
import io
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from IPython.display import HTML, display
import tabulate
from normality import stringify
from hashlib import sha1


def listfiles(path, ext=None):
    """List all files in a directory"""
    return (
        [f for f in os.listdir(path) if f.endswith(ext)]
        if ext
        else [f for f in os.listdir(path)]
    )


def from_parquet(filename, columns=[]):
    """Load a parquet file as a pandas dataframe"""
    if len(columns) > 0:
        return pq.read_table(filename, columns=columns).to_pandas()
    else:
        return pq.read_table(filename).to_pandas()


def to_parquet(df, filename, preserve_index=False):
    """Save a pandas dataframe to parquet file"""
    table = pa.Table.from_pandas(df, preserve_index=preserve_index)
    pq.write_table(table, filename)


def _total_minutes(td):
    return td.total_seconds() / 60


def total_minutes(td):
    """Convert a pandas series of timedeltas to a series of total minutes"""
    return td.apply(_total_minutes)


def _total_hours(td):
    return td.total_seconds() / 60 / 60


def total_hours(td):
    """Convert a pandas series of timedeltas to a series of total hours"""
    return td.apply(_total_hours)


def _total_days(td):
    return td.total_seconds() / 60 / 60 / 24


def total_days(td):
    """Convert a pandas series of timedeltas to a series of total days"""
    return td.apply(_total_days)


def from_julian_time(z):
    return pd.to_datetime(z - pd.Timestamp(0).to_julian_date(), unit="D")


def offset_to_charttime(offset, basedate="2018-01-01"):
    """Converts time offset in minutes to pd.Timestamp. Return a list of pd.Timestamp"""
    return [pd.Timestamp(basedate) + pd.Timedelta(f"{t}m") for t in offset]


def ffill_limit(df: pd.DataFrame, f: str, limit: str = "8h") -> pd.DataFrame:
    """Forward fill a column `f` in the dataframe `df` with a
    maximum `limit` between measurements. `limit` is the duration
    to forward fill.

    Args:
        df (pd.DataFrame): dataframe indexed by timestamp
        f (str): column name to forward fill
        limit (str, optional): max duration in time to forward fill, e.g. 8h

    Returns:
        pd.DataFrame: forward filled dataframe
    """
    tdf = df.copy()
    lastt = None
    lastv = None
    for k, r in tdf.iterrows():
        if not pd.isna(r[f]):
            lastt = k
            lastv = r[f]
        else:
            if lastt is not None and ((k - lastt) <= pd.Timedelta(limit)):
                tdf.loc[k, f] = lastv
    return tdf


def bin_timestamps(dt: pd.DataFrame, f: str, freq: str = "1h") -> pd.Series:
    """Bins pd.Timestamps in `dt` with frequency `freq` and returns
    bin category codes. Code -1 means out of range, usually for values
    at the end of the series without a full `freq` window.

    Args:
        dt (pd.DataFrame): dataframe of timestamps
        f (str): feature column name
        freq (str, optional): frequency

    Returns:
        pd.Series: bins
    """
    bins_dt = pd.date_range(dt.loc[:, f].min(), d.loc[:, f].max(), freq=freq)
    bins_str = bins_dt.astype(str).values
    return pd.cut(
        dt.loc[:, f].astype(np.int64) // 10 ** 9,
        bins=bins_dt.astype(np.int64) // 10 ** 9,
        include_lowest=True,
    ).cat.codes


def to_categorical(
    y: np.ndarray, num_classes: int = None, dtype: str = "float32"
) -> np.ndarray:
    """Transform a list of discrete values to categorical

    Args:
        y (np.ndarray): list of discrete values
        num_classes (int, optional): number of classes
        dtype (str, optional): data type

    Returns:
        np.ndarray: Description
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def reload_module(modulename: str):
    """Reload a module and it's submodules"""
    from importlib import reload
    from sys import modules

    for k, v in modules.items():
        if k.startswith(modulename):
            reload(v)


def zscores(x: pd.Series, mean_dict: dict, std_dict: dict, na=9) -> pd.Series:
    """Convert values to z-scores clipped between -4 and 4.
    Missing values are assigned 9. Expects series name to also be feature
    name corresponding to a key in mean_dict and std_dict.

    Args:
        x (pd.Series): data
        mean_dict (dict): dictionary of mean values
        std_dict (dict): dictionary of standard deviations

    Returns:
        pd.Series: z-scores
    """
    x = 1.0 * (x - mean_dict[x.name]) / std_dict[x.name]
    x = x.round()
    x = x.clip(-4, 4)
    x = x.fillna(na)
    x = x.round(0).astype(int)
    return x


def physiological_words(mat: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Converts numerical values into one-hot vectors of number of z-scores
    above/below the mean.

    Args:
        mat (pd.DataFrame): dataframe of feature values

    Returns:
        pd.DataFrame: a DataFrame where each features is a set of
        indicator columns signifying number of z-scores above or below the mean.
    """
    mean_dict = mat.groupby([group_col]).mean().mean().to_dict()
    std_dict = mat.std().to_dict()
    feature_cols = mat.columns
    print(feature_cols)
    X_words = mat.loc[:, feature_cols].apply(
        lambda x: zscores(x, mean_dict, std_dict), axis=0
    )
    mat.loc[:, feature_cols] = X_words
    X_categorized = pd.get_dummies(mat, columns=mat.columns[len(INDEX_COLS) :])
    na_columns = [col for col in X_categorized.columns if "_9" in col]
    X_categorized.drop(na_columns, axis=1, inplace=True)
    return X_categorized


def wide_to_tall(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide table to a tall table

    Args:
        df (pd.DataFrame): wide table

    Returns:
        pd.DataFrame: tall table
    """
    return df.unstack().dropna().reset_index()


def plausibility_filter(
    df: pd.DataFrame, parameters: list, ranges: dict
) -> pd.DataFrame:
    """Apply plausibility filters. Dataframe is not copied but changed in place.

    Args:
        df (pd.DataFrame): dataframe, wide format
        parameters (list): list of parameters corresponding to column names
        ranges (dict): dictionary indexed by parameter names with MinValue and MaxValue elements

    Returns:
        pd.DataFrame: dataframe
    """
    for c in parameters:
        df.loc[
            (df.loc[:, c] < ranges[c]["MinValue"])
            | (df.loc[:, c] > ranges[c]["MaxValue"]),
            c,
        ] = np.nan
    return df


def chunks(array: np.ndarray, size: int):
    """Yield successive n-sized chunks from l.

    Args:
        array (np.ndarray): multi-dimensional array
        size (int): number of samples per batch

    Returns:
        np.ndarray: batch of samples
    """
    for i in range(0, len(array), size):
        yield array[i : i + size]


def fread(filename):
    with io.open(filename, "r", encoding="utf8") as f:
        return f.read()


def freadb(filename):
    with io.open(filename, "rb") as f:
        return f.read()


def fwrite(filename, html):
    with io.open(filename, "w", encoding="utf8") as f:
        f.write(html)


def fwriteb(filename, html):
    with io.open(filename, "wb") as f:
        f.write(html)


def to_json(filename, data):
    fwrite(
        filename,
        json.dumps(
            data,
            default=lambda o: o.isoformat()
            if isinstance(o, (datetime, date))
            else None,
            indent=2,
        ),
    )


def from_json(filename, default=[]):
    if os.path.exists(filename):
        with open(filename) as f:
            data = json.load(f)
    else:
        data = default
    return data


def astable(table):
    display(HTML(tabulate.tabulate(table, tablefmt="html")))


def roundn(x, prec=2, base=0.05):
    return np.round(base * np.round(x / base), prec)


def key_bytes(key):
    """Convert the given data to a value appropriate for hashing."""
    if isinstance(key, bytes):
        return key
    key = stringify(key) or ""
    return key.encode("utf-8")


def compute_key(record, keys, key_prefix=None):
    """Generate a key for this record, based on the given fields."""
    seed = sha1(key_bytes(key_prefix))
    values = [key_bytes(record.get(k)) for k in keys]
    digest = seed.copy()
    for value in sorted(values):
        digest.update(value)
    if digest.digest() != seed.digest():
        return digest.hexdigest()


def ffill(arr: np.ndarray) -> np.ndarray:
    arr = arr.T
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx].T
    return out