import numpy as np
import pandas as pd


def describe(df, feature_cols):
    stats_df = {}
    for f in feature_cols:
        x = df.loc[:,f].dropna()
        stats = dict(map(lambda i: [f'p{i}',np.percentile(x, i)], [1,5,10,25,50,75,90,95,99]))
        inds = (x >= stats.get('p1')) & (x <= stats.get('p99'))
        x = x[inds]
        stats['mean'] = np.mean(x)
        stats['min'] = np.min(x)
        stats['max'] = np.max(x)
        stats['nunique'] = len(np.unique(x))
        stats_df[f] = stats
    stats_df = pd.DataFrame(stats_df).T
    return stats_df