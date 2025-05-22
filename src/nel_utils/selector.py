# nel_utils/selector.py

import gmpy2
import pandas as pd

def gmpy2_nsmallest(df, column, n):

    sizes = [
        (idx, gmpy2.mpz(val)) for idx, val in df[column].items()
    ]
    sizes_sorted = sorted(sizes, key=lambda x: x[1])
    top_n_indices = [idx for idx, _ in sizes_sorted[:n]]

    return df.loc[top_n_indices]

def nsmallest(df, column, n):
    return df.nsmallest(n, column)

MODEL_DICT = {
    'gp': nsmallest,
    'gsgp': gmpy2_nsmallest
}

def select_top_n(
    df_log, 
    model, 
    k=10, 
    n=3
):

    df = pd.DataFrame({
        'cv': df_log['cv'],
        'rmse_train': df_log.iloc[:, 4],
        'rmse_test': df_log.iloc[:, 5],
        'size': df_log.iloc[:, 9]
    })

    df['overfit_ratio'] = df['rmse_test'] / df['rmse_train']

    medians = df.groupby('cv')['rmse_test'].median()
    best_cv = medians.nsmallest(k).index.tolist()
    filtered = df[df['cv'].isin(best_cv)]

    top_k = filtered.nsmallest(k, 'overfit_ratio')
    top_n = MODEL_DICT[model](top_k, 'size', n)

    return sorted(top_n.index.tolist())
