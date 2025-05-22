import pandas as pd
from sklearn.impute import KNNImputer

def knn_impute(df, n_neighbors=5):
    """
    Impute missing values in a DataFrame using KNN regression.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with missing values.
    n_neighbors : int
        Number of neighbors to use for imputation.

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed values.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(df)
    return pd.DataFrame(imputed_array, columns=df.columns, index=df.index)
