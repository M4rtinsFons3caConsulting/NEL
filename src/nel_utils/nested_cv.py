import os

def run(
    X, 
    y, 
    cv_outer, 
    cv_inner, 
    fixed_params, 
    param_grid, 
    seed, 
    LOG_DIR, 
    DATASET_NAME, 
    call_slim
):
    
    """
    Run nested cross-validation with outer and inner loops.
    
    Parameters:
        X, y: Features and target arrays.
        cv_outer: Outer CV splitter object (with split method).
        cv_inner: Inner CV splitter object.
        fixed_params: dict, fixed parameters for the model call.
        param_grid: dict, grid of hyperparameters to search.
        seed: int, base seed for reproducibility.
        LOG_DIR: str, directory path to store logs.
        DATASET_NAME: str, dataset name for log files.
        call_slim: function, model training function accepting fixed_params and param_grid.
        
    Returns:
        results: list of results from inner folds.
    """
    
    # Get first outer fold indices
    data_cv_outer = [[learning_ix, test_ix] for learning_ix, test_ix in cv_outer.split(X, y)][0]
    learning_ix, test_ix = data_cv_outer
    
    X_learning, y_learning = X[learning_ix], y[learning_ix]
    X_test, y_test = X[test_ix], y[test_ix]
    
    print('\n' + '-'*41 + '\n')
    print(f'Outer CV\nLearning shape: {X_learning.shape}\nTest shape: {X_test.shape}\n')
    
    results = []
    data_cv_inner = [[train_ix, val_ix] for train_ix, val_ix in cv_inner.split(X_learning, y_learning)]
    
    for i_inner, (train_ix, val_ix) in enumerate(data_cv_inner):
        print('-----\nInner CV {}'.format(i_inner))
        
        X_train, y_train = X_learning[train_ix], y_learning[train_ix]
        X_val, y_val = X_learning[val_ix], y_learning[val_ix]
        
        print(f'Training shape: {X_train.shape}\nValidation shape: {X_val.shape}\n')
        
        # Update fixed params for this fold
        fixed_params.update({
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_val, 'y_test': y_val
        })
        
        LOG_PATH = os.path.join(LOG_DIR, f'slim_{DATASET_NAME}_{i_inner}.csv')
        if os.path.exists(LOG_PATH):
            os.remove(LOG_PATH)
        fixed_params['log_path'] = LOG_PATH
        
        res = call_slim(fixed_params, param_grid, seed=(seed + i_inner))
        results.append(res)
        
    return results
