"""
Defines model and features evaluation functions.
"""
import numpy as np
from sklearn.model_selection import KFold

def cross_validate(model, x, y, n_splits):
    """
    Evaluates a model's performances through cross-validation.
    :param model: Model to evaluate. Should implement model.fit(x, y) and
        model.score(x, y).
    :param x: input data of shape (N_samples, N_features)
    :param y: targets of shape (N_samples,) as zeros or ones.
    :param n_splits: Number of cross-validation folds.
    :return: (train_acc, std_train_acc), (test_acc, std_test_acc)
    """
    folds = KFold(n_splits=n_splits, shuffle=False)
    accs = []
    train_accs = []
    for fold_indx, (train_index, test_index) in enumerate(folds.split(x, y)):
        print(f'Cross-validating on subset {fold_indx}...      ', end='\r')
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]

        model.fit(x_train, y_train)
        accs.append(model.score(x_test, y_test))
        train_accs.append(model.score(x_train, y_train))

    return (np.mean(train_accs), np.std(train_accs)), (np.mean(accs), np.std(accs))
