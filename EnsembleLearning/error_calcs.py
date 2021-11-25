import numpy as np
import pandas as pd

import DecisionTree as dtree


def calc_weighted_gain(x: pd.Series, y: pd.Series, w: pd.Series, 
        f=dtree.calc_entropy) -> float:
    if isinstance(x, np.ndarray) | isinstance(x, list):
        x = pd.Series(x)

    if isinstance(y, np.ndarray) | isinstance(y, list):
        y = pd.Series(y)

    if isinstance(w, np.ndarray) | isinstance(w, list):
        w = pd.Series(w)

    is_y_nan = y.isna()
    x = x[~is_y_nan]
    y = y[~is_y_nan]

    H_y = f(calc_weighted_discrete_probability(y, w), 
        unique_outcomes=len(np.unique(y)))
    s = len(y)
    e = list()

    x_no_nans = x[~x.isna()]
    count_x_no_nans = len(x_no_nans)
    for v in np.unique(x_no_nans):
        is_x_eq_v = np.where(x==v)
        frac_of_x = len(x.loc[is_x_eq_v]) / count_x_no_nans
        null_x_v = len(x[x.isna()]) * frac_of_x

        null_y_v = y.loc[x.isna()]

        s_v = len(x.loc[is_x_eq_v]) + null_x_v
        y_v = y.loc[is_x_eq_v]
        w_v = w.loc[is_x_eq_v]
        prob_y_v = calc_weighted_discrete_probability(y_v, w_v,
            null_x_v=null_y_v, null_frac=frac_of_x)
        e.append(s_v / s * f(prob_y_v, 
            unique_outcomes=len(np.unique(y))))
    # print('\n\n')
    # print(H_y)
    # print(sum(e))
    # print(f'{x.name}: {H_y - np.sum(e)}')
    return H_y - np.sum(e)


def calc_weighted_discrete_probability(x: np.array, w: list, 
        null_x_v=pd.Series([]), null_frac=0) -> float:
    w = np.array(w)
    unique_vals = np.unique(x)
    null_size = len(null_x_v) * null_frac # Ignore for now
    p = pd.Series([])
    w = np.divide(w, np.sum(w))
    for i in unique_vals:
        num = np.sum(np.multiply(1 * (x==i), w)) + null_frac * np.sum(null_x_v==i)
        p[i] = num
    return p
