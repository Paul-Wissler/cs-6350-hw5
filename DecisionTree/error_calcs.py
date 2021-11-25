import math
import numpy as np
import pandas as pd


def calc_entropy(set_array: pd.Series, unique_outcomes=2) -> float:
    h = list()
    for p in set_array:
        msg = f'Probability was {p}. Probabilities may not exceed 1.'
        # if p > 1:
        #     raise ValueError(msg)
        # elif p == 0:
        if p == 0:
            h.append(0)
        else:
            h.append(-p * logn(p, 2))
    return sum(h)


def calc_gini_index(set_array: pd.Series, unique_outcomes=2) -> float:
    h = list()
    for p in set_array:
        msg = f'Probability was {p}. Probabilities may not exceed 1.'
        if p > 1:
            raise ValueError(msg)
        else:
            h.append(-p**2)
    return 1 + sum(h)


def calc_majority_error(set_array: pd.Series, unique_outcomes=2) -> float:
    if len(set_array) == 1:
        return 0
    return 1 - set_array.max()


def calc_gain(x: pd.Series, y: pd.Series, f=calc_entropy) -> float:
    if isinstance(x, np.ndarray):
        x = pd.Series(x)

    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    is_y_nan = y.isna()
    x = x[~is_y_nan]
    y = y[~is_y_nan]

    H_y = f(calc_discrete_probability(y), 
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
        prob_y_v = calc_discrete_probability(y_v, 
            null_x_v=null_y_v, null_frac=frac_of_x)
        e.append(s_v / s * f(prob_y_v, 
            unique_outcomes=len(np.unique(y))))
    return H_y - sum(e)


def calc_discrete_probability(x: np.array, null_x_v=pd.Series([]), 
        null_frac=0) -> float:
    unique_vals = np.unique(x)
    null_size = len(null_x_v) * null_frac
    p = pd.Series([])
    for i in unique_vals:
        num = sum(x==i) + null_frac * sum(null_x_v==i)
        p[i] = num / (len(x) + null_size)
    # print('test', p)
    return p


def logn(x, n) -> float:
    return math.log(x) / math.log(n)
