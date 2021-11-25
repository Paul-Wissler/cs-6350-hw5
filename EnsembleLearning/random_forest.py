import pandas as pd
import numpy as np

import DecisionTree as dtree
from DecisionTree import calc_gain, calc_entropy
from .bagger import BaggerModel


class RandomForestModel(BaggerModel):

    def __init__(self, X: pd.DataFrame, y: pd.Series, sample_rate=None, bag_rounds=100,
            error_f=calc_entropy, max_tree_depth=None, num_sample_attributes=2,
            default_value_selection='subset_majority', reproducible_seed=True):
        self.X = X.copy()
        self.y = y.copy()
        self.numeric_cols = self.determine_numeric_cols()
        self.median = self.calc_median()
        self.default_value_selection = default_value_selection
        self.error_f = error_f
        self.max_tree_depth = max_tree_depth
        self.num_sample_attributes = num_sample_attributes
        self.model = self.create_random_forest(
            self.convert_numeric_vals_to_categorical(X.copy()), y, 
            sample_rate=sample_rate, bag_rounds=bag_rounds, error_f=error_f, 
            max_tree_depth=self.max_tree_depth, reproducible_seed=reproducible_seed,
        )
        del self.X
        del self.y

    def create_random_forest(self, X: pd.DataFrame, y: pd.Series, sample_rate=None, 
            bag_rounds=100, error_f=calc_entropy, max_tree_depth=None, 
            reproducible_seed=True) -> list:
        if not sample_rate:
            sample_rate = self.auto_calc_subset(len(X))
        bag = list()
        for t in range(bag_rounds):
            if reproducible_seed:
                A = pd.Series(X.columns).sample(n=self.num_sample_attributes, random_state=t * 100000)
                X_s = X[A].sample(n=sample_rate, replace=True, random_state=t * 100000)
            else:
                A = pd.Series(X.columns).sample(n=self.num_sample_attributes)
                X_s = X[A].sample(n=sample_rate, replace=True)
            y_s = y.iloc[X_s.index]
            bag.append(dtree.DecisionTreeModel(X_s, y_s))
            # print(f'round: {t}')
        return bag
