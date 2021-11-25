import pandas as pd
import numpy as np
from multiprocessing import Pool

import DecisionTree as dtree
from DecisionTree import calc_gain, calc_entropy

class BaggerModel:

    def __init__(self, X: pd.DataFrame, y: pd.Series, sample_rate=None, 
            bag_rounds=100, error_f=calc_entropy, max_tree_depth=None, 
            default_value_selection='subset_majority', reproducible_seed=True):
        self.X = X.copy()
        self.y = y.copy()
        self.numeric_cols = self.determine_numeric_cols()
        self.median = self.calc_median()
        self.default_value_selection = default_value_selection
        self.error_f = error_f
        self.max_tree_depth = max_tree_depth
        self.model = self.create_bagger(
            self.convert_numeric_vals_to_categorical(X.copy()), y, 
            sample_rate=sample_rate, bag_rounds=bag_rounds, error_f=error_f, 
            max_tree_depth=self.max_tree_depth, reproducible_seed=reproducible_seed
        )
        del self.X
        del self.y

    def convert_numeric_vals_to_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.numeric_cols:
            return X
        for col, m in self.median.iteritems():
            is_gte_m = X[col] >= m
            X[col].loc[is_gte_m] = f'>={m}'
            X[col].loc[~is_gte_m] = f'<{m}'
        return X

    def determine_numeric_cols(self) -> list:
        return self.X.select_dtypes(include=np.number).columns.tolist()

    def calc_median(self) -> pd.Series:
        return self.X[self.numeric_cols].median()

    def create_bagger(self, X: pd.DataFrame, y: pd.Series, sample_rate=None, 
            bag_rounds=100, error_f=calc_entropy, max_tree_depth=None, 
            reproducible_seed=True) -> list:
        if not sample_rate:
            sample_rate = self.auto_calc_subset(len(X))

        bag = list()
        for t in range(bag_rounds):
            if reproducible_seed:
                X_s = X.sample(n=sample_rate, replace=True, random_state=t * 100000)
            else:
                X_s = X.sample(n=sample_rate, replace=True)
            y_s = y.iloc[X_s.index]
            bag.append(dtree.DecisionTreeModel(X_s, y_s))
            print(f'round: {t}')
        return bag

    @staticmethod
    def auto_calc_subset(i):
        return round(.2 * i)

    def test(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        X_test = self.convert_numeric_vals_to_categorical(X_test)
        predict_y = self.evaluate(X_test)
        s = y_test == predict_y
        return s.sum() / s.count()

    def test_cumulative_trees(self, X_test: pd.DataFrame, y_test: pd.Series, 
            ix: list) -> float:
        X_test = self.convert_numeric_vals_to_categorical(X_test.copy())
        test_results = pd.Series()
        for i in ix:
            # print(f'test: {i}')
            predict_y = self.evaluate_specific_trees(X_test.copy(), self.model[:i+1])
            s = y_test == predict_y
            test_results = test_results.append(
                pd.Series([s.sum() / s.count()], index=[i+1])
            )
        return test_results

    def evaluate(self, X_test: pd.DataFrame) -> float:
        i = 0
        eval_df = pd.DataFrame()
        for tree in self.model:
            # print(f'evaluating tree: {i}')
            eval_df[i] = tree.evaluate(X_test)
            i += 1
        h = eval_df.mode(axis=1)[0]
        return h

    def evaluate_specific_trees(self, X_test: pd.DataFrame, model: list) -> float:
        i = 0
        eval_df = pd.DataFrame()
        for tree in model:
            eval_df[i] = tree.evaluate(X_test)
            i += 1
        h = eval_df.mode(axis=1)[0]
        return h
