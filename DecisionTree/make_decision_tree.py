import pandas as pd
import numpy as np

from .error_calcs import calc_gain, calc_entropy


class DecisionTreeModel:

    def __init__(self, X: pd.DataFrame, y: pd.Series, error_f=calc_entropy, 
            max_tree_depth=None, default_value_selection='subset_majority'):
        self.X = X.copy()
        self.y = y.copy()
        self.y_mode = y.mode().iloc[0]
        self.numeric_cols = self.determine_numeric_cols()
        self.median = self.calc_median()
        self.default_value_selection = default_value_selection
        self.error_f = error_f
        self.input_max_tree_depth = max_tree_depth
        self.tree = self.make_decision_tree(
            self.convert_numeric_vals_to_categorical(X.copy()), y, 
            error_f=error_f, max_tree_depth=self.calc_max_tree_depth()
        )
        del self.X
    
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

    def calc_max_tree_depth(self):
        max_len = len(self.X.columns)
        if not self.input_max_tree_depth:
            return max_len
        elif self.input_max_tree_depth > max_len:
            print(f'WARNING: input tree depth of {self.input_max_tree_depth} exceeds maximum possible length, {max_len}.')
            return max_len
        return self.input_max_tree_depth

    def default_value(self, y):
        if self.default_value_selection == 'majority':
            return y.groupby(y).count().idxmax()
        elif self.default_value_selection == 'subset_majority':
            return y.groupby(y).count().idxmax()

    def make_decision_tree(self, X: pd.DataFrame, y: pd.Series, 
            error_f, max_tree_depth=None) -> dict:
        split_node = self.determine_split(X, y, error_f)
        d = {split_node: dict()}
        for v in X[split_node].unique():
            X_v_cols = X.columns[X.columns != split_node]
            X_v = X[X_v_cols].loc[X[split_node] == v]
            y_v = y.loc[X[split_node] == v]
            if len(y_v.unique()) == 1:
                d[split_node][v] = y_v.unique()[0]
            elif max_tree_depth == 1:
                d[split_node][v] = self.default_value(y_v)
            else:
                d[split_node][v] = self.make_decision_tree(X_v, y_v, 
                    error_f, max_tree_depth - 1)
        return d
    
    @staticmethod
    def determine_split(X: pd.DataFrame, y: pd.Series, error_f) -> str:
        current_gain = 0
        split_node = X.columns[0]
        for col in X.columns:
            new_gain = calc_gain(X[col].values, y.values, f=error_f)
            if new_gain > current_gain:
                split_node = col + ''
                current_gain = new_gain + 0
        return split_node

    def test(self, test_X: pd.DataFrame, test_y: pd.Series) -> float:
        test_X = self.convert_numeric_vals_to_categorical(test_X)
        predict_y = self.evaluate(test_X)
        s = test_y == predict_y
        return s.sum() / s.count()

    def evaluate(self, test_X: pd.DataFrame) -> pd.Series:
        return test_X.apply(self.check_tree, axis=1, args=[self.tree])

    def check_tree(self, row: pd.Series, tree: dict):
        node = list(tree.keys())[0]
        try:
            if isinstance(tree[node][row[node]], str):
                return tree[node][row[node]]
            elif isinstance(tree[node], dict):
                return self.check_tree(row, tree[node][row[node]])
            else:
                return self.y_mode
        except KeyError:
            return self.y_mode
