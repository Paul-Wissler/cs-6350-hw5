import math
import pandas as pd
import numpy as np

import DecisionTree as dtree
from .error_calcs import calc_weighted_gain


class AdaBoostModel:

    def __init__(self, X: pd.DataFrame, y: pd.Series, sample_rate=None, 
            boost_rounds=100, error_f=dtree.calc_entropy, max_tree_depth=2, 
            default_value_selection='subset_majority', reproducible_seed=True):
        self.y_mode = y.mode().iloc[0]
        self.X = X.copy()
        self.y = self.binarize_data(y.copy())
        self.numeric_cols = self.determine_numeric_cols()
        self.median = self.calc_median()
        self.default_value_selection = default_value_selection
        self.error_f = error_f
        self.max_tree_depth = max_tree_depth
        self.trees, self.votes = self.create_booster(
            self.convert_numeric_vals_to_categorical(X.copy()), 
            self.binarize_data(y.copy()), sample_rate=sample_rate, 
            boost_rounds=boost_rounds, error_f=error_f, 
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

    def binarize_data(self, y):
        is_mode = y == self.y_mode
        y.loc[is_mode] = '1'
        y.loc[~is_mode] = '-1'
        return y

    def create_booster(self, X: pd.DataFrame, y: pd.Series, sample_rate=None, 
            boost_rounds=100, error_f=dtree.calc_entropy, max_tree_depth=None, 
            reproducible_seed=True) -> (list, list):
        boosted_model = list()
        current_weights = [1 / len(X)] * len(X)
        weights = list()
        error = 0
        votes = list()
        for t in range(boost_rounds):
            print(t)
            tree = AdaBoostDecisionTreeModel(X, y, current_weights,
                max_tree_depth=self.max_tree_depth, error_f=error_f)
            temp_error = error + 0
            error = 1 - tree.test(X, y, pd.Series(current_weights))
            vote = self.calc_vote(error)
            votes.append(vote)
            boosted_model.append(tree)
            temp_weights = list()
            for i in range(len(y)):
                y_x_h = int(y[i]) * int(tree.check_tree(X.iloc[i], tree.tree))
                temp_weights.append(current_weights[i] * math.exp(-vote * y_x_h))
            current_weights = [weight / sum(temp_weights) for weight in temp_weights]
        return boosted_model, votes

    @staticmethod
    def calc_vote(error):
        return 0.5 * math.log((1 - error) / error)

    def test(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        X_test = self.convert_numeric_vals_to_categorical(X_test)
        predict_y = self.evaluate(X_test)
        s = self.binarize_data(y_test) == predict_y
        return s.sum() / s.count()

    def test_cumulative_trees(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        X_test = self.convert_numeric_vals_to_categorical(X_test.copy())
        test_results = pd.Series()
        cum_test_results = pd.Series()
        y_test = self.binarize_data(y_test)
        for i in range(len(self.trees)):
            predict_y = self.trees[i].evaluate(X_test.copy())
            s = y_test == predict_y
            test_results = test_results.append(
                pd.Series([s.sum() / s.count()], index=[i+1])
            )

            cum_predict_y = self.evaluate_specific_trees(X_test.copy(), self.trees[:i+1], 
                self.votes[:i+1])
            s = y_test == cum_predict_y
            cum_test_results = cum_test_results.append(
                pd.Series([s.sum() / s.count()], index=[i+1])
            )
        return test_results, cum_test_results

    def evaluate(self, X_test: pd.DataFrame) -> float:
        i = 0
        eval_df = pd.DataFrame()
        for tree, vote in zip(self.trees, self.votes):
            eval_df[i] = tree.evaluate(X_test).astype(int) * vote
            i += 1
        h = pd.DataFrame()
        h['vals'] = eval_df.mean(axis=1)
        is_gte_0 = h.vals >= 0
        h.vals.loc[is_gte_0] = '1'
        h.vals.loc[~is_gte_0] = '-1'
        return h.vals

    def evaluate_specific_trees(self, X_test: pd.DataFrame, trees: list, votes: list
            ) -> float:
        i = 0
        eval_df = pd.DataFrame()
        for tree, vote in zip(trees, votes):
            eval_df[i] = tree.evaluate(X_test).astype(int) * vote
            i += 1
        h = pd.DataFrame()
        h['vals'] = eval_df.mean(axis=1)
        is_gte_0 = h.vals >= 0
        h.vals.loc[is_gte_0] = '1'
        h.vals.loc[~is_gte_0] = '-1'
        return h.vals


class AdaBoostDecisionTreeModel(dtree.DecisionTreeModel):

    def __init__(self, X: pd.DataFrame, y: pd.Series, weights: list, 
            error_f=dtree.calc_entropy, max_tree_depth=None, 
            default_value_selection='subset_majority'):
        self.X = X.copy()
        self.y = y.copy()
        self.numeric_cols = self.determine_numeric_cols()
        self.median = self.calc_median()
        self.default_value_selection = default_value_selection
        self.error_f = error_f
        self.input_max_tree_depth = max_tree_depth
        self.tree = self.make_decision_tree(
            self.convert_numeric_vals_to_categorical(X.copy()), y.copy(), weights, 
            error_f=error_f, max_tree_depth=self.calc_max_tree_depth()
        )
        del self.X
        del self.y

    def make_decision_tree(self, X: pd.DataFrame, y: pd.Series, w: list,
            error_f=dtree.calc_entropy, max_tree_depth=None) -> dict:
        split_node = self.determine_split(X, y, w, error_f)
        d = {split_node: dict()}
        w = pd.Series(w)
        for v in X[split_node].unique():
            X_v_cols = X.columns[X.columns != split_node]
            X_v = X[X_v_cols].loc[X[split_node] == v]
            y_v = y.loc[X[split_node] == v]
            w_v = w.loc[X[split_node] == v]
            if len(y_v.unique()) == 1:
                d[split_node][v] = y_v.unique()[0]
            elif max_tree_depth == 1:
                d[split_node][v] = self.most_likely_value(y_v, w_v)
            else:
                d[split_node][v] = self.make_decision_tree(X_v, y_v, list(w.values), 
                    error_f, max_tree_depth - 1)
        return d

    @staticmethod
    def determine_split(X: pd.DataFrame, y: pd.Series, w: list, error_f) -> str:
        current_gain = 0
        split_node = X.columns[0]
        for col in X.columns:
            new_gain = calc_weighted_gain(X[col].values, y.values, w, f=error_f)
            if new_gain > current_gain:
                split_node = col + ''
                current_gain = new_gain + 0
        return split_node

    def most_likely_value(self, y, w):
        w = np.array(w)
        check = pd.Series()
        for i in np.unique(y):
            ix = np.where(y==i)
            check[i] = np.sum(w[ix])
        return check.loc[check==check.max()].index[0]

    def test(self, test_X: pd.DataFrame, test_y: pd.Series, w: pd.Series) -> float:
        test_X = self.convert_numeric_vals_to_categorical(test_X)
        predict_y = self.evaluate(test_X)
        w = np.array(w)
        s = np.multiply(1 * (test_y == predict_y), w)
        return s.sum()
