import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier


class FeatureMetrics:
    @staticmethod
    def load_data():
        address = '../dataset/boston_new.csv'
        column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
                        'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                        'PTRATIO', 'B', 'LSTAT', 'MEDV']
        data = pd.read_csv(address, delim_whitespace=True, header=None, names=column_names)
        return data

    def split_data(self, data):
        X = data.drop('MEDV', axis=1)
        prices = data['MEDV']
        y = np.round(prices)
        return X, y

    def feature_score_counter(self, X, y):
        feature_scorer = SelectKBest(score_func=chi2, k=5)
        fit_features = feature_scorer.fit(X, y)
        score_counter = pd.DataFrame(fit_features.scores_)
        columns = pd.DataFrame(X.columns)
        feature_scores = pd.concat([columns, score_counter], axis=1)
        feature_scores.columns = ['specs', 'score']
        return feature_scores

    def feature_importance_counter(self, X, y):
        clf = ExtraTreesClassifier()
        clf.fit(X, y)
        result = clf.feature_importances_
        z = dict(zip(X.columns, result))
        return z