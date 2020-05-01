import datetime
import os

import pandas as pd
import xgboost as xgb
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion, make_pipeline

import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

census_data_filename = './adult.data'

# These are the column labels from the census data files
COLUMNS = (
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income-level'
)


class PositionalSelector(BaseEstimator, TransformerMixin):
    def __init__(self, positions):
        self.positions = positions

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(X)[:, self.positions]


class StripString(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        strip = np.vectorize(str.strip)
        return strip(np.array(X))


class SimpleOneHotEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.values = []
        for c in range(X.shape[1]):
            Y = X[:, c]
            values = {v: i for i, v in enumerate(np.unique(Y))}
            self.values.append(values)
        return self

    def transform(self, X):
        X = np.array(X)
        matrices = []
        for c in range(X.shape[1]):
            Y = X[:, c]
            matrix = np.zeros(shape=(len(Y), len(self.values[c])), dtype=np.int8)
            for i, x in enumerate(Y):
                if x in self.values[c]:
                    matrix[i][self.values[c][x]] = 1
            matrices.append(matrix)
        res = np.concatenate(matrices, axis=1)
        return res


def predict(instances):
    # Load the training census dataset
    with open(census_data_filename, 'r') as train_data:
        raw_training_data = pd.read_csv(train_data, header=None, names=COLUMNS)

        raw_training_data.head()

        raw_features = raw_training_data.drop('income-level', axis=1).values
        # Create training labels list
        train_labels = (raw_training_data['income-level'] == ' >50K').values

        # Categorical features: age and hours-per-week
        # Numerical features: workclass, marital-status, and relationship
        numerical_indices = [0, 12]  # age-num, and hours-per-week
        categorical_indices = [1, 3, 5, 7]  # workclass, education, marital-status, and relationship

        p1 = make_pipeline(PositionalSelector(categorical_indices),
                           StripString(),
                           SimpleOneHotEncoder())
        p2 = make_pipeline(PositionalSelector(numerical_indices),
                           StandardScaler())

        pipeline = FeatureUnion([
            ('numericals', p1),
            ('categoricals', p2),
        ])

        train_features = pipeline.fit_transform(raw_features)

        # train the model
        model = xgb.XGBClassifier(max_depth=4)
        model.fit(train_features, train_labels)

        # save the mode
        model.save_model('model.bst')
        processed_instances = pipeline.transform(instances)
        ans = model.predict(processed_instances)

        return ans


if __name__ == '__main__':
    instances = [
        [
            42, ' State-gov', 77516, ' Bachelors', 13, ' Never-married',
            ' Adm-clerical', ' Not-in-family', ' White', ' Male', 2174, 0, 40,
            ' United-States'
        ],
        [
            50, ' Self-emp-not-inc', 83311, ' Bachelors', 13,
            ' Married-civ-spouse', ' Exec-managerial', ' Husband',
            ' White', ' Male', 0, 0, 10, ' United-States'
        ],
        [
            50, ' Federal-gov', 83311, ' Doctorate', 13,
            ' Married-civ-spouse', ' Exec-managerial', ' Husband',
            ' White', ' Male', 0, 0, 10, ' United-States'
        ]
    ]
    ans = predict(instances)
    print(ans)
