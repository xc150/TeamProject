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