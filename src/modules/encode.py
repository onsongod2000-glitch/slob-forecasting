from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.config import RunConfig


class CategoricalEncoder:
    """
    Ordinal-encodes categorical columns, 
    fit on TRAIN split only. Unseen categories 
    (at val / test) time map to -1
    """

    def __init__(self, config: RunConfig):
        self.config = config
        self.encoders_: Dict[str, Dict[str, int]] = {}
        self._is_fit = False

    def fit(self, X_train: pd.DataFrame)->CategoricalEncoder:
        for feature in self.config.categorical_features:
            categories = sorted(X_train[feature].astype(str).unique())
            self.encoders_[feature] = {category: i for i, category in enumerate(categories)}

        self._is_fit = True
        return self

    def transform(self, X: pd.DataFrame)->pd.DataFrame:
        if not self._is_fit:
            raise RuntimeError(f"{CategoricalEncoder.__name__} must be fit before transform().")

        out = X.copy()

        for feature in self.config.categorical_features:
            out[feature] = out[feature].astype(str).map(self.encoders_[feature]).fillna(-1).astype(np.float32)

        for col in [feature for feature in out.columns if feature not in self.config.categorical_features]:
            out[col] = out[col].astype(np.float32)

        return out

    def fit_transform(self, X_train: pd.DataFrame)->pd.DataFrame:
        return self.fit(X_train=X_train).transform(X=X_train)

    def to_dict(self):
        return self.encoders_