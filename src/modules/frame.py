from typing import Tuple

import pandas as pd

from src.config import RunConfig


Frame = Tuple[pd.DataFrame, pd.Series, pd.DataFrame]


class SupervisedLearningFrame:
    """
    Turn the leak-safe features table into an `(X, y, meta)`
    supervised learning frame for a given run config,
    applying the horizon shift.
    """

    def __init__(self, config: RunConfig):
        self.config = config

    def build(self, df: pd.DataFrame)->Frame:
        config = self.config
        work = df.copy()

        future_col = f"{config.target_col}_future"

        work[future_col] = work.groupby("SKU", observed=True)[config.target_col].shift(
            -config.horizon_weeks
        )

        keep_cols = ["SKU", "Week"] + config.feature_cols + [future_col]
        frame = work[keep_cols].dropna(subset=config.feature_cols + [future_col]).copy()
        meta = frame[['SKU', 'Week']].reset_index(drop=True)
        y = frame[future_col].reset_index(drop=True)
        X = frame[config.feature_cols].reset_index(drop=True)

        if config.task == 'classification':
            y = y.astype(int)

        return X, y, meta