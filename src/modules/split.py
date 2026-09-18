from typing import Dict

import numpy as np
import pandas as pd

from src.config import RunConfig
from src.modules.frame import Frame


class TemporalSplitter:
    """
    Splits an (X, y, meta) frame chronologically by
    calendar week. Appropriate for longitudinal panel.
    """

    def __init__(self, config: RunConfig):
        self.config = config


    def split(self, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame)->Dict[str, Frame]:
        config = self.config
        weeks_sorted = np.sort(meta['Week'].unique())
        n_weeks = len(weeks_sorted)
        train_split = weeks_sorted[int(n_weeks * config.train_split)]
        val_split = weeks_sorted[int(n_weeks * (config.train_split + config.val_split))]

        masks = {
            'train': meta["Week"] < train_split,
            'val': (meta["Week"] >= train_split) & (meta["Week"] < val_split),
            'test': meta["Week"] >=val_split,
        }

        return {
            name: (
                X[mask].reset_index(drop=True),
                y[mask].reset_index(drop=True),
                meta[mask].reset_index(drop=True),
            )
            for name, mask in masks.items()
        }