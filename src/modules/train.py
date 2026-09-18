from typing import Dict

import pandas as pd

from src.config import RunConfig
from src.modules.model_factory import build_model, fit_model

class ModelTrainer:
    def __init__(self, config: RunConfig):
        self.config = config
 
    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series, 
        best_params: Dict,
    ):
        config = self.config
        params = {**best_params, "random_state": config.random_state, "n_jobs": -1}
        model = build_model(config=config, params=params)
        model = fit_model(model=model, config=config, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
        return model
