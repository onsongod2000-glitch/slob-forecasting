from typing import Dict, Optional

import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error

from src.config import RunConfig
from src.utils import to_numpy_f32
from src.modules.model_factory import build_model, fit_model, suggest_params


class HyperParameterTuner:
    def __init__(self, config: RunConfig):
        self.config = config
        self.study_: Optional[optuna.Study] = None


    def tune(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series)->Dict:
        config = self.config

        def objective(trial: optuna.Trial)->float:
            params = suggest_params(trial=trial, config=config)
            model =  build_model(config=config, params=params)
            model = fit_model(model=model, config=config, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
            y_pred = model.predict(to_numpy_f32(X_val))

            if config.task == 'classification':
                return f1_score(y_true=y_val, y_pred=y_pred)

            return -np.sqrt(mean_squared_error(y_true=y_val, y_pred=y_pred))

        sampler = optuna.samplers.TPESampler(seed=config.random_state)
        self.study_ = optuna.create_study(direction='maximize', sampler=sampler)
        self.study_.optimize(objective, n_trials=config.n_optuna_trials, show_progress_bar=False)
        return self.study_.best_params