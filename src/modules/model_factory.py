from typing import Dict, Union

import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.config import RunConfig
from src.utils import to_numpy_f32

Model = Union[xgb.XGBClassifier, xgb.XGBRegressor, RandomForestClassifier, RandomForestRegressor]

def build_model(config: RunConfig, params: Dict)->Model:
    is_classification = config.task == 'classification'
    if config.model_family == 'xgboost':
        if is_classification:
            return xgb.XGBClassifier(**params, objective='binary:logistic', eval_metric='logloss')
        return xgb.XGBRegressor(**params, objective='reg:squarederror', eval_metric='rmse')

    if is_classification:
        return RandomForestClassifier(**params)
    return RandomForestRegressor(**params)

def fit_model(model: Model, config: RunConfig, X_train: pd.DataFrame, y_train:pd.Series, X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None)->Model:
    X_train_ = to_numpy_f32(X=X_train)
    if config.model_family == 'xgboost' and (X_val is not None and y_val is not None):
        X_val_ =to_numpy_f32(X=X_val)
        model.fit(X_train_, y_train, eval_set=[(X_val_, y_val)], verbose = False)
    else:
        model.fit(X_train_, y_train)
    return model

def suggest_params(trial: optuna.Trial, config: RunConfig) -> Dict:

    if config.model_family == 'xgboost':
        return {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'random_state': config.random_state,
            'n_jobs': -1,
        }
    
    # random_forest
    
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 4, 24),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.3, 1.0),
        'random_state': config.random_state,
        'n_jobs': -1,
    }
