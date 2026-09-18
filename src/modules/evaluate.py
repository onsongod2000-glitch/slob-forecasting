from typing import Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score

from src.config import RunConfig
from src.utils import to_numpy_f32


class ModelEvaluator:

    def __init__(self, config: RunConfig):
        self.config = config

    def evaluate(self, model: xgb.XGBRegressor | xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series)->Dict:
        if self.config.task == 'classification':
            return self._evaluate_classifier(model=model, X=X, y=y)
        return self._evaluate_regressor(model=model, X=X, y=y)

    @staticmethod
    def _evaluate_classifier(model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series, threshold = 0.5)->Dict:
        X_np = to_numpy_f32(X=X)
        proba = model.predict_proba(X_np)[:, 1]
        preds = (proba >threshold).astype(int)

        metrics = {
            'accuracy': float(accuracy_score(y_true=y, y_pred=preds)),
            'precision': float(precision_score(y_true=y, y_pred=preds)),
            'recall': float(recall_score(y_true=y, y_pred=preds, zero_division=0)),
            'f1': float(f1_score(y_true=y, y_pred=preds, zero_division=0))
        }

        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true=y, y_score=proba))
        except ValueError:
            metrics['roc_auc'] = float('nan')

        metrics['confusion_matrix'] = confusion_matrix(y_true=y, y_pred=preds).tolist()

        return {
            'metrics': metrics,
            'preds': preds,
            'proba': proba
        }

    @staticmethod
    def _evaluate_regressor(model: xgb.XGBRegressor, X: pd.DataFrame, y: pd.Series)->Dict:
        X_np = to_numpy_f32(X=X)
        preds = model.predict(X_np)

        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_true=y, y_pred=preds))),
            'mae': float(mean_absolute_error(y_true=y, y_pred=preds)),
            'r2': float(r2_score(y_true=y, y_pred=preds))
        }

        return {
            'metrics': metrics,
            'preds': preds,
            'proba': None
        }

