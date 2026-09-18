from typing import Literal, List
from dataclasses import dataclass

from src.constants import ABC_FEATURE, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_CLASSIFICATION, TARGET_DEMAND, TARGET_REGRESSION

@dataclass(frozen=True)
class RunConfig:
    task: Literal['classification', 'regression', 'demand'] = 'classification'
    model_family: Literal['xgboost', 'randomforest'] = 'xgboost'
    max_train_rows:int | None =None
    include_abc: bool = False
    horizon_weeks: int = 4
    train_split: float = 0.70
    val_split: float = 0.15
    n_optuna_trials: int = 30
    random_state: int = 42

    @property
    def run_name(self)->str:
        abc_tag = "with_abc" if self.include_abc else "no_abc"
        family_tag = "" if self.model_family == 'xgboost' else f'_{self.model_family}'
        return f"{self.task}{family_tag}_{abc_tag}_h{self.horizon_weeks}"

    @property
    def target_col(self)->str:
        return {
            'classification': TARGET_CLASSIFICATION,
            'regression': TARGET_REGRESSION,
            'demand': TARGET_DEMAND,
        }[self.task]

    @property
    def feature_cols(self)->List[str]:
        categorical_cols = CATEGORICAL_FEATURES + ([ABC_FEATURE] if self.include_abc else [])
        return categorical_cols + NUMERICAL_FEATURES

    @property
    def categorical_features(self)->List[str]:
        return CATEGORICAL_FEATURES + ([ABC_FEATURE] if self.include_abc else [])

