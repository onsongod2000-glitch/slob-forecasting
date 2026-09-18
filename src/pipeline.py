import json
import joblib
from typing import Dict
from pathlib import Path

import optuna
import pandas as pd
import xgboost as xgb

from src.config import RunConfig
from src.modules.artifacts import Artifacts
from src.modules.encode import CategoricalEncoder
from src.modules.evaluate import ModelEvaluator
from src.modules.frame import SupervisedLearningFrame
from src.modules.hyperparameters import HyperParameterTuner
from src.modules.split import TemporalSplitter
from src.modules.train import ModelTrainer

optuna.logging.set_verbosity(optuna.logging.WARNING)

class SlobPipeline:

    def __init__(self, config: RunConfig, features_path: Path | None = None):
        self.config = config
        self.study_: optuna.Study | None = None
        self.artifacts = Artifacts(config=config)
        self.trainer = ModelTrainer(config=config)
        self.evaluator = ModelEvaluator(config=config)
        self.tuner = HyperParameterTuner(config=config)
        self.splitter = TemporalSplitter(config=config)
        self.encoder = CategoricalEncoder(config=config)
        self.model_:xgb.XGBClassifier | xgb.XGBRegressor | None = None
        self.supervised_learning_frame = SupervisedLearningFrame(config=config)
        self.features_path = features_path or self.artifacts.data_dir / 'slob_features.parquet'

    
    def load_feature_table(self)->pd.DataFrame:
        df = pd.read_parquet(self.features_path)
        return df.sort_values(['SKU', 'Week']).reset_index(drop=True)

    def run(self, df: pd.DataFrame | None = None, verbose: bool = True)->Dict:
        config = self.config
        if df is None: df = self.load_feature_table()
        X, y, meta = self.supervised_learning_frame.build(df)
        splits = self.splitter.split(X=X, y=y, meta=meta)
        (X_train, y_train, _) = splits['train']
        (X_val, y_val, _) = splits['val']
        (X_test, y_test, meta_test) = splits['test']

        X_train_encoded = self.encoder.fit_transform(X_train)
        X_val_encoded = self.encoder.transform(X_val)
        X_test_encoded = self.encoder.transform(X_test)

        if verbose:
            print(
                f"[{config.run_name}] train={len(X_train_encoded)} val={len(X_val_encoded)} "
                f"test={len(X_test_encoded)} features={X_train_encoded.shape[1]}"
            )

        best_params = self.tuner.tune(X_train=X_train_encoded, y_train=y_train, X_val=X_val_encoded, y_val=y_val)
        self.study_ = self.tuner.study_
        self.model_ = self.trainer.fit(X_train=X_train_encoded, y_train=y_train, X_val=X_val_encoded, y_val=y_val, best_params=best_params)

        val_eval = self.evaluator.evaluate(model=self.model_, X=X_val_encoded, y=y_val)
        test_eval = self.evaluator.evaluate(model=self.model_, X=X_test_encoded, y=y_test)

        result = {
            'config': {
                'task': config.task,
                'include_abc': config.include_abc,
                'model_family': config.model_family,
                'horizon_weeks': config.horizon_weeks,
                'train_split': config.train_split,
                'val_split': config.val_split,
                'n_optuna_trials': config.n_optuna_trials,
                'random_state': config.random_state,
                'max_train_rows': config.max_train_rows,
            },
            'best_params': best_params,
            'val_metrics': val_eval['metrics'],
            'test_metrics': test_eval['metrics'],
            'n_train': len(X_train_encoded),
            'n_val': len(X_val_encoded),
            'n_test': len(X_test_encoded),
            'feature_cols': config.feature_cols
        }

        self.artifacts.save_json(obj=result, filename='metrics.json')
        self.artifacts.save_json(obj=self.encoder.to_dict(), filename='categorical_encoder.json')
        self.artifacts.save_native_model(model=self.model_)
        self.artifacts.save_onnx(model=self.model_, n_features=X_test_encoded.shape[1])

        return {
            'model': self.model_,
            'study': self.study_,
            'result': result,
            'X_test': X_test_encoded,
            'y_test': y_test,
            'preds': test_eval['preds'],
            'proba': test_eval['proba'],
            'meta_test': meta_test,
            'encoders': self.encoder.to_dict(),
        }


    def run_or_load(self, df: pd.DataFrame | None = None, verbose: bool = True, force_retrain: bool = False) -> Dict:
        """Reuse a previously-persisted model for this exact RunConfig if
        its artifacts already exist on disk, instead of retraining."""

        config = self.config
        native_name = 'model_native.json' if config.model_family == 'xgboost' else 'model_native.joblib'
        metrics_path = self.artifacts.models_dir / 'metrics.json'
        native_path = self.artifacts.models_dir / native_name
        
        if force_retrain or not (metrics_path.exists() and native_path.exists()):
            return self.run(df=df, verbose=verbose)
 
        if verbose:
            print(f"[{config.run_name}] reusing cached artifacts from {self.artifacts.models_dir}")
 
        
        with open(metrics_path) as f:
            result = json.load(f)

        result['config'].setdefault('model_family', 'xgboost')  # backward-compat
 
        if config.model_family == 'xgboost':
            self.model_ = xgb.XGBClassifier() if config.task == 'classification' else xgb.XGBRegressor()
            self.model_.load_model(str(native_path))
        else:
            self.model_ = joblib.load(native_path)
 
        if df is None:
            df = self.load_feature_table()
            
        X, y, meta = self.supervised_learning_frame.build(df)
        splits = self.splitter.split(X=X, y=y, meta=meta)
        (X_train, _, _) = splits['train']
        (X_test, y_test, meta_test) = splits['test']
 
        self.encoder.fit(X_train)
        X_test_encoded = self.encoder.transform(X_test)
        test_eval = self.evaluator.evaluate(model=self.model_, X=X_test_encoded, y=y_test)
 
        return {
            'model': self.model_,
            'study': None,
            'result': result,
            'X_test': X_test_encoded,
            'y_test': y_test,
            'preds': test_eval['preds'],
            'proba': test_eval['proba'],
            'meta_test': meta_test,
            'encoders': self.encoder.to_dict(),
        }
