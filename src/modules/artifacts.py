import json
import joblib
from pathlib import Path
from typing import Dict

import xgboost as xgb

from src.config import RunConfig
from src.modules.export import ONNXExport


class Artifacts:

    def __init__(self, config: RunConfig):
        self.config = config

    @property
    def base_dir(self):
        return Path(__file__).resolve().parent.parent.parent

    @property
    def artifacts_dir(self):
        return self.base_dir / 'artifacts'

    @property
    def data_dir(self):
        return self.artifacts_dir / 'data'

    @property
    def runs_dir(self):
        _dir = self.artifacts_dir / 'runs'
        _dir.mkdir(parents=True, exist_ok=True)
        return _dir

    @property
    def models_dir(self):
        _dir = self.runs_dir / 'models' / self.config.run_name
        _dir.mkdir(parents=True, exist_ok=True)
        return _dir
    
    @property
    def plots_dir(self):
        return self.runs_dir / 'plots'

    @property
    def trains_plot_dir(self):
        _dir = self.plots_dir / 'train'
        _dir.mkdir(parents=True, exist_ok=True)
        return _dir
    
    @property
    def benchmark_plots_dir(self):
        _dir = self.plots_dir / 'benchmark'
        _dir.mkdir(parents=True, exist_ok=True)
        return _dir

    def save_onnx(self, model: xgb.XGBRegressor | xgb.XGBClassifier, n_features: int)->Path:
        path = self.models_dir / "weights.onnx"
        ONNXExport.export(model=model, path=path, n_features=n_features, config=self.config)
        return path

    def save_native_model(self, model: xgb.XGBRegressor | xgb.XGBClassifier)->Path:
        if self.config.model_family == 'xgboost':
            path = self.models_dir / 'model_native.json'
            model.save_model(str(path))
        else:
            path = self.models_dir / 'model_native.joblib'
            joblib.dump(model, path)

        return path

    def save_json(self, obj: Dict, filename: str):
        path = self.models_dir / filename
        with open(path, 'w') as fp:
            json.dump(obj=obj, fp=fp, indent=2, default=str)
        return path