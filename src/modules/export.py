from pathlib import Path

import xgboost as xgb
from skl2onnx import convert_sklearn
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from skl2onnx.common.data_types import FloatTensorType as FTensorStype

from src.config import RunConfig

class ONNXExport:

    @staticmethod
    def export(model: xgb.XGBRegressor | xgb.XGBClassifier, n_features: int, path: Path, config: RunConfig)->None:
        path.parent.mkdir(parents=True, exist_ok=True)

        if config.model_family == 'xgboost':
            initial_type = [('input', FloatTensorType([None, n_features]))]
            onnx_model = convert_xgboost(model, initial_types=initial_type)
        else:
            initial_type = [('input', FTensorStype([None, n_features]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)

        with open(path, 'wb') as f:
            f.write(onnx_model.SerializeToString())