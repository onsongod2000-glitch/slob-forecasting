import numpy as np
import pandas as pd

def to_numpy_f32(X: pd.DataFrame)->np.ndarray:
    """Remove column names before handing data to onnxmltools/XGB"""
    return X.to_numpy(dtype=np.float32)