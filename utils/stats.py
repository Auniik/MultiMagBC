import json
import os
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays and types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def save_as_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)

def save_as_csv(data, path):
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)