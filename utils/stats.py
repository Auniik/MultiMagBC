import json
import os

def save_as_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def save_as_csv(data, path):
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)