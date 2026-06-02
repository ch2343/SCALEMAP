import json
import os
from typing import Dict, Optional

import pandas as pd


def load_json_if_exists(path: str) -> Dict:
    if path is None or not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_dataframe(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def extract_runtime_fields(runtime_dict):
    peak_memory_bytes = runtime_dict.get("peak_memory_use")
    training_time_sec = runtime_dict.get("total_runtime_min")

    if peak_memory_bytes is not None:
        peak_memory_bytes *= 1024**3  # GiB → bytes

    if training_time_sec is not None:
        training_time_sec *= 60  # minutes → seconds

    return peak_memory_bytes, training_time_sec