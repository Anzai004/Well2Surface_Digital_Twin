"""
State Persistence & Telemetry Logging Pipeline
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd


class DigitalTwinJSONEncoder(json.JSONEncoder):
    """Encodes NumPy scalar types and arrays into native JSON-serializable types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_simulation_state(state: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Serializes complete digital twin SCADA & operational states to JSON."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, cls=DigitalTwinJSONEncoder)


def load_simulation_state(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Loads saved operational states from a JSON artifact."""
    in_path = Path(filepath)
    if not in_path.exists():
        raise FileNotFoundError(f"State file {filepath} does not exist.")
    with open(in_path, "r", encoding="utf-8") as f:
        return json.load(f)


def export_telemetry_to_csv(
    telemetry_records: Union[List[Dict[str, Any]], pd.DataFrame],
    filepath: Union[str, Path],
) -> None:
    """Exports time-series dynamic telemetry logs to CSV."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(telemetry_records, pd.DataFrame):
        df = telemetry_records
    else:
        df = pd.DataFrame(telemetry_records)
    df.to_csv(out_path, index=False)