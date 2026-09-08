"""
JSON state serialization and time-series telemetry CSV logging.

Handles NumPy 2.0+ scalar/bool types transparently so simulation-state
dicts containing np.float64 / np.bool_ / np.ndarray values can be dumped
straight to JSON/CSV without a manual conversion pass at every call site.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np


def _to_native(value: Any) -> Any:
    """Recursively converts NumPy scalar/array types to native Python types."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_to_native(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(v) for v in value]
    return value


def save_simulation_state(state: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Serializes a simulation state dict to JSON, normalizing any NumPy
    scalar/bool/array types encountered along the way.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    native_state = _to_native(state)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(native_state, f, indent=2, sort_keys=True)


def export_telemetry_to_csv(
    records: List[Dict[str, Any]],
    path: Union[str, Path],
) -> None:
    """
    Appends (or creates) a time-series telemetry CSV log from a list of
    flat state-payload dicts. If the file already exists, new rows are
    appended using the existing header; new keys not present in the
    existing header are ignored (log schema is fixed at first write) to
    keep the CSV rectangular across pipeline runs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        return

    native_records = [_to_native(r) for r in records]
    file_exists = path.exists() and path.stat().st_size > 0

    if file_exists:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
    else:
        header = list(native_records[0].keys())

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for record in native_records:
            writer.writerow(record)