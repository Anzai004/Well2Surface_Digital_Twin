"""
Dyno Card Tensor Rasterizer & 5-State Heuristic Diagnostic Classifier
Asset: Baghewala Field Heavy Oil Operations (18° API)
"""

from typing import Dict, Tuple, Union
import numpy as np


def rasterize_dyno_card(
    position: np.ndarray,
    load: np.ndarray,
    image_size: Tuple[int, int] = (224, 224),
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Renders 1D downhole force-position arrays into a normalized 224x224 single-channel tensor.
    
    Parameters
    ----------
    position : np.ndarray
        Plunger axial displacement array u_plunger (m or in).
    load : np.ndarray
        Plunger load array F_plunger (N or lbf).
    image_size : Tuple[int, int]
        Output image dimensions (H, W), default (224, 224).
    line_thickness : int
        Bresenham anti-aliasing line footprint width.
        
    Returns
    -------
    np.ndarray
        2D single-channel float32 array normalized to [0.0, 1.0].
    """
    pos = np.asarray(position, dtype=np.float64)
    f_load = np.asarray(load, dtype=np.float64)

    if len(pos) != len(f_load) or len(pos) < 4:
        raise ValueError("Position and load arrays must have equal length >= 4.")

    h, w = image_size
    tensor = np.zeros((h, w), dtype=np.float32)

    # Normalize position to width [padding, w - 1 - padding]
    pos_min, pos_max = np.min(pos), np.max(pos)
    load_min, load_max = np.min(f_load), np.max(f_load)

    delta_pos = pos_max - pos_min if pos_max > pos_min else 1.0
    delta_load = load_max - load_min if load_max > load_min else 1.0

    pad = 12
    norm_x = pad + (pos - pos_min) / delta_pos * (w - 1 - 2 * pad)
    # Invert y: maximum load sits at row index 0 (top of image)
    norm_y = (h - 1 - pad) - (f_load - load_min) / delta_load * (h - 1 - 2 * pad)

    x_coords = np.clip(np.round(norm_x).astype(np.int32), 0, w - 1)
    y_coords = np.clip(np.round(norm_y).astype(np.int32), 0, h - 1)

    # Rasterize perimeter via digital differential analyzer
    n_pts = len(x_coords)
    for i in range(n_pts):
        x0, y0 = x_coords[i], y_coords[i]
        x1, y1 = x_coords[(i + 1) % n_pts], y_coords[(i + 1) % n_pts]

        num_steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        xs = np.linspace(x0, x1, num_steps + 1).round().astype(np.int32)
        ys = np.linspace(y0, y1, num_steps + 1).round().astype(np.int32)

        for offset_x in range(-line_thickness // 2, line_thickness // 2 + 1):
            for offset_y in range(-line_thickness // 2, line_thickness // 2 + 1):
                cur_x = np.clip(xs + offset_x, 0, w - 1)
                cur_y = np.clip(ys + offset_y, 0, h - 1)
                tensor[cur_y, cur_x] = 1.0

    return tensor


def classify_dyno_card(
    position: np.ndarray,
    load: np.ndarray,
    pprl_rated: float = 24000.0,
    compression_threshold: float = -100.0,
) -> Dict[str, Union[str, float, bool]]:
    """
    Evaluates downhole plunger card morphology using a 5-state heuristic classifier.
    
    States:
      - 'Normal Operation'
      - 'Rod Floating / Slack'
      - 'Fluid Pound'
      - 'Gas Locking'
      - 'Pump Unsetting'
    """
    pos = np.asarray(position, dtype=np.float64)
    f_load = np.asarray(load, dtype=np.float64)

    min_f, max_f = np.min(f_load), np.max(f_load)
    stroke_span = np.ptp(pos)
    load_span = np.ptp(f_load)
    area_work = np.abs(np.trapezoid(f_load, pos))
    bounding_box_area = (stroke_span * load_span) if (stroke_span * load_span) > 0 else 1.0
    fillage_ratio = area_work / bounding_box_area

    # Split into Upstroke (velocity > 0) and Downstroke (velocity < 0)
    vel = np.gradient(pos)
    downstroke_idx = np.where(vel < -1e-5)[0]
    upstroke_idx = np.where(vel > 1e-5)[0]

    # Feature 1: Rod Floating / Slack Check (Axial compression or near-zero load during downstroke)
    has_negative_compression = min_f < compression_threshold
    downstroke_min = np.min(f_load[downstroke_idx]) if len(downstroke_idx) > 0 else min_f

    if has_negative_compression or downstroke_min < 0:
        return {
            "state": "Rod Floating / Slack",
            "confidence": 0.94,
            "fillage_ratio": float(fillage_ratio),
            "peak_load": float(max_f),
            "min_load": float(min_f),
            "requires_action": True,
            "recommended_action": "Throttle SPM down via VFD to restore rod tension[cite: 8].",
        }

    # Feature 2: Pump Unsetting (Erratic baseline, abnormal load hysteresis)
    if min_f > 0.60 * pprl_rated or max_f > 1.15 * pprl_rated or load_span < (0.10 * pprl_rated):
        return {
            "state": "Pump Unsetting",
            "confidence": 0.89,
            "fillage_ratio": float(fillage_ratio),
            "peak_load": float(max_f),
            "min_load": float(min_f),
            "requires_action": True,
            "recommended_action": "Emergency pump inspection; stroke load envelope decoupled[cite: 8].",
        }

    # Feature 3: Fluid Pound vs Gas Locking during Downstroke
    if len(downstroke_idx) > 8:
        down_pos = pos[downstroke_idx]
        down_loads = f_load[downstroke_idx]
        
        # Sort along descending stroke
        sort_order = np.argsort(-down_pos)
        sorted_pos = down_pos[sort_order]
        sorted_load = down_loads[sort_order]

        # Calculate localized slope dF/du during downstroke
        d_load_du = np.gradient(sorted_load, sorted_pos + 1e-6)
        max_down_gradient = np.max(np.abs(d_load_du))
        norm_gradient = max_down_gradient / (load_span / (stroke_span + 1e-6))

        # Sharp vertical cliff indicates sudden impact with fluid level (Fluid Pound)
        if norm_gradient > 2.8 and fillage_ratio < 0.68:
            return {
                "state": "Fluid Pound",
                "confidence": 0.92,
                "fillage_ratio": float(fillage_ratio),
                "peak_load": float(max_f),
                "min_load": float(min_f),
                "requires_action": True,
                "recommended_action": "Reduce SPM to allow pump barrel complete liquid fillage[cite: 8].",
            }

        # Smooth concave delayed valve closure (Gas Locking)
        if norm_gradient < 2.0 and fillage_ratio < 0.55:
            return {
                "state": "Gas Locking",
                "confidence": 0.87,
                "fillage_ratio": float(fillage_ratio),
                "peak_load": float(max_f),
                "min_load": float(min_f),
                "requires_action": True,
                "recommended_action": "Cycle VFD speed to bleed gas or adjust thermal soak[cite: 8].",
            }

    # Default State: Balanced envelope
    return {
        "state": "Normal Operation",
        "confidence": 0.96,
        "fillage_ratio": float(fillage_ratio),
        "peak_load": float(max_f),
        "min_load": float(min_f),
        "requires_action": False,
        "recommended_action": "Maintain optimal setpoints[cite: 8].",
    }