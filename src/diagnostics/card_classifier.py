"""
Pipeline-facing diagnostics wrapper.

main.py drives the pipeline against `rasterize_dyno_card()` and
`classify_dyno_card()` with the signatures defined here. Both are thin
adapters over the physics-layer primitives in
`src.physics.srp_dynamics` (rasterize_dyno_card / classify_dyno_card_5state) --
this module owns the "what does main.py need to call" contract, while
srp_dynamics.py owns the underlying rod-mechanics math.
"""

from typing import Dict, Optional

import numpy as np

from src.physics.srp_dynamics import (
    classify_dyno_card_5state,
    rasterize_dyno_card as _rasterize_square,
)

_ACTION_MAP = {
    "Normal Operation": "Maintain target SPM trajectory.",
    "Fluid Pound / Low Fillage": "Reduce SPM / decrease stroke rate to fill pump barrel.",
    "Gas Interference / Lock": "Cycle VFD speed to bleed gas; adjust soak timing.",
    "Mechanical Overload / Rod Stress": "Emergency throttle override to idle baseline (SPM = 2.0).",
    "Viscous Drag Sucking / High Friction": "Reduce SPM toward the current SPM_safe(t) floor.",
}


def rasterize_dyno_card(
    position_array: np.ndarray,
    load_array: np.ndarray,
    image_size: tuple = (224, 224),
) -> np.ndarray:
    """
    Pipeline adapter: accepts an (H, W) `image_size` tuple (as main.py's
    telemetry pipeline supplies) and rasterizes via the physics-layer
    rasterize_dyno_card(), which is natively square (`grid_size`).

    Raises:
        ValueError: If image_size is not square (H != W) -- the underlying
            rasterizer only supports square grids; a non-square request
            signals a caller/config mismatch that should surface loudly
            rather than silently distorting the card aspect ratio.
    """
    height, width = image_size
    if height != width:
        raise ValueError(
            f"rasterize_dyno_card image_size must be square, got {image_size}."
        )
    return _rasterize_square(position_array, load_array, grid_size=height)


def classify_dyno_card(
    position_array: np.ndarray,
    load_array: np.ndarray,
    pprl_rated: float,
    fillage_pct: Optional[float] = None,
    current_spm: Optional[float] = None,
    spm_safe: Optional[float] = None,
    gas_lock_flag: bool = False,
) -> Dict[str, object]:
    """
    Pipeline adapter around classify_dyno_card_5state(): derives the peak
    downhole load (PPRL proxy) from the supplied load array and returns a
    diagnostic dict with 'state', 'confidence', 'recommended_action', and
    'pprl' keys.

    Fillage estimation:
        When `fillage_pct` telemetry is not supplied directly, this adapter
        falls back to a coarse card-shape heuristic: a compressive
        (negative) minimum load on the downstroke is treated as the
        low-fillage / rod-float signature (55%), otherwise nominal fillage
        (92%) is assumed. This is a documented simplification -- real
        fillage should come from pump-card area integration or direct
        telemetry once available.
    """
    load_arr = np.asarray(load_array, dtype=float)
    pprl = float(np.max(load_arr))
    min_load = float(np.min(load_arr))

    if fillage_pct is None:
        fillage = 55.0 if min_load < 0.0 else 92.0
    else:
        fillage = fillage_pct

    spm_now = current_spm if current_spm is not None else 0.0
    spm_lim = spm_safe if spm_safe is not None else float("inf")

    state = classify_dyno_card_5state(
        fillage_pct=fillage,
        pprl=pprl,
        yield_limit=pprl_rated,
        current_spm=spm_now,
        spm_safe=spm_lim,
        gas_lock_flag=gas_lock_flag,
    )

    confidence = 0.97 if state == "Normal Operation" else 0.90

    return {
        "state": state,
        "confidence": confidence,
        "recommended_action": _ACTION_MAP[state],
        "pprl": pprl,
        "fillage_pct": fillage,
    }