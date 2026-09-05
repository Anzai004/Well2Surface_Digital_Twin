"""
Unit tests for SRP Mechanical Lift Engine (Gemini Pro/Flash Components).
Validates rod stretch, viscous drag, PPRL yield checks, dyno card rasterization, and 5-state rules.
"""

import pytest
import numpy as np
from src.physics.srp_dynamics import (
    calculate_rod_stretch,
    calculate_buoyant_weight,
    calculate_viscous_drag,
    calculate_pprl_and_check_yield,
    calculate_effective_stroke_and_displacement,
    calculate_spm_safe_floor,
    rasterize_dyno_card,
    classify_dyno_card_5state,
)


def test_rod_stretch_and_buoyancy():
    """Verify Hooke's law rod stretch and buoyancy calculations."""
    F_load = 50000.0  # N
    L_rod = 1000.0    # m
    A_rod = 0.0005    # m2
    E_steel = 2.07e11 # Pa

    delta_L = calculate_rod_stretch(F_load, L_rod, A_rod, E_steel)
    expected_stretch = (50000.0 * 1000.0) / (0.0005 * 2.07e11)
    assert np.isclose(delta_L, expected_stretch)

    W_buoy = calculate_buoyant_weight(W_rod_dry=80000.0, rho_fluid=950.0, rho_steel=7850.0)
    assert W_buoy < 80000.0


def test_viscous_drag_and_pprl_yield_flag():
    """Verify viscous drag and 90% PPRL yield interrupt triggering."""
    F_drag = calculate_viscous_drag(
        mu_cp=66.0, v_rod=1.2, L_rod=1000.0, D_tubing=0.076, D_rod=0.025
    )
    assert F_drag > 0.0

    # Over-yield load scenario
    pprl, yield_flag = calculate_pprl_and_check_yield(
        W_r=50000.0, W_f=30000.0, F_drag=15000.0, yield_limit_load=100000.0
    )
    assert pprl == 95000.0
    assert yield_flag is True  # 95k > 90% of 100k


def test_spm_safe_floor_and_effective_stroke():
    """Verify SPM clamping floor and stroke kinematics."""
    spm_safe = calculate_spm_safe_floor(mu_cp=10000.0, spm_floor=2.0)
    assert spm_safe == 2.0

    S_eff, V_disp = calculate_effective_stroke_and_displacement(
        S_surface=3.0, delta_L=0.5, spm=5.0, D_plunger=0.044
    )
    assert S_eff == 2.5
    assert V_disp > 0.0


def test_dyno_card_rasterization_shape():
    """Verify dyno card tensor array generation to 224x224."""
    pos = np.sin(np.linspace(0, 2 * np.pi, 100))
    load = np.cos(np.linspace(0, 2 * np.pi, 100)) * 1000 + 5000

    tensor = rasterize_dyno_card(pos, load, grid_size=224)
    assert tensor.shape == (224, 224)
    assert np.max(tensor) == 255


def test_5state_diagnostic_classifier():
    """Verify condition branches in 5-state diagnostic classifier."""
    # Mechanical Overload
    state1 = classify_dyno_card_5state(
        fillage_pct=90.0, pprl=95000, yield_limit=100000, current_spm=4.0, spm_safe=5.0
    )
    assert state1 == "Mechanical Overload / Rod Stress"

    # Fluid Pound
    state2 = classify_dyno_card_5state(
        fillage_pct=50.0, pprl=50000, yield_limit=100000, current_spm=4.0, spm_safe=5.0
    )
    assert state2 == "Fluid Pound / Low Fillage"

    # Normal Operation
    state3 = classify_dyno_card_5state(
        fillage_pct=85.0, pprl=50000, yield_limit=100000, current_spm=3.0, spm_safe=5.0
    )
    assert state3 == "Normal Operation"