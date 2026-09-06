"""
Unit tests for SRP Dynamics & Downhole Diagnostic Physics Module.
Validates rod stretch, buoyancy, viscous drag, PPRL, safe SPM floor,
dyno card rasterization, 5-state classifier, and 1D Gibbs PDE wave solver.
"""

import numpy as np
import pytest

from src.physics.srp_dynamics import (
    calculate_buoyant_weight,
    calculate_effective_stroke_and_displacement,
    calculate_pprl_and_check_yield,
    calculate_rod_stretch,
    calculate_spm_safe_floor,
    calculate_viscous_drag,
    calculate_wave_speed_and_damping,
    check_cfl_stability,
    classify_dyno_card_5state,
    rasterize_dyno_card,
    solve_gibbs_wave_equation,
)


def test_calculate_rod_stretch():
    """Verify elastic rod stretch calculation using Hooke's Law."""
    delta_l = calculate_rod_stretch(
        F_load=100000.0, L_rod=1000.0, A_rod=0.0005, E_steel=2.07e11
    )
    expected = (100000.0 * 1000.0) / (0.0005 * 2.07e11)
    assert np.isclose(delta_l, expected)


def test_calculate_buoyant_weight():
    """Verify hydrostatic buoyancy weight reduction."""
    w_buoy = calculate_buoyant_weight(
        W_rod_dry=50000.0, rho_fluid=950.0, rho_steel=7850.0
    )
    expected = 50000.0 * (1.0 - (950.0 / 7850.0))
    assert np.isclose(w_buoy, expected)


def test_calculate_buoyant_weight_invalid_density():
    """Verify exception when fluid density exceeds or equals steel density."""
    with pytest.raises(ValueError, match="cannot exceed steel density"):
        calculate_buoyant_weight(W_rod_dry=50000.0, rho_fluid=8000.0, rho_steel=7850.0)


def test_calculate_viscous_drag():
    """Verify viscous drag force on rod string moving through fluid column."""
    f_drag = calculate_viscous_drag(
        mu_cp=100.0, v_rod=1.2, L_rod=1000.0, D_tubing=0.0889, D_rod=0.0254
    )
    assert f_drag > 0.0
    assert np.isclose(
        f_drag,
        (2.0 * np.pi * 0.1 * 1.2 * 1000.0) / np.log(0.0889 / 0.0254),
    )


def test_calculate_pprl_and_check_yield():
    """Verify PPRL load summation and yield threshold interrupt flag."""
    pprl, flag = calculate_pprl_and_check_yield(
        W_r=40000.0, W_f=20000.0, F_drag=5000.0, yield_limit_load=100000.0
    )
    assert pprl == 65000.0
    assert not flag  # 65,000 <= 90,000 threshold

    pprl_high, flag_high = calculate_pprl_and_check_yield(
        W_r=60000.0, W_f=30000.0, F_drag=5000.0, yield_limit_load=100000.0
    )
    assert pprl_high == 95000.0
    assert flag_high  # 95,000 > 90,000 threshold


def test_calculate_effective_stroke_and_displacement():
    """Verify effective stroke reduction and daily fluid displacement calculation."""
    s_eff, v_disp = calculate_effective_stroke_and_displacement(
        S_surface=3.0, delta_L=0.5, spm=5.0, D_plunger=0.05715
    )
    assert s_eff == 2.5
    expected_v_disp = (np.pi / 4.0) * (0.05715 ** 2) * 2.5 * 5.0 * 1440.0
    assert np.isclose(v_disp, expected_v_disp)


def test_calculate_spm_safe_floor():
    """Verify SPM clamping floor to prevent rod fall hesitation."""
    # High viscosity forces calculated SPM below spm_floor (2.0)
    spm_safe_clamped = calculate_spm_safe_floor(mu_cp=10000.0, spm_floor=2.0)
    assert spm_safe_clamped == 2.0

    # FIX: original mu_cp=10.0 with the default K_factor=1e-4 still computes
    # spm_calc = 1e-4*(7850-950)*9.81/10 = 0.677 SPM, which is BELOW the 2.0
    # floor and gets clamped -- the assertion `> 2.0` was checking the wrong
    # side of the clamp given the default K_factor magnitude. A sufficiently
    # low viscosity is needed to push spm_calc above the floor.
    spm_safe_high = calculate_spm_safe_floor(mu_cp=1.0, spm_floor=2.0)
    assert spm_safe_high > 2.0

    # Explicit unclamped comparison: raising K_factor with mu_cp=10.0 must
    # also clear the floor -- isolates the K_factor scaling behavior from
    # the mu_cp magnitude used above.
    spm_safe_high_k = calculate_spm_safe_floor(mu_cp=10.0, K_factor=1.0, spm_floor=2.0)
    assert spm_safe_high_k > 2.0


def test_rasterize_dyno_card():
    """Verify 2D tensor rasterization of position-load vector arrays."""
    pos = np.sin(np.linspace(0, 2 * np.pi, 100))
    load = np.cos(np.linspace(0, 2 * np.pi, 100))

    tensor = rasterize_dyno_card(position_array=pos, load_array=load, grid_size=224)

    assert tensor.shape == (224, 224)
    assert tensor.dtype == np.uint8
    assert np.max(tensor) == 255
    assert np.sum(tensor == 255) > 0


def test_classify_dyno_card_5state():
    """Verify rule-based state classification across all operational regimes."""
    assert (
        classify_dyno_card_5state(
            fillage_pct=90.0, pprl=95000.0, yield_limit=100000.0,
            current_spm=5.0, spm_safe=6.0,
        )
        == "Mechanical Overload / Rod Stress"
    )

    assert (
        classify_dyno_card_5state(
            fillage_pct=90.0, pprl=50000.0, yield_limit=100000.0,
            current_spm=5.0, spm_safe=6.0, gas_lock_flag=True,
        )
        == "Gas Interference / Lock"
    )

    assert (
        classify_dyno_card_5state(
            fillage_pct=50.0, pprl=50000.0, yield_limit=100000.0,
            current_spm=5.0, spm_safe=6.0,
        )
        == "Fluid Pound / Low Fillage"
    )

    assert (
        classify_dyno_card_5state(
            fillage_pct=90.0, pprl=50000.0, yield_limit=100000.0,
            current_spm=8.0, spm_safe=5.0,
        )
        == "Viscous Drag Sucking / High Friction"
    )

    assert (
        classify_dyno_card_5state(
            fillage_pct=90.0, pprl=50000.0, yield_limit=100000.0,
            current_spm=5.0, spm_safe=6.0,
        )
        == "Normal Operation"
    )


def test_calculate_wave_speed_and_damping():
    """Verify acoustic wave speed (~5134 m/s) and viscous damping calculation."""
    a, c = calculate_wave_speed_and_damping(
        mu_cp=50.0, A_rod=0.0005, D_tubing=0.0889, D_rod=0.0254,
        rho_steel=7850.0, E_steel=2.07e11,
    )
    expected_a = np.sqrt(2.07e11 / 7850.0)
    assert np.isclose(a, expected_a)
    assert c > 0.0


def test_check_cfl_stability():
    """Verify Courant-Friedrichs-Lewy (CFL) condition checks."""
    a = 5000.0
    dx = 100.0
    safe_dt = 0.01
    cfl_ratio = check_cfl_stability(dt=safe_dt, dx=dx, a_wave_speed=a, enforce=True)
    assert cfl_ratio == 0.5

    unsafe_dt = 0.03
    with pytest.raises(ValueError, match="CFL stability violated"):
        check_cfl_stability(dt=unsafe_dt, dx=dx, a_wave_speed=a, enforce=True)

    ratio_unsafe = check_cfl_stability(dt=unsafe_dt, dx=dx, a_wave_speed=a, enforce=False)
    assert ratio_unsafe == 1.5


def test_solve_gibbs_wave_equation_basic_execution():
    """Verify execution, output keys, and array dimensions of the Gibbs wave PDE solver."""
    dt = 0.001
    L_rod = 500.0
    num_nodes = 20

    time_vec = np.arange(0.0, 1.0, dt)
    x_surface = 1.5 + 1.0 * np.sin(2.0 * np.pi * 0.1 * time_vec)

    results = solve_gibbs_wave_equation(
        x_surface=x_surface, dt=dt, L_rod=L_rod, mu_cp=10.0,
        A_rod=0.0005, D_tubing=0.0889, D_rod=0.0254,
        num_spatial_nodes=num_nodes, enforce_cfl=True,
    )

    assert "u_downhole" in results
    assert "F_plunger" in results
    assert "u_grid" in results
    assert results["u_downhole"].shape == (len(time_vec),)
    assert results["F_plunger"].shape == (len(time_vec),)
    assert results["u_grid"].shape == (len(time_vec), num_nodes)
    assert results["cfl_ratio"] <= 1.0


def test_solve_gibbs_wave_equation_step_response_ringing():
    """Verify propagation delay and elastic damped ringing under step-response input."""
    dt = 0.0005
    L_rod = 1000.0
    num_nodes = 20

    num_steps = 4000
    x_surface = np.zeros(num_steps)
    x_surface[100:] = 1.0

    results = solve_gibbs_wave_equation(
        x_surface=x_surface, dt=dt, L_rod=L_rod, mu_cp=20.0,
        A_rod=0.0005, D_tubing=0.0889, D_rod=0.0254,
        num_spatial_nodes=num_nodes,
    )

    u_downhole = results["u_downhole"]

    # FIX: exact `== 0.0` fails on machine-epsilon-scale float noise
    # (~3e-19) propagated through the finite-difference recursion before
    # the wave has physically arrived -- use an absolute tolerance floor
    # instead of bitwise equality (3e-19 m vs. a 1.0 m step is pure noise).
    assert abs(u_downhole[200]) < 1e-9
    assert np.max(u_downhole) > 0.0


def test_solve_gibbs_wave_equation_cfl_validation():
    """Verify CFL violation handling during wave solver instantiation."""
    dt = 0.1
    x_surface = np.array([0.0, 1.0, 2.0, 1.0, 0.0])

    with pytest.raises(ValueError, match="CFL stability violated"):
        solve_gibbs_wave_equation(
            x_surface=x_surface, dt=dt, L_rod=500.0, mu_cp=10.0,
            A_rod=0.0889, D_tubing=0.0889, D_rod=0.0254,
            enforce_cfl=True,
        )