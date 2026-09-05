"""
Unit tests for CSS Thermal Physics Module.
Validates guardrail enforcement, steam enthalpy, heat capacities, Andrade viscosity,
dynamic decay rate alpha, and transient temperature ODE solutions.
"""

import numpy as np
import pytest

from src.physics.css_thermal import (
    calculate_absorbed_heat,
    calculate_andrade_viscosity,
    calculate_composite_heat_capacity,
    calculate_dynamic_alpha,
    calculate_steam_enthalpy_injection,
    calculate_temperature_decay,
    validate_thermal_guardrails,
)


def test_validate_thermal_guardrails_pass():
    """Verify that valid physical state parameters pass guardrail checks silently."""
    validate_thermal_guardrails(
        temperature_c=150.0,
        native_res_temp_c=47.0,
        steam_quality=0.8,
        viscosity_cp=10.0,
    )


def test_validate_thermal_guardrails_temperature_violation():
    """Verify exception raised when temperature drops below native reservoir floor."""
    with pytest.raises(ValueError, match="cannot fall below native reservoir temperature"):
        validate_thermal_guardrails(temperature_c=30.0, native_res_temp_c=47.0)


def test_validate_thermal_guardrails_slack_tolerance():
    """Verify that floating-point overshoot within tolerance band does not raise an error."""
    # Dip 1e-7 below 47.0°C (within default 1e-6 tolerance)
    validate_thermal_guardrails(
        temperature_c=47.0 - 1e-7,
        native_res_temp_c=47.0,
        temperature_tolerance_c=1e-6,
    )


def test_validate_thermal_guardrails_steam_quality_and_viscosity():
    """Verify guardrails for out-of-bound steam quality and non-positive viscosity."""
    with pytest.raises(ValueError, match="Steam quality"):
        validate_thermal_guardrails(temperature_c=100.0, steam_quality=1.2)

    with pytest.raises(ValueError, match="viscosity"):
        validate_thermal_guardrails(temperature_c=100.0, viscosity_cp=0.0)


def test_calculate_steam_enthalpy_injection():
    """Verify enthalpy calculation for steam injection."""
    # m_s = 1000 kg, x = 0.8, T_steam = 300°C, T_res = 47°C
    # Q = 1000 * (0.8 * 2000 + 4.184 * (300 - 47)) = 1000 * (1600 + 1058.552) = 2,658,552 kJ
    q_inj = calculate_steam_enthalpy_injection(
        m_s=1000.0, x_qual=0.8, T_steam=300.0, T_res=47.0
    )
    expected = 1000.0 * (0.8 * 2000.0 + 4.184 * (300.0 - 47.0))
    assert np.isclose(q_inj, expected)


def test_calculate_composite_heat_capacity():
    """Verify volumetric composite heat capacity of saturated rock-fluid matrix."""
    rho_c_comp = calculate_composite_heat_capacity(
        phi=0.3,
        S_o=0.7,
        S_w=0.3,
        rho_r=2650.0,
        C_pr=880.0,
        rho_o=946.5,
        C_po=2000.0,
        rho_w=1000.0,
        C_pw=4184.0,
    )
    # Matrix: 0.7 * 2650 * 880 = 1,632,400
    # Fluid: 0.3 * (0.7 * 946.5 * 2000 + 0.3 * 1000 * 4184) = 0.3 * (1,325,100 + 1,255,200) = 774,090
    # Total = 2,406,490 J/(m^3.K)
    expected = (0.7 * 2650.0 * 880.0) + 0.3 * (
        0.7 * 946.5 * 2000.0 + 0.3 * 1000.0 * 4184.0
    )
    assert np.isclose(rho_c_comp, expected)


def test_calculate_composite_heat_capacity_invalid_saturation():
    """Verify error handling when saturations do not sum to 1.0."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        calculate_composite_heat_capacity(
            phi=0.3,
            S_o=0.8,
            S_w=0.8,
            rho_r=2650.0,
            C_pr=880.0,
            rho_o=900.0,
            C_po=2000.0,
            rho_w=1000.0,
            C_pw=4184.0,
        )


def test_calculate_absorbed_heat():
    """Verify composite heat energy absorbed by reservoir volume."""
    q_abs = calculate_absorbed_heat(V_zone=5000.0, rho_C_composite=2.4e6, delta_T=200.0)
    assert np.isclose(q_abs, 5000.0 * 2.4e6 * 200.0)


def test_calculate_andrade_viscosity():
    """Verify Andrade temperature-viscosity exponential calibration."""
    # At high temperature (300°C / 573.15 K), viscosity drops dramatically
    mu_hot = calculate_andrade_viscosity(T_celsius=300.0)
    # At lower native temperature (47°C / 320.15 K), viscosity is high
    mu_cold = calculate_andrade_viscosity(T_celsius=47.0)

    assert mu_hot < mu_cold
    # Sanity check order of magnitude and trend for A=-14.28, B=5820.4
    assert 30.0 < mu_cold < 70.0
    assert mu_hot < 1.0


def test_calculate_dynamic_alpha_day_basis():
    """Verify dynamic alpha thermal decay rate calculation on a per-day production basis."""
    alpha = calculate_dynamic_alpha(
        q_o=50.0,
        rho_o=946.5,
        C_po=2000.0,
        q_w=20.0,
        rho_w=1000.0,
        C_pw=4184.0,
        V_heated=5000.0,
        rho_C_composite=2.4e6,
        rate_time_basis="day",
    )
    # Num = 50 * 946.5 * 2000 + 20 * 1000 * 4184 = 94,650,000 + 83,680,000 = 178,330,000
    # Denom = 5000 * 2,400,000 = 12,000,000,000
    # Alpha = 178,330,000 / 12,000,000,000 = 0.014860833... day^-1
    expected = (50.0 * 946.5 * 2000.0 + 20.0 * 1000.0 * 4184.0) / (5000.0 * 2.4e6)
    assert np.isclose(alpha, expected)


def test_calculate_dynamic_alpha_second_basis_conversion():
    """Verify that rate_time_basis='second' scales production rates to day basis properly."""
    q_o_sec = 50.0 / 86400.0
    q_w_sec = 20.0 / 86400.0

    alpha_sec = calculate_dynamic_alpha(
        q_o=q_o_sec,
        rho_o=946.5,
        C_po=2000.0,
        q_w=q_w_sec,
        rho_w=1000.0,
        C_pw=4184.0,
        V_heated=5000.0,
        rho_C_composite=2.4e6,
        rate_time_basis="second",
    )
    alpha_day = calculate_dynamic_alpha(
        q_o=50.0,
        rho_o=946.5,
        C_po=2000.0,
        q_w=20.0,
        rho_w=1000.0,
        C_pw=4184.0,
        V_heated=5000.0,
        rho_C_composite=2.4e6,
        rate_time_basis="day",
    )
    assert np.isclose(alpha_sec, alpha_day)


def test_calculate_dynamic_alpha_invalid_inputs():
    """Verify error checks for invalid volumes, unknown time basis, or non-positive rates."""
    with pytest.raises(ValueError, match="rate_time_basis"):
        calculate_dynamic_alpha(
            1.0, 900.0, 2000.0, 0.0, 1000.0, 4184.0, 1000.0, 2e6, rate_time_basis="hour"
        )

    with pytest.raises(ValueError, match="V_heated"):
        calculate_dynamic_alpha(
            1.0, 900.0, 2000.0, 0.0, 1000.0, 4184.0, 0.0, 2e6
        )

    with pytest.raises(ValueError, match="must be strictly positive"):
        calculate_dynamic_alpha(
            0.0, 900.0, 2000.0, 0.0, 1000.0, 4184.0, 1000.0, 2e6
        )


def test_calculate_temperature_decay_scalar():
    """Verify temperature decay ODE closed-form solution for scalar time points."""
    # T(0) must equal T_steam
    t0_temp = calculate_temperature_decay(t=0.0, T_steam=300.0, T_res=47.0, alpha=0.05)
    assert np.isclose(t0_temp, 300.0)

    # T(t) decreases monotonically toward T_res
    t10_temp = calculate_temperature_decay(t=10.0, T_steam=300.0, T_res=47.0, alpha=0.05)
    expected_t10 = 47.0 + (300.0 - 47.0) * np.exp(-0.05 * 10.0)
    assert np.isclose(t10_temp, expected_t10)
    assert 47.0 < t10_temp < 300.0


def test_calculate_temperature_decay_asymptotic_limit():
    """Verify that as t -> inf, T(t) asymptotically approaches T_res without dipping below."""
    t_inf_temp = calculate_temperature_decay(t=10000.0, T_steam=300.0, T_res=47.0, alpha=0.05)
    assert np.isclose(t_inf_temp, 47.0, atol=1e-5)
    assert t_inf_temp >= 47.0


def test_calculate_temperature_decay_array_input():
    """Verify vectorization over time step arrays."""
    t_vec = np.array([0.0, 10.0, 50.0, 100.0])
    T_vec = calculate_temperature_decay(t=t_vec, T_steam=250.0, T_res=47.0, alpha=0.02)

    assert isinstance(T_vec, np.ndarray)
    assert T_vec.shape == (4,)
    assert T_vec[0] == 250.0
    assert np.all(np.diff(T_vec) < 0)  # Strictly decreasing
    assert np.all(T_vec >= 47.0)


def test_calculate_temperature_decay_invalid_inputs():
    """Verify exception handling for negative time, invalid alpha, or T_steam below floor."""
    with pytest.raises(ValueError, match="alpha"):
        calculate_temperature_decay(t=1.0, T_steam=200.0, T_res=47.0, alpha=0.0)

    with pytest.raises(ValueError, match="T_steam"):
        calculate_temperature_decay(t=1.0, T_steam=30.0, T_res=47.0, alpha=0.01)

    with pytest.raises(ValueError, match="must be non-negative"):
        calculate_temperature_decay(t=-5.0, T_steam=200.0, T_res=47.0, alpha=0.01)