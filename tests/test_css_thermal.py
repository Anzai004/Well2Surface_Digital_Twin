"""
Unit tests for CSS Thermal EOR Engine (Gemini Pro/Flash Components).
Validates guardrails, composite heat balance, and Andrade viscosity calibration points.
"""

import pytest
import numpy as np
from src.physics.css_thermal import (
    calculate_steam_enthalpy_injection,
    calculate_composite_heat_capacity,
    calculate_absorbed_heat,
    calculate_andrade_viscosity,
    validate_thermal_guardrails,
)


def test_steam_enthalpy_injection_valid():
    """Verify steam heat injection Q_inj formula output."""
    m_s = 100.0  # kg/s
    x_qual = 0.80
    T_steam = 250.0  # °C
    T_res = 47.0    # °C
    h_fg = 1700.0   # kJ/kg
    C_w = 4.184     # kJ/kg·°C

    Q_inj = calculate_steam_enthalpy_injection(m_s, x_qual, T_steam, T_res, h_fg, C_w)
    
    # Expected: 100 * [0.8 * 1700 + 4.184 * (250 - 47)] = 100 * [1360 + 849.352] = 220935.2
    expected = 100.0 * (0.80 * 1700.0 + 4.184 * (250.0 - 47.0))
    assert np.isclose(Q_inj, expected, rtol=1e-5)


def test_composite_heat_capacity_and_absorption():
    """Verify volumetric heat capacity and reservoir energy absorption."""
    phi = 0.28
    S_o = 0.70
    S_w = 0.30
    rho_r, C_pr = 2650.0, 0.88
    rho_o, C_po = 946.5, 2.00
    rho_w, C_pw = 1000.0, 4.184

    rho_C_comp = calculate_composite_heat_capacity(
        phi, S_o, S_w, rho_r, C_pr, rho_o, C_po, rho_w, C_pw
    )
    assert rho_C_comp > 0.0

    Q_abs = calculate_absorbed_heat(V_zone=1500.0, rho_C_composite=rho_C_comp, delta_T=203.0)
    assert Q_abs == pytest.approx(1500.0 * rho_C_comp * 203.0)


def test_andrade_viscosity_calibration_points():
    """Verify viscosity bounds at Baghewala reservoir baseline and thermal injection peak."""
    A = -14.28
    B = 5820.4

    # Native reservoir temperature baseline (~47°C)
    mu_native = calculate_andrade_viscosity(47.0, A=A, B=B)
    assert 40.0 <= mu_native <= 70.0, f"Expected native viscosity ~66 cP, got {mu_native}"

    # Peak steam temperature (~250°C)
    mu_heated = calculate_andrade_viscosity(250.0, A=A, B=B)
    assert 0.01 <= mu_heated <= 2.0, f"Expected high-temp viscosity ~1 cP order, got {mu_heated}"


def test_thermal_guardrail_exceptions():
    """Ensure invalid operational inputs trigger appropriate safety exceptions."""
    with pytest.raises(ValueError, match="cannot fall below native reservoir temperature"):
        validate_thermal_guardrails(temperature_c=30.0, native_res_temp_c=47.0)

    with pytest.raises(ValueError, match="must be strictly within"):
        calculate_steam_enthalpy_injection(m_s=50.0, x_qual=1.2, T_steam=200.0)

    with pytest.raises(ValueError, match="strictly positive"):
        validate_thermal_guardrails(temperature_c=50.0, viscosity_cp=-5.0)