"""
Handles steam injection heat calculations, reservoir heat absorption,
Andrade viscosity calibrations, and dynamic safety guardrails.
"""

import numpy as np


def validate_thermal_guardrails(
    temperature_c: float,
    native_res_temp_c: float = 47.0,
    steam_quality: float = None,
    viscosity_cp: float = None,
) -> None:
    """
    Enforces physical safety bounds and dynamic guardrails for thermal operations.
    
    Raises:
        ValueError: If temperature falls below native reservoir temperature,
                    viscosity is non-positive, or steam quality is outside [0, 1].
    """
    if temperature_c < native_res_temp_c:
        raise ValueError(
            f"Temperature ({temperature_c}°C) cannot fall below native reservoir "
            f"temperature ({native_res_temp_c}°C)."
        )
    
    if steam_quality is not None and not (0.0 <= steam_quality <= 1.0):
        raise ValueError(
            f"Steam quality x_qual ({steam_quality}) must be strictly within [0.0, 1.0]."
        )
        
    if viscosity_cp is not None and viscosity_cp <= 0.0:
        raise ValueError(
            f"Calculated dynamic viscosity ({viscosity_cp} cP) must be strictly positive."
        )


def calculate_steam_enthalpy_injection(
    m_s: float,
    x_qual: float,
    T_steam: float,
    T_res: float = 47.0,
    h_fg: float = 2000.0,
    C_w: float = 4.184,
) -> float:
    """
    Calculates total heat injected by steam (Q_inj).
    
    Formula:
        Q_inj = m_s * [x_qual * h_fg + C_w * (T_steam - T_res)]
        
    Args:
        m_s: Steam mass injection rate (kg or tons).
        x_qual: Steam dryness fraction / quality [0, 1].
        T_steam: Surface/wellbore steam injection temperature (°C).
        T_res: Native reservoir temperature (°C).
        h_fg: Latent heat of vaporization (kJ/kg).
        C_w: Specific heat capacity of water (kJ/kg·°C).
        
    Returns:
        float: Total heat injected Q_inj in energy units corresponding to h_fg / C_w.
    """
    validate_thermal_guardrails(
        temperature_c=T_steam, native_res_temp_c=T_res, steam_quality=x_qual
    )
    
    Q_inj = m_s * (x_qual * h_fg + C_w * (T_steam - T_res))
    return float(Q_inj)


def calculate_composite_heat_capacity(
    phi: float,
    S_o: float,
    S_w: float,
    rho_r: float,
    C_pr: float,
    rho_o: float,
    C_po: float,
    rho_w: float,
    C_pw: float,
) -> float:
    """
    Calculates volumetric composite heat capacity ((rho * C)_composite) of saturated reservoir rock.
    
    Formula:
        (rho * C)_composite = (1 - phi) * rho_r * C_pr + phi * (S_o * rho_o * C_po + S_w * rho_w * C_pw)
    """
    if not (0.0 <= phi <= 1.0):
        raise ValueError(f"Porosity phi ({phi}) must be within [0, 1].")
    if not np.isclose(S_o + S_w, 1.0, atol=1e-3):
        raise ValueError(f"Fluid saturations S_o ({S_o}) + S_w ({S_w}) must sum to 1.0.")

    matrix_term = (1.0 - phi) * rho_r * C_pr
    fluid_term = phi * (S_o * rho_o * C_po + S_w * rho_w * C_pw)
    
    rho_C_composite = matrix_term + fluid_term
    return float(rho_C_composite)


def calculate_absorbed_heat(
    V_zone: float,
    rho_C_composite: float,
    delta_T: float,
) -> float:
    """
    Calculates composite heat energy absorbed by target reservoir volume (Q_absorbed).
    
    Formula:
        Q_absorbed = V_zone * (rho * C)_composite * delta_T
    """
    if delta_T < 0:
        raise ValueError(f"Delta T ({delta_T}°C) cannot be negative.")
    if V_zone <= 0:
        raise ValueError(f"Target heated zone volume V_zone ({V_zone}) must be positive.")

    return float(V_zone * rho_C_composite * delta_T)


def calculate_andrade_viscosity(
    T_celsius: float,
    A: float = -14.28,
    B: float = 5820.4,
    native_res_temp_c: float = 47.0,
) -> float:
    """
    Evaluates heavy crude oil dynamic viscosity mu(t) using the Andrade equation.
    
    Formula:
        mu(T) = exp(A + B / (T + 273.15))
        
    Args:
        T_celsius: Current fluid temperature in °C.
        A: Calibrated Andrade intercept constant.
        B: Calibrated Andrade activation energy constant (K).
        native_res_temp_c: Base reservoir temperature guardrail threshold.
        
    Returns:
        float: Viscosity mu in centipoise (cP).
    """
    validate_thermal_guardrails(temperature_c=T_celsius, native_res_temp_c=native_res_temp_c)
    
    T_kelvin = T_celsius + 273.15
    viscosity_cp = np.exp(A + (B / T_kelvin))
    
    validate_thermal_guardrails(
        temperature_c=T_celsius,
        native_res_temp_c=native_res_temp_c,
        viscosity_cp=viscosity_cp,
    )
    
    return float(viscosity_cp)