"""
Handles steam injection heat calculations, reservoir heat absorption,
Andrade viscosity calibrations, dynamic thermal decay (T(t)), and
dynamic safety guardrails.
"""

from typing import Union

import numpy as np

ArrayOrFloat = Union[float, np.ndarray]


def validate_thermal_guardrails(
    temperature_c: float,
    native_res_temp_c: float = 47.0,
    steam_quality: float = None,
    viscosity_cp: float = None,
    temperature_tolerance_c: float = 1e-6,
) -> None:
    """
    Enforces physical safety bounds and dynamic guardrails for thermal operations.

    Args:
        temperature_tolerance_c: Small slack band to absorb floating-point
            overshoot from exponential decay as t -> inf (T(t) asymptotically
            approaches T_res but can dip a hair below it due to round-off).
            Does NOT relax the physical constraint -- it only prevents false
            positives from numerical noise.

    Raises:
        ValueError: If temperature falls below native reservoir temperature
                    (beyond tolerance), viscosity is non-positive, or steam
                    quality is outside [0, 1].
    """
    if temperature_c < native_res_temp_c - temperature_tolerance_c:
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


# ---------------------------------------------------------------------------
# NEW: Dynamic Thermal Decay ODE  --  dT/dt = -alpha * (T - T_res)
# ---------------------------------------------------------------------------

def calculate_dynamic_alpha(
    q_o: float,
    rho_o: float,
    C_po: float,
    q_w: float,
    rho_w: float,
    C_pw: float,
    V_heated: float,
    rho_C_composite: float,
    rate_time_basis: str = "day",
) -> float:
    """
    Computes the dynamic thermal decay coefficient alpha (day^-1) that drives
    reservoir cooling after steam injection stops.

    Formula:
        alpha = (q_o * rho_o * C_po + q_w * rho_w * C_pw) / (V_heated * (rho*C)_composite)

    Unit consistency:
        q_o, q_w are fluid production rates. This function ALWAYS returns
        alpha in units of day^-1 (matching T(t) time input convention used
        throughout this module). Pass `rate_time_basis="second"` if your
        upstream telemetry reports q_o / q_w as per-second rates -- they will
        be internally converted (x 86400) before the alpha calculation so the
        returned alpha stays in day^-1 regardless of the source cadence.

        Dimensional check (day-basis):
            [m^3/day] * [kg/m^3] * [J/(kg.K)]   ->  J/(day.K)
          -------------------------------------
            [m^3] * [J/(m^3.K)]                 ->  J/K
        => alpha units = 1/day.  Consistent.

    Args:
        q_o: Oil production rate (volumetric, per rate_time_basis).
        rho_o: Oil density (kg/m^3).
        C_po: Oil specific heat capacity (J/kg.K).
        q_w: Water production rate (volumetric, per rate_time_basis).
        rho_w: Water density (kg/m^3).
        C_pw: Water specific heat capacity (J/kg.K).
        V_heated: Heated drainage volume (m^3).
        rho_C_composite: Composite volumetric heat capacity (J/m^3.K), i.e.
            the output of calculate_composite_heat_capacity().
        rate_time_basis: "day" (default) or "second" -- cadence of q_o/q_w
            as supplied by the caller.

    Returns:
        float: alpha in day^-1, guaranteed strictly positive.

    Raises:
        ValueError: On non-physical (zero/negative) volumes or heat capacity,
            unknown rate_time_basis, or a non-positive resulting alpha
            (which would indicate the reservoir is not net-cooling given the
            supplied production/absorption inputs -- check sign conventions).
    """
    if rate_time_basis not in ("day", "second"):
        raise ValueError(
            f"rate_time_basis ({rate_time_basis!r}) must be 'day' or 'second'."
        )
    if V_heated <= 0.0:
        raise ValueError(f"Heated drainage volume V_heated ({V_heated}) must be positive.")
    if rho_C_composite <= 0.0:
        raise ValueError(
            f"Composite heat capacity rho_C_composite ({rho_C_composite}) must be positive."
        )

    # Normalize production rates to a per-day basis so alpha is always day^-1.
    SECONDS_PER_DAY = 86400.0  # tunable only if a non-standard day length is ever needed
    if rate_time_basis == "second":
        q_o = q_o * SECONDS_PER_DAY
        q_w = q_w * SECONDS_PER_DAY

    numerator = (q_o * rho_o * C_po) + (q_w * rho_w * C_pw)
    denominator = V_heated * rho_C_composite
    alpha = numerator / denominator

    if alpha <= 0.0:
        raise ValueError(
            f"Computed alpha ({alpha}) must be strictly positive -- reservoir must be "
            "net-cooling (production rates, densities, and heat capacities must all be "
            "non-negative with at least one positive fluid production term)."
        )

    return float(alpha)


def calculate_temperature_decay(
    t: ArrayOrFloat,
    T_steam: float,
    T_res: float,
    alpha: float,
) -> ArrayOrFloat:
    """
    Solves the closed-form thermal decay ODE for wellbore/reservoir
    temperature during the production (Puff) phase.

    ODE:
        dT/dt = -alpha * (T - T_res)
    Closed-form solution:
        T(t) = T_res + (T_steam - T_res) * exp(-alpha * t)

    Vectorized: accepts a scalar t or a 1-D array/list of time steps (days)
    and returns a matching scalar or np.ndarray.

    Args:
        t: Time elapsed since start of production (days). Scalar or 1-D
            array-like. Must be >= 0.
        T_steam: Peak wellbore temperature at t=0 (deg C), i.e. T(0).
        T_res: Native/asymptotic reservoir temperature floor (deg C).
        alpha: Thermal decay coefficient (day^-1), strictly positive --
            typically the output of calculate_dynamic_alpha().

    Returns:
        float or np.ndarray: T(t) in deg C. Guaranteed >= T_res (clamped for
        floating-point overshoot only, never for physical violations).

    Raises:
        ValueError: If alpha <= 0, T_steam < T_res, or any t < 0.
    """
    if alpha <= 0.0:
        raise ValueError(f"Thermal decay coefficient alpha ({alpha}) must be strictly positive.")
    if T_steam < T_res:
        raise ValueError(
            f"T_steam ({T_steam}°C) cannot be below the reservoir floor T_res ({T_res}°C)."
        )

    is_scalar_input = np.isscalar(t) or (isinstance(t, np.ndarray) and t.ndim == 0)
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))

    if np.any(t_arr < 0):
        raise ValueError("Time t (days) must be non-negative.")

    T_t = T_res + (T_steam - T_res) * np.exp(-alpha * t_arr)

    # Clamp only floating-point undershoot below the asymptotic floor
    # (e.g. -1e-13 at very large t) -- never masks a real physical violation
    # since the analytic solution can never cross T_res from above.
    T_t = np.maximum(T_t, T_res)

    if is_scalar_input:
        return float(T_t[0])
    return T_t