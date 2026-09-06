"""
Multiphase Flow & Thermal-Hydraulic Coupling Module.

Provides functions for wellbore linear temperature interpolation, dynamic
two-phase mixture density calculations, and Beggs & Brill (1973) multiphase
pressure-gradient traverse calculations for the Well2Surface Digital Twin
(Phase 3).
"""

from dataclasses import dataclass
from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]

_GRAVITY_M_S2 = 9.81
_CP_TO_PA_S = 1.0e-3

# Beggs & Brill (1973) horizontal liquid-holdup coefficients: H_L(0) = a * lambda_L^b / N_FR^c
_HORIZONTAL_HOLDUP_COEFFS = {
    "segregated": (0.98, 0.4846, 0.0868),
    "intermittent": (0.845, 0.5351, 0.0173),
    "distributed": (1.065, 0.5824, 0.0609),
}

# Beggs & Brill (1973) inclination-correction coefficients for uphill flow: C = (1-lambda_L)*ln(d*lambda_L^e*N_LV^f*N_FR^g)
_UPHILL_INCLINATION_COEFFS = {
    "segregated": (0.011, -3.768, 3.539, -1.614),
    "intermittent": (2.96, 0.305, -0.4473, 0.0978),
    # "distributed" uphill has no correction: C = 0 (psi = 1)
}

# Downhill flow uses a single coefficient set applied regardless of flow pattern.
_DOWNHILL_INCLINATION_COEFFS = (4.70, -0.3692, 0.1244, -0.5056)

_MIN_HOLDUP = 1.0e-6
_MAX_HOLDUP = 1.0 - 1.0e-6


def couple_wellbore_temperature_profile(
    T_bottomhole_c: float,
    T_wellhead_c: float,
    depth_from_surface_m: np.ndarray,
    L_total_m: float,
) -> np.ndarray:
    """
    Interpolate fluid temperature along the tubing string using a linear gradient approximation.

    Formula:
        T(x) = T_bottomhole - (T_bottomhole - T_wellhead) * (x / L_total)

        where x = depth from surface (m), 0 <= x <= L_total
              T_bottomhole = bottomhole temperature (°C)
              T_wellhead = measured surface temperature (°C)
              L_total = total wellbore depth (m)

    Note:
        This linear gradient is a documented simplifying assumption for the hackathon
        scope. A full Ramey heat-loss ODE formulation is planned as a future upgrade.

    Args:
        T_bottomhole_c (float): Bottomhole fluid temperature in °C from thermal model.
        T_wellhead_c (float): Measured surface wellhead temperature in °C.
        depth_from_surface_m (np.ndarray): 1-D array of depths from surface in meters.
        L_total_m (float): Total wellbore/tubing depth in meters.

    Returns:
        np.ndarray: Interpolated fluid temperature profile in °C at given depths.

    Raises:
        ValueError: If L_total_m <= 0.
        ValueError: If T_bottomhole_c < T_wellhead_c in production mode.
        ValueError: If any depth in depth_from_surface_m is < 0 or > L_total_m.
    """
    if L_total_m <= 0:
        raise ValueError(f"Total wellbore depth L_total_m must be positive, got {L_total_m}.")

    if T_bottomhole_c < T_wellhead_c:
        raise ValueError(
            f"Bottomhole temperature ({T_bottomhole_c}°C) cannot be less than "
            f"wellhead temperature ({T_wellhead_c}°C) in production mode."
        )

    depths = np.asarray(depth_from_surface_m, dtype=float)

    if np.any(depths < 0.0) or np.any(depths > L_total_m):
        raise ValueError(
            f"All depth values must be within [0.0, {L_total_m}] meters. "
            f"Got range [{np.min(depths)}, {np.max(depths)}]."
        )

    return T_bottomhole_c - (T_bottomhole_c - T_wellhead_c) * (depths / L_total_m)


def calculate_dynamic_fluid_density(
    liquid_holdup_fraction: float,
    rho_liquid: float,
    rho_gas: float = 1.2,
) -> float:
    """
    Compute holdup-weighted mixture density from liquid/gas holdup fractions.

    Formula:
        rho_mixture = (H_L * rho_liquid) + ((1 - H_L) * rho_gas)

        where H_L = liquid holdup fraction, 0.0 <= H_L <= 1.0
              rho_liquid = liquid phase density (kg/m^3)
              rho_gas = gas phase density (kg/m^3)

    Note:
        liquid_holdup_fraction is typically the slip liquid holdup H_L(theta) produced
        node-by-node by solve_beggs_brill_pressure_profile() (see its
        ``liquid_holdup`` return field). That function already computes an
        internally-vectorized equivalent of this formula for its own pressure-gradient
        traverse; this scalar function remains available for one-off density lookups
        (e.g. re-evaluating a single node against a different rho_gas assumption).

    Args:
        liquid_holdup_fraction (float): Liquid holdup fraction H_L in [0.0, 1.0].
        rho_liquid (float): Density of liquid phase in kg/m^3.
        rho_gas (float, optional): Density of gas phase in kg/m^3. Defaults to 1.2 kg/m^3.

    Returns:
        float: Calculated dynamic mixture fluid density in kg/m^3.

    Raises:
        ValueError: If liquid_holdup_fraction is not in range [0.0, 1.0].
        ValueError: If rho_liquid <= 0 or rho_gas <= 0.
    """
    if not (0.0 <= liquid_holdup_fraction <= 1.0):
        raise ValueError(
            f"liquid_holdup_fraction must be in range [0.0, 1.0], got {liquid_holdup_fraction}."
        )

    if rho_liquid <= 0:
        raise ValueError(f"rho_liquid must be positive (> 0), got {rho_liquid}.")

    if rho_gas <= 0:
        raise ValueError(f"rho_gas must be positive (> 0), got {rho_gas}.")

    return float((liquid_holdup_fraction * rho_liquid) + ((1.0 - liquid_holdup_fraction) * rho_gas))


@dataclass
class BeggsBrillResult:
    """
    Container for a Beggs & Brill (1973) multiphase pressure-traverse solution.

    Attributes:
        depth_m (np.ndarray): Depths from surface at each node (m), as supplied.
        pressure_pa (np.ndarray): Absolute pressure at each node (Pa).
        liquid_holdup (np.ndarray): Inclination-corrected in-situ liquid holdup H_L(theta)
            at each node, dimensionless in (0.0, 1.0). Feed directly into
            calculate_dynamic_fluid_density() as ``liquid_holdup_fraction`` for
            single-node density lookups.
        flow_pattern (np.ndarray): Flow-regime label at each node, one of
            "segregated", "intermittent", "distributed", "transition".
        mixture_density_kg_m3 (np.ndarray): In-situ (slip) mixture density at each node
            (kg/m^3), i.e. the vectorized equivalent of
            calculate_dynamic_fluid_density(liquid_holdup, rho_liquid, rho_gas).
        pressure_gradient_pa_per_m (np.ndarray): Total pressure gradient at each node
            (Pa/m) = elevation_gradient_pa_per_m + friction_gradient_pa_per_m.
        elevation_gradient_pa_per_m (np.ndarray): Hydrostatic (elevation) component of
            the pressure gradient at each node (Pa/m).
        friction_gradient_pa_per_m (np.ndarray): Frictional component of the pressure
            gradient at each node (Pa/m).
    """

    depth_m: np.ndarray
    pressure_pa: np.ndarray
    liquid_holdup: np.ndarray
    flow_pattern: np.ndarray
    mixture_density_kg_m3: np.ndarray
    pressure_gradient_pa_per_m: np.ndarray
    elevation_gradient_pa_per_m: np.ndarray
    friction_gradient_pa_per_m: np.ndarray


def _classify_flow_pattern(lambda_l: np.ndarray, n_fr: np.ndarray):
    """
    Determine the Beggs & Brill (1973) flow-pattern regime at each node.

    Formula (regime boundary curves in the lambda_L - N_FR plane):
        L1 = 316 * lambda_L^0.302
        L2 = 0.0009252 * lambda_L^(-2.4684)
        L3 = 0.10 * lambda_L^(-1.4516)
        L4 = 0.5 * lambda_L^(-6.738)

        Segregated:   lambda_L < 0.01 and N_FR < L1;  OR lambda_L >= 0.01 and N_FR < L2
        Transition:   lambda_L >= 0.01 and L2 <= N_FR <= L3
        Intermittent: 0.01 <= lambda_L < 0.4 and L3 < N_FR <= L1;
                      OR lambda_L >= 0.4 and L3 < N_FR <= L4
        Distributed:  lambda_L < 0.4 and N_FR >= L1;  OR lambda_L >= 0.4 and N_FR > L4

    Args:
        lambda_l (np.ndarray): No-slip liquid holdup at each node, in (0.0, 1.0).
        n_fr (np.ndarray): Froude number of the mixture at each node (dimensionless).

    Returns:
        tuple: (pattern, l1, l2, l3, l4) where ``pattern`` is a string array with
        values in {"segregated", "transition", "intermittent", "distributed"} and
        l1..l4 are the regime boundary arrays (same shape as lambda_l).
    """
    l1 = 316.0 * lambda_l ** 0.302
    l2 = 0.0009252 * lambda_l ** (-2.4684)
    l3 = 0.10 * lambda_l ** (-1.4516)
    l4 = 0.5 * lambda_l ** (-6.738)

    pattern = np.full(lambda_l.shape, "", dtype="<U12")

    segregated = ((lambda_l < 0.01) & (n_fr < l1)) | ((lambda_l >= 0.01) & (n_fr < l2))
    transition = (lambda_l >= 0.01) & (n_fr >= l2) & (n_fr <= l3)
    intermittent = (
        ((lambda_l >= 0.01) & (lambda_l < 0.4) & (n_fr > l3) & (n_fr <= l1))
        | ((lambda_l >= 0.4) & (n_fr > l3) & (n_fr <= l4))
    )
    distributed = ((lambda_l < 0.4) & (n_fr >= l1)) | ((lambda_l >= 0.4) & (n_fr > l4))

    pattern[segregated] = "segregated"
    pattern[transition] = "transition"
    pattern[intermittent] = "intermittent"
    pattern[distributed] = "distributed"

    return pattern, l1, l2, l3, l4


def _regime_horizontal_holdup(lambda_l: np.ndarray, n_fr: np.ndarray, regime: str) -> np.ndarray:
    """
    Evaluate the horizontal (theta = 0) liquid holdup H_L(0) for a single named regime,
    applied element-wise to every node regardless of that node's actual classified
    pattern (used internally to build per-regime holdup fields for transition blending).

    Formula:
        H_L(0) = a * lambda_L^b / N_FR^c, clipped to the physical bound [lambda_L, 1.0].

    Args:
        lambda_l (np.ndarray): No-slip liquid holdup at each node.
        n_fr (np.ndarray): Mixture Froude number at each node.
        regime (str): One of "segregated", "intermittent", "distributed".

    Returns:
        np.ndarray: Horizontal liquid holdup H_L(0) at each node for this regime.
    """
    a, b, c = _HORIZONTAL_HOLDUP_COEFFS[regime]
    h_l0 = a * (lambda_l ** b) / (n_fr ** c)
    return np.clip(h_l0, lambda_l, 1.0)


def _regime_inclination_psi(
    lambda_l: np.ndarray,
    n_lv: np.ndarray,
    n_fr: np.ndarray,
    theta_rad: np.ndarray,
    regime: str,
    is_uphill_flow: bool,
) -> np.ndarray:
    """
    Evaluate the Beggs & Brill inclination-correction factor psi(theta) for a single
    named regime and flow direction.

    Formula:
        C = max(0, (1 - lambda_L) * ln(d * lambda_L^e * N_LV^f * N_FR^g))
        psi(theta) = 1 + C * [sin(1.8*theta) - (1/3)*sin^3(1.8*theta)]

    Distributed-pattern uphill flow has no correction (C = 0, psi = 1). Downhill flow
    uses a single coefficient set applied to every pattern.

    Args:
        lambda_l (np.ndarray): No-slip liquid holdup at each node.
        n_lv (np.ndarray): Liquid velocity number at each node (dimensionless).
        n_fr (np.ndarray): Mixture Froude number at each node.
        theta_rad (np.ndarray): Magnitude of pipe inclination from horizontal (radians),
            in [0, pi/2].
        regime (str): One of "segregated", "intermittent", "distributed".
        is_uphill_flow (bool): True if the fluid flows toward increasing elevation
            (typical for a producing well); False for downhill (e.g. an injection leg).

    Returns:
        np.ndarray: Inclination-correction factor psi(theta) at each node.
    """
    if is_uphill_flow:
        if regime == "distributed":
            return np.ones_like(lambda_l)
        d, e, f, g = _UPHILL_INCLINATION_COEFFS[regime]
    else:
        d, e, f, g = _DOWNHILL_INCLINATION_COEFFS

    with np.errstate(divide="ignore", invalid="ignore"):
        log_arg = d * (lambda_l ** e) * (n_lv ** f) * (n_fr ** g)
        c_raw = (1.0 - lambda_l) * np.log(log_arg)
    c_val = np.clip(np.nan_to_num(c_raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)

    return 1.0 + c_val * (np.sin(1.8 * theta_rad) - (np.sin(1.8 * theta_rad) ** 3) / 3.0)


def _beggs_brill_liquid_holdup(
    lambda_l: np.ndarray,
    n_lv: np.ndarray,
    n_fr: np.ndarray,
    theta_rad: np.ndarray,
    is_uphill_flow: bool,
):
    """
    Compute the full inclination-corrected Beggs & Brill liquid holdup H_L(theta) and
    the classified flow pattern at each node.

    Transition-regime nodes blend the segregated and intermittent holdups (each with
    their own horizontal-holdup and inclination-correction coefficients) using the
    standard interpolation factor A = (L3 - N_FR) / (L3 - L2).

    Args:
        lambda_l (np.ndarray): No-slip liquid holdup at each node.
        n_lv (np.ndarray): Liquid velocity number at each node.
        n_fr (np.ndarray): Mixture Froude number at each node.
        theta_rad (np.ndarray): Magnitude of pipe inclination from horizontal (radians).
        is_uphill_flow (bool): True for uphill (production) flow, False for downhill.

    Returns:
        tuple: (h_l_theta, pattern) where h_l_theta is the inclination-corrected liquid
        holdup at each node, clipped to [_MIN_HOLDUP, _MAX_HOLDUP], and pattern is the
        flow-regime label array.
    """
    pattern, l1, l2, l3, l4 = _classify_flow_pattern(lambda_l, n_fr)

    h_seg0 = _regime_horizontal_holdup(lambda_l, n_fr, "segregated")
    h_int0 = _regime_horizontal_holdup(lambda_l, n_fr, "intermittent")
    h_dist0 = _regime_horizontal_holdup(lambda_l, n_fr, "distributed")

    psi_seg = _regime_inclination_psi(lambda_l, n_lv, n_fr, theta_rad, "segregated", is_uphill_flow)
    psi_int = _regime_inclination_psi(lambda_l, n_lv, n_fr, theta_rad, "intermittent", is_uphill_flow)
    psi_dist = _regime_inclination_psi(lambda_l, n_lv, n_fr, theta_rad, "distributed", is_uphill_flow)

    h_seg_theta = h_seg0 * psi_seg
    h_int_theta = h_int0 * psi_int
    h_dist_theta = h_dist0 * psi_dist

    with np.errstate(divide="ignore", invalid="ignore"):
        interp_a = np.clip((l3 - n_fr) / (l3 - l2), 0.0, 1.0)
    interp_a = np.nan_to_num(interp_a, nan=0.5)
    h_transition_theta = interp_a * h_seg_theta + (1.0 - interp_a) * h_int_theta

    h_l_theta = np.select(
        [pattern == "segregated", pattern == "intermittent", pattern == "distributed", pattern == "transition"],
        [h_seg_theta, h_int_theta, h_dist_theta, h_transition_theta],
        default=np.nan,
    )

    return np.clip(h_l_theta, _MIN_HOLDUP, _MAX_HOLDUP), pattern


def _two_phase_friction_factor(
    lambda_l: np.ndarray,
    h_l: np.ndarray,
    rho_ns: np.ndarray,
    mu_ns_pa_s: np.ndarray,
    v_m: np.ndarray,
    pipe_id_m: float,
    pipe_roughness_m: float,
) -> np.ndarray:
    """
    Compute the Beggs & Brill two-phase Darcy friction factor at each node.

    Formula:
        Re_ns = rho_ns * v_m * D / mu_ns                (no-slip Reynolds number)
        f_n   = 64 / Re_ns                               for Re_ns < 2000 (laminar)
        f_n   = 0.25 / [log10(eps/(3.7D) + 5.74/Re_ns^0.9)]^2   for Re_ns >= 2000
                                                          (Swamee-Jain, applied as the
                                                          turbulent-regime approximation
                                                          across the 2000-4000
                                                          transitional band too -- a
                                                          documented simplification)
        y = lambda_L / H_L^2
        S = ln(2.2y - 1.2)                               for 1.0 < y < 1.2
        S = ln(y) / [-0.0523 + 3.182 ln(y) - 0.8725 ln(y)^2 + 0.01853 ln(y)^4]  otherwise
        f_tp = f_n * exp(S)

    Args:
        lambda_l (np.ndarray): No-slip liquid holdup at each node.
        h_l (np.ndarray): Inclination-corrected liquid holdup at each node.
        rho_ns (np.ndarray): No-slip mixture density at each node (kg/m^3).
        mu_ns_pa_s (np.ndarray): No-slip mixture viscosity at each node (Pa.s).
        v_m (np.ndarray): Mixture (no-slip) velocity at each node (m/s).
        pipe_id_m (float): Pipe inner diameter (m).
        pipe_roughness_m (float): Absolute pipe roughness (m).

    Returns:
        np.ndarray: Two-phase Darcy friction factor f_tp at each node (dimensionless).
    """
    reynolds_ns = rho_ns * v_m * pipe_id_m / mu_ns_pa_s

    f_n_laminar = 64.0 / reynolds_ns
    with np.errstate(divide="ignore", invalid="ignore"):
        f_n_turbulent = 0.25 / (
            np.log10(pipe_roughness_m / (3.7 * pipe_id_m) + 5.74 / reynolds_ns ** 0.9) ** 2
        )
    f_n = np.where(reynolds_ns < 2000.0, f_n_laminar, f_n_turbulent)

    y = lambda_l / h_l ** 2
    ln_y = np.log(y)
    denom = -0.0523 + 3.182 * ln_y - 0.8725 * ln_y ** 2 + 0.01853 * ln_y ** 4
    with np.errstate(divide="ignore", invalid="ignore"):
        s_main = ln_y / denom
    s_patch = np.log(np.clip(2.2 * y - 1.2, 1.0e-12, None))
    s_factor = np.where((y > 1.0) & (y < 1.2), s_patch, s_main)
    s_factor = np.nan_to_num(s_factor, nan=0.0, posinf=0.0, neginf=0.0)

    return f_n * np.exp(s_factor)


def solve_beggs_brill_pressure_profile(
    q_liquid_m3_s: float,
    q_gas_m3_s: ArrayLike,
    pipe_id_m: float,
    depth_from_surface_m: np.ndarray,
    P_reference_pa: float,
    rho_liquid: float,
    rho_gas: ArrayLike,
    mu_liquid_cp: float,
    mu_gas_cp: float,
    surface_tension_n_m: float,
    inclination_deg: ArrayLike = 90.0,
    pipe_roughness_m: float = 4.6e-5,
    is_uphill_flow: bool = True,
    reference_at_surface: bool = True,
) -> BeggsBrillResult:
    """
    Solve the Beggs & Brill (1973) two-phase pressure-gradient traverse along a
    tubing string and return the resulting pressure profile and liquid holdup.

    Formula references (Beggs, H.D. and Brill, J.P., "A Study of Two-Phase Flow in
    Inclined Pipes", JPT, May 1973):
        lambda_L = V_sl / V_m                            (no-slip liquid holdup)
        N_FR = V_m^2 / (g * D)                            (Froude number)
        N_LV = 1.938 * V_sl * (rho_liquid / sigma)^0.25   (liquid velocity number)
        H_L(theta) = H_L(0) * psi(theta)                  (see _beggs_brill_liquid_holdup)
        rho_s  = H_L(theta) * rho_liquid + (1 - H_L(theta)) * rho_gas   (slip density)
        rho_ns = lambda_L * rho_liquid + (1 - lambda_L) * rho_gas       (no-slip density)
        (dP/dx)_elevation = rho_s * g * sin(theta)
        (dP/dx)_friction   = f_tp * rho_ns * V_m^2 / (2 * D)
        dP/dx = (dP/dx)_elevation + (dP/dx)_friction

    Note (documented simplifying assumptions for the hackathon scope):
        * The acceleration pressure-gradient term is neglected. This is standard
          practice for the low-velocity liquid-dominated flow typical of SRP
          production tubing, but is a simplification versus the full Beggs & Brill
          formulation (dP/dx divided by (1 - E_k)). A future upgrade could add it
          back once local in-situ pressure feeds into E_k iteratively.
        * q_gas_m3_s and rho_gas are taken as the caller-supplied IN-SITU values at
          each node (not expanded from standard conditions via a real-gas EOS along
          the string). Callers modeling strong gas expansion should supply
          depth-varying arrays for both.
        * Re_ns in the 2000-4000 transitional band uses the turbulent (Swamee-Jain)
          friction-factor correlation as an approximation rather than a dedicated
          transitional-flow model.

    Args:
        q_liquid_m3_s (float): In-situ liquid volumetric flow rate (m^3/s). Must be > 0.
        q_gas_m3_s (float or np.ndarray): In-situ gas volumetric flow rate (m^3/s) at
            each node, or a scalar broadcast to every node. Must be >= 0.
        pipe_id_m (float): Tubing inner diameter (m). Must be > 0.
        depth_from_surface_m (np.ndarray): 1-D array of depths from surface (m),
            strictly increasing, length >= 2.
        P_reference_pa (float): Known boundary pressure (Pa). Must be > 0.
        rho_liquid (float): Liquid phase density (kg/m^3), assumed incompressible.
            Must be > 0.
        rho_gas (float or np.ndarray): In-situ gas phase density (kg/m^3) at each
            node, or a scalar broadcast to every node. Must be > 0.
        mu_liquid_cp (float): Liquid phase viscosity (cP). Must be > 0.
        mu_gas_cp (float): Gas phase viscosity (cP). Must be > 0.
        surface_tension_n_m (float): Liquid-gas interfacial tension (N/m). Must be > 0.
        inclination_deg (float or np.ndarray, optional): Magnitude of pipe
            inclination from horizontal (degrees), in [0, 90], at each node or as a
            scalar broadcast to every node. Defaults to 90.0 (vertical).
        pipe_roughness_m (float, optional): Absolute pipe roughness (m).
            Defaults to 4.6e-5 (commercial steel).
        is_uphill_flow (bool, optional): True if fluid flows toward increasing
            elevation (a producing well's Puff/production leg). False for downhill
            flow (e.g. a CSS Huff/injection leg). Defaults to True.
        reference_at_surface (bool, optional): If True, P_reference_pa is the
            pressure at depth_from_surface_m[0] (e.g. known wellhead pressure) and
            the profile is integrated downward. If False, P_reference_pa is the
            pressure at the last node (e.g. known bottomhole pressure) and the
            profile is integrated upward. Defaults to True.

    Returns:
        BeggsBrillResult: Pressure profile, liquid holdup, flow pattern, mixture
        density, and pressure-gradient component arrays at each depth node.

    Raises:
        ValueError: If any scalar input is non-positive where positivity is required,
            if depth_from_surface_m is not strictly increasing or has fewer than 2
            nodes, if inclination_deg is outside [0, 90], or if an array input
            (q_gas_m3_s, rho_gas, inclination_deg) cannot be broadcast to the shape
            of depth_from_surface_m.
    """
    if q_liquid_m3_s <= 0:
        raise ValueError(f"q_liquid_m3_s must be positive (> 0), got {q_liquid_m3_s}.")

    if pipe_id_m <= 0:
        raise ValueError(f"pipe_id_m must be positive (> 0), got {pipe_id_m}.")

    if rho_liquid <= 0:
        raise ValueError(f"rho_liquid must be positive (> 0), got {rho_liquid}.")

    if mu_liquid_cp <= 0:
        raise ValueError(f"mu_liquid_cp must be positive (> 0), got {mu_liquid_cp}.")

    if mu_gas_cp <= 0:
        raise ValueError(f"mu_gas_cp must be positive (> 0), got {mu_gas_cp}.")

    if surface_tension_n_m <= 0:
        raise ValueError(f"surface_tension_n_m must be positive (> 0), got {surface_tension_n_m}.")

    if P_reference_pa <= 0:
        raise ValueError(f"P_reference_pa must be positive (> 0), got {P_reference_pa}.")

    if pipe_roughness_m < 0:
        raise ValueError(f"pipe_roughness_m must be non-negative, got {pipe_roughness_m}.")

    depths = np.asarray(depth_from_surface_m, dtype=float)

    if depths.ndim != 1 or depths.shape[0] < 2:
        raise ValueError(
            f"depth_from_surface_m must be a 1-D array of at least 2 nodes, got shape {depths.shape}."
        )

    if np.any(np.diff(depths) <= 0):
        raise ValueError("depth_from_surface_m must be strictly increasing.")

    try:
        q_gas = np.broadcast_to(np.asarray(q_gas_m3_s, dtype=float), depths.shape).copy()
    except ValueError as exc:
        raise ValueError(
            f"q_gas_m3_s of shape {np.asarray(q_gas_m3_s).shape} cannot be broadcast to "
            f"depth_from_surface_m of shape {depths.shape}."
        ) from exc

    if np.any(q_gas < 0):
        raise ValueError(f"q_gas_m3_s must be non-negative everywhere, got min {np.min(q_gas)}.")

    try:
        rho_gas_arr = np.broadcast_to(np.asarray(rho_gas, dtype=float), depths.shape).copy()
    except ValueError as exc:
        raise ValueError(
            f"rho_gas of shape {np.asarray(rho_gas).shape} cannot be broadcast to "
            f"depth_from_surface_m of shape {depths.shape}."
        ) from exc

    if np.any(rho_gas_arr <= 0):
        raise ValueError(f"rho_gas must be positive everywhere (> 0), got min {np.min(rho_gas_arr)}.")

    try:
        incl_deg = np.broadcast_to(np.asarray(inclination_deg, dtype=float), depths.shape).copy()
    except ValueError as exc:
        raise ValueError(
            f"inclination_deg of shape {np.asarray(inclination_deg).shape} cannot be broadcast to "
            f"depth_from_surface_m of shape {depths.shape}."
        ) from exc

    if np.any(incl_deg < 0.0) or np.any(incl_deg > 90.0):
        raise ValueError(
            f"inclination_deg must be within [0, 90] degrees from horizontal. "
            f"Got range [{np.min(incl_deg)}, {np.max(incl_deg)}]."
        )

    theta_rad = np.radians(incl_deg)

    pipe_area_m2 = np.pi * (pipe_id_m ** 2) / 4.0
    v_sl = np.full(depths.shape, q_liquid_m3_s / pipe_area_m2)
    v_sg = q_gas / pipe_area_m2
    v_m = v_sl + v_sg

    lambda_l = v_sl / v_m
    n_fr = v_m ** 2 / (_GRAVITY_M_S2 * pipe_id_m)
    n_lv = 1.938 * v_sl * (rho_liquid / surface_tension_n_m) ** 0.25

    h_l_theta, pattern = _beggs_brill_liquid_holdup(lambda_l, n_lv, n_fr, theta_rad, is_uphill_flow)

    rho_s = h_l_theta * rho_liquid + (1.0 - h_l_theta) * rho_gas_arr
    rho_ns = lambda_l * rho_liquid + (1.0 - lambda_l) * rho_gas_arr
    mu_ns_pa_s = (lambda_l * mu_liquid_cp + (1.0 - lambda_l) * mu_gas_cp) * _CP_TO_PA_S

    f_tp = _two_phase_friction_factor(lambda_l, h_l_theta, rho_ns, mu_ns_pa_s, v_m, pipe_id_m, pipe_roughness_m)

    elevation_gradient = rho_s * _GRAVITY_M_S2 * np.sin(theta_rad)
    friction_gradient = f_tp * rho_ns * v_m ** 2 / (2.0 * pipe_id_m)
    total_gradient = elevation_gradient + friction_gradient

    segment_lengths = np.diff(depths)
    segment_avg_gradient = 0.5 * (total_gradient[:-1] + total_gradient[1:])
    cumulative_drop = np.concatenate(([0.0], np.cumsum(segment_avg_gradient * segment_lengths)))

    if reference_at_surface:
        pressure_pa = P_reference_pa + cumulative_drop
    else:
        pressure_pa = P_reference_pa - (cumulative_drop[-1] - cumulative_drop)

    return BeggsBrillResult(
        depth_m=depths,
        pressure_pa=pressure_pa,
        liquid_holdup=h_l_theta,
        flow_pattern=pattern,
        mixture_density_kg_m3=rho_s,
        pressure_gradient_pa_per_m=total_gradient,
        elevation_gradient_pa_per_m=elevation_gradient,
        friction_gradient_pa_per_m=friction_gradient,
    )