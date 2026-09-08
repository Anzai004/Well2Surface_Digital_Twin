"""
Handles rod mechanics, viscous drag, effective stroke, SPM limits,
dyno card rasterization, 5-state rule-based diagnostics, and the
1D Gibbs damped wave equation solver for downhole card reconstruction.
"""

from typing import Dict, Tuple

import numpy as np


def calculate_rod_stretch(
    F_load: float,
    L_rod: float,
    A_rod: float,
    E_steel: float = 2.07e11,
) -> float:
    """
    Calculates elastic rod stretch using Hooke's Law.

    Formula:
        delta_L = (F_load * L_rod) / (A_rod * E_steel)
    """
    if A_rod <= 0 or E_steel <= 0:
        raise ValueError("Rod cross-sectional area and Young's modulus must be positive.")
    return float((F_load * L_rod) / (A_rod * E_steel))


def calculate_buoyant_weight(
    W_rod_dry: float,
    rho_fluid: float,
    rho_steel: float = 7850.0,
) -> float:
    """
    Calculates submerged rod weight considering hydrostatic buoyancy.

    Formula:
        W_buoy = W_rod_dry * (1 - rho_fluid / rho_steel)
    """
    if rho_fluid >= rho_steel:
        raise ValueError("Fluid density cannot exceed steel density.")
    return float(W_rod_dry * (1.0 - (rho_fluid / rho_steel)))


def calculate_viscous_drag(
    mu_cp: float,
    v_rod: float,
    L_rod: float,
    D_tubing: float,
    D_rod: float,
) -> float:
    """
    Calculates viscous drag force acting on the rod string inside fluid column.

    Formula:
        F_drag(t) = (2 * pi * mu(t) * v_rod * L_rod) / ln(D_tubing / D_rod)
    """
    if D_tubing <= D_rod:
        raise ValueError("Tubing inner diameter must be strictly greater than rod diameter.")

    # Convert viscosity from cP to Pa·s (1 cP = 1e-3 Pa·s)
    mu_pascal_sec = mu_cp * 1e-3
    numerator = 2.0 * np.pi * mu_pascal_sec * v_rod * L_rod
    denominator = np.log(D_tubing / D_rod)

    return float(numerator / denominator)


def calculate_pprl_and_check_yield(
    W_r: float,
    W_f: float,
    F_drag: float,
    yield_limit_load: float,
) -> Tuple[float, bool]:
    """
    Calculates Peak Surface Rod Load (PPRL) and triggers an interrupt flag if load > 90% yield limit.

    Formula:
        PPRL = W_r + W_f + F_drag
    """
    pprl = W_r + W_f + F_drag
    threshold = 0.90 * yield_limit_load
    yield_interrupt_flag = bool(pprl > threshold)

    return float(pprl), yield_interrupt_flag


def calculate_effective_stroke_and_displacement(
    S_surface: float,
    delta_L: float,
    spm: float,
    D_plunger: float,
) -> Tuple[float, float]:
    """
    Calculates downhole effective stroke length (S_eff) and fluid volumetric displacement.

    Formulas:
        S_eff = S_surface - delta_L
        V_disp (m^3/day) = (pi / 4) * D_plunger^2 * S_eff * spm * 1440
    """
    S_eff = max(0.0, S_surface - delta_L)
    plunger_area = (np.pi / 4.0) * (D_plunger ** 2)
    daily_displacement_m3 = plunger_area * S_eff * spm * 1440.0

    return float(S_eff), float(daily_displacement_m3)


def calculate_spm_safe_floor(
    mu_cp: float,
    rho_steel: float = 7850.0,
    rho_fluid: float = 950.0,
    g: float = 9.81,
    K_factor: float = 1.0e-4,
    spm_floor: float = 2.0,
) -> float:
    """
    Calculates maximum safe strokes per minute (SPM_safe) to prevent rod fall hesitation.

    Formula:
        SPM_safe(t) = K * (rho_steel - rho_fluid) * g / mu(t)
    Clamped to a minimum floor of 2.0 SPM.
    """
    if not np.isfinite(mu_cp) or mu_cp <= 0:
        raise ValueError("Viscosity must be positive and finite.")

    spm_calc = (K_factor * (rho_steel - rho_fluid) * g) / mu_cp
    spm_safe = max(spm_floor, float(spm_calc))
    return spm_safe


def rasterize_dyno_card(
    position_array: np.ndarray,
    load_array: np.ndarray,
    grid_size: int = 224,
) -> np.ndarray:
    """
    Rasterizes continuous downhole/surface dynamometer position-load vectors
    into a 2D matrix tensor (grid_size x grid_size) normalized between [0, 255].
    """
    tensor_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)

    norm_pos = (position_array - np.min(position_array)) / (np.ptp(position_array) + 1e-8)
    norm_load = (load_array - np.min(load_array)) / (np.ptp(load_array) + 1e-8)

    grid_x = np.clip((norm_pos * (grid_size - 1)).astype(int), 0, grid_size - 1)
    grid_y = np.clip(((1.0 - norm_load) * (grid_size - 1)).astype(int), 0, grid_size - 1)

    tensor_grid[grid_y, grid_x] = 255
    return tensor_grid


def classify_dyno_card_5state(
    fillage_pct: float,
    pprl: float,
    yield_limit: float,
    current_spm: float,
    spm_safe: float,
    gas_lock_flag: bool = False,
) -> str:
    """
    5-State Rule-Based Diagnostic Classifier for SRP Operations.
    States:
        1. Normal Operation
        2. Fluid Pound / Low Fillage
        3. Gas Interference / Lock
        4. Mechanical Overload / Rod Stress
        5. Viscous Drag Sucking / High Friction
    """
    if pprl > 0.90 * yield_limit:
        return "Mechanical Overload / Rod Stress"
    if gas_lock_flag:
        return "Gas Interference / Lock"
    if fillage_pct < 70.0:
        return "Fluid Pound / Low Fillage"
    if current_spm > spm_safe:
        return "Viscous Drag Sucking / High Friction"

    return "Normal Operation"


# ---------------------------------------------------------------------------
# 1D Gibbs Damped Wave Equation Solver
#   d^2u/dt^2 = a^2 * d^2u/dx^2 - c(t) * du/dt
# ---------------------------------------------------------------------------

def calculate_wave_speed_and_damping(
    mu_cp: float,
    A_rod: float,
    D_tubing: float,
    D_rod: float,
    rho_steel: float = 7850.0,
    E_steel: float = 2.07e11,
) -> Tuple[float, float]:
    """
    Computes the rod acoustic wave speed `a` and viscous damping factor `c`
    used by the Gibbs equation.

    Formulas:
        a = sqrt(E_steel / rho_steel)
        c = (2 * pi * mu(t)) / (rho_steel * A_rod * ln(D_tubing/D_rod))
    """
    if D_tubing <= D_rod:
        raise ValueError("Tubing inner diameter must be strictly greater than rod diameter.")
    if A_rod <= 0:
        raise ValueError(f"Rod cross-sectional area A_rod ({A_rod}) must be positive.")
    if not np.isfinite(mu_cp) or mu_cp <= 0:
        raise ValueError(
            f"Viscosity mu_cp ({mu_cp}) must be finite and strictly positive -- "
            "infinite/NaN viscosity (e.g. a fully congealed cold-shutdown crude) "
            "is a physical limit, not a valid solver input; callers must clamp to "
            "a large-but-finite viscosity ceiling before calling the wave solver."
        )
    if rho_steel <= 0 or E_steel <= 0:
        raise ValueError("rho_steel and E_steel must be positive.")

    a_wave_speed = float(np.sqrt(E_steel / rho_steel))

    mu_pascal_sec = mu_cp * 1e-3  # cP -> Pa.s
    c_damping = float(
        (2.0 * np.pi * mu_pascal_sec)
        / (rho_steel * A_rod * np.log(D_tubing / D_rod))
    )
    return a_wave_speed, c_damping


def check_cfl_stability(dt: float, dx: float, a_wave_speed: float, enforce: bool = True) -> float:
    """
    Checks the Courant-Friedrichs-Lewy (CFL) stability condition:
        CFL ratio = a * dt / dx <= 1.0
    """
    if dt <= 0 or dx <= 0 or a_wave_speed <= 0:
        raise ValueError("dt, dx, and a_wave_speed must all be strictly positive.")

    cfl_ratio = (a_wave_speed * dt) / dx

    if enforce and cfl_ratio > 1.0:
        max_dt = dx / a_wave_speed
        raise ValueError(
            f"CFL stability violated: a*dt/dx = {cfl_ratio:.4f} > 1.0. "
            f"Reduce dt to <= {max_dt:.6f} s (current dt = {dt} s) or increase "
            f"num_spatial_nodes to raise dx tolerance."
        )
    return float(cfl_ratio)


def solve_gibbs_wave_equation(
    x_surface: np.ndarray,
    dt: float,
    L_rod: float,
    mu_cp: float,
    A_rod: float,
    D_tubing: float,
    D_rod: float,
    rho_steel: float = 7850.0,
    E_steel: float = 2.07e11,
    num_spatial_nodes: int = 50,
    enforce_cfl: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Solves the 1D damped Gibbs wave equation to reconstruct downhole rod
    displacement u(L, t) and plunger load F_plunger(t) from a measured
    surface position time series x_surface(t).
    """
    if num_spatial_nodes < 3:
        raise ValueError(f"num_spatial_nodes ({num_spatial_nodes}) must be >= 3.")
    if L_rod <= 0:
        raise ValueError(f"L_rod ({L_rod}) must be positive.")
    if dt <= 0:
        raise ValueError(f"dt ({dt}) must be positive.")

    x_surface = np.asarray(x_surface, dtype=float)
    if x_surface.ndim != 1 or x_surface.shape[0] < 2:
        raise ValueError("x_surface must be a 1-D array with at least 2 time samples.")

    num_time_steps = x_surface.shape[0]

    a_wave_speed, c_damping = calculate_wave_speed_and_damping(
        mu_cp=mu_cp,
        A_rod=A_rod,
        D_tubing=D_tubing,
        D_rod=D_rod,
        rho_steel=rho_steel,
        E_steel=E_steel,
    )

    dx = L_rod / (num_spatial_nodes - 1)
    cfl_ratio = check_cfl_stability(dt=dt, dx=dx, a_wave_speed=a_wave_speed, enforce=enforce_cfl)

    r2 = (a_wave_speed * dt / dx) ** 2
    one_minus_c_dt = 1.0 - c_damping * dt
    one_plus_c_dt = 1.0 + c_damping * dt

    u = np.zeros((num_time_steps, num_spatial_nodes), dtype=float)

    # Initial conditions: rod string starting at rest
    u[0, :] = x_surface[0]
    u[1, :] = x_surface[0]
    u[1, 0] = x_surface[1] if num_time_steps > 1 else x_surface[0]

    last = num_spatial_nodes - 1

    for j in range(1, num_time_steps - 1):
        u_j = u[j, :]
        u_jm1 = u[j - 1, :]

        # Interior nodes (i = 1 .. last-1)
        laplacian = np.empty(num_spatial_nodes, dtype=float)
        laplacian[1:last] = u_j[2:last + 1] - 2.0 * u_j[1:last] + u_j[0:last - 1]

        # Free-end Neumann BC via ghost node at x=L: laplacian = 2*(u[last-1] - u[last])
        laplacian[last] = 2.0 * (u_j[last - 1] - u_j[last])

        u_next = (
            2.0 * u_j - u_jm1 * one_minus_c_dt + r2 * laplacian
        ) / one_plus_c_dt

        # Top boundary: measured surface trajectory
        u_next[0] = x_surface[j + 1]
        u[j + 1, :] = u_next

    u_downhole = u[:, last]
    F_plunger = E_steel * A_rod * (u[:, last] - u[:, last - 1]) / dx

    return {
        "u_downhole": u_downhole,
        "F_plunger": F_plunger,
        "u_grid": u,
        "dx": dx,
        "a_wave_speed": a_wave_speed,
        "c_damping": c_damping,
        "cfl_ratio": cfl_ratio,
    }