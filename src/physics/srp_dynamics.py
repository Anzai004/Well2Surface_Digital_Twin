"""
Handles rod mechanics, viscous drag, effective stroke, SPM limits,
dyno card rasterization, and 5-state rule-based diagnostics.
"""

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
) -> tuple[float, bool]:
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
) -> tuple[float, float]:
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
    if mu_cp <= 0:
        raise ValueError("Viscosity must be positive.")
    
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