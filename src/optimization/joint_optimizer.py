"""
Integrated Well2Surface Optimizer (Phase 4).

Coordinates the CSS thermal engine (src.physics.css_thermal), the SRP
mechanical/wave engine (src.physics.srp_dynamics), and the multiphase
wellbore transport engine (src.physics.multiphase_flow) into a single
forward cycle simulator, then runs a constrained 4-variable non-convex
optimization (steam mass m_s, soak time t_soak, pump speed SPM, surface
stroke length S) that maximizes the FR-4.3 lifecycle economic objective J
subject to the SRP no-float (SPM <= SPMsafe) and rod-yield (PPRL <=
threshold) safety constraints.

Integration & Modeling Assumptions (documented simplifications for the
hackathon scope -- consistent with the "documented simplifying assumption"
style already used in css_thermal.py / multiphase_flow.py):

0. UNIT MISMATCH FOUND BETWEEN PHASE 1 DEFAULTS AND field_params.yaml:
   css_thermal.calculate_steam_enthalpy_injection's own default constants
   (h_fg=2000.0, C_w=4.184) are kJ-scale -- h_fg reads as an approximate
   steam-table kJ/kg figure, and C_w=4.184 is water's familiar specific
   heat in kJ/(kg.K). But config/field_params.yaml's thermal_capacities
   are explicitly J-scale (keys are literally named "*_j_kg_c", e.g.
   water_specific_heat_j_kg_c=4184.0, i.e. 4184 J/(kg.K), a factor of
   1000 larger). calculate_composite_heat_capacity() is built entirely
   from those J-scale values, so if Q_inj is computed with css_thermal's
   own kJ-scale defaults and then divided by that J-scale
   (rho*C)_composite, delta_T comes out 1000x too small (verified: with
   the module's own defaults, m_s=20,000 kg of steam raises the
   reservoir by ~0.014 C -- not physically credible). This module
   therefore does NOT use css_thermal's own h_fg/C_w defaults: it
   requires sim_config to supply J-consistent values explicitly (see
   default_sim_config(), which sets C_w_steam = field_params' own
   water_specific_heat_j_kg_c for guaranteed consistency, and h_fg to an
   approximate J/kg steam-table figure). Flag this to whoever owns
   css_thermal.py -- any other Phase-4-style caller of
   calculate_steam_enthalpy_injection that relies on its bare defaults
   alongside these J-scale field params will hit the same bug.

1. Steam energy balance (Huff+Soak -> T_peak): the peak reservoir
   temperature reached before production begins is derived by an energy
   balance, delta_T = Q_inj / (V_zone * (rho*C)_composite), i.e. all
   injected enthalpy Q_inj (css_thermal.calculate_steam_enthalpy_injection)
   is assumed to raise the heated zone by delta_T with no loss term. A
   real system would apply a thermal-efficiency factor < 1; none is
   specified in the provided config, so none is assumed.

2. t_soak does not itself change T_peak in this model (the provided
   thermal-decay ODE only cools once fluid is produced -- see
   css_thermal.calculate_dynamic_alpha, which requires nonzero production
   to return a positive alpha). t_soak instead enters the objective purely
   as non-productive cycle time: total cycle length = t_soak + t_puff,
   and the reported per-day economics are normalized by that total, so
   an optimizer that ignores soak-time cost would systematically over-soak.

3. calculate_andrade_viscosity() and calculate_pprl_and_check_yield() in
   srp_dynamics.py branch on Python `if`/`bool()` of their scalar inputs
   and are therefore NOT safe to call with array arguments (this was
   confirmed by inspecting the provided source, not assumed) -- this
   module calls them once per time node in an explicit per-node loop
   over the puff-phase time grid rather than vectorizing through them.

4. Rod velocity is approximated as the average linear speed
   v_rod = 2 * S * (SPM / 60) m/s (each stroke cycle covers the surface
   stroke length twice -- once down, once up -- SPM times per minute).
   The Gibbs 1D wave solver (solve_gibbs_wave_equation) is NOT invoked
   here: it operates on a measured surface position waveform x_surface(t)
   at telemetry rate and is designed for real-time downhole card
   reconstruction, not for a scalar per-cycle economic search. Skipping
   it here (rather than fabricating a synthetic waveform) is a deliberate
   scope decision for Phase 4; it remains available for Phase 5 dashboard
   use exactly as implemented in Phase 2.

5. Fluid load on the plunger W_f = rho_fluid * g * A_plunger * L_rod (the
   weight of the produced fluid column supported by the traveling valve
   during the upstroke) is a standard sucker-rod design quantity that is
   NOT one of the functions provided in srp_dynamics.py, so it is
   implemented locally in this module rather than assumed to exist there.

6. The multiphase engine (solve_beggs_brill_pressure_profile) is
   evaluated ONCE per candidate cycle at representative (time-averaged)
   viscosity and a fixed produced water-cut, purely liquid (q_gas = 0,
   consistent with typical SRP liquid-dominated lift -- see the "hackathon
   scope" note already in multiphase_flow.py), to obtain a physically
   grounded in-situ mixture density used as rho_fluid for buoyancy and
   SPMsafe, in place of srp_dynamics' arbitrary rho_fluid=950 default.
   It is not re-solved at every time node.

7. Oil/water split of pumped displacement uses a fixed produced water-cut
   fraction (sim_config["water_cut"]), not a full material-balance model.

8. Motor electrical power is approximated as
   P_motor = PPRL * v_rod / (1000 * motor_efficiency) kW -- the standard
   "polished-rod horsepower" style approximation -- since no motor-power
   function is provided in any module.

9. Config note: config/operational_limits.yaml defines SPM and stroke
   bounds (srp_limits.min_spm/max_spm,
   srp_limits.min_surface_stroke_length_m/max_surface_stroke_length_m) but
   defines NO bounds for steam mass m_s or soak time t_soak. Rather than
   inventing them silently, optimize_css_srp_cycle REQUIRES the caller to
   pass m_s_bounds and t_soak_bounds explicitly.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution, minimize

from src.physics import css_thermal as ct
from src.physics import multiphase_flow as mf
from src.physics import srp_dynamics as sd
from src.optimization.economic_objective import evaluate_css_cycle_economics

_GRAVITY_M_S2 = 9.81
_PSI_TO_PA = 6894.757293168
_M3_TO_BBL = 1.0 / 0.158987294928
_SECONDS_PER_DAY = 86400.0


class InfeasibleCycleError(ValueError):
    """Raised internally when a candidate (m_s, t_soak, SPM, S) violates a
    hard physical bound (e.g. T_peak above the steam cap) before any SRP/
    economics evaluation is attempted. Caught by the optimizer wrappers and
    converted into a large penalty rather than propagated."""


def _fluid_load_n(rho_fluid: float, D_plunger_m: float, L_rod: float, g: float = _GRAVITY_M_S2) -> float:
    """
    Standard SRP fluid load on the plunger: the weight of the produced
    fluid column supported above the plunger during the upstroke.

    Formula:
        W_f = rho_fluid * g * A_plunger * L_rod,  A_plunger = (pi/4) * D_plunger^2

    Not provided in srp_dynamics.py -- implemented here directly (see
    module docstring, assumption 5).
    """
    if D_plunger_m <= 0:
        raise ValueError(f"D_plunger_m ({D_plunger_m}) must be positive.")
    A_plunger = (np.pi / 4.0) * (D_plunger_m ** 2)
    return float(rho_fluid * g * A_plunger * L_rod)


def _rod_dry_weight_n(rho_steel: float, A_rod: float, L_rod: float, g: float = _GRAVITY_M_S2) -> float:
    """Dry (in-air) weight of the full sucker-rod string, W_rod_dry = rho_steel * A_rod * L_rod * g."""
    return float(rho_steel * A_rod * L_rod * g)


@dataclass
class CycleSimulationResult:
    """
    Full forward-simulation trace for one candidate (m_s, t_soak, SPM, S)
    CSS+SRP cycle, integrating Phase 1 (thermal), Phase 2 (mechanical), and
    Phase 3 (multiphase) engines.

    Attributes:
        m_s, t_soak, spm, S: The four decision-variable values simulated.
        T_peak_c: Peak reservoir temperature reached after Huff+Soak (deg C).
        alpha_day: Thermal decay coefficient used for the puff phase (day^-1).
        rho_fluid_kg_m3: Representative in-situ mixture density from the
            Beggs & Brill traverse, used for buoyancy/SPMsafe.
        t_days: Puff-phase time grid (days), shape (n,).
        T_c, mu_cp: Temperature and viscosity at each node in t_days.
        v_rod_m_s: Average rod linear speed (constant across the cycle).
        F_drag_n, pprl_n, delta_L_m, S_eff_m, q_o_bbl_day, sigma_rod_psi,
            P_motor_kw, spm_safe: Per-node mechanical/production series,
            each shape (n,).
        yield_limit_n: Force-equivalent yield threshold used for the PPRL check.
        steam_volume_cumulative_m3: Cumulative injected steam volume series
            (constant after t=0 -- steam is injected only during Huff).
        economics: EconomicCycleResult for this cycle (see economic_objective.py).
        max_pprl_n: Peak PPRL over the puff phase (N).
        min_spm_safe: Minimum SPMsafe over the puff phase (the binding value
            for the SPM <= SPMsafe(t) constraint).
        total_cycle_days: t_soak + t_days[-1] (Huff duration is not modeled
            as a separate time cost here -- see assumption 2).
        J_per_day: economics.J_total / total_cycle_days.
    """

    m_s: float
    t_soak: float
    spm: float
    S: float
    T_peak_c: float
    alpha_day: float
    rho_fluid_kg_m3: float
    t_days: np.ndarray
    T_c: np.ndarray
    mu_cp: np.ndarray
    v_rod_m_s: float
    F_drag_n: np.ndarray
    pprl_n: np.ndarray
    delta_L_m: np.ndarray
    S_eff_m: np.ndarray
    q_o_bbl_day: np.ndarray
    sigma_rod_psi: np.ndarray
    P_motor_kw: np.ndarray
    spm_safe: np.ndarray
    yield_limit_n: float
    steam_volume_cumulative_m3: np.ndarray
    economics: object
    max_pprl_n: float
    min_spm_safe: float
    total_cycle_days: float
    J_per_day: float


def simulate_css_srp_cycle(
    m_s: float,
    t_soak: float,
    spm: float,
    S: float,
    field_params: dict,
    operational_limits: dict,
    sim_config: dict,
) -> CycleSimulationResult:
    """
    Forward-simulates one CSS+SRP cycle for a candidate decision vector and
    evaluates its lifecycle economics. See module docstring for the full
    list of coupling/integration assumptions.

    Args:
        m_s: Steam mass injected during Huff (kg). Must be positive.
        t_soak: Soak duration (days). Must be non-negative.
        spm: Surface pump speed (strokes/min). Must be positive.
        S: Surface stroke length (m). Must be positive.
        field_params: Parsed contents of config/field_params.yaml.
        operational_limits: Parsed contents of config/operational_limits.yaml.
        sim_config: Fixed exogenous scenario assumptions not covered by the
            two config files (steam quality, boiler temperature, plunger
            diameter, water cut, puff-phase evaluation horizon, etc). See
            `default_sim_config()` for the required keys and their meaning.

    Returns:
        CycleSimulationResult: Full time-series trace and summary economics.

    Raises:
        ValueError: If any of m_s, t_soak, spm, S is non-physical (m_s<=0,
            t_soak<0, spm<=0, S<=0).
        InfeasibleCycleError: If the resulting peak temperature exceeds the
            configured steam injection cap (steam_limits.max_injection_temp_c),
            or if steam quality / other guardrails in the underlying physics
            modules are violated for this candidate.
    """
    if m_s <= 0:
        raise ValueError(f"m_s ({m_s}) must be positive.")
    if t_soak < 0:
        raise ValueError(f"t_soak ({t_soak}) cannot be negative.")
    if spm <= 0:
        raise ValueError(f"spm ({spm}) must be positive.")
    if S <= 0:
        raise ValueError(f"S ({S}) must be positive.")

    reservoir = field_params["reservoir"]
    pvt = field_params["pvt_fluid"]
    thermal = field_params["thermal_capacities"]
    geom = field_params["wellbore_geometry"]

    T_res = float(reservoir["native_temperature_c"])
    V_zone = float(reservoir["heated_zone_volume_m3"])
    phi = float(reservoir["porosity_fraction"])
    S_o = float(reservoir["initial_oil_saturation"])
    S_w = float(reservoir["initial_water_saturation"])

    rho_o = float(pvt["oil_density_kg_m3"])
    rho_w_pvt = float(pvt["water_density_kg_m3"])
    A_andrade = float(pvt["andrade_constant_a"])
    B_andrade = float(pvt["andrade_constant_b"])

    rho_r = float(thermal["rock_matrix_density_kg_m3"])
    C_pr = float(thermal["rock_specific_heat_j_kg_c"])
    C_po = float(thermal["oil_specific_heat_j_kg_c"])
    C_pw = float(thermal["water_specific_heat_j_kg_c"])

    L_rod = float(geom["depth_m"])
    D_tubing = float(geom["tubing_inner_diameter_m"])
    D_rod = float(geom["rod_outer_diameter_m"])
    A_rod = float(geom["rod_cross_section_m2"])
    rho_steel = float(geom["steel_density_kg_m3"])
    E_steel = float(geom["steel_youngs_modulus_pa"])

    steam_limits = operational_limits["steam_limits"]
    max_injection_temp_c = float(steam_limits["max_injection_temp_c"])
    max_rod_stress_psi = float(operational_limits["srp_limits"]["max_rod_stress_psi"])

    x_qual = float(sim_config["x_qual"])
    T_boiler_c = float(sim_config["T_boiler_c"])
    # IMPORTANT UNIT FIX (see module docstring, assumption 0): css_thermal's
    # own defaults for calculate_steam_enthalpy_injection (h_fg=2000.0,
    # C_w=4.184) are kJ-scale (h_fg ~ steam-table kJ/kg, C_w = water's
    # familiar 4.184 kJ/(kg.K)), but calculate_composite_heat_capacity is
    # fed J-scale specific heats from field_params.yaml (keys literally
    # named "*_j_kg_c", e.g. water_specific_heat_j_kg_c=4184.0). Dividing a
    # kJ-scale Q_inj by a J-scale rho_C_composite silently suppresses
    # delta_T by 1000x. sim_config must therefore supply J-consistent
    # h_fg/C_w_steam explicitly; this module does NOT fall back to
    # css_thermal's own (kJ-scale) defaults.
    h_fg = float(sim_config["h_fg"])
    C_w_steam = float(sim_config["C_w_steam"])
    water_cut = float(sim_config["water_cut"])
    D_plunger_m = float(sim_config["D_plunger_m"])
    puff_duration_days = float(sim_config["puff_duration_days"])
    n_time_nodes = int(sim_config["n_time_nodes"])
    motor_efficiency = float(sim_config["motor_efficiency"])
    spm_safe_K_factor = float(sim_config.get("spm_safe_K_factor", 1.0e-4))
    wellhead_pressure_pa = float(sim_config["wellhead_pressure_pa"])
    surface_tension_n_m = float(sim_config["surface_tension_n_m"])
    mu_gas_cp = float(sim_config.get("mu_gas_cp", 0.015))
    rho_gas_kg_m3 = float(sim_config.get("rho_gas_kg_m3", 1.2))

    # --- Stage 1 (Phase 1): Huff steam energy balance -> T_peak ---------
    Q_inj = ct.calculate_steam_enthalpy_injection(
        m_s=m_s, x_qual=x_qual, T_steam=T_boiler_c, T_res=T_res, h_fg=h_fg, C_w=C_w_steam
    )
    rho_C_composite = ct.calculate_composite_heat_capacity(
        phi=phi, S_o=S_o, S_w=S_w, rho_r=rho_r, C_pr=C_pr,
        rho_o=rho_o, C_po=C_po, rho_w=rho_w_pvt, C_pw=C_pw,
    )
    delta_T = Q_inj / (V_zone * rho_C_composite)
    T_peak = T_res + delta_T

    if T_peak > max_injection_temp_c:
        raise InfeasibleCycleError(
            f"Achieved T_peak ({T_peak:.2f} C) exceeds steam_limits.max_injection_temp_c "
            f"({max_injection_temp_c} C) for m_s={m_s}."
        )

    # --- Representative production-rate estimate (drives alpha) ---------
    # Optimistic first-pass estimate at delta_L=0, T_peak (see assumption 2/3).
    delta_L_guess = 0.0
    S_eff0, disp0_m3_day = sd.calculate_effective_stroke_and_displacement(
        S_surface=S, delta_L=delta_L_guess, spm=spm, D_plunger=D_plunger_m
    )
    q_o_guess_m3_day = disp0_m3_day * (1.0 - water_cut)
    q_w_guess_m3_day = disp0_m3_day * water_cut

    alpha_day = ct.calculate_dynamic_alpha(
        q_o=q_o_guess_m3_day, rho_o=rho_o, C_po=C_po,
        q_w=q_w_guess_m3_day, rho_w=rho_w_pvt, C_pw=C_pw,
        V_heated=V_zone, rho_C_composite=rho_C_composite, rate_time_basis="day",
    )

    # --- Stage 2/3 (Phase 1): puff-phase T(t), mu(t) ---------------------
    t_days = np.linspace(0.0, puff_duration_days, n_time_nodes)
    T_c = ct.calculate_temperature_decay(t_days, T_steam=T_peak, T_res=T_res, alpha=alpha_day)

    # calculate_andrade_viscosity() branches on a scalar `if`, so it cannot
    # be called with an array (see module docstring, assumption 3).
    mu_cp = np.array([
        ct.calculate_andrade_viscosity(float(T), A=A_andrade, B=B_andrade, native_res_temp_c=T_res)
        for T in T_c
    ])

    # --- Stage 3 (Phase 3): representative multiphase mixture density ---
    rho_liquid_blend = (1.0 - water_cut) * rho_o + water_cut * rho_w_pvt
    mu_liquid_cp_avg = float(np.mean(mu_cp))
    q_liquid_m3_s = max(disp0_m3_day, 1.0e-6) / _SECONDS_PER_DAY

    bb_result = mf.solve_beggs_brill_pressure_profile(
        q_liquid_m3_s=q_liquid_m3_s,
        q_gas_m3_s=0.0,
        pipe_id_m=D_tubing,
        depth_from_surface_m=np.array([0.0, L_rod]),
        P_reference_pa=wellhead_pressure_pa,
        rho_liquid=rho_liquid_blend,
        rho_gas=rho_gas_kg_m3,
        mu_liquid_cp=mu_liquid_cp_avg,
        mu_gas_cp=mu_gas_cp,
        surface_tension_n_m=surface_tension_n_m,
        inclination_deg=90.0,
        is_uphill_flow=True,
        reference_at_surface=True,
    )
    rho_fluid = float(np.mean(bb_result.mixture_density_kg_m3))

    # --- Stage 4 (Phase 2): per-node mechanics --------------------------
    v_rod = 2.0 * S * (spm / 60.0)  # m/s, average linear rod speed

    W_rod_dry = _rod_dry_weight_n(rho_steel, A_rod, L_rod)
    W_buoy = sd.calculate_buoyant_weight(W_rod_dry, rho_fluid, rho_steel)
    W_f = _fluid_load_n(rho_fluid, D_plunger_m, L_rod)

    # yield_limit_load is scaled so that calculate_pprl_and_check_yield's
    # internal 0.90x threshold reproduces max_rod_stress_psi exactly (that
    # config value is documented as ALREADY being 90% of the Grade D yield
    # limit -- see config/operational_limits.yaml comment).
    yield_limit_n = (max_rod_stress_psi / 0.90) * _PSI_TO_PA * A_rod

    n = t_days.shape[0]
    F_drag_n = np.empty(n)
    pprl_n = np.empty(n)
    delta_L_m = np.empty(n)
    S_eff_m = np.empty(n)
    disp_m3_day = np.empty(n)
    spm_safe_arr = np.empty(n)

    for i in range(n):
        F_drag_n[i] = sd.calculate_viscous_drag(
            mu_cp=float(mu_cp[i]), v_rod=v_rod, L_rod=L_rod, D_tubing=D_tubing, D_rod=D_rod
        )
        pprl_n[i], _ = sd.calculate_pprl_and_check_yield(
            W_r=W_buoy, W_f=W_f, F_drag=F_drag_n[i], yield_limit_load=yield_limit_n
        )
        delta_L_m[i] = sd.calculate_rod_stretch(F_load=pprl_n[i], L_rod=L_rod, A_rod=A_rod, E_steel=E_steel)
        S_eff_m[i], disp_m3_day[i] = sd.calculate_effective_stroke_and_displacement(
            S_surface=S, delta_L=delta_L_m[i], spm=spm, D_plunger=D_plunger_m
        )
        spm_safe_arr[i] = sd.calculate_spm_safe_floor(
            mu_cp=float(mu_cp[i]), rho_steel=rho_steel, rho_fluid=rho_fluid,
            g=_GRAVITY_M_S2, K_factor=spm_safe_K_factor,
            spm_floor=float(operational_limits["srp_limits"]["min_spm"]),
        )

    q_o_m3_day = disp_m3_day * (1.0 - water_cut)
    q_o_bbl_day = q_o_m3_day * _M3_TO_BBL
    sigma_rod_psi = (pprl_n / A_rod) / _PSI_TO_PA
    P_motor_kw = (pprl_n * v_rod) / (1000.0 * motor_efficiency)

    steam_volume_cumulative_m3 = np.full(n, m_s / rho_w_pvt)  # steam mass -> liquid-equivalent volume

    economics_cfg = operational_limits["economics"]
    P_oil = float(economics_cfg["oil_price_usd_bbl"])
    C_elec = float(economics_cfg["electricity_cost_usd_kwh"]) * 24.0  # USD/(kW.day)
    C_wear = float(economics_cfg["equipment_wear_penalty_factor"])
    C_steam = float(economics_cfg["steam_cost_usd_ton"]) / 1000.0  # USD/kg

    economics = evaluate_css_cycle_economics(
        t=t_days, q_o=q_o_bbl_day, P_motor=P_motor_kw, sigma_rod=sigma_rod_psi,
        steam_volume_cumulative=steam_volume_cumulative_m3,
        m_s=m_s, P_oil=P_oil, C_elec=C_elec, C_wear=C_wear, C_steam=C_steam,
        sor_threshold=sim_config.get("sor_threshold"),
    )

    total_cycle_days = t_soak + float(t_days[-1])
    J_per_day = economics.J_total / total_cycle_days if total_cycle_days > 0 else economics.J_total

    return CycleSimulationResult(
        m_s=m_s, t_soak=t_soak, spm=spm, S=S,
        T_peak_c=T_peak, alpha_day=alpha_day, rho_fluid_kg_m3=rho_fluid,
        t_days=t_days, T_c=T_c, mu_cp=mu_cp, v_rod_m_s=v_rod,
        F_drag_n=F_drag_n, pprl_n=pprl_n, delta_L_m=delta_L_m, S_eff_m=S_eff_m,
        q_o_bbl_day=q_o_bbl_day, sigma_rod_psi=sigma_rod_psi, P_motor_kw=P_motor_kw,
        spm_safe=spm_safe_arr, yield_limit_n=yield_limit_n,
        steam_volume_cumulative_m3=steam_volume_cumulative_m3,
        economics=economics, max_pprl_n=float(np.max(pprl_n)),
        min_spm_safe=float(np.min(spm_safe_arr)), total_cycle_days=total_cycle_days,
        J_per_day=float(J_per_day),
    )


def default_sim_config(field_params: Optional[dict] = None) -> Dict[str, float]:
    """
    Returns a documented set of default exogenous scenario assumptions for
    simulate_css_srp_cycle(), covering everything needed that is not
    present in field_params.yaml or operational_limits.yaml.

    Args:
        field_params: If supplied, C_w_steam is set to
            field_params["thermal_capacities"]["water_specific_heat_j_kg_c"]
            so it is guaranteed J-unit-consistent with
            calculate_composite_heat_capacity() (see module docstring,
            assumption 0). If None, falls back to the literal value 4184.0.

    NOTE: h_fg, D_plunger_m, water_cut's exact value, surface_tension_n_m,
    and wellhead_pressure_pa are not specified anywhere in the provided
    config or PDFs. h_fg=1,716,000 J/kg is an approximate steam-table
    latent-heat value at ~250 C saturation (~3.97 MPa, comfortably under
    the 12 MPa boiler cap) -- confirm against real Baghewala boiler
    conditions. wellhead_pressure_pa is taken from the TRD's sample
    telemetry payload, "wellhead_pressure_psi": 165.2. Callers should
    override all flagged values with real field values where available.
    """
    C_w_steam = 4184.0
    if field_params is not None:
        C_w_steam = float(field_params["thermal_capacities"]["water_specific_heat_j_kg_c"])

    return {
        "x_qual": 0.80,
        "T_boiler_c": 250.0,
        "h_fg": 1_716_000.0,  # J/kg -- approximate steam-table value, confirm against real PVT/boiler data
        "C_w_steam": C_w_steam,
        "water_cut": 0.30,  # matches field_params initial_water_saturation
        "D_plunger_m": 0.044,  # NOT in provided config -- confirm real value
        "puff_duration_days": 30.0,
        "n_time_nodes": 30,
        "motor_efficiency": 0.85,
        "spm_safe_K_factor": 1.0e-4,
        "wellhead_pressure_pa": 165.2 * _PSI_TO_PA,  # from TRD sample telemetry
        "surface_tension_n_m": 0.02,  # NOT in provided config -- confirm real value
        "mu_gas_cp": 0.015,
        "rho_gas_kg_m3": 1.2,
        "sor_threshold": None,
    }


@dataclass
class OptimizationResult:
    """
    Result of optimize_css_srp_cycle().

    Attributes:
        m_s, t_soak, spm, S: Optimized decision variables.
        J_total: Full-cycle net profit at the optimum (USD).
        J_per_day: J_total normalized by total cycle time (USD/day) --
            this is the actual quantity maximized (see module docstring,
            assumption 2).
        cycle_result: Full CycleSimulationResult at the optimum.
        de_success: differential_evolution convergence flag.
        de_message: differential_evolution termination message.
        de_nfev: Number of objective evaluations used by differential_evolution.
        slsqp_success: SLSQP polishing-step convergence flag.
        slsqp_message: SLSQP termination message.
        slsqp_nit: Number of SLSQP iterations.
        constraints_satisfied: True if both SPM<=SPMsafe(t) and
            PPRL<=yield_limit hold at the returned optimum.
    """

    m_s: float
    t_soak: float
    spm: float
    S: float
    J_total: float
    J_per_day: float
    cycle_result: CycleSimulationResult
    de_success: bool
    de_message: str
    de_nfev: int
    slsqp_success: bool
    slsqp_message: str
    slsqp_nit: int
    constraints_satisfied: bool


_PENALTY = 1.0e12


_SPM_FEASIBILITY_TOL = 1.0e-4
_PPRL_FEASIBILITY_TOL_N = 1.0


def _is_feasible(result: Optional[CycleSimulationResult], spm: float) -> bool:
    """Feasibility check with a small numerical tolerance (see module notes
    on the SPM lower bound coinciding exactly with the SPMsafe hard floor,
    which otherwise produces spurious sub-1e-5 'violations' from floating
    point noise at that shared boundary)."""
    if result is None:
        return False
    return (
        spm <= result.min_spm_safe + _SPM_FEASIBILITY_TOL
        and result.max_pprl_n <= result.yield_limit_n + _PPRL_FEASIBILITY_TOL_N
    )


def _make_evaluators(
    field_params: dict, operational_limits: dict, sim_config: dict
) -> Tuple[Callable, Callable, Callable]:
    """
    Builds (objective, constraint_spm, constraint_pprl) callables sharing a
    1-slot memoized simulate_css_srp_cycle() call, so that SLSQP -- which
    evaluates the objective and every constraint at the same x in
    back-to-back calls -- does not re-simulate the cycle 3x per iteration.
    """
    cache: Dict[str, object] = {"x": None, "result": None, "error": False}

    def _get_result(x: np.ndarray) -> Optional[CycleSimulationResult]:
        key = tuple(np.round(x, 10))
        if cache["x"] == key:
            return cache["result"] if not cache["error"] else None
        m_s, t_soak, spm, S = x
        try:
            result = simulate_css_srp_cycle(m_s, t_soak, spm, S, field_params, operational_limits, sim_config)
            cache["x"], cache["result"], cache["error"] = key, result, False
            return result
        except (ValueError, InfeasibleCycleError):
            cache["x"], cache["result"], cache["error"] = key, None, True
            return None

    def objective(x: np.ndarray) -> float:
        result = _get_result(x)
        if result is None:
            return _PENALTY
        return -result.J_per_day  # minimize -J to maximize J

    def constraint_spm(x: np.ndarray) -> float:
        """g(x) = SPMsafe_min(x) - SPM  >= 0 required."""
        result = _get_result(x)
        if result is None:
            return -_PENALTY
        return result.min_spm_safe - x[2]

    def constraint_pprl(x: np.ndarray) -> float:
        """g(x) = yield_limit - PPRL_max(x)  >= 0 required."""
        result = _get_result(x)
        if result is None:
            return -_PENALTY
        return result.yield_limit_n - result.max_pprl_n

    return objective, constraint_spm, constraint_pprl


def optimize_css_srp_cycle(
    field_params: dict,
    operational_limits: dict,
    sim_config: dict,
    m_s_bounds: Tuple[float, float],
    t_soak_bounds: Tuple[float, float],
    spm_bounds: Optional[Tuple[float, float]] = None,
    stroke_bounds: Optional[Tuple[float, float]] = None,
    seed: Optional[int] = 42,
    de_maxiter: int = 40,
    de_popsize: int = 12,
    de_workers: int = 1,
    enforce_spm_constraint: bool = True,
    enforce_pprl_constraint: bool = True,
) -> OptimizationResult:
    """
    Runs the constrained 4-variable non-convex optimization over
    (m_s, t_soak, SPM, S) maximizing the per-day economic objective
    (economic_objective.evaluate_css_cycle_economics), subject to:
        SPM <= min_t SPMsafe(t)        (no rod-float, from srp_dynamics)
        max_t PPRL(t) <= yield_limit   (rod-yield safety, from srp_dynamics)

    Strategy: scipy.optimize.differential_evolution for the global/
    non-convex search (handles the non-smooth, potentially multi-modal
    landscape created by the SPMsafe/PPRL feasibility boundary), then
    scipy.optimize.minimize(method="SLSQP") seeded at the DE result for
    local polishing to a precise KKT-stationary point.

    Args:
        field_params: Parsed config/field_params.yaml.
        operational_limits: Parsed config/operational_limits.yaml.
        sim_config: Exogenous scenario assumptions -- see default_sim_config().
        m_s_bounds: (low, high) bounds on steam mass (kg). REQUIRED: not
            present in operational_limits.yaml (see module docstring,
            assumption 9) -- caller must supply a physically sane range.
        t_soak_bounds: (low, high) bounds on soak time (days). REQUIRED for
            the same reason as m_s_bounds.
        spm_bounds: (low, high) SPM bounds. Defaults to
            (operational_limits["srp_limits"]["min_spm"],
             operational_limits["srp_limits"]["max_spm"]) if None.
        stroke_bounds: (low, high) surface stroke length bounds (m).
            Defaults to
            (operational_limits["srp_limits"]["min_surface_stroke_length_m"],
             operational_limits["srp_limits"]["max_surface_stroke_length_m"])
            if None.
        seed: Random seed for differential_evolution (reproducibility).
        de_maxiter: Max generations for differential_evolution.
        de_popsize: Population size multiplier for differential_evolution.
        de_workers: Parallel worker count for differential_evolution
            (1 = serial; -1 = all cores. Note the objective is a Python
            per-node loop, so parallelism helps for expensive n_time_nodes).
        enforce_spm_constraint: If False, drops the SPM <= SPMsafe(t)
            constraint from both solver stages. Provided mainly for
            sensitivity testing (e.g. to demonstrate how much the
            constraint lowers the achievable optimum) -- production use
            should leave this True.
        enforce_pprl_constraint: Same as enforce_spm_constraint, for the
            PPRL <= yield_limit constraint.

    Returns:
        OptimizationResult: Optimized variables, resulting J, full cycle
        trace, and convergence diagnostics from both solver stages.

    Raises:
        ValueError: If any bound tuple is malformed (low >= high, or a
            bound is non-physical, e.g. m_s_bounds[0] <= 0).
    """
    if spm_bounds is None:
        spm_bounds = (
            float(operational_limits["srp_limits"]["min_spm"]),
            float(operational_limits["srp_limits"]["max_spm"]),
        )
    if stroke_bounds is None:
        stroke_bounds = (
            float(operational_limits["srp_limits"]["min_surface_stroke_length_m"]),
            float(operational_limits["srp_limits"]["max_surface_stroke_length_m"]),
        )

    bounds = [m_s_bounds, t_soak_bounds, spm_bounds, stroke_bounds]
    names = ("m_s_bounds", "t_soak_bounds", "spm_bounds", "stroke_bounds")
    for name, (lo, hi) in zip(names, bounds):
        if lo >= hi:
            raise ValueError(f"{name} = ({lo}, {hi}) must have low < high.")
    if m_s_bounds[0] <= 0:
        raise ValueError(f"m_s_bounds lower bound ({m_s_bounds[0]}) must be positive.")
    if t_soak_bounds[0] < 0:
        raise ValueError(f"t_soak_bounds lower bound ({t_soak_bounds[0]}) cannot be negative.")
    if spm_bounds[0] <= 0:
        raise ValueError(f"spm_bounds lower bound ({spm_bounds[0]}) must be positive.")
    if stroke_bounds[0] <= 0:
        raise ValueError(f"stroke_bounds lower bound ({stroke_bounds[0]}) must be positive.")

    objective, constraint_spm, constraint_pprl = _make_evaluators(field_params, operational_limits, sim_config)

    de_constraints = []
    slsqp_constraints = []
    if enforce_spm_constraint:
        de_constraints.append(NonlinearConstraint(constraint_spm, 0.0, np.inf))
        slsqp_constraints.append({"type": "ineq", "fun": constraint_spm})
    if enforce_pprl_constraint:
        de_constraints.append(NonlinearConstraint(constraint_pprl, 0.0, np.inf))
        slsqp_constraints.append({"type": "ineq", "fun": constraint_pprl})

    de_result = differential_evolution(
        objective,
        bounds=bounds,
        constraints=tuple(de_constraints) if de_constraints else (),
        seed=seed,
        maxiter=de_maxiter,
        popsize=de_popsize,
        polish=False,
        workers=de_workers,
        updating="deferred" if de_workers != 1 else "immediate",
    )

    slsqp_result = minimize(
        objective,
        x0=de_result.x,
        method="SLSQP",
        bounds=bounds,
        constraints=slsqp_constraints,
        options={"maxiter": 100, "ftol": 1e-9},
    )

    # Pick the best candidate among DE's and SLSQP's proposals rather than
    # trusting scipy's `success` flags blindly: both solvers can report
    # `success=False` on a numerically-fine point that sits exactly on an
    # active bound (e.g. SPM pinned at its lower bound, which here happens
    # to coincide with SPMsafe's own hard floor -- see _SPM_FEASIBILITY_TOL).
    candidates = [de_result.x, slsqp_result.x]
    candidate_results = []
    for x in candidates:
        m_s_c, t_soak_c, spm_c, S_c = x
        try:
            r = simulate_css_srp_cycle(m_s_c, t_soak_c, spm_c, S_c, field_params, operational_limits, sim_config)
            feasible = True
            if enforce_spm_constraint:
                feasible = feasible and (spm_c <= r.min_spm_safe + _SPM_FEASIBILITY_TOL)
            if enforce_pprl_constraint:
                feasible = feasible and (r.max_pprl_n <= r.yield_limit_n + _PPRL_FEASIBILITY_TOL_N)
            candidate_results.append((x, r, feasible))
        except (ValueError, InfeasibleCycleError):
            continue

    feasible_candidates = [(x, r) for x, r, feasible in candidate_results if feasible]
    if feasible_candidates:
        x_final, final_result = max(feasible_candidates, key=lambda item: item[1].J_per_day)
        constraints_satisfied = True
    elif candidate_results:
        # Nothing strictly feasible within tolerance -- report the least
        # bad candidate (by objective) so the caller still gets a usable
        # answer, flagged honestly as not fully constraint-satisfying.
        x_final, final_result, _ = max(candidate_results, key=lambda item: item[1].J_per_day)
        constraints_satisfied = False
    else:
        raise InfeasibleCycleError(
            "No candidate from differential_evolution or SLSQP produced a valid simulation; "
            "check bounds and sim_config for physical consistency."
        )

    m_s, t_soak, spm, S = x_final

    return OptimizationResult(
        m_s=float(m_s), t_soak=float(t_soak), spm=float(spm), S=float(S),
        J_total=final_result.economics.J_total, J_per_day=final_result.J_per_day,
        cycle_result=final_result,
        de_success=bool(de_result.success), de_message=str(de_result.message), de_nfev=int(de_result.nfev),
        slsqp_success=bool(slsqp_result.success), slsqp_message=str(slsqp_result.message),
        slsqp_nit=int(slsqp_result.get("nit", -1)),
        constraints_satisfied=constraints_satisfied,
    )