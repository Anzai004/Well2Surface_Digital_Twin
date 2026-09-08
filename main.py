#!/usr/bin/env python3
"""
Well2Surface Digital Twin: Autonomous Control & Optimization Engine
Baghewala Field Heavy Oil Operations (Rajasthan)

Phase 7-8 deep-wiring revision: replaces the fast algebraic approximations
used for end-to-end pipeline verification with direct calls into the
production numerical solvers (src/physics, src/optimization), and adds a
single-iteration holdup-density feedback loop plus hard mechanical
fail-safe overrides ahead of persistence/reporting.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.diagnostics.card_classifier import classify_dyno_card, rasterize_dyno_card
from src.exporters.latex_report import generate_latex_report
from src.exporters.persistence import export_telemetry_to_csv, save_simulation_state
from src.optimization.economic_objective import calculate_instantaneous_profit_rate
from src.physics.css_thermal import (
    calculate_andrade_viscosity,
    calculate_composite_heat_capacity,
    calculate_dynamic_alpha,
    calculate_temperature_decay,
)
from src.physics.multiphase_flow import solve_beggs_brill_pressure_profile
from src.physics.srp_dynamics import (
    calculate_buoyant_weight,
    calculate_effective_stroke_and_displacement,
    calculate_rod_stretch,
    calculate_spm_safe_floor,
    calculate_viscous_drag,
    solve_gibbs_wave_equation,
)

# ---------------------------------------------------------------------------
# Documented simplifying assumptions (hackathon scope -- not in config/*.yaml
# because they are not asset-measured parameters, just reasonable defaults
# needed to close the physics loop end-to-end):
# ---------------------------------------------------------------------------
_PLUNGER_DIAMETER_M = 0.0445          # ~1.75" standard SRP plunger
# Reference wellhead pressure per the TRD sample telemetry payload
# (baghewala/well-07/telemetry: wellhead_pressure_psi = 165.2); the Beggs &
# Brill traverse is integrated downward from this known surface condition
# rather than assuming an (unmeasured) bottomhole pressure.
_ASSUMED_WELLHEAD_PRESSURE_PA = 165.2 / (1.0 / 6894.757293168)
_ASSUMED_GAS_DENSITY_KG_M3 = 1.2
_ASSUMED_GAS_VISCOSITY_CP = 0.015
_ASSUMED_SURFACE_TENSION_N_M = 0.020
_GAS_LIQUID_RATIO_FRACTION = 1.0e-4   # near-negligible free gas (undersaturated CSS oil)
_MOTOR_MECHANICAL_EFFICIENCY = 0.85
_M3_PER_BBL = 0.158987
_PSI_PER_PA = 1.0 / 6894.757293168
_N_PER_LBF = 4.4482216153
_IN2_PER_M2 = 1550.0031

_WAVE_NUM_SPATIAL_NODES = 50
_CFL_SAFETY_MARGIN = 0.90   # solve at 90% of the CFL dt ceiling for numerical headroom
_N_MULTIPHASE_NODES = 25


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cfl_safe_sample_count(
    period_s: float, L_rod: float, num_spatial_nodes: int,
    rho_steel: float, E_steel: float, safety_margin: float = _CFL_SAFETY_MARGIN,
) -> int:
    """
    Chooses a surface-sample count (and thus dt = period/n_samples) that
    satisfies the CFL condition dt <= dx/a for the given spatial mesh,
    with `safety_margin` headroom below the exact stability ceiling.
    """
    a_wave_speed = float(np.sqrt(E_steel / rho_steel))
    dx = L_rod / (num_spatial_nodes - 1)
    dt_ceiling = dx / a_wave_speed
    dt_target = dt_ceiling * safety_margin
    n_samples = int(np.ceil(period_s / dt_target)) + 1
    return max(n_samples, 50)


def _build_surface_position_series(stroke_m: float, spm: float, n_samples: int):
    """
    Builds one full pumping-stroke cycle of surface (polished rod) position
    samples x_surface(t) plus the matching uniform time step dt, from a
    simple-harmonic surface-motion approximation (crank-driven walking
    beam): x(theta) = (S/2)*(1 - cos(theta)).
    """
    period_s = 60.0 / spm
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples)
    x_surface = (stroke_m / 2.0) * (1.0 - np.cos(theta))
    dt = period_s / n_samples
    return x_surface, dt, theta


def main():
    print("=" * 70)
    print(" WELL2SURFACE DIGITAL TWIN - CYCLIC STEAM & SUCKER ROD PUMP ENGINE ")
    print(" Field Asset: Baghewala-07 (Jodhpur Sandstone, 18° API) ")
    print(" [Phase 7/8: Deep Physics Wiring -- Production Solvers Active] ")
    print("=" * 70)

    field = _load_yaml(ROOT_DIR / "config" / "field_params.yaml")
    limits = _load_yaml(ROOT_DIR / "config" / "operational_limits.yaml")

    reservoir = field["reservoir"]
    pvt = field["pvt_fluid"]
    thermal_caps = field["thermal_capacities"]
    geom = field["wellbore_geometry"]
    srp_limits = limits["srp_limits"]
    economics_cfg = limits["economics"]

    # 1. SCADA Ingestion & State Acquisition
    print("\n[Step 1/7] Ingesting Live Wellhead Telemetry...")
    telemetry_packet = {
        "well_id": "BGW-JH-07",
        "production_days": 16.5,
        "surface_stroke_length_m": 2.5,
        "target_spm": 5.0,
        "steam_injected_tons": 1200.0,
        "steam_quality": 0.80,
        "steam_temp_c": 180.0,
        "native_res_temp_c": reservoir["native_temperature_c"],
    }
    print(f"  Ingested Well: {telemetry_packet['well_id']} at Day {telemetry_packet['production_days']}")

    t_days = telemetry_packet["production_days"]
    T_res = telemetry_packet["native_res_temp_c"]
    T_steam = telemetry_packet["steam_temp_c"]

    L_rod = geom["depth_m"]
    A_rod = geom["rod_cross_section_m2"]
    D_tubing = geom["tubing_inner_diameter_m"]
    D_rod = geom["rod_outer_diameter_m"]
    rho_steel = geom["steel_density_kg_m3"]
    E_steel = geom["steel_youngs_modulus_pa"]

    # 2. Coupled Thermodynamic & Dynamic Viscosity Computation (production solver)
    print("\n[Step 2/7] Solving Coupled Thermal Decay & Andrade Viscosity (css_thermal.py)...")

    rho_C_composite = calculate_composite_heat_capacity(
        phi=reservoir["porosity_fraction"],
        S_o=reservoir["initial_oil_saturation"],
        S_w=reservoir["initial_water_saturation"],
        rho_r=thermal_caps["rock_matrix_density_kg_m3"],
        C_pr=thermal_caps["rock_specific_heat_j_kg_c"],
        rho_o=pvt["oil_density_kg_m3"],
        C_po=thermal_caps["oil_specific_heat_j_kg_c"],
        rho_w=pvt["water_density_kg_m3"],
        C_pw=thermal_caps["water_specific_heat_j_kg_c"],
    )

    # First-pass nominal production-rate estimate to seed alpha (chicken/egg:
    # alpha needs q_o, q_o needs T(t)/mu(t), which need alpha). One
    # coarse seed iteration is sufficient for a lumped-capacitance model
    # whose alpha is only weakly sensitive to the exact q_o used here.
    q_o_seed_m3_day = 48.0 * np.exp(-0.035 * t_days) * _M3_PER_BBL * 5.615  # bbl/day-ish -> m3/day
    q_w_seed_m3_day = q_o_seed_m3_day * (
        reservoir["initial_water_saturation"] / max(reservoir["initial_oil_saturation"], 1e-6)
    )

    alpha = calculate_dynamic_alpha(
        q_o=q_o_seed_m3_day,
        rho_o=pvt["oil_density_kg_m3"],
        C_po=thermal_caps["oil_specific_heat_j_kg_c"],
        q_w=q_w_seed_m3_day,
        rho_w=pvt["water_density_kg_m3"],
        C_pw=thermal_caps["water_specific_heat_j_kg_c"],
        V_heated=reservoir["heated_zone_volume_m3"],
        rho_C_composite=rho_C_composite,
        rate_time_basis="day",
    )

    t_current = calculate_temperature_decay(t=t_days, T_steam=T_steam, T_res=T_res, alpha=alpha)
    mu_cp = calculate_andrade_viscosity(
        T_celsius=t_current,
        A=pvt["andrade_constant_a"],
        B=pvt["andrade_constant_b"],
        native_res_temp_c=T_res,
    )

    rho_fluid_nominal = (
        reservoir["initial_oil_saturation"] * pvt["oil_density_kg_m3"]
        + reservoir["initial_water_saturation"] * pvt["water_density_kg_m3"]
    )
    spm_safe = calculate_spm_safe_floor(
        mu_cp=mu_cp, rho_steel=rho_steel, rho_fluid=rho_fluid_nominal,
        spm_floor=srp_limits["min_spm"],
    )
    spm_active = float(np.clip(telemetry_packet["target_spm"], srp_limits["min_spm"],
                                min(spm_safe, srp_limits["max_spm"])))

    print(f"  Dynamic Thermal Decay Coefficient alpha: {alpha:.5f} day^-1")
    print(f"  Wellbore Temperature T(t): {t_current:.2f} °C")
    print(f"  Heavy Crude Viscosity μ(t): {mu_cp:.2f} cP")
    print(f"  Dynamic Safe Speed Limit SPM_safe: {spm_safe:.2f} SPM (Active: {spm_active:.2f})")

    # 3. 1D Gibbs Damped Wave Solver -- downhole reconstruction (production solver)
    print("\n[Step 3/7] Solving 1D Gibbs Damped Wave Equation (srp_dynamics.py)...")
    stroke_m = telemetry_packet["surface_stroke_length_m"]

    def _solve_wave(spm_for_solve: float, mu_for_solve: float):
        period_s = 60.0 / spm_for_solve
        n_samples = _cfl_safe_sample_count(
            period_s, L_rod, _WAVE_NUM_SPATIAL_NODES, rho_steel, E_steel
        )
        x_surface, dt, _ = _build_surface_position_series(stroke_m, spm_for_solve, n_samples)
        return solve_gibbs_wave_equation(
            x_surface=x_surface, dt=dt, L_rod=L_rod, mu_cp=mu_for_solve,
            A_rod=A_rod, D_tubing=D_tubing, D_rod=D_rod,
            rho_steel=rho_steel, E_steel=E_steel,
            num_spatial_nodes=_WAVE_NUM_SPATIAL_NODES, enforce_cfl=True,
        )

    wave_result = _solve_wave(spm_active, mu_cp)
    u_downhole = wave_result["u_downhole"]
    F_plunger = wave_result["F_plunger"]

    rod_stretch_m = calculate_rod_stretch(F_load=float(np.max(F_plunger)), L_rod=L_rod, A_rod=A_rod, E_steel=E_steel)
    print(f"  CFL Ratio: {wave_result['cfl_ratio']:.4f} (stable <= 1.0)")
    print(f"  Rod Stretch ΔL: {rod_stretch_m * 39.37:.2f} in")
    print(f"  Peak Downhole Plunger Load: {np.max(F_plunger):.1f} N / Min: {np.min(F_plunger):.1f} N")

    # 4. Dyno Card Tensor Rasterization & Anomaly Classification (production solver)
    print("\n[Step 4/7] Rasterizing Dyno Card & Running 5-State Diagnostic Classifier...")

    yield_stress_pa = (srp_limits["max_rod_stress_psi"] / 0.90) / _PSI_PER_PA  # full yield, not the 90% band
    F_yield_n = yield_stress_pa * A_rod

    tensor = rasterize_dyno_card(u_downhole, F_plunger, image_size=(224, 224))
    diag_result = classify_dyno_card(
        position_array=u_downhole, load_array=F_plunger, pprl_rated=F_yield_n,
        current_spm=spm_active, spm_safe=spm_safe,
    )
    print(f"  Tensor Shape: {tensor.shape}, Density: {np.mean(tensor):.4f}")
    print(f"  Operational State: {diag_result['state']} (Confidence: {diag_result['confidence'] * 100:.1f}%)")
    print(f"  Recommended Control: {diag_result['recommended_action']}")

    # 5. Multiphase Transport (Beggs & Brill) -- production solver
    print("\n[Step 5/7] Evaluating Multiphase Beggs & Brill Tubing Pressure Traverse...")

    v_rod_avg = 2.0 * stroke_m * (spm_active / 60.0)  # mean rod speed over a stroke, m/s
    S_eff, daily_disp_m3 = calculate_effective_stroke_and_displacement(
        S_surface=stroke_m, delta_L=rod_stretch_m, spm=spm_active, D_plunger=_PLUNGER_DIAMETER_M,
    )
    q_liquid_m3_s = max(daily_disp_m3 / 86400.0, 1.0e-6)

    rho_liquid_mix = (
        reservoir["initial_oil_saturation"] * pvt["oil_density_kg_m3"]
        + reservoir["initial_water_saturation"] * pvt["water_density_kg_m3"]
    )
    depth_nodes = np.linspace(0.0, L_rod, _N_MULTIPHASE_NODES)

    bb_result = solve_beggs_brill_pressure_profile(
        q_liquid_m3_s=q_liquid_m3_s,
        q_gas_m3_s=q_liquid_m3_s * _GAS_LIQUID_RATIO_FRACTION,
        pipe_id_m=D_tubing,
        depth_from_surface_m=depth_nodes,
        P_reference_pa=_ASSUMED_WELLHEAD_PRESSURE_PA,
        rho_liquid=rho_liquid_mix,
        rho_gas=_ASSUMED_GAS_DENSITY_KG_M3,
        mu_liquid_cp=mu_cp,
        mu_gas_cp=_ASSUMED_GAS_VISCOSITY_CP,
        surface_tension_n_m=_ASSUMED_SURFACE_TENSION_N_M,
        inclination_deg=90.0,
        is_uphill_flow=True,
        reference_at_surface=True,
    )
    rho_fluid_holdup = float(np.mean(bb_result.mixture_density_kg_m3))
    wellhead_pressure_pa = float(bb_result.pressure_pa[0])
    bottomhole_pressure_pa = float(bb_result.pressure_pa[-1])
    print(f"  Liquid Rate: {q_liquid_m3_s * 86400.0:.2f} m^3/day ({q_liquid_m3_s * 86400.0 / _M3_PER_BBL:.1f} bbl/day)")
    print(f"  Holdup-Weighted Mixture Density (mean): {rho_fluid_holdup:.1f} kg/m^3")
    print(f"  Wellhead Pressure (reference): {wellhead_pressure_pa * _PSI_PER_PA:.1f} psi")
    print(f"  Reconstructed Bottomhole Pressure: {bottomhole_pressure_pa * _PSI_PER_PA:.1f} psi")

    # 6. Feedback Loop: dynamic holdup density + T(t) fed back into damping/drag
    print("\n[Step 6/7] Closing Feedback Loop: rho_fluid(t) & T(t) -> c(t), F_drag, SPM_safe...")

    spm_safe_refined = calculate_spm_safe_floor(
        mu_cp=mu_cp, rho_steel=rho_steel, rho_fluid=rho_fluid_holdup,
        spm_floor=srp_limits["min_spm"],
    )
    spm_active_refined = float(np.clip(telemetry_packet["target_spm"], srp_limits["min_spm"],
                                        min(spm_safe_refined, srp_limits["max_spm"])))

    F_drag_refined = calculate_viscous_drag(
        mu_cp=mu_cp, v_rod=v_rod_avg, L_rod=L_rod, D_tubing=D_tubing, D_rod=D_rod,
    )
    W_rod_dry = rho_steel * A_rod * L_rod * 9.81
    W_buoy_refined = calculate_buoyant_weight(W_rod_dry=W_rod_dry, rho_fluid=rho_fluid_holdup, rho_steel=rho_steel)

    resolved_wave = abs(spm_active_refined - spm_active) > 1.0e-6
    if resolved_wave:
        print(f"  SPM_safe refined by holdup density: {spm_safe:.2f} -> {spm_safe_refined:.2f} SPM"
              f" | Active SPM: {spm_active:.2f} -> {spm_active_refined:.2f} -- re-solving wave equation.")
        wave_result = _solve_wave(spm_active_refined, mu_cp)
        u_downhole = wave_result["u_downhole"]
        F_plunger = wave_result["F_plunger"]
        rod_stretch_m = calculate_rod_stretch(
            F_load=float(np.max(F_plunger)), L_rod=L_rod, A_rod=A_rod, E_steel=E_steel
        )
        diag_result = classify_dyno_card(
            position_array=u_downhole, load_array=F_plunger, pprl_rated=F_yield_n,
            current_spm=spm_active_refined, spm_safe=spm_safe_refined,
        )
    else:
        print(f"  SPM_safe stable under refined holdup density ({spm_safe_refined:.2f} SPM) -- no re-solve needed.")

    spm_active = spm_active_refined
    spm_safe = spm_safe_refined
    pprl_n = W_buoy_refined + F_drag_refined + float(np.max(F_plunger))
    print(f"  Refined Drag Force F_drag: {F_drag_refined:.1f} N | Buoyant Weight: {W_buoy_refined:.1f} N")
    print(f"  Feedback-Coupled PPRL: {pprl_n:.1f} N ({pprl_n / _N_PER_LBF:.1f} lbf)")

    # 7. Edge Fail-Safe Overrides & Watchdog Protection (Task 3)
    print("\n[Step 7/7] Evaluating Mechanical Fail-Safe Interlocks & CSS Cycle Economics...")

    min_downstroke_load_n = float(np.min(F_plunger))
    yield_interrupt = pprl_n > 0.90 * F_yield_n
    rod_float_interrupt = min_downstroke_load_n <= 0.0

    emergency_override_triggered = bool(yield_interrupt or rod_float_interrupt)
    if emergency_override_triggered:
        reason = []
        if yield_interrupt:
            reason.append(f"PPRL {pprl_n:.0f} N > 90% yield ({0.90 * F_yield_n:.0f} N)")
        if rod_float_interrupt:
            reason.append(f"compressive downstroke load {min_downstroke_load_n:.0f} N <= 0 (rod floating)")
        print(f"  *** EMERGENCY THROTTLE OVERRIDE TRIGGERED: {'; '.join(reason)} ***")
        print(f"  Forcing VFD speed to idle baseline SPM = {srp_limits['min_spm']:.1f}")
        spm_active = srp_limits["min_spm"]
    else:
        print("  Mechanical interlocks nominal -- no override required.")

    q_oil_bpd = max(0.5, (q_liquid_m3_s * 86400.0) / _M3_PER_BBL)
    p_motor_kw = (pprl_n * stroke_m * (spm_active / 60.0)) / (_MOTOR_MECHANICAL_EFFICIENCY * 1000.0)
    sigma_rod_psi = (pprl_n / A_rod) * _PSI_PER_PA

    dj_dt = calculate_instantaneous_profit_rate(
        q_o=q_oil_bpd, P_motor=p_motor_kw, sigma_rod=sigma_rod_psi,
        P_oil=economics_cfg["oil_price_usd_bbl"],
        C_elec=economics_cfg["electricity_cost_usd_kwh"] * 24.0,
        C_wear=economics_cfg["equipment_wear_penalty_factor"],
    )
    cutoff_triggered = dj_dt <= 0.0

    print(f"  Oil Flow Rate: {q_oil_bpd:.2f} bbl/day")
    print(f"  Motor Power Draw: {p_motor_kw:.2f} kW")
    print(f"  Marginal Profit Rate dJ/dt: ${dj_dt:.2f}/day")
    print(f"  Cycle Cutoff Condition: {'TRIGGERED (Proceed to Huff Injection)' if cutoff_triggered else 'NOMINAL PRODUCTION'}")

    # Persistence, Logging, & LaTeX Summary Compilation
    print("\nPersisting State & Compiling LaTeX Summary Artifacts...")
    state_payload = {
        "well_id": telemetry_packet["well_id"],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "production_days": t_days,
        "reservoir_temperature_c": t_current,
        "dynamic_viscosity_cp": mu_cp,
        "thermal_decay_alpha_day": alpha,
        "spm_actual": spm_active,
        "spm_safe": spm_safe,
        "vfd_hz": spm_active * 6.0,
        "pprl_n": pprl_n,
        "pprl_lbf": pprl_n / _N_PER_LBF,
        "rod_stretch_in": rod_stretch_m * 39.37,
        "diagnostic_state": diag_result["state"],
        "cfl_ratio": wave_result["cfl_ratio"],
        "wave_solver_a_m_s": wave_result["a_wave_speed"],
        "wave_solver_c_damping": wave_result["c_damping"],
        "holdup_mixture_density_kg_m3": rho_fluid_holdup,
        "wellhead_pressure_psi": wellhead_pressure_pa * _PSI_PER_PA,
        "bottomhole_pressure_psi": bottomhole_pressure_pa * _PSI_PER_PA,
        "profit_rate_dj_dt": dj_dt,
        "cutoff_triggered": cutoff_triggered,
        "emergency_override_triggered": emergency_override_triggered,
        "cum_sor": telemetry_packet["steam_injected_tons"] / max(q_oil_bpd * t_days * 0.159, 1.0e-6),
    }

    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    save_simulation_state(state_payload, reports_dir / "latest_state.json")
    export_telemetry_to_csv([state_payload], reports_dir / "telemetry_log.csv")
    generate_latex_report(state_payload, str(reports_dir / "cycle_summary.pdf"))

    print("  Artifacts successfully exported:")
    print(f"    - JSON State:   {reports_dir / 'latest_state.json'}")
    print(f"    - Telemetry CSV:{reports_dir / 'telemetry_log.csv'}")
    print(f"    - LaTeX Report: {reports_dir / 'cycle_summary.tex'}")
    print("\nPipeline execution complete.")

    return state_payload


if __name__ == "__main__":
    main()