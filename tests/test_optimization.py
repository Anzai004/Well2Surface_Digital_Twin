"""
Tests for src/optimization/economic_objective.py and
src/optimization/joint_optimizer.py (Phase 4 -- Integrated Well2Surface
Optimizer).

NOTE ON SCOPE: this environment was given the current source of
src/physics/css_thermal.py, src/physics/srp_dynamics.py, and
src/physics/multiphase_flow.py, plus config/field_params.yaml and
config/operational_limits.yaml, but NOT the existing
test_css_thermal.py / test_srp_dynamics.py / test_multiphase_flow.py
files referenced in the hand-off ("67/67 tests passing"). Those are not
re-created or re-run here -- only the new Phase 4 module is tested.
A couple of lightweight import/smoke checks against the Phase 1-3
modules are included so a missing/renamed function is caught early.
"""

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.physics import css_thermal as ct
from src.physics import multiphase_flow as mf
from src.physics import srp_dynamics as sd
from src.optimization import economic_objective as eco
from src.optimization import joint_optimizer as jo

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture(scope="module")
def field_params():
    with open(CONFIG_DIR / "field_params.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def operational_limits():
    with open(CONFIG_DIR / "operational_limits.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture()
def sim_config(field_params):
    cfg = jo.default_sim_config(field_params)
    cfg["n_time_nodes"] = 15  # keep tests fast
    return cfg


# ---------------------------------------------------------------------------
# Smoke checks: Phase 1-3 modules import and behave as this module assumes.
# ---------------------------------------------------------------------------

class TestPhysicsModuleSmoke:
    def test_css_thermal_andrade_calibration(self):
        assert ct.calculate_andrade_viscosity(47.0) == pytest.approx(66.0, abs=0.01)
        assert ct.calculate_andrade_viscosity(250.0) == pytest.approx(1.0, abs=0.01)

    def test_srp_dynamics_pprl_formula(self):
        pprl, flag = sd.calculate_pprl_and_check_yield(W_r=100.0, W_f=50.0, F_drag=10.0, yield_limit_load=1000.0)
        assert pprl == pytest.approx(160.0)
        assert flag is False

    def test_multiphase_flow_pressure_profile_runs(self):
        result = mf.solve_beggs_brill_pressure_profile(
            q_liquid_m3_s=0.01, q_gas_m3_s=0.0, pipe_id_m=0.0762,
            depth_from_surface_m=np.array([0.0, 1200.0]), P_reference_pa=1.138e6,
            rho_liquid=950.0, rho_gas=1.2, mu_liquid_cp=30.0, mu_gas_cp=0.015,
            surface_tension_n_m=0.02,
        )
        assert result.pressure_pa.shape == (2,)
        assert result.pressure_pa[1] > result.pressure_pa[0]  # pressure builds with depth


# ---------------------------------------------------------------------------
# economic_objective.py
# ---------------------------------------------------------------------------

class TestInstantaneousProfitRate:
    def test_basic_formula(self):
        rate = eco.calculate_instantaneous_profit_rate(
            q_o=100.0, P_motor=20.0, sigma_rod=500.0, P_oil=75.0, C_elec=2.88, C_wear=0.05
        )
        expected = 75.0 * 100.0 - 2.88 * 20.0 - 0.05 * 500.0
        assert rate == pytest.approx(expected)

    def test_array_input(self):
        rate = eco.calculate_instantaneous_profit_rate(
            q_o=np.array([0.0, 100.0]), P_motor=np.array([10.0, 20.0]),
            sigma_rod=np.array([100.0, 500.0]), P_oil=75.0, C_elec=2.88, C_wear=0.05,
        )
        assert rate.shape == (2,)
        assert rate[0] < rate[1]

    @pytest.mark.parametrize("kwargs", [
        dict(q_o=-1.0, P_motor=1.0, sigma_rod=1.0, P_oil=1.0, C_elec=1.0, C_wear=1.0),
        dict(q_o=1.0, P_motor=-1.0, sigma_rod=1.0, P_oil=1.0, C_elec=1.0, C_wear=1.0),
        dict(q_o=1.0, P_motor=1.0, sigma_rod=-1.0, P_oil=1.0, C_elec=1.0, C_wear=1.0),
        dict(q_o=1.0, P_motor=1.0, sigma_rod=1.0, P_oil=-1.0, C_elec=1.0, C_wear=1.0),
        dict(q_o=1.0, P_motor=1.0, sigma_rod=1.0, P_oil=1.0, C_elec=-1.0, C_wear=1.0),
        dict(q_o=1.0, P_motor=1.0, sigma_rod=1.0, P_oil=1.0, C_elec=1.0, C_wear=-1.0),
    ])
    def test_negative_inputs_raise(self, kwargs):
        with pytest.raises(ValueError):
            eco.calculate_instantaneous_profit_rate(**kwargs)


class TestSteamOilRatio:
    def test_zero_oil_returns_inf(self):
        assert eco.calculate_steam_oil_ratio(steam_volume=10.0, oil_volume=0.0) == np.inf

    def test_normal_ratio(self):
        assert eco.calculate_steam_oil_ratio(steam_volume=30.0, oil_volume=10.0) == pytest.approx(3.0)

    def test_array_input(self):
        sor = eco.calculate_steam_oil_ratio(np.array([0.0, 10.0, 20.0]), np.array([0.0, 5.0, 10.0]))
        assert sor[0] == np.inf
        assert sor[1] == pytest.approx(2.0)
        assert sor[2] == pytest.approx(2.0)

    @pytest.mark.parametrize("steam,oil", [(-1.0, 5.0), (5.0, -1.0)])
    def test_negative_inputs_raise(self, steam, oil):
        with pytest.raises(ValueError):
            eco.calculate_steam_oil_ratio(steam, oil)


class TestEvaluateCssCycleEconomics:
    def _series(self, n=10):
        return np.linspace(0.0, 30.0, n)

    def test_typical_cycle_positive_profit(self):
        t = self._series()
        n = t.shape[0]
        result = eco.evaluate_css_cycle_economics(
            t=t, q_o=np.full(n, 100.0), P_motor=np.full(n, 15.0), sigma_rod=np.full(n, 20000.0),
            steam_volume_cumulative=np.full(n, 50.0), m_s=20000.0,
            P_oil=75.0, C_elec=2.88, C_wear=0.05, C_steam=0.025,
        )
        assert isinstance(result, eco.EconomicCycleResult)
        assert result.J_total > 0
        assert not result.cutoff_triggered

    def test_boundary_zero_production_triggers_immediate_cutoff(self):
        """Boundary case: zero oil production for the whole series -- pure
        cost, no revenue -- must trigger the cutoff at t=0 (profit_rate)."""
        t = self._series()
        n = t.shape[0]
        result = eco.evaluate_css_cycle_economics(
            t=t, q_o=np.zeros(n), P_motor=np.full(n, 15.0), sigma_rod=np.full(n, 20000.0),
            steam_volume_cumulative=np.full(n, 50.0), m_s=20000.0,
            P_oil=75.0, C_elec=2.88, C_wear=0.05, C_steam=0.025,
        )
        assert result.cutoff_triggered is True
        assert result.cutoff_index == 0
        assert result.cutoff_reason == "profit_rate"
        assert result.J_total < 0
        assert np.all(result.profit_rate <= 0.0)
        # cumulative SOR should be +inf everywhere (no oil produced at all)
        assert np.all(np.isinf(result.cumulative_sor))

    def test_sor_threshold_triggers_cutoff_even_if_profit_positive(self):
        t = np.linspace(0.0, 10.0, 6)
        n = t.shape[0]
        q_o = np.full(n, 50.0)  # steady, positive profit throughout
        steam_vol = np.linspace(0.0, 5000.0, n)  # steam keeps accumulating fast
        result = eco.evaluate_css_cycle_economics(
            t=t, q_o=q_o, P_motor=np.full(n, 5.0), sigma_rod=np.full(n, 1000.0),
            steam_volume_cumulative=steam_vol, m_s=100.0,
            P_oil=75.0, C_elec=0.5, C_wear=0.001, C_steam=0.001,
            sor_threshold=2.0,
        )
        assert np.all(result.profit_rate > 0.0)  # profit alone would never cut off
        assert result.cutoff_triggered is True
        assert result.cutoff_reason == "sor_threshold"

    def test_empty_time_array_raises(self):
        with pytest.raises(ValueError):
            eco.evaluate_css_cycle_economics(
                t=np.array([]), q_o=np.array([]), P_motor=np.array([]), sigma_rod=np.array([]),
                steam_volume_cumulative=np.array([]), m_s=1.0,
                P_oil=1.0, C_elec=1.0, C_wear=1.0, C_steam=1.0,
            )

    def test_non_monotonic_time_raises(self):
        t = np.array([0.0, 2.0, 1.0, 3.0])
        n = t.shape[0]
        with pytest.raises(ValueError):
            eco.evaluate_css_cycle_economics(
                t=t, q_o=np.full(n, 1.0), P_motor=np.full(n, 1.0), sigma_rod=np.full(n, 1.0),
                steam_volume_cumulative=np.full(n, 1.0), m_s=1.0,
                P_oil=1.0, C_elec=1.0, C_wear=1.0, C_steam=1.0,
            )

    def test_mismatched_shapes_raise(self):
        t = np.linspace(0.0, 10.0, 5)
        with pytest.raises(ValueError):
            eco.evaluate_css_cycle_economics(
                t=t, q_o=np.full(4, 1.0), P_motor=np.full(5, 1.0), sigma_rod=np.full(5, 1.0),
                steam_volume_cumulative=np.full(5, 1.0), m_s=1.0,
                P_oil=1.0, C_elec=1.0, C_wear=1.0, C_steam=1.0,
            )

    def test_negative_rate_raises(self):
        t = np.linspace(0.0, 10.0, 5)
        with pytest.raises(ValueError):
            eco.evaluate_css_cycle_economics(
                t=t, q_o=np.full(5, -1.0), P_motor=np.full(5, 1.0), sigma_rod=np.full(5, 1.0),
                steam_volume_cumulative=np.full(5, 1.0), m_s=1.0,
                P_oil=1.0, C_elec=1.0, C_wear=1.0, C_steam=1.0,
            )

    def test_decreasing_cumulative_steam_raises(self):
        t = np.linspace(0.0, 10.0, 5)
        with pytest.raises(ValueError):
            eco.evaluate_css_cycle_economics(
                t=t, q_o=np.full(5, 1.0), P_motor=np.full(5, 1.0), sigma_rod=np.full(5, 1.0),
                steam_volume_cumulative=np.array([10.0, 8.0, 9.0, 11.0, 12.0]), m_s=1.0,
                P_oil=1.0, C_elec=1.0, C_wear=1.0, C_steam=1.0,
            )

    def test_non_positive_sor_threshold_raises(self):
        t = np.linspace(0.0, 10.0, 5)
        with pytest.raises(ValueError):
            eco.evaluate_css_cycle_economics(
                t=t, q_o=np.full(5, 1.0), P_motor=np.full(5, 1.0), sigma_rod=np.full(5, 1.0),
                steam_volume_cumulative=np.full(5, 1.0), m_s=1.0,
                P_oil=1.0, C_elec=1.0, C_wear=1.0, C_steam=1.0, sor_threshold=0.0,
            )

    def test_negative_m_s_raises(self):
        t = np.linspace(0.0, 10.0, 5)
        with pytest.raises(ValueError):
            eco.evaluate_css_cycle_economics(
                t=t, q_o=np.full(5, 1.0), P_motor=np.full(5, 1.0), sigma_rod=np.full(5, 1.0),
                steam_volume_cumulative=np.full(5, 1.0), m_s=-1.0,
                P_oil=1.0, C_elec=1.0, C_wear=1.0, C_steam=1.0,
            )


# ---------------------------------------------------------------------------
# joint_optimizer.py -- simulate_css_srp_cycle
# ---------------------------------------------------------------------------

class TestSimulateCssSrpCycle:
    def test_basic_sanity(self, field_params, operational_limits, sim_config):
        result = jo.simulate_css_srp_cycle(
            m_s=20000.0, t_soak=5.0, spm=4.0, S=2.0,
            field_params=field_params, operational_limits=operational_limits, sim_config=sim_config,
        )
        assert result.T_peak_c >= field_params["reservoir"]["native_temperature_c"]
        # Reservoir cools during the puff phase: T non-increasing, mu non-decreasing.
        assert np.all(np.diff(result.T_c) <= 1e-9)
        assert np.all(np.diff(result.mu_cp) >= -1e-9)
        assert np.all(result.q_o_bbl_day >= 0.0)
        assert result.min_spm_safe >= operational_limits["srp_limits"]["min_spm"]
        assert isinstance(result.economics, eco.EconomicCycleResult)
        assert result.total_cycle_days == pytest.approx(result.t_soak + result.t_days[-1])

    @pytest.mark.parametrize("kwargs", [
        dict(m_s=0.0, t_soak=5.0, spm=4.0, S=2.0),
        dict(m_s=-100.0, t_soak=5.0, spm=4.0, S=2.0),
        dict(m_s=20000.0, t_soak=-1.0, spm=4.0, S=2.0),
        dict(m_s=20000.0, t_soak=5.0, spm=0.0, S=2.0),
        dict(m_s=20000.0, t_soak=5.0, spm=4.0, S=0.0),
    ])
    def test_rejects_non_physical_inputs(self, kwargs, field_params, operational_limits, sim_config):
        with pytest.raises(ValueError):
            jo.simulate_css_srp_cycle(
                field_params=field_params, operational_limits=operational_limits, sim_config=sim_config, **kwargs
            )

    def test_infeasible_when_steam_cap_exceeded(self, field_params, operational_limits, sim_config):
        """Boundary case: enough steam mass that the energy-balance T_peak
        would exceed steam_limits.max_injection_temp_c -- must raise
        InfeasibleCycleError rather than silently returning an
        out-of-spec temperature."""
        with pytest.raises(jo.InfeasibleCycleError):
            jo.simulate_css_srp_cycle(
                m_s=5_000_000.0, t_soak=5.0, spm=4.0, S=2.0,
                field_params=field_params, operational_limits=operational_limits, sim_config=sim_config,
            )

    def test_pprl_flagged_when_yield_limit_artificially_tiny(self, field_params, operational_limits, sim_config):
        """PPRL <= yield_limit boundary: with a deliberately tiny yield
        threshold, the (otherwise unremarkable) PPRL from a normal cycle
        must exceed it, and the joint_optimizer constraint wrapper must
        report that violation with a negative margin."""
        tiny_limits = copy.deepcopy(operational_limits)
        tiny_limits["srp_limits"]["max_rod_stress_psi"] = 1.0  # ~0 N of headroom

        result = jo.simulate_css_srp_cycle(
            m_s=20000.0, t_soak=5.0, spm=4.0, S=2.0,
            field_params=field_params, operational_limits=tiny_limits, sim_config=sim_config,
        )
        assert result.max_pprl_n > result.yield_limit_n

        _, _, constraint_pprl = jo._make_evaluators(field_params, tiny_limits, sim_config)
        margin = constraint_pprl(np.array([20000.0, 5.0, 4.0, 2.0]))
        assert margin < 0.0


# ---------------------------------------------------------------------------
# joint_optimizer.py -- optimize_css_srp_cycle
# ---------------------------------------------------------------------------

class TestOptimizeCssSrpCycle:
    def test_bounds_validation(self, field_params, operational_limits, sim_config):
        with pytest.raises(ValueError):
            jo.optimize_css_srp_cycle(
                field_params, operational_limits, sim_config,
                m_s_bounds=(10000.0, 1000.0), t_soak_bounds=(1.0, 10.0),  # low >= high
            )
        with pytest.raises(ValueError):
            jo.optimize_css_srp_cycle(
                field_params, operational_limits, sim_config,
                m_s_bounds=(-1.0, 1000.0), t_soak_bounds=(1.0, 10.0),  # m_s lower bound <= 0
            )
        with pytest.raises(ValueError):
            jo.optimize_css_srp_cycle(
                field_params, operational_limits, sim_config,
                m_s_bounds=(1000.0, 10000.0), t_soak_bounds=(-5.0, 10.0),  # t_soak lower bound < 0
            )

    def test_typical_full_cycle_optimum_is_feasible(self, field_params, operational_limits, sim_config):
        result = jo.optimize_css_srp_cycle(
            field_params, operational_limits, sim_config,
            m_s_bounds=(5000.0, 80000.0), t_soak_bounds=(2.0, 15.0),
            de_maxiter=20, de_popsize=8, seed=7,
        )
        m_s_lo, m_s_hi = 5000.0, 80000.0
        t_lo, t_hi = 2.0, 15.0
        spm_lo, spm_hi = operational_limits["srp_limits"]["min_spm"], operational_limits["srp_limits"]["max_spm"]
        s_lo, s_hi = (
            operational_limits["srp_limits"]["min_surface_stroke_length_m"],
            operational_limits["srp_limits"]["max_surface_stroke_length_m"],
        )

        assert m_s_lo - 1e-6 <= result.m_s <= m_s_hi + 1e-6
        assert t_lo - 1e-6 <= result.t_soak <= t_hi + 1e-6
        assert spm_lo - 1e-6 <= result.spm <= spm_hi + 1e-6
        assert s_lo - 1e-6 <= result.S <= s_hi + 1e-6

        assert np.isfinite(result.J_total)
        assert result.constraints_satisfied is True
        assert result.spm <= result.cycle_result.min_spm_safe + jo._SPM_FEASIBILITY_TOL
        assert result.cycle_result.max_pprl_n <= result.cycle_result.yield_limit_n + jo._PPRL_FEASIBILITY_TOL_N

    def test_spm_safe_constraint_forces_lower_optimum(self, field_params, operational_limits, sim_config):
        """Boundary case: in a cold/high-viscosity scenario (small m_s
        bound keeps T_peak low, so mu stays high and SPMsafe is pinned at
        its 2.0 floor), the SPM<=SPMsafe(t) constraint must force the
        optimizer to a materially lower SPM -- and lower profit -- than an
        otherwise-identical unconstrained run would pick."""
        cold_bounds = dict(m_s_bounds=(1000.0, 15000.0), t_soak_bounds=(1.0, 10.0))

        constrained = jo.optimize_css_srp_cycle(
            field_params, operational_limits, sim_config, **cold_bounds,
            de_maxiter=15, de_popsize=8, seed=1,
        )
        unconstrained = jo.optimize_css_srp_cycle(
            field_params, operational_limits, sim_config, **cold_bounds,
            de_maxiter=15, de_popsize=8, seed=1,
            enforce_spm_constraint=False,
        )

        # The unconstrained run must actually want to violate the safety
        # floor it's ignoring -- otherwise this isn't testing anything.
        assert unconstrained.spm > unconstrained.cycle_result.min_spm_safe + jo._SPM_FEASIBILITY_TOL

        assert constrained.spm <= constrained.cycle_result.min_spm_safe + jo._SPM_FEASIBILITY_TOL
        assert constrained.spm < unconstrained.spm
        assert constrained.J_per_day < unconstrained.J_per_day
        assert constrained.constraints_satisfied is True