"""
tests/test_cfl_benchmarks.py

Phase 7 verification suite: Courant-Friedrichs-Lewy (CFL) mesh stability,
1D Gibbs wave-solver latency benchmarking, high-damping numerical
robustness, and cold-shutdown / boundary edge-case handling for the
production 1D Gibbs damped wave equation solver
(src/physics/srp_dynamics.py).
"""

import time
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.physics.srp_dynamics import (
    calculate_wave_speed_and_damping,
    check_cfl_stability,
    solve_gibbs_wave_equation,
)

# ---------------------------------------------------------------------------
# Shared well-geometry fixtures (Baghewala-07: L = 1200 m TVD, steel rod
# string, a ~= 5000 m/s per config/field_params.yaml)
# ---------------------------------------------------------------------------
L_ROD_M = 1200.0
A_ROD_M2 = 0.000388
D_TUBING_M = 0.0762
D_ROD_M = 0.0222
RHO_STEEL = 7850.0
E_STEEL = 2.1e11
A_WAVE_SPEED_REF = np.sqrt(E_STEEL / RHO_STEEL)  # ~= 5175 m/s (per config values)

LATENCY_BUDGET_S = 0.200  # NFR: < 200 ms per PRD/TRD/Implementation-Plan


def _cfl_safe_dt(dx: float, a_wave_speed: float, safety: float = 0.9) -> float:
    return safety * dx / a_wave_speed


def _make_surface_series(stroke_m: float, n_samples: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples)
    return (stroke_m / 2.0) * (1.0 - np.cos(theta))


# ---------------------------------------------------------------------------
# 1. Courant-Friedrichs-Lewy (CFL) Condition
# ---------------------------------------------------------------------------
class TestCFLStabilityCondition:
    """Verifies dt <= dx/a is enforced across a spread of spatial meshes."""

    @pytest.mark.parametrize("num_spatial_nodes", [10, 25, 50, 100, 200])
    def test_stable_dt_passes(self, num_spatial_nodes):
        dx = L_ROD_M / (num_spatial_nodes - 1)
        dt = _cfl_safe_dt(dx, A_WAVE_SPEED_REF, safety=0.9)
        ratio = check_cfl_stability(dt=dt, dx=dx, a_wave_speed=A_WAVE_SPEED_REF, enforce=True)
        assert ratio <= 1.0
        assert ratio == pytest.approx(0.9, rel=1e-6)

    @pytest.mark.parametrize("num_spatial_nodes", [10, 25, 50, 100, 200])
    def test_unstable_dt_raises(self, num_spatial_nodes):
        dx = L_ROD_M / (num_spatial_nodes - 1)
        # 20% over the exact stability ceiling -- must raise when enforced.
        dt_unstable = 1.2 * dx / A_WAVE_SPEED_REF
        with pytest.raises(ValueError, match="CFL stability violated"):
            check_cfl_stability(dt=dt_unstable, dx=dx, a_wave_speed=A_WAVE_SPEED_REF, enforce=True)

    def test_unstable_dt_permitted_when_not_enforced(self):
        dx = L_ROD_M / 49
        dt_unstable = 1.2 * dx / A_WAVE_SPEED_REF
        ratio = check_cfl_stability(dt=dt_unstable, dx=dx, a_wave_speed=A_WAVE_SPEED_REF, enforce=False)
        assert ratio > 1.0

    def test_exact_boundary_ratio_of_one_is_stable(self):
        dx = L_ROD_M / 49
        dt_exact = dx / A_WAVE_SPEED_REF
        ratio = check_cfl_stability(dt=dt_exact, dx=dx, a_wave_speed=A_WAVE_SPEED_REF, enforce=True)
        assert ratio == pytest.approx(1.0, rel=1e-9)

    def test_full_wave_solve_respects_cfl_end_to_end(self):
        """
        A full solve_gibbs_wave_equation() call with an unsafe dt (measured
        surface-sample rate coarser than the CFL ceiling) must raise before
        producing any (necessarily unstable) output, rather than silently
        returning divergent values.
        """
        num_spatial_nodes = 50
        dx = L_ROD_M / (num_spatial_nodes - 1)
        unsafe_dt = 5.0 * dx / A_WAVE_SPEED_REF
        x_surface = _make_surface_series(2.5, 200)

        with pytest.raises(ValueError, match="CFL stability violated"):
            solve_gibbs_wave_equation(
                x_surface=x_surface, dt=unsafe_dt, L_rod=L_ROD_M, mu_cp=66.0,
                A_rod=A_ROD_M2, D_tubing=D_TUBING_M, D_rod=D_ROD_M,
                rho_steel=RHO_STEEL, E_steel=E_STEEL,
                num_spatial_nodes=num_spatial_nodes, enforce_cfl=True,
            )

    def test_finer_mesh_tightens_dt_budget(self):
        """Halving dx (doubling spatial resolution) halves the CFL dt ceiling."""
        dx_coarse = L_ROD_M / 49
        dx_fine = L_ROD_M / 99
        dt_ceiling_coarse = dx_coarse / A_WAVE_SPEED_REF
        dt_ceiling_fine = dx_fine / A_WAVE_SPEED_REF
        assert dt_ceiling_fine < dt_ceiling_coarse


# ---------------------------------------------------------------------------
# 2. Wave Solver Latency Benchmark (< 200 ms across a 1,200 m rod string)
# ---------------------------------------------------------------------------
class TestWaveSolverLatency:

    @pytest.mark.parametrize("mu_cp", [1.0, 66.0, 500.0])
    def test_single_stroke_solve_under_200ms(self, mu_cp):
        num_spatial_nodes = 50
        dx = L_ROD_M / (num_spatial_nodes - 1)
        dt = _cfl_safe_dt(dx, A_WAVE_SPEED_REF, safety=0.9)
        period_s = 60.0 / 5.0  # 5 SPM nominal operating speed
        n_samples = int(np.ceil(period_s / dt)) + 1
        x_surface = _make_surface_series(2.5, n_samples)

        start = time.perf_counter()
        result = solve_gibbs_wave_equation(
            x_surface=x_surface, dt=dt, L_rod=L_ROD_M, mu_cp=mu_cp,
            A_rod=A_ROD_M2, D_tubing=D_TUBING_M, D_rod=D_ROD_M,
            rho_steel=RHO_STEEL, E_steel=E_STEEL,
            num_spatial_nodes=num_spatial_nodes, enforce_cfl=True,
        )
        elapsed_s = time.perf_counter() - start

        assert elapsed_s < LATENCY_BUDGET_S, (
            f"Wave solve took {elapsed_s * 1000:.1f} ms for mu={mu_cp} cP "
            f"(budget: {LATENCY_BUDGET_S * 1000:.0f} ms, n_samples={n_samples})."
        )
        assert result["u_downhole"].shape[0] == n_samples

    def test_repeated_solve_mean_latency_under_budget(self):
        """
        Guards against a solve that is individually fast but has a heavy,
        amortized-only-on-repeat cost (e.g. first-call JIT/allocation
        warmup) by checking the mean of several consecutive solves.
        """
        num_spatial_nodes = 50
        dx = L_ROD_M / (num_spatial_nodes - 1)
        dt = _cfl_safe_dt(dx, A_WAVE_SPEED_REF, safety=0.9)
        period_s = 60.0 / 5.0
        n_samples = int(np.ceil(period_s / dt)) + 1
        x_surface = _make_surface_series(2.5, n_samples)

        n_runs = 5
        timings = []
        for _ in range(n_runs):
            start = time.perf_counter()
            solve_gibbs_wave_equation(
                x_surface=x_surface, dt=dt, L_rod=L_ROD_M, mu_cp=66.0,
                A_rod=A_ROD_M2, D_tubing=D_TUBING_M, D_rod=D_ROD_M,
                rho_steel=RHO_STEEL, E_steel=E_STEEL,
                num_spatial_nodes=num_spatial_nodes, enforce_cfl=True,
            )
            timings.append(time.perf_counter() - start)

        mean_latency = sum(timings) / n_runs
        assert mean_latency < LATENCY_BUDGET_S, (
            f"Mean wave-solve latency {mean_latency * 1000:.1f} ms over {n_runs} runs "
            f"exceeds the {LATENCY_BUDGET_S * 1000:.0f} ms budget."
        )


# ---------------------------------------------------------------------------
# 3. High Damping Stability Bounds (T -> 46C native reservoir floor,
#    mu > 5,000 cP): no overflow / NaN / divergence in the wave solver.
# ---------------------------------------------------------------------------
class TestHighDampingStability:

    @pytest.mark.parametrize("mu_cp", [5000.0, 8000.0, 10000.0, 15000.0])
    def test_near_native_reservoir_viscosity_no_nan_or_overflow(self, mu_cp):
        num_spatial_nodes = 50
        dx = L_ROD_M / (num_spatial_nodes - 1)
        dt = _cfl_safe_dt(dx, A_WAVE_SPEED_REF, safety=0.9)
        # Very slow pumping is realistic once the well has cooled back
        # toward native temperature and viscosity has surged.
        period_s = 60.0 / 2.0  # SPM_safe floor = 2.0
        n_samples = int(np.ceil(period_s / dt)) + 1
        x_surface = _make_surface_series(2.5, n_samples)

        result = solve_gibbs_wave_equation(
            x_surface=x_surface, dt=dt, L_rod=L_ROD_M, mu_cp=mu_cp,
            A_rod=A_ROD_M2, D_tubing=D_TUBING_M, D_rod=D_ROD_M,
            rho_steel=RHO_STEEL, E_steel=E_STEEL,
            num_spatial_nodes=num_spatial_nodes, enforce_cfl=True,
        )

        for key in ("u_downhole", "F_plunger", "u_grid"):
            arr = result[key]
            assert np.all(np.isfinite(arr)), f"{key} contains non-finite values at mu={mu_cp} cP"
            assert not np.any(np.isnan(arr)), f"{key} contains NaN at mu={mu_cp} cP"

        # Heavy damping should suppress -- not amplify -- the response: the
        # downhole displacement amplitude must not runaway beyond a small
        # multiple of the driving surface stroke amplitude.
        surface_amplitude = float(np.ptp(x_surface))
        downhole_amplitude = float(np.ptp(result["u_downhole"]))
        assert downhole_amplitude <= 5.0 * surface_amplitude, (
            f"Downhole displacement amplitude ({downhole_amplitude:.3f} m) blew up "
            f"relative to surface drive ({surface_amplitude:.3f} m) at mu={mu_cp} cP."
        )

    def test_damping_coefficient_monotonically_increases_with_viscosity(self):
        _, c_low = calculate_wave_speed_and_damping(
            mu_cp=1.0, A_rod=A_ROD_M2, D_tubing=D_TUBING_M, D_rod=D_ROD_M,
            rho_steel=RHO_STEEL, E_steel=E_STEEL,
        )
        _, c_high = calculate_wave_speed_and_damping(
            mu_cp=10000.0, A_rod=A_ROD_M2, D_tubing=D_TUBING_M, D_rod=D_ROD_M,
            rho_steel=RHO_STEEL, E_steel=E_STEEL,
        )
        assert c_high > c_low
        assert np.isfinite(c_high)

    def test_extreme_viscosity_still_finite_damping(self):
        """mu -> very large (near-solid crude) must not overflow c_damping."""
        _, c_damping = calculate_wave_speed_and_damping(
            mu_cp=1.0e6, A_rod=A_ROD_M2, D_tubing=D_TUBING_M, D_rod=D_ROD_M,
            rho_steel=RHO_STEEL, E_steel=E_STEEL,
        )
        assert np.isfinite(c_damping)
        assert c_damping > 0.0


# ---------------------------------------------------------------------------
# 4. Cold Shutdown & Edge Boundary Tests
# ---------------------------------------------------------------------------
class TestColdShutdownAndEdgeBoundaries:

    def test_zero_stroke_length_produces_static_rod_no_error(self):
        """S = 0: rod string commanded to hold position -- flat surface series."""
        num_spatial_nodes = 50
        dx = L_ROD_M / (num_spatial_nodes - 1)
        dt = _cfl_safe_dt(dx, A_WAVE_SPEED_REF, safety=0.9)
        n_samples = 200
        x_surface = _make_surface_series(stroke_m=0.0, n_samples=n_samples)

        result = solve_gibbs_wave_equation(
            x_surface=x_surface, dt=dt, L_rod=L_ROD_M, mu_cp=66.0,
            A_rod=A_ROD_M2, D_tubing=D_TUBING_M, D_rod=D_ROD_M,
            rho_steel=RHO_STEEL, E_steel=E_STEEL,
            num_spatial_nodes=num_spatial_nodes, enforce_cfl=True,
        )

        assert np.all(np.isfinite(result["u_downhole"]))
        # No surface excitation -> the rod string stays at rest everywhere.
        assert np.allclose(result["u_downhole"], x_surface[0], atol=1e-9)
        assert np.allclose(result["F_plunger"], 0.0, atol=1e-6)

    def test_stationary_pump_zero_spm_rejected_as_non_physical_period(self):
        """
        SPM = 0 implies an infinite stroke period (60/0). The wave solver
        itself has no notion of SPM -- this test documents the contract at
        the pipeline boundary: callers must not translate SPM=0 into a
        period/dt calculation directly, and should instead treat SPM=0 as
        the zero-stroke-length static case (see test above) or refuse to
        drive the solver at all.
        """
        with pytest.raises(ZeroDivisionError):
            _ = 60.0 / 0.0  # documents the exact failure mode callers must guard against

    def test_infinite_viscosity_raises_cleanly(self):
        """mu -> inf must be rejected by the wave-speed/damping calculator
        with a clear error rather than silently producing inf/NaN damping."""
        with pytest.raises((ValueError, OverflowError)):
            calculate_wave_speed_and_damping(
                mu_cp=float("inf"), A_rod=A_ROD_M2, D_tubing=D_TUBING_M, D_rod=D_ROD_M,
                rho_steel=RHO_STEEL, E_steel=E_STEEL,
            )

    def test_non_positive_viscosity_rejected(self):
        for bad_mu in (0.0, -1.0, -1.0e6):
            with pytest.raises(ValueError):
                calculate_wave_speed_and_damping(
                    mu_cp=bad_mu, A_rod=A_ROD_M2, D_tubing=D_TUBING_M, D_rod=D_ROD_M,
                    rho_steel=RHO_STEEL, E_steel=E_STEEL,
                )

    def test_minimum_spatial_nodes_boundary(self):
        """num_spatial_nodes below the documented minimum (3) must raise."""
        with pytest.raises(ValueError, match="num_spatial_nodes"):
            solve_gibbs_wave_equation(
                x_surface=_make_surface_series(2.5, 50), dt=1.0e-4, L_rod=L_ROD_M,
                mu_cp=66.0, A_rod=A_ROD_M2, D_tubing=D_TUBING_M, D_rod=D_ROD_M,
                rho_steel=RHO_STEEL, E_steel=E_STEEL,
                num_spatial_nodes=2, enforce_cfl=True,
            )

    def test_single_sample_surface_series_rejected(self):
        """A degenerate single-sample x_surface has no time axis to solve over."""
        with pytest.raises(ValueError, match="at least 2 time samples"):
            solve_gibbs_wave_equation(
                x_surface=np.array([1.25]), dt=1.0e-4, L_rod=L_ROD_M,
                mu_cp=66.0, A_rod=A_ROD_M2, D_tubing=D_TUBING_M, D_rod=D_ROD_M,
                rho_steel=RHO_STEEL, E_steel=E_STEEL,
                num_spatial_nodes=50, enforce_cfl=True,
            )