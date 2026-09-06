"""
Unit tests for multiphase flow and thermal-hydraulic coupling engine.
"""

import numpy as np
import pytest

from src.physics.multiphase_flow import (
    BeggsBrillResult,
    calculate_dynamic_fluid_density,
    couple_wellbore_temperature_profile,
    solve_beggs_brill_pressure_profile,
)


class TestCoupleWellboreTemperatureProfile:
    """Unit tests for couple_wellbore_temperature_profile."""

    def test_linear_profile_exact_values(self):
        T_bh = 180.0
        T_wh = 60.0
        L_total = 1200.0
        depths = np.array([0.0, 300.0, 600.0, 900.0, 1200.0])

        temps = couple_wellbore_temperature_profile(
            T_bottomhole_c=T_bh,
            T_wellhead_c=T_wh,
            depth_from_surface_m=depths,
            L_total_m=L_total,
        )

        expected = np.array([180.0, 150.0, 120.0, 90.0, 60.0])
        np.testing.assert_allclose(temps, expected, rtol=1e-6)

    def test_single_depth_array(self):
        T_bh = 100.0
        T_wh = 40.0
        L_total = 1000.0
        depths = np.array([500.0])

        temps = couple_wellbore_temperature_profile(T_bh, T_wh, depths, L_total)
        assert np.isclose(temps[0], 70.0)

    def test_guardrail_invalid_L_total(self):
        with pytest.raises(ValueError, match="Total wellbore depth L_total_m must be positive"):
            couple_wellbore_temperature_profile(100.0, 50.0, np.array([0.0]), -500.0)

        with pytest.raises(ValueError, match="Total wellbore depth L_total_m must be positive"):
            couple_wellbore_temperature_profile(100.0, 50.0, np.array([0.0]), 0.0)

    def test_guardrail_temperature_inversion(self):
        with pytest.raises(ValueError, match="cannot be less than wellhead temperature"):
            couple_wellbore_temperature_profile(40.0, 80.0, np.array([100.0]), 1000.0)

    def test_guardrail_depth_out_of_bounds(self):
        L_total = 1200.0
        with pytest.raises(ValueError, match="All depth values must be within"):
            couple_wellbore_temperature_profile(150.0, 50.0, np.array([-10.0, 600.0]), L_total)

        with pytest.raises(ValueError, match="All depth values must be within"):
            couple_wellbore_temperature_profile(150.0, 50.0, np.array([0.0, 1205.0]), L_total)


class TestCalculateDynamicFluidDensity:
    """Unit tests for calculate_dynamic_fluid_density."""

    def test_holdup_sweep_boundary_and_midpoint(self):
        rho_liq = 946.5
        rho_gas = 1.2

        # H_L = 0.0 -> pure gas density
        rho_0 = calculate_dynamic_fluid_density(0.0, rho_liq, rho_gas)
        assert np.isclose(rho_0, rho_gas)

        # H_L = 0.5 -> exact arithmetic midpoint
        rho_50 = calculate_dynamic_fluid_density(0.5, rho_liq, rho_gas)
        assert np.isclose(rho_50, 0.5 * rho_liq + 0.5 * rho_gas)

        # H_L = 1.0 -> pure liquid density
        rho_100 = calculate_dynamic_fluid_density(1.0, rho_liq, rho_gas)
        assert np.isclose(rho_100, rho_liq)

    def test_default_gas_density(self):
        rho_liq = 1000.0
        rho_mix = calculate_dynamic_fluid_density(0.8, rho_liq)
        expected = (0.8 * 1000.0) + (0.2 * 1.2)
        assert np.isclose(rho_mix, expected)

    def test_guardrail_invalid_holdup(self):
        with pytest.raises(ValueError, match="liquid_holdup_fraction must be in range"):
            calculate_dynamic_fluid_density(-0.01, 946.5, 1.2)

        with pytest.raises(ValueError, match="liquid_holdup_fraction must be in range"):
            calculate_dynamic_fluid_density(1.01, 946.5, 1.2)

    def test_guardrail_invalid_densities(self):
        with pytest.raises(ValueError, match="rho_liquid must be positive"):
            calculate_dynamic_fluid_density(0.5, -946.5, 1.2)

        with pytest.raises(ValueError, match="rho_liquid must be positive"):
            calculate_dynamic_fluid_density(0.5, 0.0, 1.2)

        with pytest.raises(ValueError, match="rho_gas must be positive"):
            calculate_dynamic_fluid_density(0.5, 946.5, -1.2)

        with pytest.raises(ValueError, match="rho_gas must be positive"):
            calculate_dynamic_fluid_density(0.5, 946.5, 0.0)


# ---------------------------------------------------------------------------
# Shared fixtures for Beggs & Brill tests: representative Baghewala-style
# 18 deg API heavy crude SRP tubing-string conditions.
# ---------------------------------------------------------------------------

def _baghewala_common_kwargs(**overrides):
    depths = np.linspace(0.0, 1200.0, 13)
    kwargs = dict(
        q_liquid_m3_s=0.0045,       # ~2450 bpd equivalent
        q_gas_m3_s=0.0025,          # modest associated gas
        pipe_id_m=0.062,            # ~2 7/8" tubing
        depth_from_surface_m=depths,
        P_reference_pa=1_500_000.0,  # ~15.2 bar wellhead reference
        rho_liquid=946.5,           # 18 deg API heavy crude
        rho_gas=1.2,
        mu_liquid_cp=66.0,          # per Andrade calibration at ~47 C
        mu_gas_cp=0.012,
        surface_tension_n_m=0.02,
        inclination_deg=90.0,
        is_uphill_flow=True,
        reference_at_surface=True,
    )
    kwargs.update(overrides)
    return kwargs


class TestSolveBeggsBrillPressureProfileVertical:
    """Typical vertical production-tubing conditions (inclination = 90 deg)."""

    def test_returns_beggs_brill_result_with_matching_shapes(self):
        kwargs = _baghewala_common_kwargs()
        result = solve_beggs_brill_pressure_profile(**kwargs)

        assert isinstance(result, BeggsBrillResult)
        n = kwargs["depth_from_surface_m"].shape[0]
        for field in (
            result.pressure_pa,
            result.liquid_holdup,
            result.flow_pattern,
            result.mixture_density_kg_m3,
            result.pressure_gradient_pa_per_m,
            result.elevation_gradient_pa_per_m,
            result.friction_gradient_pa_per_m,
        ):
            assert field.shape == (n,)

    def test_liquid_holdup_is_physically_bounded(self):
        result = solve_beggs_brill_pressure_profile(**_baghewala_common_kwargs())
        assert np.all(result.liquid_holdup > 0.0)
        assert np.all(result.liquid_holdup < 1.0)

    def test_flow_pattern_is_always_classified(self):
        result = solve_beggs_brill_pressure_profile(**_baghewala_common_kwargs())
        valid_labels = {"segregated", "intermittent", "distributed", "transition"}
        assert set(np.unique(result.flow_pattern)).issubset(valid_labels)
        assert not np.any(result.flow_pattern == "")

    def test_pressure_increases_with_depth_for_producing_well(self):
        # Reference at surface: pressure should rise monotonically going downhole
        # since both elevation and friction gradients are positive for uphill
        # (production) flow.
        result = solve_beggs_brill_pressure_profile(**_baghewala_common_kwargs())
        assert np.all(np.diff(result.pressure_pa) > 0)
        assert np.isclose(result.pressure_pa[0], 1_500_000.0)

    def test_vertical_elevation_gradient_uses_full_gravity_component(self):
        # At theta = 90 deg, sin(theta) = 1, so elevation gradient = rho_s * g exactly.
        result = solve_beggs_brill_pressure_profile(**_baghewala_common_kwargs())
        expected_elevation = result.mixture_density_kg_m3 * 9.81
        np.testing.assert_allclose(result.elevation_gradient_pa_per_m, expected_elevation, rtol=1e-6)

    def test_mixture_density_matches_calculate_dynamic_fluid_density(self):
        # Ties solve_beggs_brill_pressure_profile's liquid_holdup output directly
        # into calculate_dynamic_fluid_density, per the module's documented
        # integration point.
        kwargs = _baghewala_common_kwargs()
        result = solve_beggs_brill_pressure_profile(**kwargs)

        for idx in (0, 3, 7, 12):
            expected = calculate_dynamic_fluid_density(
                float(result.liquid_holdup[idx]), kwargs["rho_liquid"], kwargs["rho_gas"]
            )
            assert np.isclose(result.mixture_density_kg_m3[idx], expected, rtol=1e-9)

    def test_reference_at_bottomhole_matches_boundary_condition(self):
        # A realistic bottomhole reference: the surface-referenced run's own bottomhole
        # pressure, so the resulting profile stays physically positive throughout.
        surface_run = solve_beggs_brill_pressure_profile(**_baghewala_common_kwargs())
        kwargs = _baghewala_common_kwargs(
            reference_at_surface=False, P_reference_pa=float(surface_run.pressure_pa[-1])
        )
        result = solve_beggs_brill_pressure_profile(**kwargs)
        assert np.isclose(result.pressure_pa[-1], kwargs["P_reference_pa"])
        # Pressure should still increase with depth (surface value is lower).
        assert np.all(np.diff(result.pressure_pa) > 0)
        assert np.all(result.pressure_pa > 0)

    def test_surface_vs_bottomhole_reference_consistency(self):
        # Anchoring at the surface pressure produced by a bottomhole-referenced run
        # should reproduce the same overall profile shape. Use a bottomhole reference
        # large enough to keep the whole column physically positive (~1200 m of heavy
        # crude is on the order of 10 MPa of hydrostatic head alone).
        kwargs_bh = _baghewala_common_kwargs(reference_at_surface=False, P_reference_pa=12_000_000.0)
        result_bh = solve_beggs_brill_pressure_profile(**kwargs_bh)

        kwargs_wh = _baghewala_common_kwargs(
            reference_at_surface=True, P_reference_pa=float(result_bh.pressure_pa[0])
        )
        result_wh = solve_beggs_brill_pressure_profile(**kwargs_wh)

        np.testing.assert_allclose(result_bh.pressure_pa, result_wh.pressure_pa, rtol=1e-9)


class TestSolveBeggsBrillPressureProfileHorizontalAndInclined:
    """Horizontal and inclined-wellbore conditions."""

    def test_horizontal_run_has_zero_elevation_gradient(self):
        kwargs = _baghewala_common_kwargs(inclination_deg=0.0)
        result = solve_beggs_brill_pressure_profile(**kwargs)
        np.testing.assert_allclose(result.elevation_gradient_pa_per_m, 0.0, atol=1e-9)
        # Total gradient is then pure friction, still positive for a moving fluid.
        assert np.all(result.friction_gradient_pa_per_m > 0.0)
        np.testing.assert_allclose(
            result.pressure_gradient_pa_per_m, result.friction_gradient_pa_per_m, rtol=1e-9
        )

    def test_inclined_run_between_horizontal_and_vertical(self):
        kwargs_incl = _baghewala_common_kwargs(inclination_deg=45.0)
        kwargs_vert = _baghewala_common_kwargs(inclination_deg=90.0)
        result_incl = solve_beggs_brill_pressure_profile(**kwargs_incl)
        result_vert = solve_beggs_brill_pressure_profile(**kwargs_vert)

        # sin(45) < sin(90), so the elevation gradient at 45 deg must be strictly
        # smaller than the fully-vertical case for matching mixture density.
        assert np.all(result_incl.elevation_gradient_pa_per_m < result_vert.elevation_gradient_pa_per_m)
        assert np.all(result_incl.liquid_holdup > 0.0)
        assert np.all(result_incl.liquid_holdup < 1.0)

    def test_varying_inclination_array_along_depth(self):
        depths = np.linspace(0.0, 1000.0, 6)
        inclination_profile = np.array([90.0, 90.0, 60.0, 60.0, 30.0, 0.0])
        kwargs = _baghewala_common_kwargs(
            depth_from_surface_m=depths, inclination_deg=inclination_profile
        )
        result = solve_beggs_brill_pressure_profile(**kwargs)
        # Elevation gradient should trend downward as the pipe flattens out.
        assert result.elevation_gradient_pa_per_m[-1] < result.elevation_gradient_pa_per_m[0]
        assert np.isclose(result.elevation_gradient_pa_per_m[-1], 0.0, atol=1e-9)

    def test_downhill_flow_runs_without_error(self):
        # Models a CSS Huff (steam-injection) leg where flow moves downward.
        kwargs = _baghewala_common_kwargs(is_uphill_flow=False, inclination_deg=90.0)
        result = solve_beggs_brill_pressure_profile(**kwargs)
        assert np.all(result.liquid_holdup > 0.0)
        assert np.all(result.liquid_holdup < 1.0)
        assert np.all(np.isfinite(result.pressure_pa))


class TestSolveBeggsBrillPressureProfileFlowRegimes:
    """Flow-regime coverage across the lambda_L - N_FR plane."""

    def test_very_low_rate_trends_segregated(self):
        # A very low mixture velocity (small N_FR) at a larger pipe diameter and
        # balanced liquid/gas loading (lambda_L = 0.5) should land in the
        # segregated regime, since gravity-driven phase separation dominates at
        # low velocity.
        kwargs = _baghewala_common_kwargs(
            q_liquid_m3_s=0.00017,
            q_gas_m3_s=0.00017,
            pipe_id_m=0.15,
            inclination_deg=0.0,
        )
        result = solve_beggs_brill_pressure_profile(**kwargs)
        assert np.all(result.flow_pattern == "segregated")

    def test_high_rate_trends_distributed(self):
        # High mixture velocity drives N_FR up sharply, which should push the
        # regime toward distributed flow regardless of liquid loading.
        kwargs = _baghewala_common_kwargs(
            q_liquid_m3_s=0.05, q_gas_m3_s=0.05, inclination_deg=0.0
        )
        result = solve_beggs_brill_pressure_profile(**kwargs)
        assert np.all(result.flow_pattern == "distributed")

    def test_moderate_rate_can_produce_intermittent_regime(self):
        kwargs = _baghewala_common_kwargs(
            q_liquid_m3_s=0.006, q_gas_m3_s=0.004, inclination_deg=0.0
        )
        result = solve_beggs_brill_pressure_profile(**kwargs)
        assert np.all(np.isin(result.flow_pattern, ["intermittent", "transition", "distributed"]))


class TestSolveBeggsBrillPressureProfileGuardrails:
    """Guardrail / error-raising tests for invalid parameters."""

    def test_guardrail_non_positive_liquid_rate(self):
        with pytest.raises(ValueError, match="q_liquid_m3_s must be positive"):
            solve_beggs_brill_pressure_profile(**_baghewala_common_kwargs(q_liquid_m3_s=0.0))

        with pytest.raises(ValueError, match="q_liquid_m3_s must be positive"):
            solve_beggs_brill_pressure_profile(**_baghewala_common_kwargs(q_liquid_m3_s=-0.001))

    def test_guardrail_negative_gas_rate(self):
        with pytest.raises(ValueError, match="q_gas_m3_s must be non-negative"):
            solve_beggs_brill_pressure_profile(**_baghewala_common_kwargs(q_gas_m3_s=-0.001))

    def test_guardrail_non_positive_pipe_id(self):
        with pytest.raises(ValueError, match="pipe_id_m must be positive"):
            solve_beggs_brill_pressure_profile(**_baghewala_common_kwargs(pipe_id_m=0.0))

        with pytest.raises(ValueError, match="pipe_id_m must be positive"):
            solve_beggs_brill_pressure_profile(**_baghewala_common_kwargs(pipe_id_m=-0.05))

    def test_guardrail_non_positive_liquid_density(self):
        with pytest.raises(ValueError, match="rho_liquid must be positive"):
            solve_beggs_brill_pressure_profile(**_baghewala_common_kwargs(rho_liquid=0.0))

    def test_guardrail_non_positive_gas_density(self):
        with pytest.raises(ValueError, match="rho_gas must be positive"):
            solve_beggs_brill_pressure_profile(**_baghewala_common_kwargs(rho_gas=0.0))

        with pytest.raises(ValueError, match="rho_gas must be positive"):
            bad_rho_gas = np.full(13, 1.2)
            bad_rho_gas[5] = -1.0
            solve_beggs_brill_pressure_profile(**_baghewala_common_kwargs(rho_gas=bad_rho_gas))

    def test_guardrail_non_positive_viscosities(self):
        with pytest.raises(ValueError, match="mu_liquid_cp must be positive"):
            solve_beggs_brill_pressure_profile(**_baghewala_common_kwargs(mu_liquid_cp=0.0))

        with pytest.raises(ValueError, match="mu_gas_cp must be positive"):
            solve_beggs_brill_pressure_profile(**_baghewala_common_kwargs(mu_gas_cp=-0.01))

    def test_guardrail_non_positive_surface_tension(self):
        with pytest.raises(ValueError, match="surface_tension_n_m must be positive"):
            solve_beggs_brill_pressure_profile(**_baghewala_common_kwargs(surface_tension_n_m=0.0))

    def test_guardrail_non_positive_reference_pressure(self):
        with pytest.raises(ValueError, match="P_reference_pa must be positive"):
            solve_beggs_brill_pressure_profile(**_baghewala_common_kwargs(P_reference_pa=0.0))

    def test_guardrail_negative_roughness(self):
        with pytest.raises(ValueError, match="pipe_roughness_m must be non-negative"):
            solve_beggs_brill_pressure_profile(**_baghewala_common_kwargs(pipe_roughness_m=-1e-5))

    def test_guardrail_depth_array_too_short(self):
        with pytest.raises(ValueError, match="at least 2 nodes"):
            solve_beggs_brill_pressure_profile(**_baghewala_common_kwargs(depth_from_surface_m=np.array([0.0])))

    def test_guardrail_depth_array_not_increasing(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            solve_beggs_brill_pressure_profile(
                **_baghewala_common_kwargs(depth_from_surface_m=np.array([0.0, 500.0, 300.0, 1000.0]))
            )

        with pytest.raises(ValueError, match="strictly increasing"):
            solve_beggs_brill_pressure_profile(
                **_baghewala_common_kwargs(depth_from_surface_m=np.array([0.0, 500.0, 500.0, 1000.0]))
            )

    def test_guardrail_inclination_out_of_range(self):
        with pytest.raises(ValueError, match="inclination_deg must be within"):
            solve_beggs_brill_pressure_profile(**_baghewala_common_kwargs(inclination_deg=95.0))

        with pytest.raises(ValueError, match="inclination_deg must be within"):
            solve_beggs_brill_pressure_profile(**_baghewala_common_kwargs(inclination_deg=-1.0))

    def test_guardrail_mismatched_array_shape(self):
        depths = np.linspace(0.0, 1200.0, 13)
        with pytest.raises(ValueError, match="cannot be broadcast"):
            solve_beggs_brill_pressure_profile(
                **_baghewala_common_kwargs(
                    depth_from_surface_m=depths, rho_gas=np.array([1.2, 1.3, 1.4])
                )
            )

        with pytest.raises(ValueError, match="cannot be broadcast"):
            solve_beggs_brill_pressure_profile(
                **_baghewala_common_kwargs(
                    depth_from_surface_m=depths, q_gas_m3_s=np.array([0.001, 0.002])
                )
            )

        with pytest.raises(ValueError, match="cannot be broadcast"):
            solve_beggs_brill_pressure_profile(
                **_baghewala_common_kwargs(
                    depth_from_surface_m=depths, inclination_deg=np.array([90.0, 45.0])
                )
            )