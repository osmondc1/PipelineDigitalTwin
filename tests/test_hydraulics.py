"""Tests for the hydraulic calculations."""

import math

import pytest

from pipeline.hydraulics import (
    elevation_pressure_change,
    flow_regime,
    fluid_velocity,
    friction_factor,
    friction_pressure_loss,
    pipe_area,
    reynolds_number,
    segment_outlet_pressure,
)


# --- pipe_area ---


def test_area_matches_hand_calculation() -> None:
    # Hand check: A = pi * 0.5^2 / 4 = 0.19635 m^2
    assert pipe_area(0.5) == pytest.approx(0.19635, rel=1e-4)


def test_area_of_one_metre_pipe_is_pi_over_four() -> None:
    assert pipe_area(1.0) == pytest.approx(math.pi / 4)


def test_doubling_diameter_quadruples_area() -> None:
    assert pipe_area(1.0) == pytest.approx(4 * pipe_area(0.5))


@pytest.mark.parametrize("bad_diameter", [0.0, -0.5, math.nan, math.inf])
def test_area_rejects_impossible_diameter(bad_diameter: float) -> None:
    with pytest.raises(ValueError, match="internal_diameter"):
        pipe_area(bad_diameter)


# --- fluid_velocity ---


def test_velocity_matches_hand_calculation() -> None:
    # Hand check: v = 0.25 / 0.19635 = 1.2732 m/s
    assert fluid_velocity(0.25, 0.19635) == pytest.approx(1.2732, rel=1e-4)


def test_doubling_flow_rate_doubles_velocity() -> None:
    assert fluid_velocity(0.5, 0.2) == pytest.approx(2 * fluid_velocity(0.25, 0.2))


def test_zero_flow_gives_zero_velocity() -> None:
    assert fluid_velocity(0.0, 0.2) == 0.0


@pytest.mark.parametrize("bad_flow", [-0.25, math.nan, math.inf])
def test_velocity_rejects_impossible_flow_rate(bad_flow: float) -> None:
    with pytest.raises(ValueError, match="flow_rate"):
        fluid_velocity(bad_flow, 0.2)


@pytest.mark.parametrize("bad_area", [0.0, -0.2, math.nan, math.inf])
def test_velocity_rejects_impossible_area(bad_area: float) -> None:
    with pytest.raises(ValueError, match="area"):
        fluid_velocity(0.25, bad_area)


# --- reynolds_number ---


def test_reynolds_matches_hand_calculation_for_water() -> None:
    # Hand check: Re = 1000 * 1 * 0.1 / 0.001 = 100,000
    assert reynolds_number(1000.0, 1.0, 0.1, 0.001) == pytest.approx(100_000.0)


def test_reynolds_for_thick_oil_is_laminar() -> None:
    # Hand check: Re = 900 * 1 * 0.5 / 1.0 = 450, well below 2300
    assert reynolds_number(900.0, 1.0, 0.5, 1.0) == pytest.approx(450.0)


def test_zero_velocity_gives_zero_reynolds_number() -> None:
    assert reynolds_number(850.0, 0.0, 0.5, 0.01) == 0.0


@pytest.mark.parametrize(
    "bad_input",
    [
        {"density": 0.0},
        {"density": -850.0},
        {"density": math.nan},
        {"velocity": -1.0},
        {"velocity": math.nan},
        {"internal_diameter": 0.0},
        {"internal_diameter": math.inf},
        {"dynamic_viscosity": 0.0},
        {"dynamic_viscosity": -0.01},
        {"dynamic_viscosity": math.nan},
    ],
)
def test_reynolds_rejects_impossible_inputs(bad_input: dict[str, float]) -> None:
    values = {
        "density": 850.0,
        "velocity": 1.27,
        "internal_diameter": 0.5,
        "dynamic_viscosity": 0.01,
    }
    values.update(bad_input)
    field = next(iter(bad_input))

    with pytest.raises(ValueError, match=field):
        reynolds_number(**values)


# --- All three functions together ---


def test_worked_example_from_flow_rate_to_reynolds_number() -> None:
    # Oil at 850 kg/m^3 and 0.01 Pa·s, flowing at 0.25 m^3/s in a 0.5 m pipe.
    # Hand check: A = 0.19635 m^2, v = 1.2732 m/s,
    #             Re = 850 * 1.2732 * 0.5 / 0.01 = 54,113
    area = pipe_area(0.5)
    velocity = fluid_velocity(0.25, area)
    reynolds = reynolds_number(850.0, velocity, 0.5, 0.01)

    assert area == pytest.approx(0.19635, rel=1e-4)
    assert velocity == pytest.approx(1.2732, rel=1e-4)
    assert reynolds == pytest.approx(54_113.0, rel=1e-4)


# --- friction_factor: helper for independent checking ---


def colebrook_friction_factor(reynolds: float, relative_roughness: float) -> float:
    """Solve the Colebrook equation by repeated substitution.

    1 / sqrt(f) = -2 * log10(rr / 3.7 + 2.51 / (Re * sqrt(f)))

    This is deliberately separate from the model code, so that it acts as an
    independent reference. f appears on both sides, so we guess f, calculate
    the right-hand side, and repeat until the answer stops changing.
    """
    f = 0.02
    for _ in range(100):
        right_side = -2 * math.log10(
            relative_roughness / 3.7 + 2.51 / (reynolds * math.sqrt(f))
        )
        f = 1 / right_side**2
    return f


# --- friction_factor: laminar flow ---


def test_laminar_friction_matches_hand_calculation() -> None:
    # Hand check: f = 64 / 1000 = 0.064
    assert friction_factor(1000.0, 0.0001) == pytest.approx(0.064)


def test_laminar_friction_ignores_roughness() -> None:
    assert friction_factor(1000.0, 0.0) == friction_factor(1000.0, 0.05)


def test_laminar_formula_applies_at_regime_limit() -> None:
    # Hand check: f = 64 / 2300 = 0.027826
    assert friction_factor(2300.0, 0.0001) == pytest.approx(0.027826, rel=1e-4)


# --- friction_factor: turbulent flow ---


def test_turbulent_friction_matches_worked_example() -> None:
    # Hand check with Re = 54,113 and rr = 45e-6 / 0.5 = 9e-5:
    # f = 0.25 / [log10(2.432e-5 + 3.154e-4)]^2 = 0.0208
    assert friction_factor(54_113.0, 9e-5) == pytest.approx(0.0208, rel=0.005)


@pytest.mark.parametrize(
    "reynolds, relative_roughness, published_f",
    [
        (1e5, 0.0, 0.0180),  # Smooth pipe
        (1e5, 0.001, 0.0222),  # Moderately rough pipe
        (1e8, 0.01, 0.0380),  # Fully rough pipe
    ],
)
def test_turbulent_friction_matches_moody_chart(
    reynolds: float, relative_roughness: float, published_f: float
) -> None:
    result = friction_factor(reynolds, relative_roughness)

    assert result == pytest.approx(published_f, rel=0.015)


@pytest.mark.parametrize("reynolds", [1e4, 1e5, 1e6, 1e7])
@pytest.mark.parametrize("relative_roughness", [1e-6, 1e-5, 1e-4, 1e-3])
def test_swamee_jain_within_1_percent_for_realistic_pipelines(
    reynolds: float, relative_roughness: float
) -> None:
    # The conditions a steel transmission pipeline normally operates in.
    expected = colebrook_friction_factor(reynolds, relative_roughness)

    result = friction_factor(reynolds, relative_roughness)

    assert result == pytest.approx(expected, rel=0.01)


@pytest.mark.parametrize("reynolds", [5e3, 1e4, 1e5, 1e6, 1e7, 1e8])
@pytest.mark.parametrize("relative_roughness", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
def test_swamee_jain_within_3_percent_across_rated_range(
    reynolds: float, relative_roughness: float
) -> None:
    # Measured worst case is 2.8%, at Re = 5000 and relative roughness = 0.01.
    expected = colebrook_friction_factor(reynolds, relative_roughness)

    result = friction_factor(reynolds, relative_roughness)

    assert result == pytest.approx(expected, rel=0.03)


def test_friction_falls_as_reynolds_number_rises() -> None:
    assert friction_factor(1e6, 0.0001) < friction_factor(1e5, 0.0001)


def test_friction_rises_with_roughness() -> None:
    assert friction_factor(1e5, 0.001) > friction_factor(1e5, 0.0001)


# --- friction_factor: transitional flow ---


def test_transitional_midpoint_is_average_of_regime_limits() -> None:
    # Re = 3150 is exactly halfway between 2300 and 4000.
    laminar_limit_f = friction_factor(2300.0, 0.0001)
    turbulent_limit_f = friction_factor(4000.0, 0.0001)

    result = friction_factor(3150.0, 0.0001)

    assert result == pytest.approx((laminar_limit_f + turbulent_limit_f) / 2)


@pytest.mark.parametrize("limit", [2300.0, 4000.0])
def test_friction_has_no_jump_at_regime_limits(limit: float) -> None:
    just_below = friction_factor(limit - 1e-6, 0.0001)
    just_above = friction_factor(limit + 1e-6, 0.0001)

    assert just_below == pytest.approx(just_above, rel=1e-6)


# --- friction_factor: boundary conditions ---


def test_maximum_relative_roughness_is_allowed() -> None:
    assert friction_factor(1e5, 0.05) > 0


def test_smooth_pipe_is_allowed_in_turbulent_flow() -> None:
    assert friction_factor(1e5, 0.0) > 0


# --- friction_factor: invalid inputs ---


@pytest.mark.parametrize("bad_reynolds", [0.0, -1000.0, math.nan, math.inf])
def test_friction_rejects_impossible_reynolds_number(bad_reynolds: float) -> None:
    with pytest.raises(ValueError, match="reynolds"):
        friction_factor(bad_reynolds, 0.0001)


@pytest.mark.parametrize("bad_roughness", [-0.0001, 0.0501, math.nan, math.inf])
def test_friction_rejects_impossible_relative_roughness(bad_roughness: float) -> None:
    with pytest.raises(ValueError, match="relative_roughness"):
        friction_factor(1e5, bad_roughness)


def test_roughness_in_wrong_units_is_caught() -> None:
    # 0.045 mm entered as 0.045 m, in a 0.5 m pipe, gives 0.045 / 0.5 = 0.09.
    with pytest.raises(ValueError, match="metres"):
        friction_factor(1e5, 0.045 / 0.5)


# --- friction_pressure_loss ---


def worked_example_loss(**changes: float) -> float:
    """Friction loss for the worked example, replacing any values a test changes."""
    values = {
        "density": 850.0,  # kg/m^3
        "velocity": 1.27324,  # m/s, from Q = 0.25 m^3/s
        "length": 100_000.0,  # m
        "internal_diameter": 0.5,  # m
        "dynamic_viscosity": 0.01,  # Pa·s
        "roughness": 45e-6,  # m
    }
    values.update(changes)
    return friction_pressure_loss(**values)


def test_friction_loss_matches_hand_calculation() -> None:
    # Hand check: f = 0.02078, L / D = 200,000,
    # rho * v^2 / 2 = 850 * 1.27324^2 / 2 = 689.0 Pa
    # delta_p = 0.02078 * 200,000 * 689.0 = 2.863e6 Pa, about 28.6 bar
    assert worked_example_loss() == pytest.approx(2.863e6, rel=0.002)


def test_friction_loss_agrees_with_colebrook_within_1_percent() -> None:
    f = colebrook_friction_factor(54_113.0, 9e-5)
    expected = f * (100_000.0 / 0.5) * (850.0 * 1.27324**2 / 2)

    assert worked_example_loss() == pytest.approx(expected, rel=0.01)


def test_laminar_loss_matches_hagen_poiseuille() -> None:
    # Hagen-Poiseuille is an exact solution for laminar pipe flow:
    # delta_p = 128 * mu * L * Q / (pi * D^4)
    density, viscosity, length, diameter, flow = 900.0, 1.0, 1000.0, 0.5, 0.1
    expected = 128 * viscosity * length * flow / (math.pi * diameter**4)
    velocity = fluid_velocity(flow, pipe_area(diameter))

    result = friction_pressure_loss(
        density, velocity, length, diameter, viscosity, 45e-6
    )

    assert result == pytest.approx(expected)


def test_doubling_length_doubles_friction_loss() -> None:
    doubled = worked_example_loss(length=200_000.0)

    assert doubled == pytest.approx(2 * worked_example_loss())


def test_doubling_velocity_stays_between_smooth_and_rough_limits() -> None:
    # Smooth-pipe turbulent flow (Blasius): loss scales with v^1.75, giving
    # 2^1.75 = 3.36. Fully rough flow: loss scales with v^2, giving 4.
    # Our pipe is in between, so its ratio must lie between those limits.
    ratio = worked_example_loss(velocity=2 * 1.27324) / worked_example_loss()

    assert 2**1.75 < ratio < 4.0


def test_zero_velocity_gives_zero_friction_loss() -> None:
    assert worked_example_loss(velocity=0.0) == 0.0


def test_smooth_pipe_loses_less_than_rough_pipe() -> None:
    assert worked_example_loss(roughness=0.0) < worked_example_loss(roughness=45e-6)


@pytest.mark.parametrize(
    "bad_input",
    [
        {"length": 0.0},
        {"length": -100.0},
        {"length": math.nan},
        {"velocity": -1.0},
        {"density": 0.0},
        {"internal_diameter": 0.0},
        {"dynamic_viscosity": 0.0},
        {"roughness": -1e-6},
        {"roughness": math.nan},
    ],
)
def test_friction_loss_rejects_impossible_inputs(bad_input: dict[str, float]) -> None:
    field = next(iter(bad_input))

    with pytest.raises(ValueError, match=field):
        worked_example_loss(**bad_input)


def test_friction_loss_catches_roughness_in_wrong_units() -> None:
    with pytest.raises(ValueError, match="metres"):
        worked_example_loss(roughness=0.045)


# --- elevation_pressure_change ---


def test_elevation_change_matches_hand_calculation() -> None:
    # Hand check: 850 * 9.80665 * 30 = 250,070 Pa, about 2.5 bar
    assert elevation_pressure_change(850.0, 30.0) == pytest.approx(250_070.0, rel=1e-4)


def test_ten_metres_of_water_is_about_one_bar() -> None:
    # A well-known rule of thumb.
    assert elevation_pressure_change(1000.0, 10.0) == pytest.approx(1e5, rel=0.02)


def test_downhill_gives_negative_pressure_change() -> None:
    assert elevation_pressure_change(850.0, -30.0) == pytest.approx(-250_070.0, rel=1e-4)


def test_level_pipe_gives_zero_pressure_change() -> None:
    assert elevation_pressure_change(850.0, 0.0) == 0.0


@pytest.mark.parametrize("bad_density", [0.0, -850.0, math.nan, math.inf])
def test_elevation_change_rejects_impossible_density(bad_density: float) -> None:
    with pytest.raises(ValueError, match="density"):
        elevation_pressure_change(bad_density, 30.0)


@pytest.mark.parametrize("bad_change", [math.nan, math.inf, -math.inf])
def test_elevation_change_rejects_non_finite_height(bad_change: float) -> None:
    with pytest.raises(ValueError, match="elevation_change"):
        elevation_pressure_change(850.0, bad_change)


# --- segment_outlet_pressure ---


def test_outlet_pressure_matches_hand_calculation() -> None:
    # Hand check, in bar: 50 + 30 - 5 - 2.5 = 72.5
    result = segment_outlet_pressure(50e5, 5e5, 2.5e5, pump_boost=30e5)

    assert result == pytest.approx(72.5e5)


def test_pump_boost_defaults_to_zero() -> None:
    # Hand check, in bar: 50 - 5 - 2.5 = 42.5
    assert segment_outlet_pressure(50e5, 5e5, 2.5e5) == pytest.approx(42.5e5)


def test_friction_reduces_outlet_pressure() -> None:
    assert segment_outlet_pressure(50e5, 5e5, 0.0) < 50e5


def test_uphill_reduces_and_downhill_raises_outlet_pressure() -> None:
    uphill = elevation_pressure_change(850.0, 30.0)
    downhill = elevation_pressure_change(850.0, -30.0)

    assert segment_outlet_pressure(50e5, 0.0, uphill) < 50e5
    assert segment_outlet_pressure(50e5, 0.0, downhill) > 50e5


def test_stationary_fluid_only_changes_pressure_with_height() -> None:
    # With no flow there is no friction, so only the hydrostatic term remains.
    friction = worked_example_loss(velocity=0.0)
    elevation = elevation_pressure_change(850.0, 30.0)

    result = segment_outlet_pressure(50e5, friction, elevation)

    assert result == pytest.approx(50e5 - 250_070.0, rel=1e-6)


def test_outlet_pressure_may_be_negative() -> None:
    # Flagging this is the pipeline model's job, so it must not raise here.
    assert segment_outlet_pressure(1e5, 5e5, 0.0) < 0


@pytest.mark.parametrize(
    "bad_input",
    [
        {"inlet_pressure": math.nan},
        {"inlet_pressure": math.inf},
        {"friction_loss": -1.0},
        {"friction_loss": math.nan},
        {"elevation_change_pressure": math.nan},
        {"pump_boost": -1.0},
        {"pump_boost": math.nan},
    ],
)
def test_outlet_pressure_rejects_impossible_inputs(bad_input: dict[str, float]) -> None:
    values = {
        "inlet_pressure": 50e5,
        "friction_loss": 5e5,
        "elevation_change_pressure": 2.5e5,
        "pump_boost": 0.0,
    }
    values.update(bad_input)
    field = next(iter(bad_input))

    with pytest.raises(ValueError, match=field):
        segment_outlet_pressure(**values)


# --- flow_regime ---


@pytest.mark.parametrize(
    "reynolds, expected",
    [
        (0.0, "no flow"),
        (1000.0, "laminar"),
        (2300.0, "laminar"),
        (2300.1, "transitional"),
        (3999.9, "transitional"),
        (4000.0, "turbulent"),
        (54_113.0, "turbulent"),
    ],
)
def test_flow_regime_classifies_reynolds_number(
    reynolds: float, expected: str
) -> None:
    assert flow_regime(reynolds) == expected


@pytest.mark.parametrize("bad_reynolds", [-1.0, math.nan, math.inf])
def test_flow_regime_rejects_impossible_reynolds_number(bad_reynolds: float) -> None:
    with pytest.raises(ValueError, match="reynolds"):
        flow_regime(bad_reynolds)
