"""Tests for the hydraulic calculations."""

import math

import pytest

from pipeline.hydraulics import fluid_velocity, pipe_area, reynolds_number


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
