"""Tests for the Fluid class."""

import dataclasses
import math

import pytest

from pipeline.fluid_properties import Fluid


def make_fluid(**changes: float) -> Fluid:
    """Build a valid oil, replacing any values a test wants to change."""
    values = {
        "density": 850.0,  # kg/m^3
        "dynamic_viscosity": 0.01,  # Pa·s
        "vapor_pressure": 30_000.0,  # Pa (absolute)
    }
    values.update(changes)
    return Fluid(**values)


# --- Normal operation ---


def test_fluid_stores_its_properties() -> None:
    fluid = make_fluid()

    assert fluid.density == 850.0
    assert fluid.dynamic_viscosity == 0.01
    assert fluid.vapor_pressure == 30_000.0


def test_kinematic_viscosity_matches_hand_calculation() -> None:
    # Hand check: nu = mu / rho = 0.01 / 850 = 1.17647e-5 m^2/s
    fluid = make_fluid()

    assert fluid.kinematic_viscosity == pytest.approx(1.17647e-5, rel=1e-5)


def test_water_kinematic_viscosity_matches_known_value() -> None:
    # Published value for water at 20 °C is about 1.0e-6 m^2/s.
    water = Fluid(density=998.0, dynamic_viscosity=0.001, vapor_pressure=2_340.0)

    assert water.kinematic_viscosity == pytest.approx(1.0e-6, rel=0.01)


def test_fluid_cannot_be_changed_after_creation() -> None:
    fluid = make_fluid()

    with pytest.raises(dataclasses.FrozenInstanceError):
        fluid.density = 900.0


# --- Boundary conditions ---


def test_zero_vapor_pressure_is_allowed() -> None:
    assert make_fluid(vapor_pressure=0.0).vapor_pressure == 0.0


def test_very_small_positive_viscosity_is_allowed() -> None:
    assert make_fluid(dynamic_viscosity=1e-9).dynamic_viscosity == 1e-9


# --- Invalid inputs ---


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("density", 0.0),
        ("density", -850.0),
        ("density", math.nan),
        ("density", math.inf),
        ("dynamic_viscosity", 0.0),
        ("dynamic_viscosity", -0.01),
        ("dynamic_viscosity", math.nan),
        ("dynamic_viscosity", math.inf),
        ("vapor_pressure", -1.0),
        ("vapor_pressure", math.nan),
        ("vapor_pressure", math.inf),
    ],
)
def test_impossible_values_are_rejected(field: str, bad_value: float) -> None:
    with pytest.raises(ValueError, match=field):
        make_fluid(**{field: bad_value})


def test_text_instead_of_a_number_is_rejected() -> None:
    with pytest.raises(TypeError):
        make_fluid(density="850")
