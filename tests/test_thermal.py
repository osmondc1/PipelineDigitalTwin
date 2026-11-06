"""Tests for the temperature model and its effect on pressure."""

import math

import pytest

from pipeline.fluid_properties import Fluid
from pipeline.parameters import PipeSegment
from pipeline.pipeline_model import calculate_pipeline
from pipeline.thermal import (
    ThermalProperties,
    calculate_temperature_profile,
    segment_outlet_temperature,
    segment_viscosities,
    viscosity_at_temperature,
)

BAR = 1e5  # Pa
DENSITY = 850.0  # kg/m^3
FLOW_RATE = 0.25  # m^3/s
MASS_FLOW_RATE = DENSITY * FLOW_RATE  # 212.5 kg/s
OIL_IN = 313.15  # K, which is 40 °C
GROUND = 288.15  # K, which is 15 °C


def make_thermal(**changes: float) -> ThermalProperties:
    """Build illustrative thermal properties, replacing any a test changes."""
    values = {
        "specific_heat": 2000.0,  # J/(kg·K)
        "heat_transfer_coefficient": 2.0,  # W/(m^2·K)
        "ground_temperature": GROUND,
        "reference_temperature": 303.15,  # K, which is 30 °C
        "viscosity_temperature_coefficient": 0.03,  # 1/K
    }
    values.update(changes)
    return ThermalProperties(**values)


def make_segment(name: str, length_km: float) -> PipeSegment:
    """Build a level 0.5 m segment."""
    return PipeSegment(name, length_km * 1000, 0.5, 45e-6, 100.0, 100.0)


def outlet_after(length_km: float, **thermal_changes: float) -> float:
    """Return the outlet temperature of one segment of the given length."""
    return segment_outlet_temperature(
        OIL_IN, make_thermal(**thermal_changes), 0.5, length_km * 1000, MASS_FLOW_RATE
    )


OIL = Fluid(DENSITY, 0.01, 30_000.0)


# --- segment_outlet_temperature ---


def test_outlet_temperature_matches_hand_calculation() -> None:
    # Hand check over 100 km:
    # exponent = 2 * pi * 0.5 * 100,000 / (212.5 * 2000) = 0.7392
    # T_out = 15 + 25 * exp(-0.7392) = 15 + 25 * 0.4775 = 26.94 °C = 300.09 K
    assert outlet_after(100.0) == pytest.approx(300.09, abs=0.01)


def test_oil_cools_toward_ground_but_never_reaches_it() -> None:
    temperature = outlet_after(100.0)

    assert GROUND < temperature < OIL_IN


def test_very_long_pipe_reaches_ground_temperature() -> None:
    assert outlet_after(10_000.0) == pytest.approx(GROUND, abs=1e-6)


def test_cold_oil_warms_toward_ground() -> None:
    cold_oil = 278.15
    thermal = make_thermal()

    result = segment_outlet_temperature(
        cold_oil, thermal, 0.5, 100_000.0, MASS_FLOW_RATE
    )

    assert cold_oil < result < GROUND


def test_oil_at_ground_temperature_does_not_change() -> None:
    result = segment_outlet_temperature(
        GROUND, make_thermal(), 0.5, 100_000.0, MASS_FLOW_RATE
    )

    assert result == pytest.approx(GROUND)


def test_faster_flow_loses_less_heat() -> None:
    thermal = make_thermal()

    slow = segment_outlet_temperature(OIL_IN, thermal, 0.5, 100_000.0, 100.0)
    fast = segment_outlet_temperature(OIL_IN, thermal, 0.5, 100_000.0, 300.0)

    assert fast > slow


def test_perfect_insulation_keeps_temperature() -> None:
    assert outlet_after(100.0, heat_transfer_coefficient=0.0) == OIL_IN


def test_zero_flow_settles_to_ground_temperature() -> None:
    result = segment_outlet_temperature(OIL_IN, make_thermal(), 0.5, 100_000.0, 0.0)

    assert result == GROUND


def test_perfect_insulation_at_zero_flow_keeps_temperature() -> None:
    thermal = make_thermal(heat_transfer_coefficient=0.0)

    assert segment_outlet_temperature(OIL_IN, thermal, 0.5, 100_000.0, 0.0) == OIL_IN


@pytest.mark.parametrize(
    "field, args",
    [
        ("inlet_temperature", (0.0, 0.5, 1000.0, 200.0)),
        ("inlet_temperature", (math.nan, 0.5, 1000.0, 200.0)),
        ("internal_diameter", (OIL_IN, 0.0, 1000.0, 200.0)),
        ("length", (OIL_IN, 0.5, -1.0, 200.0)),
        ("mass_flow_rate", (OIL_IN, 0.5, 1000.0, -1.0)),
    ],
)
def test_outlet_temperature_rejects_impossible_inputs(
    field: str, args: tuple[float, float, float, float]
) -> None:
    inlet, diameter, length, mass_flow = args

    with pytest.raises(ValueError, match=field):
        segment_outlet_temperature(inlet, make_thermal(), diameter, length, mass_flow)


# --- viscosity_at_temperature ---


def test_viscosity_at_reference_temperature_is_unchanged() -> None:
    assert viscosity_at_temperature(0.01, 303.15, make_thermal()) == pytest.approx(0.01)


def test_viscosity_matches_hand_calculation() -> None:
    # Hand check, 10 K hotter: 0.01 * exp(-0.03 * 10) = 0.0074082 Pa·s
    result = viscosity_at_temperature(0.01, 313.15, make_thermal())

    assert result == pytest.approx(0.0074082, rel=1e-4)


def test_colder_oil_is_more_viscous() -> None:
    thermal = make_thermal()

    cold = viscosity_at_temperature(0.01, 293.15, thermal)
    warm = viscosity_at_temperature(0.01, 313.15, thermal)

    assert cold > 0.01 > warm


def test_zero_coefficient_means_no_temperature_effect() -> None:
    thermal = make_thermal(viscosity_temperature_coefficient=0.0)

    assert viscosity_at_temperature(0.01, 280.0, thermal) == 0.01


@pytest.mark.parametrize(
    "field, viscosity, temperature",
    [
        ("reference_viscosity", 0.0, 300.0),
        ("reference_viscosity", -0.01, 300.0),
        ("temperature", 0.01, 0.0),
        ("temperature", 0.01, math.nan),
    ],
)
def test_viscosity_rejects_impossible_inputs(
    field: str, viscosity: float, temperature: float
) -> None:
    with pytest.raises(ValueError, match=field):
        viscosity_at_temperature(viscosity, temperature, make_thermal())


# --- ThermalProperties ---


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("specific_heat", 0.0),
        ("specific_heat", math.nan),
        ("heat_transfer_coefficient", -1.0),
        ("ground_temperature", 0.0),
        ("reference_temperature", -1.0),
        ("viscosity_temperature_coefficient", -0.01),
    ],
)
def test_impossible_thermal_properties_are_rejected(
    field: str, bad_value: float
) -> None:
    with pytest.raises(ValueError, match=field):
        make_thermal(**{field: bad_value})


# --- calculate_temperature_profile ---


def test_splitting_a_line_does_not_change_the_outlet_temperature() -> None:
    whole = [make_segment("S1", 100.0)]
    split = [make_segment("S1", 30.0), make_segment("S2", 70.0)]
    thermal = make_thermal()

    whole_result = calculate_temperature_profile(
        whole, thermal, DENSITY, FLOW_RATE, OIL_IN
    )
    split_result = calculate_temperature_profile(
        split, thermal, DENSITY, FLOW_RATE, OIL_IN
    )

    assert split_result[-1].outlet_temperature == pytest.approx(
        whole_result[-1].outlet_temperature
    )


def test_each_segment_starts_at_the_previous_outlet_temperature() -> None:
    segments = [make_segment("S1", 30.0), make_segment("S2", 70.0)]

    first, second = calculate_temperature_profile(
        segments, make_thermal(), DENSITY, FLOW_RATE, OIL_IN
    )

    assert first.inlet_temperature == OIL_IN
    assert second.inlet_temperature == first.outlet_temperature


def test_average_temperature_is_the_midpoint() -> None:
    result = calculate_temperature_profile(
        [make_segment("S1", 100.0)], make_thermal(), DENSITY, FLOW_RATE, OIL_IN
    )[0]

    expected = (result.inlet_temperature + result.outlet_temperature) / 2
    assert result.average_temperature == pytest.approx(expected)


@pytest.mark.parametrize(
    "field, density, flow_rate",
    [("density", 0.0, 0.25), ("flow_rate", 850.0, -0.25)],
)
def test_temperature_profile_rejects_impossible_inputs(
    field: str, density: float, flow_rate: float
) -> None:
    with pytest.raises(ValueError, match=field):
        calculate_temperature_profile(
            [make_segment("S1", 10.0)], make_thermal(), density, flow_rate, OIL_IN
        )


# --- Viscosity input to the pipeline model ---


def test_leaving_out_viscosities_uses_the_fluid_value() -> None:
    segments = [make_segment("S1", 50.0), make_segment("S2", 50.0)]

    default = calculate_pipeline(OIL, FLOW_RATE, 60 * BAR, segments)
    explicit = calculate_pipeline(
        OIL, FLOW_RATE, 60 * BAR, segments, segment_viscosities=[0.01, 0.01]
    )

    assert explicit.outlet_pressure == default.outlet_pressure


def test_uniform_viscosities_match_a_thicker_fluid() -> None:
    segments = [make_segment("S1", 50.0), make_segment("S2", 50.0)]
    thicker = Fluid(DENSITY, 0.02, 30_000.0)

    expected = calculate_pipeline(thicker, FLOW_RATE, 60 * BAR, segments)
    result = calculate_pipeline(
        OIL, FLOW_RATE, 60 * BAR, segments, segment_viscosities=[0.02, 0.02]
    )

    assert result.outlet_pressure == pytest.approx(expected.outlet_pressure)


def test_each_segment_uses_its_own_viscosity() -> None:
    segments = [make_segment("S1", 50.0), make_segment("S2", 50.0)]

    result = calculate_pipeline(
        OIL, FLOW_RATE, 60 * BAR, segments, segment_viscosities=[0.01, 0.02]
    )

    s1, s2 = result.segments
    assert s1.reynolds == pytest.approx(2 * s2.reynolds)
    assert s2.friction_loss > s1.friction_loss


def test_wrong_number_of_viscosities_is_rejected() -> None:
    segments = [make_segment("S1", 50.0), make_segment("S2", 50.0)]

    with pytest.raises(ValueError, match="one value per segment"):
        calculate_pipeline(
            OIL, FLOW_RATE, 60 * BAR, segments, segment_viscosities=[0.01]
        )


@pytest.mark.parametrize("bad_viscosity", [0.0, -0.01, math.nan])
def test_impossible_segment_viscosity_is_rejected(bad_viscosity: float) -> None:
    with pytest.raises(ValueError, match="segment_viscosities"):
        calculate_pipeline(
            OIL,
            FLOW_RATE,
            60 * BAR,
            [make_segment("S1", 50.0)],
            segment_viscosities=[bad_viscosity],
        )


# --- Temperature, viscosity and pressure working together ---


def run_with_inlet_temperature(inlet_temperature: float) -> float:
    """Return the outlet pressure for oil entering at the given temperature."""
    segments = [make_segment(f"S{i}", 25.0) for i in range(4)]
    thermal = make_thermal()
    temperatures = calculate_temperature_profile(
        segments, thermal, DENSITY, FLOW_RATE, inlet_temperature
    )
    viscosities = segment_viscosities(temperatures, 0.01, thermal)
    result = calculate_pipeline(
        OIL, FLOW_RATE, 60 * BAR, segments, segment_viscosities=viscosities
    )
    return result.outlet_pressure


def test_viscosity_rises_as_oil_cools_along_the_line() -> None:
    segments = [make_segment("S1", 50.0), make_segment("S2", 50.0)]
    thermal = make_thermal()
    temperatures = calculate_temperature_profile(
        segments, thermal, DENSITY, FLOW_RATE, OIL_IN
    )

    upstream, downstream = segment_viscosities(temperatures, 0.01, thermal)

    assert downstream > upstream


def test_colder_oil_gives_lower_outlet_pressure() -> None:
    assert run_with_inlet_temperature(293.15) < run_with_inlet_temperature(313.15)
