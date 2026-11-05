"""Tests for the PipeSegment and PumpStation classes."""

import dataclasses
import math

import pytest

from pipeline.parameters import PipeSegment, PumpStation


def make_segment(**changes: float | str) -> PipeSegment:
    """Build a valid 25 km segment, replacing any values a test wants to change."""
    values = {
        "name": "S1",
        "length": 25_000.0,  # m
        "internal_diameter": 0.5,  # m
        "roughness": 45e-6,  # m
        "inlet_elevation": 100.0,  # m
        "outlet_elevation": 130.0,  # m
    }
    values.update(changes)
    return PipeSegment(**values)


def make_pump(**changes: float | str) -> PumpStation:
    """Build a valid 30 bar pump station, replacing any values a test wants to change."""
    values = {
        "name": "PS1",
        "segment_name": "S1",
        "pressure_boost": 3_000_000.0,  # Pa
    }
    values.update(changes)
    return PumpStation(**values)


# --- PipeSegment: normal operation ---


def test_segment_stores_its_values() -> None:
    segment = make_segment()

    assert segment.name == "S1"
    assert segment.length == 25_000.0
    assert segment.internal_diameter == 0.5
    assert segment.roughness == 45e-6
    assert segment.inlet_elevation == 100.0
    assert segment.outlet_elevation == 130.0


def test_uphill_segment_has_positive_elevation_change() -> None:
    # Hand check: 130 - 100 = +30 m
    assert make_segment().elevation_change == pytest.approx(30.0)


def test_downhill_segment_has_negative_elevation_change() -> None:
    # Hand check: 60 - 100 = -40 m
    segment = make_segment(outlet_elevation=60.0)

    assert segment.elevation_change == pytest.approx(-40.0)


def test_level_segment_has_zero_elevation_change() -> None:
    segment = make_segment(outlet_elevation=100.0)

    assert segment.elevation_change == 0.0


def test_segment_cannot_be_changed_after_creation() -> None:
    segment = make_segment()

    with pytest.raises(dataclasses.FrozenInstanceError):
        segment.length = 1.0


# --- PipeSegment: boundary conditions ---


def test_perfectly_smooth_pipe_is_allowed() -> None:
    assert make_segment(roughness=0.0).roughness == 0.0


def test_elevations_below_datum_are_allowed() -> None:
    # Hand check: -20 - (-50) = +30 m
    segment = make_segment(inlet_elevation=-50.0, outlet_elevation=-20.0)

    assert segment.elevation_change == pytest.approx(30.0)


def test_vertical_segment_is_allowed() -> None:
    # Rising exactly its own length is the steepest possible pipe.
    segment = make_segment(length=30.0, inlet_elevation=0.0, outlet_elevation=30.0)

    assert segment.elevation_change == pytest.approx(segment.length)


# --- PipeSegment: invalid inputs ---


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("length", 0.0),
        ("length", -25_000.0),
        ("length", math.nan),
        ("length", math.inf),
        ("internal_diameter", 0.0),
        ("internal_diameter", -0.5),
        ("internal_diameter", math.nan),
        ("internal_diameter", math.inf),
        ("roughness", -1e-6),
        ("roughness", math.nan),
        ("roughness", math.inf),
        ("inlet_elevation", math.nan),
        ("inlet_elevation", math.inf),
        ("outlet_elevation", math.nan),
        ("outlet_elevation", -math.inf),
    ],
)
def test_impossible_segment_values_are_rejected(field: str, bad_value: float) -> None:
    with pytest.raises(ValueError, match=field):
        make_segment(**{field: bad_value})


@pytest.mark.parametrize("bad_name", ["", "   "])
def test_blank_segment_name_is_rejected(bad_name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        make_segment(name=bad_name)


def test_roughness_equal_to_diameter_is_rejected() -> None:
    with pytest.raises(ValueError, match="roughness"):
        make_segment(roughness=0.5)


def test_climbing_further_than_length_is_rejected() -> None:
    # A 25 km length entered as 25 m, with a 30 m climb.
    with pytest.raises(ValueError, match="elevation change"):
        make_segment(length=25.0)


def test_descending_further_than_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="elevation change"):
        make_segment(length=25.0, inlet_elevation=130.0, outlet_elevation=100.0)


# --- PumpStation ---


def test_pump_stores_its_values() -> None:
    pump = make_pump()

    assert pump.name == "PS1"
    assert pump.segment_name == "S1"
    assert pump.pressure_boost == 3_000_000.0


def test_switched_off_pump_is_allowed() -> None:
    assert make_pump(pressure_boost=0.0).pressure_boost == 0.0


def test_pump_cannot_be_changed_after_creation() -> None:
    pump = make_pump()

    with pytest.raises(dataclasses.FrozenInstanceError):
        pump.pressure_boost = 0.0


@pytest.mark.parametrize("bad_boost", [-1.0, math.nan, math.inf])
def test_impossible_pump_boost_is_rejected(bad_boost: float) -> None:
    with pytest.raises(ValueError, match="pressure_boost"):
        make_pump(pressure_boost=bad_boost)


@pytest.mark.parametrize("field", ["name", "segment_name"])
@pytest.mark.parametrize("bad_name", ["", "   "])
def test_blank_pump_names_are_rejected(field: str, bad_name: str) -> None:
    with pytest.raises(ValueError, match=field):
        make_pump(**{field: bad_name})
