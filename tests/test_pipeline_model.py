"""Tests for the multi-segment pipeline model."""

import math

import pytest

from pipeline.fluid_properties import Fluid
from pipeline.parameters import PipeSegment, PumpStation
from pipeline.pipeline_model import calculate_pipeline

BAR = 1e5  # Pa
FLOW_RATE = 0.25  # m^3/s

# Hand-calculated friction loss for this oil, flow rate and pipe [Pa/km].
FRICTION_PER_KM = 28.63 * BAR / 100


def make_oil(**changes: float) -> Fluid:
    """Build the worked-example oil, replacing any values a test changes."""
    values = {
        "density": 850.0,  # kg/m^3
        "dynamic_viscosity": 0.01,  # Pa·s
        "vapor_pressure": 30_000.0,  # Pa, absolute (illustrative value)
    }
    values.update(changes)
    return Fluid(**values)


def make_segment(
    name: str, length_km: float, inlet_elevation: float, outlet_elevation: float
) -> PipeSegment:
    """Build a 0.5 m steel segment with the given length and elevations."""
    return PipeSegment(
        name=name,
        length=length_km * 1000,
        internal_diameter=0.5,
        roughness=45e-6,
        inlet_elevation=inlet_elevation,
        outlet_elevation=outlet_elevation,
    )


def three_segments() -> list[PipeSegment]:
    """A 100 km line: uphill, then downhill, then level."""
    return [
        make_segment("S1", 40.0, 100.0, 130.0),
        make_segment("S2", 30.0, 130.0, 90.0),
        make_segment("S3", 30.0, 90.0, 90.0),
    ]


def boost_at(segment_name: str, bar: float = 30.0) -> PumpStation:
    """Build a pump station at the inlet of the named segment."""
    return PumpStation("PS", segment_name, bar * BAR)


# --- Normal operation ---


def test_single_level_segment_loses_only_friction() -> None:
    # Hand check, in bar: 50 - 28.63 = 21.37
    segments = [make_segment("S1", 100.0, 100.0, 100.0)]

    result = calculate_pipeline(make_oil(), FLOW_RATE, 50 * BAR, segments)

    assert result.outlet_pressure == pytest.approx(21.37 * BAR, rel=0.002)


def test_three_segments_match_hand_calculation() -> None:
    # Hand check, in bar. The height terms are rho * g * dz:
    #   +30 m = 2.501 bar, -40 m = -3.334 bar
    # S1: 60 - 11.452 - 2.501          = 46.047
    # S2: 46.047 + 30 - 8.589 + 3.334  = 70.792
    # S3: 70.792 - 8.589               = 62.203
    result = calculate_pipeline(
        make_oil(), FLOW_RATE, 60 * BAR, three_segments(), [boost_at("S2")]
    )

    s1, s2, s3 = result.segments
    assert s1.outlet_pressure == pytest.approx(46.047 * BAR, rel=0.002)
    assert s2.pressure_after_pump == pytest.approx(76.047 * BAR, rel=0.002)
    assert s2.outlet_pressure == pytest.approx(70.792 * BAR, rel=0.002)
    assert s3.outlet_pressure == pytest.approx(62.203 * BAR, rel=0.002)


def test_whole_line_pressure_balance() -> None:
    # An independent check of the marching loop: the total change across the
    # line must equal the sum of every pump, friction and height term.
    result = calculate_pipeline(
        make_oil(), FLOW_RATE, 60 * BAR, three_segments(), [boost_at("S2")]
    )

    total_change = sum(
        s.pump_boost - s.friction_loss - s.elevation_pressure for s in result.segments
    )

    assert result.outlet_pressure - result.inlet_pressure == pytest.approx(total_change)


def test_each_segment_starts_at_the_previous_outlet_pressure() -> None:
    result = calculate_pipeline(
        make_oil(), FLOW_RATE, 60 * BAR, three_segments(), [boost_at("S2")]
    )

    for upstream, downstream in zip(result.segments, result.segments[1:]):
        assert downstream.inlet_pressure == upstream.outlet_pressure


def test_splitting_a_segment_does_not_change_the_answer() -> None:
    # A uniform pipe must give the same result however it is divided up.
    whole = [make_segment("S1", 100.0, 100.0, 200.0)]
    split = [
        make_segment("S1", 40.0, 100.0, 140.0),
        make_segment("S2", 60.0, 140.0, 200.0),
    ]

    whole_result = calculate_pipeline(make_oil(), FLOW_RATE, 60 * BAR, whole)
    split_result = calculate_pipeline(make_oil(), FLOW_RATE, 60 * BAR, split)

    assert split_result.outlet_pressure == pytest.approx(whole_result.outlet_pressure)


def test_results_stay_in_upstream_to_downstream_order() -> None:
    result = calculate_pipeline(make_oil(), FLOW_RATE, 60 * BAR, three_segments())

    names = [segment.segment_name for segment in result.segments]
    assert names == ["S1", "S2", "S3"]


def test_segment_reports_worked_example_flow_conditions() -> None:
    result = calculate_pipeline(make_oil(), FLOW_RATE, 60 * BAR, three_segments())

    s1 = result.segments[0]
    assert s1.velocity == pytest.approx(1.2732, rel=1e-4)
    assert s1.reynolds == pytest.approx(54_113.0, rel=1e-4)
    assert s1.darcy_friction_factor == pytest.approx(0.0208, rel=0.005)


def test_pump_on_first_segment_acts_on_inlet_pressure() -> None:
    result = calculate_pipeline(
        make_oil(), FLOW_RATE, 10 * BAR, three_segments(), [boost_at("S1")]
    )

    assert result.inlet_pressure == 10 * BAR
    assert result.segments[0].pressure_after_pump == pytest.approx(40 * BAR)


def test_pump_order_in_the_list_does_not_matter() -> None:
    first = [PumpStation("A", "S1", 10 * BAR), PumpStation("B", "S2", 20 * BAR)]
    second = list(reversed(first))

    result_1 = calculate_pipeline(make_oil(), FLOW_RATE, 60 * BAR, three_segments(), first)
    result_2 = calculate_pipeline(make_oil(), FLOW_RATE, 60 * BAR, three_segments(), second)

    assert result_1.outlet_pressure == result_2.outlet_pressure


def test_narrower_segment_has_higher_velocity_and_more_loss() -> None:
    wide = make_segment("S1", 10.0, 100.0, 100.0)
    narrow = PipeSegment("S2", 10_000.0, 0.25, 45e-6, 100.0, 100.0)

    result = calculate_pipeline(make_oil(), FLOW_RATE, 60 * BAR, [wide, narrow])

    s1, s2 = result.segments
    assert s2.velocity == pytest.approx(4 * s1.velocity)
    assert s2.friction_loss > s1.friction_loss


# --- What-if behaviour ---


def test_more_flow_means_lower_outlet_pressure() -> None:
    low = calculate_pipeline(make_oil(), 0.2, 60 * BAR, three_segments())
    high = calculate_pipeline(make_oil(), 0.25, 60 * BAR, three_segments())

    assert high.outlet_pressure < low.outlet_pressure


def test_higher_viscosity_means_lower_outlet_pressure() -> None:
    thin = calculate_pipeline(make_oil(), FLOW_RATE, 60 * BAR, three_segments())
    thick = calculate_pipeline(
        make_oil(dynamic_viscosity=0.05), FLOW_RATE, 60 * BAR, three_segments()
    )

    assert thick.outlet_pressure < thin.outlet_pressure


def test_stopped_pump_lowers_outlet_by_its_boost() -> None:
    running = calculate_pipeline(
        make_oil(), FLOW_RATE, 60 * BAR, three_segments(), [boost_at("S2", 30.0)]
    )
    stopped = calculate_pipeline(
        make_oil(), FLOW_RATE, 60 * BAR, three_segments(), [boost_at("S2", 0.0)]
    )

    assert running.outlet_pressure - stopped.outlet_pressure == pytest.approx(30 * BAR)


def test_very_viscous_oil_runs_in_laminar_flow() -> None:
    result = calculate_pipeline(
        make_oil(dynamic_viscosity=2.0), 0.05, 60 * BAR, three_segments()
    )

    assert result.segments[0].reynolds < 2300


# --- Boundary conditions ---


def test_zero_flow_leaves_only_height_differences() -> None:
    # Hand check: S1 to S3 ends 10 m lower, so pressure rises by
    # 850 * 9.80665 * 10 = 83,357 Pa.
    result = calculate_pipeline(make_oil(), 0.0, 60 * BAR, three_segments())

    assert result.outlet_pressure == pytest.approx(60 * BAR + 83_357.0, rel=1e-5)


def test_zero_flow_has_no_friction_factor() -> None:
    result = calculate_pipeline(make_oil(), 0.0, 60 * BAR, three_segments())

    for segment in result.segments:
        assert segment.darcy_friction_factor is None
        assert segment.friction_loss == 0.0


def test_pipeline_with_no_pumps_is_allowed() -> None:
    result = calculate_pipeline(make_oil(), FLOW_RATE, 60 * BAR, three_segments())

    assert all(segment.pump_boost == 0.0 for segment in result.segments)


# --- Low-pressure flag ---


def test_healthy_line_is_not_flagged() -> None:
    result = calculate_pipeline(make_oil(), FLOW_RATE, 60 * BAR, three_segments())

    assert not result.has_low_pressure


def test_outlet_below_vapor_pressure_is_flagged() -> None:
    # Only 20 bar at the inlet, but 28.63 bar of friction across the line.
    result = calculate_pipeline(make_oil(), FLOW_RATE, 20 * BAR, three_segments())

    assert result.has_low_pressure
    assert result.segments[-1].below_vapor_pressure


def test_only_the_affected_segments_are_flagged() -> None:
    result = calculate_pipeline(make_oil(), FLOW_RATE, 20 * BAR, three_segments())

    assert not result.segments[0].below_vapor_pressure


def test_low_pressure_before_a_pump_is_flagged_even_if_outlet_recovers() -> None:
    # S1 ends below vapour pressure, then a large pump restores the pressure.
    # Hand check, in bar: S2 outlet is about -2.0 + 30 - 8.6 + 3.3 = 22.7.
    result = calculate_pipeline(
        make_oil(), FLOW_RATE, 12 * BAR, three_segments(), [boost_at("S2")]
    )

    s1, s2, _ = result.segments
    assert s1.outlet_pressure < 0
    assert s2.outlet_pressure > 20 * BAR
    assert s2.below_vapor_pressure


def test_pressure_exactly_at_vapor_pressure_is_not_flagged() -> None:
    # A level line with no flow sits at its inlet pressure throughout.
    segments = [make_segment("S1", 10.0, 100.0, 100.0)]

    result = calculate_pipeline(make_oil(), 0.0, 30_000.0, segments)

    assert not result.has_low_pressure


def test_minimum_pressure_can_be_just_before_a_pump() -> None:
    result = calculate_pipeline(
        make_oil(), FLOW_RATE, 60 * BAR, three_segments(), [boost_at("S2")]
    )

    assert result.minimum_pressure == result.segments[0].outlet_pressure


# --- Invalid inputs ---


@pytest.mark.parametrize("bad_flow", [-0.25, math.nan, math.inf])
def test_impossible_flow_rate_is_rejected(bad_flow: float) -> None:
    with pytest.raises(ValueError, match="flow_rate"):
        calculate_pipeline(make_oil(), bad_flow, 60 * BAR, three_segments())


@pytest.mark.parametrize("bad_pressure", [-1.0, math.nan, math.inf])
def test_impossible_inlet_pressure_is_rejected(bad_pressure: float) -> None:
    with pytest.raises(ValueError, match="inlet_pressure"):
        calculate_pipeline(make_oil(), FLOW_RATE, bad_pressure, three_segments())


def test_empty_pipeline_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one segment"):
        calculate_pipeline(make_oil(), FLOW_RATE, 60 * BAR, [])


def test_duplicate_segment_names_are_rejected() -> None:
    segments = [
        make_segment("S1", 10.0, 100.0, 100.0),
        make_segment("S1", 10.0, 100.0, 100.0),
    ]

    with pytest.raises(ValueError, match="unique"):
        calculate_pipeline(make_oil(), FLOW_RATE, 60 * BAR, segments)


def test_elevation_gap_between_segments_is_rejected() -> None:
    segments = [
        make_segment("S1", 10.0, 100.0, 120.0),
        make_segment("S2", 10.0, 125.0, 130.0),
    ]

    with pytest.raises(ValueError, match="elevation gap"):
        calculate_pipeline(make_oil(), FLOW_RATE, 60 * BAR, segments)


def test_pump_on_unknown_segment_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown segment"):
        calculate_pipeline(
            make_oil(), FLOW_RATE, 60 * BAR, three_segments(), [boost_at("S9")]
        )


def test_two_pumps_on_one_segment_are_rejected() -> None:
    pumps = [boost_at("S2"), boost_at("S2")]

    with pytest.raises(ValueError, match="one pump"):
        calculate_pipeline(make_oil(), FLOW_RATE, 60 * BAR, three_segments(), pumps)
