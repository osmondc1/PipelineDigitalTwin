"""Tests for the pressure profile data and chart."""

from pathlib import Path

import pytest

from pipeline.config_loader import PipelineCase, load_case
from pipeline.fluid_properties import Fluid
from pipeline.parameters import PipeSegment, PumpStation
from pipeline.pipeline_model import calculate_pipeline
from pipeline.plotting import plot_pressure_profile, pressure_profile

BASE_CASE_PATH = Path(__file__).resolve().parent.parent / "config" / "base_case.toml"
BAR = 1e5  # Pa


def run_case(case: PipelineCase):
    """Run a case through the model."""
    return calculate_pipeline(
        case.fluid, case.flow_rate, case.inlet_pressure, case.segments, case.pumps
    )


def two_segment_case(boost_bar: float | None = None) -> PipelineCase:
    """Build a level 20 km line, with an optional pump on the second segment."""
    pumps = ()
    if boost_bar is not None:
        pumps = (PumpStation("PS1", "S2", boost_bar * BAR),)
    return PipelineCase(
        name="Test case",
        fluid=Fluid(850.0, 0.01, 30_000.0),
        flow_rate=0.25,
        inlet_pressure=60 * BAR,
        segments=(
            PipeSegment("S1", 10_000.0, 0.5, 45e-6, 100.0, 100.0),
            PipeSegment("S2", 10_000.0, 0.5, 45e-6, 100.0, 100.0),
        ),
        pumps=pumps,
    )


# --- pressure_profile: base case ---


def test_base_case_has_a_point_per_location_and_per_pump() -> None:
    # Hand check: 5 positions (km 0, 30, 55, 80, 100) + 2 pump jumps = 7
    case = load_case(BASE_CASE_PATH)

    assert len(pressure_profile(case, run_case(case))) == 7


def test_base_case_distances_follow_the_segments() -> None:
    case = load_case(BASE_CASE_PATH)

    points = pressure_profile(case, run_case(case))

    distances_km = [point.distance / 1000 for point in points]
    assert distances_km == pytest.approx([0, 0, 30, 55, 55, 80, 100])


def test_base_case_elevations_follow_the_segments() -> None:
    case = load_case(BASE_CASE_PATH)

    points = pressure_profile(case, run_case(case))

    assert [point.elevation for point in points] == [50, 50, 180, 120, 120, 200, 80]


def test_base_case_pressures_match_the_model_results() -> None:
    case = load_case(BASE_CASE_PATH)
    result = run_case(case)
    s1, s2, s3, s4 = result.segments

    points = pressure_profile(case, result)

    assert [point.pressure for point in points] == [
        s1.inlet_pressure,
        s1.pressure_after_pump,
        s1.outlet_pressure,
        s2.outlet_pressure,
        s3.pressure_after_pump,
        s3.outlet_pressure,
        s4.outlet_pressure,
    ]


# --- pressure_profile: pumps ---


def test_pump_appears_as_a_vertical_jump() -> None:
    case = two_segment_case(boost_bar=30.0)

    _, arrival, discharge, _ = pressure_profile(case, run_case(case))

    assert discharge.distance == arrival.distance
    assert discharge.pressure - arrival.pressure == pytest.approx(30 * BAR)
    assert discharge.pump_name == "PS1"
    assert arrival.pump_name is None


def test_line_without_pumps_has_no_jump() -> None:
    case = two_segment_case()

    points = pressure_profile(case, run_case(case))

    assert len(points) == 3
    assert all(point.pump_name is None for point in points)


def test_stopped_pump_adds_no_point() -> None:
    case = two_segment_case(boost_bar=0.0)

    assert len(pressure_profile(case, run_case(case))) == 3


# --- plot_pressure_profile ---


def test_chart_has_a_pressure_panel_and_an_elevation_panel() -> None:
    case = load_case(BASE_CASE_PATH)

    figure = plot_pressure_profile(case, run_case(case))

    pressure_axes, elevation_axes = figure.axes
    assert pressure_axes.get_ylabel() == "Pressure (barg)"
    assert elevation_axes.get_ylabel() == "Elevation (m)"


def test_chart_plots_the_profile_in_barg() -> None:
    # Hand check: 21.87 bar abs at the outlet is 21.87 - 1.01325 = 20.86 barg
    case = load_case(BASE_CASE_PATH)

    figure = plot_pressure_profile(case, run_case(case))

    pressure_line = figure.axes[0].lines[0]
    assert pressure_line.get_ydata()[-1] == pytest.approx(20.86, rel=0.003)


def test_chart_labels_each_running_pump() -> None:
    case = load_case(BASE_CASE_PATH)

    figure = plot_pressure_profile(case, run_case(case))

    labels = [text.get_text() for text in figure.axes[0].texts]
    assert "PS1: +35 bar" in labels
    assert "PS2: +15 bar" in labels


def test_chart_can_be_saved(tmp_path: Path) -> None:
    case = load_case(BASE_CASE_PATH)
    path = tmp_path / "profile.png"

    plot_pressure_profile(case, run_case(case)).savefig(path)

    assert path.stat().st_size > 0
