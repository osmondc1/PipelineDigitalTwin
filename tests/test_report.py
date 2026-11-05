"""Tests for the text report and the run script."""

import subprocess
import sys
from pathlib import Path

import pytest

from pipeline.config_loader import PipelineCase
from pipeline.fluid_properties import Fluid
from pipeline.parameters import PipeSegment
from pipeline.pipeline_model import calculate_pipeline
from pipeline.report import format_results, to_bar_gauge

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def report_for(
    inlet_pressure: float = 60e5,
    flow_rate: float = 0.25,
    dynamic_viscosity: float = 0.01,
) -> str:
    """Run a one-segment, level 100 km line and return its report."""
    segment = PipeSegment("S1", 100_000.0, 0.5, 45e-6, 100.0, 100.0)
    case = PipelineCase(
        name="Test case",
        fluid=Fluid(850.0, dynamic_viscosity, 30_000.0),
        flow_rate=flow_rate,
        inlet_pressure=inlet_pressure,
        segments=(segment,),
        pumps=(),
    )
    result = calculate_pipeline(
        case.fluid, case.flow_rate, case.inlet_pressure, case.segments, case.pumps
    )
    return format_results(case, result)


# --- Unit conversion ---


def test_atmospheric_pressure_is_zero_gauge() -> None:
    assert to_bar_gauge(101_325.0) == 0.0


def test_gauge_conversion_matches_hand_calculation() -> None:
    # Hand check: (6,000,000 - 101,325) / 100,000 = 58.98675 barg
    assert to_bar_gauge(6_000_000.0) == pytest.approx(58.98675)


def test_pressure_below_atmosphere_is_negative_gauge() -> None:
    assert to_bar_gauge(50_000.0) < 0


# --- Report contents ---


def test_report_shows_the_key_results() -> None:
    report = report_for()

    assert "Test case" in report
    assert "S1" in report
    assert "100.0 km" in report
    assert "turbulent" in report
    assert "31.37" in report  # Outlet: 60 - 28.63 bar abs.


def test_healthy_line_has_no_warnings() -> None:
    assert "Warnings:\n  None" in report_for()


def test_low_pressure_is_warned_about() -> None:
    assert "below vapour pressure" in report_for(inlet_pressure=20e5)


def test_transitional_flow_is_warned_about() -> None:
    # Re = 850 * 0.509 * 0.5 / 0.07 = about 3090
    # Check the warning text: "transitional" alone also appears as the flow type.
    report = report_for(flow_rate=0.1, dynamic_viscosity=0.07)

    assert "Flow type:  transitional" in report
    assert "interpolated estimate" in report


def test_zero_flow_shows_friction_factor_as_not_available() -> None:
    report = report_for(flow_rate=0.0)

    assert "n/a" in report
    assert "no flow" in report


def test_report_states_it_is_not_for_operational_use() -> None:
    assert "Not for operational use" in report_for()


# --- The run script ---


def test_run_script_works_from_any_folder(tmp_path: Path) -> None:
    # Run the script as a separate program, from an unrelated folder.
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_model.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Illustrative 100 km crude oil pipeline" in completed.stdout
    assert "S4" in completed.stdout
