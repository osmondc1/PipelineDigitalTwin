"""Tests for loading pipeline cases from TOML files."""

import tomllib
from pathlib import Path

import pytest

from pipeline.config_loader import load_case
from pipeline.pipeline_model import calculate_pipeline

BASE_CASE_PATH = Path(__file__).resolve().parent.parent / "config" / "base_case.toml"
BAR = 1e5  # Pa

MINIMAL_CASE = """
name = "Test case"
flow_rate_m3_per_s = 0.25
inlet_pressure_pa_abs = 5000000.0

[fluid]
density_kg_per_m3 = 850.0
dynamic_viscosity_pa_s = 0.01
vapor_pressure_pa_abs = 30000.0

[[segments]]
name = "S1"
length_m = 10000.0
internal_diameter_m = 0.5
roughness_m = 45e-6
inlet_elevation_m = 100.0
outlet_elevation_m = 100.0
"""


def write_case(tmp_path: Path, text: str) -> Path:
    """Write text to a temporary TOML file and return its path."""
    path = tmp_path / "case.toml"
    path.write_text(text)
    return path


# --- The real base case ---


def test_base_case_describes_a_100_km_line() -> None:
    case = load_case(BASE_CASE_PATH)

    assert len(case.segments) == 4
    assert len(case.pumps) == 2
    assert sum(segment.length for segment in case.segments) == pytest.approx(100_000.0)


def test_base_case_outlet_matches_hand_calculation() -> None:
    # Hand check, in bar: 3 + 35 + 15 - 28.63 - 2.50 = 21.87
    # (28.63 is the friction over 100 km, and 2.50 is the net 30 m climb.)
    case = load_case(BASE_CASE_PATH)

    result = calculate_pipeline(
        case.fluid, case.flow_rate, case.inlet_pressure, case.segments, case.pumps
    )

    assert result.outlet_pressure == pytest.approx(21.87 * BAR, rel=0.002)
    assert not result.has_low_pressure


# --- Loading ---


def test_values_are_loaded_into_the_right_fields(tmp_path: Path) -> None:
    case = load_case(write_case(tmp_path, MINIMAL_CASE))

    assert case.name == "Test case"
    assert case.flow_rate == 0.25
    assert case.inlet_pressure == 5_000_000.0
    assert case.fluid.density == 850.0
    assert case.fluid.dynamic_viscosity == 0.01
    assert case.fluid.vapor_pressure == 30_000.0
    assert case.segments[0].name == "S1"
    assert case.segments[0].length == 10_000.0
    assert case.segments[0].internal_diameter == 0.5
    assert case.segments[0].roughness == 45e-6


def test_pumps_are_optional(tmp_path: Path) -> None:
    case = load_case(write_case(tmp_path, MINIMAL_CASE))

    assert case.pumps == ()


# --- Invalid files ---


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_case(tmp_path / "does_not_exist.toml")


@pytest.mark.parametrize(
    "setting",
    ["flow_rate_m3_per_s", "density_kg_per_m3", "length_m"],
)
def test_missing_setting_is_named_in_the_error(tmp_path: Path, setting: str) -> None:
    lines = [line for line in MINIMAL_CASE.splitlines() if not line.startswith(setting)]

    with pytest.raises(ValueError, match=setting):
        load_case(write_case(tmp_path, "\n".join(lines)))


def test_impossible_value_is_rejected(tmp_path: Path) -> None:
    text = MINIMAL_CASE.replace("density_kg_per_m3 = 850.0", "density_kg_per_m3 = -1.0")

    with pytest.raises(ValueError, match="density"):
        load_case(write_case(tmp_path, text))


def test_broken_toml_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(tomllib.TOMLDecodeError):
        load_case(write_case(tmp_path, "this is not toml"))
