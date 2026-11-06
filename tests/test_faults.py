"""Tests for sensor faults and simulated readings."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.config_loader import load_case
from sensors.faults import (
    SensorFault,
    apply_fault,
    check_faults,
    fault_window,
    load_faults,
)
from sensors.readings import generate_sensor_readings
from sensors.sensor import Sensor, load_sensors
from simulation.operations import generate_operating_data, load_plan

CONFIG_FOLDER = Path(__file__).resolve().parent.parent / "config"
SEED = 12345

# One day of readings, one per minute, all exactly 50 unless a test adds noise.
DAYS = np.arange(1440) / 1440


def clean_readings() -> np.ndarray:
    """Return a day of perfectly steady readings."""
    return np.full(1440, 50.0)


def noisy_readings() -> np.ndarray:
    """Return a day of readings that all differ from each other."""
    return 50.0 + np.random.default_rng(SEED).normal(0.0, 1.0, 1440)


def make_fault(**changes: object) -> SensorFault:
    """Build a fault for the middle of the day, replacing values a test changes."""
    values = {
        "sensor_id": "PT-001",
        "kind": "bias_shift",
        "start_day": 0.25,
        "end_day": 0.75,
        "magnitude": 2.0,
    }
    values.update(changes)
    return SensorFault(**values)


def apply(readings: np.ndarray, **changes: object) -> np.ndarray:
    """Apply a fault to some readings."""
    return apply_fault(
        readings, make_fault(**changes), DAYS, np.random.default_rng(SEED)
    )


def make_sensor(sensor_id: str = "PT-001", **changes: float) -> Sensor:
    """Build a perfect sensor, replacing any values a test changes."""
    values = {
        "sensor_id": sensor_id,
        "quantity": "pressure",
        "location": "Inlet",
        "location_km": 0.0,
        "sampling_interval": 60.0,
        "noise_std": 0.0,
        "bias_std": 0.0,
        "missing_probability": 0.0,
    }
    values.update(changes)
    return Sensor(**values)


def make_truth(*sensor_ids: str) -> pd.DataFrame:
    """Build one day of true data, with every sensor truly reading 50."""
    timestamps = pd.date_range("2026-01-01", periods=1440, freq="60s", tz="UTC")
    return pd.DataFrame({"timestamp": timestamps} | {s: 50.0 for s in sensor_ids})


# --- fault_window ---


def test_window_includes_the_start_and_excludes_the_end() -> None:
    days = np.array([0.24, 0.25, 0.74, 0.75])

    assert fault_window(make_fault(), days).tolist() == [False, True, True, False]


# --- Fault types ---


def test_bias_shift_adds_a_constant_error_only_inside_its_window() -> None:
    faulty = apply(clean_readings())

    assert faulty[359] == 50.0
    assert np.all(faulty[360:1080] == 52.0)
    assert faulty[1080] == 50.0


def test_drift_grows_steadily_and_ends_with_the_fault() -> None:
    # Hand check: halfway through the window the error is 2 / 2 = 1.
    faulty = apply(clean_readings(), kind="drift")

    assert faulty[360] == pytest.approx(50.0)
    assert faulty[720] == pytest.approx(51.0)
    assert faulty[1079] == pytest.approx(52.0, abs=0.01)
    assert faulty[1080] == 50.0


def test_negative_drift_moves_readings_down() -> None:
    faulty = apply(clean_readings(), kind="drift", magnitude=-2.0)

    assert faulty[1079] == pytest.approx(48.0, abs=0.01)


def test_stuck_sensor_repeats_one_value() -> None:
    readings = noisy_readings()

    faulty = apply(readings, kind="stuck")

    assert np.all(faulty[360:1080] == readings[360])
    assert faulty[359] == readings[359]
    assert faulty[1080] == readings[1080]


def test_stuck_uses_the_last_reading_if_the_first_is_missing() -> None:
    readings = noisy_readings()
    readings[360] = np.nan

    faulty = apply(readings, kind="stuck")

    assert faulty[361] == readings[359]


def test_stuck_at_the_very_start_uses_the_first_available_reading() -> None:
    readings = noisy_readings()
    readings[0] = np.nan

    faulty = apply(readings, kind="stuck", start_day=0.0)

    assert faulty[1] == readings[1]


def test_stuck_sensor_keeps_its_missing_readings() -> None:
    readings = noisy_readings()
    readings[500] = np.nan

    assert np.isnan(apply(readings, kind="stuck")[500])


def test_dropout_removes_every_reading_in_its_window() -> None:
    faulty = apply(clean_readings(), kind="dropout")

    assert np.all(np.isnan(faulty[360:1080]))
    assert not np.isnan(faulty[:360]).any()
    assert not np.isnan(faulty[1080:]).any()


def test_spikes_have_the_right_size_and_both_directions() -> None:
    faulty = apply(clean_readings(), kind="spike", spike_probability=0.1)

    changes = set(np.unique(faulty[360:1080] - 50.0))
    assert changes == {-2.0, 0.0, 2.0}


def test_spike_rate_matches_its_setting() -> None:
    faulty = apply(clean_readings(), kind="spike", spike_probability=0.1)

    spike_count = np.count_nonzero(faulty[360:1080] != 50.0)
    assert spike_count / 720 == pytest.approx(0.1, rel=0.2)


def test_spikes_stay_inside_the_window() -> None:
    faulty = apply(clean_readings(), kind="spike", spike_probability=1.0)

    assert np.all(faulty[:360] == 50.0)
    assert np.all(faulty[1080:] == 50.0)


def test_faults_keep_missing_readings_missing() -> None:
    readings = clean_readings()
    readings[500] = np.nan

    assert np.isnan(apply(readings)[500])


def test_applying_a_fault_leaves_the_original_readings_alone() -> None:
    readings = clean_readings()

    apply(readings)

    assert np.all(readings == 50.0)


# --- Fault validation ---


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"sensor_id": ""}, "sensor_id"),
        ({"kind": "melted"}, "kind"),
        ({"start_day": -1.0}, "start_day"),
        ({"end_day": 0.1}, "end_day"),
        ({"magnitude": 0.0}, "magnitude"),
        ({"kind": "drift", "magnitude": 0.0}, "magnitude"),
        ({"kind": "spike", "spike_probability": 0.0}, "spike_probability"),
        ({"kind": "spike", "spike_probability": 1.5}, "spike_probability"),
    ],
)
def test_impossible_fault_is_rejected(changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_fault(**changes)


@pytest.mark.parametrize("kind", ["stuck", "dropout"])
def test_stuck_and_dropout_need_no_magnitude(kind: str) -> None:
    assert make_fault(kind=kind, magnitude=0.0).kind == kind


def test_fault_on_unknown_sensor_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown sensor"):
        check_faults((make_fault(sensor_id="PT-999"),), ["PT-001"])


def test_overlapping_faults_on_one_sensor_are_rejected() -> None:
    faults = (make_fault(), make_fault(start_day=0.5, end_day=0.9))

    with pytest.raises(ValueError, match="overlap"):
        check_faults(faults, ["PT-001"])


def test_overlapping_faults_on_different_sensors_are_allowed() -> None:
    faults = (make_fault(), make_fault(sensor_id="PT-002"))

    check_faults(faults, ["PT-001", "PT-002"])


def test_faults_that_only_touch_are_allowed() -> None:
    faults = (make_fault(), make_fault(start_day=0.75, end_day=0.9))

    check_faults(faults, ["PT-001"])


def test_file_with_no_faults_gives_an_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "plan.toml"
    path.write_text('seed = 1\n')

    assert load_faults(path) == ()


# --- generate_sensor_readings ---


def test_perfect_sensor_reports_the_true_value() -> None:
    result = generate_sensor_readings(make_truth("PT-001"), (make_sensor(),), (), SEED)

    assert (result.readings["PT-001"] == 50.0).all()


def test_fault_only_changes_the_named_sensor() -> None:
    sensors = (make_sensor("PT-001"), make_sensor("PT-002"))
    truth = make_truth("PT-001", "PT-002")

    result = generate_sensor_readings(truth, sensors, (make_fault(),), SEED)

    assert result.readings["PT-001"].iloc[500] == 52.0
    assert (result.readings["PT-002"] == 50.0).all()


def test_labels_mark_exactly_the_fault_window() -> None:
    result = generate_sensor_readings(
        make_truth("PT-001"), (make_sensor(),), (make_fault(),), SEED
    )

    labels = result.fault_labels["PT-001"]
    assert (labels.iloc[360:1080] == "bias_shift").all()
    assert (labels.iloc[:360] == "none").all()
    assert (labels.iloc[1080:] == "none").all()


def test_true_data_is_not_changed() -> None:
    truth = make_truth("PT-001")
    before = truth.copy()

    generate_sensor_readings(truth, (make_sensor(),), (make_fault(),), SEED)

    pd.testing.assert_frame_equal(truth, before)


def test_readings_include_bias_and_noise() -> None:
    sensor = make_sensor(noise_std=0.5, bias_std=2.0)

    result = generate_sensor_readings(make_truth("PT-001"), (sensor,), (), SEED)

    values = result.readings["PT-001"]
    assert values.mean() == pytest.approx(50.0 + result.biases["PT-001"], abs=0.1)
    assert values.std() == pytest.approx(0.5, rel=0.1)


def test_missing_readings_are_nan() -> None:
    sensor = make_sensor(missing_probability=0.1)

    result = generate_sensor_readings(make_truth("PT-001"), (sensor,), (), SEED)

    assert result.readings["PT-001"].isna().mean() == pytest.approx(0.1, rel=0.2)


def test_same_seed_gives_the_same_readings() -> None:
    sensor = make_sensor(noise_std=0.5, bias_std=2.0, missing_probability=0.1)

    first = generate_sensor_readings(make_truth("PT-001"), (sensor,), (), SEED)
    second = generate_sensor_readings(make_truth("PT-001"), (sensor,), (), SEED)

    pd.testing.assert_frame_equal(first.readings, second.readings)


def test_different_seed_gives_different_readings() -> None:
    sensor = make_sensor(noise_std=0.5)

    first = generate_sensor_readings(make_truth("PT-001"), (sensor,), (), SEED)
    second = generate_sensor_readings(make_truth("PT-001"), (sensor,), (), SEED + 1)

    assert not first.readings["PT-001"].equals(second.readings["PT-001"])


def test_sensor_with_a_different_sampling_interval_is_rejected() -> None:
    sensor = make_sensor(sampling_interval=300.0)

    with pytest.raises(ValueError, match="samples every"):
        generate_sensor_readings(make_truth("PT-001"), (sensor,), (), SEED)


# --- The real 30-day plan ---


@pytest.fixture(scope="module")
def full_run() -> tuple[pd.DataFrame, object]:
    """The real plan's true data and sensor readings."""
    case = load_case(CONFIG_FOLDER / "base_case.toml")
    sensors = load_sensors(CONFIG_FOLDER / "sensors.toml")
    plan = load_plan(CONFIG_FOLDER / "simulation.toml")
    faults = load_faults(CONFIG_FOLDER / "simulation.toml")
    truth = generate_operating_data(case, sensors, plan)
    return truth, generate_sensor_readings(truth, sensors, faults, plan.seed)


def test_network_produces_345600_expected_readings(full_run) -> None:
    # Hand check: 43,200 steps * 8 sensors = 345,600
    _, sensor_data = full_run

    assert sensor_data.readings.drop(columns="timestamp").size == 345_600


def test_missing_readings_are_mostly_the_planned_rates(full_run) -> None:
    # Hand check: 345,600 * 0.002 = 691 random gaps, plus a 720-minute dropout.
    _, sensor_data = full_run

    missing = sensor_data.readings.drop(columns="timestamp").isna().sum().sum()
    assert missing == pytest.approx(691 + 720, abs=60)


def test_every_planned_fault_is_labelled(full_run) -> None:
    _, sensor_data = full_run

    labels = sensor_data.fault_labels
    assert (labels["PT-002"] == "stuck").sum() == 360
    assert (labels["FT-002"] == "drift").sum() == 5760
    assert (labels["TT-002"] == "spike").sum() == 1440
    assert (labels["PT-003"] == "bias_shift").sum() == 2880
    assert (labels["FT-001"] == "dropout").sum() == 720


def test_healthy_readings_stay_close_to_the_truth(full_run) -> None:
    truth, sensor_data = full_run

    healthy = sensor_data.fault_labels["PT-001"] == "none"
    error = (sensor_data.readings["PT-001"] - truth["PT-001"])[healthy]

    assert error.abs().max() < 0.3  # barg: bias plus about 5 noise deviations


def test_flow_drift_looks_like_an_imbalance_while_true_flow_matches(full_run) -> None:
    # Day 0 is 1 January, so the drift runs from 11 January to the end of 14
    # January. The last half day is days 13.5 to 14, where the average error is
    # 12 * 3.75 / 4 = 11.25 m^3/h. The day before the drift is the baseline.
    # The pipeline has no leak, yet the two flow sensors drift apart.
    truth, sensor_data = full_run
    readings = sensor_data.readings.set_index("timestamp")
    imbalance = readings["FT-002"] - readings["FT-001"]

    before = imbalance.loc["2026-01-10 00:00":"2026-01-10 11:59"].mean()
    late = imbalance.loc["2026-01-14 12:00":"2026-01-14 23:59"].mean()

    assert late - before == pytest.approx(11.25, abs=0.5)
    assert truth["FT-001"].equals(truth["FT-002"])
