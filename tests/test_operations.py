"""Tests for the synthetic operating data generator."""

import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.config_loader import load_case
from pipeline.pipeline_model import calculate_pipeline
from sensors.sensor import load_sensors
from simulation.operations import (
    Event,
    SimulationPlan,
    correlated_noise,
    daily_cycle,
    departure_steps,
    event_strength,
    generate_operating_data,
    load_plan,
    transit_flow_rates,
)

CONFIG_FOLDER = Path(__file__).resolve().parent.parent / "config"
CASE = load_case(CONFIG_FOLDER / "base_case.toml")
SENSORS = load_sensors(CONFIG_FOLDER / "sensors.toml")
START = datetime(2026, 1, 1, tzinfo=UTC)
SEED = 12345


def make_plan(**changes: object) -> SimulationPlan:
    """Build a one-day plan with no variation, replacing any values a test changes."""
    values = {
        "seed": SEED,
        "start_time": START,
        "duration_days": 1.0,
        "time_step": 600.0,
        "noise_correlation_hours": 2.0,
        "flow_daily_amplitude": 0.0,
        "flow_peak_hour": 18.0,
        "flow_noise_std": 0.0,
        "pump_noise_std": 0.0,
        "inlet_pressure_noise_std": 0.0,
        "inlet_temperature_mean": 313.15,
        "inlet_temperature_amplitude": 0.0,
        "inlet_temperature_peak_hour": 15.0,
    }
    values.update(changes)
    return SimulationPlan(**values)


def make_event(**changes: object) -> Event:
    """Build a two-day event with no ramps, replacing any values a test changes."""
    values = {
        "label": "test_event",
        "start_day": 1.0,
        "end_day": 3.0,
        "ramp_up_hours": 0.0,
        "ramp_down_hours": 0.0,
    }
    values.update(changes)
    return Event(**values)


def generate(**plan_changes: object) -> pd.DataFrame:
    """Generate base-case data from a plan."""
    return generate_operating_data(CASE, SENSORS, make_plan(**plan_changes))


@pytest.fixture(scope="module")
def full_run() -> pd.DataFrame:
    """The real 30-day plan. It takes seconds, so it is only generated once."""
    return generate_operating_data(CASE, SENSORS, load_plan(CONFIG_FOLDER / "simulation.toml"))


# --- correlated_noise ---


def test_noise_has_the_requested_spread_and_a_zero_mean() -> None:
    noise = correlated_noise(np.random.default_rng(SEED), 200_000, 0.5, 10.0)

    assert np.mean(noise) == pytest.approx(0.0, abs=0.02)
    assert np.std(noise) == pytest.approx(0.5, rel=0.02)


def test_neighbouring_noise_values_are_related() -> None:
    # Hand check: correlation at one step is exp(-1 / 10) = 0.905
    noise = correlated_noise(np.random.default_rng(SEED), 200_000, 0.5, 10.0)

    assert np.corrcoef(noise[:-1], noise[1:])[0, 1] == pytest.approx(0.905, abs=0.01)


def test_zero_spread_gives_zero_noise() -> None:
    noise = correlated_noise(np.random.default_rng(SEED), 100, 0.0, 10.0)

    assert np.all(noise == 0.0)


# --- daily_cycle ---


def test_cycle_is_highest_at_the_peak_hour_and_lowest_12_hours_later() -> None:
    values = daily_cycle(np.array([18.0, 6.0, 42.0]), 18.0)

    assert values == pytest.approx([1.0, -1.0, 1.0])


# --- event_strength ---


def test_event_with_no_ramps_switches_on_and_off() -> None:
    hours = np.array([23.0, 24.0, 71.0, 72.0])

    assert event_strength(make_event(), hours) == pytest.approx([0, 1, 1, 0])


def test_event_ramps_up_and_down_in_straight_lines() -> None:
    event = make_event(ramp_up_hours=10.0, ramp_down_hours=4.0)
    hours = np.array([24.0, 29.0, 34.0, 72.0, 74.0, 76.0])

    assert event_strength(event, hours) == pytest.approx([0, 0.5, 1, 1, 0.5, 0])


def test_event_finish_day_includes_the_ramp_down() -> None:
    assert make_event(ramp_down_hours=12.0).finish_day == pytest.approx(3.5)


# --- departure_steps ---


def test_travel_time_matches_hand_calculation_at_constant_flow() -> None:
    # Hand check: 19,635 m^3 / 0.25 m^3/s = 78,540 s = 1309 one-minute steps
    flows = np.full(5000, 0.25)

    entered = departure_steps(flows, 60.0, 19_635.0)

    assert 4999 - entered[4999] == pytest.approx(1309, abs=1)


def test_oil_at_the_inlet_entered_just_now() -> None:
    steps = departure_steps(np.full(100, 0.25), 60.0, 0.0)

    assert np.array_equal(steps, np.arange(100))


def test_oil_already_in_the_line_counts_as_step_zero() -> None:
    entered = departure_steps(np.full(100, 0.25), 60.0, 19_635.0)

    assert np.all(entered == 0)


def test_slower_flow_means_longer_travel_time() -> None:
    fast = departure_steps(np.full(5000, 0.25), 60.0, 5000.0)
    slow = departure_steps(np.full(5000, 0.125), 60.0, 5000.0)

    assert (4999 - slow[4999]) == pytest.approx(2 * (4999 - fast[4999]), abs=2)


# --- transit_flow_rates ---


def test_transit_flow_equals_a_constant_flow() -> None:
    flows = np.full(2000, 0.25)

    transit = transit_flow_rates(flows, 60.0, 3000.0)

    assert transit == pytest.approx(flows)


def test_transit_flow_averages_a_journey_across_a_flow_change() -> None:
    # Flow doubles at step 1000. By step 1050, 50 steps at 0.5 m^3/s have pumped
    # 1500 m^3. The other 1500 m^3 took 100 steps at 0.25 m^3/s.
    # Hand check: 3000 m^3 / (150 steps * 60 s) = 0.3333 m^3/s
    flows = np.concatenate([np.full(1000, 0.25), np.full(1000, 0.5)])

    transit = transit_flow_rates(flows, 60.0, 3000.0)

    assert transit[1049] == pytest.approx(0.3333, abs=0.002)


def test_transit_flow_uses_current_flow_before_the_line_has_filled() -> None:
    flows = np.linspace(0.2, 0.3, 100)

    transit = transit_flow_rates(flows, 60.0, 19_635.0)

    assert transit == pytest.approx(flows)


def test_transit_flow_at_the_inlet_is_the_current_flow() -> None:
    flows = np.linspace(0.2, 0.3, 100)

    assert transit_flow_rates(flows, 60.0, 0.0) == pytest.approx(flows)


# --- Plan and event validation ---


def test_steps_for_30_days_at_one_minute() -> None:
    # Hand check: 30 * 24 * 60 = 43,200
    assert make_plan(duration_days=30.0, time_step=60.0).steps == 43_200


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("duration_days", 0.0),
        ("time_step", -60.0),
        ("noise_correlation_hours", 0.0),
        ("flow_noise_std", -0.1),
        ("inlet_temperature_mean", 0.0),
        ("start_time", datetime(2026, 1, 1)),
    ],
)
def test_impossible_plan_is_rejected(field: str, bad_value: object) -> None:
    match = "time zone" if field == "start_time" else field

    with pytest.raises(ValueError, match=match):
        make_plan(**{field: bad_value})


def test_overlapping_events_are_rejected() -> None:
    events = (
        make_event(label="first", ramp_down_hours=24.0),
        make_event(label="second", start_day=3.5, end_day=5.0),
    )

    with pytest.raises(ValueError, match="overlap"):
        make_plan(events=events)


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("label", ""),
        ("start_day", -1.0),
        ("flow_fraction", 0.0),
        ("boost_fraction", -0.5),
        ("ramp_up_hours", -1.0),
        ("ramp_down_hours", -1.0),
    ],
)
def test_impossible_event_is_rejected(field: str, bad_value: object) -> None:
    with pytest.raises(ValueError, match=field):
        make_event(**{field: bad_value})


def test_event_ending_before_it_starts_is_rejected() -> None:
    with pytest.raises(ValueError, match="end_day"):
        make_event(start_day=5.0, end_day=4.0)


def test_ramp_longer_than_event_is_rejected() -> None:
    with pytest.raises(ValueError, match="ramp_up_hours"):
        make_event(ramp_up_hours=49.0)


def test_event_on_unknown_pump_is_rejected() -> None:
    events = (make_event(start_day=0.0, end_day=0.5, pump="PS9", boost_fraction=0.5),)

    with pytest.raises(ValueError, match="unknown pump"):
        generate(events=events)


def test_case_without_thermal_properties_is_rejected() -> None:
    case = dataclasses.replace(CASE, thermal=None)

    with pytest.raises(ValueError, match="thermal"):
        generate_operating_data(case, SENSORS, make_plan())


def test_missing_plan_setting_is_named_in_the_error(tmp_path: Path) -> None:
    text = (CONFIG_FOLDER / "simulation.toml").read_text()
    path = tmp_path / "plan.toml"
    path.write_text(text.replace("seed = 42\n", ""))

    with pytest.raises(ValueError, match="seed"):
        load_plan(path)


# --- Generated data ---


def test_steady_plan_matches_a_direct_model_run() -> None:
    # With no variation, the simulation must agree with the pressure model when
    # both are given the same temperature-dependent viscosities. At 0.25 m^3/s
    # with 40 °C oil, the temperature model's result for this line is 22.34 bar
    # abs, as reported when the temperature stage was built.
    data = generate()

    assert data["PT-003"].iloc[-1] == pytest.approx(22.34 - 1.01325, abs=0.01)


def test_steady_plan_stays_constant_once_the_line_has_filled() -> None:
    data = generate(duration_days=2.0)

    assert data["PT-003"].std() == pytest.approx(0.0, abs=1e-9)


def test_flow_sensors_agree_with_the_flow_input() -> None:
    data = generate(flow_noise_std=0.01)

    expected = data["flow_rate_m3_s"] * 3600
    assert data["FT-001"].to_numpy() == pytest.approx(expected.to_numpy())
    assert data["FT-002"].to_numpy() == pytest.approx(expected.to_numpy())


def test_timestamps_are_evenly_spaced_in_utc() -> None:
    data = generate()

    assert data["timestamp"].iloc[0] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert (data["timestamp"].diff().dropna() == pd.Timedelta(seconds=600)).all()


def test_same_seed_gives_the_same_data() -> None:
    plan = {"flow_noise_std": 0.01, "pump_noise_std": 0.005}

    pd.testing.assert_frame_equal(generate(**plan), generate(**plan))


def test_different_seed_gives_different_data() -> None:
    first = generate(flow_noise_std=0.01)
    second = generate(flow_noise_std=0.01, seed=SEED + 1)

    assert not first["FT-001"].equals(second["FT-001"])


def test_daily_flow_cycle_peaks_at_the_chosen_hour() -> None:
    data = generate(flow_daily_amplitude=0.04, time_step=3600.0)

    assert data["flow_rate_m3_s"].idxmax() == 18
    assert data["flow_rate_m3_s"].max() == pytest.approx(0.26)  # 0.25 * 1.04


def test_outlet_temperature_lags_the_inlet_by_the_travel_time() -> None:
    # Hand check: the oil takes 19,635 m^3 / 0.25 m^3/s = 21.8 hours to arrive.
    data = generate(
        duration_days=5.0, time_step=300.0, inlet_temperature_amplitude=1.5
    )
    last_day = data.iloc[-288:]

    inlet_peak_hour = last_day["TT-001"].idxmax() * 300 / 3600
    outlet_peak_hour = last_day["TT-003"].idxmax() * 300 / 3600

    assert (outlet_peak_hour - inlet_peak_hour) % 24 == pytest.approx(21.8, abs=0.2)


def test_outlet_temperature_changes_gradually_when_flow_drops() -> None:
    # Oil arriving just after a flow drop spent nearly all of its journey at the
    # old speed, so its temperature must not jump.
    event = make_event(start_day=3.0, end_day=3.5, flow_fraction=0.75)

    data = generate(duration_days=4.0, time_step=300.0, events=(event,))

    assert data["TT-003"].diff().abs().max() < 0.05  # degC per 5 minutes


# --- Scripted events ---


def test_steps_outside_events_are_labelled_normal() -> None:
    event = make_event(start_day=0.5, end_day=0.75, flow_fraction=1.1)

    data = generate(events=(event,))

    assert set(data["event"]) == {"normal", "test_event"}
    assert (data["event"].iloc[:72] == "normal").all()
    assert (data["event"].iloc[72:108] == "test_event").all()
    assert (data["event"].iloc[108:] == "normal").all()


def test_demand_event_raises_flow_and_lowers_outlet_pressure() -> None:
    event = make_event(start_day=0.5, end_day=0.75, flow_fraction=1.15)

    data = generate(events=(event,))

    during = data[data["event"] == "test_event"]
    assert during["flow_rate_m3_s"].iloc[0] == pytest.approx(0.2875)  # 0.25 * 1.15
    assert during["PT-003"].mean() < data["PT-003"].iloc[0]


def test_stopped_pump_drops_its_boost_to_zero() -> None:
    event = make_event(start_day=0.5, end_day=0.75, pump="PS2", boost_fraction=0.0)

    data = generate(events=(event,))

    during = data[data["event"] == "test_event"]
    assert (during["boost_PS2_pa"] == 0.0).all()
    assert (data["boost_PS2_pa"].iloc[0]) == pytest.approx(1_500_000.0)


def test_stopped_pump_lowers_outlet_pressure_by_its_boost() -> None:
    event = make_event(start_day=0.5, end_day=0.75, pump="PS2", boost_fraction=0.0)

    data = generate(events=(event,))

    drop = data["PT-003"].iloc[0] - data[data["event"] == "test_event"]["PT-003"].iloc[0]
    assert drop == pytest.approx(15.0, abs=0.01)


def test_ps2_stopping_does_not_change_pressure_upstream_of_it() -> None:
    event = make_event(start_day=0.5, end_day=0.75, pump="PS2", boost_fraction=0.0)

    data = generate(events=(event,))

    during = data[data["event"] == "test_event"]
    assert during["PT-002"].iloc[0] == pytest.approx(data["PT-002"].iloc[0])


# --- The real 30-day plan ---


def test_full_run_has_43200_steps_and_every_sensor(full_run: pd.DataFrame) -> None:
    assert len(full_run) == 43_200
    assert {sensor.sensor_id for sensor in SENSORS} <= set(full_run.columns)


def test_full_run_never_reaches_low_pressure(full_run: pd.DataFrame) -> None:
    assert not full_run["low_pressure"].any()


def test_full_run_has_no_missing_values(full_run: pd.DataFrame) -> None:
    assert not full_run.isna().any().any()


def test_full_run_includes_every_operating_state(full_run: pd.DataFrame) -> None:
    assert set(full_run["event"]) == {
        "normal",
        "demand_increase",
        "ps1_decline",
        "ps2_trip",
    }


def test_full_run_stays_below_the_low_pressure_flow_limit(
    full_run: pd.DataFrame,
) -> None:
    assert full_run["flow_rate_m3_s"].max() < 0.34


def test_flow_is_most_of_the_time_near_base_flow(full_run: pd.DataFrame) -> None:
    normal = full_run[full_run["event"] == "normal"]

    assert normal["FT-001"].mean() == pytest.approx(900.0, rel=0.01)


def test_ps1_decline_recovers_after_repair(full_run: pd.DataFrame) -> None:
    decline = full_run[full_run["event"] == "ps1_decline"]
    after = full_run.iloc[decline.index[-1] + 1]

    assert decline["boost_PS1_pa"].iloc[-1] < 3_300_000.0
    assert after["boost_PS1_pa"] > 3_400_000.0
