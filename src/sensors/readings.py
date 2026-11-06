"""Turn true operating data into what the sensor network would report.

All readings are SIMULATED.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sensors.faults import SensorFault, apply_fault, check_faults, fault_window
from sensors.sensor import Sensor, draw_biases, take_reading

SECONDS_PER_DAY = 86_400.0
NO_FAULT = "none"

# Keeps sensor randomness separate from the operating data's, so that adding a
# sensor does not change the simulated pipeline.
SENSOR_RANDOM_STREAM = 1


@dataclass(frozen=True)
class SensorData:
    """The output of the sensor network.

    Keep labels and biases away from anything that learns from the readings.
    They are the answers, and a real monitoring system would not have them.

    Attributes:
        readings: Timestamp plus one column per sensor. NaN means no reading.
        fault_labels: Timestamp plus each sensor's active fault at that time.
        biases: Each sensor's fixed calibration error.
    """

    readings: pd.DataFrame
    fault_labels: pd.DataFrame
    biases: dict[str, float]


def _check_sampling(sensors: tuple[Sensor, ...], data: pd.DataFrame) -> None:
    """Raise ValueError unless every sensor samples at the data's time step."""
    time_step = (data["timestamp"].iloc[1] - data["timestamp"].iloc[0]).total_seconds()
    for sensor in sensors:
        if sensor.sampling_interval != time_step:
            raise ValueError(
                f"{sensor.sensor_id} samples every {sensor.sampling_interval} s, "
                f"but the data has a {time_step} s time step"
            )


def generate_sensor_readings(
    data: pd.DataFrame,
    sensors: tuple[Sensor, ...],
    faults: tuple[SensorFault, ...],
    seed: int,
) -> SensorData:
    """Simulate every sensor reading.

    Args:
        data: True operating data, with a timestamp and a column per sensor.
        sensors: The sensor network.
        faults: Scripted sensor faults.
        seed: Random seed. The same seed gives the same readings.

    Returns:
        The readings, fault labels and biases.

    Raises:
        ValueError: If a sensor's sampling interval does not match the data, or
            a fault is invalid.
    """
    _check_sampling(sensors, data)
    check_faults(faults, [sensor.sensor_id for sensor in sensors])

    rng = np.random.default_rng([seed, SENSOR_RANDOM_STREAM])
    biases = draw_biases(sensors, rng)
    elapsed = data["timestamp"] - data["timestamp"].iloc[0]
    days = elapsed.dt.total_seconds().to_numpy() / SECONDS_PER_DAY

    readings = {"timestamp": data["timestamp"]}
    labels = {"timestamp": data["timestamp"]}

    for sensor in sensors:
        bias = biases[sensor.sensor_id]
        # dtype=float converts None (no reading) to NaN.
        values = np.array(
            [
                take_reading(sensor, value, bias, rng)
                for value in data[sensor.sensor_id].to_numpy()
            ],
            dtype=float,
        )
        sensor_labels = np.full(len(data), NO_FAULT, dtype=object)

        for fault in faults:
            if fault.sensor_id == sensor.sensor_id:
                values = apply_fault(values, fault, days, rng)
                sensor_labels[fault_window(fault, days)] = fault.kind

        readings[sensor.sensor_id] = values
        labels[sensor.sensor_id] = sensor_labels

    return SensorData(pd.DataFrame(readings), pd.DataFrame(labels), biases)
