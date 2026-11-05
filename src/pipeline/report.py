"""Turn model results into a text report.

The model works in pascals (absolute). This module only converts values for
display, and contains no engineering calculations of its own.
"""

from pipeline.config_loader import PipelineCase
from pipeline.hydraulics import flow_regime
from pipeline.pipeline_model import PipelineResult, SegmentResult

PA_PER_BAR = 1e5
ATMOSPHERIC_PRESSURE = 101_325.0  # Pa, standard atmosphere


def to_bar_gauge(absolute_pressure: float) -> float:
    """Convert absolute pressure [Pa] to gauge pressure [bar]."""
    return (absolute_pressure - ATMOSPHERIC_PRESSURE) / PA_PER_BAR


def _format_friction_factor(darcy_f: float | None) -> str:
    """Return the friction factor as text, or n/a when it is undefined."""
    return "n/a" if darcy_f is None else f"{darcy_f:.4f}"


def _format_segment_row(segment: SegmentResult) -> str:
    """Return one line of the results table."""
    return (
        f"{segment.segment_name:<8}"
        f"{segment.pump_boost / PA_PER_BAR:>7.1f}"
        f"{to_bar_gauge(segment.inlet_pressure):>10.2f}"
        f"{to_bar_gauge(segment.pressure_after_pump):>10.2f}"
        f"{to_bar_gauge(segment.outlet_pressure):>10.2f}"
        f"{segment.friction_loss / PA_PER_BAR:>10.2f}"
        f"{segment.elevation_pressure / PA_PER_BAR:>10.2f}"
        f"{segment.velocity:>8.2f}"
        f"{segment.reynolds:>10.0f}"
        f"{_format_friction_factor(segment.darcy_friction_factor):>8}"
    )


def _warnings(result: PipelineResult) -> list[str]:
    """Return a warning for every condition the user should look at."""
    warnings = []
    for segment in result.segments:
        if segment.below_vapor_pressure:
            warnings.append(
                f"{segment.segment_name}: pressure is below vapour pressure. "
                "Results from here downstream are not physically meaningful."
            )
        if flow_regime(segment.reynolds) == "transitional":
            warnings.append(
                f"{segment.segment_name}: flow is transitional, so its friction "
                "factor is an interpolated estimate."
            )
    return warnings


def format_results(case: PipelineCase, result: PipelineResult) -> str:
    """Return a text report of a pipeline calculation.

    Args:
        case: The case that was run.
        result: The calculated results.

    Returns:
        The report, ready to print.
    """
    total_length_km = sum(segment.length for segment in case.segments) / 1000
    regimes = sorted({flow_regime(s.reynolds) for s in result.segments})

    lines = [
        case.name,
        "=" * len(case.name),
        f"Length:     {total_length_km:.1f} km in {len(case.segments)} segments",
        f"Flow rate:  {result.flow_rate:.3f} m3/s",
        f"Flow type:  {', '.join(regimes)}",
        "",
        f"{'Segment':<8}{'Pump':>7}{'Arrive':>10}{'After':>10}{'Outlet':>10}"
        f"{'Friction':>10}{'Height':>10}{'Speed':>8}{'Re':>10}{'f':>8}",
        f"{'':<8}{'bar':>7}{'barg':>10}{'barg':>10}{'barg':>10}"
        f"{'bar':>10}{'bar':>10}{'m/s':>8}{'':>10}{'':>8}",
        "-" * 91,
    ]
    lines += [_format_segment_row(segment) for segment in result.segments]
    lines += [
        "",
        f"{'Pressure':<12}{'bar abs':>10}{'barg':>10}",
        f"{'Inlet':<12}{result.inlet_pressure / PA_PER_BAR:>10.2f}"
        f"{to_bar_gauge(result.inlet_pressure):>10.2f}",
        f"{'Outlet':<12}{result.outlet_pressure / PA_PER_BAR:>10.2f}"
        f"{to_bar_gauge(result.outlet_pressure):>10.2f}",
        f"{'Minimum':<12}{result.minimum_pressure / PA_PER_BAR:>10.2f}"
        f"{to_bar_gauge(result.minimum_pressure):>10.2f}",
        "",
        "Warnings:",
    ]
    lines += [f"  - {warning}" for warning in _warnings(result)] or ["  None"]
    lines += [
        "",
        "Educational model with illustrative inputs. Not for operational use.",
    ]
    return "\n".join(lines)
