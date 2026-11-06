"""Build and draw the pressure profile along a pipeline.

pressure_profile only prepares the numbers, so it can be tested and reused.
plot_pressure_profile only draws them, and contains no engineering.
"""

from dataclasses import dataclass

from matplotlib.figure import Figure

from pipeline.config_loader import PipelineCase
from pipeline.pipeline_model import PipelineResult
from pipeline.report import PA_PER_BAR, to_bar_gauge

M_PER_KM = 1000.0

SURFACE_COLOR = "#fcfcfb"
TEXT_COLOR = "#0b0b0b"
MUTED_TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"
PRESSURE_COLOR = "#2a78d6"
ELEVATION_COLOR = "#a3a29c"


@dataclass(frozen=True)
class ProfilePoint:
    """One point on the pressure profile.

    Attributes:
        distance: Distance from the start of the pipeline [m].
        elevation: Height above datum [m].
        pressure: Absolute pressure [Pa].
        pump_name: Name of the pump that raised the pressure at this point, or
            None if this point is not a pump discharge.
    """

    distance: float
    elevation: float
    pressure: float
    pump_name: str | None = None


def pressure_profile(case: PipelineCase, result: PipelineResult) -> list[ProfilePoint]:
    """Return the points that describe pressure along the pipeline.

    Pressure changes linearly along a segment, so straight lines between these
    points are exact. A running pump adds a second point at the same distance,
    which draws as a vertical jump. A stopped pump (zero boost) adds no point.

    Args:
        case: The case that was run.
        result: The calculated results.

    Returns:
        The profile points, ordered from upstream to downstream.
    """
    pump_names = {pump.segment_name: pump.name for pump in case.pumps}
    points = [
        ProfilePoint(0.0, case.segments[0].inlet_elevation, result.inlet_pressure)
    ]
    distance = 0.0

    for segment, conditions in zip(case.segments, result.segments):
        if conditions.pump_boost > 0:
            points.append(
                ProfilePoint(
                    distance,
                    segment.inlet_elevation,
                    conditions.pressure_after_pump,
                    pump_names[segment.name],
                )
            )
        distance += segment.length
        points.append(
            ProfilePoint(distance, segment.outlet_elevation, conditions.outlet_pressure)
        )

    return points


def plot_pressure_profile(case: PipelineCase, result: PipelineResult) -> Figure:
    """Draw pressure and ground elevation against distance.

    They are drawn as two panels, not on one chart with two axes, because
    pressure and elevation have different units.

    Args:
        case: The case that was run.
        result: The calculated results.

    Returns:
        The finished figure. Save it with figure.savefig(path).
    """
    points = pressure_profile(case, result)
    distances_km = [point.distance / M_PER_KM for point in points]
    pressures_barg = [to_bar_gauge(point.pressure) for point in points]
    elevations_m = [point.elevation for point in points]

    figure = Figure(figsize=(10, 6.5), layout="constrained", facecolor=SURFACE_COLOR)
    pressure_axes, elevation_axes = figure.subplots(
        2, 1, sharex=True, height_ratios=[3, 1]
    )

    # clip_on=False stops the panel edge cutting a pump jump at km 0 in half.
    pressure_axes.plot(
        distances_km, pressures_barg,
        color=PRESSURE_COLOR, linewidth=2, clip_on=False,
    )
    _draw_vapor_pressure_line(pressure_axes, case, distances_km[-1])
    _label_pumps(pressure_axes, points)
    _label_segments(pressure_axes, case)

    elevation_axes.fill_between(
        distances_km, elevations_m, min(elevations_m) - 20,
        color=ELEVATION_COLOR, linewidth=0,
    )

    pressure_axes.set_title(
        f"{case.name}: {result.flow_rate:.3f} m3/s",
        loc="left", color=TEXT_COLOR, fontsize=13, pad=26,
    )
    pressure_axes.set_ylabel("Pressure (barg)", color=MUTED_TEXT_COLOR)
    elevation_axes.set_ylabel("Elevation (m)", color=MUTED_TEXT_COLOR)
    elevation_axes.set_xlabel("Distance along pipeline (km)", color=MUTED_TEXT_COLOR)
    elevation_axes.set_ylim(min(elevations_m) - 20, max(elevations_m) + 20)
    elevation_axes.set_xlim(0, distances_km[-1])
    elevation_axes.set_xticks(sorted(set(distances_km)))

    for axes in (pressure_axes, elevation_axes):
        _apply_style(axes)

    return figure


def _draw_vapor_pressure_line(axes, case: PipelineCase, end_km: float) -> None:
    """Draw and label the fluid's vapour pressure as a dashed reference line."""
    vapor_barg = to_bar_gauge(case.fluid.vapor_pressure)
    axes.axhline(vapor_barg, color=MUTED_TEXT_COLOR, linewidth=1, linestyle="--")
    axes.text(
        end_km, vapor_barg, "Vapour pressure ",
        color=MUTED_TEXT_COLOR, fontsize=9, ha="right", va="bottom",
    )


def _label_pumps(axes, points: list[ProfilePoint]) -> None:
    """Label each pump at the top of its pressure jump."""
    for previous, point in zip(points, points[1:]):
        if point.pump_name is not None:
            boost_bar = (point.pressure - previous.pressure) / PA_PER_BAR
            axes.annotate(
                f"{point.pump_name}: +{boost_bar:.0f} bar",
                (point.distance / M_PER_KM, to_bar_gauge(point.pressure)),
                xytext=(6, 4), textcoords="offset points",
                color=TEXT_COLOR, fontsize=9,
            )


def _label_segments(axes, case: PipelineCase) -> None:
    """Write each segment's name above the panel, centred on the segment."""
    start = 0.0
    for segment in case.segments:
        middle_km = (start + segment.length / 2) / M_PER_KM
        axes.text(
            middle_km, 1.01, segment.name,
            transform=axes.get_xaxis_transform(),
            color=MUTED_TEXT_COLOR, fontsize=9, ha="center", va="bottom",
        )
        start += segment.length


def _apply_style(axes) -> None:
    """Keep the grid and axes quiet so that the data stands out."""
    axes.set_facecolor(SURFACE_COLOR)
    axes.grid(color=GRID_COLOR, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED_TEXT_COLOR, length=0)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID_COLOR)
