# Oil Pipeline Digital Twin

An educational Python digital twin of an oil transmission pipeline, built as a
learning and portfolio project. It is not a safety-certified monitoring system,
and its results must not be used for real pipeline operations.

## Current status

The steady-state hydraulic model is complete. Sensor simulation, leak detection,
corrosion modelling, machine learning and the dashboard are planned.

The model takes a fluid, a flow rate, an inlet pressure, pipe segments and pump
stations, then calculates velocity, Reynolds number, friction factor and
pressure through each segment.

## Setup

These commands are for Windows PowerShell, from the project folder. They require
Python 3.11 or newer.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
```

## Usage

Run the base case. This prints a report and saves a pressure profile chart to
`outputs/pressure_profile.png`:

```powershell
.venv\Scripts\python scripts\run_model.py
```

Run the tests:

```powershell
.venv\Scripts\python -m pytest
```

To try a different scenario, edit [config/base_case.toml](config/base_case.toml).
Its values are illustrative and do not describe a real pipeline.

## Project structure

| Path | Purpose |
|---|---|
| `config/base_case.toml` | Base-case inputs |
| `scripts/run_model.py` | Runs the model, prints a report and saves a chart |
| `src/pipeline/fluid_properties.py` | `Fluid` class |
| `src/pipeline/parameters.py` | `PipeSegment` and `PumpStation` classes |
| `src/pipeline/hydraulics.py` | Single-segment equations |
| `src/pipeline/pipeline_model.py` | Multi-segment pressure calculation |
| `src/pipeline/config_loader.py` | Reads TOML cases |
| `src/pipeline/report.py` | Formats text reports |
| `src/pipeline/plotting.py` | Builds and draws the pressure profile |
| `src/pipeline/validation.py` | Shared input checks |
| `tests/` | Automated tests |

## Engineering basis

All calculations use SI units, and pressures are absolute inside the model.

| Quantity | Equation |
|---|---|
| Area | A = pi D^2 / 4 |
| Velocity | v = Q / A |
| Reynolds number | Re = rho v D / mu |
| Friction factor, Re <= 2300 | f = 64 / Re |
| Friction factor, Re >= 4000 | Swamee-Jain approximation of Colebrook |
| Friction factor, in between | Linear interpolation (a modelling assumption) |
| Friction loss | dp = f (L / D) (rho v^2 / 2) |
| Elevation term | dp = rho g dz |
| Segment outlet | p_out = p_in + pump boost - friction loss - rho g dz |

Calculations are checked against hand calculations, an independent Colebrook
solver and the Hagen-Poiseuille equation. Swamee-Jain is within 1% of Colebrook
for Re from 1e4 to 1e7 with relative roughness up to 1e-3, and about 3% at worst
across the range the model uses.

## Assumptions and limitations

- Flow is steady, single-phase and incompressible, with no leaks or offtakes.
- Fluid properties are constant, so temperature effects are not modelled.
- Flow rate and inlet pressure are inputs, not solved values.
- Pumps add a fixed pressure boost, and pump curves are not modelled.
- Losses from fittings, bends and valves are excluded.
- Elevation varies linearly along a segment, so a high point needs a segment
  boundary to be represented.
- Pressure below vapour pressure is flagged, but results downstream of that
  point are not physically meaningful.
- Maximum allowable operating pressure is not checked.
