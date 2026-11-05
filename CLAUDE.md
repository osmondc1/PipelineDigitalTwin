# Oil Pipeline Digital Twin — Project Instructions

## 1. Project Overview

This project aims to develop an interactive, Python-based digital twin of an oil transmission pipeline.

The objective is to simulate pipeline operating conditions, monitor pipeline integrity, identify abnormal behaviour, and demonstrate how engineering modelling, data analytics, and machine learning can support pipeline operations.

The project is also intended to develop the user's skills in:
- Python programming
- Software engineering
- AI-assisted development using Claude Code
- Engineering simulation
- Data analysis and visualization
- Machine learning
- Git and GitHub
- Application development and deployment

This is an educational and portfolio project intended to demonstrate engineering and software development capabilities.

It is not a safety-certified pipeline monitoring system.

## 2. Project Objectives

The digital twin should eventually include the following functionality:

### Core Pipeline Simulation
- Model an approximately 100 km oil transmission pipeline.
- Include multiple pipeline segments.
- Include pumping stations.
- Model pressure, flow rate, temperature, and fluid properties.
- Calculate pressure losses using fluid mechanics principles.
- Account for elevation changes and pumping contributions.

### Sensor Simulation
- Generate simulated pressure, flow, and temperature measurements.
- Include multiple sensor locations.
- Introduce realistic measurement noise.
- Simulate sensor faults and missing readings.
- Store historical sensor data.

### Leak Detection
- Identify abnormal flow imbalances.
- Monitor unexpected pressure deviations.
- Detect potential leak scenarios.
- Estimate potential leak locations using segment-level information.
- Display leak alerts and associated confidence or severity indicators.

### Corrosion and Degradation
- Model simplified pipeline wall thickness degradation.
- Estimate remaining wall thickness.
- Calculate percentage wall loss.
- Track corrosion by pipeline segment.
- Estimate remaining life under explicitly stated assumptions.
- Identify segments requiring further inspection.

### Machine Learning
- Implement anomaly detection.
- Investigate predictive maintenance models.
- Predict degradation using simulated historical data.
- Evaluate model performance using appropriate metrics.
- Compare machine learning predictions with engineering-based calculations.

### What-If Simulations
Allow users to modify operating conditions and evaluate their effects.

Examples:
- Increased throughput.
- Pump failure.
- Increased oil viscosity.
- Pipeline degradation.
- Simulated pipeline leak.

### Interactive Dashboard
Develop a Streamlit application that displays:
- Pipeline operating conditions.
- Sensor readings.
- Pipeline visualization.
- Historical trends.
- Leak detection alerts.
- Corrosion indicators.
- Machine learning predictions.
- What-if simulation results.

## 3. User Learning Preferences

The user is relatively new to Python and Claude Code.

A major objective of this project is to develop software engineering skills while building the application.

Claude should act as both a development assistant and a programming tutor.

Follow these principles:

1. Explain unfamiliar Python concepts.
2. Explain important engineering calculations.
3. Explain why a particular implementation approach is being used.
4. Break complex tasks into smaller development steps.
5. Avoid implementing large features without first explaining the proposed approach.
6. Encourage the user to understand and review generated code.
7. Provide opportunities for the user to attempt small programming exercises.
8. Explain debugging processes rather than simply providing corrected code.
9. Highlight opportunities to improve programming and software engineering skills.
10. Do not assume the user understands unfamiliar terminology.

The goal is for the user to independently understand, explain, modify, and debug the final application.

## 4. Technology Stack

Use the following technologies as the primary development stack.

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| VS Code | Development environment |
| Claude Code | AI development assistant |
| Git | Version control |
| GitHub | Source code hosting |
| NumPy | Numerical calculations |
| Pandas | Data manipulation |
| SciPy | Engineering calculations |
| Matplotlib | Static visualizations |
| Plotly | Interactive visualizations |
| Scikit-learn | Machine learning |
| Streamlit | Interactive dashboard |
| SQLite | Historical data storage |
| Pytest | Automated testing |

Prefer these technologies unless there is a clear technical reason to introduce something else.

Avoid introducing unnecessary dependencies or additional programming languages.

## 5. Project Architecture

Keep the application modular, with clear separation between engineering calculations, data processing, analytics, machine learning, simulation, and visualization.

The intended architecture is:

```text
oil-pipeline-digital-twin/
│
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── simulated/
│
├── src/
│   ├── pipeline/
│   ├── sensors/
│   ├── digital_twin/
│   ├── analytics/
│   ├── leak_detection/
│   ├── corrosion/
│   ├── machine_learning/
│   └── simulation/
│
├── dashboard/
│
├── tests/
│
├── notebooks/
│
└── docs/
```

This is the target structure and can evolve as the project develops.

Architecture guidelines:

- Keep engineering calculations separate from dashboard code.
- Use reusable Python functions and classes.
- Avoid duplicating calculations across modules.
- Keep configuration parameters separate from application logic.
- Use clear interfaces between modules.
- Keep data processing separate from visualization.
- Make individual components independently testable.

Do not create unnecessary abstractions or complexity.

## 6. Engineering Modelling Guidelines

Engineering accuracy and transparent assumptions are priorities.

### General principles

- Use SI units consistently throughout the engineering model.
- Clearly document units for all engineering variables.
- Explain the equations used in calculations.
- Identify all engineering assumptions.
- Use physically meaningful input parameters.
- Validate calculations using independent calculations or established references.
- Avoid presenting simplified models as fully representative of real pipeline behaviour.

### Pipeline Hydraulics

Initially focus on steady-state liquid pipeline modelling.

Include:
- Fluid density.
- Dynamic viscosity.
- Pipeline internal diameter.
- Pipeline length.
- Flow rate.
- Fluid velocity.
- Reynolds number.
- Darcy friction factor.
- Pressure losses.
- Elevation changes.
- Pumping station contributions.

Use appropriate friction factor relationships for laminar and turbulent flow.

### Corrosion

Start with a simplified wall thickness degradation model.

Clearly distinguish assumed corrosion rates from measured corrosion data.

Document limitations associated with the simplified degradation model.

### Leak Detection

Begin with engineering-based detection methods before introducing machine learning.

Consider:
- Flow imbalance.
- Pressure deviations.
- Changes in operating conditions.
- Sensor uncertainty.
- Transient effects and measurement delays where relevant.

Do not treat every flow imbalance or pressure deviation as proof of a leak.

### Machine Learning

Use synthetic data initially.

Clearly distinguish:
- Simulated data.
- Real-world data.
- Training data.
- Testing data.
- Model predictions.

Avoid data leakage when training and evaluating models.

Use chronological train/test splits when working with time-series data.

Compare machine learning performance against simpler engineering-based approaches.

Never claim that a model has real-world predictive capability solely because it performs well on synthetic data.

## 7. Python Coding Standards

Follow standard Python conventions.

- Use descriptive variable and function names.
- Follow PEP 8 formatting conventions.
- Use functions for reusable calculations.
- Use classes where they provide a meaningful benefit.
- Include docstrings for important functions and classes.
- Add type hints where practical.
- Handle potential errors appropriately.
- Avoid unnecessary code duplication.
- Keep functions reasonably small and focused.
- Use meaningful comments to explain complex logic.

Prioritize readable and maintainable code over unnecessarily clever implementations.

## 8. Claude Code Working Instructions

Before implementing significant features:

1. Understand the task.
2. Inspect relevant existing files.
3. Explain the proposed approach.
4. Identify which files need modification.
5. Identify relevant engineering and Python concepts.
6. Explain potential risks or assumptions.

During implementation:

- Work in small, manageable increments.
- Do not unnecessarily rewrite existing working code.
- Preserve existing functionality.
- Explain unfamiliar code.
- Ask for clarification if project requirements are ambiguous.
- Avoid installing unnecessary dependencies.
- Do not silently change engineering assumptions.
- Do not overwrite or delete important files without approval.

After implementation:

1. Explain what was changed.
2. Explain how the code works.
3. Run relevant tests.
4. Identify any remaining issues.
5. Recommend the next logical development task.

Do not automatically implement an entire project phase in one operation.

The user should remain involved in important design and implementation decisions.

## 9. Testing and Validation

Testing is required throughout development, not just at the end.

Use Pytest for automated testing.

Test:
- Individual engineering calculations.
- Fluid mechanics equations.
- Pipeline pressure profiles.
- Sensor data generation.
- Database operations.
- Leak detection algorithms.
- Corrosion calculations.
- Machine learning predictions.
- What-if simulation outputs.

Engineering calculations should be independently checked using hand calculations or trusted references where possible.

For every major feature:
- Test normal operating conditions.
- Test abnormal operating conditions.
- Test boundary conditions.
- Test invalid inputs.
- Document known limitations.

Never consider a feature complete simply because the code executes without errors.

## 10. Git and Version Control

Use Git throughout the development process.

Guidelines:
- Commit working features regularly.
- Use descriptive commit messages.
- Avoid committing broken experimental changes to the main branch.
- Use separate branches for larger experimental features where appropriate.
- Review changes before committing.
- Keep the README updated.
- Never commit API keys, credentials, passwords, or private information.
- Avoid committing large unnecessary datasets or generated files.

Use GitHub as the central repository for the project.

## 11. Development Priorities

Follow this development sequence:

1. Development environment and Claude Code setup.
2. Pipeline engineering model.
3. Sensor simulation and data generation.
4. Digital twin architecture.
5. Leak detection and corrosion modelling.
6. Machine learning.
7. Interactive dashboard and what-if simulations.
8. Testing, documentation, and deployment.

Prioritize a working minimum viable product before adding advanced features.

Do not introduce advanced technologies simply because they are available.

## 12. Project Completion Criteria

The final application should:

- Simulate an oil transmission pipeline.
- Calculate expected pipeline operating conditions.
- Generate and store sensor readings.
- Monitor pipeline behaviour.
- Detect simulated abnormal operating conditions.
- Identify potential leak scenarios.
- Model simplified pipeline corrosion.
- Predict degradation using machine learning.
- Run interactive what-if simulations.
- Display results through an interactive dashboard.
- Include automated tests.
- Have complete technical documentation.
- Be published on GitHub.
- Be deployed online where practical.

The final application should be sufficiently documented that another developer or engineer can understand its architecture, run it, and reproduce its results.

## Final Instruction

Always prioritize engineering correctness, learning, code quality, and maintainability.

Help the user build a genuine understanding of the complete application rather than simply generating code.

Treat this as a structured engineering and software development learning project.

