# thermal_pipe package

This package contains the core simulation model and solver support code.

## Responsibilities by file

- `config.py`
  - Centralized defaults and parameter normalization.
  - Unit-safe bounds and compatibility helpers for solver overrides.

- `numerics.py`
  - Core numerical kernels and update routines for heat transfer state evolution.
  - Time-step helpers and stability-related calculations.

- `runtime.py`
  - Simulation orchestration for a run (initialization, stepping loop, stop logic).
  - Bridges user inputs and numerics into a coherent run lifecycle.

- `outputs.py`
  - Result packaging, summaries, and output artifact helpers.
  - Plot/data export utilities used by higher-level entrypoints.

- `__init__.py`
  - Re-exports package-level public API for compatibility imports.

## Notes for contributors

- Put physics/state update logic in `numerics.py`.
- Put run-control flow in `runtime.py`.
- Keep user-facing defaults/override validation in `config.py`.
- Keep file/plot/export side effects in `outputs.py`.
