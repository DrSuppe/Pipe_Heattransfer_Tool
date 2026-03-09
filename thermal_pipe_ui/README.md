# thermal_pipe_ui package

This package contains the PyQt6 desktop UI, split into focused modules.

## Responsibilities by file

- `window.py`
  - Thin application window/orchestration class (`ThermalPipeWindow`).
  - Shared constants, unit conversions, and high-level run wiring.

- `panels.py`
  - UI construction for tabs/groups/widgets.
  - Keeps layout and control creation separate from behavior logic.

- `plotting.py`
  - Live/results plot rendering and interaction behavior.
  - Plot popup, hover datapoint labels, right-click save, and image export helpers.

- `optimization.py`
  - Worker-thread simulation execution and optimization search logic.
  - Target metric utilities and stress-estimation helpers for optimization mode.

- `persistence.py`
  - Presets, ledger I/O, README display, and export bundle generation.

- `__init__.py`
  - Package entrypoint exports (`ThermalPipeWindow`, `main`).

## Notes for contributors

- Add new widgets/tabs in `panels.py`.
- Add plot behavior in `plotting.py`.
- Add save/load/export logic in `persistence.py`.
- Keep `window.py` as orchestration glue, not a catch-all module.
