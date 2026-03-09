# Build Standalone Binaries (Beta)

This app can be packaged as a desktop executable for macOS and Windows with PyInstaller.

## Important

- Build on the target OS:
  - macOS binary on macOS
  - Windows binary on Windows
- Use a clean virtual environment.

## 1) Install build tooling

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install pyinstaller
```

## 2) Compile/syntax readiness check (recommended before packaging)

```bash
python3 -m compileall -q Pipe_Sim_V4.py sim_controller.py gui.py pyqt6_app.py thermal_pipe thermal_pipe_ui
```

Optional no-write solver smoke check:

```bash
python3 - <<'PY'
from Pipe_Sim_V4 import run_simulation
r = run_simulation({"Nx": 64, "t_end": 2.0, "save_frames": 4})
print("smoke_ok", r.n_steps, round(r.Tg_outlet_final, 3))
PY
```

## 3) Build command

From project root:

```bash
pyinstaller --noconfirm --windowed --name ThermalPipeSimulator pyqt6_app.py
```

Result:

- macOS: `dist/ThermalPipeSimulator.app`
- Windows: `dist/ThermalPipeSimulator/ThermalPipeSimulator.exe`

## 4) File behavior in packaged app

The executable keeps the same file behavior as Python source mode (subject to user permissions):

- Creates run folders and files (`runs/run_*`, logs, arrays, plots).
- Maintains ledger file (`run_history.csv` or selected path).
- Reads/writes custom pipe preset library (`pipe_presets.json`).
- Supports one-click export bundle (`.zip`).

If colleagues cannot access the default working directory, they should choose a writable save folder in the UI.

## 5) Distribution checks

Before sharing:

1. Run a smoke test in packaged app (`Fixed time` mode, save enabled).
2. Verify ledger append/delete in `Ledger` tab.
3. Verify `Export Bundle` creates expected `.zip`.
4. Confirm `run_*` retention cap behaves as intended.
5. Review [SECURITY.md](SECURITY.md) checklist.
