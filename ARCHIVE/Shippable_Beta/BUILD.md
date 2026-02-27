# Build Standalone Binaries (Beta)

This app can be packaged as a desktop executable for macOS and Windows with PyInstaller.

## Important

- Build on the target OS:
  - macOS binary on macOS
  - Windows binary on Windows
- Use a clean virtual environment.

## 1) Install build tooling

```bash
pip install pyinstaller PyQt6 matplotlib numpy
```

Optional runtime accelerators:

```bash
pip install numba openpyxl
```

## 2) Build command

From project root:

```bash
pyinstaller --noconfirm --windowed --name ThermalPipeSimulator pyqt6_app.py
```

Result:

- macOS: `dist/ThermalPipeSimulator.app`
- Windows: `dist/ThermalPipeSimulator/ThermalPipeSimulator.exe`

## 3) File behavior in packaged app

The executable keeps the same file behavior as Python source mode (subject to user permissions):

- Creates run folders and files (`runs/run_*`, logs, arrays, plots).
- Maintains ledger file (`run_history.csv` or selected path).
- Reads/writes custom pipe preset library (`pipe_presets.json`).
- Supports one-click export bundle (`.zip`).

If colleagues cannot access the default working directory, they should choose a writable save folder in the UI.

## 4) Distribution checks

Before sharing:

1. Run a smoke test in packaged app (`Fixed time` mode, save enabled).
2. Verify ledger append/delete in `Ledger` tab.
3. Verify `Export Bundle` creates expected `.zip`.
4. Confirm `run_*` retention cap behaves as intended.
5. Review [SECURITY.md](SECURITY.md) checklist.
