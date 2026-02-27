# Security Notes (Beta)

This is a practical security baseline for distributing the Thermal Pipe Simulator to colleagues.

## Threat Model (Pragmatic)

- Trusted internal users on company-managed machines.
- Inputs are user-entered engineering values and local file paths.
- No intended network communication.

## Current Safeguards

- GUI expression parsing in legacy Tk path uses restricted AST evaluation (no arbitrary code execution path expected).
- No shell command execution in the main simulation/GUI runtime.
- No dynamic plugin loading from untrusted sources.
- Run artifacts are written as local data files (`json/csv/npz/png/log`).
- Run-directory retention only deletes `run_*` folders within configured output root.

## Residual Risks

- Any desktop binary can write files where user has permission; user training is still required.
- Very large simulations can still consume CPU/RAM/disk if configured aggressively.
- External dependencies (PyQt6, numpy, matplotlib, optional openpyxl/numba) may introduce supply-chain risk if unmanaged.

## Release Checklist

1. Dependency pinning:
- Pin exact package versions in a lockfile for release builds.
- Build in a clean environment.

2. Static checks:
- Search for dangerous APIs (`eval`, `exec`, `subprocess`, `os.system`, unsafe deserialization).
- Verify no new network clients were introduced.

3. File-write controls:
- Confirm run-output retention cap works as expected.
- Confirm ledger/library files are not touched by retention pruning.

4. Packaging:
- Build per-platform binary on native OS.
- Sign/notarize binaries according to company policy.

5. Operational limits:
- Validate defaults for `Nx`, `save_frames`, and simulation duration.
- Confirm optimization modes provide explicit constraint status/warnings.

## Recommended Next Hardening

- Add a lightweight "safe mode" profile for distributed builds:
  - upper bounds on `Nx`, `save_frames`, and `t_end`,
  - disabled legacy Tk UI path if not needed.
- Add optional integrity check for bundled material/preset files.
- Add CI security job (`pip-audit`, Bandit, simple forbidden-pattern grep).
