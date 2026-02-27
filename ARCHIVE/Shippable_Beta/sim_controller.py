# sim_controller.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime

# Adjust this import to your actual module name
from Pipe_Sim_V4 import main, SimRunResult

@dataclass
class HardwareConfig:
    # keep this small in the GUI; add more as needed
    L: float = 65.0
    Di: float = 0.13
    t_wall: float = 0.018
    t_ins: float = 0.15
    h_out: float = 8.0
    eps_rad: float = 0.7
    # You can add materials (rho, cp, k) if you want them editable

    def to_overrides(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class RunInputs:
    p: float               # Pa
    Tin: float             # K
    m_dot: float = 2.5     # kg/s
    # control mode
    mode: str = "time"     # "time" or "target"
    t_end: float = 5000.0  # s (used if mode == "time")
    Tg_out_target: Optional[float] = None  # K (used if mode == "target")
    stop_dir: Optional[str] = None         # "le" or "ge" (auto if None)

@dataclass
class RunSpec:
    hardware: HardwareConfig
    run: RunInputs
    overrides: Optional[Dict[str, Any]] = None
    save_dir: Optional[Path] = None
    make_plots: bool = True
    save_results: bool = True
    snapshot_callback: Optional[Callable[[float, Any, Any, Any], None]] = None
    abort_callback: Optional[Callable[[], bool]] = None

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def run_once(spec: RunSpec) -> SimRunResult:
    overrides = {
        "p": spec.run.p,
        "Tin": spec.run.Tin,
        "m_dot": spec.run.m_dot,
        **spec.hardware.to_overrides(),
    }
    if spec.overrides:
        overrides.update(spec.overrides)

    mode = spec.run.mode

    if mode == "time":
        overrides["t_end"] = float(spec.run.t_end)
        res = main(
            overrides,
            make_plots=spec.make_plots,
            save_results=spec.save_results,
            outdir=spec.save_dir,
            stop_at_Tg_outlet=None,
            stop_dir=None,
            max_sim_time=None,
            snapshot_callback=spec.snapshot_callback,
            abort_callback=spec.abort_callback,
        )

    elif mode == "target":
        if spec.run.Tg_out_target is None:
            raise ValueError("Tg_out_target must be set when mode='target'.")
        cap = float(spec.run.t_end) if spec.run.t_end else None
        res = main(
            overrides,
            make_plots=spec.make_plots,
            save_results=spec.save_results,
            outdir=spec.save_dir,
            stop_at_Tg_outlet=float(spec.run.Tg_out_target),
            stop_dir=spec.run.stop_dir,
            max_sim_time=cap,
            snapshot_callback=spec.snapshot_callback,
            abort_callback=spec.abort_callback,
        )

    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'time' or 'target'.")

    return res

def run_sweep(specs: List[RunSpec]) -> List[SimRunResult]:
    results = []
    for i, spec in enumerate(specs):
        # Give each run a folder unless caller passed a specific one
        outdir = spec.save_dir
        if spec.save_results and outdir is None:
            tag = f"run_{i:02d}_{_timestamp()}"
            outdir = Path("runs") / tag
        rs = RunSpec(
            hardware=spec.hardware,
            run=spec.run,
            overrides=spec.overrides,
            save_dir=outdir,
            make_plots=spec.make_plots,
            save_results=spec.save_results,
            snapshot_callback=spec.snapshot_callback,
            abort_callback=spec.abort_callback,
        )
        results.append(run_once(rs))
    return results
