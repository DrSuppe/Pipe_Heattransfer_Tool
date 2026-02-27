# sim_controller.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List
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
    m_dot: float = 1.0     # kg/s
    # control mode
    mode: str = "time"     # "time" or "target"
    t_end: float = 1800.0  # s (used if mode == "time")
    Tg_out_target: Optional[float] = None  # K (used if mode == "target")
    stop_dir: Optional[str] = None         # "le" or "ge" (auto if None)

@dataclass
class RunSpec:
    hardware: HardwareConfig
    run: RunInputs
    save_dir: Optional[Path] = None
    make_plots: bool = True
    save_results: bool = True

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def run_once(spec: RunSpec) -> SimRunResult:
    overrides = {
        "p": spec.run.p,
        "Tin": spec.run.Tin,
        "m_dot": spec.run.m_dot,
        **spec.hardware.to_overrides(),
    }
    # Select control mode
    if spec.run.mode == "time":
        overrides["t_end"] = float(spec.run.t_end)
        res = main(
            overrides,
            make_plots=spec.make_plots,
            save_results=spec.save_results,
            outdir=spec.save_dir,
            stop_at_Tg_outlet=None,
            stop_dir=None,
            max_sim_time=None,  # rely on overrides["t_end"]
        )
    elif spec.run.mode == "target":
        # Use max_sim_time as a cap; t_end still exists in defaults but we override with cap
        cap = float(spec.run.t_end) if spec.run.t_end else None
        res = main(
            overrides,
            make_plots=spec.make_plots,
            save_results=spec.save_results,
            outdir=spec.save_dir,
            stop_at_Tg_outlet=float(spec.run.Tg_out_target),
            stop_dir=spec.run.stop_dir,
            max_sim_time=cap,
        )
    else:
        raise ValueError(f"Unknown mode: {spec.run.mode}")

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
            save_dir=outdir,
            make_plots=spec.make_plots,
            save_results=spec.save_results,
        )
        results.append(run_once(rs))
    return results