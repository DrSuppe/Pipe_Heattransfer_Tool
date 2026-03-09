##########################################################################
# __author__         = "Tim Kayser"
# __date__           = "09.03.2026"
# __version__        = "2.1"
# __maintainer__     = "Tim Kayser"
# __email__          = "kaysert@purdue.edu"
# __status__         = "Open Beta"
# __copyright__      = "Copyright 2026"
# __credits__        = ["Tim Kayser"]
# __license__        = "GPL"
##########################################################################
"""Simulation defaults and parameter validation helpers."""

import datetime
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

def _prune_run_dirs(prefix: str, max_keep: int) -> None:
    if max_keep <= 0:
        return
    root = Path(prefix)
    if not root.exists():
        return
    dirs = [p for p in root.glob("run_*") if p.is_dir()]
    # Keep at most max_keep historical run directories; reserve one slot for the upcoming run.
    n_remove = max(0, len(dirs) - max_keep + 1)
    if n_remove <= 0:
        return
    dirs.sort(key=lambda p: p.stat().st_mtime)
    for d in dirs[:n_remove]:
        try:
            shutil.rmtree(d)
            logging.info("Pruned old run directory: %s", d)
        except Exception as exc:
            logging.warning("Failed to prune old run directory %s: %s", d, exc)


def _make_run_dir(prefix="runs", max_keep: int = 1000):
    _prune_run_dirs(prefix, max_keep=max_keep)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    outdir = Path(prefix) / f"run_{ts}"
    outdir.mkdir(parents=True, exist_ok=False)
    return outdir

params = {
    "L": 65, "Di": 0.13, "t_wall": 0.018, "t_ins": 0.15, "Nx": 2500,
    "p": 5.0e6, "m_dot": 2.5, "Tin": 1100.0,
    "T_init_wall": 300.0, "T_init_ins": 300.0, "T_init_gas": 300.0, "Tamb": 300.0,
    "rho_w": 8220.0, "cp_w": 564.0, "k_w": 21.9,
    "rho_i": 128.0, "cp_i": 1246, "k_i": 0.3,
    "cp_g": 1005.0, "k_g": 0.028, "mu_g": 2.0e-5, "Pr": 0.71, "dittus_boelter_n": 0.35,
    "h_out": 14.0, "h_out_mode": "auto", "eps_rad": 0.23,
    "t_end": 5000.0, "CFL": 0.6, "theta_cond": 0.5,
    "target_metric": "gas_outlet",      # stop metric for target mode

    # Performance/stability controls
    "adv_scheme": "semi_lagrangian",   # options: "semi_lagrangian" or "upwind"
    "semi_lag_courant_max": 2.0,       # semi-Lagrangian accuracy cap (Courant-like)
    "dt_max": 5.0,                     # hard cap for adaptive time step [s]
    "dt_min": 1.0e-4,  # or some small positive number
    "Tin_ramp_s": 900.0,               # inlet temperature ramp duration [s]
    "Tin_ramp_model": "logistic",      # inlet ramp model: "logistic" (default), "linear", or "heater_exp"
    "Tin_ramp_shape": 8.0,             # logistic steepness parameter (dimensionless, >0)
    "save_frames": 240,                # reduce I/O and RAM (was 1000)
    "log_interval_s": 10.0,            # runtime log interval in wall seconds
    "log_interval_steps": 1000,        # runtime log interval in steps
    "use_float32": True,
    "dt_quantize_pct": 0.1,
    "update_props_every": 5,  # recompute h_in,u every 5 steps (reduces div/pow work)
    "prop_update_temp_threshold_k": 2.0,  # rebuild temp-dependent solids only if max ΔT exceeds threshold
    "prop_update_force_steps": 25,        # hard upper bound between temp-dependent solid rebuilds
    "insulation_mass_mode": "penetration",  # "full" or "penetration"
    "insulation_mass_min_frac": 0.25,       # minimum effective insulation mass fraction for penetration mode
    "insulation_penetration_time_s": 0.0,   # <=0 uses run horizon (t_end)
    "target_asymptote_check": True,          # early-stop target runs if metric asymptotes before target
    "target_asymptote_window_s": 600.0,      # trailing window used for slope estimate
    "target_asymptote_rate_tol_k_per_s": 2.5e-4,  # improvement-rate threshold (~0.9 K/hr)
    "target_asymptote_min_gap_k": 1.0,       # only trigger if still this far from target
    "target_asymptote_min_time_s": 900.0,    # do not evaluate asymptote too early
    "target_asymptote_projection_factor": 1.25,  # projected time-to-target vs remaining time factor
    "target_asymptote_stall_windows": 3,     # consecutive windows required before stop

    "parallel": False,  # single-threaded kernel (faster on mixed-core CPUs)
    "progress": "basic",  # "none" (quiet) or "basic" (periodic prints)
    "enable_numba": not sys.platform.startswith("win"),  # default off on Windows for stability
    "log_to_file": True,  # write run.log into output directory
    "write_trace_csv": True,  # write runtime_trace.csv
    "max_run_dirs": 1000,  # cap number of run_* folders in output root (oldest pruned)

    # Solid-property and local thermal-mass extensions
    "use_temp_dependent_props": True,
    "pipe_prop_table": None,  # {"T":[...], "cp":[...], "k":[...]}
    "ins_prop_table": None,   # {"T":[...], "cp":[...], "k":[...]}
    "thermal_mass_count": 0,
    "thermal_mass_factor": 0.0,        # added local wall heat-capacity multiplier at each attachment
    "thermal_mass_positions_frac": [],  # normalized [0..1]
    "thermal_mass_spread_frac": 0.03,   # spatial spreading of each attachment influence

}

def validate_params(p: dict) -> None:
    """Cheap sanity checks to catch bad input early."""
    # geometry
    if p["L"] <= 0:
        raise ValueError("L must be > 0")
    if p["Di"] <= 0:
        raise ValueError("Di must be > 0")
    if p["t_wall"] < 0:
        raise ValueError("t_wall must be >= 0")
    if p["t_ins"] < 0:
        raise ValueError("t_ins must be >= 0")

    # materials / thermo
    for k in ("rho_w", "cp_w", "k_w", "rho_i", "cp_i", "k_i", "cp_g", "k_g"):
        if p[k] <= 0:
            raise ValueError(f"{k} must be > 0")
    if p["mu_g"] <= 0:
        raise ValueError("mu_g must be > 0")
    if p["p"] <= 0:
        raise ValueError("p must be > 0")
    if p["m_dot"] <= 0:
        raise ValueError("m_dot must be > 0")

    # HT / radiation
    if not (0.0 <= p["eps_rad"] <= 1.0):
        raise ValueError("eps_rad must be in [0, 1]")
    if p["h_out"] < 0.0:
        raise ValueError("h_out must be >= 0")
    if p.get("h_out_mode", "auto") not in ("auto", "manual"):
        raise ValueError("h_out_mode must be 'auto' or 'manual'")

    # numerics
    if p["Nx"] < 3:
        raise ValueError("Nx must be >= 3")
    if p["dt_max"] <= 0 or p["dt_min"] <= 0:
        raise ValueError("dt_max and dt_min must be > 0")
    if p["dt_min"] > p["dt_max"]:
        raise ValueError("dt_min must be <= dt_max")
    if p.get("Tin_ramp_s", 0.0) < 0:
        raise ValueError("Tin_ramp_s must be >= 0")
    if str(p.get("Tin_ramp_model", "logistic")).lower() not in ("heater_exp", "linear", "logistic"):
        raise ValueError("Tin_ramp_model must be 'heater_exp', 'linear', or 'logistic'")
    if float(p.get("Tin_ramp_shape", 8.0)) <= 0.0:
        raise ValueError("Tin_ramp_shape must be > 0")
    if not (0.0 < p["theta_cond"] <= 1.0):
        raise ValueError("theta_cond must be in (0, 1]")
    if p["CFL"] <= 0:
        raise ValueError("CFL must be > 0")
    if float(p.get("semi_lag_courant_max", 2.0)) <= 0.0:
        raise ValueError("semi_lag_courant_max must be > 0")
    if float(p.get("prop_update_temp_threshold_k", 2.0)) < 0.0:
        raise ValueError("prop_update_temp_threshold_k must be >= 0")
    if int(p.get("prop_update_force_steps", 25)) < 1:
        raise ValueError("prop_update_force_steps must be >= 1")
    if str(p.get("insulation_mass_mode", "penetration")).lower() not in ("full", "penetration"):
        raise ValueError("insulation_mass_mode must be 'full' or 'penetration'")
    if float(p.get("insulation_mass_min_frac", 0.25)) <= 0.0 or float(p.get("insulation_mass_min_frac", 0.25)) > 1.0:
        raise ValueError("insulation_mass_min_frac must be in (0, 1]")
    if float(p.get("insulation_penetration_time_s", 0.0)) < 0.0:
        raise ValueError("insulation_penetration_time_s must be >= 0")
    if float(p.get("target_asymptote_window_s", 600.0)) <= 0.0:
        raise ValueError("target_asymptote_window_s must be > 0")
    if float(p.get("target_asymptote_rate_tol_k_per_s", 2.5e-4)) < 0.0:
        raise ValueError("target_asymptote_rate_tol_k_per_s must be >= 0")
    if float(p.get("target_asymptote_min_gap_k", 1.0)) < 0.0:
        raise ValueError("target_asymptote_min_gap_k must be >= 0")
    if float(p.get("target_asymptote_min_time_s", 900.0)) < 0.0:
        raise ValueError("target_asymptote_min_time_s must be >= 0")
    if float(p.get("target_asymptote_projection_factor", 1.25)) <= 0.0:
        raise ValueError("target_asymptote_projection_factor must be > 0")
    if int(p.get("target_asymptote_stall_windows", 3)) < 1:
        raise ValueError("target_asymptote_stall_windows must be >= 1")
    if p.get("max_run_dirs", 1000) < 0:
        raise ValueError("max_run_dirs must be >= 0")
    if p.get("target_metric", "gas_outlet") not in (
        "gas_outlet",
        "wall_inner_outlet",
        "wall_outer_outlet",
        "insulation_outlet",
    ):
        raise ValueError("target_metric must be one of gas_outlet/wall_inner_outlet/wall_outer_outlet/insulation_outlet")
    if p.get("thermal_mass_count", 0) < 0:
        raise ValueError("thermal_mass_count must be >= 0")
    if p.get("thermal_mass_factor", 0.0) < 0.0:
        raise ValueError("thermal_mass_factor must be >= 0")
    if p.get("thermal_mass_spread_frac", 0.03) < 0.0:
        raise ValueError("thermal_mass_spread_frac must be >= 0")


def _prepare_prop_table(raw_table: Any, cp_default: float, k_default: float):
    if not isinstance(raw_table, dict):
        return None
    try:
        t = np.asarray(raw_table.get("T", []), dtype=float).reshape(-1)
        cp = np.asarray(raw_table.get("cp", []), dtype=float).reshape(-1)
        k = np.asarray(raw_table.get("k", []), dtype=float).reshape(-1)
    except Exception:
        return None
    if t.size < 2 or cp.size != t.size or k.size != t.size:
        return None
    order = np.argsort(t)
    t = t[order]
    cp = np.maximum(cp[order], 1.0e-9)
    k = np.maximum(k[order], 1.0e-9)
    if np.any(np.diff(t) <= 0.0):
        return None
    # Keep default values as fallback for extrapolation guards.
    return {
        "T": t,
        "cp": cp,
        "k": k,
        "cp_default": float(cp_default),
        "k_default": float(k_default),
    }


def _interp_props(temp_vec: np.ndarray, table: dict | None, cp_default: float, k_default: float):
    if table is None:
        cp = np.full_like(temp_vec, float(cp_default), dtype=float)
        k = np.full_like(temp_vec, float(k_default), dtype=float)
        return cp, k
    cp = np.interp(
        temp_vec,
        table["T"],
        table["cp"],
        left=float(table.get("cp_default", cp_default)),
        right=float(table.get("cp_default", cp_default)),
    )
    k = np.interp(
        temp_vec,
        table["T"],
        table["k"],
        left=float(table.get("k_default", k_default)),
        right=float(table.get("k_default", k_default)),
    )
    return np.maximum(cp, 1.0e-9), np.maximum(k, 1.0e-9)


def _build_thermal_mass_profile(nx: int, count: int, factor: float, positions_frac: list[float] | np.ndarray, spread_frac: float):
    profile = np.ones(int(max(1, nx)), dtype=float)
    n = int(max(0, count))
    if n <= 0 or factor <= 0.0 or nx <= 1:
        return profile
    if positions_frac is None:
        pos = []
    else:
        pos = [float(v) for v in positions_frac]
    if len(pos) != n:
        pos = [float(v) for v in np.linspace(0.15, 0.85, n)]
    spread = max(1, int(max(1.0e-6, float(spread_frac)) * nx))
    radius = 3 * spread
    for frac in pos:
        j0 = int(round(max(0.0, min(1.0, float(frac))) * (nx - 1)))
        for dj in range(-radius, radius + 1):
            j = j0 + dj
            if j < 0 or j >= nx:
                continue
            # Raised-cosine bump keeps the same compact support as before but with smooth edges.
            w = 0.5 * (1.0 + np.cos(np.pi * float(dj) / float(radius)))
            profile[j] += float(factor) * w
    return np.maximum(profile, 1.0)
