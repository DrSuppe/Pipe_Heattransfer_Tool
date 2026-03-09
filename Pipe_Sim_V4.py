##########################################################################
# __author__         = "Tim Kayser"
# __date__           = "27.02.2026"
# __version__        = "2.0"
# __maintainer__     = "Tim Kayser"
# __email__          = "kaysert@purdue.edu"
# __status__         = "Open Beta"
# __copyright__      = "Copyright 2026"
# __credits__        = ["Tim Kayser"]
# __license__        = "GPL"
##########################################################################
"""Core transient thermal solver for gas, wall, and insulation temperature fields."""

import datetime
import json
import logging
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

try:
    from numba import njit, prange
except Exception:
    def njit(*args, **kwargs):
        def _identity(func):
            return func
        return _identity

    def prange(*args):
        return range(*args)

# --- Global tolerances and feature gates ---
EPS = 1e-12          # small positive epsilon for denominators
CN_REL_TOL = 0.05    # relative threshold to rebuild CN factors
HAS_NUMBA = False     # flipped to True if numba import succeeds

# --- Output directory and logging setup ---
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


# --- Global dtype selector and helpers ---
DTYPE = np.float32 if params.get("use_float32", False) else np.float64
def _as_dtype(x):
    return np.asarray(x, dtype=DTYPE)
def _float(x):
    # Cast scalar to the configured numpy scalar type (e.g., np.float32 or np.float64)
    return DTYPE(x)

R = _float(287.058)
sigma = _float(5.670374419e-8)
L, Di = _float(params["L"]), _float(params["Di"])
r_i = _float(Di/2)
r_w_o = _float(r_i + params["t_wall"])
r_ins_o = _float(r_w_o + params["t_ins"])
A_flow = _float(np.pi*(Di**2)/4)
P_in = _float(np.pi*Di)
P_out = _float(np.pi*2*r_ins_o)
Nx = params["Nx"]
x = np.linspace(0, L, Nx, dtype=DTYPE)
dx = _float(x[1] - x[0])

Vw_cell = _float(np.pi*(r_w_o**2 - r_i**2)*dx)
Vi_cell = _float(np.pi*(r_ins_o**2 - r_w_o**2)*dx)
Cw_cell = _float(params["rho_w"]*Vw_cell*params["cp_w"])
Ci_cell = _float(params["rho_i"]*Vi_cell*params["cp_i"])
aw = _float(params["k_w"]/(params["rho_w"]*params["cp_w"]))
ai = _float(params["k_i"]/(params["rho_i"]*params["cp_i"]))
R_wall = _float(np.log(r_w_o/r_i)/(2*np.pi*params["k_w"]*dx))
R_ins = _float(np.log(r_ins_o/r_w_o)/(2*np.pi*params["k_i"]*dx))
R_Tw_to_Ti = _float(R_wall + R_ins)

def compute_h_in(Tg):
    """Internal convection via Gnielinski correlation (with laminar/transition blending)."""
    rho = params["p"]/(R*Tg)
    u = params["m_dot"]/(rho*A_flow)
    Re = rho*u*Di/params["mu_g"]
    Re = np.maximum(Re,1.0)

    # Gnielinski valid for turbulent flow; blend with laminar Nu=3.66 through transition.
    Re_eff = np.maximum(Re, 3000.0)
    f = (0.79*np.log(Re_eff) - 1.64)**(-2.0)
    Nu_turb = (f/8.0)*(Re_eff - 1000.0)*params["Pr"] / (
        1.0 + 12.7*np.sqrt(f/8.0)*(params["Pr"]**(2.0/3.0) - 1.0)
    )
    Nu_lam = np.full_like(Re, 3.66)
    w = np.clip((Re - 2300.0)/700.0, 0.0, 1.0)
    Nu = np.where(Re < 2300.0, Nu_lam, np.where(Re < 3000.0, (1.0-w)*Nu_lam + w*Nu_turb, Nu_turb))
    return Nu*params["k_g"]/Di, u


# Natural-convection constants (air, Churchill-Chu correlation).
# Precomputing the denominator avoids repeated pow() work in per-step calls.
_G_AIR = 9.81
_NU_AIR = 1.6e-5
_ALPHA_AIR = 2.3e-5
_K_AIR = 0.026
_PR_AIR = _NU_AIR / _ALPHA_AIR
_CC_DENOM = (1.0 + (0.559 / _PR_AIR) ** (9.0 / 16.0)) ** (8.0 / 27.0)


@njit(cache=True, fastmath=True)
def _h_out_natural_conv_scalar(Ts, Tamb, D_out):
    # Churchill-Chu natural convection over a horizontal cylinder (air properties approximated)
    tf = 0.5 * (Ts + Tamb)
    if tf < 1.0:
        tf = 1.0
    beta = 1.0 / tf
    dT = abs(Ts - Tamb) + 1.0e-6
    D = D_out if D_out > 1.0e-6 else 1.0e-6
    Ra = _G_AIR * beta * dT * (D ** 3) / (_NU_AIR * _ALPHA_AIR + 1.0e-20)
    if Ra < 1.0e-14:
        return 1.0
    Nu = (0.60 + (0.387 * (Ra ** (1.0 / 6.0))) / _CC_DENOM) ** 2
    h = Nu * _K_AIR / D
    if h < 1.0:
        h = 1.0
    return h


def _h_out_natural_conv_vec(Ts_vec, Tamb, D_out):
    tf = np.maximum(0.5 * (Ts_vec + Tamb), 1.0)
    beta = 1.0 / tf
    dT = np.abs(Ts_vec - Tamb) + 1.0e-6
    D = max(float(D_out), 1.0e-6)
    Ra = _G_AIR * beta * dT * (D ** 3) / (_NU_AIR * _ALPHA_AIR + 1.0e-20)
    Nu = (0.60 + (0.387 * np.power(np.maximum(Ra, 1.0e-14), 1.0 / 6.0)) / _CC_DENOM) ** 2
    h = Nu * _K_AIR / D
    return np.maximum(h, 1.0)

try:
    from numba import njit, prange
    HAS_NUMBA = True

    @njit(cache=True, fastmath=True)
    def _diffuse_axial_CN_numba(T, alpha, dt, dx, theta):
        if alpha <= 0.0:
            return T
        n = T.shape[0]
        lam = alpha * dt / (dx * dx)
        a = np.zeros(n, dtype=T.dtype)
        b = np.ones(n, dtype=T.dtype)
        c = np.zeros(n, dtype=T.dtype)
        RHS = np.empty(n, dtype=T.dtype)
        L_T = np.empty(n, dtype=T.dtype)

        for i in range(1, n - 1):
            a[i] = -theta * lam
            b[i] = 1.0 + 2.0 * theta * lam
            c[i] = -theta * lam

        b[0] = 1.0 + 2.0 * theta * lam
        c[0] = -2.0 * theta * lam
        a[n - 1] = -2.0 * theta * lam
        b[n - 1] = 1.0 + 2.0 * theta * lam

        # Laplacian (Neumann-like end treatment used in original code)
        L_T[0] = 2.0 * (T[1] - T[0])
        for i in range(1, n - 1):
            L_T[i] = T[i + 1] - 2.0 * T[i] + T[i - 1]
        L_T[n - 1] = 2.0 * (T[n - 2] - T[n - 1])

        for i in range(n):
            RHS[i] = T[i] + (1.0 - theta) * lam * L_T[i]

        # Thomas algorithm
        for i in range(1, n):
            m = a[i] / b[i - 1]
            b[i] -= m * c[i - 1]
            RHS[i] -= m * RHS[i - 1]

        Tnew = np.empty_like(T)
        Tnew[n - 1] = RHS[n - 1] / b[n - 1]
        for i in range(n - 2, -1, -1):
            Tnew[i] = (RHS[i] - c[i] * Tnew[i + 1]) / b[i]

        return Tnew

    @njit(cache=True, fastmath=True)
    def _cn_solve_with_cache(T, lam, theta, a, cprime, inv_denom):
        n = T.shape[0]
        # Build RHS = T + (1-theta)*lam*L(T)
        RHS = np.empty(n, dtype=T.dtype)
        # Laplacian with Neumann-like end treatment
        if n > 1:
            RHS[0] = T[0] + (1.0 - theta) * lam * (2.0 * (T[1] - T[0]))
            for i in range(1, n - 1):
                RHS[i] = T[i] + (1.0 - theta) * lam * (T[i + 1] - 2.0 * T[i] + T[i - 1])
            RHS[n - 1] = T[n - 1] + (1.0 - theta) * lam * (2.0 * (T[n - 2] - T[n - 1]))
        else:
            RHS[0] = T[0]

        # Forward sweep using cached c' and inv_denom
        y = np.empty(n, dtype=T.dtype)
        y[0] = RHS[0] * inv_denom[0]
        for i in range(1, n):
            y[i] = (RHS[i] - a[i] * y[i - 1]) * inv_denom[i]

        # Back substitution using cached c'
        Tnew = np.empty_like(T)
        Tnew[n - 1] = y[n - 1]
        for i in range(n - 2, -1, -1):
            Tnew[i] = y[i] - cprime[i] * Tnew[i + 1]
        return Tnew


    @njit(cache=True, fastmath=True)
    def _cn_solve_with_cache_noalloc(T, lam, theta, a, cprime, inv_denom, RHS, y, out):
        n = T.shape[0]
        # Build RHS = T + (1-theta)*lam*L(T) into RHS (Neumann-like ends)
        if n > 1:
            RHS[0] = T[0] + (1.0 - theta) * lam * (2.0 * (T[1] - T[0]))
            for i in range(1, n - 1):
                RHS[i] = T[i] + (1.0 - theta) * lam * (T[i + 1] - 2.0 * T[i] + T[i - 1])
            RHS[n - 1] = T[n - 1] + (1.0 - theta) * lam * (2.0 * (T[n - 2] - T[n - 1]))
        else:
            RHS[0] = T[0]

        # Forward sweep using cached factors
        y[0] = RHS[0] * inv_denom[0]
        for i in range(1, n):
            y[i] = (RHS[i] - a[i] * y[i - 1]) * inv_denom[i]

        # Back substitution using cached c'
        out[n - 1] = y[n - 1]
        for i in range(n - 2, -1, -1):
            out[i] = y[i] - cprime[i] * out[i + 1]
        return out

    def diffuse_axial_CN(T, alpha, dt, dx, theta):
        # wrapper so callers can pass numpy arrays and get a copy back
        return _diffuse_axial_CN_numba(T, alpha, dt, dx, theta)

    @njit(cache=True, fastmath=True)
    def _compute_adaptive_dt_numba(Tg, Ti, u, h_in, p, R_g, A_flow, dx, cp_g, P_in, Cw_cell_vec, Ci_cell_vec, R_Tw_to_Ti_vec, Tamb, D_out, h_out_manual, use_auto_hout, eps_rad, sigma, P_out):
        n = Tg.shape[0]
        tau_g_min = 1.0e9
        tau_w_min = 1.0e9
        tau_i_min = 1.0e9
        ci_max = 0.0
        for i in range(n):
            rho_g = p / (R_g * Tg[i])
            Cg = rho_g * A_flow * dx * cp_g
            hia = h_in[i] * P_in * dx
            inv_r = 1.0 / (R_Tw_to_Ti_vec[i] if R_Tw_to_Ti_vec[i] > 1e-16 else 1e-16)
            tg_l = Cg / (hia + 1e-12)
            if tg_l < tau_g_min: tau_g_min = tg_l
            tw_l = Cw_cell_vec[i] / (hia + inv_r + 1e-12)
            if tw_l < tau_w_min: tau_w_min = tw_l
            
            ci = Ci_cell_vec[i]
            if ci > ci_max: ci_max = ci
            
            h_o_l = h_out_manual
            if use_auto_hout == 1:
                h_o_l = _h_out_natural_conv_scalar(Ti[i], Tamb, D_out)
            h_o_term = h_o_l * P_out * dx
            ti_safe = Ti[i] if Ti[i] > 1.0 else 1.0
            rad_term = 4.0 * eps_rad * sigma * P_out * dx * (ti_safe * ti_safe * ti_safe)
            ti_l = ci / (inv_r + h_o_term + rad_term + 1e-12)
            if ti_l < tau_i_min: tau_i_min = ti_l
            
        if ci_max <= 1e-12:
            tau_i_min = 1.0e9
            
        u_max = 0.0
        for i in range(n):
            if u[i] > u_max: u_max = u[i]
            
        return tau_g_min, tau_w_min, tau_i_min, u_max

    @njit(cache=True, fastmath=True)
    def _compute_h_in_numba(Tg_vec, h_in, u, p, R_g, m_dot, A_flow, Di, mu_g, Pr, k_g):
        n = Tg_vec.shape[0]
        for i in range(n):
            rho = p / (R_g * Tg_vec[i])
            u_i = m_dot / (rho * A_flow)
            u[i] = u_i
            Re = rho * u_i * Di / mu_g
            if Re < 1.0: Re = 1.0
            Re_eff = Re if Re > 3000.0 else 3000.0
            
            f_log = np.log(Re_eff)
            f = 1.0 / ((0.79 * f_log - 1.64)**2)
            
            Nu_turb = (f / 8.0) * (Re_eff - 1000.0) * Pr / (
                1.0 + 12.7 * np.sqrt(f / 8.0) * (Pr**(2.0/3.0) - 1.0)
            )
            Nu_lam = 3.66
            w = (Re - 2300.0) / 700.0
            if w < 0.0: w = 0.0
            elif w > 1.0: w = 1.0
            
            if Re < 2300.0:
                Nu = Nu_lam
            elif Re < 3000.0:
                Nu = (1.0 - w) * Nu_lam + w * Nu_turb
            else:
                Nu = Nu_turb
            h_in[i] = Nu * k_g / Di

    logging.info("Numba acceleration is ENABLED (HAS_NUMBA=True).")

except Exception as _e:
    logging.warning("Numba not available (HAS_NUMBA=False); using pure-Python kernels. Reason: %s", _e)

    def diffuse_axial_CN(T, alpha, dt, dx, theta):
        if alpha <= 0:
            return T

        n = T.size
        lam = alpha * dt / (dx * dx)
        a = np.zeros(n)
        b = np.ones(n)
        c = np.zeros(n)

        for i in range(1, n - 1):
            a[i] = -theta * lam
            b[i] = 1 + 2 * theta * lam
            c[i] = -theta * lam

        b[0] = 1 + 2 * theta * lam
        c[0] = -2 * theta * lam
        a[-1] = -2 * theta * lam
        b[-1] = 1 + 2 * theta * lam

        RHS = T.copy()
        L_T = np.empty_like(T)
        L_T[1:-1] = T[2:] - 2 * T[1:-1] + T[:-2]
        L_T[0] = 2 * (T[1] - T[0])
        L_T[-1] = 2 * (T[-2] - T[-1])
        RHS += (1 - theta) * lam * L_T

        for i in range(1, n):
            m = a[i] / b[i - 1]
            b[i] -= m * c[i - 1]
            RHS[i] -= m * RHS[i - 1]

        Tnew = np.empty_like(T)
        Tnew[-1] = RHS[-1] / b[-1]
        for i in range(n - 2, -1, -1):
            Tnew[i] = (RHS[i] - c[i] * Tnew[i + 1]) / b[i]

        return Tnew

    def _cn_solve_with_cache(T, lam, theta, a, cprime, inv_denom):
        n = T.shape[0]
        RHS = np.empty(n, dtype=T.dtype)
        if n > 1:
            RHS[0] = T[0] + (1.0 - theta) * lam * (2.0 * (T[1] - T[0]))
            for i in range(1, n - 1):
                RHS[i] = T[i] + (1.0 - theta) * lam * (T[i + 1] - 2.0 * T[i] + T[i - 1])
            RHS[n - 1] = T[n - 1] + (1.0 - theta) * lam * (2.0 * (T[n - 2] - T[n - 1]))
        else:
            RHS[0] = T[0]

        y = np.empty(n, dtype=T.dtype)
        y[0] = RHS[0] * inv_denom[0]
        for i in range(1, n):
            y[i] = (RHS[i] - a[i] * y[i - 1]) * inv_denom[i]

        out = np.empty_like(T)
        out[n - 1] = y[n - 1]
        for i in range(n - 2, -1, -1):
            out[i] = y[i] - cprime[i] * out[i + 1]
        return out

    def _cn_solve_with_cache_noalloc(T, lam, theta, a, cprime, inv_denom, RHS, y, out):
        n = T.shape[0]
        if n > 1:
            RHS[0] = T[0] + (1.0 - theta) * lam * (2.0 * (T[1] - T[0]))
            for i in range(1, n - 1):
                RHS[i] = T[i] + (1.0 - theta) * lam * (T[i + 1] - 2.0 * T[i] + T[i - 1])
            RHS[n - 1] = T[n - 1] + (1.0 - theta) * lam * (2.0 * (T[n - 2] - T[n - 1]))
        else:
            RHS[0] = T[0]

        y[0] = RHS[0] * inv_denom[0]
        for i in range(1, n):
            y[i] = (RHS[i] - a[i] * y[i - 1]) * inv_denom[i]

        out[n - 1] = y[n - 1]
        for i in range(n - 2, -1, -1):
            out[i] = y[i] - cprime[i] * out[i + 1]
        return out

def _numba_sanity_check():
    """One-step CN check: ensure accelerated path matches reference."""
    global HAS_NUMBA  # <-- must be first statement in the function

    try:
        import numpy as _np
    except ImportError:
        return

    if not HAS_NUMBA:
        return

    # tiny test
    x = _np.linspace(0.0, 1.0, 32)
    dx = x[1] - x[0]
    alpha = 1e-5
    dt = 0.01
    theta = 0.5
    T0 = 300.0 + 10.0 * _np.sin(2.0 * _np.pi * x)

    # reference using non-numba CN (or py_func if you're wrapping the njit)
    T_ref = diffuse_axial_CN(T0.copy(), alpha, dt, dx, theta)

    # accelerated path (if different)
    T_acc = diffuse_axial_CN(T0.copy(), alpha, dt, dx, theta)

    if not _np.allclose(T_acc, T_ref, rtol=1e-4, atol=1e-5):
        logging.warning(
            "Numba CN kernel deviates from reference; disabling Numba."
        )
        HAS_NUMBA = False

try:
    from numba import njit
    @njit(cache=True, fastmath=True)
    def _interp1d_uniform_vec(x0, dx, y, xp, left, y_right):
        n = y.shape[0]
        x_end = x0 + dx*(n-1)
        out = np.empty(xp.shape[0], dtype=y.dtype)
        for k in range(xp.shape[0]):
            xv = xp[k]
            if xv <= x0:
                out[k] = left
            elif xv >= x_end:
                out[k] = y_right
            else:
                r = (xv - x0)/dx
                i = int(r)
                t = r - i
                out[k] = (1.0 - t)*y[i] + t*y[i+1]
        return out
except Exception as _e:
    # If Numba import or JIT fails here, fall back to a pure-NumPy version.
    def _interp1d_uniform_vec(x0, dx, y, xp, left, y_right):
        n = y.shape[0]
        x_end = x0 + dx*(n-1)
        out = np.empty(xp.shape[0], dtype=y.dtype)
        for k in range(xp.shape[0]):
            xv = xp[k]
            if xv <= x0:
                out[k] = left
            elif xv >= x_end:
                out[k] = y_right
            else:
                r = (xv - x0)/dx
                i = int(r)
                t = r - i
                out[k] = (1.0 - t)*y[i] + t*y[i+1]
        return out

# --- Helper to precompute and cache CN factors ---
def _build_cn_factors(n, lam, theta, dtype):
    a = np.zeros(n, dtype=dtype)
    b = np.ones(n, dtype=dtype)
    c = np.zeros(n, dtype=dtype)
    # interior
    for i in range(1, n - 1):
        a[i] = -theta * lam
        b[i] = 1.0 + 2.0 * theta * lam
        c[i] = -theta * lam
    # boundaries: Neumann-like end treatment consistent with original
    b[0] = 1.0 + 2.0 * theta * lam
    c[0] = -2.0 * theta * lam
    a[n - 1] = -2.0 * theta * lam
    b[n - 1] = 1.0 + 2.0 * theta * lam

    # Thomas factorization pieces independent of RHS: c' and inv_denom
    cprime = np.zeros(n, dtype=dtype)
    inv_denom = np.zeros(n, dtype=dtype)
    inv_denom[0] = 1.0 / b[0]
    cprime[0] = c[0] * inv_denom[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cprime[i - 1]
        inv_denom[i] = 1.0 / denom
        cprime[i] = c[i] * inv_denom[i]
    return a, cprime, inv_denom

# --- CN cache helper class ---
class CNCache:
    """Small helper that caches Crank–Nicolson tridiagonal factors vs. lambda.
    Rebuilds when |Δlam|/lam > CN_REL_TOL.
    """
    def __init__(self, n: int, theta: float, dtype, rel_tol: float = CN_REL_TOL):
        self.n = n
        self.theta = theta
        self.dtype = dtype
        self.rel_tol = rel_tol
        self.last_lam = None
        self.a = None
        self.cprime = None
        self.inv = None

    def ensure(self, lam: float):
        if self.last_lam is None or abs(lam - self.last_lam) > self.rel_tol * max(lam, EPS):
            self.a, self.cprime, self.inv = _build_cn_factors(self.n, lam, self.theta, self.dtype)
            self.last_lam = lam
        return self.a, self.cprime, self.inv


@njit(cache=True, fastmath=True, nogil=True)
def _timestep_numba_seq(Tg, Tw, Ti, dt,
                        x0, dx,
                        p, m_dot, Tin_now, Tin_next, R_g, Pr, mu_g, k_g,
                        Di, A_flow, P_in, P_out,
                        cp_g, Cw_cell_vec, Ci_cell_vec, R_Tw_to_Ti_vec,
                        h_out_manual, use_auto_hout, D_out, eps_rad, sigma,
                        theta_cond, dittus_n, use_semi_lag,
                        Tamb,
                        lam_w, lam_i,
                        a_w, cprime_w, inv_w,
                        a_i, cprime_i, inv_i,
                        xp_buf, Cg_buf, q_gw_buf,
                        Tg_out, Tw_e, Ti_e, Tw_out, Ti_out,
                        h_in, u,
                        RHS_w, Y_w, RHS_i, Y_i):
    n = Tg.shape[0]
    x_end = x0 + dx * (n - 1)

    # --- Gas advection + source (semi-Lagrangian or upwind), no allocations ---
    if use_semi_lag == 1:
        for i in range(n):
            x_i = x0 + i * dx
            x_dep = x_i - u[i] * dt
            xp_buf[i] = x_dep
            if x_dep <= x0:
                denom = u[i] * dt
                if denom <= 1.0e-12:
                    Tg_out[i] = Tin_now
                else:
                    frac = 1.0 - x_i / denom
                    if frac < 0.0:
                        frac = 0.0
                    elif frac > 1.0:
                        frac = 1.0
                    Tg_out[i] = Tin_now + frac * (Tin_next - Tin_now)
            elif x_dep >= x_end:
                Tg_out[i] = Tg[-1]
            else:
                r = (x_dep - x0) / dx
                j = int(r)
                if j >= n - 1:
                    j = n - 2
                tloc = r - j
                Tg_out[i] = (1.0 - tloc) * Tg[j] + tloc * Tg[j + 1]
        for i in range(n):
            # Local gas cell heat capacity (rho depends on current Tg)
            rho_i   = p / (R_g * Tg[i])
            Cg_buf[i]  = rho_i * A_flow * dx * cp_g
            q_gw_buf[i]= h_in[i] * P_in * dx * (Tw[i] - Tg_out[i])
            Tg_out[i]  = Tg_out[i] + dt * (q_gw_buf[i] / (Cg_buf[i] if Cg_buf[i] > 1e-12 else 1e-12))
    else:
        for i in range(n):
            rho_i = p / (R_g * Tg[i])
            Cg    = rho_i * A_flow * dx * cp_g
            Tg_up = Tin_now if i == 0 else Tg[i-1]
            adv   = -u[i] * (Tg[i] - Tg_up) / dx
            q_gw  = h_in[i] * P_in * dx * (Tw[i] - Tg[i])
            dTg_dt= adv + q_gw / (Cg if Cg > 1e-12 else 1e-12)
            Tg_out[i] = Tg[i] + dt * dTg_dt

    # --- Local 2x2 implicit (per-cell) for Tw, Ti (no allocations) ---
    for i in range(n):
        inv_R = 1.0 / (R_Tw_to_Ti_vec[i] if R_Tw_to_Ti_vec[i] > 1e-16 else 1e-16)
        h_out_i = h_out_manual
        if use_auto_hout == 1:
            h_out_i = _h_out_natural_conv_scalar(Ti[i], Tamb, D_out)
        k_rad = 4.0 * eps_rad * sigma * P_out * dx * (Ti[i] ** 3)
        a11 = Cw_cell_vec[i]/dt + h_in[i]*P_in*dx + inv_R
        a12 = -inv_R
        a21 = -inv_R
        a22 = Ci_cell_vec[i]/dt + inv_R + h_out_i*P_out*dx + k_rad
        rhs1 = (Cw_cell_vec[i]/dt)*Tw[i] + h_in[i]*P_in*dx*Tg_out[i]
        rhs2 = (Ci_cell_vec[i]/dt)*Ti[i] + h_out_i*P_out*dx*Tamb + eps_rad*sigma*P_out*dx*((Tamb**4) - (Ti[i]**4)) + k_rad*Ti[i]
        det = a11*a22 - a12*a21
        Tw_e[i] = (rhs1*a22 - a12*rhs2) / det
        Ti_e[i] = (a11*rhs2 - rhs1*a21) / det

    # --- Axial diffusion (CN) using cached factors and scratch buffers (no allocations) ---
    _cn_solve_with_cache_noalloc(Tw_e, lam_w, theta_cond, a_w, cprime_w, inv_w, RHS_w, Y_w, Tw_out)
    _cn_solve_with_cache_noalloc(Ti_e, lam_i, theta_cond, a_i, cprime_i, inv_i, RHS_i, Y_i, Ti_out)
    return  # all results written into Tg_out, Tw_out, Ti_out





@njit(cache=True, fastmath=True, parallel=True)
def _timestep_numba(Tg, Tw, Ti, dt,
                    x0, dx,
                    p, m_dot, Tin_now, Tin_next, R_g, Pr, mu_g, k_g,
                    Di, A_flow, P_in, P_out,
                    cp_g, Cw_cell_vec, Ci_cell_vec, R_Tw_to_Ti_vec,
                    h_out_manual, use_auto_hout, D_out, eps_rad, sigma,
                    theta_cond, _legacy_corr_param, use_semi_lag,
                    Tamb,
                    lam_w, lam_i,
                    a_w, cprime_w, inv_w,
                    a_i, cprime_i, inv_i):
    n = Tg.shape[0]
    x_end = x0 + dx * (n - 1)
    # Flow properties + Gnielinski internal convection
    rho = p/(R_g*Tg)
    u = m_dot/(rho*A_flow)
    h_in = np.empty_like(Tg)
    for i in prange(n):
        Re_i = rho[i]*u[i]*Di/mu_g
        if Re_i < 1.0:
            Re_i = 1.0
        if Re_i < 2300.0:
            Nu_i = 3.66
        else:
            Re_eff = Re_i
            if Re_eff < 3000.0:
                Re_eff = 3000.0
            f = (0.79*np.log(Re_eff) - 1.64)
            f = 1.0 / (f*f)
            Nu_turb = (f/8.0)*(Re_eff - 1000.0)*Pr / (1.0 + 12.7*np.sqrt(f/8.0)*(Pr**(2.0/3.0) - 1.0))
            if Re_i < 3000.0:
                w = (Re_i - 2300.0) / 700.0
                Nu_i = (1.0 - w)*3.66 + w*Nu_turb
            else:
                Nu_i = Nu_turb
        h_in[i] = Nu_i*k_g/Di

    # Gas advection + source
    if use_semi_lag == 1:
        Tg_adv = np.empty_like(Tg)
        for i in prange(n):
            x_i = x0 + i * dx
            x_dep = x_i - u[i] * dt
            if x_dep <= x0:
                denom = u[i] * dt
                if denom <= 1.0e-12:
                    Tg_adv[i] = Tin_now
                else:
                    frac = 1.0 - x_i / denom
                    if frac < 0.0:
                        frac = 0.0
                    elif frac > 1.0:
                        frac = 1.0
                    Tg_adv[i] = Tin_now + frac * (Tin_next - Tin_now)
            elif x_dep >= x_end:
                Tg_adv[i] = Tg[-1]
            else:
                r = (x_dep - x0) / dx
                j = int(r)
                if j >= n - 1:
                    j = n - 2
                tloc = r - j
                Tg_adv[i] = (1.0 - tloc) * Tg[j] + tloc * Tg[j + 1]
        Cg_cell = rho*A_flow*dx*cp_g
        q_gw = h_in*P_in*dx*(Tw - Tg_adv)
        Tg_new = Tg_adv + dt*(q_gw/np.maximum(Cg_cell, 1e-12))
    else:
        Tg_new = np.empty_like(Tg)
        for i in prange(n):
            Cg = rho[i]*A_flow*dx*cp_g
            if i == 0:
                Tg_up = Tin_now
            else:
                Tg_up = Tg[i-1]
            adv = -u[i]*(Tg[i] - Tg_up)/dx
            q_gw = h_in[i]*P_in*dx*(Tw[i] - Tg[i])
            dTg_dt = adv + q_gw/np.maximum(Cg, 1e-12)
            Tg_new[i] = Tg[i] + dt*dTg_dt

    # Semi-implicit local 2x2 for (Tw, Ti) with linearized radiation
    Tw_e = np.empty_like(Tw)
    Ti_e = np.empty_like(Ti)
    for i in prange(n):
        inv_R = 1.0/np.maximum(R_Tw_to_Ti_vec[i], 1e-16)
        h_out_i = h_out_manual
        if use_auto_hout == 1:
            h_out_i = _h_out_natural_conv_scalar(Ti[i], Tamb, D_out)
        # coefficients
        k_rad = 4.0 * eps_rad * sigma * P_out * dx * (Ti[i]**3)
        a11 = Cw_cell_vec[i]/dt + h_in[i]*P_in*dx + inv_R
        a12 = -inv_R
        a21 = -inv_R
        a22 = Ci_cell_vec[i]/dt + inv_R + h_out_i*P_out*dx + k_rad
        rhs1 = (Cw_cell_vec[i]/dt)*Tw[i] + h_in[i]*P_in*dx*Tg_new[i]
        rhs2 = (Ci_cell_vec[i]/dt)*Ti[i] + h_out_i*P_out*dx*Tamb + eps_rad*sigma*P_out*dx*((Tamb**4) - (Ti[i]**4)) + k_rad*Ti[i]
        det = a11*a22 - a12*a21
        Tw_e[i] = (rhs1*a22 - a12*rhs2)/det
        Ti_e[i] = (a11*rhs2 - rhs1*a21)/det

    # Axial diffusion (CN) for wall and insulation using cached factors
    Tw_new = _cn_solve_with_cache(Tw_e, lam_w, theta_cond, a_w, cprime_w, inv_w)
    Ti_new = _cn_solve_with_cache(Ti_e, lam_i, theta_cond, a_i, cprime_i, inv_i)
    return Tg_new, Tw_new, Ti_new


# --- Semi-Lagrangian advection and snapshot scheduling ---
def advect_semi_lagrangian(T_old, u, dt, x, Tin_now, Tin_next):
    """Backward-characteristic semi-Lagrangian advection for positive u.
    For x_dep < x0, use inlet value at boundary crossing time within the current step.
    """
    x_depart = x - u*dt
    T_adv = np.interp(x_depart, x, T_old, left=Tin_now, right=T_old[-1])
    mask = x_depart < x[0]
    if np.any(mask):
        denom = np.maximum(u[mask] * dt, 1.0e-12)
        frac = 1.0 - x[mask] / denom
        frac = np.clip(frac, 0.0, 1.0)
        T_adv[mask] = Tin_now + frac * (Tin_next - Tin_now)
    return T_adv.astype(DTYPE, copy=False)

class SnapshotScheduler:
    def __init__(self, t_end, nframes):
        self.t_end = float(t_end)
        self.n = int(max(2, nframes))
        self.dt_save = self.t_end/(self.n-1)
        self.next_t = 0.0
    def should_save(self, t):
        return t >= self.next_t - 1e-12
    def mark_saved(self):
        self.next_t += self.dt_save


# --- Human-readable time formatting for logs ---
def _fmt_hms(sec):
    s = int(sec)
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# --- RuntimeTracker for progress and trace ---
class RuntimeTracker:
    def __init__(
        self,
        t_end,
        interval_s=5.0,
        interval_steps=1000,
        progress_mode="none",
        write_trace_csv=False,
    ):
        self.t_end = float(t_end)
        self.interval_s = float(interval_s)
        self.interval_steps = int(max(1, interval_steps))
        self.progress_mode = str(progress_mode)
        self.write_trace_csv = bool(write_trace_csv)
        self.t0_wall = None
        self.last_log_wall = None
        self.records = []  # (wall_s, sim_s, steps, mean_dt, speed_x, eta_s)
        self.started = False
        self.total_steps = 0

    def start(self):
        self.t0_wall = time.perf_counter()
        self.last_log_wall = self.t0_wall
        self.started = True

    def _metrics(self, sim_t, steps):
        wall = time.perf_counter() - self.t0_wall
        mean_dt = (sim_t / steps) if steps > 0 else 0.0
        speed = (sim_t / wall) if wall > 0 else 0.0  # simulated seconds per wall second
        eta = (self.t_end - sim_t) / speed if speed > 1e-12 else float("inf")
        return wall, mean_dt, speed, eta

    def log_if_needed(self, sim_t, steps):
        if not self.started:
            self.start()
        if self.progress_mode == "none":
            return
        now = time.perf_counter()
        need = (now - self.last_log_wall) >= self.interval_s or (steps % self.interval_steps == 0)
        if need:
            wall, mean_dt, speed, eta = self._metrics(sim_t, steps)
            self.records.append((wall, sim_t, steps, mean_dt, speed, eta))
            logging.info(
                "Progress: steps=%d, sim_t=%.2fs (%s), wall=%.2fs (%s), "
                "mean_dt=%.4fs, speed=%.2fx, ETA=%s",
                steps,
                sim_t,
                _fmt_hms(sim_t),
                wall,
                _fmt_hms(wall),
                mean_dt,
                speed,
                f"{eta:.1f}s ({_fmt_hms(eta)})" if eta != float("inf") else "--"
            )
            self.last_log_wall = now

    def finalize(self, sim_t, steps, outdir: Path | None):
        wall, mean_dt, speed, eta = self._metrics(sim_t, steps)
        logging.info(
            "Runtime summary: steps=%d, sim_t=%.2fs (%s), wall=%.2fs (%s), mean_dt=%.4fs, sim_speed=%.2fx real-time",
            steps, sim_t, _fmt_hms(sim_t), wall, _fmt_hms(wall), mean_dt, speed
        )
        # Persist trace CSV
        if outdir is not None and self.write_trace_csv and self.records:
            import csv
            csv_path = outdir / "runtime_trace.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["wall_s", "sim_s", "steps", "mean_dt_s", "speed_x", "eta_s"])
                for r in self.records:
                    w.writerow(list(r))
            logging.info("Saved runtime_trace.csv with %d rows", len(self.records))

        logging.info("Total wall-clock runtime: %.3f seconds (%.2f minutes)", wall, wall / 60)
        return wall  # allow caller to use wall time

# --- Helper plotting and saving utilities (modularity; no behavior change) ---
def _import_pyplot():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        logging.warning("Matplotlib unavailable; skipping plot generation. Reason: %s", exc)
        return None


def plot_heatmaps(x, times, Tw_hist, Tg_hist, Ti_hist, OUTDIR, params):
    plt = _import_pyplot()
    if plt is None:
        return

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Heatmaps — m_dot={params['m_dot']} kg/s, Tin={params['Tin']} K")
    im0 = axs[0].imshow(Tw_hist, aspect='auto', extent=[x[0], x[-1], times[-1], times[0]])
    axs[0].set_title('Wall Tw(x,t) [K]')
    axs[0].set_xlabel('x [m]')
    axs[0].set_ylabel('time [s]')
    plt.colorbar(im0, ax=axs[0], label='K')

    im1 = axs[1].imshow(Tg_hist, aspect='auto', extent=[x[0], x[-1], times[-1], times[0]])
    axs[1].set_title('Gas Tg(x,t) [K]')
    axs[1].set_xlabel('x [m]')
    axs[1].set_ylabel('time [s]')
    plt.colorbar(im1, ax=axs[1], label='K')

    im2 = axs[2].imshow(Ti_hist, aspect='auto', extent=[x[0], x[-1], times[-1], times[0]])
    axs[2].set_title('Insulation Ti(x,t) [K]')
    axs[2].set_xlabel('x [m]')
    axs[2].set_ylabel('time [s]')
    plt.colorbar(im2, ax=axs[2], label='K')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTDIR / "heatmaps.png", dpi=200)
    if params.get("show_plots", False):
        plt.show()
    plt.close(fig)
    logging.info("Saved heatmaps.png")


def plot_profiles(x, times, Tw_hist, Tg_hist, Ti_hist, OUTDIR, params):
    plt = _import_pyplot()
    if plt is None:
        return

    fig = plt.figure(figsize=(10, 4))
    # Select at most 30 evenly spaced time indices
    nmax = 30
    idx = np.linspace(0, max(0, times.size - 1), min(nmax, max(1, times.size)), dtype=int)

    for i in idx:
        plt.plot(x, Tw_hist[i], label=f"Tw {times[i]:.0f}s")
    for i in idx:
        plt.plot(x, Tg_hist[i], '--', label=f"Tg {times[i]:.0f}s")
    for i in idx:
        plt.plot(x, Ti_hist[i], ':', label=f"Ti {times[i]:.0f}s")

    plt.xlabel('x [m]')
    plt.ylabel('Temperature [K]')
    plt.title(f"Profiles over time — m_dot={params['m_dot']} kg/s, Tin={params['Tin']} K")
    from matplotlib.lines import Line2D
    nlabels = len(idx) * 3  # Tw/Tg/Ti per time
    if nlabels <= 12:
        plt.legend(ncol=3, fontsize=7)
    elif nlabels <= 30:
        plt.legend(ncol=1, fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    else:
        proxy = [
            Line2D([0], [0], linestyle='-', linewidth=1.5, label='Tw'),
            Line2D([0], [0], linestyle='--', linewidth=1.5, label='Tg'),
            Line2D([0], [0], linestyle=':', linewidth=1.5, label='Ti'),
        ]
        plt.legend(handles=proxy, ncol=1, fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, title="Line styles")
    plt.tight_layout(rect=[0, 0, 0.88, 0.93])
    plt.savefig(OUTDIR / "profiles.png", dpi=200)
    if params.get("show_plots", False):
        plt.show()
    plt.close(fig)
    logging.info("Saved profiles.png")


def save_arrays_and_csv(OUTDIR, x, times, Tw_hist, Tg_hist, Ti_hist, Nx):
    # Persist arrays (npz)
    np.savez_compressed(OUTDIR / "fields.npz", x=x, times=times, Tw=Tw_hist, Tg=Tg_hist, Ti=Ti_hist)
    logging.info("Saved fields.npz")

    # Summary CSV: outlet/inlet/midpoint temperatures over time
    mid_idx = Nx // 2
    summary = np.column_stack([
        times,
        Tg_hist[:, -1],
        Tw_hist[:, 0], Tw_hist[:, mid_idx], Tw_hist[:, -1],
        Ti_hist[:, 0], Ti_hist[:, mid_idx], Ti_hist[:, -1],
    ]).astype(np.float64, copy=False)
    header = "time_s,Tg_outlet_K,Tw_inlet_K,Tw_mid_K,Tw_outlet_K,Ti_inlet_K,Ti_mid_K,Ti_outlet_K"
    np.savetxt(OUTDIR / "summary.csv", summary, delimiter=",", header=header, comments="")
    logging.info("Saved summary.csv")

    # Final prints for quick inspection (unchanged behavior)
    logging.info("Final outlet gas T [K]: %.2f", float(Tg_hist[-1, -1]))
    logging.info(
        "Final Tw inlet/mid/outlet [K]: %.2f, %.2f, %.2f",
        float(Tw_hist[-1, 0]), float(Tw_hist[-1, Nx // 2]), float(Tw_hist[-1, -1])
    )
    logging.info(
        "Final Ti inlet/mid/outlet [K]: %.2f, %.2f, %.2f",
        float(Ti_hist[-1, 0]), float(Ti_hist[-1, Nx // 2]), float(Ti_hist[-1, -1])
    )
    logging.info("Outputs saved to: %s", OUTDIR)


# === Functional entry point for reuse ===

@dataclass
class SimRunResult:
    outdir: Path | None
    x: np.ndarray
    times: np.ndarray
    Tw_hist: np.ndarray
    Tg_hist: np.ndarray
    Ti_hist: np.ndarray
    wall_time_s: float
    n_steps: int
    reached_Tg_target: bool
    Tg_outlet_final: float
    Tg_outlet_target: float | None
    target_metric: str
    target_outlet_final: float
    stop_reason: str


def main(
        params_override: Optional[Dict[str, Any]] = None,
        *,
        make_plots: bool = True,
        save_results: bool = True,
        outdir: Optional[Path] = None,
        stop_at_Tg_outlet: float | None = None,
        stop_dir: str | None = None,  # "le" or "ge"; if None, inferred
        max_sim_time: float | None = None,
        snapshot_callback: Optional[Callable[[float, np.ndarray, np.ndarray, np.ndarray], None]] = None,
        abort_callback: Optional[Callable[[], bool]] = None,
    ) -> SimRunResult:
    """Run one simulation. Optionally override entries in the global `params` dict.
    Returns a SimRunResult with arrays and file outputs location.
    """
    # --- merge params ---
    p = dict(params)
    if params_override:
        p.update(params_override)

    validate_params(p)
    use_numba = bool(HAS_NUMBA and p.get("enable_numba", True))

    # --- Output directory and logging setup ---
    logger = logging.getLogger()

    # If no handlers yet (CLI / bare run), create a default console handler
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )

    OUTDIR = None
    run_file_handler = None
    if save_results:
        OUTDIR = outdir if outdir is not None else _make_run_dir(prefix="runs", max_keep=int(p.get("max_run_dirs", 1000)))
        logger.info("Run directory: %s", OUTDIR)

        if p.get("log_to_file", False):
            for h in list(logger.handlers):
                if getattr(h, "_thermal_pipe_file_handler", False):
                    logger.removeHandler(h)
                    try:
                        h.close()
                    except Exception:
                        pass
            # Add a file handler in addition to existing handlers (GUI, console, etc.)
            run_file_handler = logging.FileHandler(OUTDIR / "run.log", mode="w")
            run_file_handler._thermal_pipe_file_handler = True
            run_file_handler.setLevel(logging.INFO)
            run_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logger.addHandler(run_file_handler)

        # Save parameters snapshot at start
        with open(OUTDIR / "params.json", "w") as _f:
            json.dump(p, _f, indent=2)
        logger.info("Saved params.json")
    if HAS_NUMBA and not use_numba:
        logger.warning(
            "Numba kernels disabled for this run (enable_numba=False). "
            "Windows defaults to this mode for stability."
        )

    # --- local constants (compute from params for reusability) ---
    DTYPE = np.float32 if p.get("use_float32", False) else np.float64
    def _float(x):
        return (np.float32 if DTYPE is np.float32 else np.float64)(x)

    R = _float(287.058)
    sigma = _float(5.670374419e-8)
    L, Di = _float(p["L"]), _float(p["Di"])
    r_i = _float(Di/2)
    r_w_o = _float(r_i + p["t_wall"])
    r_ins_o = _float(r_w_o + p["t_ins"])
    D_out = _float(2.0 * r_ins_o)
    A_flow = _float(np.pi*(Di**2)/4)
    P_in = _float(np.pi*Di)
    P_out = _float(np.pi*2*r_ins_o)
    Nx = int(p["Nx"])
    x = np.linspace(0, L, Nx, dtype=DTYPE)
    dx = _float(x[1] - x[0])

    Vw_cell = _float(np.pi*(r_w_o**2 - r_i**2)*dx)
    Vi_cell = _float(np.pi*(r_ins_o**2 - r_w_o**2)*dx)
    Cw_cell_base = _float(p["rho_w"] * Vw_cell * p["cp_w"])
    Ci_cell_base = _float(p["rho_i"] * Vi_cell * p["cp_i"])
    aw_base = _float(p["k_w"] / (p["rho_w"] * p["cp_w"]))
    ai_base = _float(p["k_i"] / (p["rho_i"] * p["cp_i"]))
    R_wall_base = _float(np.log(r_w_o / r_i) / (2 * np.pi * p["k_w"] * dx))
    R_ins_base = _float(np.log(r_ins_o / r_w_o) / (2 * np.pi * p["k_i"] * dx)) if p["t_ins"] > 1.0e-12 else _float(0.0)
    R_Tw_to_Ti_base = _float(R_wall_base + R_ins_base)
    t_end_horizon = float(p["t_end"]) if max_sim_time is None else float(max_sim_time)

    ins_mass_mode = str(p.get("insulation_mass_mode", "penetration")).strip().lower()
    insulation_mass_scale = 1.0
    if p["t_ins"] > 1.0e-12 and ins_mass_mode == "penetration":
        t_pen = float(p.get("insulation_penetration_time_s", 0.0))
        if t_pen <= 0.0:
            t_pen = t_end_horizon
        alpha_i_ref = float(p["k_i"]) / max(float(p["rho_i"] * p["cp_i"]), 1.0e-12)
        penetration_frac = np.sqrt(max(alpha_i_ref * t_pen, 0.0)) / max(float(p["t_ins"]), 1.0e-12)
        insulation_mass_scale = min(
            1.0,
            max(float(p.get("insulation_mass_min_frac", 0.25)), float(penetration_frac)),
        )

    thermal_mass_profile = _build_thermal_mass_profile(
        nx=Nx,
        count=int(p.get("thermal_mass_count", 0)),
        factor=float(p.get("thermal_mass_factor", 0.0)),
        positions_frac=p.get("thermal_mass_positions_frac", []),
        spread_frac=float(p.get("thermal_mass_spread_frac", 0.03)),
    ).astype(DTYPE, copy=False)

    use_temp_dep = bool(p.get("use_temp_dependent_props", False))
    semi_lag_courant_max = float(max(1.0e-6, p.get("semi_lag_courant_max", 2.0)))
    solid_rebuild_temp_k = float(max(0.0, p.get("prop_update_temp_threshold_k", 2.0)))
    pipe_table = _prepare_prop_table(p.get("pipe_prop_table"), cp_default=float(p["cp_w"]), k_default=float(p["k_w"]))
    ins_table = _prepare_prop_table(p.get("ins_prop_table"), cp_default=float(p["cp_i"]), k_default=float(p["k_i"]))
    if not use_temp_dep:
        pipe_table = None
        ins_table = None

    _numba_sanity_check()

    def compute_h_in_local(Tg_vec):
        rho = p["p"]/(R*Tg_vec)
        u = p["m_dot"]/(rho*A_flow)
        Re = rho*u*Di/p["mu_g"]
        Re = np.maximum(Re, 1.0)
        Re_eff = np.maximum(Re, 3000.0)
        f = (0.79*np.log(Re_eff) - 1.64)**(-2.0)
        Nu_turb = (f/8.0)*(Re_eff - 1000.0)*p["Pr"] / (
            1.0 + 12.7*np.sqrt(f/8.0)*(p["Pr"]**(2.0/3.0) - 1.0)
        )
        Nu_lam = np.full_like(Re, 3.66)
        w = np.clip((Re - 2300.0)/700.0, 0.0, 1.0)
        Nu = np.where(Re < 2300.0, Nu_lam, np.where(Re < 3000.0, (1.0-w)*Nu_lam + w*Nu_turb, Nu_turb))
        return Nu*p["k_g"]/Di, u

    def build_solid_state(Tw_vec, Ti_vec):
        tw_f = np.asarray(Tw_vec, dtype=float)
        ti_f = np.asarray(Ti_vec, dtype=float)

        cp_w_vec, k_w_vec = _interp_props(tw_f, pipe_table, cp_default=float(p["cp_w"]), k_default=float(p["k_w"]))
        cp_i_vec, k_i_vec = _interp_props(ti_f, ins_table, cp_default=float(p["cp_i"]), k_default=float(p["k_i"]))

        Cw_vec = (p["rho_w"] * Vw_cell * cp_w_vec * thermal_mass_profile).astype(DTYPE, copy=False)
        Ci_vec = (p["rho_i"] * Vi_cell * cp_i_vec * insulation_mass_scale).astype(DTYPE, copy=False)

        if p["t_ins"] > 1.0e-12:
            R_ins_vec = np.log(np.maximum(r_ins_o / r_w_o, 1.0 + 1.0e-12)) / (2.0 * np.pi * np.maximum(k_i_vec, 1.0e-12) * dx)
        else:
            R_ins_vec = np.zeros_like(cp_i_vec, dtype=float)
        R_wall_vec = np.log(np.maximum(r_w_o / r_i, 1.0 + 1.0e-12)) / (2.0 * np.pi * np.maximum(k_w_vec, 1.0e-12) * dx)
        R_tot_vec = np.maximum(R_wall_vec + R_ins_vec, 1.0e-16).astype(DTYPE, copy=False)
        R_wall_vec = np.maximum(R_wall_vec, 1.0e-16).astype(DTYPE, copy=False)

        aw_eff = float(np.mean(k_w_vec / np.maximum(p["rho_w"] * cp_w_vec * thermal_mass_profile, 1.0e-12)))
        ai_eff = float(np.mean(k_i_vec / np.maximum(p["rho_i"] * cp_i_vec, 1.0e-12)))
        return Cw_vec, Ci_vec, R_tot_vec, R_wall_vec, aw_eff, ai_eff, cp_w_vec, cp_i_vec

    # --- diagnostics: simple energy accounting (optional) ---
    def _energy_state(Tg, Tw, Ti, cp_w_local, cp_i_local):
        # Gas: ideal-gas rho from p = rho R T
        rho_g = p["p"] / (R * Tg)
        vol = A_flow * dx
        E_g = float(np.sum(rho_g * p["cp_g"] * Tg * vol))

        # Wall: cylindrical shell
        vol_w = np.pi * (r_w_o**2 - r_i**2) * dx
        E_w = float(np.sum(p["rho_w"] * cp_w_local * thermal_mass_profile * Tw * vol_w))

        # Insulation
        vol_i = np.pi * (r_ins_o**2 - r_w_o**2) * dx
        E_i = float(np.sum(p["rho_i"] * cp_i_local * Ti * vol_i))

        return E_g + E_w + E_i

    # --- initial conditions ---
    Tg_init = float(p.get("T_init_gas", p["T_init_wall"]))
    Tin_target = float(p["Tin"])
    Tin_ramp_s = float(max(0.0, p.get("Tin_ramp_s", 0.0)))
    Tin_ramp_model = str(p.get("Tin_ramp_model", "logistic")).strip().lower()
    if Tin_ramp_model not in ("heater_exp", "linear", "logistic"):
        Tin_ramp_model = "logistic"
    Tin_ramp_shape = float(max(1.0e-6, p.get("Tin_ramp_shape", 8.0)))
    Tin_start = float(Tg_init)
    Tg = np.full(Nx, Tg_init, dtype=DTYPE)
    Tw = np.full(Nx, p["T_init_wall"], dtype=DTYPE)
    Ti = np.full(Nx, p["T_init_ins"], dtype=DTYPE)
    Cw_cell_vec, Ci_cell_vec, R_Tw_to_Ti_vec, R_wall_vec, aw_eff, ai_eff, cp_w_local, cp_i_local = build_solid_state(Tw, Ti)
    static_solids = not use_temp_dep
    Tw_props_ref = Tw.copy()
    Ti_props_ref = Ti.copy()
    h0, u0 = compute_h_in_local(Tg)
    u0_max = float(np.max(u0))
    if p["adv_scheme"] == "semi_lagrangian":
        dt_adv0 = semi_lag_courant_max * dx / max(u0_max, 1.0e-6)
    else:
        dt_adv0 = p["CFL"] * dx / max(u0_max, 1.0e-6)

    energy_interval = int(p.get("energy_diag_interval_steps", 0))
    E0 = _energy_state(Tg, Tw, Ti, cp_w_local, cp_i_local) if energy_interval > 0 else None

    # --- Optional early-stop on selected outlet metric ---
    reached_target = False
    stop_reason = "t_end"
    Tg_outlet_target = stop_at_Tg_outlet
    target_metric = str(p.get("target_metric", "gas_outlet"))
    if target_metric not in ("gas_outlet", "wall_inner_outlet", "wall_outer_outlet", "insulation_outlet"):
        target_metric = "gas_outlet"

    def metric_outlet(Tg_vec, Tw_vec, Ti_vec, Rw_vec, Rtot_vec):
        if target_metric == "gas_outlet":
            return float(Tg_vec[-1])
        if target_metric == "wall_inner_outlet":
            return float(Tw_vec[-1])
        if target_metric == "insulation_outlet":
            return float(Ti_vec[-1])
        frac = float(Rw_vec[-1] / max(Rtot_vec[-1], 1.0e-16))
        return float(Tw_vec[-1] - frac * (Tw_vec[-1] - Ti_vec[-1]))

    initial_target_outlet = metric_outlet(Tg, Tw, Ti, R_wall_vec, R_Tw_to_Ti_vec)
    # Determine comparison direction if user didn't specify
    if Tg_outlet_target is not None:
        if stop_dir in ("le", "ge"):
            _cmp_dir = stop_dir
        else:
            _cmp_dir = "le" if Tg_outlet_target <= initial_target_outlet else "ge"
    else:
        _cmp_dir = None
    asymptote_check_enabled = bool(p.get("target_asymptote_check", True)) and (Tg_outlet_target is not None)
    asymptote_window_s = float(max(1.0, p.get("target_asymptote_window_s", 600.0)))
    asymptote_rate_tol = float(max(0.0, p.get("target_asymptote_rate_tol_k_per_s", 2.5e-4)))
    asymptote_min_gap = float(max(0.0, p.get("target_asymptote_min_gap_k", 1.0)))
    asymptote_min_time = float(max(0.0, p.get("target_asymptote_min_time_s", 900.0)))
    asymptote_proj_factor = float(max(1.0e-6, p.get("target_asymptote_projection_factor", 1.25)))
    asymptote_need_windows = int(max(1, p.get("target_asymptote_stall_windows", 3)))
    asymptote_hist_t: deque[float] = deque()
    asymptote_hist_m: deque[float] = deque()
    asymptote_stall_windows = 0
    asymptote_stop = False

    # Cache for flow properties
    _h_in = h0.copy(); _u = u0.copy()
    _update_props_every = int(max(1, p.get("update_props_every", 1)))
    _last_props_step = -1
    _solid_force_steps = int(max(1, p.get("prop_update_force_steps", 5 * _update_props_every)))
    _last_solid_step = 0
    use_auto_hout = 1 if p.get("h_out_mode", "auto") == "auto" else 0
    h_out_manual = float(p.get("h_out", 0.0))

    h0_area = h0 * P_in * dx
    inv_r0 = 1.0 / np.maximum(R_Tw_to_Ti_vec, 1.0e-16)
    tau_g = ((p["p"]/(R*p["Tin"]))*A_flow*dx*p["cp_g"]) / (np.max(h0_area)+1e-9)
    tau_wi = float(np.min(Cw_cell_vec / (h0_area + inv_r0 + 1.0e-12)))
    if use_auto_hout == 1:
        h_out0 = _h_out_natural_conv_vec(Ti, p["Tamb"], float(D_out))
    else:
        h_out0 = h_out_manual
    h_out0_term = h_out0 * P_out * dx
    rad0_term = 4.0 * p["eps_rad"] * sigma * P_out * dx * np.maximum(Ti, 1.0) ** 3
    tau_i0 = float(np.min(Ci_cell_vec / (inv_r0 + h_out0_term + rad0_term + 1.0e-12)))
    dt_src = 0.3*min(tau_g, tau_wi, tau_i0, 1e9)
    dt = float(min(dt_adv0, dt_src, 0.25))
    if ins_mass_mode == "penetration" and p["t_ins"] > 1.0e-12:
        logging.info(
            "Insulation effective mass scale=%.3f (penetration model)",
            insulation_mass_scale,
        )

    # --- Setup runtime tracker ---
    t = 0.0
    t_end = t_end_horizon
    tracker = RuntimeTracker(
        t_end,
        p.get("log_interval_s", 5.0),
        p.get("log_interval_steps", 1000),
        progress_mode=p.get("progress", "none"),
        write_trace_csv=p.get("write_trace_csv", False),
    )
    tracker.start()
    saver = SnapshotScheduler(t_end, p["save_frames"])

    # Preallocate snapshots
    _nframes = int(max(2, p["save_frames"]))
    Tw_hist = np.empty((_nframes, Nx), dtype=DTYPE)
    Tg_hist = np.empty((_nframes, Nx), dtype=DTYPE)
    Ti_hist = np.empty((_nframes, Nx), dtype=DTYPE)
    times   = np.empty((_nframes,),   dtype=DTYPE)
    _frame_idx = 0
    n = 0
    _snapshot_cb_errors = 0

    # Work/output buffers for the sequential kernel (no per-step allocations)
    xp_buf    = np.empty(Nx, dtype=DTYPE)
    Cg_buf    = np.empty(Nx, dtype=DTYPE)
    q_gw_buf  = np.empty(Nx, dtype=DTYPE)

    # Gas + wall/insulation outputs
    Tg_new_b  = np.empty(Nx, dtype=DTYPE)
    Tw_e_b    = np.empty(Nx, dtype=DTYPE)
    Ti_e_b    = np.empty(Nx, dtype=DTYPE)
    Tw_new_b  = np.empty(Nx, dtype=DTYPE)
    Ti_new_b  = np.empty(Nx, dtype=DTYPE)

    # Scratch for CN solves
    RHS_w = np.empty(Nx, dtype=DTYPE)
    Y_w   = np.empty(Nx, dtype=DTYPE)
    RHS_i = np.empty(Nx, dtype=DTYPE)
    Y_i   = np.empty(Nx, dtype=DTYPE)

    # CN caches for wall and insulation
    cn_w = CNCache(Nx, p["theta_cond"], DTYPE, rel_tol=CN_REL_TOL)
    cn_i = CNCache(Nx, p["theta_cond"], DTYPE, rel_tol=CN_REL_TOL)

    def inlet_temp_at(sim_t: float) -> float:
        if Tin_ramp_s <= 1.0e-12:
            return Tin_target
        if Tin_ramp_model == "linear":
            frac = min(1.0, max(0.0, sim_t / Tin_ramp_s))
        elif Tin_ramp_model == "logistic":
            u = min(1.0, max(0.0, sim_t / Tin_ramp_s))
            lo = 1.0 / (1.0 + np.exp(0.5 * Tin_ramp_shape))
            hi = 1.0 / (1.0 + np.exp(-0.5 * Tin_ramp_shape))
            sig = 1.0 / (1.0 + np.exp(-Tin_ramp_shape * (u - 0.5)))
            frac = (sig - lo) / max(hi - lo, 1.0e-12)
            frac = min(1.0, max(0.0, float(frac)))
        else:
            # First-order heater warm-up model: 99% of target delta reached at t = Tin_ramp_s.
            frac = 1.0 - np.exp(-np.log(100.0) * max(0.0, sim_t) / Tin_ramp_s)
            frac = min(1.0, max(0.0, float(frac)))
        return Tin_start + (Tin_target - Tin_start) * frac

    def _python_timestep_fallback(Tg, Tw, Ti, dt, Tin_eff, Tin_eff_next, h_in, u,
                                  lam_w, lam_i, _a_w, _cprime_w, _inv_w, _a_i, _cprime_i, _inv_i):
        if p["adv_scheme"] == "semi_lagrangian":
            Tg_adv = advect_semi_lagrangian(Tg, u, dt, x, Tin_eff, Tin_eff_next)
            q_gw = h_in * P_in * dx * (Tw - Tg_adv)
            Cg_cell = (p["p"] / (R * Tg)) * A_flow * dx * p["cp_g"]
            Tg_new = Tg_adv + dt * (q_gw / np.maximum(Cg_cell, 1e-12))
        else:
            Tg_up = np.roll(Tg, 1)
            Tg_up[0] = Tin_eff
            adv = -u * (Tg - Tg_up) / dx
            q_gw = h_in * P_in * dx * (Tw - Tg)
            Cg_cell = (p["p"] / (R * Tg)) * A_flow * dx * p["cp_g"]
            dTg_dt = adv + q_gw / np.maximum(Cg_cell, 1e-12)
            Tg_new = Tg + dt * dTg_dt

        inv_R = 1.0 / np.maximum(R_Tw_to_Ti_vec, 1.0e-16)
        if use_auto_hout == 1:
            h_out_vec = _h_out_natural_conv_vec(Ti, p["Tamb"], float(D_out))
        else:
            h_out_vec = np.full_like(Ti, h_out_manual)
        Ti3 = np.maximum(Ti, 1.0) ** 3
        k_rad_vec = 4.0 * p["eps_rad"] * sigma * P_out * dx * Ti3
        a11 = Cw_cell_vec / dt + h_in * P_in * dx + inv_R
        a22 = Ci_cell_vec / dt + inv_R + h_out_vec * P_out * dx + k_rad_vec
        rhs1 = (Cw_cell_vec / dt) * Tw + h_in * P_in * dx * Tg_new
        rhs2 = (
            (Ci_cell_vec / dt) * Ti
            + h_out_vec * P_out * dx * p["Tamb"]
            + p["eps_rad"] * sigma * P_out * dx * ((p["Tamb"] ** 4) - (Ti ** 4))
            + k_rad_vec * Ti
        )
        det = a11 * a22 - inv_R * inv_R
        det_safe = np.where(np.abs(det) > 1.0e-16, det, np.where(det >= 0.0, 1.0e-16, -1.0e-16))
        Tw_new = (rhs1 * a22 + inv_R * rhs2) / det_safe
        Ti_new = (a11 * rhs2 + inv_R * rhs1) / det_safe

        if _a_w is not None and _a_i is not None:
            Tw_new = _cn_solve_with_cache(Tw_new, lam_w, p["theta_cond"], _a_w, _cprime_w, _inv_w)
            Ti_new = _cn_solve_with_cache(Ti_new, lam_i, p["theta_cond"], _a_i, _cprime_i, _inv_i)
        else:
            Tw_new = diffuse_axial_CN(Tw_new, aw_eff, dt, dx, p["theta_cond"])
            Ti_new = diffuse_axial_CN(Ti_new, ai_eff, dt, dx, p["theta_cond"])
        return Tg_new, Tw_new, Ti_new

    # Pre-cast constants outside the timestep loop
    p_f = _float(p["p"])
    m_dot_f = _float(p["m_dot"])
    R_f = _float(R)
    Pr_f = _float(p["Pr"])
    mu_g_f = _float(p["mu_g"])
    k_g_f = _float(p["k_g"])
    Di_f = _float(Di)
    A_flow_f = _float(A_flow)
    P_in_f = _float(P_in)
    P_out_f = _float(P_out)
    cp_g_f = _float(p["cp_g"])
    h_out_manual_f = _float(h_out_manual)
    D_out_f = _float(D_out)
    eps_rad_f = _float(p["eps_rad"])
    sigma_f = _float(sigma)
    theta_cond_f = _float(p["theta_cond"])
    dittus_boelter_n_f = _float(p.get("dittus_boelter_n", 0.35))
    Tamb_f = _float(p["Tamb"])
    x0_f = _float(x[0])
    dx_f = _float(dx)
    use_semi = 1 if p["adv_scheme"] == "semi_lagrangian" else 0

    while t < t_end - 1e-12:
        if abort_callback is not None:
            try:
                if bool(abort_callback()):
                    stop_reason = "aborted_by_user"
                    reached_target = False
                    break
            except Exception as abort_exc:
                logging.warning("abort_callback failed: %s", abort_exc)
        Tin_eff = inlet_temp_at(t)

        if saver.should_save(t) and _frame_idx < _nframes:
            Tw_hist[_frame_idx, :] = Tw
            Tg_hist[_frame_idx, :] = Tg
            Ti_hist[_frame_idx, :] = Ti
            times[_frame_idx] = t
            _frame_idx += 1
            saver.mark_saved()
            if snapshot_callback is not None:
                try:
                    snapshot_callback(float(t), Tg.copy(), Tw.copy(), Ti.copy())
                except Exception as cb_exc:
                    _snapshot_cb_errors += 1
                    if _snapshot_cb_errors <= 3 or (_snapshot_cb_errors % 50 == 0):
                        logging.warning("snapshot_callback failed (%d). Reason: %s", _snapshot_cb_errors, cb_exc)

        # Only recompute h_in,u every N steps
        if (n - _last_props_step) >= _update_props_every:
            if use_numba:
                _compute_h_in_numba(Tg, _h_in, _u, p_f, R_f, m_dot_f, A_flow_f, Di_f, mu_g_f, Pr_f, k_g_f)
            else:
                _h_in[:], _u[:] = compute_h_in_local(Tg)
            _last_props_step = n
        if not static_solids and (n - _last_solid_step) >= _update_props_every:
            delta_tw = float(np.max(np.abs(Tw - Tw_props_ref)))
            delta_ti = float(np.max(np.abs(Ti - Ti_props_ref)))
            force_due = (n - _last_solid_step) >= _solid_force_steps
            if force_due or delta_tw >= solid_rebuild_temp_k or delta_ti >= solid_rebuild_temp_k:
                Cw_cell_vec, Ci_cell_vec, R_Tw_to_Ti_vec, R_wall_vec, aw_eff, ai_eff, cp_w_local, cp_i_local = build_solid_state(Tw, Ti)
                Tw_props_ref[:] = Tw
                Ti_props_ref[:] = Ti
                _last_solid_step = n
        h_in = _h_in; u = _u

        # Adaptive dt
        if use_numba:
            tau_g_min, tau_w_min, tau_i_min, u_max = _compute_adaptive_dt_numba(
                Tg, Ti, u, h_in, p_f, R_f, A_flow_f, dx_f, cp_g_f, P_in_f,
                Cw_cell_vec, Ci_cell_vec, R_Tw_to_Ti_vec, Tamb_f, D_out_f,
                h_out_manual_f, use_auto_hout, eps_rad_f, sigma_f, P_out_f
            )
            dt_src_local = 0.35 * float(min(tau_g_min, tau_w_min, tau_i_min))
        else:
            u_max = float(np.max(u))
            rho_g_local = p["p"]/(R*Tg)
            Cg_cell = rho_g_local*A_flow*dx*p["cp_g"]
            h_in_area = h_in * P_in * dx
            inv_R_local = 1.0/np.maximum(R_Tw_to_Ti_vec, 1.0e-16)
            tau_g_local = Cg_cell/(h_in_area + 1e-12)
            tau_w_local = Cw_cell_vec/(h_in_area + inv_R_local + 1e-12)
            if use_auto_hout == 1:
                h_out_local = _h_out_natural_conv_vec(Ti, p["Tamb"], float(D_out))
            else:
                h_out_local = h_out_manual
            h_out_term = h_out_local * P_out * dx
            rad_term = 4 * p["eps_rad"] * sigma * P_out * dx * np.maximum(Ti, 1.0) ** 3
            if float(np.max(Ci_cell_vec)) > 1.0e-12:
                tau_i_local = Ci_cell_vec / (inv_R_local + h_out_term + rad_term + 1e-12)
            else:
                tau_i_local = np.full_like(Ti, 1.0e9)
            dt_src_local = 0.35*min(np.min(tau_g_local), np.min(tau_w_local), np.min(tau_i_local))

        if p["adv_scheme"] == "semi_lagrangian":
            dt_cfl_local = semi_lag_courant_max * float(dx) / max(float(u_max), 1.0e-6)
        else:
            dt_cfl_local = p["CFL"] * float(dx) / max(float(u_max), 1.0e-6)

        dt = float(max(p["dt_min"], min(dt_cfl_local, dt_src_local, p["dt_max"])))
        dq = max(1e-9, float(p.get("dt_quantize_pct", 0.0)) * dt)
        if dq > 0.0:
            dt = float(dq * np.round(dt / dq))
        if t + dt > t_end:
            dt = t_end - t
        dt_f = _float(dt)
        Tin_eff_next = inlet_temp_at(t + dt)
        Tin_eff_next_f = _float(Tin_eff_next)
        Tin_eff_f = _float(Tin_eff)

        lam_w = float(max(1.0e-16, aw_eff) * dt / (float(dx) * float(dx)))
        lam_i = float(max(1.0e-16, ai_eff) * dt / (float(dx) * float(dx)))
        _a_w, _cprime_w, _inv_w = cn_w.ensure(lam_w)
        _a_i, _cprime_i, _inv_i = cn_i.ensure(lam_i)
        
        lam_w_f = _float(lam_w)
        lam_i_f = _float(lam_i)
        used_seq_buffers = False

        if use_numba:
            try:
                if p.get("parallel", False):
                    Tg_new, Tw_new, Ti_new = _timestep_numba(
                        Tg, Tw, Ti, dt_f,
                        x0_f, dx_f,
                        p_f, m_dot_f, Tin_eff_f, Tin_eff_next_f, R_f, Pr_f,
                        mu_g_f, k_g_f,
                        Di_f, A_flow_f, P_in_f, P_out_f,
                        cp_g_f, Cw_cell_vec, Ci_cell_vec, R_Tw_to_Ti_vec,
                        h_out_manual_f, use_auto_hout, D_out_f, eps_rad_f, sigma_f,
                        theta_cond_f, dittus_boelter_n_f, use_semi,
                        Tamb_f,
                        lam_w_f, lam_i_f,
                        _a_w, _cprime_w, _inv_w,
                        _a_i, _cprime_i, _inv_i
                    )
                else:
                    _timestep_numba_seq(
                        Tg, Tw, Ti, dt_f,
                        x0_f, dx_f,
                        p_f, m_dot_f, Tin_eff_f, Tin_eff_next_f, R_f, Pr_f,
                        mu_g_f, k_g_f,
                        Di_f, A_flow_f, P_in_f, P_out_f,
                        cp_g_f, Cw_cell_vec, Ci_cell_vec, R_Tw_to_Ti_vec,
                        h_out_manual_f, use_auto_hout, D_out_f, eps_rad_f, sigma_f,
                        theta_cond_f, dittus_boelter_n_f, use_semi,
                        Tamb_f,
                        lam_w_f, lam_i_f,
                        _a_w, _cprime_w, _inv_w,
                        _a_i, _cprime_i, _inv_i,
                        xp_buf, Cg_buf, q_gw_buf,
                        Tg_new_b, Tw_e_b, Ti_e_b, Tw_new_b, Ti_new_b,
                        h_in, u,
                        RHS_w, Y_w, RHS_i, Y_i
                    )
                    Tg_new, Tw_new, Ti_new = Tg_new_b, Tw_new_b, Ti_new_b
                    used_seq_buffers = True
            except Exception as exc:
                logging.exception("Accelerated timestep path failed; using fallback kernels. Reason: %s", exc)
                Tg_new, Tw_new, Ti_new = _python_timestep_fallback(
                    Tg, Tw, Ti, dt, Tin_eff, Tin_eff_next, h_in, u,
                    lam_w, lam_i, _a_w, _cprime_w, _inv_w, _a_i, _cprime_i, _inv_i
                )
        else:
            Tg_new, Tw_new, Ti_new = _python_timestep_fallback(
                Tg, Tw, Ti, dt, Tin_eff, Tin_eff_next, h_in, u,
                lam_w, lam_i, _a_w, _cprime_w, _inv_w, _a_i, _cprime_i, _inv_i
            )

        # Early-stop check on selected outlet target metric
        if Tg_outlet_target is not None:
            target_out = metric_outlet(Tg_new, Tw_new, Ti_new, R_wall_vec, R_Tw_to_Ti_vec)
            if (_cmp_dir == "le" and target_out <= Tg_outlet_target) or (_cmp_dir == "ge" and target_out >= Tg_outlet_target):
                reached_target = True
                stop_reason = f"{target_metric} {'<=' if _cmp_dir == 'le' else '>='} {float(Tg_outlet_target):.1f} K"
            elif asymptote_check_enabled:
                t_next = float(t + dt)
                asymptote_hist_t.append(t_next)
                asymptote_hist_m.append(float(target_out))
                while asymptote_hist_t and (t_next - asymptote_hist_t[0]) > asymptote_window_s:
                    asymptote_hist_t.popleft()
                    asymptote_hist_m.popleft()
                if t_next >= asymptote_min_time and len(asymptote_hist_t) >= 2:
                    span = max(1.0e-9, t_next - asymptote_hist_t[0])
                    if _cmp_dir == "ge":
                        gain = float(target_out - asymptote_hist_m[0])
                        gap = float(Tg_outlet_target - target_out)
                    else:
                        gain = float(asymptote_hist_m[0] - target_out)
                        gap = float(target_out - Tg_outlet_target)
                    rate = gain / span
                    remaining = max(0.0, t_end - t_next)
                    projected = gap / max(rate, 1.0e-12) if rate > 0.0 else float("inf")
                    stalled_by_rate = gap > asymptote_min_gap and rate <= asymptote_rate_tol
                    stalled_by_projection = (
                        gap > asymptote_min_gap
                        and projected > max(2.0 * asymptote_window_s, asymptote_proj_factor * max(remaining, 1.0))
                    )
                    if stalled_by_rate or stalled_by_projection:
                        asymptote_stall_windows += 1
                    else:
                        asymptote_stall_windows = 0
                    if asymptote_stall_windows >= asymptote_need_windows:
                        asymptote_stop = True
                        reason_tag = "asymptote" if stalled_by_rate else "projected_miss"
                        stop_reason = (
                            f"{target_metric} {reason_tag}: target {float(Tg_outlet_target):.1f} K not reachable "
                            f"(current {float(target_out):.1f} K, rate {rate:.4e} K/s, projected {projected:.1f} s)"
                        )

        if energy_interval and (n % energy_interval == 0):
            E = _energy_state(Tg, Tw, Ti, cp_w_local, cp_i_local)
            drift = (E - E0) / (abs(E0) + 1e-12)
            if abs(drift) > 0.05:  # 5% drift threshold
                logging.warning(
                    "Energy drift %.1f%% at t=%.1f s (step %d)",
                    100.0 * drift, t, n
                )

        if used_seq_buffers:
            # Swap input/output buffers to avoid in-place aliasing on next step.
            Tg, Tg_new_b = Tg_new_b, Tg
            Tw, Tw_new_b = Tw_new_b, Tw
            Ti, Ti_new_b = Ti_new_b, Ti
        else:
            Tg, Tw, Ti = Tg_new, Tw_new, Ti_new
        t += dt
        n += 1
        tracker.log_if_needed(t, n)

        if reached_target or asymptote_stop:
            break

    # ensure final frame is captured
    if _frame_idx < _nframes:
        Tw_hist[_frame_idx, :] = Tw
        Tg_hist[_frame_idx, :] = Tg
        Ti_hist[_frame_idx, :] = Ti
        times[_frame_idx] = t
        _frame_idx += 1
        if snapshot_callback is not None:
            try:
                snapshot_callback(float(t), Tg.copy(), Tw.copy(), Ti.copy())
            except Exception as cb_exc:
                _snapshot_cb_errors += 1
                logging.warning("snapshot_callback failed at final frame (%d). Reason: %s", _snapshot_cb_errors, cb_exc)

    total_wall = tracker.finalize(t, n, OUTDIR)
    logging.info("=== Simulation complete ===")
    logging.info(
        "Simulated time: %.2f s (%s) over %d steps",
        t, _fmt_hms(t), n
    )
    logging.info(
        "Total wall-clock time: %.2f s (%.2f min)",
        total_wall, total_wall / 60.0
    )
    logging.info(
        "Average simulation speed: %.2fx real-time",
        (t / total_wall) if total_wall > 0 else 0.0
    )

    # trim unused rows
    Tw_hist = Tw_hist[:_frame_idx, :]
    Tg_hist = Tg_hist[:_frame_idx, :]
    Ti_hist = Ti_hist[:_frame_idx, :]
    times   = times[:_frame_idx]

    # Plot & save (optional)
    if make_plots and save_results and OUTDIR is not None:
        plot_heatmaps(x, times, Tw_hist, Tg_hist, Ti_hist, OUTDIR, p)
        plot_profiles(x, times, Tw_hist, Tg_hist, Ti_hist, OUTDIR, p)
    if save_results and OUTDIR is not None:
        save_arrays_and_csv(OUTDIR, x, times, Tw_hist, Tg_hist, Ti_hist, Nx)
        logging.info("Run complete. Outputs saved to: %s", OUTDIR)

    result = SimRunResult(
        outdir=OUTDIR,
        x=x,
        times=times,
        Tw_hist=Tw_hist,
        Tg_hist=Tg_hist,
        Ti_hist=Ti_hist,
        wall_time_s=total_wall,
        n_steps=n,
        reached_Tg_target=reached_target,
        Tg_outlet_final=float(Tg_hist[-1, -1]) if Tg_hist.size else float('nan'),
        Tg_outlet_target=Tg_outlet_target,
        target_metric=target_metric,
        target_outlet_final=metric_outlet(Tg, Tw, Ti, R_wall_vec, R_Tw_to_Ti_vec),
        stop_reason=stop_reason,
    )
    if run_file_handler is not None:
        logger.removeHandler(run_file_handler)
        run_file_handler.close()
    return result

def run_simulation(config: Optional[Dict[str, Any]] = None) -> SimRunResult:
    """Pure API variant: no plots, no filesystem writes, quiet logging."""
    return main(config, make_plots=False, save_results=False, outdir=None)



if __name__ == "__main__":
    # CLI / direct execution path (uses defaults; plots and saves)
    _ = main()
