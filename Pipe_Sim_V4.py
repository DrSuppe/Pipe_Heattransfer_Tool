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
"""Core solver runner (compatibility entrypoint) with modular helper imports."""

import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from thermal_pipe.config import (
    _build_thermal_mass_profile,
    _interp_props,
    _make_run_dir,
    _prepare_prop_table,
    params,
    validate_params,
)
from thermal_pipe import numerics as _numerics
from thermal_pipe.outputs import plot_heatmaps, plot_profiles, save_arrays_and_csv
from thermal_pipe.runtime import RuntimeTracker, SnapshotScheduler, _fmt_hms

# Re-export key numerics symbols for backwards compatibility with existing imports.
HAS_NUMBA = _numerics.HAS_NUMBA
EPS = _numerics.EPS
CN_REL_TOL = _numerics.CN_REL_TOL
_as_dtype = _numerics._as_dtype
_float = _numerics._float
R = _numerics.R
sigma = _numerics.sigma
compute_h_in = _numerics.compute_h_in
CNCache = _numerics.CNCache
_build_cn_factors = _numerics._build_cn_factors
_h_out_natural_conv_scalar = _numerics._h_out_natural_conv_scalar
_h_out_natural_conv_vec = _numerics._h_out_natural_conv_vec
_numba_sanity_check = _numerics._numba_sanity_check
_timestep_numba = _numerics._timestep_numba
_timestep_numba_seq = _numerics._timestep_numba_seq
advect_semi_lagrangian = _numerics.advect_semi_lagrangian
diffuse_axial_CN = _numerics.diffuse_axial_CN
_cn_solve_with_cache = _numerics._cn_solve_with_cache
_compute_h_in_numba = getattr(_numerics, "_compute_h_in_numba", None)
_compute_adaptive_dt_numba = getattr(_numerics, "_compute_adaptive_dt_numba", None)

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
    use_numba = bool(
        HAS_NUMBA
        and p.get("enable_numba", True)
        and _compute_h_in_numba is not None
        and _compute_adaptive_dt_numba is not None
    )

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
