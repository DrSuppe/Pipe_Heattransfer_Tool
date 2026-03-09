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
"""Optimization support and worker thread logic for the thermal pipe application."""

from __future__ import annotations

import logging
import traceback
from dataclasses import replace
from typing import Any, Dict

import numpy as np

from sim_controller import RunSpec, run_once

try:
    from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "PyQt6 is required to run this UI.\n"
        "Install with: pip install PyQt6\n"
        f"Import error: {exc}"
    )

from Pipe_Sim_V4 import HAS_NUMBA

TARGET_METRIC_CHOICES: list[tuple[str, str]] = [
    ("Gas outlet Tg", "gas_outlet"),
    ("Wall inner outlet Tw(in)", "wall_inner_outlet"),
    ("Wall outer outlet Tw(out)", "wall_outer_outlet"),
    ("Insulation outlet Ti", "insulation_outlet"),
]

TARGET_METRIC_LABELS: Dict[str, str] = {key: label for label, key in TARGET_METRIC_CHOICES}

def _sanitize_target_metric(metric: str) -> str:
    key = str(metric or "").strip().lower()
    valid = {k for _label, k in TARGET_METRIC_CHOICES}
    return key if key in valid else "gas_outlet"

def _wall_fraction_from_geom(geom: Dict[str, Any]) -> float:
    di = max(float(geom.get("Di", 0.13)), 1.0e-9)
    t_wall = max(float(geom.get("t_wall", 0.018)), 1.0e-9)
    t_ins = max(float(geom.get("t_ins", 0.0)), 0.0)
    k_w = max(float(geom.get("k_w", 15.0)), 1.0e-12)
    k_i = max(float(geom.get("k_i", 0.05)), 1.0e-12)
    ri = 0.5 * di
    ro = ri + t_wall
    r_ins = ro + t_ins
    r_wall = np.log(max(ro / ri, 1.0 + 1.0e-12)) / (2.0 * np.pi * k_w)
    if t_ins > 1.0e-12:
        r_insu = np.log(max(r_ins / ro, 1.0 + 1.0e-12)) / (2.0 * np.pi * k_i)
    else:
        r_insu = 0.0
    return float(r_wall / max(r_wall + r_insu, 1.0e-12))

def _outlet_series_for_metric(
    tg_hist: np.ndarray,
    tw_hist: np.ndarray,
    ti_hist: np.ndarray,
    metric: str,
    geom: Dict[str, Any],
) -> np.ndarray:
    key = _sanitize_target_metric(metric)
    tg_out = np.asarray(tg_hist[:, -1], dtype=float)
    tw_out = np.asarray(tw_hist[:, -1], dtype=float)
    ti_out = np.asarray(ti_hist[:, -1], dtype=float)
    if key == "gas_outlet":
        return tg_out
    if key == "wall_inner_outlet":
        return tw_out
    if key == "insulation_outlet":
        return ti_out
    wall_frac = _wall_fraction_from_geom(geom)
    return tw_out - wall_frac * (tw_out - ti_out)

def _target_crossing_time_series(times: np.ndarray, values: np.ndarray, target: float) -> float | None:
    if values.size == 0 or times.size == 0:
        return None
    mode_le = bool(target <= float(values[0]))
    cond = values <= target if mode_le else values >= target
    hit = np.where(cond)[0]
    if hit.size == 0:
        return None
    i = int(hit[0])
    if i == 0:
        return float(times[0])
    t0, t1 = float(times[i - 1]), float(times[i])
    y0, y1 = float(values[i - 1]), float(values[i])
    if abs(y1 - y0) <= 1.0e-12:
        return t1
    frac = (target - y0) / (y1 - y0)
    frac = max(0.0, min(1.0, frac))
    return t0 + frac * (t1 - t0)


def _estimate_vm_total_max_mpa(
    tw_si: np.ndarray,
    ti_si: np.ndarray,
    *,
    length_m: float,
    geom: Dict[str, Any],
    run_si: Dict[str, Any],
    mech: Dict[str, Any],
    include_pressure: bool,
    nr_wall: int,
) -> float:
    nr = int(max(3, nr_wall))
    E = float(mech.get("E", 210.0e9))
    alpha = float(mech.get("alpha", 12.0e-6))
    nu = float(mech.get("nu", 0.30))

    di_m = float(geom.get("Di", 0.1))
    t_wall_m = float(geom.get("t_wall", 0.01))
    t_ins_m = float(geom.get("t_ins", 0.0))
    k_w = float(geom.get("k_w", 15.0))
    k_i = float(geom.get("k_i", 0.04))
    n_elbows = int(geom.get("n_elbows", 0))
    elbow_sif = float(geom.get("elbow_sif", 1.0))
    elbow_positions_frac = list(geom.get("elbow_positions_frac", []))

    p_si = float(run_si.get("p", 0.0))
    tamb = float(run_si.get("Tamb", 300.0))
    axial_restraint = float(run_si.get("axial_restraint", 0.0))

    ri = max(0.5 * di_m, 1.0e-9)
    ro = ri + max(t_wall_m, 1.0e-9)
    nx = max(1, tw_si.shape[1])
    dx_si = max(length_m, 1.0e-12) / max(1, nx - 1)

    r_wall = np.log(max(ro / ri, 1.0 + 1.0e-12)) / (2.0 * np.pi * max(k_w, 1.0e-12) * max(dx_si, 1.0e-12))
    if t_ins_m > 1.0e-12:
        r_ins_o = ro + t_ins_m
        r_ins = np.log(max(r_ins_o / ro, 1.0 + 1.0e-12)) / (2.0 * np.pi * max(k_i, 1.0e-12) * max(dx_si, 1.0e-12))
    else:
        r_ins = 0.0
    wall_frac = float(r_wall / max(r_wall + r_ins, 1.0e-12))

    delta_t_wall_si = np.abs(tw_si - ti_si) * wall_frac
    xi = np.linspace(0.0, 1.0, nr, dtype=float)
    r = ri + xi * max(t_wall_m, 1.0e-9)
    grad_term = (0.5 - xi)[:, None, None] * delta_t_wall_si[None, :, :]
    temp_r = tw_si[None, :, :] + grad_term

    thermo_coeff = E * alpha / max(1.0e-9, (1.0 - nu))
    sigma_theta_th = thermo_coeff * (tw_si[None, :, :] - temp_r)
    sigma_z_th = -axial_restraint * thermo_coeff * (tw_si - tamb)[None, :, :]
    sigma_r_th = np.zeros_like(sigma_theta_th)

    if include_pressure:
        denom = max(ro * ro - ri * ri, 1.0e-12)
        r2 = np.maximum(r * r, 1.0e-12)[:, None, None]
        coeff = p_si * ri * ri / denom
        sigma_r_p = coeff * (1.0 - (ro * ro) / r2)
        sigma_theta_p = coeff * (1.0 + (ro * ro) / r2)
        sigma_z_p = np.full_like(sigma_r_p, coeff)
    else:
        sigma_r_p = np.zeros_like(temp_r)
        sigma_theta_p = np.zeros_like(temp_r)
        sigma_z_p = np.zeros_like(temp_r)

    sigma_theta = sigma_theta_th + sigma_theta_p
    sigma_r = sigma_r_th + sigma_r_p
    sigma_z = sigma_z_th + sigma_z_p
    sigma_vm_total = np.sqrt(
        0.5 * ((sigma_theta - sigma_z) ** 2 + (sigma_z - sigma_r) ** 2 + (sigma_r - sigma_theta) ** 2)
    )
    vm_total_mpa = np.max(sigma_vm_total, axis=0) / 1.0e6

    elbow_profile = np.ones(nx, dtype=float)
    if n_elbows > 0 and nx > 2:
        if len(elbow_positions_frac) != n_elbows:
            elbow_positions_frac = [float(v) for v in np.linspace(0.15, 0.85, n_elbows)]
        spread = max(1, int(0.03 * nx))
        for frac in elbow_positions_frac:
            j0 = int(round(max(0.0, min(1.0, float(frac))) * (nx - 1)))
            for dj in range(-3 * spread, 3 * spread + 1):
                j = j0 + dj
                if j < 0 or j >= nx:
                    continue
                w = max(0.0, 1.0 - abs(dj) / (3.0 * spread + 1.0e-9))
                elbow_profile[j] = max(elbow_profile[j], 1.0 + (elbow_sif - 1.0) * w)

    vm_total_elbow_mpa = vm_total_mpa * elbow_profile[None, :]
    return float(np.nanmax(vm_total_elbow_mpa))


class UserCancelledError(RuntimeError):
    """Signal user-requested cancellation without surfacing an error dialog."""

class SimulationWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    snapshot = pyqtSignal(float, object, object, object)

    def __init__(self, spec: RunSpec, optimization: Dict[str, Any] | None = None):
        super().__init__()
        self.spec = spec
        self.optimization = optimization or None
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def _abort_requested(self) -> bool:
        return bool(self._cancel_requested)

    def _check_cancel(self):
        if self._cancel_requested:
            raise UserCancelledError("Simulation cancelled by user.")

    @pyqtSlot()
    def run(self):
        try:
            if self.optimization:
                result = self._run_optimization()
            else:
                spec = replace(
                    self.spec,
                    snapshot_callback=self._emit_snapshot,
                    abort_callback=self._abort_requested,
                )
                result = run_once(spec)
            if str(getattr(result, "stop_reason", "")) == "aborted_by_user":
                self.finished.emit(result)
                return
            self.finished.emit(result)
        except UserCancelledError:
            self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _emit_snapshot(self, t_s: float, tg, tw, ti):
        self.snapshot.emit(float(t_s), tg, tw, ti)

    def _target_value(self, opt: Dict[str, Any]) -> float:
        return float(opt.get("target_temp_K", opt.get("target_Tg_K", np.nan)))

    def _target_metric(self, opt: Dict[str, Any]) -> str:
        return _sanitize_target_metric(str(opt.get("target_metric", "gas_outlet")))

    def _candidate_result(self, m_dot_kg_s: float, nx: int, opt: Dict[str, Any]) -> Dict[str, Any]:
        self._check_cancel()
        run = replace(
            self.spec.run,
            mode="target",
            m_dot=float(m_dot_kg_s),
            Tg_out_target=self._target_value(opt),
            t_end=float(opt.get("opt_t_end_s", self.spec.run.t_end)),
            stop_dir=None,
        )
        overrides = dict(self.spec.overrides or {})
        overrides.update(
            {
                "Nx": int(nx),
                "t_end": float(opt.get("opt_t_end_s", self.spec.run.t_end)),
                "save_frames": int(max(20, min(120, int(overrides.get("save_frames", 120))))),
                "progress": "none",
                "log_to_file": False,
                "write_trace_csv": False,
                "show_plots": False,
            }
        )
        spec = RunSpec(
            hardware=self.spec.hardware,
            run=run,
            overrides=overrides,
            save_dir=None,
            make_plots=False,
            save_results=False,
            snapshot_callback=None,
            abort_callback=self._abort_requested,
        )
        r = run_once(spec)
        if str(getattr(r, "stop_reason", "")) == "aborted_by_user":
            raise UserCancelledError("Simulation cancelled by user.")
        target_series = _outlet_series_for_metric(
            np.asarray(r.Tg_hist, dtype=float),
            np.asarray(r.Tw_hist, dtype=float),
            np.asarray(r.Ti_hist, dtype=float),
            self._target_metric(opt),
            opt.get("geom", {}),
        )
        t_hit = _target_crossing_time_series(np.asarray(r.times, dtype=float), target_series, self._target_value(opt))
        sigma_max = _estimate_vm_total_max_mpa(
            np.asarray(r.Tw_hist, dtype=float),
            np.asarray(r.Ti_hist, dtype=float),
            length_m=float(self.spec.hardware.L),
            geom=opt["geom"],
            run_si=opt["run_si"],
            mech=opt["mech"],
            include_pressure=bool(opt["include_pressure"]),
            nr_wall=int(opt["nr_wall"]),
        )
        return {
            "m_dot": float(m_dot_kg_s),
            "reached": bool(r.reached_Tg_target),
            "t_hit_s": t_hit,
            "sigma_max_mpa": float(sigma_max),
        }

    @staticmethod
    def _heatup_score(
        t_hit_s: float | None,
        sigma_mpa: float,
        target_t_s: float,
        tol_t_s: float,
        sigma_lim_mpa: float,
    ) -> tuple[float, float, float]:
        stress_violation = max(0.0, float(sigma_mpa) - float(sigma_lim_mpa))
        if t_hit_s is None:
            return (stress_violation, 1.0e9, 1.0e9)
        abs_err = abs(float(t_hit_s) - float(target_t_s))
        time_violation = max(0.0, abs_err - float(tol_t_s))
        return (stress_violation, time_violation, abs_err)

    def _full_nx_for_opt(self, opt: Dict[str, Any]) -> int:
        nx_from_overrides = None
        if isinstance(self.spec.overrides, dict):
            try:
                nx_from_overrides = int(self.spec.overrides.get("Nx", 0))
            except Exception:
                nx_from_overrides = None
        if nx_from_overrides is None or nx_from_overrides <= 0:
            nx_from_overrides = int(max(120, opt.get("search_nx", 120)))
        return int(max(80, nx_from_overrides))

    def _run_final_candidate(self, m_dot_kg_s: float, opt: Dict[str, Any]):
        self._check_cancel()
        run_final = replace(
            self.spec.run,
            mode="target",
            m_dot=float(m_dot_kg_s),
            Tg_out_target=self._target_value(opt),
            t_end=float(opt.get("opt_t_end_s", self.spec.run.t_end)),
            stop_dir=None,
        )
        overrides_final = dict(self.spec.overrides or {})
        overrides_final["t_end"] = float(run_final.t_end)
        final_spec = replace(
            self.spec,
            run=run_final,
            overrides=overrides_final,
            snapshot_callback=self._emit_snapshot,
            abort_callback=self._abort_requested,
        )
        result = run_once(final_spec)
        if str(getattr(result, "stop_reason", "")) == "aborted_by_user":
            raise UserCancelledError("Simulation cancelled by user.")

        target_series = _outlet_series_for_metric(
            np.asarray(result.Tg_hist, dtype=float),
            np.asarray(result.Tw_hist, dtype=float),
            np.asarray(result.Ti_hist, dtype=float),
            self._target_metric(opt),
            opt.get("geom", {}),
        )
        t_hit_final = _target_crossing_time_series(np.asarray(result.times, dtype=float), target_series, self._target_value(opt))
        sigma_final_mpa = _estimate_vm_total_max_mpa(
            np.asarray(result.Tw_hist, dtype=float),
            np.asarray(result.Ti_hist, dtype=float),
            length_m=float(self.spec.hardware.L),
            geom=opt["geom"],
            run_si=opt["run_si"],
            mech=opt["mech"],
            include_pressure=bool(opt["include_pressure"]),
            nr_wall=int(opt["nr_wall"]),
        )
        return result, t_hit_final, float(sigma_final_mpa)

    def _run_heatup_time_opt(self, opt: Dict[str, Any]) -> Dict[str, Any]:
        self._check_cancel()
        md_min = float(opt["mdot_min_kg_s"])
        md_max = float(opt["mdot_max_kg_s"])
        target_t = float(opt["heatup_target_s"])
        tol_t = float(opt["heatup_tol_s"])
        sigma_lim = float(opt["stress_limit_mpa"])
        nx_search = int(opt["search_nx"])
        coarse_points = int(max(5, opt.get("coarse_points", 9)))
        refine_iters = int(max(2, opt.get("refine_iters", 6)))

        cache: Dict[float, Dict[str, Any]] = {}

        def evaluate(m: float) -> Dict[str, Any]:
            m_key = float(np.round(m, 10))
            if m_key not in cache:
                cache[m_key] = self._candidate_result(m_key, nx_search, opt)
            return cache[m_key]

        def score(c: Dict[str, Any]) -> tuple[float, float, float]:
            t_hit = c["t_hit_s"]
            stress_violation = max(0.0, float(c["sigma_max_mpa"]) - sigma_lim)
            if t_hit is None:
                time_violation = 1.0e9
                abs_time_err = 1.0e9
            else:
                abs_time_err = abs(float(t_hit) - target_t)
                time_violation = max(0.0, abs_time_err - tol_t)
            return (stress_violation, time_violation, abs_time_err)

        grid = np.linspace(md_min, md_max, coarse_points)
        coarse = [evaluate(float(m)) for m in grid]
        best = min(coarse, key=score)
        best_score = score(best)

        idx_best = int(np.argmin([score(c) for c in coarse]))
        left = float(grid[max(0, idx_best - 1)])
        right = float(grid[min(len(grid) - 1, idx_best + 1)])
        if right <= left:
            left, right = md_min, md_max

        for _ in range(refine_iters):
            self._check_cancel()
            pts = np.linspace(left, right, 5)
            local = [evaluate(float(m)) for m in pts]
            local_best = min(local, key=score)
            local_score = score(local_best)
            if local_score < best_score:
                best, best_score = local_best, local_score

            if (
                local_best["t_hit_s"] is not None
                and abs(float(local_best["t_hit_s"]) - target_t) <= tol_t
                and float(local_best["sigma_max_mpa"]) <= sigma_lim
            ):
                best = local_best
                break

            i_local = int(np.argmin([score(c) for c in local]))
            l_idx = max(0, i_local - 1)
            r_idx = min(len(pts) - 1, i_local + 1)
            new_left = float(pts[l_idx])
            new_right = float(pts[r_idx])
            if new_right - new_left <= 1.0e-9:
                break
            left, right = new_left, new_right

        return best

    def _run_stress_limit_opt(self, opt: Dict[str, Any]) -> Dict[str, Any]:
        self._check_cancel()
        md_min = float(opt["mdot_min_kg_s"])
        md_max = float(opt["mdot_max_kg_s"])
        sigma_lim = float(opt["stress_limit_mpa"])
        sigma_tol = float(opt.get("stress_tol_mpa", 2.0))
        nx_search = int(opt["search_nx"])
        max_iter = int(max(4, opt.get("bisection_iters", 10)))

        def evaluate(m: float) -> Dict[str, Any]:
            return self._candidate_result(float(m), nx_search, opt)

        lo = evaluate(md_min)
        hi = evaluate(md_max)
        f_lo = float(lo["sigma_max_mpa"]) - sigma_lim
        f_hi = float(hi["sigma_max_mpa"]) - sigma_lim

        if f_lo <= 0.0 and f_hi <= 0.0:
            return hi
        if f_lo > 0.0 and f_hi > 0.0:
            return lo

        l_m, h_m = md_min, md_max
        c_best = lo if abs(f_lo) <= abs(f_hi) else hi
        for _ in range(max_iter):
            self._check_cancel()
            mid = 0.5 * (l_m + h_m)
            c_mid = evaluate(mid)
            f_mid = float(c_mid["sigma_max_mpa"]) - sigma_lim
            if abs(f_mid) < abs(float(c_best["sigma_max_mpa"]) - sigma_lim):
                c_best = c_mid
            if abs(f_mid) <= sigma_tol:
                c_best = c_mid
                break
            if f_lo * f_mid <= 0.0:
                h_m = mid
                hi = c_mid
                f_hi = f_mid
            else:
                l_m = mid
                lo = c_mid
                f_lo = f_mid
            if h_m - l_m <= 1.0e-9:
                break
        return c_best

    def _run_optimization(self):
        self._check_cancel()
        opt = dict(self.optimization or {})
        mode = str(opt.get("mode", ""))
        if mode not in ("heatup_time_opt", "stress_limit_opt"):
            raise ValueError(f"Unsupported optimization mode: {mode}")

        if mode == "heatup_time_opt":
            best = self._run_heatup_time_opt(opt)
        else:
            best = self._run_stress_limit_opt(opt)

        mdot_best = float(best["m_dot"])
        result, t_hit_final, sigma_final_mpa = self._run_final_candidate(mdot_best, opt)

        sigma_lim = float(opt["stress_limit_mpa"])
        nx_full = self._full_nx_for_opt(opt)
        correction_applied = False

        if mode == "heatup_time_opt":
            tgt = float(opt.get("heatup_target_s", np.nan))
            tol = float(opt.get("heatup_tol_s", np.nan))
            cur_score = self._heatup_score(t_hit_final, sigma_final_mpa, tgt, tol, sigma_lim)
            if cur_score[0] > 0.0 or cur_score[1] > 0.0:
                md_min = float(opt["mdot_min_kg_s"])
                md_max = float(opt["mdot_max_kg_s"])
                span = max(md_max - md_min, 1.0e-8)
                local_half = max(0.2 * span, 0.25 * mdot_best)
                left = max(md_min, mdot_best - local_half)
                right = min(md_max, mdot_best + local_half)
                probes = np.unique(
                    np.concatenate(
                        (
                            np.array([md_min, left, mdot_best, right, md_max], dtype=float),
                            np.linspace(left, right, 7),
                        )
                    )
                )
                candidates = [self._candidate_result(float(m), nx_full, opt) for m in probes]
                best_full = min(
                    candidates,
                    key=lambda c: self._heatup_score(
                        c.get("t_hit_s"),
                        float(c.get("sigma_max_mpa", np.inf)),
                        tgt,
                        tol,
                        sigma_lim,
                    ),
                )
                best_full_score = self._heatup_score(
                    best_full.get("t_hit_s"),
                    float(best_full.get("sigma_max_mpa", np.inf)),
                    tgt,
                    tol,
                    sigma_lim,
                )
                if best_full_score < cur_score and abs(float(best_full["m_dot"]) - mdot_best) > 1.0e-12:
                    mdot_best = float(best_full["m_dot"])
                    result, t_hit_final, sigma_final_mpa = self._run_final_candidate(mdot_best, opt)
                    correction_applied = True
        else:
            md_min = float(opt["mdot_min_kg_s"])
            md_max = float(opt["mdot_max_kg_s"])
            sigma_tol = float(opt.get("stress_tol_mpa", 2.0))
            if sigma_final_mpa > sigma_lim + sigma_tol:
                probes = np.linspace(md_min, max(md_min, mdot_best), 9)
                candidates = [self._candidate_result(float(m), nx_full, opt) for m in probes]
                passing = [c for c in candidates if float(c.get("sigma_max_mpa", np.inf)) <= sigma_lim + sigma_tol]
                if passing:
                    c_pick = max(passing, key=lambda c: float(c["m_dot"]))
                    if abs(float(c_pick["m_dot"]) - mdot_best) > 1.0e-12:
                        mdot_best = float(c_pick["m_dot"])
                        result, t_hit_final, sigma_final_mpa = self._run_final_candidate(mdot_best, opt)
                        correction_applied = True
            elif sigma_final_mpa <= sigma_lim:
                probes = np.linspace(min(mdot_best, md_max), md_max, 7)
                candidates = [self._candidate_result(float(m), nx_full, opt) for m in probes]
                passing = [c for c in candidates if float(c.get("sigma_max_mpa", np.inf)) <= sigma_lim + sigma_tol]
                if passing:
                    c_pick = max(passing, key=lambda c: float(c["m_dot"]))
                    if abs(float(c_pick["m_dot"]) - mdot_best) > 1.0e-12:
                        mdot_best = float(c_pick["m_dot"])
                        result, t_hit_final, sigma_final_mpa = self._run_final_candidate(mdot_best, opt)
                        correction_applied = True

        meets_stress = bool(sigma_final_mpa <= sigma_lim)
        meets_heatup = True
        if mode == "heatup_time_opt":
            tgt = float(opt.get("heatup_target_s", np.nan))
            tol = float(opt.get("heatup_tol_s", np.nan))
            meets_heatup = bool(t_hit_final is not None and abs(float(t_hit_final) - tgt) <= tol)

        logging.info(
            "Optimization summary: mode=%s, m_dot=%.5f kg/s, t_hit_final=%s, sigma_final=%.2f MPa, "
            "meets_stress=%s, meets_heatup=%s, correction=%s",
            mode,
            mdot_best,
            "n/a" if t_hit_final is None else f"{float(t_hit_final):.2f}s",
            sigma_final_mpa,
            meets_stress,
            meets_heatup,
            correction_applied,
        )
        result.opt_summary = {
            "mode": mode,
            "m_dot_kg_s": mdot_best,
            "target_metric": self._target_metric(opt),
            "target_temp_K": self._target_value(opt),
            "sigma_est_search_mpa": float(best["sigma_max_mpa"]),
            "sigma_final_mpa": float(sigma_final_mpa),
            "time_to_target_search_s": None if best["t_hit_s"] is None else float(best["t_hit_s"]),
            "time_to_target_final_s": None if t_hit_final is None else float(t_hit_final),
            "stress_limit_mpa": float(opt["stress_limit_mpa"]),
            "heatup_target_s": float(opt.get("heatup_target_s", np.nan)),
            "heatup_tol_s": float(opt.get("heatup_tol_s", np.nan)),
            "meets_stress_limit": meets_stress,
            "meets_heatup_tolerance": meets_heatup,
            "target_reached_final": bool(result.reached_Tg_target),
            "correction_applied": correction_applied,
            "search_nx": int(opt.get("search_nx", 0)),
            "full_nx": int(nx_full),
        }
        return result
