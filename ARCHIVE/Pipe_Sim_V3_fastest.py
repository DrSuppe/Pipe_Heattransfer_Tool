# Rerun with smaller grid and capped frames to keep execution quick in this environment.
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
import datetime
import time

# --- Output directory and logging setup ---
def _make_run_dir(prefix="runs"):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(prefix) / f"run_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

params = {
    "L": 65, "Di": 0.13, "t_wall": 0.018, "t_ins": 0.15, "Nx": 2500,
    "p": 5.0e6, "m_dot": 1, "Tin": 1000.0,
    "T_init_wall": 300.0, "T_init_ins": 300.0, "Tamb": 300.0,
    "rho_w": 8220.0, "cp_w": 564.0, "k_w": 21.9,
    "rho_i": 128.0, "cp_i": 1246, "k_i": 0.09,
    "cp_g": 1005.0, "k_g": 0.028, "mu_g": 2.0e-5, "Pr": 0.71, "dittus_boelter_n": 0.4,
    "h_out": 8.0, "eps_rad": 0.7,
    "t_end": 1800.0, "CFL": 0.6, "theta_cond": 0.5,

    # Performance/stability controls
    "adv_scheme": "semi_lagrangian",   # options: "semi_lagrangian" or "upwind"
    "dt_max": 5.0,                     # hard cap for adaptive time step [s]
    "save_frames": 240,                # reduce I/O and RAM (was 1000)
    "log_interval_s": 10.0,            # runtime log interval in wall seconds
    "log_interval_steps": 1000,        # runtime log interval in steps
    "use_float32": True,
    "dt_quantize_pct": 0.1,
    "update_props_every": 5,  # recompute h_in,u every 5 steps (reduces div/pow work)

    "parallel": False,  # single-threaded kernel (faster on mixed-core CPUs)
    "progress": "basic",  # "none" (quiet) or "basic" (periodic prints)
    "log_to_file": True,  # disable disk logging to avoid I/O
    "write_trace_csv": True  # skip runtime_trace.csv

}

OUTDIR = _make_run_dir()
if params.get("log_to_file", False):
    logging.basicConfig(
        filename=str(OUTDIR / "run.log"),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.info("Run directory: %s", OUTDIR)
else:
    logging.getLogger().handlers.clear()
    logging.disable(logging.CRITICAL)

# Save parameters snapshot at start
with open(OUTDIR / "params.json", "w") as _f:
    json.dump(params, _f, indent=2)
logging.info("Saved params.json")

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
    rho = params["p"]/(R*Tg)
    u = params["m_dot"]/(rho*A_flow)
    Re = rho*u*Di/params["mu_g"]
    Re = np.maximum(Re,1.0)
    Nu = 0.023*(Re**0.8)*(params["Pr"]**params["dittus_boelter_n"])
    return Nu*params["k_g"]/Di, u

try:
    from numba import njit, prange

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

    logging.info("Numba acceleration for diffuse_axial_CN is ENABLED.")

except Exception as _e:
    logging.warning("Numba not available; using pure-Python diffuse_axial_CN. Reason: %s", _e)

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

# --- CN factor cache keyed by dt ---
_CN_REL_TOL = 0.05  # rebuild CN factors if |Δdt|/dt > 5%
_dt_last = None
_lam_w_last = None
_lam_i_last = None
_a_w = _cprime_w = _inv_w = None
_a_i = _cprime_i = _inv_i = None


@njit(cache=True, fastmath=True, nogil=True)
def _timestep_numba_seq(Tg, Tw, Ti, dt,
                        x0, dx,
                        p, m_dot, Tin, R_g, Pr, mu_g, k_g,
                        Di, A_flow, P_in, P_out,
                        cp_g, Cw_cell, Ci_cell, R_Tw_to_Ti,
                        h_out, eps_rad, sigma,
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

    # --- Gas advection + source (semi-Lagrangian or upwind), no allocations ---
    if use_semi_lag == 1:
        for i in range(n):
            xp_buf[i] = (x0 + i*dx) - u[i]*dt
        Tg_adv = _interp1d_uniform_vec(x0, dx, Tg, xp_buf, Tin, Tg[-1])  # returns a view; ok
        for i in range(n):
            # Local gas cell heat capacity (rho depends on current Tg)
            rho_i   = p / (R_g * Tg[i])
            Cg_buf[i]  = rho_i * A_flow * dx * cp_g
            q_gw_buf[i]= h_in[i] * P_in * (Tw[i] - Tg_adv[i])
            Tg_out[i]  = Tg_adv[i] + dt * (q_gw_buf[i] / (Cg_buf[i] if Cg_buf[i] > 1e-12 else 1e-12))
    else:
        for i in range(n):
            rho_i = p / (R_g * Tg[i])
            Cg    = rho_i * A_flow * dx * cp_g
            Tg_up = Tin if i == 0 else Tg[i-1]
            adv   = -u[i] * (Tg[i] - Tg_up) / dx
            q_gw  = h_in[i] * P_in * (Tw[i] - Tg[i])
            dTg_dt= adv + q_gw / (Cg if Cg > 1e-12 else 1e-12)
            Tg_out[i] = Tg[i] + dt * dTg_dt

    # --- Local 2x2 implicit (per-cell) for Tw, Ti (no allocations) ---
    inv_R = 1.0 / (R_Tw_to_Ti if R_Tw_to_Ti > 1e-16 else 1e-16)
    for i in range(n):
        k_rad = 4.0 * eps_rad * sigma * P_out * (Ti[i] ** 3)
        a11 = Cw_cell/dt + h_in[i]*P_in + inv_R
        a12 = -inv_R
        a21 = -inv_R
        a22 = Ci_cell/dt + inv_R + h_out*P_out + k_rad
        rhs1 = (Cw_cell/dt)*Tw[i] + h_in[i]*P_in*Tg_out[i]
        rhs2 = (Ci_cell/dt)*Ti[i] + h_out*P_out*Tamb + eps_rad*sigma*P_out*((Tamb**4) - (Ti[i]**4)) + k_rad*Ti[i]
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
                    p, m_dot, Tin, R_g, Pr, mu_g, k_g,
                    Di, A_flow, P_in, P_out,
                    cp_g, Cw_cell, Ci_cell, R_Tw_to_Ti,
                    h_out, eps_rad, sigma,
                    theta_cond, dittus_n, use_semi_lag,
                    Tamb,
                    lam_w, lam_i,
                    a_w, cprime_w, inv_w,
                    a_i, cprime_i, inv_i):
    n = Tg.shape[0]
    # Flow properties
    rho = p/(R_g*Tg)
    u = m_dot/(rho*A_flow)
    Re = rho*u*Di/mu_g
    for i in prange(n):
        if Re[i] < 1.0:
            Re[i] = 1.0
    Nu = 0.023*(Re**0.8)*(Pr**dittus_n)
    h_in = Nu*k_g/Di

    # Gas advection + source
    if use_semi_lag == 1:
        xp = np.empty(n, dtype=Tg.dtype)
        for i in prange(n):
            xp[i] = (x0 + i*dx) - u[i]*dt
        Tg_adv = _interp1d_uniform_vec(x0, dx, Tg, xp, Tin, Tg[-1])
        Cg_cell = rho*A_flow*dx*cp_g
        q_gw = h_in*P_in*(Tw - Tg_adv)
        Tg_new = Tg_adv + dt*(q_gw/np.maximum(Cg_cell, 1e-12))
    else:
        Tg_new = np.empty_like(Tg)
        for i in prange(n):
            Cg = rho[i]*A_flow*dx*cp_g
            if i == 0:
                Tg_up = Tin
            else:
                Tg_up = Tg[i-1]
            adv = -u[i]*(Tg[i] - Tg_up)/dx
            q_gw = h_in[i]*P_in*(Tw[i] - Tg[i])
            dTg_dt = adv + q_gw/np.maximum(Cg, 1e-12)
            Tg_new[i] = Tg[i] + dt*dTg_dt

    # Semi-implicit local 2x2 for (Tw, Ti) with linearized radiation
    Tw_e = np.empty_like(Tw)
    Ti_e = np.empty_like(Ti)
    inv_R = 1.0/np.maximum(R_Tw_to_Ti, 1e-16)
    for i in prange(n):
        # coefficients
        k_rad = 4.0 * eps_rad * sigma * P_out * (Ti[i]**3)
        a11 = Cw_cell/dt + h_in[i]*P_in + inv_R
        a12 = -inv_R
        a21 = -inv_R
        a22 = Ci_cell/dt + inv_R + h_out*P_out + k_rad
        rhs1 = (Cw_cell/dt)*Tw[i] + h_in[i]*P_in*Tg_new[i]
        rhs2 = (Ci_cell/dt)*Ti[i] + h_out*P_out*Tamb + eps_rad*sigma*P_out*((Tamb**4) - (Ti[i]**4)) + k_rad*Ti[i]
        det = a11*a22 - a12*a21
        Tw_e[i] = (rhs1*a22 - a12*rhs2)/det
        Ti_e[i] = (a11*rhs2 - rhs1*a21)/det

    # Axial diffusion (CN) for wall and insulation using cached factors
    Tw_new = _cn_solve_with_cache(Tw_e, lam_w, theta_cond, a_w, cprime_w, inv_w)
    Ti_new = _cn_solve_with_cache(Ti_e, lam_i, theta_cond, a_i, cprime_i, inv_i)
    return Tg_new, Tw_new, Ti_new


# --- Semi-Lagrangian advection and snapshot scheduling ---
def advect_semi_lagrangian(T_old, u, dt, x, Tin):
    """Backward-characteristic semi-Lagrangian advection for positive u.
    For each Eulerian point x_i, sample T at x_i - u_i*dt; if it is < 0, use Tin.
    """
    x_depart = x - u*dt
    # np.interp handles vectorized sampling; left=Tin applies for x<0
    T_adv = np.interp(x_depart, x, T_old, left=Tin, right=T_old[-1])
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
    def __init__(self, t_end, interval_s=5.0, interval_steps=1000):
        self.t_end = float(t_end)
        self.interval_s = float(interval_s)
        self.interval_steps = int(max(1, interval_steps))
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
        if params.get("progress", "none") == "none":
            return
        now = time.perf_counter()
        need = (now - self.last_log_wall) >= self.interval_s or (steps % self.interval_steps == 0)
        if need:
            wall, mean_dt, speed, eta = self._metrics(sim_t, steps)
            self.records.append((wall, sim_t, steps, mean_dt, speed, eta))
            print(
                f"Progress: steps={steps}, sim_t={sim_t:.2f}s ({_fmt_hms(sim_t)}), "
                f"wall={wall:.2f}s ({_fmt_hms(wall)}), mean_dt={mean_dt:.4f}s, "
                f"speed={speed:.2f}x, ETA={(eta if eta != float('inf') else -1.0):.1f}s "
                f"({_fmt_hms(eta) if eta != float('inf') else '--:--:--'})"
            )
            self.last_log_wall = now
    def finalize(self, sim_t, steps, outdir: Path):
        wall, mean_dt, speed, eta = self._metrics(sim_t, steps)
        logging.info(
            "Runtime summary: steps=%d, sim_t=%.2fs (%s), wall=%.2fs (%s), mean_dt=%.4fs, sim_speed=%.2fx real-time",
            steps, sim_t, _fmt_hms(sim_t), wall, _fmt_hms(wall), mean_dt, speed
        )
        # Persist trace CSV
        if params.get("write_trace_csv", False) and self.records:
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

def plot_heatmaps(x, times, Tw_hist, Tg_hist, Ti_hist, OUTDIR, params):
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
    plt.show()
    logging.info("Saved heatmaps.png")


def plot_profiles(x, times, Tw_hist, Tg_hist, Ti_hist, OUTDIR, params):
    plt.figure(figsize=(10, 4))
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
    plt.show()
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
    print("Final outlet gas T [K]:", float(Tg_hist[-1, -1]))
    print("Final Tw inlet/mid/outlet [K]:", float(Tw_hist[-1, 0]), float(Tw_hist[-1, Nx // 2]), float(Tw_hist[-1, -1]))
    print("Final Ti inlet/mid/outlet [K]:", float(Ti_hist[-1, 0]), float(Ti_hist[-1, Nx // 2]), float(Ti_hist[-1, -1]))
    print(f"Outputs saved to: {OUTDIR}")

Tg = np.full(Nx, params["Tin"], dtype=DTYPE)
Tw = np.full(Nx, params["T_init_wall"], dtype=DTYPE)
Ti = np.full(Nx, params["T_init_ins"], dtype=DTYPE)
h0, u0 = compute_h_in(np.full(Nx, params["Tin"], dtype=DTYPE))
dt_cfl = params["CFL"]*dx/max(u0.max(), 1e-6)

# --- NEW: cache for flow properties to avoid recomputing every step ---
_h_in = h0.copy()
_u = u0.copy()
_update_props_every = int(max(1, params.get("update_props_every", 1)))
_last_props_step = -1

tau_g = ( (params["p"]/(R*params["Tin"]))*A_flow*dx*params["cp_g"] )/((h0*P_in).max()+1e-9)
tau_wi = Cw_cell.min()/(1.0/R_Tw_to_Ti)
dt_src = 0.3*min(tau_g, tau_wi, 1e9)
dt = float(min(dt_cfl, dt_src, 0.25))

# --- Setup runtime tracker ---
t = 0.0
t_end = params["t_end"]

tracker = RuntimeTracker(t_end, params.get("log_interval_s", 5.0), params.get("log_interval_steps", 1000))
tracker.start()
saver = SnapshotScheduler(t_end, params["save_frames"])

# Preallocate snapshots
_nframes = int(max(2, params["save_frames"]))
Tw_hist = np.empty((_nframes, Nx), dtype=DTYPE)
Tg_hist = np.empty((_nframes, Nx), dtype=DTYPE)
Ti_hist = np.empty((_nframes, Nx), dtype=DTYPE)
times   = np.empty((_nframes,),   dtype=DTYPE)
_frame_idx = 0
n = 0

# Work/output buffers for the sequential kernel (no per-step allocations)
xp_buf    = np.empty(Nx, dtype=DTYPE)
Cg_buf    = np.empty(Nx, dtype=DTYPE)
q_gw_buf  = np.empty(Nx, dtype=DTYPE)

# Gas + wall/insulation outputs
Tg_new_b  = np.empty(Nx, dtype=DTYPE)
Tw_e_b    = np.empty(Nx, dtype=DTYPE)  # explicit (pre-diffusion) wall temp per cell
Ti_e_b    = np.empty(Nx, dtype=DTYPE)  # explicit (pre-diffusion) insulation temp per cell
Tw_new_b  = np.empty(Nx, dtype=DTYPE)  # post-CN wall
Ti_new_b  = np.empty(Nx, dtype=DTYPE)  # post-CN insulation

# Scratch for CN solves (RHS and y vectors) — separate for wall/ins or reuse sequentially
RHS_w = np.empty(Nx, dtype=DTYPE)
Y_w   = np.empty(Nx, dtype=DTYPE)
RHS_i = np.empty(Nx, dtype=DTYPE)
Y_i   = np.empty(Nx, dtype=DTYPE)

while t < t_end - 1e-12:

    if saver.should_save(t) and _frame_idx < _nframes:
        Tw_hist[_frame_idx, :] = Tw
        Tg_hist[_frame_idx, :] = Tg
        Ti_hist[_frame_idx, :] = Ti
        times[_frame_idx] = t
        _frame_idx += 1
        saver.mark_saved()

    # Only recompute h_in,u every N steps
    if (n - _last_props_step) >= _update_props_every:
        _h_in[:], _u[:] = compute_h_in(Tg)
        _last_props_step = n
    h_in = _h_in
    u = _u

    # Adaptive dt: source limits always apply; CFL only for upwind scheme
    if params["adv_scheme"] == "semi_lagrangian":
        dt_cfl = np.inf
    else:
        dt_cfl = params["CFL"]*dx/np.maximum(u.max(),1e-6)
    rho_g_local = params["p"]/(R*Tg)
    Cg_cell = rho_g_local*A_flow*dx*params["cp_g"]
    tau_g_local = Cg_cell/(h_in*P_in + 1e-12)
    tau_w_local = Cw_cell/( (1.0/R_Tw_to_Ti) + 1e-12 )
    tau_i_local = Ci_cell/( params["h_out"]*P_out + 4*params["eps_rad"]*sigma*P_out*np.maximum(Ti,1)**3 + 1e-12 )
    dt_src = 0.35*min(np.min(tau_g_local), np.min(tau_w_local), np.min(tau_i_local))

    dt = float(max(1e-6, min(dt_cfl, dt_src, params["dt_max"])))
    # Quantize dt to stabilize CN caching (e.g., 5% steps)
    dq = max(1e-9, float(params.get("dt_quantize_pct", 0.0)) * dt)
    if dq > 0.0:
        dt = float(dq * np.round(dt / dq))
    if t + dt > t_end:
        dt = t_end - t

    # --- CN factor cache logic (keyed by dt) ---
    lam_w = float(aw * dt / (dx * dx))
    lam_i = float(ai * dt / (dx * dx))
    need_rebuild = (_dt_last is None) or (abs(dt - _dt_last) > _CN_REL_TOL * max(dt, 1e-16))
    if need_rebuild:
        _a_w, _cprime_w, _inv_w = _build_cn_factors(Nx, lam_w, params["theta_cond"], DTYPE)
        _a_i, _cprime_i, _inv_i = _build_cn_factors(Nx, lam_i, params["theta_cond"], DTYPE)
        _dt_last = dt
        _lam_w_last = lam_w
        _lam_i_last = lam_i

    use_semi = 1 if params["adv_scheme"] == "semi_lagrangian" else 0
    try:
        if params.get("parallel", False):
            Tg_new, Tw_new, Ti_new = _timestep_numba(
                Tg, Tw, Ti, _float(dt),
                _float(x[0]), _float(dx),
                _float(params["p"]), _float(params["m_dot"]), _float(params["Tin"]), _float(R), _float(params["Pr"]),
                _float(params["mu_g"]), _float(params["k_g"]),
                _float(Di), _float(A_flow), _float(P_in), _float(P_out),
                _float(params["cp_g"]), _float(Cw_cell), _float(Ci_cell), _float(R_Tw_to_Ti),
                _float(params["h_out"]), _float(params["eps_rad"]), _float(sigma),
                _float(params["theta_cond"]), _float(params["dittus_boelter_n"]), use_semi,
                _float(params["Tamb"]),
                _float(lam_w), _float(lam_i),
                _a_w, _cprime_w, _inv_w,
                _a_i, _cprime_i, _inv_i
            )
        else:
            # No-alloc sequential kernel: writes into *_b buffers
            _timestep_numba_seq(
                Tg, Tw, Ti, _float(dt),
                _float(x[0]), _float(dx),
                _float(params["p"]), _float(params["m_dot"]), _float(params["Tin"]), _float(R), _float(params["Pr"]),
                _float(params["mu_g"]), _float(params["k_g"]),
                _float(Di), _float(A_flow), _float(P_in), _float(P_out),
                _float(params["cp_g"]), _float(Cw_cell), _float(Ci_cell), _float(R_Tw_to_Ti),
                _float(params["h_out"]), _float(params["eps_rad"]), _float(sigma),
                _float(params["theta_cond"]), _float(params["dittus_boelter_n"]), use_semi,
                _float(params["Tamb"]),
                _float(lam_w), _float(lam_i),
                _a_w, _cprime_w, _inv_w,
                _a_i, _cprime_i, _inv_i,
                xp_buf, Cg_buf, q_gw_buf,
                Tg_new_b, Tw_e_b, Ti_e_b, Tw_new_b, Ti_new_b,
                h_in, u,
                RHS_w, Y_w, RHS_i, Y_i
            )
            Tg_new, Tw_new, Ti_new = Tg_new_b, Tw_new_b, Ti_new_b

    except Exception:
        # Fallback to original Python path if Numba is unavailable
        if params["adv_scheme"] == "semi_lagrangian":
            Tg_adv = advect_semi_lagrangian(Tg, u, dt, x, params["Tin"])
            q_gw = h_in*P_in*(Tw - Tg_adv)
            Tg_new = Tg_adv + dt * ( q_gw / np.maximum(Cg_cell,1e-12) )
        else:
            Tg_up = np.roll(Tg,1); Tg_up[0] = params["Tin"]
            adv = -u*(Tg - Tg_up)/dx
            q_gw = h_in*P_in*(Tw - Tg)
            dTg_dt = adv + q_gw/np.maximum(Cg_cell,1e-12)
            Tg_new = Tg + dt*dTg_dt
        # Semi-implicit 2x2 local solve for (Tw, Ti)
        Tw_new = np.empty_like(Tw)
        Ti_new = np.empty_like(Ti)
        inv_R = 1.0/np.maximum(R_Tw_to_Ti,1e-16)
        Ti3 = np.maximum(Ti,1.0)**3
        k_rad_vec = 4.0 * params["eps_rad"]*sigma*P_out * Ti3
        for i in range(Nx):
            a11 = Cw_cell/dt + h_in[i]*P_in + inv_R
            a12 = -inv_R
            a21 = -inv_R
            a22 = Ci_cell/dt + inv_R + params["h_out"]*P_out + k_rad_vec[i]
            rhs1 = (Cw_cell/dt)*Tw[i] + h_in[i]*P_in*Tg_new[i]
            rhs2 = (Ci_cell/dt)*Ti[i] + params["h_out"]*P_out*params["Tamb"] + params["eps_rad"]*sigma*P_out*((params["Tamb"]**4) - (Ti[i]**4)) + k_rad_vec[i]*Ti[i]
            det = a11*a22 - a12*a21
            Tw_new[i] = (rhs1*a22 - a12*rhs2)/det
            Ti_new[i] = (a11*rhs2 - rhs1*a21)/det
        # Diffusion with cached solver if available
        if _a_w is not None and _a_i is not None:
            Tw_new = _cn_solve_with_cache(Tw_new, lam_w, params["theta_cond"], _a_w, _cprime_w, _inv_w)
            Ti_new = _cn_solve_with_cache(Ti_new, lam_i, params["theta_cond"], _a_i, _cprime_i, _inv_i)
        else:
            Tw_new = diffuse_axial_CN(Tw_new, aw, dt, dx, params["theta_cond"])
            Ti_new = diffuse_axial_CN(Ti_new, ai, dt, dx, params["theta_cond"])
    Tg = Tg_new; Tw = Tw_new; Ti = Ti_new

    t += dt
    n += 1
    tracker.log_if_needed(t, n)

# ensure final frame is captured
if _frame_idx < _nframes:
    Tw_hist[_frame_idx, :] = Tw
    Tg_hist[_frame_idx, :] = Tg
    Ti_hist[_frame_idx, :] = Ti
    times[_frame_idx] = t
    _frame_idx += 1

total_wall = tracker.finalize(t, n, OUTDIR)
print(f"\n=== Simulation complete ===")
print(f"Simulated time: {t:.2f} s ({_fmt_hms(t)}) over {n} steps")
print(f"Total wall-clock time: {total_wall:.2f} s ({total_wall/60:.2f} min)")
print(f"Average simulation speed: {t/total_wall:.2f}× real-time")

# trim unused rows
Tw_hist = Tw_hist[:_frame_idx, :]
Tg_hist = Tg_hist[:_frame_idx, :]
Ti_hist = Ti_hist[:_frame_idx, :]
times   = times[:_frame_idx]


# Plot and save figures (modularized)
plot_heatmaps(x, times, Tw_hist, Tg_hist, Ti_hist, OUTDIR, params)
plot_profiles(x, times, Tw_hist, Tg_hist, Ti_hist, OUTDIR, params)

# --- Persist arrays and concise CSV summary (modularized) ---
save_arrays_and_csv(OUTDIR, x, times, Tw_hist, Tg_hist, Ti_hist, Nx)
logging.info("Run complete. Outputs saved to: %s", OUTDIR)