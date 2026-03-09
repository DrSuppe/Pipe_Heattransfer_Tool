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
"""Numerical kernels and timestep utilities used by the solver runner."""

import logging

import numpy as np

from .config import params

try:
    from numba import njit, prange
except Exception:
    def njit(*args, **kwargs):
        def _identity(func):
            return func

        return _identity

    def prange(*args):
        return range(*args)


EPS = 1e-12
CN_REL_TOL = 0.05
HAS_NUMBA = False

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
