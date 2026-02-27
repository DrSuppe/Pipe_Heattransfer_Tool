# Rerun with smaller grid and capped frames to keep execution quick in this environment.
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
import datetime

# --- Output directory and logging setup ---
def _make_run_dir(prefix="runs"):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(prefix) / f"run_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

OUTDIR = _make_run_dir()
logging.basicConfig(
    filename=str(OUTDIR / "run.log"),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logging.info("Run directory: %s", OUTDIR)

params = {
    "L": 65, "Di": 0.13, "t_wall": 0.018, "t_ins": 0.15, "Nx": 2500,
    "p": 5.0e6, "m_dot": 1, "Tin": 1000.0,
    "T_init_wall": 300.0, "T_init_ins": 300.0, "Tamb": 300.0,
    "rho_w": 8220.0, "cp_w": 564.0, "k_w": 21.9,
    "rho_i": 128.0, "cp_i": 1246, "k_i": 0.09,
    "cp_g": 1005.0, "k_g": 0.028, "mu_g": 2.0e-5, "Pr": 0.71, "dittus_boelter_n": 0.4,
    "h_out": 8.0, "eps_rad": 0.7,
    "t_end": 8*1800.0, "CFL": 0.6, "theta_cond": 0.5,
    # Performance/stability controls
    "adv_scheme": "semi_lagrangian",  # options: "semi_lagrangian" or "upwind"
    "dt_max": 1.0,                     # hard cap for adaptive time step [s]
    "save_frames": 240                 # reduce I/O and RAM (was 1000)
}
# Save parameters snapshot at start
with open(OUTDIR / "params.json", "w") as _f:
    json.dump(params, _f, indent=2)
logging.info("Saved params.json")

R = 287.058
sigma = 5.670374419e-8
L, Di = params["L"], params["Di"]
r_i = Di/2
r_w_o = r_i + params["t_wall"]
r_ins_o = r_w_o + params["t_ins"]
A_flow = np.pi*(Di**2)/4
P_in = np.pi*Di
P_out = np.pi*2*r_ins_o
Nx = params["Nx"]
x = np.linspace(0,L,Nx)
dx = x[1]-x[0]

Vw_cell = np.pi*(r_w_o**2 - r_i**2)*dx
Vi_cell = np.pi*(r_ins_o**2 - r_w_o**2)*dx
Cw_cell = params["rho_w"]*Vw_cell*params["cp_w"]
Ci_cell = params["rho_i"]*Vi_cell*params["cp_i"]
aw = params["k_w"]/(params["rho_w"]*params["cp_w"])
ai = params["k_i"]/(params["rho_i"]*params["cp_i"])
R_wall = np.log(r_w_o/r_i)/(2*np.pi*params["k_w"]*dx)
R_ins=np.log(r_ins_o/r_w_o)/(2*np.pi*params["k_i"]*dx)
R_Tw_to_Ti = R_wall+R_ins

def compute_h_in(Tg):
    rho = params["p"]/(R*Tg)
    u = params["m_dot"]/(rho*A_flow)
    Re = rho*u*Di/params["mu_g"]
    Re = np.maximum(Re,1.0)
    Nu = 0.023*(Re**0.8)*(params["Pr"]**params["dittus_boelter_n"])

    return Nu*params["k_g"]/Di, u

# --- Crank–Nicolson axial diffusion (Numba-accelerated with safe fallback) ---
try:
    from numba import njit

    @njit(cache=True, fastmath=True)
    def _diffuse_axial_CN_numba(T, alpha, dt, dx, theta):
        if alpha <= 0.0:
            return T
        n = T.shape[0]
        lam = alpha * dt / (dx * dx)
        a = np.zeros(n)
        b = np.ones(n)
        c = np.zeros(n)
        RHS = np.empty(n)
        L_T = np.empty(n)

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


# --- Semi-Lagrangian advection and snapshot scheduling ---
def advect_semi_lagrangian(T_old, u, dt, x, Tin):
    """Backward-characteristic semi-Lagrangian advection for positive u.
    For each Eulerian point x_i, sample T at x_i - u_i*dt; if it is < 0, use Tin.
    """
    x_depart = x - u*dt
    # np.interp handles vectorized sampling; left=Tin applies for x<0
    T_adv = np.interp(x_depart, x, T_old, left=Tin, right=T_old[-1])
    return T_adv

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

Tg = np.full(Nx, params["Tin"])
Tw = np.full(Nx, params["T_init_wall"])
Ti = np.full(Nx, params["T_init_ins"])
h0, u0 = compute_h_in(np.full(Nx, params["Tin"]))
dt_cfl = params["CFL"]*dx/max(u0.max(),1e-6)
tau_g = ( (params["p"]/(R*params["Tin"]))*A_flow*dx*params["cp_g"] )/((h0*P_in).max()+1e-9)
tau_wi = Cw_cell.min()/(1.0/R_Tw_to_Ti)
dt_src = 0.3*min(tau_g, tau_wi, 1e9)
dt = float(min(dt_cfl, dt_src, 0.25))

t = 0.0
t_end = params["t_end"]
saver = SnapshotScheduler(t_end, params["save_frames"])
Tw_hist=[]; Tg_hist=[]; Ti_hist=[]; times=[]
n = 0

while t < t_end - 1e-12:

    if saver.should_save(t):
        Tw_hist.append(Tw.copy()); Tg_hist.append(Tg.copy()); Ti_hist.append(Ti.copy()); times.append(t)
        saver.mark_saved()

    h_in, u = compute_h_in(Tg)
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
    if t + dt > t_end:
        dt = t_end - t

    # Gas advection: scheme switch
    if params["adv_scheme"] == "semi_lagrangian":
        # Operator-split: advect, then apply source
        Tg_adv = advect_semi_lagrangian(Tg, u, dt, x, params["Tin"])
        q_gw = h_in*P_in*(Tw - Tg_adv)
        Tg_new = Tg_adv + dt * ( q_gw / np.maximum(Cg_cell,1e-12) )
    else:
        Tg_up = np.roll(Tg,1); Tg_up[0] = params["Tin"]
        adv = -u*(Tg - Tg_up)/dx
        q_gw = h_in*P_in*(Tw - Tg)
        dTg_dt = adv + q_gw/np.maximum(Cg_cell,1e-12)
        Tg_new = Tg + dt*dTg_dt

    q_wi = (Ti - Tw)/np.maximum(R_Tw_to_Ti,1e-16)
    dTw_dt = ( h_in*P_in*(Tg - Tw) + q_wi )/np.maximum(Cw_cell,1e-12)
    Tw_new = Tw + dt*dTw_dt

    q_iw = (Tw - Ti)/np.maximum(R_Tw_to_Ti,1e-16)
    q_out = params["h_out"]*P_out*(params["Tamb"] - Ti) + params["eps_rad"]*sigma*P_out*(params["Tamb"]**4 - Ti**4)
    dTi_dt = ( q_iw + q_out )/np.maximum(Ci_cell,1e-12)
    Ti_new = Ti + dt*dTi_dt

    Tw_new = diffuse_axial_CN(Tw_new, aw, dt, dx, params["theta_cond"])
    Ti_new = diffuse_axial_CN(Ti_new, ai, dt, dx, params["theta_cond"])

    Tg = Tg_new
    Tw = Tw_new
    Ti = Ti_new
    t += dt
    n += 1

Tw_hist.append(Tw.copy())
Tg_hist.append(Tg.copy())
Ti_hist.append(Ti.copy())
times.append(t)
Tw_hist=np.array(Tw_hist)
Tg_hist=np.array(Tg_hist)
Ti_hist=np.array(Ti_hist)
times=np.array(times)

fig,axs=plt.subplots(1,3,figsize=(15,4))
fig.suptitle(f"Heatmaps — m_dot={params['m_dot']} kg/s, Tin={params['Tin']} K")
im0=axs[0].imshow(Tw_hist,aspect='auto',extent=[x[0],x[-1],times[-1],times[0]]);axs[0].set_title('Wall Tw(x,t) [K]')
axs[0].set_xlabel('x [m]');axs[0].set_ylabel('time [s]');plt.colorbar(im0,ax=axs[0],label='K')

im1=axs[1].imshow(Tg_hist,aspect='auto',extent=[x[0],x[-1],times[-1],times[0]]);axs[1].set_title('Gas Tg(x,t) [K]')
axs[1].set_xlabel('x [m]');axs[1].set_ylabel('time [s]');plt.colorbar(im1,ax=axs[1],label='K')

im2=axs[2].imshow(Ti_hist,aspect='auto',extent=[x[0],x[-1],times[-1],times[0]]);axs[2].set_title('Insulation Ti(x,t) [K]')
axs[2].set_xlabel('x [m]');axs[2].set_ylabel('time [s]');plt.colorbar(im2,ax=axs[2],label='K')

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUTDIR / "heatmaps.png", dpi=200)
plt.show()
logging.info("Saved heatmaps.png")


plt.figure(figsize=(10,4))
# Select at most 30 evenly spaced time indices
nmax = 30  # ~1.5x density vs 20
if times.size > 0:
    idx = np.linspace(0, times.size - 1, min(nmax, times.size), dtype=int)
else:
    idx = np.array([], dtype=int)
for i in idx:
    plt.plot(x, Tw_hist[i], label=f"Tw {times[i]:.0f}s")
for i in idx:
    plt.plot(x, Tg_hist[i], '--', label=f"Tg {times[i]:.0f}s")
for i in idx:
    plt.plot(x, Ti_hist[i], ':', label=f"Ti {times[i]:.0f}s")

plt.xlabel('x [m]')
plt.ylabel('Temperature [K]')
plt.title(f"Profiles over time — m_dot={params['m_dot']} kg/s, Tin={params['Tin']} K")
# Place legend outside if many entries; if too many, show style-only legend (Tw/Tg/Ti)
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

# --- Persist arrays and concise CSV summary ---
np.savez_compressed(OUTDIR / "fields.npz", x=x, times=times, Tw=Tw_hist, Tg=Tg_hist, Ti=Ti_hist)
logging.info("Saved fields.npz")

# Summary CSV: outlet/inlet/midpoint temperatures over time
mid_idx = Nx // 2
summary = np.column_stack([
    times,
    Tg_hist[:, -1],
    Tw_hist[:, 0], Tw_hist[:, mid_idx], Tw_hist[:, -1],
    Ti_hist[:, 0], Ti_hist[:, mid_idx], Ti_hist[:, -1],
])
header = "time_s,Tg_outlet_K,Tw_inlet_K,Tw_mid_K,Tw_outlet_K,Ti_inlet_K,Ti_mid_K,Ti_outlet_K"
np.savetxt(OUTDIR / "summary.csv", summary, delimiter=",", header=header, comments="")
logging.info("Saved summary.csv")

print("Saved times [s]:", np.round(times,1))
print("Final outlet gas T [K]:", float(Tg_hist[-1,-1]))
print("Final Tw inlet/mid/outlet [K]:", float(Tw_hist[-1,0]), float(Tw_hist[-1,Nx//2]), float(Tw_hist[-1,-1]))
print("Final Ti inlet/mid/outlet [K]:", float(Ti_hist[-1,0]), float(Ti_hist[-1,Nx//2]), float(Ti_hist[-1,-1]))
print(f"Outputs saved to: {OUTDIR}")
logging.info("Run complete. Outputs saved to: %s", OUTDIR)