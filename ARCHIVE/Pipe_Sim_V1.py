# Pipe_Sim_V1.py

import numpy as np
import matplotlib.pyplot as plt

# Parameters dictionary
params = {
    # Geometry
    "L": 1.0,          # pipe length [m]
    "D": 0.05,         # pipe diameter [m]
    "dx": 0.01,        # spatial step [m]
    "Nx": 101,         # number of cells

    # Fluid properties
    "rho_g": 1.2,      # gas density [kg/m3]
    "cp_g": 1005.0,    # gas specific heat [J/kg/K]
    "mu_g": 1.8e-5,    # gas dynamic viscosity [Pa.s]
    "k_g": 0.025,      # gas thermal conductivity [W/m/K]

    # Wall properties
    "rho_w": 7800.0,   # wall density [kg/m3]
    "cp_w": 500.0,     # wall specific heat [J/kg/K]
    "k_w": 15.0,       # wall thermal conductivity [W/m/K]
    "thickness": 0.005,# wall thickness [m]

    # Flow conditions
    "u": 1.0,          # gas velocity [m/s]
    "Tin": 400.0,      # inlet gas temperature [K]
    "Tw_init": 300.0,  # initial wall temperature [K]

    # Correlation toggle
    "use_DittusBoelter": True,

    # Physics toggles
    "enable_wall_axial_cond": True,  # include axial wall conduction between cells
    "theta_cond": 0.5,               # 0.5 = Crank–Nicolson; 1 = fully implicit; 0 = explicit

    # Numerics
    "dt": 10,         # time step [s]
    "t_end": 2000.0,     # end time [s]

    # Radiation toggle
    "radiation": False,
}

# Derived parameters
L = params["L"]
D = params["D"]
dx = params["dx"]
Nx = params["Nx"]
Nt = int(params["t_end"] / params["dt"])
dt = params["dt"]
u = params["u"]
rho_g = params["rho_g"]
cp_g = params["cp_g"]
rho_w = params["rho_w"]
cp_w = params["cp_w"]
k_w = params["k_w"]
thickness = params["thickness"]
Tw_init = params["Tw_init"]
Tin = params["Tin"]

x = np.linspace(0, L, Nx)

# Initial conditions
Tg = np.ones(Nx) * Tin
Tw = np.ones(Nx) * Tw_init

# Precompute wall masses per cell
Aw = np.pi * D * dx
Vw = Aw * thickness
Cw_cell = rho_w * cp_w * Vw

# Helper: implicit axial conduction step for wall temperature using CN/implicit scheme
def diffuse_axial_CN(T, alpha, dt, dx, theta):
    """
    Advance T by dt for 1D diffusion: T_t = alpha * T_xx, with CN/implicit blend `theta`.
    Neumann BC at both ends (dT/dx = 0). Returns new T.
    """
    n = T.size
    lam = alpha * dt / (dx*dx)
    # Build tri-diagonal for (I - theta*lam*L)
    a = np.zeros(n)  # sub-diagonal
    b = np.ones(n)   # main diagonal
    c = np.zeros(n)  # super-diagonal

    # Interior nodes
    for i in range(1, n-1):
        a[i] = -theta * lam
        b[i] = 1.0 + 2.0 * theta * lam
        c[i] = -theta * lam
    # Neumann BCs via mirrored ghost: dT/dx = 0 => L*T @ 0 = (T1 - T0)*2, same for last
    b[0] = 1.0 + 2.0 * theta * lam
    c[0] = -2.0 * theta * lam
    a[-1] = -2.0 * theta * lam
    b[-1] = 1.0 + 2.0 * theta * lam

    # Right-hand side (I + (1-theta)*lam*L) T^*
    RHS = T.copy()
    # Apply L operator with Neumann ends
    L_T = np.empty_like(T)
    L_T[1:-1] = T[2:] - 2.0*T[1:-1] + T[:-2]
    L_T[0]    = 2.0*(T[1] - T[0])
    L_T[-1]   = 2.0*(T[-2] - T[-1])
    RHS += (1.0 - theta) * lam * L_T

    # Thomas algorithm
    # Forward sweep
    for i in range(1, n):
        m = a[i] / b[i-1]
        b[i] -= m * c[i-1]
        RHS[i] -= m * RHS[i-1]
    # Back substitution
    Tnew = np.empty_like(T)
    Tnew[-1] = RHS[-1] / b[-1]
    for i in range(n-2, -1, -1):
        Tnew[i] = (RHS[i] - c[i] * Tnew[i+1]) / b[i]
    return Tnew

sigma = 5.670374419e-8

aw = params["k_w"] / (params["rho_w"] * params["cp_w"])  # wall thermal diffusivity [m^2/s]
enable_wall_axial_cond = params.get("enable_wall_axial_cond", False)
theta_cond = params.get("theta_cond", 0.5)

def compute_h_in(Tg_local):
    # Dummy placeholder for heat transfer coefficient and related values
    h_in = 10.0  # W/m2/K
    P_in = np.pi * D
    A_in = P_in * dx
    return h_in, P_in, A_in, None

# History arrays for plotting
Tg_hist = []
Tw_hist = []
times = []

for n in range(Nt):
    t = n * dt

    # Compute heat transfer coefficient and geometry
    h_in, P_in, A_in, _ = compute_h_in(Tg)

    # Gas temperature update (simple convection + heat exchange)
    dTg_dx = np.zeros_like(Tg)
    dTg_dx[1:] = (Tg[1:] - Tg[:-1]) / dx
    dTg_dx[0] = (Tg[1] - Tg[0]) / dx

    q_gw = h_in * P_in * (Tw - Tg)  # heat flux W/m along pipe

    dTg_dt = -u * dTg_dx + q_gw / (rho_g * cp_g * np.pi * (D/2)**2)

    Tg_new = Tg + dt * dTg_dt

    # Wall temperature update (heat conduction + heat exchange)
    dTw_dt = q_gw / Cw_cell

    Tw_new = Tw + dt * dTw_dt

    # Optional axial conduction coupling between wall cells (stabilized CN/implicit)
    if enable_wall_axial_cond:
        Tw_new = diffuse_axial_CN(Tw_new, aw, dt, dx, theta_cond)

    Tg = Tg_new
    Tw = Tw_new

    if n % 10 == 0:
        Tg_hist.append(Tg.copy())
        Tw_hist.append(Tw.copy())
        times.append(t)

# Convert history to arrays
Tg_hist = np.array(Tg_hist)
Tw_hist = np.array(Tw_hist)
times = np.array(times)

# Plot 1: Space–time (imshow) for wall temperature
plt.figure(figsize=(8, 4))
plt.imshow(Tw_hist, aspect='auto', extent=[x[0], x[-1], times[-1], times[0]])
plt.xlabel('x [m]')
plt.ylabel('time [s]')
plt.title('Wall temperature Tw(x,t) [K]')
cbar = plt.colorbar()
cbar.set_label('K')
plt.tight_layout()
plt.show()

# Plot 1b: Space–time (imshow) for gas temperature
plt.figure(figsize=(8, 4))
plt.imshow(Tg_hist, aspect='auto', extent=[x[0], x[-1], times[-1], times[0]])
plt.xlabel('x [m]')
plt.ylabel('time [s]')
plt.title('Gas temperature Tg(x,t) [K]')
cbar = plt.colorbar()
cbar.set_label('K')
plt.tight_layout()
plt.show()

# Plot 2: Snapshots of wall temperature
plt.figure()
for i in range(len(times)):
    plt.plot(x, Tw_hist[i], label=f't={times[i]:.1f}s')
plt.xlabel('x [m]')
plt.ylabel('Wall Temperature [K]')
plt.title('Wall temperature profiles over time')
plt.legend()
plt.show()

# --- Diagnostic: energy balance consistency (rough check)
# Computes net convective heat from gas to wall over the whole pipe at the last snapshot
# (Perfect balance is not expected here if h_out or radiation are non-zero.)
h_in_last, _, _, _ = compute_h_in(Tg_hist[-1])
q_gw_axial = h_in_last * P_in * (Tw_hist[-1] - Tg_hist[-1])  # W/m along x
Q_gw_total = np.trapz(q_gw_axial, x)  # W
print(f"Approx. convective heat from gas to wall at final snapshot [W]: {Q_gw_total:.3e}")

print("Time snapshots saved:", times)
