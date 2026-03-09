
import numpy as np
import sys
import os

# Add the project dir to path to import the solver
sys.path.append(os.getcwd())
from Pipe_Sim_V4 import run_simulation

def run_test(nx):
    params = {
        "Nx": nx,
        "L": 65.0,
        "t_end": 100.0,
        "m_dot": 2.5,
        "p": 5.0e6,
        "Tin": 1100.0,
        "Tin_ramp_s": 0.0,
        "T_init_gas": 300.0,
        "T_init_wall": 300.0,
        "progress": "none",
        "save_frames": 2,
        "parallel": False
    }
    res = run_simulation(params)
    return res.Tg_outlet_final

print("--- Nx Sensitivity Test (Current Code) ---")
# If the physics is correct, changing Nx should only affect accuracy, not the bulk result.
t_100 = run_test(100)
t_500 = run_test(500)
t_2500 = run_test(2500)

print(f"Final Tg_out at Nx=100:  {t_100:.2f} K")
print(f"Final Tg_out at Nx=500:  {t_500:.2f} K")
print(f"Final Tg_out at Nx=2500: {t_2500:.2f} K")

if abs(t_100 - t_2500) > 10.0:
    print(f"\nRESULT: Large sensitivity to Nx detected (Difference: {abs(t_100 - t_2500):.1f} K).")
    print("This confirms that the heat transfer rate is being scaled by Nx (1/dx).")
else:
    print("\nRESULT: Nx sensitivity is low. Physics is likely correctly scaled or compensated.")
