from Pipe_Sim_V4 import main

res = main(
    {"Tin": 1000.0, "t_end": 3600.0},
    stop_at_Tg_outlet=950.0,
    stop_dir="ge",             # explicitly rising to ≥ target
    max_sim_time=4*3600.0,       # don’t run past 1200 s
)
print(res.stop_reason, res.Tg_outlet_final)