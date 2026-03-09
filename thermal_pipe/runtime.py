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
"""Runtime tracking and snapshot scheduling utilities."""

import logging
import time
from pathlib import Path

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
