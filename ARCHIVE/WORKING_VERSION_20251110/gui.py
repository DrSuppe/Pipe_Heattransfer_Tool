# gui.py
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import re

# --- Unit helpers ---
FT2M = 0.3048
IN2M = 0.0254
M2FT = 1.0 / FT2M
M2IN = 1.0 / IN2M
PSI2PA = 6894.757293168
PA2PSI = 1.0 / PSI2PA
W_M2K__TO__BTU_HR_FT2_F = 0.1761101838
BTU_HR_FT2_F__TO__W_M2K = 1.0 / W_M2K__TO__BTU_HR_FT2_F
KG_S__TO__LBM_S = 2.20462262185
LBM_S__TO__KG_S = 1.0 / KG_S__TO__LBM_S

def K_to_F(K: float) -> float:
    return (K - 273.15) * 9.0/5.0 + 32.0

def F_to_K(F: float) -> float:
    return (F - 32.0) * 5.0/9.0 + 273.15

SAFE_EXPR_RE = re.compile(r'^[0-9eE\.+\-*/()\s]+$')

def parse_num(text: str) -> float:
    s = str(text).strip()
    try:
        return float(s)
    except Exception:
        if SAFE_EXPR_RE.match(s):
            return float(eval(s, {"__builtins__": None}, {}))
        raise

from sim_controller import HardwareConfig, RunInputs, RunSpec, run_once

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Thermal Pipe Simulator (starter GUI)")
        self.geometry("640x520")

        # Units: "SI" or "Imperial"
        self.var_units = tk.StringVar(value="SI")
        self._units_last = "SI"

        # Unit label StringVars
        self.lbl_L_unit = tk.StringVar(value="m")
        self.lbl_Di_unit = tk.StringVar(value="m")
        self.lbl_tw_unit = tk.StringVar(value="m")
        self.lbl_ti_unit = tk.StringVar(value="m")
        self.lbl_hout_unit = tk.StringVar(value="W/m²K")
        self.lbl_eps_unit = tk.StringVar(value="–")
        self.lbl_p_unit = tk.StringVar(value="Pa")
        self.lbl_Tin_unit = tk.StringVar(value="K")
        self.lbl_mdot_unit = tk.StringVar(value="kg/s")
        self.lbl_target_unit = tk.StringVar(value="K")

        self.save_dir: Path | None = None

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        # --- Hardware tab ---
        self.hw_frame = ttk.Frame(nb)
        nb.add(self.hw_frame, text="Hardware")

        self.var_L      = tk.DoubleVar(value=65.0)
        self.var_Di     = tk.DoubleVar(value=0.13)
        self.var_tw     = tk.DoubleVar(value=0.018)
        self.var_ti     = tk.DoubleVar(value=0.15)
        self.var_hout   = tk.DoubleVar(value=8.0)
        self.var_eps    = tk.DoubleVar(value=0.7)

        grid = ttk.Frame(self.hw_frame)
        grid.pack(padx=10, pady=10, fill="x")
        # Show units header at right
        ttk.Label(grid, textvariable=self.var_units, foreground="#555").grid(row=0, column=3, sticky="e", padx=4)
        def row(r, label, var, unitvar):
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky="w", padx=4, pady=4)
            ttk.Entry(grid, textvariable=var, width=12).grid(row=r, column=1, sticky="w")
            ttk.Label(grid, textvariable=unitvar).grid(row=r, column=2, sticky="w")
        row(0, "Length L",     self.var_L,    self.lbl_L_unit)
        row(1, "Inner Dia Di", self.var_Di,   self.lbl_Di_unit)
        row(2, "Wall t_wall",  self.var_tw,   self.lbl_tw_unit)
        row(3, "Ins t_ins",    self.var_ti,   self.lbl_ti_unit)
        row(4, "h_out",        self.var_hout, self.lbl_hout_unit)
        row(5, "ε_rad",        self.var_eps,  self.lbl_eps_unit)

        # --- Run tab ---
        self.run_frame = ttk.Frame(nb)
        nb.add(self.run_frame, text="Run")

        self.var_p     = tk.DoubleVar(value=5.0e6)
        self.var_Tin   = tk.DoubleVar(value=1000.0)
        self.var_mdot  = tk.DoubleVar(value=1.0)

        self.var_mode  = tk.StringVar(value="time")  # "time" or "target"
        self.var_tend  = tk.StringVar(value="1800.0")
        self.var_target= tk.StringVar(value="950.0")
        self.var_stopdir = tk.StringVar(value="auto")  # "auto", "le", "ge"

        self.var_makeplots  = tk.BooleanVar(value=True)
        self.var_saveresults= tk.BooleanVar(value=True)

        frm = ttk.Frame(self.run_frame)
        frm.pack(padx=10, pady=10, fill="x")

        # Units selector
        ttk.Label(frm, text="Units").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        units_combo = ttk.Combobox(frm, textvariable=self.var_units, values=("SI","Imperial"), width=12, state="readonly")
        units_combo.grid(row=0, column=1, sticky="w")
        units_combo.bind("<<ComboboxSelected>>", lambda e: self._on_units_changed())

        # Basic inputs
        def rrow(r, label, var, unitvar, width=16):
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", padx=4, pady=4)
            ttk.Entry(frm, textvariable=var, width=width).grid(row=r, column=1, sticky="w")
            ttk.Label(frm, textvariable=unitvar).grid(row=r, column=2, sticky="w")

        rrow(1, "Pressure p", self.var_p,   self.lbl_p_unit)
        rrow(2, "Inlet Tin",  self.var_Tin, self.lbl_Tin_unit)
        rrow(3, "m_dot",      self.var_mdot,self.lbl_mdot_unit)

        # Control mode
        ttk.Label(frm, text="Control mode").grid(row=4, column=0, sticky="w", padx=4, pady=(12,4))
        modes = ttk.Frame(frm); modes.grid(row=4, column=1, columnspan=2, sticky="w", pady=(12,4))
        ttk.Radiobutton(modes, text="Fixed time",   variable=self.var_mode, value="time",   command=self._update_mode).pack(side="left")
        ttk.Radiobutton(modes, text="Outlet target",variable=self.var_mode, value="target", command=self._update_mode).pack(side="left")

        # Time / Target controls
        self.lbl_time = ttk.Label(frm, text="t_end")
        self.lbl_time.grid(row=5, column=0, sticky="w", padx=4, pady=4)
        self.ent_time = ttk.Entry(frm, textvariable=self.var_tend, width=16)
        self.ent_time.grid(row=5, column=1, sticky="w")
        ttk.Label(frm, text="s").grid(row=5, column=2, sticky="w")

        self.lbl_target = ttk.Label(frm, text="Outlet target")
        self.ent_target = ttk.Entry(frm, textvariable=self.var_target, width=16)
        self.lbl_target_u = ttk.Label(frm, textvariable=self.lbl_target_unit)

        # Stop direction
        ttk.Label(frm, text="Stop direction").grid(row=7, column=0, sticky="w", padx=4, pady=4)
        combo = ttk.Combobox(frm, textvariable=self.var_stopdir, values=("auto","le","ge"), width=12, state="readonly")
        combo.grid(row=7, column=1, sticky="w")
        combo.current(0)

        # Options
        opts = ttk.Frame(frm); opts.grid(row=8, column=0, columnspan=3, sticky="w", pady=(8,4))
        ttk.Checkbutton(opts, text="Make plots", variable=self.var_makeplots).pack(side="left", padx=(0,12))
        ttk.Checkbutton(opts, text="Save results", variable=self.var_saveresults).pack(side="left")

        # Save dir chooser
        pick = ttk.Frame(frm); pick.grid(row=9, column=0, columnspan=3, sticky="w", pady=(8,4))
        ttk.Button(pick, text="Choose save folder…", command=self._choose_dir).pack(side="left")
        self.lbl_dir = ttk.Label(pick, text="(auto)")
        self.lbl_dir.pack(side="left", padx=8)

        # Run
        runbar = ttk.Frame(self.run_frame); runbar.pack(fill="x", padx=10, pady=6)
        self.btn_run = ttk.Button(runbar, text="Run", command=self._run_async)
        self.btn_run.pack(side="left")
        self.lbl_status = ttk.Label(runbar, text="")
        self.lbl_status.pack(side="left", padx=12)

        self._update_mode()  # set initial widget visibility

    def _update_mode(self):
        mode = self.var_mode.get()
        if mode == "time":
            self.lbl_time.config(text="t_end")
            # hide target row
            self.lbl_target.grid_forget()
            self.ent_target.grid_forget()
            self.lbl_target_u.grid_forget()
            # show time row
            self.lbl_time.grid(row=5, column=0, sticky="w", padx=4, pady=4)
            self.ent_time.grid(row=5, column=1, sticky="w")
        else:
            self.lbl_time.config(text="t_max")
            # show target
            self.lbl_target.grid(row=6, column=0, sticky="w", padx=4, pady=4)
            self.ent_target.grid(row=6, column=1, sticky="w")
            self.lbl_target_u.grid(row=6, column=2, sticky="w")
            # ensure time row is shown (used as cap)
            self.lbl_time.grid(row=5, column=0, sticky="w", padx=4, pady=4)
            self.ent_time.grid(row=5, column=1, sticky="w")

    def _on_units_changed(self):
        # Convert displayed values when unit system toggles
        try:
            old = getattr(self, "_units_last", "SI")
            new = self.var_units.get()
            if old == new:
                return
            # Hardware
            if old == "SI" and new == "Imperial":
                self.var_L.set(self.var_L.get() * M2FT)
                self.var_Di.set(self.var_Di.get() * M2IN)
                self.var_tw.set(self.var_tw.get() * M2IN)
                self.var_ti.set(self.var_ti.get() * M2IN)
                self.var_hout.set(self.var_hout.get() * W_M2K__TO__BTU_HR_FT2_F)
            elif old == "Imperial" and new == "SI":
                self.var_L.set(self.var_L.get() * FT2M)
                self.var_Di.set(self.var_Di.get() * IN2M)
                self.var_tw.set(self.var_tw.get() * IN2M)
                self.var_ti.set(self.var_ti.get() * IN2M)
                self.var_hout.set(self.var_hout.get() * BTU_HR_FT2_F__TO__W_M2K)
            # Run inputs
            if old == "SI" and new == "Imperial":
                self.var_p.set(self.var_p.get() * PA2PSI)
                self.var_Tin.set(K_to_F(self.var_Tin.get()))
                self.var_mdot.set(self.var_mdot.get() * KG_S__TO__LBM_S)
                # target and t_end/t_max are strings; convert if numeric-looking
                try: self.var_target.set(f"{K_to_F(parse_num(self.var_target.get())):.6g}")
                except Exception: pass
            elif old == "Imperial" and new == "SI":
                self.var_p.set(self.var_p.get() * PSI2PA)
                self.var_Tin.set(F_to_K(self.var_Tin.get()))
                self.var_mdot.set(self.var_mdot.get() * LBM_S__TO__KG_S)
                try: self.var_target.set(f"{F_to_K(parse_num(self.var_target.get())):.6g}")
                except Exception: pass
            # Update unit labels
            if new == "Imperial":
                self.lbl_L_unit.set("ft")
                self.lbl_Di_unit.set("in")
                self.lbl_tw_unit.set("in")
                self.lbl_ti_unit.set("in")
                self.lbl_hout_unit.set("BTU/(hr·ft²·°F)")
                self.lbl_eps_unit.set("–")
                self.lbl_p_unit.set("psi")
                self.lbl_Tin_unit.set("°F")
                self.lbl_mdot_unit.set("lbm/s")
                self.lbl_target_unit.set("°F")
            else:
                self.lbl_L_unit.set("m")
                self.lbl_Di_unit.set("m")
                self.lbl_tw_unit.set("m")
                self.lbl_ti_unit.set("m")
                self.lbl_hout_unit.set("W/m²K")
                self.lbl_eps_unit.set("–")
                self.lbl_p_unit.set("Pa")
                self.lbl_Tin_unit.set("K")
                self.lbl_mdot_unit.set("kg/s")
                self.lbl_target_unit.set("K")
            self._units_last = new
        except Exception as e:
            messagebox.showwarning("Units", f"Could not convert values: {e}")

    def _choose_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.save_dir = Path(path)
            self.lbl_dir.config(text=str(self.save_dir))
        else:
            self.save_dir = None
            self.lbl_dir.config(text="(auto)")

    def _collect_spec(self) -> RunSpec:
        units = self.var_units.get()
        # Parse numbers (allow simple expressions in t_end/target)
        try:
            t_end_val = parse_num(self.var_tend.get())
        except Exception:
            raise ValueError("t_end / t_max must be a number or simple expression (e.g., 5*3600)")
        mode = self.var_mode.get()
        try:
            target_val = parse_num(self.var_target.get()) if mode == "target" else None
        except Exception:
            raise ValueError("Outlet target must be a number or simple expression")

        # Hardware inputs (display units)
        L = self.var_L.get()
        Di = self.var_Di.get()
        t_wall = self.var_tw.get()
        t_ins = self.var_ti.get()
        h_out = self.var_hout.get()
        eps_rad = self.var_eps.get()

        # Run inputs (display units)
        p = self.var_p.get()
        Tin = self.var_Tin.get()
        m_dot = self.var_mdot.get()

        # Convert to SI if needed
        if units == "Imperial":
            L *= FT2M
            Di *= IN2M
            t_wall *= IN2M
            t_ins *= IN2M
            h_out *= BTU_HR_FT2_F__TO__W_M2K
            p *= PSI2PA
            Tin = F_to_K(Tin)
            m_dot *= LBM_S__TO__KG_S
            if target_val is not None:
                target_val = F_to_K(target_val)
        # (If SI, leave as-is; t_end is seconds in both.)

        hw = HardwareConfig(L=L, Di=Di, t_wall=t_wall, t_ins=t_ins, h_out=h_out, eps_rad=eps_rad)
        run = RunInputs(
            p=p, Tin=Tin, m_dot=m_dot,
            mode=mode,
            t_end=float(t_end_val),
            Tg_out_target=(float(target_val) if mode == "target" else None),
            stop_dir=(None if self.var_stopdir.get()=="auto" else self.var_stopdir.get()),
        )
        return RunSpec(
            hardware=hw,
            run=run,
            save_dir=self.save_dir,
            make_plots=self.var_makeplots.get(),
            save_results=self.var_saveresults.get(),
        )

    def _run_async(self):
        spec = self._collect_spec()
        self.btn_run.config(state="disabled")
        self.lbl_status.config(text="Running…")
        def worker():
            try:
                res = run_once(spec)
                msg = f"Done: {res.stop_reason}; Tg_out={res.Tg_outlet_final:.2f} K"
                if res.outdir:
                    msg += f" | saved → {res.outdir}"
                self.after(0, lambda: self.lbl_status.config(text=msg))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.after(0, lambda: self.btn_run.config(state="normal"))
        threading.Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    App().mainloop()