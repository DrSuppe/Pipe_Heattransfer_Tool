# Thermal Pipe Heat Transfer Simulator

This project estimates transient heating/cooling of flowing gas in a pipe with wall + optional insulation, including:

- axial advection of gas temperature,
- radial heat exchange gas -> wall -> insulation -> ambient,
- axial conduction in wall and insulation,
- external convection to ambient (auto-estimated) + radiation.

The GUI is designed for rapid first estimates and report-ready visual outputs.

## Current App Entrypoints

- Tkinter legacy UI: `gui.py`
- PyQt6 main UI (recommended): `pyqt6_app.py`
- Solver core: `Pipe_Sim_V4.py`
- Controller/dataclasses glue: `sim_controller.py`
- GitHub CI smoke check: `.github/workflows/compile-smoke.yml`

## Start Here (No Prior Knowledge)

If you are opening this project for the first time, follow this exact flow.

### 1) Install and launch

```bash
python3 -m pip install -r requirements.txt
python3 pyqt6_app.py
```

### 2) Choose a starting preset

- In **Scenario**, keep the default numerical preset (`Balanced`) unless you need faster/slower runs.
- In **Library**, optionally apply a pipe default if your team has saved one.

### 3) Fill only the minimum required inputs

- Pipe geometry: `L`, `Di`, `t_wall`, optional `t_ins`.
- Material: pipe material, optional insulation material.
- Ambient and inlet: `Tamb`, `Tin`, `p`, `m_dot`.
- Run control: fixed-time (`t_end`) or target mode (`Outlet target`).

### 4) Pick your target condition

- Use **Target variable** to choose what "target reached" means:
  - `Gas outlet Tg`
  - `Wall inner outlet Tw(in)`
  - `Wall outer outlet Tw(out)`
  - `Insulation outlet Ti`
- Set the target temperature in the same panel.

### 5) (Optional) enable advanced realism

- Check `use temperature-dependent solids` to use built-in `cp(T)` and `k(T)` tables.
- In **Library -> Temperature-Dependent Tables**, choose material type + material and edit `T`, `cp`, `k` lists directly.
- Add thermal masses if needed (tees/welded attachments) with:
  - count,
  - mass factor,
  - positions,
  - spread.

### 6) Run and inspect

- Click **Run Simulation**.
- Watch **Live View** for evolving profiles and heatmap.
- Use **Run Animation** and slider for time-scrubbing.
- Open **Results** for static report plots and statistics.

### 7) Save evidence

- Enable `Save run artifacts` if you want run folders.
- Use **Export Bundle** for a portable `.zip` with plots, logs, settings, and snapshots.
- Use **Ledger** tab to append selected runs/conditions into long-term history.

### 8) Interpret conservatively

- This tool is a first-estimate engineering simulator.
- Treat stress as a screening indicator, not code qualification.
- Use FEA/piping code checks for final design decisions.

## Core Terms (Plain Language)

- `Tg`: gas temperature in the pipe center flow path.
- `Tw`: wall representative temperature (inner wall node in this model).
- `Ti`: insulation representative temperature (outer layer node in this model).
- `Outlet`: the last axial cell at `x = L`.
- `Heatup time`: simulated time required for selected target variable to cross target temperature.
- `Stress indicator`: rapid estimate combining thermal-gradient effects and optional pressure/restraint terms.
- `Elbow-adjusted`: straight-section stress indicator amplified locally by elbow screening factors.

## Physics Model

The solver is a 1D axial model with lumped radial layers per axial cell.

- State fields: `Tg(x,t)`, `Tw(x,t)`, `Ti(x,t)` (gas, wall, insulation).
- Spatial domain: `x in [0, L]`, discretized with `Nx` cells.
- Gas properties use ideal-gas density from pressure and local temperature [4].

### 1) Gas Energy (Advection + Source)

For each axial cell:

```text
dTg/dt = advection + q_gw / Cg
```

where:

- `advection` is upwind or semi-Lagrangian transport,
- `q_gw = h_in * P_in * (Tw - Tg)`,
- `Cg = rho_g * A_flow * dx * cp_g`.

Internal convection `h_in` uses Gnielinski (with laminar/transition blending in code):

```text
f = (0.79*ln(Re) - 1.64)^(-2)
Nu_turb = (f/8)*(Re-1000)*Pr / [1 + 12.7*(f/8)^(1/2)*(Pr^(2/3)-1)]
h_in = Nu * k_g / Di
```

Implementation notes:

- `Re < 2300`: laminar fallback (`Nu = 3.66`)
- `2300 <= Re < 3000`: linear blend laminar -> turbulent
- `Re >= 3000`: Gnielinski turbulent form

This is consistent with the validity region and practical use of the Gnielinski/Petukhov framework [1,2].

Inlet heater warm-up model:

- The inlet boundary ramps from initial gas temperature (`T_init_gas`) toward heater setpoint `Tin`.
- Default model is **logistic (S-curve)**, normalized to reach setpoint at `t = Tin_ramp_s`.

```text
u = clip(t / Tin_ramp_s, 0, 1)
sig = 1 / (1 + exp(-k*(u-0.5)))
frac = (sig - sig(u=0)) / (sig(u=1) - sig(u=0))
Tin_eff(t) = Tin_start + (Tin_target - Tin_start) * frac
```

Special case:

- If `Tin_ramp_s = 0`, inlet temperature steps immediately to `Tin`.
- Alternative modes are available via `Tin_ramp_model`:
  - `linear`: constant-slope ramp to setpoint,
  - `heater_exp`: first-order heater-style curve (asymptotic-looking, normalized with 99% characteristic scale).

### 2) Wall/Insulation Radial Coupling

Per axial cell, a coupled implicit 2x2 solve is done for `(Tw, Ti)`:

- wall storage and gas-wall transfer,
- conduction resistance wall -> insulation,
- ambient loss from insulation via convection + radiation.

Radiation is linearized per step with:

```text
k_rad ~ 4 * eps * sigma * P_out * Ti^3
```

This linearization is standard for implicit transient radiation coupling around a local operating temperature [4].

### 3) External Ambient Convection (Auto Mode)

`h_out` is now auto-estimated (default) from natural convection over a horizontal cylinder using a Churchill-Chu style correlation [3].

```text
Ra = g * beta * |Ts - Tamb| * D_out^3 / (nu * alpha)
Nu = [0.60 + 0.387*Ra^(1/6) / (1 + (0.559/Pr)^(9/16))^(8/27)]^2
h_out = Nu * k_air / D_out
```

`manual` mode still exists internally for compatibility.

### 4) Axial Conduction in Solids

Wall and insulation axial diffusion are solved with Crank-Nicolson tridiagonal solves [5].

### 5) Temperature-Dependent Solid Properties

Pipe wall and insulation can use temperature-dependent material curves:

- `cp_w(T), k_w(T)` for pipe material.
- `cp_i(T), k_i(T)` for insulation material.

At runtime, the solver interpolates from tabulated values and updates local:

- thermal capacities (`Cw_cell`, `Ci_cell`),
- radial resistances (`R_wall`, `R_ins`),
- effective diffusion parameters used in CN solves.

This improves realism for large temperature excursions where constant properties would bias time constants and heat-loss behavior.

Notes:

- Built-in tables are representative first-estimate data.
- For production work, replace with certified supplier/project data.

### 6) Attached Thermal-Mass Model (Tees / Welded Masses)

The UI supports localized attached thermal masses along the pipe.

- These are treated as added local wall heat capacity of the same material.
- Physical interpretation: this is a **dead-end T-branch approximation** (no branch flow solved), where attached metal adds thermal inertia to the main line.
- You control:
  - number of attachments,
  - mass factor per attachment,
  - positions,
  - spatial spread,
  - optional rough dead-leg sizing fields.

Implementation concept:

```text
Cw_eff(x) = Cw_base(x) * [1 + mass_profile(x)]
```

where `mass_profile(x)` is constructed from user-defined localized peaks.

Detailed meaning of each thermal-mass control:

- `Attached thermal masses`:
  - number of attachment zones added to the model.
  - if set to `0`, no extra thermal mass is applied and `Cw_eff = Cw_base`.
- `Mass factor per attach`:
  - peak local multiplier contribution from one attachment.
  - at the exact attachment center, local wall capacity is approximately `Cw_base * (1 + mass_factor)`.
- `Mass spread (%L)`:
  - smoothing width for each attachment profile along `x`.
  - internally, each attachment is a triangular influence over roughly `±3 * spread_frac * L`; larger spread distributes the same attachment effect over more axial cells.
- `Mass positions`:
  - center locations for attachments.
  - accepts absolute distances (current unit system) or percentages (`20%, 60%`).
  - if blank or count mismatch, the app **auto-distributes** positions from 15% to 85% of line length with equal spacing.

How this influences simulation behavior:

- higher `mass_factor` and/or more attachments increases local thermal inertia, which slows local wall heating/cooling.
- larger `spread` broadens this inertia effect and smooths spatial gradients.
- because gas-wall heat transfer depends on wall temperature, outlet gas transients can lag when significant mass is attached upstream.
- stress indicators may shift in time and space because through-wall and axial gradients evolve differently with added local inertia.

Rough scaling estimate (per attachment, same material, dead-end branch approximation):

```text
mass_factor ~= (D_ratio * L_dead_leg / L_main) / (3 * spread_frac)
```

with:

- `D_ratio = D_dead_leg / D_main`,
- `spread_frac = spread_%L / 100`.

The GUI provides this estimate and can auto-fill `mass_factor`.

### Worked Translation Example: 4 Tees, each dead-end length 12 m

Assume:

- main line length `L_main = 65 m`,
- `4` identical tees with dead-end branches,
- each branch has same material and same nominal diameter as main pipe (`D_ratio = 1.0`),
- choose `spread = 3%L` as a starting point (`spread_frac = 0.03`).

Then per-attachment rough estimate:

```text
mass_factor ~= (1.0 * 12 / 65) / (3 * 0.03) ~= 2.05
```

Recommended GUI entry for this case:

- `Attached thermal masses` = `4`
- `Mass factor per attach` = `2.0` to `2.1` (start at `2.05`)
- `Mass spread` = `3.0 %L`
- `Mass positions`:
  - if tees are truly even: leave blank and use auto-distribution
  - or enter explicit values if known (for example in SI: `10, 25, 40, 55`)
- `Dead-leg length (rough)` = `12 m`
- `Dead-leg diameter ratio` = `1.0`
- click `Apply rough mass factor` to auto-fill factor.

What auto-distribute does for this example:

- with count `4` and blank positions, centers are placed at `15%`, `38.3%`, `61.7%`, `85%` of `L`.
- for `L = 65 m`, this corresponds to approximately `9.75`, `24.92`, `40.08`, `55.25 m`.

## Numerical Scheme

- Gas advection:
  - default: `semi_lagrangian` (stable at larger `dt`),
  - optional: upwind.
- Solid diffusion:
  - Crank-Nicolson with cached factorization.
- Adaptive time step:
  - bounded by source and CFL-like constraints,
  - clamped by `dt_min <= dt <= dt_max`.
- Float precision:
  - configurable (`float32` default for speed, `float64` if needed).

## PyQt6 Interface Design

The UI is workflow-first:

1. Scenario (preset, run mode, stop criteria)
2. Materials/Ambient (pipe material, optional insulation, ambient)
3. Geometry
4. Flow/Inlet
5. Outputs
6. Advanced numerics

Right side includes:

- **Live View** tab:
  - live outlet trend with inlet heater setpoint overlay (`Tin_eff`),
  - live inlet gas (`Tg_in(cell0)`) trend to compare boundary response against setpoint,
  - live axial profile,
  - live heatmap during run,
  - playback controls (`Run Animation`, slider, time indicator).
- **Results** tab:
  - static heatmap,
  - waterfall profiles,
  - outlet-vs-time curves including inlet heater setpoint (`Tin_eff(t)`),
  - inlet tracking curve (`Tg_in(cell0)`) to verify slow heater ramp effect,
  - stress indicator plots (final profile, max-vs-time, stress heatmap),
  - run statistics panel,
  - health/fatigue warning panel.
- **Library** tab:
  - add/update pipe materials (rho, cp, k, E, alpha, nu, emissivity),
  - add/update nominal yield strength (`Sy`) for screening,
  - add/update insulation materials (rho, cp, k),
  - edit temperature-dependent material tables (`T`, `cp`, `k`) for both pipe and insulation materials,
  - save/apply/delete custom pipe presets.
- **Ledger** tab:
  - preview current run-history ledger (`CSV`/`XLSX`),
  - append current configuration row,
  - delete selected ledger rows.
- **README** tab:
  - renders this project README in-app (Markdown when supported by Qt).
- **Simulation Log** tab.

The UI includes a units toggle (SI / Imperial) that affects:

- parameter inputs,
- live/result plot axes and legends,
- status/statistics outputs.

Run modes now include:

- `Fixed time`
- `Outlet target`
- `Heatup-time optimize` (solve for `m_dot` to hit target time within tolerance under stress cap)
- `Stress-limit optimize` (bisection on `m_dot` to satisfy stress cap, then report time-to-target)

Target condition now supports selectable variable:

- gas outlet,
- inner-wall outlet,
- estimated outer-wall outlet,
- insulation outlet.

This allows commissioning/operations to align with whichever thermocouple location is actually available.

Run controls include one-click `Export Bundle` to package run artifacts, settings snapshot, stats/warnings, ledger snapshot, and library snapshots into a `.zip`.

## GUI Parameter Reference (What Each Input Does)

Each parameter below is user-settable in the GUI and includes its simulation effect.

### 1) Scenario

- `Units`: switches SI/Imperial for inputs, plots, and reported values. This is a display/unit-conversion layer and does not change physics itself.
- `Preset`: sets numerical defaults (mesh, frame count, timestep caps) for speed vs fidelity tradeoff.
- `Pipe default`: loads a saved geometry/material operating template to reduce repeated data entry.
- `Asset ID`: tag for the physical line in the ledger so runs can be grouped by equipment.
- `Branch ID`: tag for path/branch context in shared-trunk systems for later filtering.
- `Run mode`: selects fixed-time, target stop, or optimization behavior.
- `t_end / t_max`: upper bound of simulated physical time before forced stop.
- `Outlet target`: numeric target temperature used by target/optimization modes.
- `Target variable`: defines which outlet temperature channel is compared to the target (`Tg`, wall inner, wall outer estimate, or insulation).
- `Stop direction`: forces `<=` or `>=` logic for target mode (or auto-infers from initial condition).
- `Heatup target time`: desired time-to-target used in heatup-time optimization.
- `Heatup tolerance`: allowable deviation around heatup target time for optimization pass/fail.
- `Stress limit`: screening cap for stress-aware optimization modes.

### 2) Materials / Ambient

- `Pipe material`: selects base density/thermal/mechanical properties for the wall.
- `Ambient temp`: external reference temperature driving convective/radiative losses.
- `Surface emissivity`: scales radiative heat transfer from outer surface.
- `Enable insulation`: toggles insulation layer modeling on/off.
- `Insulation material`: selects insulation density and thermal properties.
- `Insulation thickness`: changes radial resistance and insulation thermal storage.
- `Use temperature-dependent cp(T), k(T)`: enables interpolation from material tables; if disabled, scalar cp/k values are used.

### 3) Pipe Geometry

- `Length L`: advection travel distance and total conduction path.
- `Inner diameter Di`: sets flow area, velocity, Reynolds number, and internal convection scaling.
- `Wall thickness`: sets wall thermal mass and radial resistance.
- `Number of elbows`: count of localized elbow screening zones for stress amplification.
- `Elbow SIF factor`: peak elbow stress amplification in the screening model.
- `Elbow positions`: elbow centers along the line (distance or `%`), controlling where amplification is applied.
- `Attached thermal masses`: number of localized dead-end T-like mass attachments.
- `Mass factor per attach`: additional local wall thermal inertia intensity per attachment. At each center, effective wall capacity scales approximately as `Cw_local ~ Cw_base * (1 + mass_factor)` before overlap effects.
- `Mass positions`: attachment centers along the line (distance or `%`). If blank or mismatched with count, positions auto-distribute uniformly from 15% to 85% of total length.
- `Mass spread`: axial width of each attachment influence; wider spread distributes added mass over more cells (triangular profile with support of about `±3 * spread_frac * L`).
- `Dead-leg length (rough)`: helper input used only for rough `mass_factor` estimation, not direct physics.
- `Dead-leg diameter ratio`: helper ratio `D_dead_leg / D_main` used only for rough `mass_factor` estimation.
- `Apply rough mass factor`: computes and applies a first-cut `mass_factor` from dead-leg estimate assumptions.

### 4) Flow / Inlet

- `Pressure p`: affects gas density (ideal gas relation) and therefore velocity/Reynolds/internal convection.
- `Heater setpoint Tin`: target heater outlet temperature used as the inlet boundary setpoint.
- `Heater ramp profile`: shape of heater startup ramp (`Logistic`, `Linear`, or `Exponential`).
- `Heater rise time to setpoint`: ramp duration for inlet setpoint; with `0 s` the inlet jumps to `Tin`, otherwise it ramps to setpoint over this time according to selected profile (GUI default: `900 s`).
- `Boundary vs first cell`: `Tin_eff` is the imposed boundary/setpoint value at `x=0`; `Tg_in(cell0)` is the first gas control-volume temperature and can be slightly lower due immediate wall heat transfer.
- `Mass flow m_dot`: primary transport lever; higher flow generally accelerates axial front propagation.
- `Search m_dot min`: lower bound for optimization search.
- `Search m_dot max`: upper bound for optimization search.

### 5) Outputs

- `Save plot images`: writes static plot artifacts to run folder.
- `Save run artifacts`: writes run folder (`params`, logs, arrays, summaries).
- `Append run history ledger`: appends key run metrics/configuration to CSV/XLSX ledger after run.
- `Choose save folder`: sets destination root for run directories.
- `Choose ledger file`: sets ledger file path for append/preview operations.

### 6) Advanced Numerics

- `Nx`: axial grid resolution; higher values increase fidelity and runtime.
- `save_frames`: number of stored snapshots for live playback and result history.
- `dt_max`: upper cap on adaptive timestep for stability/accuracy control.
- `dt_min`: lower cap to prevent collapse of timestep in stiff regions.
- `update_props_every`: recomputation interval for flow/transfer coefficients and temperature-dependent state.
- `Stress Nr_wall`: radial reconstruction points for stress screening post-processing.
- `Axial restraint`: scaling for axial thermal restraint contribution in stress indicator.
- `Ignore inlet cells`: excludes first N cells in hotspot diagnostics to reduce BC-dominated false alarms.
- `progress`: controls runtime logging verbosity in UI/log file.
- `use_float32`: improves speed/memory at potential precision cost vs float64.
- `write run.log`: writes textual runtime logs in run folder.
- `write runtime_trace.csv`: writes sampled runtime performance trace.
- `show popup plots after run`: opens matplotlib windows after run completes.
- `include pressure in total stress`: adds pressure terms to total stress indicator.
- `compute stress sensitivity diagnostics`: runs coarse sensitivity checks for stress post-processing confidence.

### 7) Library Tab (Editable Content)

- `Add / Update Pipe Material`: edits scalar baseline properties used for new runs and default flat tables.
- `Add / Update Insulation Material`: edits scalar baseline insulation properties used for new runs.
- `Temperature-Dependent Tables`: edits comma-separated `T`, `cp`, `k` arrays for each material.
- `Load`: loads current stored table for the selected material into editor fields.
- `Save Table`: validates and stores the edited table (strictly increasing `T`, positive `cp/k`).
- `Reset Flat`: regenerates a constant-property table from scalar cp/k values.
- `Pipe Presets`: saves/applies/deletes reusable end-to-end setup presets.

## Thermal Stress Estimate (Current)

The current result plot uses a first-order **internal stress indicator with radial wall discretization** (screening metric), not a full code/FEA qualification model.

```text
R_wall = ln(r_w_o/r_i) / (2*pi*k_w*dx)
R_ins  = ln(r_ins_o/r_w_o) / (2*pi*k_i*dx)
DeltaT_wall_est ~ |Tw - Ti| * R_wall / (R_wall + R_ins)

Nr_wall radial points are reconstructed across the wall thickness.

Thermal hoop contribution (screening):
sigma_theta,th(r) ~ E*alpha/(1-nu) * (Twall_mean - T(r))

Optional pressure contribution:
sigma_r,p(r), sigma_theta,p(r), sigma_z,p (Lamé thick-wall style [6,7])

Axial restraint term:
sigma_z,th ~ -k_restraint * E*alpha/(1-nu) * (Twall_mean - Tamb)

Von Mises indicator:
sigma_vm = sqrt(0.5 * [(sigma_theta-sigma_z)^2 + (sigma_z-sigma_r)^2 + (sigma_r-sigma_theta)^2])

Elbow screening:
sigma_vm,elbow(x,t) = A_elbow(x) * sigma_vm(x,t),  A_elbow(x) >= 1
```

with:

- `A_elbow(x)` localized around each elbow position entered by the user (or auto-distributed if omitted),
- peak amplification approximately controlled by the elbow SIF input,
- triangular spatial decay around each elbow center (screening profile, not a flexibility solve).

Interpretation:

- This is a **risk indicator** intended to locate hot spots and compare operating cases.
- It is focused on stress generated by thermal gradients through the wall thickness.
- Axial restraint stress is intentionally not the primary metric in this tool, because your use case is a one-end-fixed pipe where global axial thermal stress is not dominant.
- Elbow influence is a **localized stress-intensification screening factor**, not a full piping flexibility solve [8].
- The Results tab is presented as a **total stress indicator** view (thermal + optional pressure/restraint), with elbow-adjusted overlays for comparison.
- Mechanical properties (`E`, `alpha`, `nu`) are currently treated as constant per run (material-level values), while thermal properties can be temperature-dependent.

For final design allowables, use a dedicated thermo-mechanical model (axisymmetric/3D FEA with realistic constraints and pressure loading) [7,8].

## Practical Setup Recipes

### Recipe A: Fast commissioning estimate

1. Use `Balanced` preset.
2. Enable insulation if physically present.
3. Enable temperature-dependent solids.
4. Set `Target variable = Gas outlet Tg`.
5. Run fixed time first to understand trend.
6. Switch to outlet-target mode for exact crossing estimate.

### Recipe B: Thermocouple-on-skin matching

1. Set `Target variable = Wall outer outlet Tw(out)`.
2. Ensure insulation thickness/material are correct.
3. Keep ambient/emissivity realistic.
4. Compare predicted outer-wall trace to measured skin sensor.

### Recipe C: Tee-heavy line with local thermal inertia

1. Set thermal mass count to number of major attached masses.
2. Enter approximate positions (absolute distance or `%`).
3. Start with `mass_factor = 1.0` to `3.0`.
4. Tune factor/spread against measured lag in local temperatures.

## Statistics Panel (Current)

The Results tab computes and displays:

- time to target condition (when target mode is active),
- max stress indicator with optional elbow amplification,
- time/location of maximum stress,
- free thermal expansion estimate,
- max estimated wall through-thickness `DeltaT`,
- time/location of max wall `DeltaT`,
- final gas/wall/insulation temperatures at inlet/outlet,
- final outlet outer-wall surface estimate,
- final inlet heater setpoint (`Tin_eff`) and ramp metadata,
- final axial mean temperatures and runtime summary.

Additional diagnostics:

- radial sensitivity (`Nr_wall` vs refined `2*Nr_wall`),
- axial sensitivity (coarsened-x check),
- inlet-hotspot ratio (global max vs excluding first inlet cells).

## Artifact Controls and Fatigue Warnings

This phase adds controls and warnings to improve interpretability:

- Inlet heater warm-up ramp (`Tin_ramp_s`, default logistic model) to reduce step-change inlet artifacts.
- Ignore-first-cells parameter for hotspot interpretation near inlet.
- Health warning panel in Results tab with screening checks for:
  - high stress vs nominal yield (`Sy`),
  - thermal strain range (cyclic fatigue proxy),
  - high-temperature exposure (creep/oxidation risk),
  - ratcheting screening index.

These are **screening-level** warnings and should be treated as prioritization aids, anchored to classical low-cycle fatigue and thermal ratcheting concepts [9,10].

## Run Ledger for Long-Term Tracking

The app can append each run to a growing ledger (`CSV`, optional `XLSX` if `openpyxl` is installed):

- timestamp,
- major input conditions (geometry, flow, material, ramp, mesh settings),
- outcome (`stop_reason`, target reached),
- key stress/temperature metrics,
- warning text.

This supports trend tracking and predictive maintenance workflows across many heats.

Note: automatic ledger append is configurable in the Outputs panel and can be left off by default for manual control.

## Running

Install dependencies (example):

```bash
python3 -m pip install -r requirements.txt
```

Run PyQt6 app:

```bash
python3 pyqt6_app.py
```

## Outputs

When saving is enabled, runs are written under `runs/run_YYYYmmdd_HHMMSS_microsec/` with:

- `params.json`
- `run.log` (if enabled)
- `runtime_trace.csv` (if enabled)
- `fields.npz`
- `summary.csv`
- `heatmaps.png`, `profiles.png` (if plot saving enabled)

Run folder retention:

- The solver can cap historical `run_*` folders (`max_run_dirs`, default `1000`).
- Oldest run folders are pruned when creating new runs.
- Ledger and library files are not affected by this pruning.
- For GitHub workflows, generated folders/files are listed in `.gitignore` so commits stay code-focused.

## Error Message Guide (Engineering)

### Input validation errors (raised before solve starts)

| Message | Physical/Numerical Meaning | Typical fix |
|---|---|---|
| `L must be > 0` | Non-physical pipe length. | Set positive length. |
| `Di must be > 0` | Non-physical inner diameter. | Set positive diameter. |
| `t_wall must be >= 0` / `t_ins must be >= 0` | Negative thickness is non-physical. | Use zero or positive thickness. |
| `... must be > 0` (density, cp, k, mu, p, m_dot) | Transport/storage model would divide by zero or become non-physical. | Enter strictly positive property/flow values. |
| `eps_rad must be in [0, 1]` | Emissivity outside thermodynamic bounds. | Keep emissivity between 0 and 1. |
| `h_out must be >= 0` | Negative convection coefficient is non-physical. | Use non-negative value (or auto mode). |
| `h_out_mode must be 'auto' or 'manual'` | Unsupported model selection. | Choose valid mode. |
| `Nx must be >= 3` | Grid too coarse for stable transport/diffusion operators. | Increase axial cells. |
| `dt_max and dt_min must be > 0` / `dt_min must be <= dt_max` | Invalid adaptive-step bounds. | Use positive, ordered time-step limits. |
| `Tin_ramp_s must be >= 0` | Negative ramp time has no physical meaning. | Use zero or positive ramp duration. |
| `Tin_ramp_model must be 'heater_exp', 'linear', or 'logistic'` | Unsupported inlet ramp function selection. | Use one of the supported ramp model strings. |
| `theta_cond must be in (0, 1]` | Implicit diffusion weighting out of scheme bounds. | Keep in `(0, 1]` (default `0.5`). |
| `CFL must be > 0` | Non-positive advection stability scale. | Set positive CFL-like parameter. |
| `Tg_out_target must be set when mode='target'` | Target mode requested without target temperature. | Enter outlet target. |
| `Unknown mode ... expected 'time' or 'target'` | Invalid run control mode. | Select supported mode. |

### Runtime warnings (simulation still runs unless fatal)

- `Numba not available ... using pure-Python kernels`: run is valid but slower.
- `Accelerated timestep path failed; using fallback kernels`: code switched to conservative implementation for robustness.
- `snapshot_callback failed`: live GUI update callback failed; solver continues.
- `Matplotlib unavailable; skipping plot generation`: numerical solve runs but plot artifacts are not saved.
- `openpyxl not installed; falling back to CSV ledger`: ledger remains active in CSV format.
- `Simulation error` dialog in GUI: uncaught exception traceback from worker thread; treat as run failure and inspect stack trace + last input set.

### Common `stop_reason` values shown in status/log

- `t_end`: fixed-time run completed full requested duration.
- `<metric> <= target K`: target reached while cooling.
- `<metric> >= target K`: target reached while heating.
- `manual_config_append`: row appended from Ledger tab without executing a solve.

## Output and Log Window Field Guide

### Live status row

- `Sim time`: latest simulated physical time represented on screen.
- `Outlet Tg`: current/selected outlet gas temperature.
- `Frames`: number of stored snapshots in memory for playback.

### Simulation log (`Simulation Log` tab, `run.log`)

Progress line format:

```text
Progress: steps=..., sim_t=..., wall=..., mean_dt=..., speed=...x, ETA=...
```

- `steps`: total solver time steps taken.
- `sim_t`: simulated physical time.
- `wall`: elapsed wall-clock runtime.
- `mean_dt`: average numerical step size (`sim_t / steps`).
- `speed=x`: simulated-time / wall-time factor.
- `ETA`: estimated wall-clock time remaining to end condition.

Runtime summary line:

- `sim_speed=...x real-time`: aggregate speed factor over whole run.

### Results tab statistics box

- `Time to target condition`: interpolated crossing time (target mode).
- `Max stress indicator (thermal only)`: best estimate of gradient-driven internal stress.
- `Max stress indicator (straight/elbow-adjusted)`: total indicator with optional pressure/restraint and local elbow amplification.
- `Time/location of max stress`: where and when peak indicator occurs.
- `Free thermal expansion estimate`: unconstrained axial extension estimate.
- `Max wall through-thickness dT`: strongest radial wall temperature gradient proxy.
- `Radial/Axial sensitivity diagnostics`: coarse convergence checks for stress post-processing confidence.
- `Inlet-hotspot ratio`: compares global max vs max excluding early inlet cells to identify potential numerical/BC artifacts.

## Elbow Estimation Method (Current Screening Model)

The elbow feature is intentionally a **fast screening layer**:

- User enters elbow count, SIF, and optional positions along `x/L`.
- Elbow position input accepts absolute distance values (current length unit) or explicit percentages using `%`.
- If positions are blank, elbows are auto-distributed between ~15% and ~85% of length.
- A local amplification profile `A_elbow(x)` is built around each elbow.
- Stress maps are multiplied by `A_elbow(x)` to estimate elbow-critical hot spots.

This does not replace piping flexibility analysis. It is intended for comparative case ranking and run-history tracking [8].

## Branching Systems and Ledger Strategy

For systems with a shared trunk and multiple branch paths:

- Use `asset_id` for the physical parent line (shared section).
- Use `branch_id` for the active downstream path for each run.
- Keep one ledger row per heat/cycle with actual active branch.
- For long-term health estimation, split cumulative damage accounting into:
  - shared-trunk exposure (all runs with same `asset_id`),
  - branch-specific exposure (group by `asset_id + branch_id`).

Practical next schema extension if needed:

- add `segment_id` (e.g., `trunk`, `branch_A`, `branch_B`),
- add `duty_weight` (fraction of cycle spent in each segment),
- aggregate fatigue/stress indicators per segment for predictive maintenance planning.

## Optimization Modes (Beta)

### Heatup-time optimize

- Uses coarse-to-fine `m_dot` search:
  - coarse scan over search bounds on reduced `Nx`,
  - local refinement around best candidate,
  - final full run at selected `m_dot`.
- Objective:
  - primary: satisfy stress limit,
  - secondary: minimize `|time_to_target - target_time|` (default tolerance can be set, e.g., ±20 s).

### Stress-limit optimize

- Uses bisection on `m_dot` against stress-limit residual.
- If bracket does not straddle limit, falls back to boundary candidate closest to feasibility.
- Runs one final full-fidelity case and reports resulting time-to-target.

Constraint reporting:

- Optimization outputs now explicitly report whether:
  - target was reached,
  - stress limit was satisfied,
  - heatup tolerance was satisfied (heatup-time mode).

## Roadmap

- Improve live animation controls (speed presets, loop mode, scrub marker syncing all plots).
- Add stress model options (free vs restrained, hoop/axial, pressure coupling).
- Add report export templates for design studies.
- Add regression tests for physics + UI workflows.

## Security

- See [SECURITY.md](SECURITY.md) for distribution-oriented security checklist and hardening notes.
- See [BUILD.md](BUILD.md) for macOS/Windows standalone packaging instructions.

## Scientific References

[1] V. Gnielinski, "New equations for heat and mass transfer in turbulent pipe and channel flow," *International Chemical Engineering*, 16(2), 359-368, 1976.  
[2] B. S. Petukhov, "Heat Transfer and Friction in Turbulent Pipe Flow with Variable Physical Properties," *Advances in Heat Transfer*, 6, 503-564, 1970.  
[3] S. W. Churchill and H. H. S. Chu, "Correlating equations for laminar and turbulent free convection from a horizontal cylinder," *International Journal of Heat and Mass Transfer*, 18(9), 1049-1053, 1975.  
[4] F. P. Incropera, D. P. DeWitt, T. L. Bergman, and A. S. Lavine, *Fundamentals of Heat and Mass Transfer*, Wiley.  
[5] J. Crank and P. Nicolson, "A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type," *Mathematical Proceedings of the Cambridge Philosophical Society*, 43(1), 50-67, 1947.  
[6] W. C. Young, R. G. Budynas, and A. M. Sadegh, *Roark's Formulas for Stress and Strain*, McGraw-Hill.  
[7] S. P. Timoshenko and J. N. Goodier, *Theory of Elasticity*, McGraw-Hill.  
[8] ASME B31.3 Process Piping and ASME B31J Stress Intensification Factors and Flexibility Factors.  
[9] L. F. Coffin Jr., "A Study of the Effects of Cyclic Thermal Stresses on a Ductile Metal," *Transactions of the ASME*, 76, 931-950, 1954; S. S. Manson, "Behavior of Materials under Conditions of Thermal Stress," NACA Report 1170, 1954.  
[10] J. Bree, "Elastic-Plastic Behaviour of Thin Tubes Subjected to Internal Pressure and Intermittent High-Heat Fluxes with Application to Fast-Nuclear-Reactor Fuel Elements," *Journal of Strain Analysis*, 2(3), 226-238, 1967.

## Progress Log (Living)

Use this section as an always-updated engineering log for final reporting.

### 2026-02-25

- Added Phase-1 reliability fixes:
  - no-numba fallback import stability,
  - target stop-direction fix,
  - per-run override support,
  - run directory uniqueness,
  - file handler cleanup,
  - `dt_min` enforcement,
  - safer expression parsing in Tk GUI.
- Introduced PyQt6 scaffold and then upgraded to workflow-first UI.
- Added live solver snapshot callback path from core solver -> controller -> PyQt worker.
- Added embedded live plots and post-run playback slider/time cursor.
- Added static Results tab with heatmap, waterfall, outlet trends, and first-pass stress plot.
- Shifted boundary condition workflow toward material + optional insulation + ambient with auto external convection estimate (`h_out_mode=auto`).
- Changed initial gas temperature handling to initialize from wall/ambient conditions (`T_init_gas` / `T_init_wall`) instead of forcing `Tin`.
- Added runtime material library editing tab (pipe + insulation).
- Added SI/Imperial unit toggle affecting inputs, live plots, result plots, and statistics.
- Replaced axial stress display with a thermal-gradient internal stress indicator + added statistics panel (target time, max stress, stress location/time, expansion, wall dT metrics).
- Added built-in material presets for Haynes 282 and Hastelloy X.
- Added Material Library tab for user-defined pipe/insulation materials during runtime.
- Fixed live-plot rendering path to keep a single heatmap colorbar axis (prevents multi-colorbar/axis clutter).
- Ensured gas initial condition can be set from ambient/wall initialization (`T_init_gas`) rather than forcing `Tin`.
- Added stress-v2 screening with radial wall mesh (`Nr_wall`) and von-Mises indicator outputs.
- Added elbow screening support (count + SIF factor) to report elbow-amplified stress indicators.
- Added inlet temperature ramp control (`Tin_ramp_s`) in solver and UI.
- Added stress sensitivity diagnostics (radial refinement and axial coarse checks) + inlet hotspot ratio.
- Added health/fatigue warning panel including ratcheting-oriented screening index.
- Added run history ledger append workflow (`CSV`, optional `XLSX`).
- Added localized elbow-position screening input (`auto` or explicit `%L` list) and localized elbow amplification profile.
- Added `Ledger` tab (preview, append config row, delete selected rows) and `README` tab (in-app Markdown rendering).
- Added scenario metadata fields (`asset_id`, `branch_id`) into run ledger rows for shared-trunk/branch analytics.
- Switched internal convection model to Gnielinski (laminar/transition/turbulent blending).
- Expanded README with engineering error guide, output-field definitions, elbow-model assumptions, branching-ledger method, and scientific citations.
- Set left sidebar minimum width to prevent lateral clipping/scroll behavior in dense parameter forms.
- Renamed `Material Library` tab to `Library` and added custom pipe preset management (save/apply/delete).
- Changed elbow-position parsing to accept explicit distances (`m`/`ft` based on active units) and `%` notation.
- Changed results presentation to focus on total stress indicators (with elbow-adjusted comparison).
- Changed automatic run-ledger append default to off.
- Stabilized Ledger tab table behavior to avoid persistent UI resizing from wide auto-sized columns.
- Added `Heatup-time optimize` mode (coarse-to-fine search on `m_dot` with stress cap + time tolerance target).
- Added `Stress-limit optimize` mode (bisection on `m_dot` against stress cap, report heatup time).
- Added one-click `Export Bundle` action to package run + config + stats/warnings + ledger/library snapshots.
- Added run-folder retention cap (`max_run_dirs`, default `1000`) to prevent uncontrolled run-directory growth.
- Added `SECURITY.md` with a release security checklist and hardening recommendations.
- Added ledger-tab width stabilization (compact status text + curated preview columns) to avoid UI spring-open behavior.
- Added optimization constraint status reporting (target reached, stress-limit pass/fail, heatup tolerance pass/fail).
- Added standalone packaging guide (`BUILD.md`) for macOS/Windows via PyInstaller.
- Added ledger schema migration handling (legacy/new columns) for both preview and append workflows to avoid malformed CSV/XLSX rows over time.
- Added compact table size-hint behavior on the Ledger tab to avoid window expansion when opening wide ledgers.
- Added optimization post-correction pass on full-resolution mesh to better honor stress/time constraints and record correction status.
- Added per-run `optimization_summary.json` artifact in run folders for easier log/debug review.
- Aligned optimization-run `params.json` `t_end` with the effective optimization cap to remove confusing `params` vs `run.log` time mismatches.

### 2026-02-26

- Added temperature-dependent solid property plumbing for both pipe and insulation (`cp(T)`, `k(T)` interpolation in solver).
- Added representative built-in temperature-property tables for supported metals and insulation materials.
- Added attached thermal-mass controls (count/factor/positions/spread) and coupled them into local wall thermal capacity.
- Added selectable target variable (`gas_outlet`, `wall_inner_outlet`, `wall_outer_outlet`, `insulation_outlet`) across solve, optimization, and results/statistics.
- Expanded README with beginner-first operating workflow, plain-language terminology, setup recipes, and feature usage guidance.
- Added Library-tab temperature-table editor for direct `T/cp/k` material table editing (pipe + insulation).
- Added dead-leg rough-scaling helper for thermal mass factor (with one-click apply).
- Expanded parameter reference so every GUI input has short effect-oriented documentation.
- Added explicit documentation for thermal-mass semantics (mass factor, spread, auto-distribute behavior, and simulation impact).
- Added worked example mapping: `4` dead-end tees of `12 m` each -> concrete GUI settings and rough mass-factor calculation.
- Added inlet heater warm-up documentation and solver equations (linear default with optional `heater_exp` mode).
- Aligned UI wording to heater semantics (`Heater setpoint Tin`, `Heater rise time to setpoint`) and added inlet tracking visualization (`Tin_eff` vs `Tg_in(cell0)`).
- Set new defaults: `m_dot = 2.5 kg/s` and `t_end = 5000 s`.
- Added selectable heater ramp profile (`Logistic`, `Linear`, `Exponential`) with logistic default and `Tin_ramp_s = 900 s` GUI default.
- Reduced left control-panel width and tightened top status row to improve fit on smaller displays.
