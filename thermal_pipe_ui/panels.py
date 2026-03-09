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
"""UI panel/tab construction mixin for the thermal pipe application."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .optimization import TARGET_METRIC_CHOICES
from .window import (
    CompactTableWidget,
    Figure,
    FigureCanvas,
    HAS_MPL,
    INSULATION_MATERIALS,
    INSULATION_TEMP_PROPS,
    PIPE_MATERIALS,
    PIPE_TEMP_PROPS,
)

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QAbstractScrollArea,
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFormLayout,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QHeaderView,
        QScrollArea,
        QSlider,
        QSpinBox,
        QSizePolicy,
        QSplitter,
        QTabWidget,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "PyQt6 is required to run this UI.\n"
        "Install with: pip install PyQt6\n"
        f"Import error: {exc}"
    )


class UIPanelMixin:
    def _build_left_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.horizontalScrollBar().setEnabled(False)
        scroll.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)

        panel = QWidget()
        panel.setMinimumWidth(0)
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_scenario_group())
        layout.addWidget(self._build_material_group())
        layout.addWidget(self._build_pipe_group())
        layout.addWidget(self._build_flow_group())
        layout.addWidget(self._build_output_group())
        layout.addWidget(self._build_advanced_group())
        layout.addStretch(1)

        scroll.setWidget(panel)
        return scroll

    def _build_right_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(0)
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addLayout(self._build_metrics_row())

        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(0)
        self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.tabs.addTab(self._build_live_tab(), "Live View")
        self.tabs.addTab(self._build_results_tab(), "Results")
        self.tabs.addTab(self._build_material_library_tab(), "Library")
        self.tabs.addTab(self._build_ledger_tab(), "Ledger")
        self.tabs.addTab(self._build_readme_tab(), "README")
        self.tabs.addTab(self._build_log_tab(), "Simulation Log")
        layout.addWidget(self.tabs, 1)
        return panel

    def _build_metrics_row(self):
        row = QHBoxLayout()
        row.setSpacing(10)
        self.lbl_runtime = QLabel("Sim time: 0.0 s")
        self.lbl_target_time = QLabel("Target time: --")
        self.lbl_target_time.setStyleSheet("color: #b8d3ff;")
        self.lbl_frames = QLabel("Frames: 0")
        self.lbl_live_readout = QLabel("Time: -- s || Tg_out -- | Tw_out -- | Ti_out -- | Tin_eff -- | Tg_in -- K")
        self.lbl_live_readout.setStyleSheet("color: #d0d0d0;")
        # Keep individual labels for compatibility with existing update paths.
        self.lbl_outlet = QLabel("Outlet Tg: -- K")
        self.lbl_wall_out = QLabel("Outlet Tw: -- K")
        self.lbl_ins_out = QLabel("Outlet Ti: -- K")
        self.lbl_inlet = QLabel("Tin_eff(bc): -- K")
        self.lbl_inlet_cell = QLabel("Tg_in(c0): -- K")
        self.lbl_mode_hint = QLabel("Workflow: configure -> run")
        self.lbl_mode_hint.setStyleSheet("color: #808080;")
        self.lbl_mode_hint.setMaximumWidth(210)
        for lbl in (
            self.lbl_runtime,
            self.lbl_target_time,
            self.lbl_frames,
            self.lbl_live_readout,
            self.lbl_mode_hint,
        ):
            lbl.setMinimumWidth(0)
            lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row.addWidget(self.lbl_runtime)
        row.addWidget(self.lbl_target_time)
        row.addWidget(self.lbl_frames)
        row.addWidget(self.lbl_live_readout, 1)
        row.addWidget(self.lbl_mode_hint)
        return row

    def _build_scenario_group(self):
        g = QGroupBox("1) Scenario")
        form = QFormLayout(g)

        self.preset = QComboBox()
        self.preset.addItems(list(self.PRESETS))
        self.preset.setCurrentText("Fast estimate")
        self.preset.currentTextChanged.connect(self._apply_preset)

        self.pipe_default = QComboBox()
        self.pipe_default.addItems(list(self.PIPE_DEFAULTS))
        btn_apply_pipe_default = QPushButton("Apply Pipe Default")
        btn_apply_pipe_default.clicked.connect(self._apply_pipe_default)

        self.units = QComboBox()
        self.units.addItems(["SI", "Imperial"])
        self.units.currentTextChanged.connect(self._on_units_changed)

        self.asset_id = QLineEdit("PIPE-001")
        self.branch_id = QLineEdit("main")

        self.mode = QComboBox()
        self.mode.addItems(
            [
                "Fixed time",
                "Outlet target",
                "Heatup-time optimize",
                "Stress-limit optimize",
            ]
        )
        self.mode.currentIndexChanged.connect(self._sync_mode_widgets)

        self.in_t_end = self._dbl(5000.0, 0.001, 1.0e8, 3, " s")
        self.in_target = self._dbl(950.0, 1.0, 5000.0, 2, " K")
        self.target_metric = QComboBox()
        for label, key in TARGET_METRIC_CHOICES:
            self.target_metric.addItem(label, key)
        self.target_metric.setCurrentIndex(0)
        self.stop_dir = QComboBox()
        self.stop_dir.addItems(["auto", "le", "ge"])
        self.in_heatup_target = self._dbl(1800.0, 1.0, 1.0e8, 1, " s")
        self.in_heatup_tol = self._dbl(20.0, 1.0, 1.0e5, 1, " s")
        self.in_stress_limit = self._dbl(250.0, 1.0, 1.0e5, 2, " MPa")

        form.addRow("Units", self.units)
        form.addRow("Performance mode", self.preset)
        form.addRow("Pipe default", self.pipe_default)
        form.addRow("", btn_apply_pipe_default)
        form.addRow("Asset ID", self.asset_id)
        form.addRow("Branch ID", self.branch_id)
        form.addRow("Run mode", self.mode)
        form.addRow("t_end / t_max", self.in_t_end)
        form.addRow("Outlet target", self.in_target)
        form.addRow("Target variable", self.target_metric)
        form.addRow("Stop direction", self.stop_dir)
        form.addRow("Heatup target time", self.in_heatup_target)
        form.addRow("Heatup tolerance", self.in_heatup_tol)
        form.addRow("Stress limit", self.in_stress_limit)
        return g

    def _build_material_group(self):
        g = QGroupBox("2) Materials / Ambient")
        form = QFormLayout(g)

        self.pipe_material = QComboBox()
        self.pipe_material.addItems(list(PIPE_MATERIALS))
        self.pipe_material.currentTextChanged.connect(self._on_pipe_material_changed)

        self.in_ambient = self._dbl(300.0, 100.0, 2000.0, 2, " K")
        self.in_eps = self._dbl(0.70, 0.0, 1.0, 3, "")

        self.chk_use_insulation = QCheckBox("Enable insulation")
        self.chk_use_insulation.setChecked(True)
        self.chk_use_insulation.toggled.connect(self._sync_insulation_widgets)

        self.ins_material = QComboBox()
        self.ins_material.addItems(list(INSULATION_MATERIALS))
        self.in_t_ins = self._dbl(0.15, 0.0, 2.0, 4, " m")
        self.chk_temp_dep_props = QCheckBox("Use temperature-dependent\ncp(T), k(T)")
        self.chk_temp_dep_props.setChecked(True)
        self.lbl_temp_dep_hint = QLabel(
            "Edit property tables in Library"
            "-> Temperature-Dependent Tables."
            "If disabled, scalar cp/k values are used."
        )
        self.lbl_temp_dep_hint.setWordWrap(True)
        self.lbl_temp_dep_hint.setStyleSheet("color: #8c8c8c;")

        form.addRow("Pipe material", self.pipe_material)
        form.addRow("Ambient temp", self.in_ambient)
        form.addRow("Surface emissivity", self.in_eps)
        form.addRow("", self.chk_use_insulation)
        form.addRow("Insulation material", self.ins_material)
        form.addRow("Insulation thickness", self.in_t_ins)
        form.addRow("", self.chk_temp_dep_props)
        form.addRow("", self.lbl_temp_dep_hint)
        return g

    def _build_pipe_group(self):
        g = QGroupBox("3) Pipe Geometry")
        form = QFormLayout(g)

        self.in_L = self._dbl(65.0, 0.01, 1e6, 3, " m")
        self.in_Di = self._dbl(0.13, 0.001, 1e3, 4, " m")
        self.in_t_wall = self._dbl(0.018, 0.0001, 1.0, 4, " m")
        self.in_elbows = QSpinBox()
        self.in_elbows.setRange(0, 40)
        self.in_elbows.setValue(0)
        self.in_elbow_sif = self._dbl(1.30, 1.00, 5.00, 2, "")
        self.in_elbow_positions = QLineEdit("")
        self.in_elbow_positions.setPlaceholderText("auto or e.g. 45, 50 (m/ft) or 20%, 60%")
        self.in_tmass_count = QSpinBox()
        self.in_tmass_count.setRange(0, 100)
        self.in_tmass_count.setValue(0)
        self.in_tmass_factor = self._dbl(0.0, 0.0, 50.0, 2, " x")
        self.in_tmass_positions = QLineEdit("")
        self.in_tmass_positions.setPlaceholderText("auto or e.g. 12, 35 (m/ft) or 20%, 60%")
        self.in_tmass_spread = self._dbl(3.0, 0.1, 50.0, 1, " %L")
        self.in_tmass_deadleg_len = self._dbl(2.0, 0.0, 1.0e6, 3, " m")
        self.in_tmass_deadleg_d_ratio = self._dbl(1.0, 0.1, 10.0, 2, " x")
        btn_apply_tmass_est = QPushButton("Apply rough mass factor")
        btn_apply_tmass_est.clicked.connect(self._apply_tmass_rough_estimate)
        self.lbl_tmass_help = QLabel(
            "Mass factor increases local wall thermal capacity around each attachment center. "
            "If positions are blank, centers auto-distribute from 15% to 85% of length."
        )
        self.lbl_tmass_help.setWordWrap(True)
        self.lbl_tmass_help.setStyleSheet("color: #8c8c8c;")
        self.lbl_tmass_est = QLabel("")
        self.lbl_tmass_est.setWordWrap(True)
        self.lbl_tmass_est.setStyleSheet("color: #8c8c8c;")
        self.in_tmass_deadleg_len.valueChanged.connect(self._update_tmass_estimate_label)
        self.in_tmass_deadleg_d_ratio.valueChanged.connect(self._update_tmass_estimate_label)
        self.in_tmass_spread.valueChanged.connect(self._update_tmass_estimate_label)
        self.in_L.valueChanged.connect(self._update_tmass_estimate_label)

        form.addRow("Length L", self.in_L)
        form.addRow("Inner diameter Di", self.in_Di)
        form.addRow("Wall thickness", self.in_t_wall)
        form.addRow("Number of elbows", self.in_elbows)
        form.addRow("Elbow SIF factor", self.in_elbow_sif)
        form.addRow("Elbow positions", self.in_elbow_positions)
        form.addRow("Attached thermal masses", self.in_tmass_count)
        form.addRow("Mass factor per attach", self.in_tmass_factor)
        form.addRow("Mass positions", self.in_tmass_positions)
        form.addRow("Mass spread", self.in_tmass_spread)
        form.addRow("Dead-leg length (rough)", self.in_tmass_deadleg_len)
        form.addRow("Dead-leg diameter ratio", self.in_tmass_deadleg_d_ratio)
        form.addRow("", btn_apply_tmass_est)
        form.addRow("", self.lbl_tmass_help)
        form.addRow("", self.lbl_tmass_est)
        return g

    def _build_flow_group(self):
        g = QGroupBox("4) Flow / Inlet")
        form = QFormLayout(g)

        self.in_p = self._dbl(5.0e6, 1.0, 1.0e9, 1, " Pa")
        self.in_Tin = self._dbl(1000.0, 1.0, 5000.0, 2, " K")
        self.in_tin_ramp = self._dbl(900.0, 0.0, 1.0e6, 1, " s")
        self.in_tin_ramp_model = QComboBox()
        self.in_tin_ramp_model.addItem("Logistic (S-curve)", "logistic")
        self.in_tin_ramp_model.addItem("Linear", "linear")
        self.in_tin_ramp_model.addItem("Exponential", "heater_exp")
        self.in_tin_ramp_model.setCurrentIndex(0)
        self.in_mdot = self._dbl(2.5, 0.0001, 1.0e4, 4, " kg/s")
        self.in_mdot_min = self._dbl(0.2, 0.0001, 1.0e4, 4, " kg/s")
        self.in_mdot_max = self._dbl(5.0, 0.0001, 1.0e4, 4, " kg/s")

        form.addRow("Pressure p", self.in_p)
        form.addRow("Heater setpoint Tin", self.in_Tin)
        form.addRow("Heater ramp profile", self.in_tin_ramp_model)
        form.addRow("Heater rise time\nto setpoint", self.in_tin_ramp)
        form.addRow("Mass flow m_dot", self.in_mdot)
        form.addRow("Search m_dot min", self.in_mdot_min)
        form.addRow("Search m_dot max", self.in_mdot_max)
        return g

    def _build_output_group(self):
        g = QGroupBox("5) Outputs")
        form = QFormLayout(g)

        self.chk_make_plots = QCheckBox("Save plot images")
        self.chk_make_plots.setChecked(False)
        self.chk_save_results = QCheckBox("Save run artifacts")
        self.chk_save_results.setChecked(True)
        self.chk_append_ledger = QCheckBox("Append run-history\nledger")
        self.chk_append_ledger.setChecked(False)

        checks = QWidget()
        checks_l = QVBoxLayout(checks)
        checks_l.setContentsMargins(0, 0, 0, 0)
        checks_l.setSpacing(2)
        checks_l.addWidget(self.chk_make_plots)
        checks_l.addWidget(self.chk_save_results)
        checks_l.addWidget(self.chk_append_ledger)

        self.dir_label = QLabel("(auto)")
        self.dir_label.setMinimumWidth(0)
        self.dir_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        btn_dir = QPushButton("Choose save folder...")
        btn_dir.clicked.connect(self._choose_dir)

        out_row = QWidget()
        out_l = QHBoxLayout(out_row)
        out_l.setContentsMargins(0, 0, 0, 0)
        out_l.setSpacing(6)
        out_l.addWidget(btn_dir)
        out_l.addWidget(self.dir_label, 1)

        self.ledger_label = QLabel("(auto)")
        self.ledger_label.setMinimumWidth(0)
        self.ledger_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.ledger_label.setToolTip("Auto: run_history.csv")
        btn_ledger = QPushButton("Choose ledger...")
        btn_ledger.clicked.connect(self._choose_ledger_file)
        ledger_row = QWidget()
        ledger_l = QHBoxLayout(ledger_row)
        ledger_l.setContentsMargins(0, 0, 0, 0)
        ledger_l.setSpacing(6)
        ledger_l.addWidget(btn_ledger)
        ledger_l.addWidget(self.ledger_label, 1)

        form.addRow("Options", checks)
        form.addRow("Folder", out_row)
        form.addRow("Run ledger", ledger_row)
        return g

    def _build_advanced_group(self):
        g = QGroupBox("6) Advanced Numerics")
        form = QFormLayout(g)

        self.in_Nx = QSpinBox()
        self.in_Nx.setRange(50, 30000)
        self.in_Nx.setValue(1300)

        self.in_save_frames = QSpinBox()
        self.in_save_frames.setRange(2, 5000)
        self.in_save_frames.setValue(240)

        self.in_dt_max = self._dbl(2.5, 0.0001, 1.0e3, 4, " s")
        self.in_dt_min = self._dbl(1.0e-4, 1.0e-8, 100.0, 6, " s")
        self.in_update_props = QSpinBox()
        self.in_update_props.setRange(1, 1000)
        self.in_update_props.setValue(5)
        self.in_adv_scheme = QComboBox()
        self.in_adv_scheme.addItem("semi_lagrangian", "semi_lagrangian")
        self.in_adv_scheme.addItem("upwind", "upwind")
        self.in_adv_scheme.setCurrentIndex(0)
        self.in_semi_lag_cmax = self._dbl(1.2, 0.1, 20.0, 2, "")
        self.in_ins_mass_mode = QComboBox()
        self.in_ins_mass_mode.addItem("penetration", "penetration")
        self.in_ins_mass_mode.addItem("full", "full")
        self.in_ins_mass_mode.setCurrentIndex(0)
        self.in_ins_mass_min_frac = self._dbl(0.25, 0.01, 1.0, 2, "")

        self.in_nr_wall = QSpinBox()
        self.in_nr_wall.setRange(3, 80)
        self.in_nr_wall.setValue(12)
        self.in_axial_restraint = self._dbl(0.00, 0.0, 1.0, 2, "")
        self.in_ignore_inlet_cells = QSpinBox()
        self.in_ignore_inlet_cells.setRange(0, 200)
        self.in_ignore_inlet_cells.setValue(2)

        self.progress = QComboBox()
        self.progress.addItems(["none", "basic"])
        self.progress.setCurrentText("basic")

        self.chk_use_float32 = QCheckBox("use_float32")
        self.chk_use_float32.setChecked(True)
        self.chk_log_to_file = QCheckBox("write run.log")
        self.chk_log_to_file.setChecked(True)
        self.chk_write_trace = QCheckBox("write runtime trace")
        self.chk_write_trace.setChecked(True)
        self.chk_show_plots = QCheckBox("open results popup\nafter run")
        self.chk_show_plots.setChecked(False)
        self.chk_include_pressure = QCheckBox("include pressure in\ntotal stress")
        self.chk_include_pressure.setChecked(True)
        self.chk_convergence_diag = QCheckBox("stress sensitivity\ndiagnostics")
        self.chk_convergence_diag.setChecked(True)
        self.chk_target_asymptote = QCheckBox("target asymptote stop")
        self.chk_target_asymptote.setChecked(True)

        flags = QWidget()
        flags_l = QVBoxLayout(flags)
        flags_l.setContentsMargins(0, 0, 0, 0)
        flags_l.setSpacing(2)
        flags_l.addWidget(self.chk_use_float32)
        flags_l.addWidget(self.chk_log_to_file)
        flags_l.addWidget(self.chk_write_trace)
        flags_l.addWidget(self.chk_show_plots)
        flags_l.addWidget(self.chk_include_pressure)
        flags_l.addWidget(self.chk_convergence_diag)
        flags_l.addWidget(self.chk_target_asymptote)

        form.addRow("Nx", self.in_Nx)
        form.addRow("Save frames", self.in_save_frames)
        form.addRow("dt max", self.in_dt_max)
        form.addRow("dt min", self.in_dt_min)
        form.addRow("Props update step", self.in_update_props)
        form.addRow("Advection", self.in_adv_scheme)
        form.addRow("Semi-Lag Courant max", self.in_semi_lag_cmax)
        form.addRow("Insulation mass model", self.in_ins_mass_mode)
        form.addRow("Insulation min frac", self.in_ins_mass_min_frac)
        form.addRow("Wall Nr", self.in_nr_wall)
        form.addRow("Axial restraint", self.in_axial_restraint)
        form.addRow("Ignore inlet cells", self.in_ignore_inlet_cells)
        form.addRow("Progress", self.progress)
        form.addRow("Flags", flags)
        return g

    def _build_live_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        controls = QHBoxLayout()
        self.btn_run_anim = QPushButton("Run Animation")
        self.btn_pause_anim = QPushButton("Pause")
        self.slider_time = QSlider(Qt.Orientation.Horizontal)
        self.slider_time.setEnabled(False)
        self.lbl_time_cursor = QLabel("t = -- s / -- s")

        self.btn_run_anim.clicked.connect(self._run_animation)
        self.btn_pause_anim.clicked.connect(self._pause_animation)
        self.slider_time.valueChanged.connect(self._on_slider_changed)

        if not HAS_MPL:
            msg = QLabel("Matplotlib is required for embedded live plots.\nInstall with: pip install matplotlib")
            msg.setWordWrap(True)
            layout.addWidget(msg)
            self.btn_run_anim.setEnabled(False)
            self.btn_pause_anim.setEnabled(False)
            controls.addWidget(self.btn_run_anim)
            controls.addWidget(self.btn_pause_anim)
            controls.addWidget(self.slider_time, 1)
            controls.addWidget(self.lbl_time_cursor)
            layout.addLayout(controls)
            return tab

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self._build_profile_canvas())
        split.addWidget(self._build_heatmap_canvas())
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setSizes([430, 360])
        layout.addWidget(split, 1)

        controls.addWidget(self.btn_run_anim)
        controls.addWidget(self.btn_pause_anim)
        controls.addWidget(self.slider_time, 1)
        controls.addWidget(self.lbl_time_cursor)
        layout.addLayout(controls)
        return tab

    def _build_profile_canvas(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)

        self.live_fig = Figure(figsize=(8, 6), constrained_layout=True)
        self.ax_outlet = self.live_fig.add_subplot(2, 1, 1)
        self.ax_profile = self.live_fig.add_subplot(2, 1, 2)
        self.live_canvas = FigureCanvas(self.live_fig)
        self.live_canvas.mpl_connect("button_press_event", self._on_live_plot_click)
        l.addWidget(self.live_canvas, 1)
        return w

    def _build_heatmap_canvas(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)

        self.heat_fig = Figure(figsize=(8, 4), constrained_layout=True)
        self.ax_heat = self.heat_fig.add_subplot(1, 1, 1)
        self.heat_canvas = FigureCanvas(self.heat_fig)
        self.heat_canvas.mpl_connect("button_press_event", self._on_live_plot_click)
        l.addWidget(self.heat_canvas, 1)
        return w

    def _build_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        if not HAS_MPL:
            msg = QLabel("Matplotlib is required to render results plots.")
            msg.setWordWrap(True)
            layout.addWidget(msg)
            return tab

        self.stats_box = QPlainTextEdit()
        self.stats_box.setReadOnly(True)
        self.stats_box.setMaximumBlockCount(500)
        self.stats_box.setFixedHeight(140)
        layout.addWidget(self.stats_box)

        self.warning_box = QPlainTextEdit()
        self.warning_box.setReadOnly(True)
        self.warning_box.setMaximumBlockCount(500)
        self.warning_box.setFixedHeight(120)
        self.warning_box.setStyleSheet("QPlainTextEdit { background: #1f1f1f; color: #ffb347; }")
        layout.addWidget(self.warning_box)

        tip = QLabel("Tip: left-click any results plot to enlarge, right-click to save.")
        tip.setStyleSheet("color: #8c8c8c;")
        layout.addWidget(tip)

        self.results_fig = Figure(figsize=(12, 8), constrained_layout=True)
        self.results_canvas = FigureCanvas(self.results_fig)
        self.results_canvas.mpl_connect("button_press_event", self._on_results_plot_click)
        layout.addWidget(self.results_canvas, 1)
        return tab

    def _build_material_library_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        left = QVBoxLayout()
        right = QVBoxLayout()
        preset_col = QVBoxLayout()

        pipe_group = QGroupBox("Add / Update Pipe Material")
        pipe_form = QFormLayout(pipe_group)
        self.new_pipe_name = QLineEdit()
        self.new_pipe_rho = self._dbl(8000.0, 1.0, 50000.0, 2, " kg/m^3")
        self.new_pipe_cp = self._dbl(500.0, 1.0, 5000.0, 2, " J/kgK")
        self.new_pipe_k = self._dbl(16.0, 0.01, 2000.0, 3, " W/mK")
        self.new_pipe_E = self._dbl(200.0, 1.0, 500.0, 3, " GPa")
        self.new_pipe_alpha = self._dbl(14.0, 0.1, 50.0, 3, " um/mK")
        self.new_pipe_nu = self._dbl(0.30, 0.01, 0.49, 3, "")
        self.new_pipe_sy = self._dbl(300.0, 1.0, 3000.0, 1, " MPa")
        self.new_pipe_eps = self._dbl(0.70, 0.0, 1.0, 3, "")
        btn_add_pipe = QPushButton("Add / Update Pipe Material")
        btn_add_pipe.clicked.connect(self._add_pipe_material)

        pipe_form.addRow("Name", self.new_pipe_name)
        pipe_form.addRow("Density", self.new_pipe_rho)
        pipe_form.addRow("cp", self.new_pipe_cp)
        pipe_form.addRow("k", self.new_pipe_k)
        pipe_form.addRow("E", self.new_pipe_E)
        pipe_form.addRow("alpha", self.new_pipe_alpha)
        pipe_form.addRow("nu", self.new_pipe_nu)
        pipe_form.addRow("Yield strength", self.new_pipe_sy)
        pipe_form.addRow("Default emissivity", self.new_pipe_eps)
        pipe_form.addRow(btn_add_pipe)

        ins_group = QGroupBox("Add / Update Insulation Material")
        ins_form = QFormLayout(ins_group)
        self.new_ins_name = QLineEdit()
        self.new_ins_rho = self._dbl(128.0, 1.0, 5000.0, 2, " kg/m^3")
        self.new_ins_cp = self._dbl(840.0, 1.0, 5000.0, 2, " J/kgK")
        self.new_ins_k = self._dbl(0.045, 0.001, 10.0, 4, " W/mK")
        btn_add_ins = QPushButton("Add / Update Insulation Material")
        btn_add_ins.clicked.connect(self._add_ins_material)

        ins_form.addRow("Name", self.new_ins_name)
        ins_form.addRow("Density", self.new_ins_rho)
        ins_form.addRow("cp", self.new_ins_cp)
        ins_form.addRow("k", self.new_ins_k)
        ins_form.addRow(btn_add_ins)

        self.pipe_list = QListWidget()
        self.ins_list = QListWidget()

        pipe_list_group = QGroupBox("Pipe Materials")
        pipe_list_l = QVBoxLayout(pipe_list_group)
        pipe_list_l.addWidget(self.pipe_list)

        ins_list_group = QGroupBox("Insulation Materials")
        ins_list_l = QVBoxLayout(ins_list_group)
        ins_list_l.addWidget(self.ins_list)

        temp_table_group = QGroupBox("Temperature-Dependent Tables (cp(T), k(T))")
        temp_form = QFormLayout(temp_table_group)
        self.temp_table_type = QComboBox()
        self.temp_table_type.addItems(["Pipe", "Insulation"])
        self.temp_table_material = QComboBox()
        self.temp_table_type.currentTextChanged.connect(self._refresh_temp_prop_materials)
        self.temp_table_material.currentTextChanged.connect(self._load_temp_prop_editor)
        self.temp_table_T = QLineEdit()
        self.temp_table_cp = QLineEdit()
        self.temp_table_k = QLineEdit()
        self.temp_table_T.setPlaceholderText("300, 500, 700, 900, 1100")
        self.temp_table_cp.setPlaceholderText("e.g. 500, 530, 560, 600, 640")
        self.temp_table_k.setPlaceholderText("e.g. 14.5, 16.0, 17.8, 19.5, 21.2")

        temp_btns = QWidget()
        temp_btns_l = QHBoxLayout(temp_btns)
        temp_btns_l.setContentsMargins(0, 0, 0, 0)
        btn_load_table = QPushButton("Load")
        btn_save_table = QPushButton("Save Table")
        btn_flat_table = QPushButton("Reset Flat")
        btn_load_table.clicked.connect(self._load_temp_prop_editor)
        btn_save_table.clicked.connect(self._save_temp_prop_editor)
        btn_flat_table.clicked.connect(self._reset_temp_prop_editor_flat)
        temp_btns_l.addWidget(btn_load_table)
        temp_btns_l.addWidget(btn_save_table)
        temp_btns_l.addWidget(btn_flat_table)

        temp_hint = QLabel(
            "Enter comma-separated values with matching lengths. "
            "T must be strictly increasing and cp/k must stay positive."
        )
        temp_hint.setWordWrap(True)
        temp_hint.setStyleSheet("color: #8c8c8c;")

        temp_form.addRow("Material type", self.temp_table_type)
        temp_form.addRow("Material", self.temp_table_material)
        temp_form.addRow("T [K]", self.temp_table_T)
        temp_form.addRow("cp [J/kgK]", self.temp_table_cp)
        temp_form.addRow("k [W/mK]", self.temp_table_k)
        temp_form.addRow("", temp_btns)
        temp_form.addRow("", temp_hint)

        preset_group = QGroupBox("Pipe Presets")
        preset_form = QFormLayout(preset_group)
        self.new_preset_name = QLineEdit()
        self.new_preset_name.setPlaceholderText("e.g. Branch A startup")
        btn_add_preset = QPushButton("Save Current as Preset")
        btn_add_preset.clicked.connect(self._save_current_pipe_preset)
        preset_form.addRow("Preset name", self.new_preset_name)
        preset_form.addRow("", btn_add_preset)

        preset_list_group = QGroupBox("Preset Library")
        preset_list_l = QVBoxLayout(preset_list_group)
        self.pipe_preset_list = QListWidget()
        self.pipe_preset_list.itemDoubleClicked.connect(lambda _item: self._apply_selected_pipe_preset())
        preset_buttons = QHBoxLayout()
        btn_apply_preset = QPushButton("Apply")
        btn_apply_preset.clicked.connect(self._apply_selected_pipe_preset)
        btn_delete_preset = QPushButton("Delete")
        btn_delete_preset.clicked.connect(self._delete_selected_pipe_preset)
        preset_buttons.addWidget(btn_apply_preset)
        preset_buttons.addWidget(btn_delete_preset)
        preset_list_l.addWidget(self.pipe_preset_list, 1)
        preset_list_l.addLayout(preset_buttons)

        left.addWidget(pipe_group)
        left.addWidget(ins_group)
        right.addWidget(pipe_list_group)
        right.addWidget(ins_list_group)
        right.addWidget(temp_table_group)
        preset_col.addWidget(preset_group)
        preset_col.addWidget(preset_list_group, 1)

        layout.addLayout(left, 1)
        layout.addLayout(right, 1)
        layout.addLayout(preset_col, 1)
        return tab

    def _build_ledger_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        controls = QHBoxLayout()
        self.btn_ledger_refresh = QPushButton("Refresh")
        self.btn_ledger_refresh.clicked.connect(self._refresh_ledger_preview)
        self.btn_ledger_append = QPushButton("Append Current Config")
        self.btn_ledger_append.clicked.connect(self._append_current_config_to_ledger)
        self.btn_ledger_delete = QPushButton("Delete Selected Row(s)")
        self.btn_ledger_delete.clicked.connect(self._delete_selected_ledger_rows)
        self.lbl_ledger_status = QLabel("Ledger: --")
        self.lbl_ledger_status.setMinimumWidth(0)
        self.lbl_ledger_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        controls.addWidget(self.btn_ledger_refresh)
        controls.addWidget(self.btn_ledger_append)
        controls.addWidget(self.btn_ledger_delete)
        controls.addStretch(1)
        controls.addWidget(self.lbl_ledger_status)
        layout.addLayout(controls)

        self.ledger_table = CompactTableWidget()
        self.ledger_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.ledger_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.ledger_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.ledger_table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
        self.ledger_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.ledger_table.horizontalHeader().setStretchLastSection(False)
        self.ledger_table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.ledger_table.setWordWrap(False)
        self.ledger_table.setMinimumWidth(0)
        self.ledger_table.setMinimumSize(0, 0)
        self.ledger_table.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.ledger_table, 1)
        return tab

    def _build_readme_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        btn_reload = QPushButton("Reload README")
        btn_reload.clicked.connect(self._refresh_readme_view)
        self.lbl_readme_path = QLabel(str(self._readme_path))
        controls.addWidget(btn_reload)
        controls.addWidget(self.lbl_readme_path, 1)
        layout.addLayout(controls)

        self.readme_view = QTextBrowser()
        self.readme_view.setOpenExternalLinks(True)
        layout.addWidget(self.readme_view, 1)
        return tab

    def _build_log_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, 1)
        return tab

    def _build_run_bar(self):
        row = QHBoxLayout()
        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.clicked.connect(self._run_clicked)
        self.btn_cancel = QPushButton("Cancel Simulation")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_clicked)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(14)

        self.status = QLabel("Idle")
        self.status.setFrameShape(QFrame.Shape.NoFrame)
        self.btn_export = QPushButton("Export Bundle")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_bundle_clicked)

        row.addWidget(self.btn_run)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.btn_export)
        row.addWidget(self.progress_bar, 1)
        row.addWidget(self.status, 1)
        return row

    def _dbl(self, value, minimum, maximum, decimals, suffix):
        box = QDoubleSpinBox()
        box.setDecimals(decimals)
        box.setRange(minimum, maximum)
        box.setValue(value)
        box.setSuffix(suffix)
        return box
