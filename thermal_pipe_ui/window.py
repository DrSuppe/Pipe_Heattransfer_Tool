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
"""PyQt6 desktop application for configuring, running, and reviewing simulations."""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from sim_controller import HardwareConfig, RunInputs, RunSpec
from Pipe_Sim_V4 import HAS_NUMBA

try:
    from PyQt6.QtCore import QObject, QSize, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QAbstractScrollArea,
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QDialog,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QHeaderView,
        QTableWidget,
        QTableWidgetItem,
        QTextBrowser,
        QScrollArea,
        QSlider,
        QSpinBox,
        QSizePolicy,
        QSplitter,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "PyQt6 is required to run this UI.\n"
        "Install with: pip install PyQt6\n"
        f"Import error: {exc}"
    )

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    HAS_MPL = True
except Exception:
    HAS_MPL = False
    FigureCanvas = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment]


PIPE_MATERIALS: Dict[str, Dict[str, float]] = {
    "Carbon Steel": {
        "rho_w": 7850.0,
        "cp_w": 486.0,
        "k_w": 45.0,
        "E": 210.0e9,
        "alpha": 12.0e-6,
        "nu": 0.29,
        "Sy": 250.0e6,
        "eps_default": 0.80,
    },
    "Stainless 316": {
        "rho_w": 8000.0,
        "cp_w": 500.0,
        "k_w": 16.3,
        "E": 193.0e9,
        "alpha": 16.0e-6,
        "nu": 0.30,
        "Sy": 290.0e6,
        "eps_default": 0.70,
    },
    "Copper": {
        "rho_w": 8960.0,
        "cp_w": 385.0,
        "k_w": 401.0,
        "E": 110.0e9,
        "alpha": 16.5e-6,
        "nu": 0.34,
        "Sy": 70.0e6,
        "eps_default": 0.35,
    },
    "Inconel 625": {
        "rho_w": 8440.0,
        "cp_w": 435.0,
        "k_w": 11.0,
        "E": 205.0e9,
        "alpha": 12.8e-6,
        "nu": 0.29,
        "Sy": 460.0e6,
        "eps_default": 0.65,
    },
    "Haynes 282": {
        "rho_w": 8220.0,
        "cp_w": 435.0,
        "k_w": 11.3,
        "E": 217.0e9,
        "alpha": 14.1e-6,
        "nu": 0.30,
        "Sy": 620.0e6,
        "eps_default": 0.65,
    },
    "Hastelloy X": {
        "rho_w": 8220.0,
        "cp_w": 460.0,
        "k_w": 9.2,
        "E": 205.0e9,
        "alpha": 13.2e-6,
        "nu": 0.31,
        "Sy": 355.0e6,
        "eps_default": 0.62,
    },
}

INSULATION_MATERIALS: Dict[str, Dict[str, float]] = {
    "Mineral Wool": {"rho_i": 128.0, "cp_i": 840.0, "k_i": 0.045},
    "Calcium Silicate": {"rho_i": 220.0, "cp_i": 900.0, "k_i": 0.06},
    "Aerogel Blanket": {"rho_i": 150.0, "cp_i": 1000.0, "k_i": 0.02},
}

PIPE_TEMP_PROPS: Dict[str, Dict[str, list[float]]] = {
    # Representative engineering curves for first-estimate work; replace with project-specific datasheets when available.
    "Carbon Steel": {"T": [300.0, 500.0, 700.0, 900.0, 1100.0], "cp": [470.0, 520.0, 580.0, 640.0, 690.0], "k": [47.0, 44.0, 40.0, 36.0, 32.0]},
    "Stainless 316": {"T": [300.0, 500.0, 700.0, 900.0, 1100.0], "cp": [500.0, 530.0, 560.0, 600.0, 640.0], "k": [14.5, 16.0, 17.8, 19.5, 21.2]},
    "Copper": {"T": [300.0, 500.0, 700.0, 900.0, 1100.0], "cp": [385.0, 400.0, 420.0, 440.0, 460.0], "k": [401.0, 378.0, 350.0, 320.0, 285.0]},
    "Inconel 625": {"T": [300.0, 500.0, 700.0, 900.0, 1100.0], "cp": [430.0, 470.0, 510.0, 560.0, 610.0], "k": [9.8, 11.0, 13.0, 15.8, 18.5]},
    "Haynes 282": {"T": [300.0, 500.0, 700.0, 900.0, 1100.0], "cp": [435.0, 475.0, 520.0, 575.0, 635.0], "k": [11.3, 12.8, 14.8, 17.2, 20.2]},
    "Hastelloy X": {"T": [300.0, 500.0, 700.0, 900.0, 1100.0], "cp": [460.0, 495.0, 530.0, 585.0, 640.0], "k": [9.2, 10.6, 12.4, 15.0, 18.0]},
}

INSULATION_TEMP_PROPS: Dict[str, Dict[str, list[float]]] = {
    "Mineral Wool": {"T": [300.0, 500.0, 700.0, 900.0, 1100.0], "cp": [840.0, 900.0, 960.0, 1020.0, 1080.0], "k": [0.045, 0.060, 0.080, 0.105, 0.135]},
    "Calcium Silicate": {"T": [300.0, 500.0, 700.0, 900.0, 1100.0], "cp": [900.0, 950.0, 1000.0, 1060.0, 1120.0], "k": [0.060, 0.075, 0.090, 0.108, 0.128]},
    "Aerogel Blanket": {"T": [300.0, 500.0, 700.0, 900.0, 1100.0], "cp": [1000.0, 1030.0, 1060.0, 1100.0, 1140.0], "k": [0.020, 0.024, 0.029, 0.035, 0.043]},
}



FT2M = 0.3048
IN2M = 0.0254
M2FT = 1.0 / FT2M
M2IN = 1.0 / IN2M
PSI2PA = 6894.757293168
PA2PSI = 1.0 / PSI2PA
KG_S__TO__LBM_S = 2.20462262185
LBM_S__TO__KG_S = 1.0 / KG_S__TO__LBM_S
MPA_TO_KSI = 0.1450377377
KSI_TO_MPA = 1.0 / MPA_TO_KSI








def K_to_F(value_k: float) -> float:
    return (value_k - 273.15) * 9.0 / 5.0 + 32.0


def F_to_K(value_f: float) -> float:
    return (value_f - 32.0) * 5.0 / 9.0 + 273.15




class LogEmitter(QObject):
    message = pyqtSignal(str)


class QtLogHandler(logging.Handler):
    def __init__(self, emitter: LogEmitter):
        super().__init__()
        self.emitter = emitter
        self._thermal_pipe_qt_handler = True

    def emit(self, record):
        self.emitter.message.emit(self.format(record))


class CompactTableWidget(QTableWidget):
    """Table widget with conservative size hints to avoid parent window auto-expansion."""

    def minimumSizeHint(self):
        return QSize(280, 180)

    def sizeHint(self):
        return QSize(560, 360)


from .optimization import (
    TARGET_METRIC_LABELS,
    SimulationWorker,
    _sanitize_target_metric,
    _wall_fraction_from_geom,
)
from .panels import UIPanelMixin
from .persistence import PersistenceMixin
from .plotting import PlottingMixin


class ThermalPipeWindow(UIPanelMixin, PlottingMixin, PersistenceMixin, QMainWindow):
    PRESETS: Dict[str, Dict[str, Any]] = {
        "Fast estimate": {
            "Nx": 450, "save_frames": 120, "dt_max": 4.0, "dt_min": 1e-4, "update_props_every": 10,
            "adv_scheme": "semi_lagrangian", "semi_lag_courant_max": 2.2, "use_float32": True,
        },
        "Balanced": {
            "Nx": 1300, "save_frames": 240, "dt_max": 2.5, "dt_min": 1e-4, "update_props_every": 5,
            "adv_scheme": "semi_lagrangian", "semi_lag_courant_max": 1.2, "use_float32": True,
        },
        "High fidelity": {
            "Nx": 2600, "save_frames": 320, "dt_max": 1.0, "dt_min": 1e-4, "update_props_every": 2,
            "adv_scheme": "upwind", "semi_lag_courant_max": 1.0, "use_float32": False,
        },
    }
    PIPE_DEFAULTS: Dict[str, Dict[str, Any]] = {
        "Custom / current": {},
        "Transfer Line (Insulated SS316)": {
            "pipe_material": "Stainless 316",
            "insulation": True,
            "ins_material": "Mineral Wool",
            "L": 65.0,
            "Di": 0.13,
            "t_wall": 0.018,
            "t_ins": 0.150,
            "p": 5.0e6,
            "Tin": 1000.0,
            "m_dot": 1.0,
            "Tamb": 300.0,
            "n_elbows": 0,
            "elbow_sif": 1.30,
        },
        "High-Temp Alloy (Inconel 625)": {
            "pipe_material": "Inconel 625",
            "insulation": True,
            "ins_material": "Calcium Silicate",
            "L": 45.0,
            "Di": 0.10,
            "t_wall": 0.015,
            "t_ins": 0.120,
            "p": 6.0e6,
            "Tin": 1100.0,
            "m_dot": 1.2,
            "Tamb": 320.0,
            "n_elbows": 2,
            "elbow_sif": 1.50,
            "elbow_positions": "22%, 78%",
        },
        "Bare Carbon Steel Startup": {
            "pipe_material": "Carbon Steel",
            "insulation": False,
            "L": 30.0,
            "Di": 0.09,
            "t_wall": 0.010,
            "t_ins": 0.0,
            "p": 2.0e6,
            "Tin": 700.0,
            "m_dot": 0.8,
            "Tamb": 295.0,
            "n_elbows": 1,
            "elbow_sif": 1.30,
            "elbow_positions": "60%",
        },
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thermal Pipe Simulator")
        screen = QApplication.primaryScreen()
        if screen is not None:
            ag = screen.availableGeometry()
            w = max(1080, min(1500, ag.width() - 40))
            h = max(760, min(960, ag.height() - 60))
            self.resize(w, h)
        else:
            self.resize(1400, 900)

        self._save_dir: Path | None = None
        self._thread: QThread | None = None
        self._worker: SimulationWorker | None = None
        self._units = "SI"
        self._current_L = 65.0
        self._ambient_temp = 300.0
        self._last_mech = {"E": 210e9, "alpha": 12e-6, "nu": 0.29, "Sy": 250e6}
        self._last_pipe_material = "Carbon Steel"

        self._snap_t: list[float] = []
        self._snap_outlet: list[float] = []
        self._snap_tw_out: list[float] = []
        self._snap_ti_out: list[float] = []
        self._snap_inlet: list[float] = []
        self._snap_inlet_cell: list[float] = []
        self._snap_tg_rows: list[np.ndarray] = []
        self._snapshot_counter = 0

        self._play_times = np.array([], dtype=float)
        self._play_tg = np.empty((0, 0), dtype=float)
        self._play_tw = np.empty((0, 0), dtype=float)
        self._play_ti = np.empty((0, 0), dtype=float)
        self._play_tin_eff = np.array([], dtype=float)
        self._last_result = None
        self._last_geom: Dict[str, float] = {}
        self._last_run_si: Dict[str, Any] = {}
        self._last_stats: Dict[str, Any] = {}
        self._last_warnings: list[str] = []
        self._ledger_path: Path | None = None
        self._app_root = Path(__file__).resolve().parent.parent
        self._readme_path = self._app_root / "README.md"
        self._pipe_preset_path = self._app_root / "pipe_presets.json"
        self._custom_pipe_presets: Dict[str, Dict[str, Any]] = {}
        self._heat_im = None
        self._heat_cbar = None
        self._results_axes: list[Any] = []
        self._results_cbar_parent: Dict[Any, Any] = {}
        self._cancel_pending = False

        self._play_timer = QTimer(self)
        self._play_timer.setInterval(50)
        self._play_timer.timeout.connect(self._playback_tick)

        root = QWidget(self)
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._left_panel = self._build_left_panel()
        self._left_panel.setMinimumWidth(350)
        self._left_panel.setMaximumWidth(760)
        splitter.addWidget(self._left_panel)
        splitter.addWidget(self._build_right_panel())
        splitter.setCollapsible(0, False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 980])
        root_layout.addWidget(splitter, 1)

        root_layout.addLayout(self._build_run_bar())

        self._setup_logging()
        self._load_custom_pipe_presets()
        self._refresh_pipe_default_combo()
        self._apply_preset("Fast estimate")
        self._on_pipe_material_changed(self.pipe_material.currentText())
        self._sync_mode_widgets()
        self._sync_insulation_widgets()
        self._refresh_material_lists()
        self._refresh_temp_prop_materials()
        self._refresh_pipe_preset_list()
        self._apply_unit_ranges()
        self._apply_unit_labels()
        self._update_tmass_estimate_label()
        self._reset_live_views()
        self._refresh_readme_view()
        self._refresh_ledger_preview()




















    def _setup_logging(self):
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.propagate = False
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

        self.log_emitter = LogEmitter()
        self.log_emitter.message.connect(self.log_view.appendPlainText)
        self.log_handler = QtLogHandler(self.log_emitter)
        self.log_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
        )
        root_logger.addHandler(self.log_handler)
        if not HAS_NUMBA:
            logging.warning("Numba acceleration not available; runs may be slower. Install 'numba' for best performance.")
        elif sys.platform.startswith("win"):
            logging.warning(
                "Windows stability mode active: Numba kernels are disabled by default for runs "
                "to avoid native-process crashes."
            )

    def _refresh_material_lists(self):
        if hasattr(self, "pipe_list"):
            self.pipe_list.clear()
            for name, m in PIPE_MATERIALS.items():
                sy_mpa = float(m.get("Sy", 250e6)) / 1.0e6
                t_table = PIPE_TEMP_PROPS.get(name)
                t_tag = ""
                if isinstance(t_table, dict):
                    t_vals = [float(v) for v in t_table.get("T", [])]
                    if len(t_vals) >= 2:
                        t_tag = f", Tdep={len(t_vals)}pts [{min(t_vals):.0f}-{max(t_vals):.0f} K]"
                self.pipe_list.addItem(
                    f"{name}: rho={m['rho_w']:.0f}, cp={m['cp_w']:.1f}, k={m['k_w']:.2f}, "
                    f"E={m['E']/1e9:.1f} GPa, alpha={m['alpha']*1e6:.2f}e-6, nu={m['nu']:.2f}, Sy={sy_mpa:.1f} MPa{t_tag}"
                )
        if hasattr(self, "ins_list"):
            self.ins_list.clear()
            for name, m in INSULATION_MATERIALS.items():
                t_table = INSULATION_TEMP_PROPS.get(name)
                t_tag = ""
                if isinstance(t_table, dict):
                    t_vals = [float(v) for v in t_table.get("T", [])]
                    if len(t_vals) >= 2:
                        t_tag = f", Tdep={len(t_vals)}pts [{min(t_vals):.0f}-{max(t_vals):.0f} K]"
                self.ins_list.addItem(f"{name}: rho={m['rho_i']:.0f}, cp={m['cp_i']:.1f}, k={m['k_i']:.4f}{t_tag}")

    def _refresh_material_combos(self):
        cur_pipe = self.pipe_material.currentText() if hasattr(self, "pipe_material") else None
        cur_ins = self.ins_material.currentText() if hasattr(self, "ins_material") else None
        if hasattr(self, "pipe_material"):
            self.pipe_material.blockSignals(True)
            self.pipe_material.clear()
            self.pipe_material.addItems(list(PIPE_MATERIALS))
            if cur_pipe in PIPE_MATERIALS:
                self.pipe_material.setCurrentText(cur_pipe)
            self.pipe_material.blockSignals(False)
        if hasattr(self, "ins_material"):
            self.ins_material.blockSignals(True)
            self.ins_material.clear()
            self.ins_material.addItems(list(INSULATION_MATERIALS))
            if cur_ins in INSULATION_MATERIALS:
                self.ins_material.setCurrentText(cur_ins)
            self.ins_material.blockSignals(False)
        self._refresh_temp_prop_materials()

    def _estimate_tmass_factor_from_deadleg(self) -> float:
        length_main = max(float(self._length_from_display(self.in_L.value())), 1.0e-9)
        deadleg_len = max(float(self._length_from_display(self.in_tmass_deadleg_len.value())), 0.0)
        spread_frac = max(1.0e-4, 0.01 * float(self.in_tmass_spread.value()))
        diam_ratio = max(0.1, float(self.in_tmass_deadleg_d_ratio.value()))
        # Triangular spread integrates to ~3*spread_frac of line mass for factor=1.0.
        return float((diam_ratio * (deadleg_len / length_main)) / (3.0 * spread_frac))

    def _update_tmass_estimate_label(self, *_args):
        if not hasattr(self, "lbl_tmass_est"):
            return
        est = self._estimate_tmass_factor_from_deadleg()
        l_unit = "m" if self._units == "SI" else "ft"
        self.lbl_tmass_est.setText(
            "Dead-end T rough guide: "
            f"mass factor ~= {est:.2f} per attachment "
            f"(L_dead={self.in_tmass_deadleg_len.value():.3f} {l_unit}, "
            f"D_ratio={self.in_tmass_deadleg_d_ratio.value():.2f}, spread={self.in_tmass_spread.value():.1f}%L)."
        )

    def _apply_tmass_rough_estimate(self, *_args):
        est = self._estimate_tmass_factor_from_deadleg()
        est = max(float(self.in_tmass_factor.minimum()), min(float(self.in_tmass_factor.maximum()), est))
        self._set_box_safely(self.in_tmass_factor, est)
        self._update_tmass_estimate_label()

    @staticmethod
    def _parse_float_list(text: str) -> list[float]:
        raw = str(text or "").replace(";", ",").replace("\n", ",")
        vals: list[float] = []
        for token in raw.split(","):
            tok = token.strip()
            if not tok:
                continue
            vals.append(float(tok))
        return vals

    def _active_temp_prop_store(self) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, float]]]:
        if self.temp_table_type.currentText() == "Pipe":
            return PIPE_TEMP_PROPS, PIPE_MATERIALS
        return INSULATION_TEMP_PROPS, INSULATION_MATERIALS

    def _refresh_temp_prop_materials(self, *_args):
        if not hasattr(self, "temp_table_material"):
            return
        store, base = self._active_temp_prop_store()
        current = self.temp_table_material.currentText()
        names = list(base.keys())
        self.temp_table_material.blockSignals(True)
        self.temp_table_material.clear()
        self.temp_table_material.addItems(names)
        if current in names:
            self.temp_table_material.setCurrentText(current)
        elif names:
            self.temp_table_material.setCurrentIndex(0)
        self.temp_table_material.blockSignals(False)
        if names:
            self._load_temp_prop_editor()

    def _load_temp_prop_editor(self, *_args):
        if not hasattr(self, "temp_table_material"):
            return
        store, base = self._active_temp_prop_store()
        name = self.temp_table_material.currentText().strip()
        if not name:
            return
        table = store.get(name)
        if not isinstance(table, dict):
            if self.temp_table_type.currentText() == "Pipe":
                mat = base.get(name, {})
                table = self._flat_prop_table(cp=float(mat.get("cp_w", 500.0)), k=float(mat.get("k_w", 16.0)))
            else:
                mat = base.get(name, {})
                table = self._flat_prop_table(cp=float(mat.get("cp_i", 900.0)), k=float(mat.get("k_i", 0.06)))
            store[name] = table
        self.temp_table_T.setText(", ".join(f"{float(v):g}" for v in table.get("T", [])))
        self.temp_table_cp.setText(", ".join(f"{float(v):g}" for v in table.get("cp", [])))
        self.temp_table_k.setText(", ".join(f"{float(v):g}" for v in table.get("k", [])))

    def _save_temp_prop_editor(self, *_args):
        store, _base = self._active_temp_prop_store()
        name = self.temp_table_material.currentText().strip()
        if not name:
            QMessageBox.warning(self, "Temp Property Table", "Select a material first.")
            return
        try:
            t_vals = self._parse_float_list(self.temp_table_T.text())
            cp_vals = self._parse_float_list(self.temp_table_cp.text())
            k_vals = self._parse_float_list(self.temp_table_k.text())
        except Exception:
            QMessageBox.warning(self, "Temp Property Table", "Could not parse T/cp/k lists. Use comma-separated numbers.")
            return
        if len(t_vals) < 2:
            QMessageBox.warning(self, "Temp Property Table", "At least two temperature points are required.")
            return
        if len(cp_vals) != len(t_vals) or len(k_vals) != len(t_vals):
            QMessageBox.warning(self, "Temp Property Table", "T, cp, and k lists must have matching lengths.")
            return
        if any((t_vals[i] >= t_vals[i + 1]) for i in range(len(t_vals) - 1)):
            QMessageBox.warning(self, "Temp Property Table", "T values must be strictly increasing.")
            return
        if any(v <= 0.0 for v in cp_vals) or any(v <= 0.0 for v in k_vals):
            QMessageBox.warning(self, "Temp Property Table", "cp and k values must be > 0.")
            return
        store[name] = {
            "T": [float(v) for v in t_vals],
            "cp": [float(v) for v in cp_vals],
            "k": [float(v) for v in k_vals],
        }
        self._refresh_material_lists()
        QMessageBox.information(self, "Temp Property Table", f"Saved table for {name}.")

    def _reset_temp_prop_editor_flat(self, *_args):
        store, base = self._active_temp_prop_store()
        name = self.temp_table_material.currentText().strip()
        if not name:
            return
        if self.temp_table_type.currentText() == "Pipe":
            mat = base.get(name, {})
            table = self._flat_prop_table(cp=float(mat.get("cp_w", 500.0)), k=float(mat.get("k_w", 16.0)))
        else:
            mat = base.get(name, {})
            table = self._flat_prop_table(cp=float(mat.get("cp_i", 900.0)), k=float(mat.get("k_i", 0.06)))
        store[name] = table
        self._load_temp_prop_editor()
        self._refresh_material_lists()











    def _add_pipe_material(self):
        name = self.new_pipe_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Material", "Pipe material name is required.")
            return
        PIPE_MATERIALS[name] = {
            "rho_w": self.new_pipe_rho.value(),
            "cp_w": self.new_pipe_cp.value(),
            "k_w": self.new_pipe_k.value(),
            "E": self.new_pipe_E.value() * 1.0e9,
            "alpha": self.new_pipe_alpha.value() * 1.0e-6,
            "nu": self.new_pipe_nu.value(),
            "Sy": self.new_pipe_sy.value() * 1.0e6,
            "eps_default": self.new_pipe_eps.value(),
        }
        PIPE_TEMP_PROPS[name] = self._flat_prop_table(cp=float(self.new_pipe_cp.value()), k=float(self.new_pipe_k.value()))
        self._refresh_material_combos()
        self.pipe_material.setCurrentText(name)
        self._refresh_material_lists()
        self._refresh_temp_prop_materials()
        QMessageBox.information(self, "Material", f"Saved pipe material: {name}")

    def _add_ins_material(self):
        name = self.new_ins_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Material", "Insulation material name is required.")
            return
        INSULATION_MATERIALS[name] = {
            "rho_i": self.new_ins_rho.value(),
            "cp_i": self.new_ins_cp.value(),
            "k_i": self.new_ins_k.value(),
        }
        INSULATION_TEMP_PROPS[name] = self._flat_prop_table(cp=float(self.new_ins_cp.value()), k=float(self.new_ins_k.value()))
        self._refresh_material_combos()
        self.ins_material.setCurrentText(name)
        self._refresh_material_lists()
        self._refresh_temp_prop_materials()
        QMessageBox.information(self, "Material", f"Saved insulation material: {name}")

    def _sync_mode_widgets(self):
        mode_txt = self.mode.currentText()
        is_target_based = mode_txt in ("Outlet target", "Heatup-time optimize", "Stress-limit optimize")
        is_heatup_opt = mode_txt == "Heatup-time optimize"
        is_stress_opt = mode_txt == "Stress-limit optimize"
        self.in_target.setEnabled(is_target_based)
        self.target_metric.setEnabled(is_target_based)
        self.stop_dir.setEnabled(mode_txt == "Outlet target")
        self.in_heatup_target.setEnabled(is_heatup_opt)
        self.in_heatup_tol.setEnabled(is_heatup_opt)
        self.in_stress_limit.setEnabled(is_heatup_opt or is_stress_opt)
        self.in_mdot_min.setEnabled(is_heatup_opt or is_stress_opt)
        self.in_mdot_max.setEnabled(is_heatup_opt or is_stress_opt)

    def _sync_insulation_widgets(self):
        enabled = self.chk_use_insulation.isChecked()
        self.ins_material.setEnabled(enabled)
        self.in_t_ins.setEnabled(enabled)

    def _on_pipe_material_changed(self, name: str):
        mat = PIPE_MATERIALS.get(name)
        if not mat:
            return
        self.in_eps.setValue(float(mat["eps_default"]))

    def _temp_to_display(self, value):
        if self._units == "SI":
            return value
        return (np.asarray(value) - 273.15) * 9.0 / 5.0 + 32.0

    def _temp_from_display(self, value):
        if self._units == "SI":
            return value
        return (np.asarray(value) - 32.0) * 5.0 / 9.0 + 273.15

    def _inlet_temp_eff_series_si(self, times_s: np.ndarray) -> np.ndarray:
        t = np.asarray(times_s, dtype=float)
        if t.size == 0:
            return np.empty((0,), dtype=float)
        tin_target = float(self._last_run_si.get("Tin", self._temp_from_display(self.in_Tin.value())))
        tin_start = float(self._last_run_si.get("T_init_gas", self._last_run_si.get("Tamb", self._ambient_temp)))
        ramp_s = max(0.0, float(self._last_run_si.get("Tin_ramp_s", self.in_tin_ramp.value())))
        ramp_model = str(self._last_run_si.get("Tin_ramp_model", "logistic")).strip().lower()
        if ramp_s <= 1.0e-12:
            return np.full_like(t, tin_target, dtype=float)
        if ramp_model == "linear":
            frac = np.clip(t / ramp_s, 0.0, 1.0)
        elif ramp_model == "logistic":
            k = 8.0
            u = np.clip(t / ramp_s, 0.0, 1.0)
            lo = 1.0 / (1.0 + np.exp(0.5 * k))
            hi = 1.0 / (1.0 + np.exp(-0.5 * k))
            sig = 1.0 / (1.0 + np.exp(-k * (u - 0.5)))
            frac = np.clip((sig - lo) / np.maximum(hi - lo, 1.0e-12), 0.0, 1.0)
        else:
            frac = 1.0 - np.exp(-np.log(100.0) * np.maximum(t, 0.0) / ramp_s)
            frac = np.clip(frac, 0.0, 1.0)
        return tin_start + (tin_target - tin_start) * frac

    def _inlet_temp_eff_at_si(self, t_s: float) -> float:
        vals = self._inlet_temp_eff_series_si(np.asarray([float(t_s)], dtype=float))
        if vals.size == 0:
            return float(self._last_run_si.get("Tin", self._temp_from_display(self.in_Tin.value())))
        return float(vals[0])

    def _length_to_display(self, value_m):
        if self._units == "SI":
            return value_m
        return value_m * M2FT

    def _length_from_display(self, value_disp):
        if self._units == "SI":
            return value_disp
        return value_disp * FT2M

    def _diam_to_display(self, value_m):
        if self._units == "SI":
            return value_m
        return value_m * M2IN

    def _diam_from_display(self, value_disp):
        if self._units == "SI":
            return value_disp
        return value_disp * IN2M

    def _pressure_to_display(self, value_pa):
        if self._units == "SI":
            return value_pa
        return value_pa * PA2PSI

    def _pressure_from_display(self, value_disp):
        if self._units == "SI":
            return value_disp
        return value_disp * PSI2PA

    def _mdot_to_display(self, value_kgs):
        if self._units == "SI":
            return value_kgs
        return value_kgs * KG_S__TO__LBM_S

    def _mdot_from_display(self, value_disp):
        if self._units == "SI":
            return value_disp
        return value_disp * LBM_S__TO__KG_S

    def _stress_to_display(self, value_mpa):
        if self._units == "SI":
            return value_mpa
        return value_mpa * MPA_TO_KSI

    def _stress_from_display(self, value_disp):
        if self._units == "SI":
            return value_disp
        return value_disp * KSI_TO_MPA

    @staticmethod
    def _fmt_num(value: Any, decimals: int = 1, fallback: str = "--") -> str:
        try:
            v = float(value)
        except Exception:
            return fallback
        if not np.isfinite(v):
            return fallback
        return f"{v:.{max(0, int(decimals))}f}"

    def _temp_unit_label(self) -> str:
        return "K" if self._units == "SI" else "F"

    def _length_unit_label(self) -> str:
        return "m" if self._units == "SI" else "ft"

    def _stress_unit_label(self) -> str:
        return "MPa" if self._units == "SI" else "ksi"

    def _fmt_temp_si(self, value_k: Any, decimals: int = 1) -> str:
        disp = self._temp_to_display(float(value_k))
        return self._fmt_num(disp, decimals=decimals)

    def _fmt_length_si(self, value_m: Any, decimals: int = 3) -> str:
        disp = self._length_to_display(float(value_m))
        return self._fmt_num(disp, decimals=decimals)

    def _fmt_stress_si(self, value_mpa: Any, decimals: int = 2) -> str:
        disp = self._stress_to_display(float(value_mpa))
        return self._fmt_num(disp, decimals=decimals)

    def _fmt_mdot_si(self, value_kg_s: Any, decimals: int = 3) -> str:
        disp = self._mdot_to_display(float(value_kg_s))
        return self._fmt_num(disp, decimals=decimals)

    def _fmt_time_s(self, value_s: Any, decimals: int = 1) -> str:
        return self._fmt_num(value_s, decimals=decimals)

    def _update_live_readout(
        self,
        *,
        sim_time_s: Any | None = None,
        tg_out_k: Any | None = None,
        tw_out_k: Any | None = None,
        ti_out_k: Any | None = None,
        tin_eff_k: Any | None = None,
        tg_in_k: Any | None = None,
    ):
        unit = self._temp_unit_label()
        tg_out = self._fmt_temp_si(tg_out_k, 1) if tg_out_k is not None else "--"
        tw_out = self._fmt_temp_si(tw_out_k, 1) if tw_out_k is not None else "--"
        ti_out = self._fmt_temp_si(ti_out_k, 1) if ti_out_k is not None else "--"
        tin_eff = self._fmt_temp_si(tin_eff_k, 1) if tin_eff_k is not None else "--"
        tg_in = self._fmt_temp_si(tg_in_k, 1) if tg_in_k is not None else "--"
        t_txt = self._fmt_time_s(sim_time_s, 1) if sim_time_s is not None else "--"
        if hasattr(self, "lbl_live_readout"):
            self.lbl_live_readout.setText(
                f"Time: {t_txt} s || Tg_out {tg_out} | Tw_out {tw_out} | Ti_out {ti_out} | Tin_eff {tin_eff} | Tg_in {tg_in} {unit}"
            )

    def _target_series_from_outlet_components(
        self,
        tg_out_k: np.ndarray,
        tw_out_k: np.ndarray,
        ti_out_k: np.ndarray,
    ) -> np.ndarray:
        metric = _sanitize_target_metric(str(self._last_run_si.get("target_metric", "gas_outlet")))
        tg = np.asarray(tg_out_k, dtype=float)
        tw = np.asarray(tw_out_k, dtype=float)
        ti = np.asarray(ti_out_k, dtype=float)
        if metric == "gas_outlet":
            return tg
        if metric == "wall_inner_outlet":
            return tw
        if metric == "insulation_outlet":
            return ti
        wall_frac = _wall_fraction_from_geom(self._last_geom if self._last_geom else {"Di": 0.13, "t_wall": 0.018, "t_ins": 0.0, "k_w": 15.0, "k_i": 0.05})
        return tw - wall_frac * (tw - ti)

    def _update_target_time_readout(
        self,
        *,
        times_s: np.ndarray | None = None,
        tg_out_k: np.ndarray | None = None,
        tw_out_k: np.ndarray | None = None,
        ti_out_k: np.ndarray | None = None,
    ):
        if not hasattr(self, "lbl_target_time"):
            return
        mode_target = bool(float(self._last_run_si.get("mode_target", 0.0)) >= 0.5)
        if not mode_target:
            self.lbl_target_time.setText("Target time: n/a (fixed)")
            return
        target = float(self._last_run_si.get("target", np.nan))
        if not np.isfinite(target):
            self.lbl_target_time.setText("Target time: --")
            return

        t = np.asarray(times_s if times_s is not None else self._snap_t, dtype=float)
        tg = np.asarray(tg_out_k if tg_out_k is not None else self._snap_outlet, dtype=float)
        tw = np.asarray(tw_out_k if tw_out_k is not None else self._snap_tw_out, dtype=float)
        ti = np.asarray(ti_out_k if ti_out_k is not None else self._snap_ti_out, dtype=float)
        n = int(min(t.size, tg.size, tw.size, ti.size))
        if n == 0:
            self.lbl_target_time.setText("Target time: --")
            return
        t = t[:n]
        series = self._target_series_from_outlet_components(tg[:n], tw[:n], ti[:n])
        mode_le = bool(target <= float(series[0]))
        t_hit = self._target_crossing_time(t, series, target, mode_le=mode_le)
        if t_hit is not None:
            eta = max(0.0, float(t_hit) - float(t[-1]))
            self.lbl_target_time.setText(f"Target time: {self._fmt_time_s(t_hit)} s (ETA {self._fmt_time_s(eta)} s)")
            return

        if n < 3:
            self.lbl_target_time.setText("Target time: --")
            return
        k = min(8, n)
        tt = t[-k:]
        yy = series[-k:]
        dt = float(tt[-1] - tt[0])
        if dt <= 1.0e-9:
            self.lbl_target_time.setText("Target time: --")
            return
        slope = float((yy[-1] - yy[0]) / dt)
        if (mode_le and slope >= -1.0e-12) or ((not mode_le) and slope <= 1.0e-12):
            self.lbl_target_time.setText("Target time: --")
            return
        t_pred = float(tt[-1] + (target - yy[-1]) / slope)
        if not np.isfinite(t_pred) or t_pred < float(tt[-1]):
            self.lbl_target_time.setText("Target time: --")
            return
        eta = max(0.0, t_pred - float(tt[-1]))
        self.lbl_target_time.setText(f"Target time: ~{self._fmt_time_s(t_pred)} s (ETA {self._fmt_time_s(eta)} s)")

    @staticmethod
    def _round_ledger_value(key: str, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (int, np.integer, bool)):
            return int(value)
        if not isinstance(value, (float, np.floating)):
            return value
        v = float(value)
        if not np.isfinite(v):
            return ""
        k = key.lower()
        if k.endswith("_k") or "_temp_" in k or k.startswith(("tg_", "tw_", "ti_", "tin_", "tamb_", "target_")):
            return round(v, 1)
        if k.endswith("_s"):
            return round(v, 1)
        if k.endswith("_pa"):
            return round(v, 0)
        if k.endswith("_mpa") or k.endswith("_ksi"):
            return round(v, 2)
        if "mdot" in k or k.endswith("_kg_s"):
            return round(v, 3)
        if k.endswith("_m"):
            return round(v, 4)
        if "factor" in k or "ratio" in k:
            return round(v, 3)
        return round(v, 4)

    def _format_ledger_cell(self, key: str, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (int, np.integer, bool)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            rounded = self._round_ledger_value(key, float(value))
            if rounded == "":
                return ""
            if isinstance(rounded, float) and rounded.is_integer() and key.lower().endswith("_pa"):
                return f"{int(rounded)}"
            return str(rounded)
        return str(value)

    @staticmethod
    def _set_box_safely(box: QDoubleSpinBox, value: float):
        box.blockSignals(True)
        box.setValue(float(value))
        box.blockSignals(False)

    def _apply_unit_ranges(self):
        if self._units == "SI":
            self.in_L.setRange(0.01, 1.0e6)
            self.in_Di.setRange(0.001, 1.0e3)
            self.in_t_wall.setRange(0.0001, 1.0)
            self.in_t_ins.setRange(0.0, 2.0)
            self.in_ambient.setRange(100.0, 2000.0)
            self.in_Tin.setRange(1.0, 5000.0)
            self.in_target.setRange(1.0, 5000.0)
            self.in_p.setRange(1.0, 1.0e9)
            self.in_mdot.setRange(0.0001, 1.0e4)
            self.in_mdot_min.setRange(0.0001, 1.0e4)
            self.in_mdot_max.setRange(0.0001, 1.0e4)
            self.in_stress_limit.setRange(1.0, 1.0e5)
            self.in_tmass_deadleg_len.setRange(0.0, 1.0e6)
        else:
            self.in_L.setRange(0.01 * M2FT, 1.0e6 * M2FT)
            self.in_Di.setRange(0.001 * M2IN, 1.0e3 * M2IN)
            self.in_t_wall.setRange(0.0001 * M2IN, 1.0 * M2IN)
            self.in_t_ins.setRange(0.0, 2.0 * M2IN)
            self.in_ambient.setRange(K_to_F(100.0), K_to_F(2000.0))
            self.in_Tin.setRange(K_to_F(1.0), K_to_F(5000.0))
            self.in_target.setRange(K_to_F(1.0), K_to_F(5000.0))
            self.in_p.setRange(1.0 * PA2PSI, 1.0e9 * PA2PSI)
            self.in_mdot.setRange(0.0001 * KG_S__TO__LBM_S, 1.0e4 * KG_S__TO__LBM_S)
            self.in_mdot_min.setRange(0.0001 * KG_S__TO__LBM_S, 1.0e4 * KG_S__TO__LBM_S)
            self.in_mdot_max.setRange(0.0001 * KG_S__TO__LBM_S, 1.0e4 * KG_S__TO__LBM_S)
            self.in_stress_limit.setRange(1.0 * MPA_TO_KSI, 1.0e5 * MPA_TO_KSI)
            self.in_tmass_deadleg_len.setRange(0.0, 1.0e6 * M2FT)

    def _apply_unit_labels(self):
        if self._units == "SI":
            self.in_L.setSuffix(" m")
            self.in_Di.setSuffix(" m")
            self.in_t_wall.setSuffix(" m")
            self.in_t_ins.setSuffix(" m")
            self.in_ambient.setSuffix(" K")
            self.in_Tin.setSuffix(" K")
            self.in_target.setSuffix(" K")
            self.in_p.setSuffix(" Pa")
            self.in_mdot.setSuffix(" kg/s")
            self.in_mdot_min.setSuffix(" kg/s")
            self.in_mdot_max.setSuffix(" kg/s")
            self.in_stress_limit.setSuffix(" MPa")
            self.in_tmass_deadleg_len.setSuffix(" m")
        else:
            self.in_L.setSuffix(" ft")
            self.in_Di.setSuffix(" in")
            self.in_t_wall.setSuffix(" in")
            self.in_t_ins.setSuffix(" in")
            self.in_ambient.setSuffix(" F")
            self.in_Tin.setSuffix(" F")
            self.in_target.setSuffix(" F")
            self.in_p.setSuffix(" psi")
            self.in_mdot.setSuffix(" lbm/s")
            self.in_mdot_min.setSuffix(" lbm/s")
            self.in_mdot_max.setSuffix(" lbm/s")
            self.in_stress_limit.setSuffix(" ksi")
            self.in_tmass_deadleg_len.setSuffix(" ft")
        # Keep temperature inputs and targets at engineering precision.
        for tbox in (self.in_ambient, self.in_Tin, self.in_target):
            tbox.setDecimals(1)
        if hasattr(self, "in_elbow_positions"):
            if self._units == "SI":
                self.in_elbow_positions.setPlaceholderText("auto or e.g. 45, 50 (m) or 20%, 60%")
            else:
                self.in_elbow_positions.setPlaceholderText("auto or e.g. 150, 180 (ft) or 20%, 60%")
        if hasattr(self, "in_tmass_positions"):
            if self._units == "SI":
                self.in_tmass_positions.setPlaceholderText("auto or e.g. 12, 35 (m) or 20%, 60%")
            else:
                self.in_tmass_positions.setPlaceholderText("auto or e.g. 40, 120 (ft) or 20%, 60%")
        if hasattr(self, "lbl_outlet"):
            self.lbl_outlet.setText(f"Outlet Tg: -- {'K' if self._units == 'SI' else 'F'}")
        if hasattr(self, "lbl_wall_out"):
            self.lbl_wall_out.setText(f"Outlet Tw: -- {'K' if self._units == 'SI' else 'F'}")
        if hasattr(self, "lbl_ins_out"):
            self.lbl_ins_out.setText(f"Outlet Ti: -- {'K' if self._units == 'SI' else 'F'}")
        if hasattr(self, "lbl_inlet"):
            self.lbl_inlet.setText(f"Tin_eff(bc): -- {'K' if self._units == 'SI' else 'F'}")
        if hasattr(self, "lbl_inlet_cell"):
            self.lbl_inlet_cell.setText(f"Tg_in(c0): -- {'K' if self._units == 'SI' else 'F'}")
        self._update_live_readout()
        self._update_target_time_readout()
        self._update_tmass_estimate_label()

    def _on_units_changed(self, new_units: str):
        old = self._units
        if new_units == old:
            return

        values = {
            "L": self.in_L.value(),
            "Di": self.in_Di.value(),
            "t_wall": self.in_t_wall.value(),
            "t_ins": self.in_t_ins.value(),
            "ambient": self.in_ambient.value(),
            "Tin": self.in_Tin.value(),
            "target": self.in_target.value(),
            "p": self.in_p.value(),
            "mdot": self.in_mdot.value(),
            "mdot_min": self.in_mdot_min.value(),
            "mdot_max": self.in_mdot_max.value(),
            "stress_limit": self.in_stress_limit.value(),
            "tmass_deadleg": self.in_tmass_deadleg_len.value(),
        }

        if old == "SI" and new_units == "Imperial":
            converted = {
                "L": values["L"] * M2FT,
                "Di": values["Di"] * M2IN,
                "t_wall": values["t_wall"] * M2IN,
                "t_ins": values["t_ins"] * M2IN,
                "ambient": K_to_F(values["ambient"]),
                "Tin": K_to_F(values["Tin"]),
                "target": K_to_F(values["target"]),
                "p": values["p"] * PA2PSI,
                "mdot": values["mdot"] * KG_S__TO__LBM_S,
                "mdot_min": values["mdot_min"] * KG_S__TO__LBM_S,
                "mdot_max": values["mdot_max"] * KG_S__TO__LBM_S,
                "stress_limit": values["stress_limit"] * MPA_TO_KSI,
                "tmass_deadleg": values["tmass_deadleg"] * M2FT,
            }
        elif old == "Imperial" and new_units == "SI":
            converted = {
                "L": values["L"] * FT2M,
                "Di": values["Di"] * IN2M,
                "t_wall": values["t_wall"] * IN2M,
                "t_ins": values["t_ins"] * IN2M,
                "ambient": F_to_K(values["ambient"]),
                "Tin": F_to_K(values["Tin"]),
                "target": F_to_K(values["target"]),
                "p": values["p"] * PSI2PA,
                "mdot": values["mdot"] * LBM_S__TO__KG_S,
                "mdot_min": values["mdot_min"] * LBM_S__TO__KG_S,
                "mdot_max": values["mdot_max"] * LBM_S__TO__KG_S,
                "stress_limit": values["stress_limit"] * KSI_TO_MPA,
                "tmass_deadleg": values["tmass_deadleg"] * FT2M,
            }
        else:
            return

        self._units = new_units
        self._apply_unit_ranges()
        self._apply_unit_labels()

        self._set_box_safely(self.in_L, converted["L"])
        self._set_box_safely(self.in_Di, converted["Di"])
        self._set_box_safely(self.in_t_wall, converted["t_wall"])
        self._set_box_safely(self.in_t_ins, converted["t_ins"])
        self._set_box_safely(self.in_ambient, converted["ambient"])
        self._set_box_safely(self.in_Tin, converted["Tin"])
        self._set_box_safely(self.in_target, converted["target"])
        self._set_box_safely(self.in_p, converted["p"])
        self._set_box_safely(self.in_mdot, converted["mdot"])
        self._set_box_safely(self.in_mdot_min, converted["mdot_min"])
        self._set_box_safely(self.in_mdot_max, converted["mdot_max"])
        self._set_box_safely(self.in_stress_limit, converted["stress_limit"])
        self._set_box_safely(self.in_tmass_deadleg_len, converted["tmass_deadleg"])

        if HAS_MPL and self._play_times.size > 0:
            idx = self.slider_time.value() if self.slider_time.maximum() >= 0 else 0
            self._render_playback_frame(idx)
            if self._last_result is not None:
                self._render_static_results(self._last_result)
            self.lbl_outlet.setText(f"Outlet Tg: {self._fmt_temp_si(self._play_tg[idx, -1])} {self._temp_unit_label()}")
            self.lbl_wall_out.setText(f"Outlet Tw: {self._fmt_temp_si(self._play_tw[idx, -1])} {self._temp_unit_label()}")
            self.lbl_ins_out.setText(f"Outlet Ti: {self._fmt_temp_si(self._play_ti[idx, -1])} {self._temp_unit_label()}")
            if self._play_tin_eff.size > idx:
                self.lbl_inlet.setText(f"Tin_eff(bc): {self._fmt_temp_si(self._play_tin_eff[idx])} {self._temp_unit_label()}")
            if self._play_tg.shape[0] > idx:
                self.lbl_inlet_cell.setText(
                    f"Tg_in(c0): {self._fmt_temp_si(self._play_tg[idx, 0])} {self._temp_unit_label()}"
                )
            self._update_live_readout(
                sim_time_s=self._play_times[idx] if self._play_times.size > idx else None,
                tg_out_k=self._play_tg[idx, -1],
                tw_out_k=self._play_tw[idx, -1],
                ti_out_k=self._play_ti[idx, -1],
                tin_eff_k=self._play_tin_eff[idx] if self._play_tin_eff.size > idx else None,
                tg_in_k=self._play_tg[idx, 0],
            )
        else:
            self._reset_live_views()
















    def _parse_fractional_positions(self, text: str, count: int, length_si: float) -> list[float]:
        n = int(max(0, count))
        if n == 0:
            return []
        raw = (text or "").strip().lower()
        if raw in ("", "auto", "default"):
            return [float(v) for v in np.linspace(0.15, 0.85, n)]
        length_disp = float(self._length_to_display(max(length_si, 1.0e-12)))
        vals: list[float] = []
        for token in raw.replace(";", ",").split(","):
            tok = token.strip()
            if not tok:
                continue
            is_percent = "%" in tok
            tok = tok.replace("%", "")
            try:
                v = float(tok)
            except ValueError:
                continue

            if is_percent:
                frac = v / 100.0
            elif abs(v) <= 1.0:
                frac = v
            else:
                # Plain numeric entries are interpreted as absolute distance in current length units.
                frac = v / max(length_disp, 1.0e-12)

            vals.append(max(0.0, min(1.0, float(frac))))
        if len(vals) < n:
            auto = [float(v) for v in np.linspace(0.15, 0.85, n)]
            vals.extend(auto[len(vals):])
        return vals[:n]

    def _parse_elbow_positions(self, text: str, n_elbows: int, length_si: float) -> list[float]:
        return self._parse_fractional_positions(text, n_elbows, length_si)

    def _current_target_metric(self) -> str:
        idx = self.target_metric.currentIndex()
        data = self.target_metric.itemData(idx)
        return _sanitize_target_metric(str(data))

    @staticmethod
    def _flat_prop_table(cp: float, k: float) -> Dict[str, list[float]]:
        return {"T": [300.0, 600.0, 900.0, 1200.0], "cp": [float(cp)] * 4, "k": [float(k)] * 4}

    def _resolve_pipe_prop_table(self, mat_name: str, mat: Dict[str, float]) -> Dict[str, list[float]]:
        table = PIPE_TEMP_PROPS.get(mat_name)
        if isinstance(table, dict):
            return table
        return self._flat_prop_table(cp=float(mat["cp_w"]), k=float(mat["k_w"]))

    def _resolve_ins_prop_table(self, mat_name: str, mat: Dict[str, float]) -> Dict[str, list[float]]:
        table = INSULATION_TEMP_PROPS.get(mat_name)
        if isinstance(table, dict):
            return table
        return self._flat_prop_table(cp=float(mat["cp_i"]), k=float(mat["k_i"]))

    def _apply_pipe_default(self):
        name = self.pipe_default.currentText()
        cfg = self._all_pipe_defaults().get(name, {})
        if not cfg:
            return
        if "pipe_material" in cfg and cfg["pipe_material"] in PIPE_MATERIALS:
            self.pipe_material.setCurrentText(str(cfg["pipe_material"]))
        if "ins_material" in cfg and cfg["ins_material"] in INSULATION_MATERIALS:
            self.ins_material.setCurrentText(str(cfg["ins_material"]))
        if "insulation" in cfg:
            self.chk_use_insulation.setChecked(bool(cfg["insulation"]))

        if "L" in cfg:
            self._set_box_safely(self.in_L, float(self._length_to_display(float(cfg["L"]))))
        if "Di" in cfg:
            self._set_box_safely(self.in_Di, float(self._diam_to_display(float(cfg["Di"]))))
        if "t_wall" in cfg:
            self._set_box_safely(self.in_t_wall, float(self._diam_to_display(float(cfg["t_wall"]))))
        if "t_ins" in cfg:
            self._set_box_safely(self.in_t_ins, float(self._diam_to_display(float(cfg["t_ins"]))))
        if "Tamb" in cfg:
            self._set_box_safely(self.in_ambient, float(self._temp_to_display(float(cfg["Tamb"]))))
        if "Tin" in cfg:
            self._set_box_safely(self.in_Tin, float(self._temp_to_display(float(cfg["Tin"]))))
        if "Tin_ramp_s" in cfg:
            self._set_box_safely(self.in_tin_ramp, float(cfg["Tin_ramp_s"]))
        if "Tin_ramp_model" in cfg:
            key = str(cfg["Tin_ramp_model"]).strip().lower()
            idx = self.in_tin_ramp_model.findData(key)
            if idx >= 0:
                self.in_tin_ramp_model.setCurrentIndex(idx)
        if "p" in cfg:
            self._set_box_safely(self.in_p, float(self._pressure_to_display(float(cfg["p"]))))
        if "m_dot" in cfg:
            self._set_box_safely(self.in_mdot, float(self._mdot_to_display(float(cfg["m_dot"]))))
        if "n_elbows" in cfg:
            self.in_elbows.setValue(int(cfg["n_elbows"]))
        if "elbow_sif" in cfg:
            self._set_box_safely(self.in_elbow_sif, float(cfg["elbow_sif"]))
        self.in_elbow_positions.setText(str(cfg.get("elbow_positions", cfg.get("elbow_positions_pct", ""))))
        if "n_tmass" in cfg:
            self.in_tmass_count.setValue(int(cfg["n_tmass"]))
        if "tmass_factor" in cfg:
            self._set_box_safely(self.in_tmass_factor, float(cfg["tmass_factor"]))
        if "tmass_spread_pct" in cfg:
            self._set_box_safely(self.in_tmass_spread, float(cfg["tmass_spread_pct"]))
        if "tmass_deadleg_len_m" in cfg:
            self._set_box_safely(self.in_tmass_deadleg_len, float(self._length_to_display(float(cfg["tmass_deadleg_len_m"]))))
        if "tmass_deadleg_d_ratio" in cfg:
            self._set_box_safely(self.in_tmass_deadleg_d_ratio, float(cfg["tmass_deadleg_d_ratio"]))
        self.in_tmass_positions.setText(str(cfg.get("tmass_positions", cfg.get("tmass_positions_pct", ""))))
        self._sync_insulation_widgets()
        self._update_tmass_estimate_label()

    def _apply_preset(self, name: str):
        cfg = self.PRESETS.get(name)
        if not cfg:
            return
        self.in_Nx.setValue(cfg["Nx"])
        self.in_save_frames.setValue(cfg["save_frames"])
        self.in_dt_max.setValue(cfg["dt_max"])
        self.in_dt_min.setValue(cfg["dt_min"])
        self.in_update_props.setValue(cfg["update_props_every"])
        if "adv_scheme" in cfg and hasattr(self, "in_adv_scheme"):
            idx = self.in_adv_scheme.findData(cfg["adv_scheme"])
            if idx >= 0:
                self.in_adv_scheme.setCurrentIndex(idx)
        if "semi_lag_courant_max" in cfg and hasattr(self, "in_semi_lag_cmax"):
            self._set_box_safely(self.in_semi_lag_cmax, float(cfg["semi_lag_courant_max"]))
        if "use_float32" in cfg and hasattr(self, "chk_use_float32"):
            self.chk_use_float32.setChecked(bool(cfg["use_float32"]))

    def _collect_spec(self) -> RunSpec:
        mode_ui = self.mode.currentText()
        mode = "time" if mode_ui == "Fixed time" else "target"

        pipe_mat = PIPE_MATERIALS[self.pipe_material.currentText()]
        if self.chk_use_insulation.isChecked():
            ins_mat = INSULATION_MATERIALS[self.ins_material.currentText()]
            t_ins_disp = self.in_t_ins.value()
        else:
            ins_mat = {"rho_i": 128.0, "cp_i": 840.0, "k_i": 0.045}
            t_ins_disp = 0.0

        L = self._length_from_display(self.in_L.value())
        Di = self._diam_from_display(self.in_Di.value())
        t_wall = self._diam_from_display(self.in_t_wall.value())
        t_ins = self._diam_from_display(t_ins_disp)
        ambient = float(self._temp_from_display(self.in_ambient.value()))
        Tin = float(self._temp_from_display(self.in_Tin.value()))
        target = float(self._temp_from_display(self.in_target.value()))
        target_metric = self._current_target_metric()
        p = float(self._pressure_from_display(self.in_p.value()))
        m_dot = float(self._mdot_from_display(self.in_mdot.value()))

        self._ambient_temp = ambient
        self._last_pipe_material = self.pipe_material.currentText()
        elbow_positions_frac = self._parse_elbow_positions(
            self.in_elbow_positions.text(),
            int(self.in_elbows.value()),
            L,
        )
        tmass_count = int(self.in_tmass_count.value())
        tmass_positions_frac = self._parse_fractional_positions(
            self.in_tmass_positions.text(),
            tmass_count,
            L,
        )
        self._last_mech = {
            "E": pipe_mat["E"],
            "alpha": pipe_mat["alpha"],
            "nu": pipe_mat["nu"],
            "Sy": float(pipe_mat.get("Sy", 250.0e6)),
        }
        self._last_geom = {
            "Di": Di,
            "t_wall": t_wall,
            "t_ins": t_ins,
            "k_w": float(pipe_mat["k_w"]),
            "k_i": float(ins_mat["k_i"]),
            "n_elbows": int(self.in_elbows.value()),
            "elbow_sif": float(self.in_elbow_sif.value()),
            "elbow_positions_frac": elbow_positions_frac,
            "elbow_positions_input": self.in_elbow_positions.text().strip(),
            "n_tmass": tmass_count,
            "tmass_factor": float(self.in_tmass_factor.value()),
            "tmass_positions_frac": tmass_positions_frac,
            "tmass_positions_input": self.in_tmass_positions.text().strip(),
            "tmass_spread_frac": 0.01 * float(self.in_tmass_spread.value()),
            "tmass_deadleg_len_m": float(self._length_from_display(self.in_tmass_deadleg_len.value())),
            "tmass_deadleg_d_ratio": float(self.in_tmass_deadleg_d_ratio.value()),
        }
        self._last_run_si = {
            "p": p,
            "Tin": Tin,
            "Tamb": ambient,
            "m_dot": m_dot,
            "T_init_gas": ambient,
            "T_init_wall": ambient,
            "T_init_ins": ambient,
            "L": L,
            "Di": Di,
            "t_wall": t_wall,
            "t_ins": t_ins,
            "Nx": float(self.in_Nx.value()),
            "nr_wall": float(self.in_nr_wall.value()),
            "axial_restraint": float(self.in_axial_restraint.value()),
            "ignore_inlet_cells": float(self.in_ignore_inlet_cells.value()),
            "Tin_ramp_s": float(self.in_tin_ramp.value()),
            "Tin_ramp_model": str(self.in_tin_ramp_model.currentData() or "logistic"),
            "target": target,
            "target_metric": target_metric,
            "target_metric_label": TARGET_METRIC_LABELS.get(target_metric, target_metric),
            "mode_target": 1.0 if mode == "target" else 0.0,
            "asset_id": self.asset_id.text().strip(),
            "branch_id": self.branch_id.text().strip(),
        }

        hardware = HardwareConfig(
            L=L,
            Di=Di,
            t_wall=t_wall,
            t_ins=t_ins,
            h_out=0.0,
            eps_rad=self.in_eps.value(),
        )

        run_inputs = RunInputs(
            p=p,
            Tin=Tin,
            m_dot=m_dot,
            mode=mode,
            t_end=self.in_t_end.value(),
            Tg_out_target=target if mode == "target" else None,
            stop_dir=None if self.stop_dir.currentText() == "auto" else self.stop_dir.currentText(),
        )

        overrides = {
            "Nx": self.in_Nx.value(),
            "save_frames": self.in_save_frames.value(),
            "dt_max": self.in_dt_max.value(),
            "dt_min": self.in_dt_min.value(),
            "Tin_ramp_s": self.in_tin_ramp.value(),
            "Tin_ramp_model": str(self.in_tin_ramp_model.currentData() or "logistic"),
            "update_props_every": self.in_update_props.value(),
            "adv_scheme": str(self.in_adv_scheme.currentData() or "semi_lagrangian"),
            "semi_lag_courant_max": float(self.in_semi_lag_cmax.value()),
            "progress": self.progress.currentText(),
            "use_float32": self.chk_use_float32.isChecked(),
            "enable_numba": (False if sys.platform.startswith("win") else bool(HAS_NUMBA)),
            "log_to_file": self.chk_log_to_file.isChecked(),
            "write_trace_csv": self.chk_write_trace.isChecked(),
            "target_asymptote_check": self.chk_target_asymptote.isChecked(),
            # GUI worker-thread runs must not call pyplot popups directly.
            "show_plots": False,
            "rho_w": pipe_mat["rho_w"],
            "cp_w": pipe_mat["cp_w"],
            "k_w": pipe_mat["k_w"],
            "rho_i": ins_mat["rho_i"],
            "cp_i": ins_mat["cp_i"],
            "k_i": ins_mat["k_i"],
            "pipe_prop_table": self._resolve_pipe_prop_table(self.pipe_material.currentText(), pipe_mat),
            "ins_prop_table": self._resolve_ins_prop_table(self.ins_material.currentText(), ins_mat),
            "use_temp_dependent_props": bool(self.chk_temp_dep_props.isChecked()),
            "target_metric": target_metric,
            "thermal_mass_count": tmass_count,
            "thermal_mass_factor": float(self.in_tmass_factor.value()),
            "thermal_mass_positions_frac": tmass_positions_frac,
            "thermal_mass_spread_frac": 0.01 * float(self.in_tmass_spread.value()),
            "Tamb": ambient,
            "eps_rad": self.in_eps.value(),
            "T_init_wall": ambient,
            "T_init_ins": ambient,
            "T_init_gas": ambient,
            "h_out_mode": "auto",
            "h_out": 0.0,
            "insulation_mass_mode": str(self.in_ins_mass_mode.currentData() or "penetration"),
            "insulation_mass_min_frac": float(self.in_ins_mass_min_frac.value()),
        }

        return RunSpec(
            hardware=hardware,
            run=run_inputs,
            overrides=overrides,
            save_dir=self._save_dir,
            # Save/export images is handled in the Qt main thread after completion.
            make_plots=False,
            save_results=self.chk_save_results.isChecked(),
        )

    def _collect_optimization_config(self) -> Dict[str, Any] | None:
        mode_ui = self.mode.currentText()
        if mode_ui not in ("Heatup-time optimize", "Stress-limit optimize"):
            return None

        mdot_min = float(self._mdot_from_display(self.in_mdot_min.value()))
        mdot_max = float(self._mdot_from_display(self.in_mdot_max.value()))
        if mdot_min <= 0.0 or mdot_max <= 0.0 or mdot_max <= mdot_min:
            raise ValueError("Search mass flow bounds must satisfy 0 < m_dot_min < m_dot_max.")

        stress_limit_mpa = float(self._stress_from_display(self.in_stress_limit.value()))
        if stress_limit_mpa <= 0.0:
            raise ValueError("Stress limit must be > 0.")

        target_temp = float(self._temp_from_display(self.in_target.value()))
        target_metric = self._current_target_metric()
        cfg: Dict[str, Any] = {
            "mode": "heatup_time_opt" if mode_ui == "Heatup-time optimize" else "stress_limit_opt",
            "target_temp_K": target_temp,
            "target_metric": target_metric,
            "mdot_min_kg_s": mdot_min,
            "mdot_max_kg_s": mdot_max,
            "stress_limit_mpa": stress_limit_mpa,
            "include_pressure": bool(self.chk_include_pressure.isChecked()),
            "nr_wall": int(self.in_nr_wall.value()),
            "geom": dict(self._last_geom),
            "run_si": dict(self._last_run_si),
            "mech": dict(self._last_mech),
            "search_nx": int(max(80, min(int(self.in_Nx.value()), max(120, int(0.35 * self.in_Nx.value()))))),
            "opt_t_end_s": float(self.in_t_end.value()),
            "coarse_points": 9,
            "refine_iters": 6,
            "bisection_iters": 12,
            "stress_tol_mpa": 2.0,
        }
        if cfg["mode"] == "heatup_time_opt":
            cfg["heatup_target_s"] = float(self.in_heatup_target.value())
            cfg["heatup_tol_s"] = float(self.in_heatup_tol.value())
            cfg["opt_t_end_s"] = max(
                float(self.in_t_end.value()),
                float(self.in_heatup_target.value()) + max(60.0, 3.0 * float(self.in_heatup_tol.value())),
            )
        return cfg





    def _run_clicked(self):
        if self._thread is not None and self._thread.isRunning():
            return

        try:
            spec = self._collect_spec()
            opt_cfg = self._collect_optimization_config()
        except Exception as exc:
            QMessageBox.critical(self, "Input error", str(exc))
            return

        self._current_L = float(spec.hardware.L)
        self._last_result = None
        self._last_stats = {}
        self._last_warnings = []
        self._reset_live_views()
        self._update_target_time_readout()
        if hasattr(self, "btn_export"):
            self.btn_export.setEnabled(False)
        self._cancel_pending = False

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.status.setText("Running...")
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)

        self._thread = QThread(self)
        self._worker = SimulationWorker(spec, optimization=opt_cfg)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.snapshot.connect(self._on_snapshot)
        self._worker.finished.connect(self._on_finished)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _cancel_clicked(self):
        if self._worker is None or self._thread is None or not self._thread.isRunning():
            return
        self._cancel_pending = True
        try:
            self._worker.request_cancel()
        except Exception:
            pass
        self.status.setText("Cancelling...")
        self.btn_cancel.setEnabled(False)

    @pyqtSlot(float, object, object, object)
    def _on_snapshot(self, t_s: float, tg_obj, tw_obj, ti_obj):
        self._snapshot_counter += 1
        tg = np.asarray(tg_obj, dtype=float)
        tw = np.asarray(tw_obj, dtype=float)
        ti = np.asarray(ti_obj, dtype=float)
        if tg.size == 0:
            return

        self._snap_t.append(float(t_s))
        self._snap_outlet.append(float(tg[-1]))
        self._snap_tw_out.append(float(tw[-1]))
        self._snap_ti_out.append(float(ti[-1]))
        tin_eff_si = self._inlet_temp_eff_at_si(float(t_s))
        self._snap_inlet.append(tin_eff_si)
        self._snap_inlet_cell.append(float(tg[0]))
        self._snap_tg_rows.append(tg.astype(np.float32, copy=True))

        self.lbl_runtime.setText(f"Sim time: {self._fmt_time_s(t_s)} s")
        self.lbl_outlet.setText(f"Outlet Tg: {self._fmt_temp_si(float(tg[-1]))} {self._temp_unit_label()}")
        self.lbl_wall_out.setText(f"Outlet Tw: {self._fmt_temp_si(float(tw[-1]))} {self._temp_unit_label()}")
        self.lbl_ins_out.setText(f"Outlet Ti: {self._fmt_temp_si(float(ti[-1]))} {self._temp_unit_label()}")
        self.lbl_inlet.setText(f"Tin_eff(bc): {self._fmt_temp_si(float(tin_eff_si))} {self._temp_unit_label()}")
        self.lbl_inlet_cell.setText(f"Tg_in(c0): {self._fmt_temp_si(float(tg[0]))} {self._temp_unit_label()}")
        self._update_live_readout(
            sim_time_s=float(t_s),
            tg_out_k=float(tg[-1]),
            tw_out_k=float(tw[-1]),
            ti_out_k=float(ti[-1]),
            tin_eff_k=float(tin_eff_si),
            tg_in_k=float(tg[0]),
        )
        self._update_target_time_readout()
        self.lbl_frames.setText(f"Frames: {len(self._snap_t)}")
        t_cap = max(float(self.in_t_end.value()), 1.0e-9)
        frac = max(0.0, min(1.0, float(t_s) / t_cap))
        self.progress_bar.setValue(int(round(1000.0 * frac)))

        if not HAS_MPL:
            return

        x = self._length_to_display(np.linspace(0.0, self._current_L, tg.size))
        self.line_outlet.set_data(self._snap_t, self._temp_to_display(np.asarray(self._snap_outlet)))
        self.line_inlet.set_data(self._snap_t, self._temp_to_display(np.asarray(self._snap_inlet)))
        self.line_inlet_cell.set_data(self._snap_t, self._temp_to_display(np.asarray(self._snap_inlet_cell)))
        self.ax_outlet.relim()
        self.ax_outlet.autoscale_view()

        self.line_tg.set_data(x, self._temp_to_display(tg))
        self.line_tw.set_data(x, self._temp_to_display(tw))
        self.line_ti.set_data(x, self._temp_to_display(ti))
        self.ax_profile.relim()
        self.ax_profile.autoscale_view()
        self.live_canvas.draw_idle()

        if self._snap_tg_rows:
            heat = self._temp_to_display(np.vstack(self._snap_tg_rows))
            x_max = float(self._length_to_display(self._current_L))
            extent = [0.0, x_max, self._snap_t[-1], self._snap_t[0]]
            self._update_live_heatmap(np.asarray(heat, dtype=float), extent)

    @pyqtSlot(object)
    def _on_finished(self, result):
        if str(getattr(result, "stop_reason", "")) == "aborted_by_user" or self._cancel_pending:
            self._on_cancelled(result)
            return
        self._last_result = result
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if hasattr(self, "btn_export"):
            self.btn_export.setEnabled(True)
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(1000)

        msg = f"Done: {result.stop_reason}; Tg_out={self._fmt_temp_si(result.Tg_outlet_final)} {self._temp_unit_label()}"
        opt_summary = getattr(result, "opt_summary", None)
        if isinstance(opt_summary, dict):
            md_disp = self._fmt_mdot_si(float(opt_summary.get("m_dot_kg_s", self._last_run_si.get("m_dot", 0.0))))
            md_unit = "kg/s" if self._units == "SI" else "lbm/s"
            stress_ok = bool(opt_summary.get("meets_stress_limit", False))
            reached_ok = bool(opt_summary.get("target_reached_final", False))
            if opt_summary.get("mode") == "heatup_time_opt":
                t_final = opt_summary.get("time_to_target_final_s")
                time_ok = bool(opt_summary.get("meets_heatup_tolerance", False))
                if t_final is not None:
                    msg += f" | opt m_dot={md_disp} {md_unit}, t_hit={self._fmt_time_s(float(t_final))}s"
                else:
                    msg += f" | opt m_dot={md_disp} {md_unit}"
                msg += f" | ok: target={reached_ok}, time={time_ok}, stress={stress_ok}"
            elif opt_summary.get("mode") == "stress_limit_opt":
                t_final = opt_summary.get("time_to_target_final_s")
                if t_final is not None:
                    msg += f" | stress-opt m_dot={md_disp} {md_unit}, t_hit={self._fmt_time_s(float(t_final))}s"
                else:
                    msg += f" | stress-opt m_dot={md_disp} {md_unit}"
                msg += f" | ok: target={reached_ok}, stress={stress_ok}"
        if result.outdir:
            msg += f" | saved -> {result.outdir}"
        self.status.setText(msg)
        logging.info(msg)
        if isinstance(opt_summary, dict) and result.outdir:
            try:
                outdir = Path(result.outdir)
                outdir.mkdir(parents=True, exist_ok=True)
                (outdir / "optimization_summary.json").write_text(
                    json.dumps(opt_summary, indent=2, default=float),
                    encoding="utf-8",
                )
            except Exception as exc:
                logging.warning("Could not save optimization_summary.json: %s", exc)

        self._set_playback_data(result)
        try:
            self._render_static_results(result)
        except Exception as exc:
            self._results_axes = []
            logging.exception("Failed to render embedded result plots: %s", exc)
        if self.chk_make_plots.isChecked():
            self._save_plot_images_from_result(result)
        if self.chk_show_plots.isChecked() and HAS_MPL and self._results_axes:
            try:
                self._show_results_popup(self._results_axes[0])
            except Exception as exc:
                logging.warning("Could not open post-run results popup: %s", exc)
        if self.chk_append_ledger.isChecked():
            try:
                self._append_run_ledger(result)
            except Exception as exc:
                logging.warning("Failed to append run ledger: %s", exc)

        if HAS_MPL:
            self.live_canvas.draw_idle()
            self.heat_canvas.draw_idle()

    @pyqtSlot(str)
    def _on_failed(self, tb_text: str):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self._cancel_pending = False
        self.status.setText("Error")
        QMessageBox.critical(self, "Simulation error", tb_text)

    @pyqtSlot()
    def _on_cancelled(self, result=None):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.status.setText("Cancelled by user")
        self._last_result = None
        self._last_stats = {}
        self._last_warnings = []
        self._cancel_pending = False
        # Discard all aborted run data/visuals.
        self._reset_live_views()
        if hasattr(self, "btn_export"):
            self.btn_export.setEnabled(False)
        outdir = getattr(result, "outdir", None) if result is not None else None
        if outdir:
            try:
                shutil.rmtree(Path(outdir), ignore_errors=True)
            except Exception as exc:
                logging.warning("Could not remove aborted run directory: %s", exc)
        logging.info("Simulation cancelled by user; partial data discarded.")

    @pyqtSlot()
    def _on_thread_finished(self):
        self.btn_cancel.setEnabled(False)
        self._cancel_pending = False
        self._worker = None
        self._thread = None







    def _compute_stress_v2(
        self,
        tw_si: np.ndarray,
        ti_si: np.ndarray,
        nr_wall: int,
    ) -> Dict[str, np.ndarray | float]:
        nr = int(max(3, nr_wall))
        E = float(self._last_mech.get("E", 210.0e9))
        alpha = float(self._last_mech.get("alpha", 12.0e-6))
        nu = float(self._last_mech.get("nu", 0.30))
        Di_m = float(self._last_geom.get("Di", max(self._diam_from_display(self.in_Di.value()), 1.0e-6)))
        t_wall_m = float(self._last_geom.get("t_wall", max(self._diam_from_display(self.in_t_wall.value()), 1.0e-6)))
        t_ins_m = float(self._last_geom.get("t_ins", 0.0))
        k_w = float(self._last_geom.get("k_w", 15.0))
        k_i = float(self._last_geom.get("k_i", 0.04))
        p_si = float(self._last_run_si.get("p", self._pressure_from_display(self.in_p.value())))
        Tamb = float(self._last_run_si.get("Tamb", self._ambient_temp))
        axial_restraint = float(self._last_run_si.get("axial_restraint", self.in_axial_restraint.value()))
        include_pressure = bool(self.chk_include_pressure.isChecked())
        n_elbows = int(self._last_geom.get("n_elbows", self.in_elbows.value()))
        elbow_sif = float(self._last_geom.get("elbow_sif", self.in_elbow_sif.value()))
        elbow_positions_frac = list(self._last_geom.get("elbow_positions_frac", []))

        ri = max(0.5 * Di_m, 1.0e-9)
        ro = ri + max(t_wall_m, 1.0e-9)
        x_len = max(float(self._current_L), 1.0e-6)
        nx = max(1, tw_si.shape[1])
        dx_si = x_len / max(1, nx - 1)

        R_wall = np.log(max(ro / ri, 1.0 + 1.0e-12)) / (2.0 * np.pi * max(k_w, 1.0e-12) * max(dx_si, 1.0e-12))
        if t_ins_m > 1.0e-12:
            r_ins_o = ro + t_ins_m
            R_ins = np.log(max(r_ins_o / ro, 1.0 + 1.0e-12)) / (2.0 * np.pi * max(k_i, 1.0e-12) * max(dx_si, 1.0e-12))
        else:
            R_ins = 0.0
        wall_frac = float(R_wall / max(R_wall + R_ins, 1.0e-12))

        deltaT_wall_si = np.abs(tw_si - ti_si) * wall_frac
        xi = np.linspace(0.0, 1.0, nr, dtype=float)
        r = ri + xi * max(t_wall_m, 1.0e-9)
        grad_term = (0.5 - xi)[:, None, None] * deltaT_wall_si[None, :, :]
        temp_r = tw_si[None, :, :] + grad_term

        thermo_coeff = E * alpha / max(1.0e-9, (1.0 - nu))
        sigma_theta_th = thermo_coeff * (tw_si[None, :, :] - temp_r)
        sigma_z_th = -axial_restraint * thermo_coeff * (tw_si - Tamb)[None, :, :]
        sigma_r_th = np.zeros_like(sigma_theta_th)

        if include_pressure:
            denom = max(ro * ro - ri * ri, 1.0e-12)
            r2 = np.maximum(r * r, 1.0e-12)[:, None, None]
            coeff = p_si * ri * ri / denom
            sigma_r_p = coeff * (1.0 - (ro * ro) / r2)
            sigma_theta_p = coeff * (1.0 + (ro * ro) / r2)
            sigma_z_p = np.full_like(sigma_r_p, coeff)
        else:
            sigma_r_p = np.zeros_like(temp_r)
            sigma_theta_p = np.zeros_like(temp_r)
            sigma_z_p = np.zeros_like(temp_r)

        sigma_vm_th = np.sqrt(
            0.5
            * (
                (sigma_theta_th - sigma_z_th) ** 2
                + (sigma_z_th - sigma_r_th) ** 2
                + (sigma_r_th - sigma_theta_th) ** 2
            )
        )

        sigma_theta = sigma_theta_th + sigma_theta_p
        sigma_r = sigma_r_p
        sigma_z = sigma_z_th + sigma_z_p

        sigma_vm_total = np.sqrt(
            0.5
            * (
                (sigma_theta - sigma_z) ** 2
                + (sigma_z - sigma_r) ** 2
                + (sigma_r - sigma_theta) ** 2
            )
        )

        vm_thermal_mpa = np.max(sigma_vm_th, axis=0) / 1.0e6
        vm_total_mpa = np.max(sigma_vm_total, axis=0) / 1.0e6
        vm_inner_mpa = sigma_vm_total[0, :, :] / 1.0e6
        vm_outer_mpa = sigma_vm_total[-1, :, :] / 1.0e6

        elbow_profile = np.ones(nx, dtype=float)
        if n_elbows > 0 and nx > 2:
            if len(elbow_positions_frac) != n_elbows:
                elbow_positions_frac = self._parse_elbow_positions(
                    str(self._last_geom.get("elbow_positions_input", "")),
                    n_elbows,
                    x_len,
                )
            spread = max(1, int(0.03 * nx))
            for frac in elbow_positions_frac:
                j0 = int(round(max(0.0, min(1.0, float(frac))) * (nx - 1)))
                for dj in range(-3 * spread, 3 * spread + 1):
                    j = j0 + dj
                    if j < 0 or j >= nx:
                        continue
                    w = max(0.0, 1.0 - abs(dj) / (3.0 * spread + 1.0e-9))
                    elbow_profile[j] = max(elbow_profile[j], 1.0 + (elbow_sif - 1.0) * w)

        vm_total_elbow_mpa = vm_total_mpa * elbow_profile[None, :]
        vm_thermal_elbow_mpa = vm_thermal_mpa * elbow_profile[None, :]
        elbow_factor = float(np.max(elbow_profile))

        return {
            "deltaT_wall_si": deltaT_wall_si,
            "vm_map_mpa": vm_total_mpa,
            "vm_map_elbow_mpa": vm_total_elbow_mpa,
            "vm_thermal_mpa": vm_thermal_mpa,
            "vm_thermal_elbow_mpa": vm_thermal_elbow_mpa,
            "vm_inner_mpa": vm_inner_mpa,
            "vm_outer_mpa": vm_outer_mpa,
            "nr_wall": float(nr),
            "wall_frac": wall_frac,
            "elbow_factor": elbow_factor,
            "elbow_profile": elbow_profile,
        }

    @staticmethod
    def _target_crossing_time(times: np.ndarray, values: np.ndarray, target: float, mode_le: bool) -> float | None:
        if values.size == 0 or times.size == 0:
            return None
        cond = values <= target if mode_le else values >= target
        hit = np.where(cond)[0]
        if hit.size == 0:
            return None
        i = int(hit[0])
        if i == 0:
            return float(times[0])
        t0, t1 = float(times[i - 1]), float(times[i])
        y0, y1 = float(values[i - 1]), float(values[i])
        if abs(y1 - y0) <= 1.0e-12:
            return t1
        frac = (target - y0) / (y1 - y0)
        frac = max(0.0, min(1.0, frac))
        return t0 + frac * (t1 - t0)

    def _build_health_warnings(
        self,
        tw_si: np.ndarray,
        vm_map_elbow_mpa: np.ndarray,
        times: np.ndarray,
        ignore_inlet_cells: int,
        j_max: int,
    ) -> list[str]:
        warnings: list[str] = []
        Sy_mpa = float(self._last_mech.get("Sy", 250.0e6)) / 1.0e6
        alpha = float(self._last_mech.get("alpha", 12.0e-6))
        n_elbows = int(self._last_geom.get("n_elbows", self.in_elbows.value()))
        elbow_factor = float(self._last_geom.get("elbow_sif", self.in_elbow_sif.value())) if n_elbows > 0 else 1.0

        vm_max = float(np.nanmax(vm_map_elbow_mpa))
        tw_max = float(np.nanmax(tw_si))
        tw_min = float(np.nanmin(tw_si))
        thermal_strain_range = alpha * max(0.0, tw_max - tw_min)

        if vm_max >= Sy_mpa:
            warnings.append(
                f"High: max stress indicator {vm_max:.1f} MPa exceeds nominal yield {Sy_mpa:.1f} MPa (screening)."
            )
        elif vm_max >= 0.8 * Sy_mpa:
            warnings.append(
                f"Warning: max stress indicator {vm_max:.1f} MPa is above 80% of nominal yield {Sy_mpa:.1f} MPa."
            )

        if thermal_strain_range >= 2.0e-3:
            warnings.append(
                f"High cycle-thermal strain range ({thermal_strain_range:.4f}) may drive low-cycle fatigue/ratcheting."
            )
        elif thermal_strain_range >= 1.0e-3:
            warnings.append(
                f"Moderate thermal strain range ({thermal_strain_range:.4f}); monitor cyclic loading and hold times."
            )

        if tw_max >= 1050.0:
            warnings.append(
                f"High wall temperature {tw_max:.1f} K: creep/oxidation mechanisms can accelerate."
            )
        elif tw_max >= 900.0:
            warnings.append(
                f"Elevated wall temperature {tw_max:.1f} K: include high-temperature degradation checks."
            )

        dwell_high = float(times[-1] - times[0]) if times.size > 1 and tw_max >= 900.0 else 0.0
        ratchet_index = (vm_max / max(Sy_mpa, 1.0e-6)) * (thermal_strain_range / 1.0e-3) * (1.0 + dwell_high / 3600.0)
        if ratchet_index >= 1.5:
            warnings.append(
                f"Ratcheting screening index {ratchet_index:.2f} is high; consider inelastic assessment for cyclic service."
            )
        elif ratchet_index >= 1.0:
            warnings.append(
                f"Ratcheting screening index {ratchet_index:.2f} is moderate; track cycle accumulation in run ledger."
            )

        if ignore_inlet_cells > 0 and j_max < ignore_inlet_cells:
            warnings.append(
                "Peak stress occurs in early inlet cells; verify with inlet ramp and mesh sensitivity diagnostics."
            )

        if n_elbows > 0:
            elbow_input = str(self._last_geom.get("elbow_positions_input", self.in_elbow_positions.text())).strip()
            if not elbow_input:
                warnings.append(
                    "Elbow positions are auto-distributed; enter explicit positions for branch/layout-specific screening."
                )
            warnings.append(
                f"Elbow screening active: applied stress amplification factor {elbow_factor:.2f} (count={n_elbows})."
            )

        if not warnings:
            warnings.append("No immediate screening warnings for this run. Continue monitoring with logged history.")
        return warnings











    def closeEvent(self, event):
        self._play_timer.stop()
        if hasattr(self, "_worker") and self._worker is not None:
            self._worker.canceled()
        if hasattr(self, "_thread") and self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
        
        root_logger = logging.getLogger()
        if getattr(self, "log_handler", None) is not None:
            try:
                root_logger.removeHandler(self.log_handler)
            except Exception:
                pass
            try:
                self.log_handler.close()
            except Exception:
                pass
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = ThermalPipeWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
