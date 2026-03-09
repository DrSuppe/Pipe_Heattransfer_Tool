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
"""Plot rendering and interaction mixin for the thermal pipe application."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from .optimization import TARGET_METRIC_LABELS, _outlet_series_for_metric, _sanitize_target_metric, _wall_fraction_from_geom
from .window import M2IN

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QFileDialog, QMessageBox, QDialog, QVBoxLayout
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


class PlottingMixin:
    def _clear_heat_colorbar(self):
        if self._heat_cbar is not None:
            try:
                self._heat_cbar.remove()
            except Exception:
                pass
            self._heat_cbar = None

        if HAS_MPL and hasattr(self, "heat_fig") and hasattr(self, "ax_heat"):
            # Colorbars add extra axes; keep only the main heatmap axis.
            for ax in list(self.heat_fig.axes):
                if ax is not self.ax_heat:
                    try:
                        self.heat_fig.delaxes(ax)
                    except Exception:
                        pass

    def _update_live_heatmap(self, heat: np.ndarray, extent: list[float]):
        if not HAS_MPL:
            return
        if heat.size == 0:
            return

        if self._heat_im is None or self._heat_im.axes is not self.ax_heat:
            self._heat_im = self.ax_heat.imshow(heat, aspect="auto", extent=extent, origin="upper")
            self._clear_heat_colorbar()
            self._heat_cbar = self.heat_fig.colorbar(
                self._heat_im,
                ax=self.ax_heat,
                label=("K" if self._units == "SI" else "F"),
            )
        else:
            self._heat_im.set_data(heat)
            self._heat_im.set_extent(extent)
            vmin = float(np.nanmin(heat))
            vmax = float(np.nanmax(heat))
            if vmax <= vmin:
                vmax = vmin + 1.0e-6
            self._heat_im.set_clim(vmin, vmax)
            if self._heat_cbar is not None:
                self._heat_cbar.set_label("K" if self._units == "SI" else "F")
                self._heat_cbar.update_normal(self._heat_im)
        self.heat_canvas.draw_idle()

    def _reset_live_views(self):
        self._snap_t.clear()
        self._snap_outlet.clear()
        self._snap_tw_out.clear()
        self._snap_ti_out.clear()
        self._snap_inlet.clear()
        self._snap_inlet_cell.clear()
        self._snap_tg_rows.clear()
        self._snapshot_counter = 0
        self.lbl_runtime.setText("Sim time: 0.0 s")
        self.lbl_outlet.setText(f"Outlet Tg: -- {'K' if self._units == 'SI' else 'F'}")
        self.lbl_wall_out.setText(f"Outlet Tw: -- {'K' if self._units == 'SI' else 'F'}")
        self.lbl_ins_out.setText(f"Outlet Ti: -- {'K' if self._units == 'SI' else 'F'}")
        self.lbl_inlet.setText(f"Tin_eff(bc): -- {'K' if self._units == 'SI' else 'F'}")
        self.lbl_inlet_cell.setText(f"Tg_in(c0): -- {'K' if self._units == 'SI' else 'F'}")
        self._update_live_readout()
        if hasattr(self, "lbl_target_time"):
            self.lbl_target_time.setText("Target time: --")
        self.lbl_frames.setText("Frames: 0")
        self._play_timer.stop()

        self._play_times = np.array([], dtype=float)
        self._play_tg = np.empty((0, 0), dtype=float)
        self._play_tw = np.empty((0, 0), dtype=float)
        self._play_ti = np.empty((0, 0), dtype=float)
        self._play_tin_eff = np.array([], dtype=float)
        self.slider_time.setEnabled(False)
        self.slider_time.setRange(0, 0)
        self.lbl_time_cursor.setText("t = -- s / -- s")

        if not HAS_MPL:
            return

        self._clear_heat_colorbar()

        self.ax_outlet.clear()
        self.ax_outlet.set_title("Outlet and Inlet Setpoint vs Simulation Time")
        self.ax_outlet.set_xlabel("time [s]")
        self.ax_outlet.set_ylabel(f"Temperature [{'K' if self._units == 'SI' else 'F'}]")
        (self.line_outlet,) = self.ax_outlet.plot([], [], color="#2a9d8f", linewidth=2.0, label="Tg_out")
        (self.line_inlet,) = self.ax_outlet.plot([], [], color="#6c757d", linewidth=1.4, linestyle="--", label="Tin_eff")
        (self.line_inlet_cell,) = self.ax_outlet.plot([], [], color="#264653", linewidth=1.2, linestyle=":", label="Tg_in")
        self.ax_outlet.grid(alpha=0.25)
        self.ax_outlet.legend(loc="best")

        self.ax_profile.clear()
        self.ax_profile.set_title("Current Axial Temperature Profile")
        self.ax_profile.set_xlabel(f"x [{'m' if self._units == 'SI' else 'ft'}]")
        self.ax_profile.set_ylabel(f"Temperature [{'K' if self._units == 'SI' else 'F'}]")
        (self.line_tg,) = self.ax_profile.plot([], [], "--", linewidth=1.5, label="Gas Tg")
        (self.line_tw,) = self.ax_profile.plot([], [], "-", linewidth=1.5, label="Wall Tw")
        (self.line_ti,) = self.ax_profile.plot([], [], ":", linewidth=1.5, label="Insulation Ti")
        self.ax_profile.grid(alpha=0.25)
        self.ax_profile.legend(loc="upper right")

        self.ax_heat.clear()
        self.ax_heat.set_title("Live Heatmap (Gas Temperature)")
        self.ax_heat.set_xlabel(f"x [{'m' if self._units == 'SI' else 'ft'}]")
        self.ax_heat.set_ylabel("time [s]")
        self._heat_im = None

        self.live_canvas.draw_idle()
        self.heat_canvas.draw_idle()

        if hasattr(self, "results_fig"):
            self.results_fig.clear()
            self._results_axes = []
            self._results_cbar_parent = {}
            self.results_canvas.draw_idle()
        if hasattr(self, "stats_box"):
            self.stats_box.clear()
        if hasattr(self, "warning_box"):
            self.warning_box.clear()

    def _set_playback_data(self, result):
        self._play_times = np.asarray(result.times, dtype=float)
        self._play_tg = np.asarray(result.Tg_hist, dtype=float)
        self._play_tw = np.asarray(result.Tw_hist, dtype=float)
        self._play_ti = np.asarray(result.Ti_hist, dtype=float)
        self._play_tin_eff = self._inlet_temp_eff_series_si(self._play_times)

        if self._play_times.size == 0:
            self.slider_time.setEnabled(False)
            self.slider_time.setRange(0, 0)
            self.lbl_time_cursor.setText("t = -- s / -- s")
            return

        self.slider_time.setEnabled(True)
        self.slider_time.setRange(0, self._play_times.size - 1)
        self.slider_time.setValue(self._play_times.size - 1)
        self._render_playback_frame(self._play_times.size - 1)

    def _run_animation(self):
        if self._play_times.size == 0:
            return
        if self.slider_time.value() >= self.slider_time.maximum():
            self.slider_time.setValue(0)
        self._play_timer.start()

    def _pause_animation(self):
        self._play_timer.stop()

    def _playback_tick(self):
        if self._play_times.size == 0:
            self._play_timer.stop()
            return
        idx = self.slider_time.value() + 1
        if idx > self.slider_time.maximum():
            self._play_timer.stop()
            return
        self.slider_time.setValue(idx)

    def _on_slider_changed(self, idx: int):
        self._render_playback_frame(int(idx))

    def _render_playback_frame(self, idx: int):
        if self._play_times.size == 0 or not HAS_MPL:
            return
        idx = max(0, min(idx, self._play_times.size - 1))
        temp_unit = "K" if self._units == "SI" else "F"
        x_unit = "m" if self._units == "SI" else "ft"

        t = float(self._play_times[idx])
        t_end = float(self._play_times[-1])
        self.lbl_runtime.setText(f"Sim time: {self._fmt_time_s(t)} s")
        self.lbl_time_cursor.setText(f"t = {self._fmt_time_s(t)} s / {self._fmt_time_s(t_end)} s")
        self.lbl_outlet.setText(f"Outlet Tg: {self._fmt_temp_si(self._play_tg[idx, -1])} {temp_unit}")
        self.lbl_wall_out.setText(f"Outlet Tw: {self._fmt_temp_si(self._play_tw[idx, -1])} {temp_unit}")
        self.lbl_ins_out.setText(f"Outlet Ti: {self._fmt_temp_si(self._play_ti[idx, -1])} {temp_unit}")
        if self._play_tin_eff.size > idx:
            self.lbl_inlet.setText(f"Tin_eff(bc): {self._fmt_temp_si(self._play_tin_eff[idx])} {temp_unit}")
        self.lbl_inlet_cell.setText(f"Tg_in(c0): {self._fmt_temp_si(self._play_tg[idx, 0])} {temp_unit}")
        self._update_live_readout(
            sim_time_s=t,
            tg_out_k=self._play_tg[idx, -1],
            tw_out_k=self._play_tw[idx, -1],
            ti_out_k=self._play_ti[idx, -1],
            tin_eff_k=self._play_tin_eff[idx] if self._play_tin_eff.size > idx else None,
            tg_in_k=self._play_tg[idx, 0],
        )
        self._update_target_time_readout(
            times_s=self._play_times[: idx + 1],
            tg_out_k=self._play_tg[: idx + 1, -1],
            tw_out_k=self._play_tw[: idx + 1, -1],
            ti_out_k=self._play_ti[: idx + 1, -1],
        )

        x = self._length_to_display(np.linspace(0.0, self._current_L, self._play_tg.shape[1]))
        tg = self._play_tg[idx]
        tw = self._play_tw[idx]
        ti = self._play_ti[idx]

        self.line_outlet.set_data(self._play_times[: idx + 1], self._temp_to_display(self._play_tg[: idx + 1, -1]))
        if self._play_tin_eff.size > 0:
            self.line_inlet.set_data(self._play_times[: idx + 1], self._temp_to_display(self._play_tin_eff[: idx + 1]))
        self.line_inlet_cell.set_data(self._play_times[: idx + 1], self._temp_to_display(self._play_tg[: idx + 1, 0]))
        self.ax_outlet.set_ylabel(f"Temperature [{temp_unit}]")
        self.ax_outlet.relim()
        self.ax_outlet.autoscale_view()

        self.line_tg.set_data(x, self._temp_to_display(tg))
        self.line_tw.set_data(x, self._temp_to_display(tw))
        self.line_ti.set_data(x, self._temp_to_display(ti))
        self.ax_profile.set_xlabel(f"x [{x_unit}]")
        self.ax_profile.set_ylabel(f"Temperature [{temp_unit}]")
        self.ax_profile.relim()
        self.ax_profile.autoscale_view()
        self.live_canvas.draw_idle()

        heat = self._temp_to_display(self._play_tg[: idx + 1])
        x_max = float(self._length_to_display(self._current_L))
        extent = [0.0, x_max, self._play_times[idx], self._play_times[0]]
        self.ax_heat.set_xlabel(f"x [{x_unit}]")
        self._update_live_heatmap(np.asarray(heat, dtype=float), extent)

    def _on_results_plot_click(self, event):
        if not HAS_MPL:
            return
        if event is None or event.inaxes is None:
            return
        source_ax = self._results_cbar_parent.get(event.inaxes, event.inaxes)
        if source_ax not in self._results_axes:
            return
        try:
            button = int(event.button) if event.button is not None else 0
        except Exception:
            button = 0
        if button == 3:
            self._save_axis_plot_dialog(source_ax)
            return
        if button != 1:
            return
        self._show_results_popup(source_ax)

    def _on_live_plot_click(self, event):
        if not HAS_MPL:
            return
        if event is None or event.inaxes is None:
            return
        if self._play_times.size == 0:
            return
        if self._thread is not None and self._thread.isRunning():
            return
        source_ax = event.inaxes
        if source_ax not in (self.ax_outlet, self.ax_profile, self.ax_heat):
            return
        try:
            button = int(event.button) if event.button is not None else 0
        except Exception:
            button = 0
        if button == 3:
            self._save_axis_plot_dialog(source_ax)
            return
        if button != 1:
            return
        self._show_results_popup(source_ax)

    @staticmethod
    def _slugify_filename(title: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9._-]+", "_", str(title or "").strip())
        clean = clean.strip("._")
        return clean or "plot"

    def _save_axis_plot_dialog(self, source_ax):
        if source_ax is None:
            return
        base = self._save_dir if self._save_dir is not None else Path.cwd()
        default_name = f"{self._slugify_filename(source_ax.get_title())}.png"
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot As",
            str(base / default_name),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not out_path:
            return
        try:
            fig = Figure(figsize=(10, 7), constrained_layout=True)
            ax = fig.add_subplot(1, 1, 1)
            self._copy_axis_content(source_ax, ax, fig)
            fig.savefig(out_path, dpi=220)
            logging.info("Saved plot to %s", out_path)
        except Exception as exc:
            logging.warning("Failed to save plot image: %s", exc)

    def _show_results_popup(self, source_ax):
        dlg = QDialog(self)
        dlg.setWindowTitle(source_ax.get_title() or "Plot Detail")
        dlg.resize(1000, 700)
        layout = QVBoxLayout(dlg)
        # Keep popup layout fixed to avoid resize jitter during hover redraws.
        fig = Figure(figsize=(10, 7), constrained_layout=False)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas, 1)

        ax = fig.add_subplot(1, 1, 1)
        self._copy_axis_content(source_ax, ax, fig)
        try:
            fig.tight_layout(pad=1.2)
        except Exception:
            pass
        self._attach_popup_hover(canvas, ax)
        canvas.draw_idle()
        dlg.exec()

    @staticmethod
    def _copy_axis_content(source_ax, target_ax, target_fig):
        axis_aspect = source_ax.get_aspect() if hasattr(source_ax, "get_aspect") else "auto"
        for line in source_ax.get_lines():
            label = line.get_label()
            if label.startswith("_"):
                label = "_nolegend_"
            target_ax.plot(
                line.get_xdata(),
                line.get_ydata(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=line.get_linewidth(),
                marker=line.get_marker(),
                markersize=line.get_markersize(),
                alpha=line.get_alpha(),
                label=label,
            )
        for image in source_ax.images:
            data = np.asarray(image.get_array())
            copied = target_ax.imshow(
                data,
                aspect=axis_aspect,
                extent=image.get_extent(),
                origin=getattr(image, "origin", "upper"),
                cmap=image.get_cmap(),
                vmin=image.get_clim()[0],
                vmax=image.get_clim()[1],
            )
            cbar_label = ""
            colorbar = getattr(image, "colorbar", None)
            if colorbar is not None:
                try:
                    cbar_label = str(colorbar.ax.get_ylabel() or "")
                except Exception:
                    cbar_label = ""
            target_fig.colorbar(copied, ax=target_ax, label=cbar_label)

        target_ax.set_title(source_ax.get_title(), fontsize=13, pad=10)
        target_ax.set_xlabel(source_ax.get_xlabel(), fontsize=11)
        target_ax.set_ylabel(source_ax.get_ylabel(), fontsize=11)
        target_ax.tick_params(labelsize=10)
        target_ax.set_xlim(source_ax.get_xlim())
        target_ax.set_ylim(source_ax.get_ylim())
        if any(gl.get_visible() for gl in source_ax.get_xgridlines() + source_ax.get_ygridlines()):
            target_ax.grid(alpha=0.25)

        handles, labels = target_ax.get_legend_handles_labels()
        if any(lbl and lbl != "_nolegend_" for lbl in labels):
            target_ax.legend(loc="best", fontsize=10, framealpha=0.9)

    @staticmethod
    def _attach_popup_hover(canvas, ax):
        annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1f1f1f", ec="#bbbbbb", alpha=0.9),
            color="white",
        )
        try:
            # Exclude hover annotation from layout calculations to prevent plot twitching.
            annot.set_in_layout(False)
        except Exception:
            pass
        annot.set_visible(False)
        pixel_tol = 18.0
        last_key = {"value": None}

        def _hide():
            if annot.get_visible():
                annot.set_visible(False)
                last_key["value"] = None
                canvas.draw_idle()

        def _show(xy, text, key):
            if last_key["value"] == key and annot.get_visible():
                return
            annot.xy = xy
            annot.set_text(text)
            annot.set_visible(True)
            last_key["value"] = key
            canvas.draw_idle()

        def _line_hover(event):
            best = None
            for line in ax.get_lines():
                x = np.asarray(line.get_xdata(), dtype=float)
                y = np.asarray(line.get_ydata(), dtype=float)
                if x.size == 0 or y.size == 0:
                    continue
                pts = np.column_stack((x, y))
                disp = ax.transData.transform(pts)
                d = np.hypot(disp[:, 0] - float(event.x), disp[:, 1] - float(event.y))
                i = int(np.argmin(d))
                dist = float(d[i])
                if best is None or dist < best[0]:
                    best = (dist, line, i, x[i], y[i])
            return best

        def _image_hover(event):
            if not ax.images:
                return None
            im = ax.images[0]
            data = np.asarray(im.get_array(), dtype=float)
            if data.ndim < 2:
                return None
            x0, x1, y0, y1 = [float(v) for v in im.get_extent()]
            if abs(x1 - x0) <= 1.0e-12 or abs(y1 - y0) <= 1.0e-12:
                return None
            fx = (float(event.xdata) - min(x0, x1)) / abs(x1 - x0)
            fy = (float(event.ydata) - min(y0, y1)) / abs(y1 - y0)
            if fx < 0.0 or fx > 1.0 or fy < 0.0 or fy > 1.0:
                return None
            nrows, ncols = data.shape[0], data.shape[1]
            col = int(np.clip(round(fx * (ncols - 1)), 0, ncols - 1))
            origin = str(getattr(im, "origin", "upper")).lower()
            if origin == "upper":
                row = int(np.clip(round((1.0 - fy) * (nrows - 1)), 0, nrows - 1))
            else:
                row = int(np.clip(round(fy * (nrows - 1)), 0, nrows - 1))
            return row, col, float(data[row, col])

        def _on_move(event):
            if event is None or event.inaxes is not ax or event.xdata is None or event.ydata is None:
                _hide()
                return
            line_hit = _line_hover(event)
            if line_hit is not None and line_hit[0] <= pixel_tol:
                _dist, line, i, xv, yv = line_hit
                label = str(line.get_label() or "line")
                if not label or label.startswith("_"):
                    label = "line"
                _show(
                    (xv, yv),
                    f"{label}\nx={xv:.4g}\ny={yv:.4g}\nindex={i}",
                    ("line", label, int(i)),
                )
                return
            image_hit = _image_hover(event)
            if image_hit is not None:
                row, col, val = image_hit
                _show(
                    (float(event.xdata), float(event.ydata)),
                    (
                        f"heatmap\nx={event.xdata:.4g}\ny={event.ydata:.4g}\n"
                        f"value={val:.4g}\nrow={row}, col={col}"
                    ),
                    ("image", int(row), int(col)),
                )
                return
            _hide()

        canvas.mpl_connect("motion_notify_event", _on_move)
        canvas.mpl_connect("axes_leave_event", lambda _evt: _hide())

    def _save_plot_images_from_result(self, result):
        if not HAS_MPL:
            return
        outdir = getattr(result, "outdir", None)
        if outdir is None:
            logging.warning("Plot image save requested, but no run output directory is available.")
            return
        try:
            outdir = Path(outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            saved_files: list[str] = []
            x_si = np.asarray(result.x, dtype=float)
            times = np.asarray(result.times, dtype=float)
            tw_si = np.asarray(result.Tw_hist, dtype=float)
            tg_si = np.asarray(result.Tg_hist, dtype=float)
            ti_si = np.asarray(result.Ti_hist, dtype=float)
            if times.size == 0 or x_si.size == 0:
                return

            x = np.asarray(self._length_to_display(x_si), dtype=float)
            tw = np.asarray(self._temp_to_display(tw_si), dtype=float)
            tg = np.asarray(self._temp_to_display(tg_si), dtype=float)
            ti = np.asarray(self._temp_to_display(ti_si), dtype=float)
            t = np.asarray(times, dtype=float)
            temp_unit = self._temp_unit_label()
            x_unit = "m" if self._units == "SI" else "ft"

            heat_fig = Figure(figsize=(15, 4), constrained_layout=True)
            axw = heat_fig.add_subplot(1, 3, 1)
            axg = heat_fig.add_subplot(1, 3, 2)
            axi = heat_fig.add_subplot(1, 3, 3)
            ext = [float(x[0]), float(x[-1]), float(t[-1]), float(t[0])]
            imw = axw.imshow(tw, aspect="auto", extent=ext)
            img = axg.imshow(tg, aspect="auto", extent=ext)
            imi = axi.imshow(ti, aspect="auto", extent=ext)
            axw.set_title("Wall Tw(x,t)")
            axg.set_title("Gas Tg(x,t)")
            axi.set_title("Insulation Ti(x,t)")
            for ax in (axw, axg, axi):
                ax.set_xlabel(f"x [{x_unit}]")
                ax.set_ylabel("time [s]")
            heat_fig.colorbar(imw, ax=axw, label=temp_unit)
            heat_fig.colorbar(img, ax=axg, label=temp_unit)
            heat_fig.colorbar(imi, ax=axi, label=temp_unit)
            heat_fig.savefig(outdir / "heatmaps.png", dpi=200)
            saved_files.append("heatmaps.png")

            wall_frac = _wall_fraction_from_geom(
                self._last_geom
                if self._last_geom
                else {"Di": 0.13, "t_wall": 0.018, "t_ins": 0.0, "k_w": 15.0, "k_i": 0.05}
            )
            tw_surface_si = tw_si - wall_frac * (tw_si - ti_si)
            tw_surface = np.asarray(self._temp_to_display(tw_surface_si), dtype=float)
            wall_fig = Figure(figsize=(8, 4), constrained_layout=True)
            axws = wall_fig.add_subplot(1, 1, 1)
            imws = axws.imshow(tw_surface, aspect="auto", extent=ext, origin="upper")
            axws.set_title("Wall Surface Heatmap Tw(outside, x,t)")
            axws.set_xlabel(f"x [{x_unit}]")
            axws.set_ylabel("time [s]")
            wall_fig.colorbar(imws, ax=axws, label=temp_unit)
            wall_fig.savefig(outdir / "wall_temp_surface_heatmap.png", dpi=220)
            saved_files.append("wall_temp_surface_heatmap.png")

            prof_fig = Figure(figsize=(10, 4), constrained_layout=True)
            axp = prof_fig.add_subplot(1, 1, 1)
            nmax = 30
            idx = np.linspace(0, max(0, t.size - 1), min(nmax, max(1, t.size)), dtype=int)
            for i in idx:
                axp.plot(x, tw[i], linewidth=1.0, alpha=0.8)
            for i in idx:
                axp.plot(x, tg[i], "--", linewidth=1.0, alpha=0.8)
            for i in idx:
                axp.plot(x, ti[i], ":", linewidth=1.0, alpha=0.8)
            axp.set_xlabel(f"x [{x_unit}]")
            axp.set_ylabel(f"Temperature [{temp_unit}]")
            axp.set_title("Profiles Over Time")
            axp.grid(alpha=0.25)
            prof_fig.savefig(outdir / "profiles.png", dpi=200)
            saved_files.append("profiles.png")

            if hasattr(self, "results_fig") and self._results_axes:
                self.results_fig.savefig(outdir / "results_6panel.png", dpi=220)
                saved_files.append("results_6panel.png")
                for n, source_ax in enumerate(self._results_axes, start=1):
                    title_slug = self._slugify_filename(source_ax.get_title())
                    fig_single = Figure(figsize=(10, 7), constrained_layout=True)
                    ax_single = fig_single.add_subplot(1, 1, 1)
                    self._copy_axis_content(source_ax, ax_single, fig_single)
                    fname = f"results_plot_{n:02d}_{title_slug}.png"
                    fig_single.savefig(outdir / fname, dpi=220)
                    saved_files.append(fname)

            logging.info("Saved plot images: %s", ", ".join(saved_files))
        except Exception as exc:
            logging.warning("Failed to save plot images: %s", exc)

    def _render_static_results(self, result):
        if not HAS_MPL or not hasattr(self, "results_fig"):
            return
        if result.times.size == 0:
            self.results_fig.clear()
            self._results_axes = []
            self._results_cbar_parent = {}
            self.results_canvas.draw_idle()
            if hasattr(self, "stats_box"):
                self.stats_box.setPlainText("No results available.")
            if hasattr(self, "warning_box"):
                self.warning_box.setPlainText("No warnings available.")
            self._last_stats = {}
            self._last_warnings = []
            return

        self.results_fig.clear()
        axs = self.results_fig.subplots(2, 3)
        self._results_axes = [axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 1], axs[1, 2]]
        self._results_cbar_parent = {}

        x_si = np.asarray(result.x, dtype=float)
        times = np.asarray(result.times, dtype=float)
        tg_si = np.asarray(result.Tg_hist, dtype=float)
        tw_si = np.asarray(result.Tw_hist, dtype=float)
        ti_si = np.asarray(result.Ti_hist, dtype=float)

        temp_unit = self._temp_unit_label()
        x_unit = self._length_unit_label()
        stress_unit = self._stress_unit_label()

        x = np.asarray(self._length_to_display(x_si), dtype=float)
        tg = np.asarray(self._temp_to_display(tg_si), dtype=float)
        tw = np.asarray(self._temp_to_display(tw_si), dtype=float)
        ti = np.asarray(self._temp_to_display(ti_si), dtype=float)
        target_metric = _sanitize_target_metric(str(self._last_run_si.get("target_metric", "gas_outlet")))
        target_metric_label = TARGET_METRIC_LABELS.get(target_metric, target_metric)
        target_series_si = _outlet_series_for_metric(tg_si, tw_si, ti_si, target_metric, self._last_geom)
        target_series = np.asarray(self._temp_to_display(target_series_si), dtype=float)
        tin_eff_si = self._inlet_temp_eff_series_si(times)
        tin_eff = np.asarray(self._temp_to_display(tin_eff_si), dtype=float)
        ramp_model = str(self._last_run_si.get("Tin_ramp_model", "logistic"))
        ramp_s = float(self._last_run_si.get("Tin_ramp_s", 0.0))
        alpha = float(self._last_mech.get("alpha", 12.0e-6))
        nr_wall = int(self.in_nr_wall.value())
        ignore_inlet_cells = int(self.in_ignore_inlet_cells.value())
        stress_v2 = self._compute_stress_v2(tw_si, ti_si, nr_wall=nr_wall)
        vm_map_mpa = np.asarray(stress_v2["vm_map_mpa"], dtype=float)
        vm_map_elbow_mpa = np.asarray(stress_v2["vm_map_elbow_mpa"], dtype=float)
        deltaT_wall_si = np.asarray(stress_v2["deltaT_wall_si"], dtype=float)

        vm_map_disp = self._stress_to_display(vm_map_mpa)
        vm_map_elbow_disp = self._stress_to_display(vm_map_elbow_mpa)
        vm_time_disp = np.max(vm_map_disp, axis=1)
        vm_time_elbow_disp = np.max(vm_map_elbow_disp, axis=1)

        # 1) Time-distance heatmap
        ax = axs[0, 0]
        im = ax.imshow(tg, aspect="auto", extent=[x[0], x[-1], times[-1], times[0]], origin="upper")
        ax.set_title("Gas Heatmap Tg(x,t)", fontsize=12, pad=8)
        ax.set_xlabel(f"x [{x_unit}]")
        ax.set_ylabel("time [s]")
        cbar_t = self.results_fig.colorbar(im, ax=ax, label=temp_unit, fraction=0.046, pad=0.04)
        self._results_cbar_parent[cbar_t.ax] = ax

        # 2) Waterfall-style profiles
        ax = axs[0, 1]
        idx = np.linspace(0, times.size - 1, min(14, times.size), dtype=int)
        for i in idx:
            ax.plot(x, tg[i], linewidth=1.0, alpha=0.85, label=f"t={times[i]:.0f}s")
        ax.set_title("Waterfall Tg(x)", fontsize=12, pad=8)
        ax.set_xlabel(f"x [{x_unit}]")
        ax.set_ylabel(f"Tg [{temp_unit}]")
        ax.grid(alpha=0.25)

        # 3) Outlet temperatures over time
        ax = axs[0, 2]
        ax.plot(times, tg[:, -1], label="Tg_out", linewidth=1.8)
        ax.plot(times, tg[:, 0], label="Tg_in(c0)", linewidth=1.2, linestyle=":")
        ax.plot(times, tw[:, -1], label="Tw_out", linewidth=1.2)
        ax.plot(times, ti[:, -1], label="Ti_out", linewidth=1.2)
        ax.plot(times, tin_eff, label="Tin_eff", linewidth=1.2, linestyle="--", color="#6c757d")
        if target_metric == "wall_outer_outlet":
            ax.plot(times, target_series, label="Tw_out(surface)", linewidth=1.2, linestyle="-.")
        if result.Tg_outlet_target is not None:
            target_disp = float(self._temp_to_display(float(result.Tg_outlet_target)))
            ax.axhline(target_disp, color="#666666", linestyle="--", linewidth=1.0, label="Target")
        ax.set_title("Outlet & Inlet vs Time", fontsize=12, pad=8)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"Temperature [{temp_unit}]")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.88)

        # 4) Final total stress profile (straight vs elbow-adjusted)
        ax = axs[1, 0]
        ax.plot(x, vm_map_disp[-1], color="#d55e00", linewidth=1.6, label="Total (straight)")
        if np.nanmax(vm_map_elbow_disp - vm_map_disp) > 1.0e-9:
            ax.plot(x, vm_map_elbow_disp[-1], color="#cc79a7", linewidth=1.2, linestyle="--", label="Total (elbow)")
        ax.set_title("Final Total Stress Profile", fontsize=12, pad=8)
        ax.set_xlabel(f"x [{x_unit}]")
        ax.set_ylabel(f"sigma [{stress_unit}]")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.88)

        # 5) Max total stress vs time
        ax = axs[1, 1]
        ax.plot(times, vm_time_disp, color="#d55e00", linewidth=1.5, label="Total (straight)")
        if np.nanmax(vm_time_elbow_disp - vm_time_disp) > 1.0e-9:
            ax.plot(times, vm_time_elbow_disp, color="#cc79a7", linewidth=1.2, linestyle="--", label="Total (elbow)")
        ax.set_title("Max Total Stress vs Time", fontsize=12, pad=8)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"max sigma [{stress_unit}]")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.88)

        # 6) Stress map over time and distance
        ax = axs[1, 2]
        im_sigma = ax.imshow(
            vm_map_elbow_disp,
            aspect="auto",
            extent=[x[0], x[-1], times[-1], times[0]],
            origin="upper",
        )
        ax.set_title("Total Stress Heatmap |sigma_vm|", fontsize=12, pad=8)
        ax.set_xlabel(f"x [{x_unit}]")
        ax.set_ylabel("time [s]")
        cbar_s = self.results_fig.colorbar(im_sigma, ax=ax, label=stress_unit, fraction=0.046, pad=0.04)
        self._results_cbar_parent[cbar_s.ax] = ax

        for a in self._results_axes:
            a.tick_params(labelsize=9)

        # Keep explicit spacing modest; constrained_layout handles final fit.
        try:
            self.results_fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.04, wspace=0.04, hspace=0.05)
        except Exception:
            pass

        # Statistics text
        i_max, j_max = np.unravel_index(np.nanargmax(vm_map_elbow_mpa), vm_map_elbow_mpa.shape)
        sigma_max_mpa = float(vm_map_mpa[i_max, j_max])
        sigma_max_elbow_mpa = float(vm_map_elbow_mpa[i_max, j_max])
        t_sigma_max = float(times[i_max])
        x_sigma_max_m = float(x_si[j_max])
        x_sigma_max_disp = float(x[j_max])

        dL_m = float(alpha * np.trapezoid((tw_si[-1] - float(self._ambient_temp)), x_si))
        if self._units == "SI":
            dL_text = f"{self._fmt_num(dL_m * 1000.0, 2)} mm"
        else:
            dL_text = f"{self._fmt_num(dL_m * M2IN, 3)} in"

        t_hit_s = None
        if result.Tg_outlet_target is None:
            target_time_text = "N/A (fixed-time mode)"
        else:
            target_si = float(result.Tg_outlet_target)
            mode_le = bool(target_si <= float(target_series_si[0]))
            t_hit_s = self._target_crossing_time(times, target_series_si, target_si, mode_le=mode_le)
            target_time_text = f"{self._fmt_time_s(t_hit_s)} s" if t_hit_s is not None else "Not reached"

        i_dt, j_dt = np.unravel_index(np.nanargmax(deltaT_wall_si), deltaT_wall_si.shape)
        deltaT_wall_scale = 1.0 if self._units == "SI" else 9.0 / 5.0
        max_deltaT_wall = float(deltaT_wall_si[i_dt, j_dt] * deltaT_wall_scale)
        t_max_dT = float(times[i_dt])
        x_max_dT = float(x[j_dt])

        sigma_max_disp = self._fmt_stress_si(sigma_max_mpa, 2)
        sigma_max_elbow_disp = self._fmt_stress_si(sigma_max_elbow_mpa, 2)
        wall_frac = _wall_fraction_from_geom(self._last_geom)
        tw_outer_surface_series_si = tw_si[:, -1] - wall_frac * (tw_si[:, -1] - ti_si[:, -1])

        tg_out_final_disp = self._fmt_temp_si(tg_si[-1, -1], 1)
        tg_in_final_disp = self._fmt_temp_si(tg_si[-1, 0], 1)
        tw_out_final_disp = self._fmt_temp_si(tw_si[-1, -1], 1)
        tw_in_final_disp = self._fmt_temp_si(tw_si[-1, 0], 1)
        tw_outer_surface_final_disp = self._fmt_temp_si(tw_outer_surface_series_si[-1], 1)
        ti_out_final_disp = self._fmt_temp_si(ti_si[-1, -1], 1)
        ti_in_final_disp = self._fmt_temp_si(ti_si[-1, 0], 1)
        tin_eff_final_disp = self._fmt_temp_si(tin_eff_si[-1], 1) if tin_eff_si.size else "--"
        tg_mean_final_disp = self._fmt_temp_si(np.mean(tg_si[-1]), 1)
        tw_mean_final_disp = self._fmt_temp_si(np.mean(tw_si[-1]), 1)
        ti_mean_final_disp = self._fmt_temp_si(np.mean(ti_si[-1]), 1)
        target_final_disp = self._fmt_temp_si(target_series_si[-1], 1)

        radial_diag_text = "off"
        axial_diag_text = "off"
        artifact_ratio_text = "off"
        radial_delta = np.nan
        axial_delta = np.nan
        sigma_excluding_inlet_disp = np.nan
        hotspot_inlet_flag = bool(ignore_inlet_cells > 0 and j_max < ignore_inlet_cells)
        if self.chk_convergence_diag.isChecked():
            nr_ref = max(2 * nr_wall, nr_wall + 2)
            stress_refined = self._compute_stress_v2(tw_si, ti_si, nr_wall=nr_ref)
            vm_ref_mpa = np.asarray(stress_refined["vm_map_elbow_mpa"], dtype=float)
            sigma_ref_mpa = float(np.nanmax(vm_ref_mpa))
            radial_delta = abs(sigma_max_elbow_mpa - sigma_ref_mpa) / max(abs(sigma_ref_mpa), 1.0e-9)
            radial_diag_text = f"{self._fmt_num(100.0 * radial_delta, 1)}% (Nr={nr_wall}->{nr_ref})"

            vm_coarse = vm_map_elbow_mpa[:, ::2] if vm_map_elbow_mpa.shape[1] > 2 else vm_map_elbow_mpa
            sigma_coarse_mpa = float(np.nanmax(vm_coarse))
            axial_delta = abs(sigma_max_elbow_mpa - sigma_coarse_mpa) / max(abs(sigma_max_elbow_mpa), 1.0e-9)
            axial_diag_text = f"{self._fmt_num(100.0 * axial_delta, 1)}% (coarse-x)"

            if ignore_inlet_cells < vm_map_elbow_mpa.shape[1]:
                sigma_excluding_inlet_mpa = float(np.nanmax(vm_map_elbow_mpa[:, ignore_inlet_cells:]))
                sigma_excluding_inlet_disp = float(self._stress_to_display(sigma_excluding_inlet_mpa))
                artifact_ratio = sigma_max_elbow_mpa / max(sigma_excluding_inlet_mpa, 1.0e-9)
                artifact_ratio_text = f"{self._fmt_num(artifact_ratio, 2)}x"
            else:
                artifact_ratio_text = "n/a"

        tracking_error_disp = (
            float(self._temp_to_display(float(tin_eff_si[-1]))) - float(self._temp_to_display(float(tg_si[-1, 0])))
            if tin_eff_si.size
            else np.nan
        )

        stats_lines = [
            "Run Statistics",
            f"- Time to target condition ({target_metric_label}): {target_time_text}",
            f"- Max total stress indicator (straight): {sigma_max_disp} {stress_unit}",
            f"- Max total stress indicator (elbow-adjusted): {sigma_max_elbow_disp} {stress_unit}",
            f"- Time of max stress: {self._fmt_time_s(t_sigma_max)} s",
            f"- Location of max stress: {self._fmt_num(x_sigma_max_disp, 3)} {x_unit}",
            f"- Free thermal expansion estimate: {dL_text}",
            f"- Max estimated wall through-thickness dT: {self._fmt_num(max_deltaT_wall, 1)} {temp_unit}",
            f"- Time/location of max wall dT: {self._fmt_time_s(t_max_dT)} s @ {self._fmt_num(x_max_dT, 3)} {x_unit}",
            f"- Radial sensitivity diagnostic: {radial_diag_text}",
            f"- Axial sensitivity diagnostic: {axial_diag_text}",
            f"- Inlet-hotspot ratio (global / excluding first cells): {artifact_ratio_text}",
            f"- Final target variable ({target_metric_label}): {target_final_disp} {temp_unit}",
            f"- Final gas temperature: inlet={tg_in_final_disp}, outlet={tg_out_final_disp} {temp_unit}",
            f"- Final inlet tracking error (Tin_eff - Tg_in): {self._fmt_num(tracking_error_disp, 1)} {temp_unit}",
            f"- Final wall temperature (inner node): inlet={tw_in_final_disp}, outlet={tw_out_final_disp} {temp_unit}",
            f"- Final wall outer-surface estimate at outlet: {tw_outer_surface_final_disp} {temp_unit}",
            f"- Final insulation temperature: inlet={ti_in_final_disp}, outlet={ti_out_final_disp} {temp_unit}",
            f"- Final axial mean temperatures: Tg={tg_mean_final_disp}, Tw={tw_mean_final_disp}, Ti={ti_mean_final_disp} {temp_unit}",
            f"- Inlet heater setpoint at end: {tin_eff_final_disp} {temp_unit} (model={ramp_model}, warm-up={self._fmt_time_s(ramp_s)} s)",
            f"- Simulation runtime: sim={self._fmt_time_s(times[-1])} s, steps={int(result.n_steps)}, wall={self._fmt_num(result.wall_time_s, 2)} s",
        ]
        opt_summary = getattr(result, "opt_summary", None)
        if isinstance(opt_summary, dict):
            md_disp = self._fmt_mdot_si(float(opt_summary.get("m_dot_kg_s", np.nan)))
            md_unit = "kg/s" if self._units == "SI" else "lbm/s"
            mode_tag = str(opt_summary.get("mode", ""))
            sigma_lim_disp = self._fmt_stress_si(float(opt_summary.get("stress_limit_mpa", np.nan)), 2)
            sigma_final_disp = self._fmt_stress_si(float(opt_summary.get("sigma_final_mpa", np.nan)), 2)
            meets_stress = bool(opt_summary.get("meets_stress_limit", False))
            reached_final = bool(opt_summary.get("target_reached_final", False))
            if mode_tag == "heatup_time_opt":
                t_goal = float(opt_summary.get("heatup_target_s", np.nan))
                t_tol = float(opt_summary.get("heatup_tol_s", np.nan))
                t_hit = opt_summary.get("time_to_target_final_s")
                t_hit_txt = "not reached" if t_hit is None else f"{self._fmt_time_s(float(t_hit))} s"
                meets_heatup = bool(opt_summary.get("meets_heatup_tolerance", False))
                stats_lines.extend(
                    [
                        "- Optimization mode: Heatup-time optimize",
                        f"- Selected mass flow: {md_disp} {md_unit}",
                        f"- Heatup target: {self._fmt_time_s(t_goal)} s +/- {self._fmt_time_s(t_tol)} s",
                        f"- Final run time-to-target: {t_hit_txt}",
                        f"- Final total stress: {sigma_final_disp} {stress_unit} (limit {sigma_lim_disp} {stress_unit})",
                        f"- Constraint status: target_reached={reached_final}, heatup_tol={meets_heatup}, stress_limit={meets_stress}",
                    ]
                )
            elif mode_tag == "stress_limit_opt":
                t_hit = opt_summary.get("time_to_target_final_s")
                t_hit_txt = "not reached" if t_hit is None else f"{self._fmt_time_s(float(t_hit))} s"
                stats_lines.extend(
                    [
                        "- Optimization mode: Stress-limit optimize",
                        f"- Selected mass flow: {md_disp} {md_unit}",
                        f"- Final run time-to-target: {t_hit_txt}",
                        f"- Final total stress: {sigma_final_disp} {stress_unit} (limit {sigma_lim_disp} {stress_unit})",
                        f"- Constraint status: target_reached={reached_final}, stress_limit={meets_stress}",
                    ]
                )

        warnings = self._build_health_warnings(
            tw_si=tw_si,
            vm_map_elbow_mpa=vm_map_elbow_mpa,
            times=times,
            ignore_inlet_cells=ignore_inlet_cells,
            j_max=j_max,
        )
        if self.chk_convergence_diag.isChecked() and np.isfinite(radial_delta) and radial_delta > 0.15:
            warnings.append(
                "Stress changes by >15% under radial refinement; increase Nr_wall or treat result as low-confidence."
            )
        if self.chk_convergence_diag.isChecked() and np.isfinite(axial_delta) and axial_delta > 0.20:
            warnings.append(
                "Stress changes by >20% under coarse axial sensitivity check; consider higher Nx and inlet ramp."
            )
        if np.isfinite(sigma_excluding_inlet_disp) and hotspot_inlet_flag:
            warnings.append(
                f"Inlet-cell exclusion check: max stress away from inlet is {self._fmt_num(sigma_excluding_inlet_disp, 2)} {stress_unit}."
            )
        opt_summary = getattr(result, "opt_summary", None)
        if isinstance(opt_summary, dict):
            if not bool(opt_summary.get("target_reached_final", True)):
                warnings.append("Optimization final run did not reach outlet target within allowed simulation time.")
            if not bool(opt_summary.get("meets_stress_limit", True)):
                lim_disp = self._fmt_stress_si(float(opt_summary.get("stress_limit_mpa", np.nan)), 2)
                sig_disp = self._fmt_stress_si(float(opt_summary.get("sigma_final_mpa", np.nan)), 2)
                warnings.append(
                    f"Optimization final stress exceeds limit ({sig_disp} > {lim_disp} {stress_unit})."
                )
            if opt_summary.get("mode") == "heatup_time_opt" and not bool(opt_summary.get("meets_heatup_tolerance", True)):
                warnings.append("Heatup-time optimization did not satisfy requested time tolerance in final run.")

        if hasattr(self, "stats_box"):
            self.stats_box.setPlainText("\n".join(stats_lines))
        if hasattr(self, "warning_box"):
            self.warning_box.setPlainText("Health / Fatigue Warnings (screening)\n" + "\n".join(f"- {w}" for w in warnings))

        self._last_warnings = warnings
        self._last_stats = {
            "time_to_target_s": t_hit_s,
            "target_metric": target_metric,
            "target_metric_label": target_metric_label,
            "sigma_max_thermal_mpa": sigma_max_mpa,
            "sigma_max_thermal_elbow_mpa": sigma_max_elbow_mpa,
            "sigma_max_mpa": sigma_max_mpa,
            "sigma_max_elbow_mpa": sigma_max_elbow_mpa,
            "t_sigma_max": t_sigma_max,
            "x_sigma_max_m": x_sigma_max_m,
            "max_wall_dT_K": float(np.nanmax(deltaT_wall_si)),
            "Tg_outlet_final_K": float(result.Tg_outlet_final),
            "Tg_inlet_final_K": float(tg_si[-1, 0]),
            "Tw_outlet_final_K": float(tw_si[-1, -1]),
            "Tw_inlet_final_K": float(tw_si[-1, 0]),
            "Tw_outer_surface_outlet_K": float(tw_outer_surface_series_si[-1]),
            "Ti_outlet_final_K": float(ti_si[-1, -1]),
            "Ti_inlet_final_K": float(ti_si[-1, 0]),
            "Tin_eff_final_K": float(tin_eff_si[-1]) if tin_eff_si.size else float("nan"),
            "Tg_mean_final_K": float(np.mean(tg_si[-1])),
            "Tw_mean_final_K": float(np.mean(tw_si[-1])),
            "Ti_mean_final_K": float(np.mean(ti_si[-1])),
            "sim_time_s": float(times[-1]),
            "n_steps": int(result.n_steps),
            "wall_time_s": float(result.wall_time_s),
            "tin_ramp_model": ramp_model,
            "tin_ramp_s": ramp_s,
            "radial_diag": float(radial_delta) if np.isfinite(radial_delta) else None,
            "axial_diag": float(axial_delta) if np.isfinite(axial_delta) else None,
            "hotspot_inlet": bool(hotspot_inlet_flag),
        }

        self.results_canvas.draw_idle()
