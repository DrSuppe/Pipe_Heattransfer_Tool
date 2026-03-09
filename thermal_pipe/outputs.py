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
"""Plotting and output persistence helpers for simulation runs."""

import logging

import numpy as np

def _import_pyplot():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        logging.warning("Matplotlib unavailable; skipping plot generation. Reason: %s", exc)
        return None


def plot_heatmaps(x, times, Tw_hist, Tg_hist, Ti_hist, OUTDIR, params):
    plt = _import_pyplot()
    if plt is None:
        return

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Heatmaps — m_dot={params['m_dot']} kg/s, Tin={params['Tin']} K")
    im0 = axs[0].imshow(Tw_hist, aspect='auto', extent=[x[0], x[-1], times[-1], times[0]])
    axs[0].set_title('Wall Tw(x,t) [K]')
    axs[0].set_xlabel('x [m]')
    axs[0].set_ylabel('time [s]')
    plt.colorbar(im0, ax=axs[0], label='K')

    im1 = axs[1].imshow(Tg_hist, aspect='auto', extent=[x[0], x[-1], times[-1], times[0]])
    axs[1].set_title('Gas Tg(x,t) [K]')
    axs[1].set_xlabel('x [m]')
    axs[1].set_ylabel('time [s]')
    plt.colorbar(im1, ax=axs[1], label='K')

    im2 = axs[2].imshow(Ti_hist, aspect='auto', extent=[x[0], x[-1], times[-1], times[0]])
    axs[2].set_title('Insulation Ti(x,t) [K]')
    axs[2].set_xlabel('x [m]')
    axs[2].set_ylabel('time [s]')
    plt.colorbar(im2, ax=axs[2], label='K')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTDIR / "heatmaps.png", dpi=200)
    if params.get("show_plots", False):
        plt.show()
    plt.close(fig)
    logging.info("Saved heatmaps.png")


def plot_profiles(x, times, Tw_hist, Tg_hist, Ti_hist, OUTDIR, params):
    plt = _import_pyplot()
    if plt is None:
        return

    fig = plt.figure(figsize=(10, 4))
    # Select at most 30 evenly spaced time indices
    nmax = 30
    idx = np.linspace(0, max(0, times.size - 1), min(nmax, max(1, times.size)), dtype=int)

    for i in idx:
        plt.plot(x, Tw_hist[i], label=f"Tw {times[i]:.0f}s")
    for i in idx:
        plt.plot(x, Tg_hist[i], '--', label=f"Tg {times[i]:.0f}s")
    for i in idx:
        plt.plot(x, Ti_hist[i], ':', label=f"Ti {times[i]:.0f}s")

    plt.xlabel('x [m]')
    plt.ylabel('Temperature [K]')
    plt.title(f"Profiles over time — m_dot={params['m_dot']} kg/s, Tin={params['Tin']} K")
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
    if params.get("show_plots", False):
        plt.show()
    plt.close(fig)
    logging.info("Saved profiles.png")


def save_arrays_and_csv(OUTDIR, x, times, Tw_hist, Tg_hist, Ti_hist, Nx):
    # Persist arrays (npz)
    np.savez_compressed(OUTDIR / "fields.npz", x=x, times=times, Tw=Tw_hist, Tg=Tg_hist, Ti=Ti_hist)
    logging.info("Saved fields.npz")

    # Summary CSV: outlet/inlet/midpoint temperatures over time
    mid_idx = Nx // 2
    summary = np.column_stack([
        times,
        Tg_hist[:, -1],
        Tw_hist[:, 0], Tw_hist[:, mid_idx], Tw_hist[:, -1],
        Ti_hist[:, 0], Ti_hist[:, mid_idx], Ti_hist[:, -1],
    ]).astype(np.float64, copy=False)
    header = "time_s,Tg_outlet_K,Tw_inlet_K,Tw_mid_K,Tw_outlet_K,Ti_inlet_K,Ti_mid_K,Ti_outlet_K"
    np.savetxt(OUTDIR / "summary.csv", summary, delimiter=",", header=header, comments="")
    logging.info("Saved summary.csv")

    # Final prints for quick inspection (unchanged behavior)
    logging.info("Final outlet gas T [K]: %.2f", float(Tg_hist[-1, -1]))
    logging.info(
        "Final Tw inlet/mid/outlet [K]: %.2f, %.2f, %.2f",
        float(Tw_hist[-1, 0]), float(Tw_hist[-1, Nx // 2]), float(Tw_hist[-1, -1])
    )
    logging.info(
        "Final Ti inlet/mid/outlet [K]: %.2f, %.2f, %.2f",
        float(Ti_hist[-1, 0]), float(Ti_hist[-1, Nx // 2]), float(Ti_hist[-1, -1])
    )
    logging.info("Outputs saved to: %s", OUTDIR)
