from matplotlib.ticker import EngFormatter
import matplotlib.pyplot as plt
import os 
import numpy as np
from scipy.stats import gaussian_kde #type: ignore 


def save_training_curves(self, it):
    """
    Three-axis plot:
        - Left  (black): Compliance
        - Right (blue):  Stress metric (percentile/KS)
        - Right offset (green): Absolute Volume
    """

    if len(self.hist_iters) == 0:
        return

    iters = self.hist_iters
    comp = self.hist_compliance
    vol = self.hist_volume
    sigma = self.hist_sigma_metric

    fig, axC = plt.subplots(figsize=(9, 4.5))
    fig.subplots_adjust(right=0.83)

    lC, = axC.plot(iters, comp, 'k-', lw=1.8, label='Compliance')
    axC.set_xlabel('Iteration')
    axC.set_ylabel('Compliance', color='k')
    axC.tick_params(axis='y', labelcolor='k')
    axC.grid(True, alpha=0.25)

    axS = axC.twinx()
    lS, = axS.plot(iters, sigma, 'b--', lw=1.8, label='Stress Metric')
    axS.set_ylabel('Stress metric', color='b')
    axS.tick_params(axis='y', labelcolor='b')

    axV = axC.twinx()
    axV.spines['right'].set_position(('axes', 1.12))
    axV.spines['right'].set_visible(True)
    lV, = axV.plot(iters, vol, 'g-', lw=1.8, label='Volume (abs)')
    axV.set_ylabel('Volume (abs)', color='g')
    axV.tick_params(axis='y', colors='g')
    try:
        axV.yaxis.set_major_formatter(EngFormatter(places=1))
    except Exception:
        pass

    lines = [lC, lV, lS]
    labels = [ln.get_label() for ln in lines]
    axC.legend(lines, labels, loc='upper right', frameon=False)

    plt.tight_layout()
    out = os.path.join(self.save_path, f"progress-{it:06d}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)

def save_ginn_training_curves(self, it):
    """
    Save two plots of GINN geometry losses vs iteration:
        1) log-scale y
        2) linear-scale y
    """

    if len(self.ginn_hist_iters) == 0:
        return

    iters = self.ginn_hist_iters

    def _plot(ax):
        ax.plot(iters, self.ginn_hist_total, 'k-', lw=1.8, label='Total GINN loss')
        ax.plot(iters, self.ginn_hist_eik, '--', label='Eikonal')
        ax.plot(iters, self.ginn_hist_env, '--', label='Envelope')
        ax.plot(iters, self.ginn_hist_connect, '--', label='Connectivity')
        ax.plot(iters, self.ginn_hist_hole, '--', label='Holes')
        ax.plot(iters, self.ginn_hist_int, '--', label='Interface')
        ax.plot(iters, self.ginn_hist_norm, '--', label='Surface normals')
        ax.plot(iters, self.ginn_hist_thick, '--', label='Thickness')
        ax.plot(iters, self.ginn_hist_prohib, '--', label='Prohibited')
        ax.plot(iters, self.ginn_hist_smooth, '--', label='Smoothness')

        ax.set_xlabel('GINN pre-training iteration')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper right', frameon=False, ncol=2)

    # log-scale
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_yscale('log')
    _plot(ax)
    plt.tight_layout()
    out_log = os.path.join(self.save_path, f"ginn-progress-log-{it:06d}.png")
    fig.savefig(out_log, dpi=150)
    plt.close(fig)

    # linear-scale
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _plot(ax)
    plt.tight_layout()
    out_lin = os.path.join(self.save_path, f"ginn-progress-lin-{it:06d}.png")
    fig.savefig(out_lin, dpi=150)
    plt.close(fig)

def save_topology_density_curves(self, it):
    """
    Plot topology-density ALM losses vs topo step index (log-scale y).
    """
 
    if len(self.topo_hist_iters) == 0:
        return

    iters = self.topo_hist_iters

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_yscale('log')

    ax.plot(iters, self.topo_hist_total, 'k-', lw=1.8, label='Total density loss')
    ax.plot(iters, self.topo_hist_topo, '--', label='Topo objective (MSE)')
    ax.plot(iters, self.topo_hist_env_rho, '--', label='Density envelope')
    ax.plot(iters, self.topo_hist_conn_rho, '--', label='Density connectivity')

    ax.set_xlabel('Topology ALM step')
    ax.set_ylabel('Loss (log scale)')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', frameon=False, ncol=2)

    plt.tight_layout()
    out = os.path.join(self.save_path, f"topo-density-progress-{it:06d}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def save_stress_histogram(self, it, sigma_img, rho_img, sigma_metric, sigma_max, num_bins=80):
    """
    Creates: 
        1) Histogram of von Mises stresses in the solid region with lines for
            stress metric (current) and Max stress.
        2) A combined plot showing KS, Percentile, and Max stress metrics as
            vertical lines on the same histogram, with a smooth distribution curve.
    """

    # ----------------- Extract solid region stresses -----------------
    solid = (rho_img >= self.plot_threshold)
    if not np.any(solid):
        flat = rho_img.reshape(-1)
        t = float(np.quantile(flat, 0.5))
        solid = (rho_img >= t)

    sigma_vals = np.asarray(sigma_img, dtype=np.float64)[solid]
    sigma_vals = sigma_vals[np.isfinite(sigma_vals)]

    if sigma_vals.size == 0:
        return  # nothing to plot

    # Compute alternate metrics (so we can show them all)
    ks_metric, _, _ = self._measure_stress_volume_KS_2d()
    perc_metric, _, _ = self._measure_stress_volume_percentile_2d()
    max_metric = float(np.max(sigma_vals))

    # ----------------- 1) Primary Histogram for current stress metric -----------------
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(
        sigma_vals,
        bins=num_bins,
        alpha=0.75,
        edgecolor='k',
        linewidth=0.4
    )

    # Stress metric line (selected metric) 
    ax.axvline(
        sigma_metric,
        color='r',
        linestyle='--',
        linewidth=2.0,
        label=f"Current {self.stress_metric.upper()} metric = {sigma_metric:.3g}"
    )

    # Max stress line
    ax.axvline(
        sigma_max,
        color='orange',
        linestyle='-.',
        linewidth=2.0,
        label=f"Max stress = {sigma_max:.3g}"
    )

    ax.set_xlabel("von Mises stress")
    ax.set_ylabel("Count of solid points")
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    out = os.path.join(self.save_path, f"stress-hist-{it:06d}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7.5, 4))

    # Histogram 
    counts, bins, _ = ax2.hist(
        sigma_vals,
        bins=num_bins,
        alpha=0.6,
        edgecolor='k',
        linewidth=0.4,
        label="Stress distribution"
    )

    # Smooth KDE curve
    kde = gaussian_kde(sigma_vals)
    xs = np.linspace(np.min(sigma_vals), np.max(sigma_vals), 400)
    ys = kde(xs)
    ys = ys * (np.max(counts) / np.max(ys))  
    ax2.plot(xs, ys, color='orange', linewidth=2.5, label="Smoothed stress curve")

    # Vertical lines for all metrics
    ax2.axvline(
        ks_metric,
        color='purple',
        linestyle='--',
        linewidth=1.8,
        label=f"KS metric = {ks_metric:.3g}"
    )
    ax2.axvline(
        perc_metric,
        color='red',
        linestyle='--',
        linewidth=1.8,
        label=f"Percentile metric = {perc_metric:.3g}"
    )
    ax2.axvline(
        max_metric,
        color='orange',
        linestyle='-.',
        linewidth=1.8,
        label=f"Max stress = {max_metric:.3g}"
    )

    ax2.set_xlabel("von Mises stress")
    ax2.set_ylabel("Count of solid points")
    ax2.grid(True, alpha=0.25)

    fig2.subplots_adjust(right=0.78)
    ax2.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False
    )
    fig2.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])

    out2 = os.path.join(self.save_path, f"stress-hist-allmetrics-{it:06d}.png")
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)

