
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter 


# ---------------- plots ----------------
def save_training_curves(self, it): 
    """
    Three-axis plot:
        - Left  (black): Compliance
        - Right (blue):  Stress metric
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
    save two plots of GINN geometry losses vs iteration:
        1) log-scale y  
        2) linear-scale 
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