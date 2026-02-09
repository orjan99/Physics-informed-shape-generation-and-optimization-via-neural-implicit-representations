import matplotlib.pyplot as plt 
import torch 
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon


jet_list = [
            (0.0,   "#000080"),  # navy
            (0.15,  "#0000FF"),  # blue 
            (0.33,  "#00FFFF"),  # cyan
            (0.50,  "#00FF00"),  # green
            (0.67,  "#FFFF00"),  # yellow
            (0.85,  "#FF8000"),  # orange
            (1.0,   "#FF0000"),  # red
        ]
ansys_jet = LinearSegmentedColormap.from_list("ansys_jet", [c for _,c in jet_list], N=256) 

def plot_2d_PINN_results(points, values, domain, quantity=None):
    # Convert tensors to numpy arrays
    pts = points.detach().cpu().numpy()
    vals = values.detach().cpu().numpy()

    if quantity == 'x_disp': 
        if vals.ndim != 2 or vals.shape[1] != 2:
            raise ValueError("Displacement values must have two columns for x and y displacements.")
        plot_color = vals[:, 0]
    elif quantity == 'y_disp':
        if vals.ndim != 2 or vals.shape[1] != 2:
            raise ValueError("Displacement values must have two columns for x and y displacements.")
        plot_color = vals[:, 1]
    elif quantity == 'Stress':
        if vals.ndim != 1:
            raise ValueError("Stress values must be a 1D array of shape (N,).")
        plot_color = vals
    elif quantity == 'displacement_magnitude':
        if vals.ndim != 2 or vals.shape[1] != 2:
            raise ValueError("Displacement values must have two columns for x and y displacements.")
        plot_color = np.linalg.norm(vals, axis=1)
    else:
        raise ValueError("Invalid quantity specified. Choose from 'x_disp', 'y_disp', 'Stress', or 'displacement_magnitude'.")

    # Set up plot
    fig, ax = plt.subplots(figsize=(6, 5))

    # compute color limits and ticks
    vmin, vmax = plot_color.min(), plot_color.max()
    ticks = np.linspace(vmin, vmax, 10)
    ticklabels = [f"{t:.6f}" for t in ticks]
    ticklabels[0]  = ticklabels[0] + " Min"
    ticklabels[-1] = ticklabels[-1] + " Max"

    scatter = ax.scatter(
        pts[:, 0], pts[:, 1],
        c=plot_color,
        cmap=ansys_jet,
        vmin=vmin,
        vmax=vmax,
        s=5
    )
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal')
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)

    # label colorbar
    if quantity == 'x_disp':
        cbar.set_label('X Displacement')
    elif quantity == 'y_disp':
        cbar.set_label('Y Displacement')
    elif quantity == 'Stress':
        cbar.set_label('Von Mises Stress')
    else:
        cbar.set_label('Displacement Magnitude')

    # Handle domain format
    if len(domain) == 4:
        x_min, x_max, y_min, y_max = domain
    else:
        (x_min, x_max), (y_min, y_max) = domain

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # set title
    if quantity == 'x_disp':
        ax.set_title('X Displacement Field')
    elif quantity == 'y_disp':
        ax.set_title('Y Displacement Field')
    elif quantity == 'Stress':
        ax.set_title('Von Mises Stress Field')
    else:
        ax.set_title('Displacement Magnitude Field')

    ax.set_aspect('equal')
    return fig