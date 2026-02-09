from Functions.Point_Sampling.point_sampler import Point_Sampler
import torch
import matplotlib.pyplot as plt 
import numpy as np 
import matplotlib.tri as mtri

def plot_GINN_geometry(test_case, num_points, GINN_model):
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else
        "mps"    if torch.backends.mps.is_available() else
        "cpu"
    )
    domain = test_case.domain
    point_sampler = Point_Sampler(domain=domain, num_points_domain=num_points)
    points = next(point_sampler).to(device)

    with torch.no_grad():
        GINN_model.eval() 
        if not test_case.Symmetry:
            SDF = GINN_model(points).view(-1, 1)
            density = 1 / (1 + torch.exp(100 * SDF))
            points = points.cpu().numpy()
            density = density.cpu().numpy().ravel()
        else:
            SDF, points_full, SDF_full = GINN_model(points)
            density = 1 / (1 + torch.exp(100 * SDF_full))
            points = points_full.cpu().numpy()
            density = density.cpu().numpy().ravel()

    interface_points = test_case.interfaces.sample_points_from_all_interfaces(
        num_points=1000,
        output_type='torch_tensor',
        device=device
    ).cpu().numpy()

    # Build triangulation for contouring
    tri_obj = mtri.Triangulation(points[:, 0], points[:, 1])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(
        points[:, 0], points[:, 1],
        c=density, cmap='coolwarm', s=5, label='Density Values'
    )
    fig.colorbar(sc, label='Density', orientation='horizontal')

    # 0.5 isoline
    cs = ax.tricontour(
        tri_obj, density,
        levels=[0.5],
        colors='k',
        linewidths=2,
        linestyles='-'
    )
    ax.clabel(cs, fmt='ρ=%.1f', inline=True, fontsize=10)

    # interface points
    ax.scatter(
        interface_points[:, 0], interface_points[:, 1],
        c='green', s=5, label='Interface Points'
    )

    ax.set_title('Density Distribution from SDF')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.legend(loc='upper right')

    plt.show()
    return fig
