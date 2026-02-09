from scipy.interpolate import griddata #type: ignore 
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_FEM_results_smooth(
                            coords_scaled,
                            von_mises_FEM,
                            x_disp_FEM,
                            y_disp_FEM,
                            BRIDGE): 
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") 

    grid_resolution = 300
    x = np.linspace(BRIDGE.domain[0], BRIDGE.domain[1], grid_resolution)
    y = np.linspace(BRIDGE.domain[2], BRIDGE.domain[3], grid_resolution)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Flatten grid for SDF and interpolation
    grid_points = np.column_stack([X.ravel(), Y.ravel()])

    # Density / SDF mask 
    with torch.no_grad():
        coords_torch = torch.tensor(grid_points,
                                    device=device,
                                    dtype=torch.float32)
        density = torch.sigmoid(-1000 * BRIDGE.interfaces.calculate_SDF(coords_torch))

    mask = (density < 0.5)                               # shape (N,)
    mask = mask.reshape(grid_resolution, grid_resolution).detach().cpu().numpy()
    points_FEM = coords_scaled

    y_grid = griddata(points_FEM, y_disp_FEM, (X, Y), method='linear')
    x_grid = griddata(points_FEM, x_disp_FEM, (X, Y), method='linear')
    s_grid = griddata(points_FEM, von_mises_FEM, (X, Y), method='linear')

    y_grid = np.ma.masked_where(mask, y_grid)
    x_grid = np.ma.masked_where(mask, x_grid)
    s_grid = np.ma.masked_where(mask, s_grid)

    y_grid = np.ma.masked_invalid(y_grid)
    x_grid = np.ma.masked_invalid(x_grid)
    s_grid = np.ma.masked_invalid(s_grid)


    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    y_nodes_scaled = y_disp_FEM
    ymin = y_nodes_scaled.min()
    ymax = y_nodes_scaled.max()

    if ymin != ymax:
        levels_y = np.linspace(ymin, ymax + 1e-9, 1000)
        im0 = axes[0].contourf(
            X, Y, y_grid,
            levels=levels_y, cmap='jet', vmin=ymin, vmax=ymax
        )
        axes[0].set_title('y_disp (FEM, interpolated)')
        axes[0].axis('image')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        cbar_y = fig.colorbar(im0, ax=axes[0], orientation='horizontal')
        cbar_y.ax.tick_params(labelrotation=-45)
        cbar_y.ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        if ymax not in cbar_y.get_ticks():
            ticks = cbar_y.get_ticks()
            ticks = np.append(ticks, ymax)
            cbar_y.set_ticks(np.sort(ticks))
    else:
        axes[0].set_title('y_disp (constant)')
        axes[0].axis('image')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].text(0.5, 0.5, f'Constant: {ymin:.4f}',
                    ha='center', va='center',
                    transform=axes[0].transAxes)

    # x_disp
    x_nodes_scaled = x_disp_FEM
    xmin = x_nodes_scaled.min()
    xmax = x_nodes_scaled.max()

    if xmin != xmax:
        levels_x = np.linspace(xmin, xmax + 1e-9, 1000)
        im1 = axes[1].contourf(X, Y, x_grid, levels=levels_x, cmap='jet', vmin=xmin, vmax=xmax)
        axes[1].set_title('x_disp (FEM, interpolated)')
        axes[1].axis('image')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        cbar_x = fig.colorbar(im1, ax=axes[1], orientation='horizontal')
        cbar_x.ax.tick_params(labelrotation=-45)
        cbar_x.ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        if xmax not in cbar_x.get_ticks():
            ticks = cbar_x.get_ticks()
            ticks = np.append(ticks, xmax)
            cbar_x.set_ticks(np.sort(ticks))
    else:
        axes[1].set_title('x_disp (constant)')
        axes[1].axis('image')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].text(0.5, 0.5, f'Constant: {xmin:.4f}',
                    ha='center', va='center',
                    transform=axes[1].transAxes)

    # Von Mises stress 
    s_nodes_scaled = von_mises_FEM
    smin = s_nodes_scaled.min()
    smax = s_nodes_scaled.max()

    if smin != smax:
        levels_s = np.linspace(smin, smax + 1e-9, 1000)
        im2 = axes[2].contourf(
            X, Y, s_grid,
            levels=levels_s, cmap='jet', vmin=smin, vmax=smax
        )
        axes[2].set_title('Von Mises Stress (FEM, interpolated)')
        axes[2].axis('image')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        cbar_s = fig.colorbar(im2, ax=axes[2],
                              orientation='horizontal', extend='both')
        cbar_s.ax.tick_params(labelrotation=-45)
        if smin not in cbar_s.get_ticks():
            ticks = cbar_s.get_ticks()
            ticks = np.append(ticks, smin)
            cbar_s.set_ticks(np.sort(ticks))
        if smax not in cbar_s.get_ticks():
            ticks = cbar_s.get_ticks()
            ticks = np.append(ticks, smax)
            cbar_s.set_ticks(np.sort(ticks))
    else:
        axes[2].set_title('Von Mises Stress (constant)')
        axes[2].axis('image')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        axes[2].text(0.5, 0.5, f'Constant: {smin:.4f}',
                    ha='center', va='center',
                    transform=axes[2].transAxes)

    plt.tight_layout()
    plt.show()