import os 
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata #type: ignore 
from Functions.Training.Properties import Properties

# FEM-only 1×3 plots 
def save_FEM_results_smooth(fem_ref, filename, BRIDGE, grid_resolution=300):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    coords_scaled = fem_ref["coords_scaled"]
    von_mises_FEM = fem_ref["sigma_vm"]
    x_disp_FEM = fem_ref["x_disp"]
    y_disp_FEM = fem_ref["y_disp"]

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    x = np.linspace(BRIDGE.domain[0], BRIDGE.domain[1], grid_resolution)
    y = np.linspace(BRIDGE.domain[2], BRIDGE.domain[3], grid_resolution)
    X, Y = np.meshgrid(x, y, indexing="ij")

    grid_points = np.column_stack([X.ravel(), Y.ravel()])

    with torch.no_grad():
        coords_torch = torch.tensor(grid_points, device=device, dtype=torch.float32)
        density = torch.sigmoid(-1000 * BRIDGE.interfaces.calculate_SDF(coords_torch))

    mask = density < 0.5
    mask = mask.reshape(grid_resolution, grid_resolution).detach().cpu().numpy()

    points_FEM = coords_scaled

    y_grid = griddata(points_FEM, y_disp_FEM, (X, Y), method="linear")
    x_grid = griddata(points_FEM, x_disp_FEM, (X, Y), method="linear")
    s_grid = griddata(points_FEM, von_mises_FEM, (X, Y), method="linear")

    y_grid = np.ma.masked_where(mask, y_grid)
    x_grid = np.ma.masked_where(mask, x_grid)
    s_grid = np.ma.masked_where(mask, s_grid)

    y_grid = np.ma.masked_invalid(y_grid)
    x_grid = np.ma.masked_invalid(x_grid)
    s_grid = np.ma.masked_invalid(s_grid)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # y_disp FEM
    y_nodes = y_disp_FEM
    ymin = y_nodes.min()
    ymax = y_nodes.max()
    if ymin != ymax:
        levels_y = np.linspace(ymin, ymax + 1e-9, 1000)
        im0 = axes[0].contourf(
            X, Y, y_grid, levels=levels_y, cmap="jet", vmin=ymin, vmax=ymax
        )
        axes[0].set_title("y_disp (FEM, interpolated)")
        axes[0].axis("image")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        cbar_y = fig.colorbar(im0, ax=axes[0], orientation="horizontal")
        cbar_y.ax.tick_params(labelrotation=-45)
        cbar_y.ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        if ymax not in cbar_y.get_ticks():
            ticks = cbar_y.get_ticks()
            ticks = np.append(ticks, ymax)
            cbar_y.set_ticks(np.sort(ticks))
    else:
        axes[0].set_title("y_disp (constant)")
        axes[0].axis("image")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].text(
            0.5,
            0.5,
            f"Constant: {ymin:.4f}",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )

    # x_disp FEM
    x_nodes = x_disp_FEM
    xmin = x_nodes.min()
    xmax = x_nodes.max()
    if xmin != xmax:
        levels_x = np.linspace(xmin, xmax + 1e-9, 1000)
        im1 = axes[1].contourf(
            X, Y, x_grid, levels=levels_x, cmap="jet", vmin=xmin, vmax=xmax
        )
        axes[1].set_title("x_disp (FEM, interpolated)")
        axes[1].axis("image")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        cbar_x = fig.colorbar(im1, ax=axes[1], orientation="horizontal")
        cbar_x.ax.tick_params(labelrotation=-45)
        cbar_x.ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        if xmax not in cbar_x.get_ticks():
            ticks = cbar_x.get_ticks()
            ticks = np.append(ticks, xmax)
            cbar_x.set_ticks(np.sort(ticks))
    else:
        axes[1].set_title("x_disp (constant)")
        axes[1].axis("image")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].text(
            0.5,
            0.5,
            f"Constant: {xmin:.4f}",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )

    # von Mises FEM
    s_nodes = von_mises_FEM
    smin = s_nodes.min()
    smax = s_nodes.max()
    if smin != smax:
        levels_s = np.linspace(smin, smax + 1e-9, 1000)
        im2 = axes[2].contourf(
            X, Y, s_grid, levels=levels_s, cmap="jet", vmin=smin, vmax=smax
        )
        axes[2].set_title("Von Mises Stress (FEM, interpolated)")
        axes[2].axis("image")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        cbar_s = fig.colorbar(im2, ax=axes[2], orientation="horizontal", extend="both")
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
        axes[2].set_title("Von Mises Stress (constant)")
        axes[2].axis("image")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].text(
            0.5,
            0.5,
            f"Constant: {smin:.4f}",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )

    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)



def predict_uv_sigma_image_2d_binary(
    u_model, v_model, density_model, domain, BRIDGE ,rho_threshold_plot: float = 0.5
):
    """
    Binary-geometry evaluation:
      -density from model (no enforce); mask = (rho < threshold)
      -set rho[~mask] = 1.0 (solid)
      -stresses with SOLID Lamé (no SIMP)
       - returns stress and displacement plots
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    grid_resolution = 400

    x = np.linspace(domain[0], domain[1], grid_resolution)
    y = np.linspace(domain[2], domain[3], grid_resolution)
    X, Y = np.meshgrid(x, y, indexing="ij")

    points = np.column_stack([X.ravel(), Y.ravel()])
    xt = torch.tensor(points, device=device, dtype=torch.float32).requires_grad_(True)

    prev_mode = density_model.training
    density_model.eval()
    with torch.no_grad():
        rho = density_model(xt).view(-1)
    if prev_mode:
        density_model.train()

    mask = rho < rho_threshold_plot
    rho = rho.clone()
    rho[~mask] = 1.0  # solid

    u = u_model(xt)
    v = v_model(xt)

    gu = torch.autograd.grad(
        u, xt, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    gv = torch.autograd.grad(
        v, xt, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True
    )[0]

    eps11 = gu[:, 0]
    eps22 = gv[:, 1]
    eps12 = 0.5 * (gu[:, 1] + gv[:, 0])
    tr = eps11 + eps22

    lam = Properties(BRIDGE).lame_lambda
    mu = Properties(BRIDGE).lame_mu 

    s11 = 2.0 * mu * eps11 + lam * tr
    s22 = 2.0 * mu * eps22 + lam * tr
    s12 = 2.0 * mu * eps12

    svm2 = s11**2 - s11 * s22 + s22**2 + 3.0 * s12**2
    svm = torch.sqrt(torch.clamp(svm2, min=1e-32)) * BRIDGE.domain_scaling_factor

    ny = nx = grid_resolution
    u_img = u.view(ny, nx).detach().cpu().numpy()
    v_img = v.view(ny, nx).detach().cpu().numpy()
    sigma_img = svm.view(ny, nx).detach().cpu().numpy()
    rho_img = rho.view(ny, nx).detach().cpu().numpy()

    return u_img, v_img, sigma_img, rho_img


def save_uv_sigma_to_file(
    u_img, v_img, sigma_img, rho_img, filename, domain, BRIDGE, rho_threshold: float = 0.5
):
    """
    1×3 panels (y_disp, x_disp, von Mises) - horizontal colorbars,
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    ny, nx = u_img.shape
    x = np.linspace(domain[0], domain[1], nx)
    y = np.linspace(domain[2], domain[3], ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    mask = rho_img < rho_threshold

    L0 = Properties(BRIDGE).L0 if hasattr(Properties(BRIDGE), "L0") else 1.0
    v_plot = np.ma.masked_where(mask, v_img) * L0
    u_plot = np.ma.masked_where(mask, u_img) * L0
    s_plot = np.ma.masked_where(mask, sigma_img)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # y_disp
    ymin, ymax = float(v_plot.min()), float(v_plot.max())
    if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
        levels_y = np.linspace(ymin, ymax + 1e-9, 1000)
        c0 = axes[0].contourf(X, Y, v_plot, cmap="jet", levels=levels_y)
        axes[0].set_title("y_disp")
        axes[0].axis("image")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        cb0 = fig.colorbar(c0, ax=axes[0], orientation="horizontal")
        cb0.ax.tick_params(labelrotation=-45)
        cb0.ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        if ymax not in cb0.get_ticks():
            ticks = np.append(cb0.get_ticks(), ymax)
            cb0.set_ticks(np.sort(ticks))
    else:
        axes[0].set_title("y_disp (constant)")
        axes[0].axis("image")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].text(
            0.5,
            0.5,
            f"Constant: {ymin:.4f}",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )

    xmin, xmax = float(u_plot.min()), float(u_plot.max())
    if np.isfinite(xmin) and np.isfinite(xmax) and xmin != xmax:
        levels_x = np.linspace(xmin, xmax + 1e-9, 1000)
        c1 = axes[1].contourf(X, Y, u_plot, cmap="jet", levels=levels_x)
        axes[1].set_title("x_disp")
        axes[1].axis("image")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        cb1 = fig.colorbar(c1, ax=axes[1], orientation="horizontal")
        cb1.ax.tick_params(labelrotation=-45)
        cb1.ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        if xmax not in cb1.get_ticks():
            ticks = np.append(cb1.get_ticks(), xmax)
            cb1.set_ticks(np.sort(ticks))
    else:
        axes[1].set_title("x_disp (constant)")
        axes[1].axis("image")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].text(
            0.5,
            0.5,
            f"Constant: {xmin:.4f}",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )

    # von Mises
    smin, smax = float(s_plot.min()), float(s_plot.max())
    if np.isfinite(smin) and np.isfinite(smax) and smin != smax:
        levels_s = np.linspace(smin, smax + 1e-9, 1000)
        c2 = axes[2].contourf(X, Y, s_plot, cmap="jet", levels=levels_s)
        axes[2].set_title("Von Mises Stress")
        axes[2].axis("image")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        cb2 = fig.colorbar(c2, ax=axes[2], orientation="horizontal", extend="both")
        cb2.ax.tick_params(labelrotation=-45)
        ticks = cb2.get_ticks()
        if smin not in ticks:
            ticks = np.append(ticks, smin)
        if smax not in ticks:
            ticks = np.append(ticks, smax)
        cb2.set_ticks(np.sort(ticks))
    else:
        axes[2].set_title("Von Mises Stress (constant)")
        axes[2].axis("image")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].text(
            0.5,
            0.5,
            f"Constant: {smin:.4f}",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )

    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)





# 3×3 grid: NN vs FEM vs |error|  
def save_uv_sigma_comparison_grid_2d(
    u_model,
    v_model,
    material_properties,
    fem_ref,
    filename,
    BRIDGE,
    grid_resolution=300,
    rho_threshold=0.5,
):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if fem_ref is None:
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Grid
    x = np.linspace(BRIDGE.domain[0], BRIDGE.domain[1], grid_resolution)
    y = np.linspace(BRIDGE.domain[2], BRIDGE.domain[3], grid_resolution)
    X, Y = np.meshgrid(x, y, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel()])

    coords_grid = torch.tensor(
        pts, dtype=torch.float32, device=device, requires_grad=True
    )

    # Mask from SDF
    with torch.no_grad():
        SDF = BRIDGE.interfaces.calculate_SDF(coords_grid)
        density = torch.sigmoid(-1000 * SDF)
    mask = (
        density < rho_threshold
    ).reshape(grid_resolution, grid_resolution).detach().cpu().numpy()

    points_FEM = fem_ref["coords_scaled"]
    x_disp_FEM = fem_ref["x_disp"]
    y_disp_FEM = fem_ref["y_disp"]
    s_FEM = fem_ref["sigma_vm"]

    y_grid = griddata(points_FEM, y_disp_FEM, (X, Y), method="linear")
    x_grid = griddata(points_FEM, x_disp_FEM, (X, Y), method="linear")
    s_grid = griddata(points_FEM, s_FEM, (X, Y), method="linear")

    y_grid = np.ma.masked_where(mask, y_grid)
    x_grid = np.ma.masked_where(mask, x_grid)
    s_grid = np.ma.masked_where(mask, s_grid)

    y_grid = np.ma.masked_invalid(y_grid)
    x_grid = np.ma.masked_invalid(x_grid)
    s_grid = np.ma.masked_invalid(s_grid)

    # NN predictions on grid
    u = u_model(coords_grid)
    v = v_model(coords_grid)

    grad_u = torch.autograd.grad(
        u,
        coords_grid,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_v = torch.autograd.grad(
        v,
        coords_grid,
        grad_outputs=torch.ones_like(v),
        create_graph=True,
        retain_graph=False,
    )[0]

    eps11 = grad_u[:, 0]
    eps22 = grad_v[:, 1]
    eps12 = 0.5 * (grad_u[:, 1] + grad_v[:, 0])
    tr = eps11 + eps22

    lam = material_properties.lame_lambda.to(device)
    mu = material_properties.lame_mu.to(device)

    s11 = 2.0 * mu * eps11 + lam * tr
    s22 = 2.0 * mu * eps22 + lam * tr
    s12 = 2.0 * mu * eps12

    svm2 = s11**2 - s11 * s22 + s22**2 + 3.0 * s12**2
    svm = torch.sqrt(torch.clamp(svm2, min=1e-32)) * BRIDGE.domain_scaling_factor

    u_nn = u.detach().cpu().numpy().reshape(grid_resolution, grid_resolution)
    v_nn = v.detach().cpu().numpy().reshape(grid_resolution, grid_resolution)
    s_nn = svm.detach().cpu().numpy().reshape(grid_resolution, grid_resolution)

    u_nn = np.ma.masked_where(mask, u_nn)
    v_nn = np.ma.masked_where(mask, v_nn)
    s_nn = np.ma.masked_where(mask, s_nn)

    err_u = np.ma.masked_where(mask, np.abs(u_nn - x_grid))
    err_v = np.ma.masked_where(mask, np.abs(v_nn - y_grid))
    err_s = np.ma.masked_where(mask, np.abs(s_nn - s_grid))

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    def _contour_with_minmax(ax, Z, title):
        zmin, zmax = float(Z.min()), float(Z.max())
        if np.isfinite(zmin) and np.isfinite(zmax) and zmin != zmax:
            levels = np.linspace(zmin, zmax + 1e-9, 300)
            c = ax.contourf(
                X, Y, Z, cmap="jet", levels=levels, vmin=zmin, vmax=zmax
            )
            ax.set_title(title)
            ax.axis("image")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            cb = fig.colorbar(c, ax=ax, orientation="horizontal")
            ticks = np.linspace(zmin, zmax, 6)
            tick_labels = [f"{t:.4f}" for t in ticks]
            if len(tick_labels) >= 2:
                tick_labels[0] += " Min"
                tick_labels[-1] += " Max"
            cb.set_ticks(ticks)
            cb.set_ticklabels(tick_labels)
            cb.ax.tick_params(labelrotation=-45)
        else:
            ax.set_title(f"{title} (constant)")
            ax.axis("image")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.text(
                0.5,
                0.5,
                f"Constant: {zmin:.4f}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    _contour_with_minmax(axes[0, 0], v_nn, "y_disp NN")
    _contour_with_minmax(axes[0, 1], y_grid, "y_disp FEM")
    _contour_with_minmax(axes[0, 2], err_v, "|y_disp NN - FEM|")

    _contour_with_minmax(axes[1, 0], u_nn, "x_disp NN")
    _contour_with_minmax(axes[1, 1], x_grid, "x_disp FEM")
    _contour_with_minmax(axes[1, 2], err_u, "|x_disp NN - FEM|")

    _contour_with_minmax(axes[2, 0], s_nn, "Von Mises NN")
    _contour_with_minmax(axes[2, 1], s_grid, "Von Mises FEM")
    _contour_with_minmax(axes[2, 2], err_s, "|Von Mises NN - FEM|")

    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)