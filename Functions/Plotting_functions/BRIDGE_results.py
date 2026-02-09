import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.tri as mtri
from Functions.Point_Sampling.point_sampler import Point_Sampler
import torch
from Functions.Point_Sampling.point_sampler import *
from Functions.Training.Properties import *
import matplotlib

def plot_GINN_geometry(test_case, num_points, GINN_model,iso):
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

    # Interface points 
    interface_points = test_case.interfaces.sample_points_from_all_interfaces(
        num_points=1000,
        output_type='torch_tensor',
        device=device
    ).cpu().numpy()

    # Build triangulation for contouring
    tri_obj = mtri.Triangulation(points[:, 0], points[:, 1])
    fig, ax = plt.subplots(figsize=(8, 8))

    sc = ax.scatter(
        points[:, 0], points[:, 1],
        c=density, cmap='Reds', s=1, label='Density Values'
    )
    fig.colorbar(sc, label='Density', orientation='horizontal')

    # Black isoline at density = 0.5 
    iso_level = iso
    cs = ax.tricontour(
        tri_obj, density,
        levels=[iso_level],
        colors='k',
        linewidths=2,
        linestyles='-'
    )

    ax.clabel(cs, fmt='ρ=%.1f', inline=False, fontsize=0)
    ax.scatter(
        interface_points[:, 0], interface_points[:, 1],
        c='green', s=5, label='Interface Points'
    )

    ax.set_title('Optimized Geometry')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)  
    ax.legend(loc='upper right')

    return fig

def predict_uv_sigma_image_2d_binary(u_model,
                                     v_model,
                                     density_model,
                                     domain,
                                     test_case,
                                     rho_threshold_plot: float = 0.3,
                                 ):
    """
    Binary-geometry evaluation - plot displacement, stresses
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") 
    grid_resolution = 400

    # 1) Grid
    x = np.linspace(domain[0], domain[1], grid_resolution)
    y = np.linspace(domain[2], domain[3], grid_resolution)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # 2) Points tensor
    points = np.column_stack([X.ravel(), Y.ravel()])
    xt = torch.tensor(points, device=device, dtype=torch.float32).requires_grad_(True)

    # 3) Density & mask
    prev_mode = density_model.training
    density_model.eval()
    with torch.no_grad():
        rho = density_model(xt).view(-1)
    if prev_mode:
        density_model.train()

    mask = (rho < rho_threshold_plot)
    rho = rho.clone()
    rho[~mask] = 1.0  # solid = 1.0

    # 4) Displacements and gradients
    u = u_model(xt)
    v = v_model(xt)

    gu = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u),
                             create_graph=True, retain_graph=True)[0]
    gv = torch.autograd.grad(v, xt, grad_outputs=torch.ones_like(v),
                             create_graph=True, retain_graph=True)[0]

    eps11 = gu[:, 0]
    eps22 = gv[:, 1]
    eps12 = 0.5 * (gu[:, 1] + gv[:, 0])
    tr = eps11 + eps22

    # 5) Stresses with solid Lamé (no SIMP)
    lam = Properties(test_case).lame_lambda
    mu  = Properties(test_case).lame_mu

    s11 = 2.0 * mu * eps11 + lam * tr
    s22 = 2.0 * mu * eps22 + lam * tr
    s12 = 2.0 * mu * eps12

    # Von Mises (2D)
    svm2 = s11**2 - s11 * s22 + s22**2 + 3.0 * s12**2
    svm  = torch.sqrt(torch.clamp(svm2, min=1e-32)) * test_case.domain_scaling_factor

    # 6) Reshape to images [ny, nx] 
    ny = nx = grid_resolution
    u_img     = u.view(ny, nx).detach().cpu().numpy()
    v_img     = v.view(ny, nx).detach().cpu().numpy()
    sigma_img = svm.view(ny, nx).detach().cpu().numpy()
    rho_img   = rho.view(ny, nx).detach().cpu().numpy()

    return u_img, v_img, sigma_img, rho_img



def save_uv_sigma_to_file(u_img,
                          v_img,
                          sigma_img,
                          rho_img,
                          filename,
                          domain,
                          test_case,
                          rho_threshold: float = 0.3):
    """
    1×3 panels (y_disp, x_disp, von Mises) 
    """

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    ny, nx = u_img.shape
    x = np.linspace(domain[0], domain[1], nx)
    y = np.linspace(domain[2], domain[3], ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Binary mask (void = True)
    mask = (rho_img < rho_threshold)

    # Apply mask + scale
    L0 = Properties(test_case).L0 if hasattr(Properties(test_case), "L0") else 1.0
    v_plot = np.ma.masked_where(mask, v_img) * L0
    u_plot = np.ma.masked_where(mask, u_img) * L0
    s_plot = np.ma.masked_where(mask, sigma_img)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # ---------- y_disp ----------
    ymin, ymax = float(v_plot.min()), float(v_plot.max())
    if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
        levels_y = np.linspace(ymin, ymax + 1e-9, 1000)
        c0 = axes[0].contourf(X, Y, v_plot, cmap='jet', levels=levels_y)
        axes[0].set_title('y_disp')
        axes[0].axis('image')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        cb0 = fig.colorbar(c0, ax=axes[0], orientation='horizontal')
        cb0.ax.tick_params(labelrotation=-45)
        cb0.ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        if ymax not in cb0.get_ticks():
            ticks = np.append(cb0.get_ticks(), ymax)
            cb0.set_ticks(np.sort(ticks))
    else:
        axes[0].set_title('y_disp (constant)')
        axes[0].axis('image')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].text(0.5, 0.5, f'Constant: {ymin:.4f}',
                     ha='center', va='center', transform=axes[0].transAxes)

    # ---------- x_disp ----------
    xmin, xmax = float(u_plot.min()), float(u_plot.max())
    if np.isfinite(xmin) and np.isfinite(xmax) and xmin != xmax:
        levels_x = np.linspace(xmin, xmax + 1e-9, 1000)
        c1 = axes[1].contourf(X, Y, u_plot, cmap='jet', levels=levels_x)
        axes[1].set_title('x_disp')
        axes[1].axis('image')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        cb1 = fig.colorbar(c1, ax=axes[1], orientation='horizontal')
        cb1.ax.tick_params(labelrotation=-45)
        cb1.ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        if xmax not in cb1.get_ticks():
            ticks = np.append(cb1.get_ticks(), xmax)
            cb1.set_ticks(np.sort(ticks))
    else:
        axes[1].set_title('x_disp (constant)')
        axes[1].axis('image')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].text(0.5, 0.5, f'Constant: {xmin:.4f}',
                     ha='center', va='center', transform=axes[1].transAxes)

    # ---------- Von Mises ----------
    smin, smax = float(s_plot.min()), float(s_plot.max())
    if np.isfinite(smin) and np.isfinite(smax) and smin != smax:
        levels_s = np.linspace(smin, smax + 1e-9, 1000)
        c2 = axes[2].contourf(X, Y, s_plot, cmap='jet', levels=levels_s)
        axes[2].set_title('Von Mises Stress')
        axes[2].axis('image')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        cb2 = fig.colorbar(c2, ax=axes[2], orientation='horizontal', extend='both')
        cb2.ax.tick_params(labelrotation=-45)
        ticks = cb2.get_ticks()
        if smin not in ticks:
            ticks = np.append(ticks, smin)
        if smax not in ticks:
            ticks = np.append(ticks, smax)
        cb2.set_ticks(np.sort(ticks))
    else:
        axes[2].set_title('Von Mises Stress (constant)')
        axes[2].axis('image')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        axes[2].text(0.5, 0.5, f'Constant: {smin:.4f}',
                     ha='center', va='center', transform=axes[2].transAxes)

    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)



def save_density_scatter_iso_to_file(density_model,
                                     domain,
                                     filename,
                                     grid_resolution: int = 600,
                                     iso: float = 0.5):
  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    x = np.linspace(domain[0], domain[1], grid_resolution)
    y = np.linspace(domain[2], domain[3], grid_resolution)
    X, Y = np.meshgrid(x, y, indexing='ij')
    points = np.column_stack([X.ravel(), Y.ravel()])
    coords = torch.tensor(points, device=device, dtype=torch.float32, requires_grad=False)

    with torch.no_grad():
        model_density = density_model(coords).squeeze()
        binary_density = (model_density > iso).to(torch.float32)

    coords_np = coords.detach().cpu().numpy()
    density_np = binary_density.detach().cpu().numpy().ravel()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.figure(figsize=(8, 8))
    sc = plt.scatter(coords_np[:, 0], coords_np[:, 1],
                     c=density_np, cmap='Reds', s=1, label='Density Values')
    plt.colorbar(sc, label='Density', orientation='horizontal')

    tri_obj = mtri.Triangulation(coords_np[:, 0], coords_np[:, 1])
    cs = plt.tricontour(tri_obj, density_np,
                        levels=[iso], colors='k', linewidths=2, linestyles='-')
    plt.clabel(cs, fmt='ρ=%.1f', inline=False, fontsize=0)

    plt.title('Optimized Geometry (binary + isoline)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()



@torch.no_grad()
def predict_densities_image_2d(density_model, domain, n_samples_xy,enforce_density):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    nx, ny = n_samples_xy
    xs = get_grid_centers(domain, [nx, ny]).astype(np.float32)
    xt = torch.tensor(xs, dtype=torch.float32, device=device)
    rho = density_model(xt)

    rho, _ = enforce_density.apply(xt, rho, None, n_samples_xy, domain)
    rho = rho.view(ny, nx).detach().cpu().numpy()
    return rho, xs

def save_densities_to_file(density_image: np.ndarray, filename: str, domain=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if domain is None:
        extent = [0, 1.5, 0, 0.5]
    else:
        extent = [domain[0], domain[1], domain[2], domain[3]]
    fig = plt.figure(figsize=(12, 4))
    plt.imshow(np.flipud(density_image), cmap='Reds', vmin=0, vmax=1, extent=extent, aspect='auto')
    plt.colorbar()
    plt.tight_layout()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.gca().set_aspect('equal', adjustable='box')
    fig.savefig(filename, dpi=150)
    plt.close(fig)



def _masked_jet(ax, field_img, mask_bool, title, extent):
        cmap = matplotlib.cm.get_cmap('jet').copy()
        cmap.set_bad(color='white')
        m = np.ma.masked_where(mask_bool, field_img)
        im = ax.imshow(np.flipud(m), cmap=cmap, extent=extent, aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return im


def predict_uv_sigma_image_2d(u_model,
                              v_model,
                              density_model,
                              domain,
                              test_case,
                              rho_threshold_plot=0.3):
    """
    Binary-geometry evaluation --> images of stresses and displacements
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") 
    grid_resolution = 400

    # 1) Grid
    x = np.linspace(domain[0], domain[1], grid_resolution)
    y = np.linspace(domain[2], domain[3], grid_resolution)
    X, Y = np.meshgrid(x, y, indexing='ij')


    points = np.column_stack([X.ravel(), Y.ravel()])
    xt = torch.tensor(points, device=device, dtype=torch.float32).requires_grad_(True)

    prev_mode = density_model.training
    density_model.eval()
    with torch.no_grad():
        rho = density_model(xt).view(-1)
    if prev_mode:
        density_model.train()

    mask = (rho < rho_threshold_plot)
    rho = rho.clone()
    rho[~mask] = 1.0  
    u = u_model(xt)
    v = v_model(xt)

    gu = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    gv = torch.autograd.grad(v, xt, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    eps11 = gu[:, 0]
    eps22 = gv[:, 1]
    eps12 = 0.5 * (gu[:, 1] + gv[:, 0])
    tr = eps11 + eps22

    # 5) Stresses with solid Lamé (no SIMP)
    lam = Properties(test_case).lame_lambda
    mu  = Properties(test_case).lame_mu

    s11 = 2.0 * mu * eps11 + lam * tr
    s22 = 2.0 * mu * eps22 + lam * tr
    s12 = 2.0 * mu * eps12

    svm2 = s11**2 - s11*s22 + s22**2 + 3.0*s12**2
    svm  = torch.sqrt(torch.clamp(svm2, min=1e-32)) * test_case.domain_scaling_factor

    # 6) Reshape to images [ny, nx] 
    ny = nx = grid_resolution
    u_img     = u.view(ny, nx).detach().cpu().numpy()
    v_img     = v.view(ny, nx).detach().cpu().numpy()
    sigma_img = svm.view(ny, nx).detach().cpu().numpy()
    rho_img   = rho.view(ny, nx).detach().cpu().numpy()

    return u_img, v_img, sigma_img, rho_img



def save_uv_sigma_to_file(u_img, v_img, sigma_img, rho_img, filename, domain, test_case, rho_threshold=0.3):
    """
    1×3 panels (y_disp, x_disp, von Mises) 
    """

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    ny, nx = u_img.shape
    x = np.linspace(domain[0], domain[1], nx)
    y = np.linspace(domain[2], domain[3], ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Binary mask (void = True)
    mask = (rho_img < rho_threshold)

    # Apply mask + scale (no transpose!)
    L0 = Properties(test_case).L0 if hasattr(Properties(test_case), "L0") else 1.0
    v_plot = np.ma.masked_where(mask, v_img) * L0
    u_plot = np.ma.masked_where(mask, u_img) * L0
    s_plot = np.ma.masked_where(mask, sigma_img)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # y_disp
    ymin, ymax = float(v_plot.min()), float(v_plot.max())
    if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
        c0 = axes[0].contourf(X, Y, v_plot, cmap='jet', levels=np.linspace(ymin, ymax + 1e-9, 1000))
        axes[0].set_title('y_disp'); axes[0].axis('image'); axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
        cb0 = fig.colorbar(c0, ax=axes[0], orientation='horizontal'); cb0.ax.tick_params(labelrotation=-45)
        cb0.ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        if ymax not in cb0.get_ticks(): cb0.set_ticks(np.sort(np.append(cb0.get_ticks(), ymax)))
    else:
        axes[0].set_title('y_disp (constant)'); axes[0].axis('image'); axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
        axes[0].text(0.5, 0.5, f'Constant: {ymin:.4f}', ha='center', va='center', transform=axes[0].transAxes)

    # x_disp
    xmin, xmax = float(u_plot.min()), float(u_plot.max())
    if np.isfinite(xmin) and np.isfinite(xmax) and xmin != xmax:
        c1 = axes[1].contourf(X, Y, u_plot, cmap='jet', levels=np.linspace(xmin, xmax + 1e-9, 1000))
        axes[1].set_title('x_disp'); axes[1].axis('image'); axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
        cb1 = fig.colorbar(c1, ax=axes[1], orientation='horizontal'); cb1.ax.tick_params(labelrotation=-45)
        cb1.ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        if xmax not in cb1.get_ticks(): cb1.set_ticks(np.sort(np.append(cb1.get_ticks(), xmax)))
    else:
        axes[1].set_title('x_disp (constant)'); axes[1].axis('image'); axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
        axes[1].text(0.5, 0.5, f'Constant: {xmin:.4f}', ha='center', va='center', transform=axes[1].transAxes)

    # von Mises
    smin, smax = float(s_plot.min()), float(s_plot.max())
    if np.isfinite(smin) and np.isfinite(smax) and smin != smax:
        c2 = axes[2].contourf(X, Y, s_plot, cmap='jet', levels=np.linspace(smin, smax + 1e-9, 1000))
        axes[2].set_title('Von Mises Stress'); axes[2].axis('image'); axes[2].set_xlabel('x'); axes[2].set_ylabel('y')
        cb2 = fig.colorbar(c2, ax=axes[2], orientation='horizontal', extend='both'); cb2.ax.tick_params(labelrotation=-45)
        ticks = cb2.get_ticks()
        if smin not in ticks: ticks = np.append(ticks, smin)
        if smax not in ticks: ticks = np.append(ticks, smax)
        cb2.set_ticks(np.sort(ticks))
    else:
        axes[2].set_title('Von Mises Stress (constant)'); axes[2].axis('image'); axes[2].set_xlabel('x'); axes[2].set_ylabel('y')
        axes[2].text(0.5, 0.5, f'Constant: {smin:.4f}', ha='center', va='center', transform=axes[2].transAxes)

    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)






