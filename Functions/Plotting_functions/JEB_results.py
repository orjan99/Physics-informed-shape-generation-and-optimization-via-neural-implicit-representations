import os 
from matplotlib import pyplot as plt #type: ignore 
import plotly.graph_objects as go #type: ignore 
from plotly.subplots import make_subplots #type: ignore 
from matplotlib import cm #type: ignore 
from skimage import measure #type: ignore  
from Functions.Point_Sampling.point_sampler import *
import torch
import numpy as np
from Functions.Training.Properties import *  
 

def save_densities_as_points_obj(densities, positions, iso_level, filename='', only_solid=True):
    """
    Save color coded points as .obj. 
    """
    if only_solid:
        solid = np.where(densities.flatten() > iso_level)[0]
        obj_x = positions[solid, 0]
        obj_y = positions[solid, 1]
        obj_z = positions[solid, 2]
        solid_rho = densities.flatten()[solid]
    else:
        obj_x = positions[:, 0]
        obj_y = positions[:, 1]
        obj_z = positions[:, 2]
        solid_rho = densities.flatten()

    mapper = cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="jet")
    rho_color_map = mapper.to_rgba(solid_rho)
    rho_color_map = np.reshape(rho_color_map, (solid_rho.size, 4))
    plt.close()

    with open(filename, "w") as f:
        for i in range(len(obj_x)):
            f.write(
                "v {} {} {} {} {} {}\n".format(
                    obj_x[i], obj_y[i], obj_z[i],
                    rho_color_map[i, 0] * 255.0,
                    rho_color_map[i, 1] * 255.0,
                    rho_color_map[i, 2] * 255.0
                )
            )

def pad_with_zeros(density_grid):
    """
    Pad a (nx, ny, nz) grid by one cell in each direction with zeros (for marching cubes).
    """
    c0 = np.full((1, density_grid.shape[1], density_grid.shape[2]), 0.0, dtype=np.float32)
    density_grid = np.concatenate((c0, density_grid, c0), axis=0)
    c1 = np.full((density_grid.shape[0], 1, density_grid.shape[2]), 0.0, dtype=np.float32)
    density_grid = np.concatenate((c1, density_grid, c1), axis=1)
    c2 = np.full((density_grid.shape[0], density_grid.shape[1], 1), 0.0, dtype=np.float32)
    density_grid = np.concatenate((c2, density_grid, c2), axis=2)
    return density_grid

def save_density_iso_surface(density_grid, spacing, iso_level, filename):
    """
    Marching cubes iso-surface from a (nx, ny, nz) density grid.
    """
    density_grid = pad_with_zeros(density_grid)
    if np.amax(density_grid) < iso_level or np.amin(density_grid) > iso_level:
        print('cannot save density grid cause the levelset is empty')
        return

    verts, faces, normals, values = measure.marching_cubes(
        density_grid, level=iso_level, spacing=spacing,
        gradient_direction="ascent", method='lewiner'
    )

    with open(filename, 'w') as file:
        for v in verts:
            file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for n in normals:
            file.write(f"vn {n[0]} {n[1]} {n[2]}\n")
        for tri in faces:
            i0, i1, i2 = (tri[0] + 1, tri[1] + 1, tri[2] + 1)
            file.write(f"f {i0}//{i0} {i1}//{i1} {i2}//{i2}\n")

@torch.no_grad()
def predict_densities_grid_3d(density_model, domain, n_cells, enforce_density, device=None, max_batch=200_000):
    """
    Sample density model on a 3D grid (centers), apply constraints, and return:
      (density_grid[nx,ny,nz], spacing_xyz, positions[n,3])
    """
    nx, ny, nz = int(n_cells[0]), int(n_cells[1]), int(n_cells[2])
    xs = get_grid_centers(domain, [nx, ny, nz]).astype(np.float32)  # (N,3)
    N = xs.shape[0]

    if device is None:
        try:
            device = next(density_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    def _run_chunk(start, end):
        xt = torch.from_numpy(xs[start:end]).to(device)
        rho = density_model(xt).view(-1, 1)
        rho = torch.clamp(rho, 0.0, 1.0)
        rho, _ = enforce_density.apply(xt, rho, None, [nx,ny,nz], domain)  
        return rho.squeeze(1).detach().cpu().numpy()

    chunks = []
    for s in range(0, N, max_batch):
        e = min(N, s + max_batch)
        chunks.append(_run_chunk(s, e))
    rho_all = np.concatenate(chunks, axis=0)

    density_grid = rho_all.reshape(nx, ny, nz)
    spacing = get_grid_centers_spacing(domain, [nx, ny, nz])
    return density_grid, spacing, xs

@torch.no_grad()
def save_density_outputs_3d(density_model, domain, n_cells, enforce_density, save_path, save_prefix, save_postfix,
                            iso_level=0.25, device=None):
    """
    Convenience wrapper
    """
    density_grid, spacing, xs = predict_densities_grid_3d(density_model, domain, n_cells, enforce_density, device=device)
    positions = xs  # (N,3)
    densities = density_grid.reshape(-1, 1)

    fn_pts = os.path.join(save_path, save_prefix + 'density'    + save_postfix + '.obj')
    fn_iso = os.path.join(save_path, save_prefix + 'density-iso' + save_postfix + '.obj')

    save_densities_as_points_obj(densities, positions, iso_level=iso_level, filename=fn_pts, only_solid=False)
    save_density_iso_surface(density_grid, spacing=spacing, iso_level=iso_level, filename=fn_iso)
    return fn_pts, fn_iso

@torch.no_grad()
def predict_densities_grid_3d_unconstrained(density_model, domain, n_cells,
                                            device=None, max_batch=200_000):
    """
    Same as predict_densities_grid_3d but without enforcing density constraints.
    """
    nx, ny, nz = int(n_cells[0]), int(n_cells[1]), int(n_cells[2])
    xs = get_grid_centers(domain, [nx, ny, nz]).astype(np.float32)  # (N,3)
    N = xs.shape[0]

    if device is None:
        try:
            device = next(density_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    chunks = []
    for s in range(0, N, max_batch):
        e = min(N, s + max_batch)
        xt = torch.from_numpy(xs[s:e]).to(device)
        rho = density_model(xt).view(-1, 1)
        rho = torch.clamp(rho, 0.0, 1.0)
        chunks.append(rho.squeeze(1).detach().cpu().numpy())

    rho_all = np.concatenate(chunks, axis=0)
    density_grid = rho_all.reshape(nx, ny, nz)
    spacing = get_grid_centers_spacing(domain, [nx, ny, nz])
    return density_grid, spacing, xs


@torch.no_grad()
def save_density_outputs_3d_unconstrained(density_model, domain, n_cells,
                                          save_path, save_prefix, save_postfix,
                                          iso_level=0.25, device=None):
    """
    Marching cubes / point cloud from density WITHOUT applying density constraints.
    """
    density_grid, spacing, xs = predict_densities_grid_3d_unconstrained(
        density_model, domain, n_cells, device=device
    )
    positions = xs
    densities = density_grid.reshape(-1, 1)

    fn_pts = os.path.join(save_path,
                          save_prefix + 'density-unconstrained' + save_postfix + '.obj')
    fn_iso = os.path.join(save_path,
                          save_prefix + 'density-unconstrained-iso' + save_postfix + '.obj')

    save_densities_as_points_obj(densities, positions, iso_level=iso_level,
                                 filename=fn_pts, only_solid=False)
    save_density_iso_surface(density_grid, spacing=spacing,
                             iso_level=iso_level, filename=fn_iso)
    return fn_pts, fn_iso



@torch.no_grad()
def save_sdf_outputs_3d(
    sdf_model,
    domain,
    n_cells,
    save_path,
    save_prefix,
    save_postfix,
    iso_level: float = 0.0,       
    device=None,
    enforce_density=None,         
    tag: str = 'sdf',
    sdf_to_rho_alpha: float = 1000.0,
    max_batch: int = 200_000,
):
    """
    Evaluate an SDF model on a 3D grid, convert SDF -> density via a steep sigmoid,
    optionally apply density constraints, and then extract an iso-surface 
    """
    nx, ny, nz = int(n_cells[0]), int(n_cells[1]), int(n_cells[2])
    xs = get_grid_centers(domain, [nx, ny, nz]).astype(np.float32)  # (N,3)
    N = xs.shape[0]

    if device is None:
        try:
            device = next(sdf_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    xs_t = torch.from_numpy(xs).to(device)

    #  Evaluate SDF on the grid in chunks 
    phi_chunks = []
    for s in range(0, N, max_batch):
        e = min(N, s + max_batch)
        phi = sdf_model(xs_t[s:e]).view(-1, 1)
        phi_chunks.append(phi)
    phi_all = torch.cat(phi_chunks, dim=0)  # [N,1]

    rho_all = sdf_to_density(phi_all, alpha=sdf_to_rho_alpha) 

    if enforce_density is not None:
        rho_all, _ = enforce_density.apply(xs_t, rho_all, None, n_cells, domain)

    # --- 4) Reshape to grid & compute spacing ---
    rho_np = rho_all.squeeze(1).detach().cpu().numpy()
    density_grid = rho_np.reshape(nx, ny, nz)
    spacing = get_grid_centers_spacing(domain, [nx, ny, nz])

    positions = xs                    # (N,3)
    densities = density_grid.reshape(-1, 1)

    # --- 5) SDF iso-level -> density iso-level ---
    # density = sigmoid(-alpha * SDF) = 1 / (1 + exp(alpha * SDF))

    iso_density = float(1.0 / (1.0 + np.exp(sdf_to_rho_alpha * iso_level)))
  

    fn_pts = os.path.join(save_path,
                          save_prefix + tag + save_postfix + '.obj')
    fn_iso = os.path.join(save_path,
                          save_prefix + tag + '-iso' + save_postfix + '.obj')


    save_densities_as_points_obj(
        densities,
        positions,
        iso_level=iso_density,
        filename=fn_pts,
        only_solid=False,            
    )


    save_density_iso_surface(
        density_grid,
        spacing=spacing,
        iso_level=iso_density,
        filename=fn_iso
    )

    return fn_pts, fn_iso



def sdf_to_density(phi: torch.Tensor, alpha: float = 1000.0) -> torch.Tensor:
    return torch.sigmoid(-alpha * phi)


# Plotting helpers (3D) -------------------------
@staticmethod
def ticks(vmin, vmax):
    tv = np.linspace(float(vmin), float(vmax), 6)
    tt = [f"{t:.4f}" for t in tv]
    if len(tt) >= 2:
        tt[0] += " Min"; tt[-1] += " Max"
    return tv, tt

@staticmethod
def density_mask_on_points_3d(density_model, pts, rho_threshold_plot: float):
    prev = density_model.training
    density_model.eval()
    with torch.no_grad():
        rho = density_model(pts).view(-1)
    if prev:
        density_model.train()
    return (rho >= rho_threshold_plot)

@staticmethod
def predict_uvw_sigma_points_3d(u_model,
                                  v_model,
                                  w_model,
                                  density_model,
                                  rho_threshold_plot: float,
                                  n_eval_points: int,
                                  max_batch: int,
                                  device,
                                  test_case): 

    # 1) sample once per snapshot
    ps = Point_Sampler(test_case.domain, test_case.interfaces,
                        num_points_domain=n_eval_points, num_points_interface=0)
    pts_full = next(ps).to(device)

    u_model.eval(); v_model.eval(); w_model.eval()

    # 2) binary mask from density
    solid_mask = density_mask_on_points_3d(density_model, pts_full, rho_threshold_plot)
    if not torch.any(solid_mask):
        with torch.no_grad():
            rho_all = density_model(pts_full).view(-1)
        q = torch.quantile(rho_all, 0.9)
        solid_mask = rho_all >= q

    pts_solid = pts_full[solid_mask].detach()

    # 3) stresses/displacements with solid Lamé (no SIMP)
    props = Properties(test_case)
    lam = props.lame_lambda.to(device)
    mu  = props.lame_mu.to(device)

    xyz_list, u_list, v_list, w_list, vm_list = [], [], [], [], []

    for s in range(0, pts_solid.shape[0], max_batch):
        e = min(pts_solid.shape[0], s + max_batch)
        x = pts_solid[s:e].clone().detach().requires_grad_(True)

        u = (u_model(x) * test_case.domain_scaling_factor)
        v = (v_model(x) * test_case.domain_scaling_factor)
        w = (w_model(x) * test_case.domain_scaling_factor)

        gu = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=False, retain_graph=False)[0]
        gv = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=False, retain_graph=False)[0]
        gw = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=False, retain_graph=False)[0]

        e11 = gu[:, 0]; e22 = gv[:, 1]; e33 = gw[:, 2]
        e12 = 0.5*(gu[:, 1] + gv[:, 0])
        e13 = 0.5*(gu[:, 2] + gw[:, 0])
        e23 = 0.5*(gv[:, 2] + gw[:, 1])
        tr  = e11 + e22 + e33

        s11 = 2*mu*e11 + lam*tr
        s22 = 2*mu*e22 + lam*tr
        s33 = 2*mu*e33 + lam*tr
        s12 = 2*mu*e12
        s13 = 2*mu*e13
        s23 = 2*mu*e23

        vm2 = 0.5*((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2) + 3*(s12**2 + s13**2 + s23**2)
        vm  = torch.sqrt(torch.clamp(vm2, min=1e-32)) * test_case.domain_scaling_factor

        xyz_list.append(x.detach().cpu().numpy())
        u_list.append(u.detach().cpu().numpy().reshape(-1))
        v_list.append(v.detach().cpu().numpy().reshape(-1))
        w_list.append(w.detach().cpu().numpy().reshape(-1))
        vm_list.append(vm.detach().cpu().numpy().reshape(-1))

    pts_np = np.concatenate(xyz_list, axis=0) if len(xyz_list) else np.zeros((0,3))
    u_np   = np.concatenate(u_list, axis=0)   if len(u_list)   else np.zeros((0,))
    v_np   = np.concatenate(v_list, axis=0)   if len(v_list)   else np.zeros((0,))
    w_np   = np.concatenate(w_list, axis=0)   if len(w_list)   else np.zeros((0,))
    vm_np  = np.concatenate(vm_list, axis=0)  if len(vm_list)  else np.zeros((0,))

    return pts_np, u_np, v_np, w_np, vm_np

@staticmethod
@torch.no_grad()
def save_uvw_sigma_to_file_3d(points_xyz, u_vals, v_vals, w_vals, vm_vals,
                              filename: str, title: str = None,
                              marker_size: int = 2):


    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if points_xyz.shape[0] == 0:
        # write an empty HTML to avoid crashes
        html_name = os.path.splitext(filename)[0] + ".html"
        with open(html_name, "w") as f:
            f.write("<html><body><h3>No solid points to plot yet.</h3></body></html>")
        return

    X, Y, Z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]

    # 2x2 with tighter domains so vertical colorbars can sit to each subplot's right
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
               [{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        horizontal_spacing=0.08,
        vertical_spacing=0.08,
        subplot_titles=('X Displacement NN', 'Y Displacement NN',
                        'Z Displacement NN', 'Von Mises NN')
    )

    def _scene(i):
        return dict(
            xaxis=dict(visible=False, showgrid=False, zeroline=False),
            yaxis=dict(visible=False, showgrid=False, zeroline=False),
            zaxis=dict(visible=False, showgrid=False, zeroline=False),
            bgcolor="white",
            aspectmode='data'
        )

    locs = [(0.46, 0.82), (0.98, 0.82), (0.46, 0.28), (0.98, 0.28)]

    # u
    fig.add_trace(go.Scatter3d(
        x=X, y=Y, z=Z, mode='markers',
        marker=dict(size=marker_size, color=u_vals,
                    colorscale='Jet',
                    colorbar=dict(x=locs[0][0], y=locs[0][1], len=0.35, thickness=18),
                    showscale=True)),
        row=1, col=1)

    # v
    fig.add_trace(go.Scatter3d(
        x=X, y=Y, z=Z, mode='markers',
        marker=dict(size=marker_size, color=v_vals,
                    colorscale='Jet',
                    colorbar=dict(x=locs[1][0], y=locs[1][1], len=0.35, thickness=18),
                    showscale=True)),
        row=1, col=2)

    # w
    fig.add_trace(go.Scatter3d(
        x=X, y=Y, z=Z, mode='markers',
        marker=dict(size=marker_size, color=w_vals,
                    colorscale='Jet',
                    colorbar=dict(x=locs[2][0], y=locs[2][1], len=0.35, thickness=18),
                    showscale=True)),
        row=2, col=1)

    # von Mises
    fig.add_trace(go.Scatter3d(
        x=X, y=Y, z=Z, mode='markers',
        marker=dict(size=marker_size, color=vm_vals,
                    colorscale='Jet',
                    colorbar=dict(x=locs[3][0], y=locs[3][1], len=0.35, thickness=18),
                    showscale=True)),
        row=2, col=2)

    fig.update_layout(
        title=title or "Displacements & Von Mises (NN)",
        height=900, width=1600,
        paper_bgcolor="white",
        margin=dict(t=60, b=20, l=20, r=20),
        scene=_scene(1), scene2=_scene(2), scene3=_scene(3), scene4=_scene(4),
        showlegend=False
    )

    try:
        fig.write_image(filename, scale=2)
    except Exception:
        fig.write_html(os.path.splitext(filename)[0] + ".html", include_plotlyjs='cdn')



# Function to plot the points on the surfae or inside the geometry (SDF < 0)
def SDF_geometry_plot(num_points,ginn_model,mesh,device,test_case): 
    point_sampler_debug = Point_Sampler(test_case.domain, test_case.interfaces,
                  num_points_domain=num_points,num_points_interface=0)


    points = next(point_sampler_debug).to(device)

    with torch.no_grad():
      ginn_model.eval()
      sdf = ginn_model(points).view(-1)

    tolerance = 1e-6
    mask = sdf <= tolerance
    pts_in   = points[mask]
    sdf_in   = sdf[mask]

    pts_in = pts_in.cpu().numpy()
    sdf_in = sdf_in.cpu().numpy()

    print("Number of points inside the mesh: ", len(pts_in))
    print("Number of points outside the mesh: ", len(points) - len(pts_in))

    # 2) Prepare the mesh trace
    verts = mesh.vertices
    faces = mesh.faces
    mesh_trace = go.Mesh3d(
        x=verts[:,0], y=verts[:,1], z=verts[:,2],
        i=faces[:,0], j=faces[:,1], k=faces[:,2],
        color='lightgray',
        opacity=0.2,
        flatshading=True
    )

    scatter_in = go.Scatter3d(
        x=pts_in[:,0],
        y=pts_in[:,1],
        z=pts_in[:,2],
        mode='markers',
        marker=dict(
            size=2,
            color=sdf_in,          
            colorscale='Blues',    
            cmin= sdf.min().item(),
            cmax=0,
            opacity=0.8
        ),
        name='SDF ≤ 0'
    )

  
    fig = go.Figure(data=[mesh_trace, scatter_in])
    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Mesh with Inside‐Surface Grid Points (SDF ≤ 0)"
    )
    return fig
