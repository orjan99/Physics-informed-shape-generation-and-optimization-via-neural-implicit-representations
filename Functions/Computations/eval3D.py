import torch 
import numpy as np 
import scipy.ndimage as ndi   #type: ignore 
 


def compute_eval_metrics_3d(self, phase: str, model_kind: str):

    prev_sdf_train = self.SDF_GINN_model.training
    prev_rho_train = self.density_GINN_model.training

    try:
        self.SDF_GINN_model.eval()
        self.density_GINN_model.eval()

        n = int(self.eval_grid_n)
        batch = int(self.eval_batch)

        bounds6 = _parse_bounds_3d(self.test_case.domain)  
        xs, ys, zs, coords = _make_eval_grid_3d(bounds6, n=n, device=self.device)  
        omega = _eval_occupancy_on_grid(self, coords, model_kind=model_kind, batch=batch)  
        in_env, in_prohib, in_thick = _eval_region_masks_on_grid(self, coords)

        # voxelize 
        omega_np, env_np, prohib_np, thick_np = _reshape_masks_to_grid(
            omega, in_env, in_prohib, in_thick, n=n
        )

        cell_vol = _cell_volume_from_linspaces(bounds6, n=n)
        vol_domain = float(self.test_case.domain_volume)

        vol_frac = _volume_fraction(omega_np, cell_vol, vol_domain)
        vol_outside_env_share = _outside_envelope_share(omega_np, env_np, cell_vol)
        prohibited_share = _region_share_voxels(omega_np, prohib_np, cell_vol)
        thickness_share = _region_share_voxels(omega_np, thick_np, cell_vol)
        boundary_share = _boundary_share_3d(
            self,
            bounds6=bounds6,
            model_kind=model_kind,
            n_total=int(self.eval_boundary_n),
        )

        b0, dc_share = _connectivity_metrics_3d(
            omega_np=omega_np,
            env_np=env_np,
            cell_vol=cell_vol,
        )

        omega_in_env = omega_np & env_np
        b0_in_env, dc_in_env_share = _connectivity_metrics_3d(
            omega_np=omega_in_env,
            env_np=env_np,   
            cell_vol=cell_vol,
        )

        cd1 = _cd1_interface_to_boundary_3d(
            self,
            omega_np=omega_np,
            xs=xs,
            ys=ys,
            zs=zs,
        )

        return {
            "grid_n": int(n),
            "vol_frac": float(vol_frac),
            "vol_outside_env_share": float(vol_outside_env_share),
            "boundary_share": float(boundary_share),
            "prohibited_share": float(prohibited_share),
            "thickness_share": float(thickness_share),
            "b0": float(b0),
            "b0_in_env": float(b0_in_env),
            "dc_share": float(dc_share),
            "dc_in_env_share": float(dc_in_env_share),
            "cd1": float(cd1),
        }

    finally:
        if prev_sdf_train:
            self.SDF_GINN_model.train()
        else:
            self.SDF_GINN_model.eval()

        if prev_rho_train:
            self.density_GINN_model.train()
        else:
            self.density_GINN_model.eval()



def _volume_fraction(omega_np: np.ndarray, cell_vol: float, vol_domain: float) -> float:
    vol_omega = float(np.sum(omega_np) * cell_vol)
    if vol_domain <= 0.0:
        return 0.0
    return vol_omega / vol_domain


def _outside_envelope_share(omega_np: np.ndarray, env_np: np.ndarray, cell_vol: float) -> float:
    outside_env = ~env_np
    vol_x_minus_e = float(np.sum(outside_env) * cell_vol)
    vol_om_minus_e = float(np.sum(omega_np & outside_env) * cell_vol)
    if vol_x_minus_e <= 0.0:
        return 0.0
    return vol_om_minus_e / vol_x_minus_e


def _region_share_voxels(omega_np: np.ndarray, region_np: np.ndarray, cell_vol: float) -> float:
    vol_R = float(np.sum(region_np) * cell_vol)
    vol_OcapR = float(np.sum(omega_np & region_np) * cell_vol)
    if vol_R <= 0.0:
        return 0.0
    return vol_OcapR / vol_R


def _boundary_share_3d(self, bounds6, model_kind: str, n_total: int) -> float:
    bd_pts = sample_domain_boundary_points_3d(bounds6, int(n_total), self.device)
    if bd_pts.numel() == 0:
        return 0.0

    with torch.no_grad():
        if str(model_kind).lower().strip() == "sdf":
            vb = self.SDF_GINN_model(bd_pts).view(-1)
            occ = (vb <= 0.0)
        else:
            vb = self.density_GINN_model(bd_pts).view(-1)
            occ = (vb >= float(self.plot_threshold))

    return float(occ.float().mean().item())


def _connectivity_metrics_3d(omega_np: np.ndarray, env_np: np.ndarray, cell_vol: float):
    
    sizes = label_connected_components_3d(omega_np)
    b0 = float(len(sizes))

    if len(sizes) <= 1:
        dc_share = 0.0
    else:
        sizes_sorted = sorted(sizes, reverse=True)
        dc_vox = float(sum(sizes_sorted[1:]))
        vol_E = float(np.sum(env_np) * cell_vol)
        dc_share = 0.0 if vol_E <= 0 else (dc_vox * cell_vol) / vol_E

    return b0, float(dc_share)


def _cd1_interface_to_boundary_3d(self, omega_np: np.ndarray, xs: torch.Tensor, ys: torch.Tensor, zs: torch.Tensor) -> float:
    
    bmask = boundary_mask_6n(omega_np)
    idx = np.argwhere(bmask)

    if idx.shape[0] == 0:
        return float("inf")

    # downsample boundary points if needed
    max_bd = int(getattr(self, "eval_max_boundary_pts", 20_000))
    if idx.shape[0] > max_bd:
        pick = np.random.choice(idx.shape[0], max_bd, replace=False)
        idx = idx[pick]

    xs_cpu = xs.detach().cpu().numpy()
    ys_cpu = ys.detach().cpu().numpy()
    zs_cpu = zs.detach().cpu().numpy()

    P_np = np.stack(
        [xs_cpu[idx[:, 0]], ys_cpu[idx[:, 1]], zs_cpu[idx[:, 2]]],
        axis=1
    ).astype(np.float32)

    # interface samples
    iface = getattr(self.test_case, "interfaces", None)
    if iface is not None and hasattr(iface, "sample_points_from_all_interfaces"):
        Q = iface.sample_points_from_all_interfaces(int(self.eval_interface_n), output_type="torch_tensor")
        if not isinstance(Q, torch.Tensor):
            Q = torch.as_tensor(Q, dtype=torch.float32)
        Q = Q.to(device=self.device, dtype=torch.float32)
    else:
        Q = torch.empty((0, 3), device=self.device, dtype=torch.float32)

    if Q.numel() == 0:
        return float("inf")

    P = torch.tensor(P_np, device=self.device, dtype=torch.float32)

    # mps fallback (kept exactly)
    use_cpu = (self.device.type == "mps")
    if use_cpu:
        P = P.cpu()
        Q = Q.cpu()

    acc = 0.0
    cnt = 0
    chunk = int(getattr(self, "eval_cdist_chunk", 4_096))

    with torch.no_grad():
        for s in range(0, Q.shape[0], chunk):
            q = Q[s:s + chunk]
            d = torch.cdist(q, P)
            md = torch.min(d, dim=1).values
            acc += torch.sum(md * md).item()
            cnt += q.shape[0]

    return float(np.sqrt(acc / max(cnt, 1)))

def _parse_bounds_3d(domain6):
    domain = [float(v) for v in list(domain6)]
    if len(domain) != 6:
        raise ValueError(f"3D domain must have 6 values [x0,x1,y0,y1,z0,z1], got: {domain6}")
    x0, x1, y0, y1, z0, z1 = domain
    return x0, x1, y0, y1, z0, z1


def _make_eval_grid_3d(bounds6, n: int, device):
    x0, x1, y0, y1, z0, z1 = bounds6
    xs = torch.linspace(x0, x1, n, device=device)
    ys = torch.linspace(y0, y1, n, device=device)
    zs = torch.linspace(z0, z1, n, device=device)

    XX, YY, ZZ = torch.meshgrid(xs, ys, zs, indexing="ij")
    coords = torch.stack([XX.reshape(-1), YY.reshape(-1), ZZ.reshape(-1)], dim=1)
    return xs, ys, zs, coords


def _cell_volume_from_linspaces(bounds6, n: int) -> float:
    x0, x1, y0, y1, z0, z1 = bounds6
    dx = float((x1 - x0) / max(n - 1, 1))
    dy = float((y1 - y0) / max(n - 1, 1))
    dz = float((z1 - z0) / max(n - 1, 1))
    return abs(dx * dy * dz)


def _eval_occupancy_on_grid(self, coords: torch.Tensor, model_kind: str, batch: int) -> torch.Tensor:
    kind = str(model_kind).lower().strip()
    if kind == "sdf":
        vals = eval_forward_batched(self.SDF_GINN_model, coords, batch=batch).view(-1)
        return (vals <= 0.0)
    elif kind == "density":
        vals = eval_forward_batched(self.density_GINN_model, coords, batch=batch).view(-1)
        return (vals >= float(self.plot_threshold))
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")


def _eval_region_masks_on_grid(self, coords: torch.Tensor):
    in_env = mask_inside_envelope(self, coords)
    in_prohib = mask_from_interface_fn(self, "is_inside_prohibited_region", coords)
    in_thick = mask_from_interface_fn(self, "is_inside_interface_thickness", coords)
    return in_env, in_prohib, in_thick


def _reshape_masks_to_grid(omega, in_env, in_prohib, in_thick, n: int):
    omega_np = omega.detach().cpu().numpy().reshape(n, n, n).astype(bool)
    env_np = in_env.detach().cpu().numpy().reshape(n, n, n).astype(bool)
    prohib_np = in_prohib.detach().cpu().numpy().reshape(n, n, n).astype(bool)
    thick_np = in_thick.detach().cpu().numpy().reshape(n, n, n).astype(bool)
    return omega_np, env_np, prohib_np, thick_np


def eval_forward_batched(model, coords, batch: int):
    outs = []
    with torch.no_grad():
        for s in range(0, coords.shape[0], batch):
            y = model(coords[s:s + batch])
            if isinstance(y, (tuple, list)):
                y = y[0]
            outs.append(y.detach())
    return torch.cat(outs, dim=0)


def mask_from_interface_fn(self, fn_name: str, coords):
    iface = getattr(self.test_case, "interfaces", None)
    if iface is None or (not hasattr(iface, fn_name)):
        return torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)

    fn = getattr(iface, fn_name)
    try:
        _, idx = fn(coords)
        m = torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)
        if isinstance(idx, torch.Tensor) and idx.numel() > 0:
            m[idx] = True
        return m
    except Exception:
        return torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)


def mask_inside_envelope(self, coords):
    iface = getattr(self.test_case, "interfaces", None)

    for name in ("is_inside_design_envelope", "is_inside_envelope", "is_inside_design_region"):
        if iface is not None and hasattr(iface, name):
            try:
                _, idx = getattr(iface, name)(coords)
                m = torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)
                if isinstance(idx, torch.Tensor) and idx.numel() > 0:
                    m[idx] = True
                    return m
            except Exception:
                pass

    return torch.ones(coords.shape[0], dtype=torch.bool, device=coords.device)


def sample_domain_boundary_points_3d(domain6, n_total: int, device):
    x0, x1, y0, y1, z0, z1 = [float(v) for v in domain6]
    n_face = max(1, int(n_total // 6))

    r = lambda n: torch.rand(n, device=device)

    pts = []
    for xv in (x0, x1):
        yy = r(n_face) * (y1 - y0) + y0
        zz = r(n_face) * (z1 - z0) + z0
        pts.append(torch.stack([torch.full_like(yy, xv), yy, zz], dim=1))
    for yv in (y0, y1):
        xx = r(n_face) * (x1 - x0) + x0
        zz = r(n_face) * (z1 - z0) + z0
        pts.append(torch.stack([xx, torch.full_like(xx, yv), zz], dim=1))
    for zv in (z0, z1):
        xx = r(n_face) * (x1 - x0) + x0
        yy = r(n_face) * (y1 - y0) + y0
        pts.append(torch.stack([xx, yy, torch.full_like(xx, zv)], dim=1))

    P = torch.cat(pts, dim=0)
    if P.shape[0] > n_total:
        idx = torch.randperm(P.shape[0], device=device)[:n_total]
        P = P[idx]
    return P


def boundary_mask_6n(mask3d):
    p = np.pad(mask3d.astype(bool), 1, mode="constant", constant_values=False)
    c = p[1:-1, 1:-1, 1:-1]
    xm = p[:-2, 1:-1, 1:-1]
    xp = p[2:, 1:-1, 1:-1]
    ym = p[1:-1, :-2, 1:-1]
    yp = p[1:-1, 2:, 1:-1]
    zm = p[1:-1, 1:-1, :-2]
    zp = p[1:-1, 1:-1, 2:]
    return c & (~xm | ~xp | ~ym | ~yp | ~zm | ~zp)


def label_connected_components_3d(mask3d):
    try:
        struct = np.array(
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
             [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
             [[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
            dtype=np.uint8
        )
        lab, n = ndi.label(mask3d.astype(np.uint8), structure=struct)
        if n <= 0:
            return []
        sizes = ndi.sum(mask3d.astype(np.uint8), lab, index=list(range(1, n + 1)))
        return [int(s) for s in sizes]
    except Exception:
        pass

    nx, ny, nz = mask3d.shape
    visited = np.zeros_like(mask3d, dtype=bool)
    sizes = []
    neigh = [(-1, 0, 0), (1, 0, 0),
             (0, -1, 0), (0, 1, 0),
             (0, 0, -1), (0, 0, 1)]

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if (not mask3d[i, j, k]) or visited[i, j, k]:
                    continue
                stack = [(i, j, k)]
                visited[i, j, k] = True
                cnt = 0
                while stack:
                    ci, cj, ck = stack.pop()
                    cnt += 1
                    for di, dj, dk in neigh:
                        ni, nj, nk = ci + di, cj + dj, ck + dk
                        if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                            if mask3d[ni, nj, nk] and (not visited[ni, nj, nk]):
                                visited[ni, nj, nk] = True
                                stack.append((ni, nj, nk))
                sizes.append(cnt)
    return sizes


