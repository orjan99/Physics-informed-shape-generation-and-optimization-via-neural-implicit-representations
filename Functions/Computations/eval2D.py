
import torch 
import numpy as np 
import scipy.ndimage as ndi   #type: ignore 
from Functions.logging.BRIDGE_logging import append_csv_row
import numpy as np
import torch
import math
import cripser  # type: ignore 



def compute_eval_metrics_2d(self, model_kind: str = "density",phase: str = None) -> dict:
    device = getattr(self, "device", torch.device("cpu"))
    model_kind = str(model_kind).lower().strip()
    if model_kind not in ("density", "sdf"):
        raise ValueError(f"model_kind must be 'density' or 'sdf', got {model_kind}")

    grid_nx = int(getattr(self, "eval_grid_nx", 256))
    grid_ny = int(getattr(self, "eval_grid_ny", 128))
    batch   = int(getattr(self, "eval_batch", 200_000))

    boundary_n       = int(getattr(self, "eval_boundary_n", 2_000))
    interface_n      = int(getattr(self, "eval_interface_n", 1_024))
    max_boundary_pts = int(getattr(self, "eval_max_boundary_pts", 20_000))
    cdist_chunk      = int(getattr(self, "eval_cdist_chunk", 4_096))

    bounds = _parse_bounds_2d(self.test_case.domain)  
    xg = _grid_centers_2d(bounds, grid_nx, grid_ny, device=device)  

    E = torch.ones((xg.shape[0],), dtype=torch.bool, device=device)
    E2d = E.view(grid_ny, grid_nx).detach().cpu().numpy().astype(bool)


    if model_kind == "density":
        rho = _eval_density(self, xg, n_samples=(grid_nx, grid_ny), batch=batch)  
        thr = float(getattr(self, "plot_threshold", 0.5))
        occ = (rho >= thr)
    else:
        sdf = _eval_sdf(self, xg, batch=batch)  
        occ = (sdf <= 0.0)

    occ2d = occ.view(grid_ny, grid_nx).detach().cpu().numpy().astype(bool)
    occ_in_E2d = (occ2d & E2d)

    vol_frac = _safe_div(float(np.count_nonzero(occ_in_E2d)), float(np.count_nonzero(E2d)))
    vol_outside_E_share = _safe_div(
        float(np.count_nonzero(occ2d & (~E2d))),
        float(np.count_nonzero((~E2d)))
    )

    prohibited_share = _region_share(
        self,
        xg,
        occ,
        region_fn=self.test_case.interfaces.is_inside_prohibited_region,
        batch=batch
    )
    thickness_share = _region_share(
        self,
        xg,
        occ,
        region_fn=self.test_case.interfaces.is_inside_interface_thickness,
        batch=batch
    )

    boundary_share = _boundary_share(self, bounds, model_kind=model_kind, n=boundary_n)

    b0 = float(_count_components_2d(occ2d)[0])
    b0_in_E = float(_count_components_2d(occ_in_E2d)[0])

    dc_share = _dc_share(occ2d, E2d)
    dc_in_E_share = _dc_share(occ_in_E2d, E2d)

    interface_pts = _sample_interface_points(self, interface_n, device=device)  # (M,2)
    boundary_pts  = _sample_boundary_points(self, model_kind=model_kind, max_pts=max_boundary_pts)  # (K,2)
    cd1 = _cd1(interface_pts, boundary_pts, cdist_chunk=cdist_chunk)

    return {
        "grid_nx": int(grid_nx),
        "grid_ny": int(grid_ny),
        "vol_frac": float(vol_frac),
        "vol_outside_E_share": float(vol_outside_E_share),
        "boundary_share": float(boundary_share),
        "prohibited_share": float(prohibited_share),
        "thickness_share": float(thickness_share),
        "b0": float(b0),
        "b0_in_E": float(b0_in_E),
        "dc_share": float(dc_share),
        "dc_in_E_share": float(dc_in_E_share),
        "cd1": float(cd1),
    }

def _parse_bounds_2d(domain) -> np.ndarray:
    b = np.array(domain, dtype=np.float64).reshape(-1)
    if b.size == 4:
        xmin, xmax, ymin, ymax = map(float, b.tolist())
        return np.array([[xmin, xmax], [ymin, ymax]], dtype=np.float64)

    b2 = np.array(domain, dtype=np.float64)
    if b2.shape == (2, 2):
        return b2

    raise ValueError(f"Unsupported domain format for 2D bounds: {domain}")


def _grid_centers_2d(bounds: np.ndarray, nx: int, ny: int, device) -> torch.Tensor:
    xmin, xmax = float(bounds[0, 0]), float(bounds[0, 1])
    ymin, ymax = float(bounds[1, 0]), float(bounds[1, 1])

    dx = (xmax - xmin) / float(nx)
    dy = (ymax - ymin) / float(ny)

    xs = torch.linspace(xmin + 0.5 * dx, xmax - 0.5 * dx, nx, device=device)
    ys = torch.linspace(ymin + 0.5 * dy, ymax - 0.5 * dy, ny, device=device)

    X, Y = torch.meshgrid(xs, ys, indexing="xy")  # X,Y: (ny,nx) effectively when using xy
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)  # (N,2)
    return pts


def _sample_domain_boundary_points_2d(bounds: np.ndarray, n: int, device) -> torch.Tensor:
 
    n = max(4, int(n))
    xmin, xmax = float(bounds[0, 0]), float(bounds[0, 1])
    ymin, ymax = float(bounds[1, 0]), float(bounds[1, 1])

    k = max(1, n // 4)
    t = torch.linspace(0.0, 1.0, k, device=device)

    xb = xmin + (xmax - xmin) * t
    xt = xb
    yb = torch.full_like(xb, ymin)
    yt = torch.full_like(xt, ymax)

    yl = ymin + (ymax - ymin) * t
    yr = yl
    xl = torch.full_like(yl, xmin)
    xr = torch.full_like(yr, xmax)

    pts = torch.cat([
        torch.stack([xb, yb], dim=1),
        torch.stack([xt, yt], dim=1),
        torch.stack([xl, yl], dim=1),
        torch.stack([xr, yr], dim=1),
    ], dim=0)

    return pts


def _safe_div(a: float, b: float) -> float:
    if b <= 0.0:
        return 0.0
    return a / b

def _unwrap_model_output(out, kind: str) -> torch.Tensor:
    if isinstance(out, (tuple, list)):
        if kind == "density":
            return out[0]
        if kind == "sdf":
            return out[1]
    return out


@torch.no_grad()
def _eval_density(self, x: torch.Tensor, n_samples, batch: int) -> torch.Tensor:
    model = self.density_GINN_model
    model.eval()
    outs = []
    nx, ny = int(n_samples[0]), int(n_samples[1])

    for i0 in range(0, x.shape[0], batch):
        xb = x[i0:i0 + batch]
        yb = _unwrap_model_output(model(xb), "density").view(-1)

        # Apply your density constraints (same call signature used in training)
        yb, _ = self.enforce_density.apply(
            xb, yb, None, [nx, ny], self.test_case.domain
        )
        yb = yb.clamp(0.0, 1.0)
        outs.append(yb)

    return torch.cat(outs, dim=0)


@torch.no_grad()
def _eval_sdf(self, x: torch.Tensor, batch: int) -> torch.Tensor:
    model = self.SDF_GINN_model
    model.eval()
    outs = []

    for i0 in range(0, x.shape[0], batch):
        xb = x[i0:i0 + batch]
        yb = _unwrap_model_output(model(xb), "sdf").view(-1)
        outs.append(yb)

    return torch.cat(outs, dim=0)



def _region_mask_from_fn(x: torch.Tensor, region_fn) -> torch.Tensor:

    out = region_fn(x)

    if isinstance(out, (tuple, list)) and len(out) == 2:
        _, idx = out
        if torch.is_tensor(idx):
            if idx.dtype == torch.bool:
                if idx.numel() != x.shape[0]:
                    raise ValueError(
                        f"region_fn returned a bool mask with wrong length: "
                        f"{idx.numel()} != {x.shape[0]}"
                    )
                return idx.view(-1).to(device=x.device)
            else:
                idx = idx.view(-1).long().to(device=x.device)
                mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
                if idx.numel() > 0:
                    mask[idx] = True
                return mask

        if isinstance(idx, np.ndarray):
            if idx.dtype == np.bool_:
                if idx.size != x.shape[0]:
                    raise ValueError(
                        f"region_fn returned a numpy bool mask with wrong length: "
                        f"{idx.size} != {x.shape[0]}"
                    )
                return torch.from_numpy(idx).to(device=x.device)
            else:
                idx_t = torch.from_numpy(idx.astype(np.int64)).to(device=x.device)
                mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
                if idx_t.numel() > 0:
                    mask[idx_t] = True
                return mask

        raise TypeError(f"Unsupported region index type: {type(idx)}")


    if torch.is_tensor(out):
        if out.dtype != torch.bool or out.numel() != x.shape[0]:
            raise ValueError(
                f"region_fn returned tensor of shape/dtype not usable as a mask: "
                f"dtype={out.dtype}, numel={out.numel()}, expected bool with numel={x.shape[0]}"
            )
        return out.view(-1).to(device=x.device)

    if isinstance(out, np.ndarray):
        if out.dtype != np.bool_ or out.size != x.shape[0]:
            raise ValueError(
                f"region_fn returned numpy array not usable as mask: "
                f"dtype={out.dtype}, size={out.size}, expected bool with size={x.shape[0]}"
            )
        return torch.from_numpy(out).to(device=x.device)

    raise TypeError(
        "region_fn must return (points_subset, idx) or a boolean mask. "
        f"Got: {type(out)}"
    )


def _region_share(self, x, occ, region_fn, batch):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if not torch.is_tensor(occ):
        occ = torch.as_tensor(occ)

    x = x.to(self.device)
    occ = occ.to(self.device).view(-1)

    
    if occ.dtype != torch.bool:
        occ = occ > 0.5

    R = _region_mask_from_fn(x, region_fn)  

    denom = float(R.sum().item())
    if denom <= 0.0:
        return 0.0

    num = float((occ & R).sum().item())
    return num / denom


def _boundary_share(self, bounds: np.ndarray, model_kind: str, n: int) -> float:
    pts = _sample_domain_boundary_points_2d(bounds, n, device=getattr(self, "device", torch.device("cpu")))
    if pts.numel() == 0:
        return 0.0

    if model_kind == "density":
        # Apply enforce_density to boundary points as well (same API)
        rho = _unwrap_model_output(self.density_GINN_model(pts), "density").view(-1)
        rho, _ = self.enforce_density.apply(
            pts, rho, None, [int(getattr(self, "eval_grid_nx", 256)), int(getattr(self, "eval_grid_ny", 128))],
            self.test_case.domain
        )
        rho = rho.clamp(0.0, 1.0)
        thr = float(getattr(self, "plot_threshold", 0.5))
        occ = (rho >= thr)
    else:
        sdf = _unwrap_model_output(self.SDF_GINN_model(pts), "sdf").view(-1)
        occ = (sdf <= 0.0)

    return float(occ.float().mean().item())


def _sample_interface_points(self, n: int, device) -> torch.Tensor:
    pts = self.test_case.interfaces.sample_points_from_all_interfaces(
        int(n),
        random_seed=None,
        output_type="torch_tensor"
    )
    if not torch.is_tensor(pts):
        pts = torch.tensor(pts, dtype=torch.float32, device=device)
    return pts.to(device=device, dtype=torch.float32).view(-1, 2)


def _sample_boundary_points(self, model_kind: str, max_pts: int) -> torch.Tensor:
  
    device = getattr(self, "device", torch.device("cpu"))

    if model_kind != "density":
        sampler = getattr(self, "boundary_sampler", None)
        if sampler is None:
            raise RuntimeError("self.boundary_sampler is required for sdf cd1 but is missing.")
        pts, _ = sampler.get_surface_pts()

        if pts is None or (torch.is_tensor(pts) and pts.numel() == 0):
            return torch.empty((0, 2), device=device, dtype=torch.float32)

        pts = pts.to(device=device, dtype=torch.float32).view(-1, 2)
        if pts.shape[0] > max_pts:
            idx = torch.randperm(pts.shape[0], device=device)[:max_pts]
            pts = pts[idx]
        return pts

    sampler = getattr(self, "boundary_sampler_rho", None)
    pts = None

    if sampler is not None:
        old_level = float(getattr(sampler, "level_set", float(getattr(self, "plot_threshold", 0.5))))
        try:
            sampler.level_set = float(getattr(self, "plot_threshold", 0.5))
            pts, _ = sampler.get_surface_pts()
        finally:
            sampler.level_set = old_level

        if pts is not None and torch.is_tensor(pts) and pts.numel() > 0:
            pts = pts.to(device=device, dtype=torch.float32).view(-1, 2)
            if pts.shape[0] > max_pts:
                idx = torch.randperm(pts.shape[0], device=device)[:max_pts]
                pts = pts[idx]
            return pts

    # Density fallback
    nx = int(getattr(self, "eval_grid_nx", 256))
    ny = int(getattr(self, "eval_grid_ny", 128))
    batch = int(getattr(self, "eval_batch", 200_000))

    bounds = _parse_bounds_2d(self.test_case.domain)  
    xg = _grid_centers_2d(bounds, nx, ny, device=device)

    rho = _eval_density(self, xg, n_samples=(nx, ny), batch=batch)  
    thr = float(getattr(self, "plot_threshold", 0.5))
    occ = (rho >= thr).view(ny, nx).detach().cpu().numpy().astype(bool)  

    bmask = _boundary_mask_4n(occ)
    idx = np.argwhere(bmask) 

    if idx.shape[0] == 0:
        return torch.empty((0, 2), device=device, dtype=torch.float32)

    if idx.shape[0] > int(max_pts):
        pick = np.random.choice(idx.shape[0], int(max_pts), replace=False)
        idx = idx[pick]

    xmin, xmax = float(bounds[0, 0]), float(bounds[0, 1])
    ymin, ymax = float(bounds[1, 0]), float(bounds[1, 1])
    dx = (xmax - xmin) / float(nx)
    dy = (ymax - ymin) / float(ny)

    xs = np.linspace(xmin + 0.5 * dx, xmax - 0.5 * dx, nx, dtype=np.float32)
    ys = np.linspace(ymin + 0.5 * dy, ymax - 0.5 * dy, ny, dtype=np.float32)

    P_np = np.stack([xs[idx[:, 1]], ys[idx[:, 0]]], axis=1).astype(np.float32)
    return torch.tensor(P_np, device=device, dtype=torch.float32)



def _cd1(interface_pts: torch.Tensor, boundary_pts: torch.Tensor, cdist_chunk: int) -> float:
    """
    one-sided Chamfer sqrt: sqrt(mean_i min_j ||pi - qj||^2)
    """
    if interface_pts is None or boundary_pts is None:
        return float("inf")
    if interface_pts.numel() == 0 or boundary_pts.numel() == 0:
        return float("inf")

    interface_pts = interface_pts.float()
    boundary_pts = boundary_pts.float()

    with torch.no_grad():
        mins2 = []
        for i0 in range(0, interface_pts.shape[0], cdist_chunk):
            p = interface_pts[i0:i0 + cdist_chunk]
            d = torch.cdist(p, boundary_pts)  # (m,K)
            m = d.min(dim=1).values
            mins2.append(m * m)
        d2 = torch.cat(mins2, dim=0)
        return float(torch.sqrt(d2.mean()).item())
    

def _boundary_mask_4n(mask2d: np.ndarray) -> np.ndarray:
    p = np.pad(mask2d.astype(bool), 1, mode="constant", constant_values=False)
    c  = p[1:-1, 1:-1]
    up = p[:-2,  1:-1]
    dn = p[2:,   1:-1]
    lf = p[1:-1, :-2]
    rt = p[1:-1, 2:]
    return c & (~up | ~dn | ~lf | ~rt)


def _dc_share(mask2d: np.ndarray, E2d: np.ndarray) -> float:
    denom = float(np.count_nonzero(E2d))
    if denom <= 0.0:
        return 0.0
    n_comp, sizes = _count_components_2d(mask2d)
    if n_comp <= 1:
        return 0.0
    total = float(np.sum(sizes))
    largest = float(np.max(sizes))
    dc = max(0.0, total - largest)
    return dc / denom


def _count_components_2d(mask: np.ndarray):
    
    if mask.size == 0:
        return 0, []

    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    sizes = []
    n_comp = 0
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(h):
        for x in range(w):
            if (not mask[y, x]) or visited[y, x]:
                continue
            n_comp += 1
            stack = [(y, x)]
            visited[y, x] = True
            sz = 0
            while stack:
                yy, xx = stack.pop()
                sz += 1
                for dy, dx in nbrs:
                    yn, xn = yy + dy, xx + dx
                    if 0 <= yn < h and 0 <= xn < w and mask[yn, xn] and (not visited[yn, xn]):
                        visited[yn, xn] = True
                        stack.append((yn, xn))
            sizes.append(sz)

    return n_comp, sizes


def log_eval_metrics_csv(self, phase: str, it: int, metrics: dict):
    fields = [
        "phase", "iter",
        "grid_nx", "grid_ny",
        "vol_frac",
        "vol_outside_E_share",       
        "boundary_share",           
        "prohibited_share",        
        "thickness_share",          
        "b0",
        "b0_in_E",
        "dc_share",                 
        "dc_in_E_share",            
        "cd1",                      
    ]
    row = {
        "phase": str(phase),
        "iter": int(it),
        "grid_nx": int(metrics.get("grid_nx", -1)),
        "grid_ny": int(metrics.get("grid_ny", -1)),
        "vol_frac": float(metrics.get("vol_frac", 0.0)),
        "vol_outside_E_share": float(metrics.get("vol_outside_E_share", 0.0)),
        "boundary_share": float(metrics.get("boundary_share", 0.0)),
        "prohibited_share": float(metrics.get("prohibited_share", 0.0)),
        "thickness_share": float(metrics.get("thickness_share", 0.0)),
        "b0": float(metrics.get("b0", 0.0)),
        "b0_in_E": float(metrics.get("b0_in_E", 0.0)),
        "dc_share": float(metrics.get("dc_share", 0.0)),
        "dc_in_E_share": float(metrics.get("dc_in_E_share", 0.0)),
        "cd1": float(metrics.get("cd1", float("inf"))),
    }
    append_csv_row(self.csv_eval_metrics_path, fields, row) 


def maybe_log_eval_metrics(self, phase: str, it: int, model_kind: str):
    if not getattr(self, "eval_metrics_enable", True):
        return
    try:
        m = self._compute_eval_metrics_2d(phase=phase, model_kind=model_kind)
        log_eval_metrics_csv(self,phase=phase, it=it, metrics=m) 
    except Exception as e:
        print(f"[WARN] eval metrics failed at phase={phase}, it={it}: {type(e).__name__}: {e}")


