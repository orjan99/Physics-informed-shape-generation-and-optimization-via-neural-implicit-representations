import torch 
import math 
import torch.nn.functional as F 

class Boundary_Sampler:
    """
    Boundary sampler for smoothness, from: https://github.com/ml-jku/GINNs-Geometry-informed-Neural-Networks 
    """

    def __init__(
        self,
        *,
        dim: int,                         # 2D or 3D
        bounds,                           # [xmin,xmax,ymin,ymax,(zmin,zmax)]
        model: torch.nn.Module,           
        x_interface: torch.Tensor,        
        n_points_surface: int,
        equidistant_init_grid: bool = True,
        interface_cutoff: float = 0.0,
        level_set: float = 0.0,
        do_uniform_resampling: bool = True,

        # flow params
        surf_pts_lr: float = 0.01,
        surf_pts_n_iter: int = 10,
        surf_pts_prec_eps: float = 1.0e-3,
        surf_pts_converged_interval: int = 1,
        surf_pts_use_newton: bool = True,
        surf_pts_newton_clip: float = 0.15,
        surf_pts_inflate_bounds_amount: float = 0.05,

        # uniformization params
        surf_pts_uniform_n_iter: int = 10,
        surf_pts_uniform_nof_neighbours: int = 16,
        surf_pts_uniform_stepsize: float = 0.75,
        surf_pts_uniform_n_iter_reproj: int = 5,
        surf_pts_uniform_prec_eps: float = 1.0e-3,
        surf_pts_uniform_min_count: int = 1000,
    ):
      
        self.nx = int(dim)
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32

        #  bounds: accept flat domain and reshape to [nx,2] on model device 
        b = torch.as_tensor(bounds, dtype=self.dtype, device=self.device).flatten()
        if b.numel() != 2 * self.nx:
            raise ValueError(f"bounds must have {2*self.nx} values (flat [xmin,xmax,...]), got {b.numel()}")
        self.bounds = torch.stack([b[2*i:2*i+2] for i in range(self.nx)], dim=0)  # [nx,2]

        # --- model / interface points ---
        self.model = model
        if x_interface is not None and x_interface.numel() > 0:
            self.x_interface = x_interface.to(self.device, dtype=self.dtype)
        else:
            self.x_interface = torch.empty(0, self.nx, device=self.device, dtype=self.dtype)

        # --- config ---
        self.interface_cutoff = float(interface_cutoff)
        self.level_set = float(level_set)

        self.surf_pts_nof_points = int(n_points_surface)
        self.equidistant_init_grid = bool(equidistant_init_grid)
        self.surf_pts_do_uniform_resampling = bool(do_uniform_resampling)

        self.surf_pts_lr = float(surf_pts_lr)
        self.surf_pts_n_iter = int(surf_pts_n_iter)
        self.surf_pts_prec_eps = float(surf_pts_prec_eps)
        self.surf_pts_converged_interval = int(surf_pts_converged_interval)
        self.surf_pts_use_newton = bool(surf_pts_use_newton)
        self.surf_pts_newton_clip = float(surf_pts_newton_clip)

        self.surf_pts_uniform_n_iter = int(surf_pts_uniform_n_iter)
        self.surf_pts_uniform_nof_neighbours = int(surf_pts_uniform_nof_neighbours)
        self.surf_pts_uniform_stepsize = float(surf_pts_uniform_stepsize)
        self.surf_pts_uniform_n_iter_reproj = int(surf_pts_uniform_n_iter_reproj)
        self.surf_pts_uniform_prec_eps = float(surf_pts_uniform_prec_eps)
        self.surf_pts_uniform_min_count = int(surf_pts_uniform_min_count)
        self.knn_k = self.surf_pts_uniform_nof_neighbours

        self._inflate_bounds(surf_pts_inflate_bounds_amount)
        self._precompute_sample_grid()

    # freeze model params during sampling 
    def _no_param_grad(self):
        class _Guard:
            def __init__(self, model):
                self.model = model
                self.prev = [p.requires_grad for p in model.parameters()]
            def __enter__(self):
                for p in self.model.parameters():
                    p.requires_grad_(False)
            def __exit__(self, exc_type, exc, tb):
                for p, r in zip(self.model.parameters(), self.prev):
                    p.requires_grad_(r)
        return _Guard(self.model)

    # ---------- setup ----------
    def _inflate_bounds(self, amount: float = 0.10) -> None:
        lengths = self.bounds[:, 1] - self.bounds[:, 0]
        self.bounds[:, 0] -= lengths * amount
        self.bounds[:, 1] += lengths * amount

    @staticmethod
    def _get_is_out_mask(x: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        return (x < bounds[:, 0]).any(1) | (x > bounds[:, 1]).any(1)

    def _precompute_sample_grid(self) -> None:
        n_points = self.surf_pts_nof_points
        bounds   = self.bounds
        nx       = self.nx
        equidistant = self.equidistant_init_grid

        bound_widths = bounds[:, 1] - bounds[:, 0]                 # [nx]
        prod_bw = bound_widths.prod().item()

        if equidistant:
            n_points_stem = int(math.pow(n_points / max(prod_bw, 1e-12), 1 / nx))
            resolution = (n_points_stem * bound_widths).to(torch.int)
        else:
            n_points_root = int(n_points ** (1 / nx))
            resolution = torch.tensor([n_points_root] * nx, device=bounds.device, dtype=torch.int)

        resolution = torch.clamp(resolution, min=1)

        if nx == 2:
            x1 = torch.linspace(bounds[0, 0], bounds[0, 1], int(resolution[0]), device=bounds.device, dtype=bounds.dtype)
            x2 = torch.linspace(bounds[1, 0], bounds[1, 1], int(resolution[1]), device=bounds.device, dtype=bounds.dtype)
            x1g, x2g = torch.meshgrid(x1, x2, indexing='ij')
            x_grid = torch.stack((x1g.reshape(-1), x2g.reshape(-1)), dim=1)
        elif nx == 3:
            x1 = torch.linspace(bounds[0, 0], bounds[0, 1], int(resolution[0]), device=bounds.device, dtype=bounds.dtype)
            x2 = torch.linspace(bounds[1, 0], bounds[1, 1], int(resolution[1]), device=bounds.device, dtype=bounds.dtype)
            x3 = torch.linspace(bounds[2, 0], bounds[2, 1], int(resolution[2]), device=bounds.device, dtype=bounds.dtype)
            x1g, x2g, x3g = torch.meshgrid(x1, x2, x3, indexing='ij')
            x_grid = torch.stack((x1g.reshape(-1), x2g.reshape(-1), x3g.reshape(-1)), dim=1)
        else:
            raise NotImplementedError("dim must be 2 or 3")

        self.grid_find_surface = x_grid
        self.grid_dist_find_surface = bound_widths / resolution.clamp_min(1)
        self.init_grid_resolution = resolution

    # ---------- starting points ----------
    def _grid_starting_pts(self) -> torch.Tensor:
        xg, gd = self.grid_find_surface, self.grid_dist_find_surface
        xc_offset = torch.rand((self.nx,), device=xg.device, dtype=xg.dtype) * gd
        x = xg + xc_offset.unsqueeze(0)
        x = x + torch.randn_like(x) * (gd / 3)
        return x

    # ---------- public API ----------
    def get_surface_pts(self):
        """
        Returns:
            surface_pts: Tensor [N, dim]
            weights:     Tensor [N], uniform, sums to 1
        """
        # grid init
        p_surface = self._grid_starting_pts()

        # flow to surface 
        with self._no_param_grad():
            ok, p_surface = self._flow(p_surface)
            if not ok:
                return None, None

            # optional uniformization
            if self.surf_pts_do_uniform_resampling and p_surface.shape[0] > self.surf_pts_uniform_min_count:
                ok, p_surface = self._uniformize(p_surface)
                if not ok:
                    return None, None

        # interface cutoff
        weights = torch.ones(p_surface.shape[0], device=p_surface.device, dtype=p_surface.dtype)
        if self.interface_cutoff > 0 and self.x_interface.numel() > 0:
            dists = torch.cdist(p_surface, self.x_interface, compute_mode='use_mm_for_euclid_dist')
            min_dist, _ = torch.min(dists, dim=1)
            mask = min_dist > self.interface_cutoff
            p_surface = p_surface[mask]
            if p_surface.shape[0] == 0:
                return None, None
            weights = torch.ones(p_surface.shape[0], device=p_surface.device, dtype=p_surface.dtype)

        weights = weights / weights.shape[0]
        return p_surface, weights

    # core flow (Newton)
    def _flow(self, p_init: torch.Tensor):
        if self.surf_pts_use_newton:
            p = p_init.clone()
            y_in = None
            for i in range(self.surf_pts_n_iter):
                out_mask = self._get_is_out_mask(p, self.bounds)
                p = p[~out_mask]
                if p.shape[0] == 0:
                    return False, None

                # grads only w.r.t. points 
                p_req = p.detach().requires_grad_(True)
                y = self._f(p_req).squeeze(1)  # [K]
                grad = torch.autograd.grad(
                    y, p_req,
                    grad_outputs=torch.ones_like(y),
                    create_graph=False, retain_graph=False, only_inputs=True
                )[0]
                delta  = torch.clamp(y - self.level_set, -self.surf_pts_newton_clip, self.surf_pts_newton_clip) / (grad.norm(dim=1) + 1e-12)
                update = grad * delta[:, None]
                with torch.no_grad():
                    p = p_req - update
                y_in = y.detach()

                if i % self.surf_pts_converged_interval == 0:
                    if (torch.abs(y_in - self.level_set) < self.surf_pts_prec_eps).all():
                        break

            converged_mask = torch.abs(y_in - self.level_set) < 1e-3
            p = p[converged_mask]
            if p.shape[0] < 100:
                return False, None
            return True, p

        # ---- fallback: small GD projection  ----
        p = p_init.clone()
        y_in = None
        for i in range(self.surf_pts_n_iter):
            out_mask = self._get_is_out_mask(p, self.bounds)
            p = p[~out_mask]
            if p.shape[0] == 0:
                return False, None

            p_req = p.detach().requires_grad_(True)
            y = self._f(p_req).squeeze(1)
            loss = (y - self.level_set).square().mean()
            if torch.isnan(loss):
                return False, None

            g, = torch.autograd.grad(loss, p_req, create_graph=False, retain_graph=False, only_inputs=True)
            with torch.no_grad():
                p = p_req - self.surf_pts_lr * g

            y_in = y.detach()
            if i % self.surf_pts_converged_interval == 0:
                if (torch.abs(y_in - self.level_set) < self.surf_pts_prec_eps).all():
                    break

        converged_mask = torch.abs(y_in - self.level_set) < 1e-3
        p = p.detach()[converged_mask]
        if p.shape[0] < 100:
            return False, None
        return True, p

    # ---------- uniformization (repel + reprojection) ----------
    def _uniformize(self, pts: torch.Tensor, num_iters=None):
        if num_iters is None:
            num_iters = self.surf_pts_uniform_n_iter
        p = pts.clone()

        for _ in range(num_iters):
            normals = self.get_normals_nograd(p)       # [N, nx]
            if p.shape[0] == 0:
                return False, None

            num_points = p.shape[0]
            diag = (p.max(dim=0).values - p.min(dim=0).values).norm().item()
            if diag < 1e-6:
                return False, None
            inv_sigma = num_points / diag

            knn_idx = self._get_nn_idcs(p, self.knn_k)       # [N, k]
            knn_nn  = p[knn_idx]                              # [N, k, nx]
            knn_diff = p.unsqueeze(1) - knn_nn                # [N, k, nx]
            knn_d2   = torch.sum(knn_diff**2, dim=-1)         # [N, k]
            w = torch.exp(-knn_d2 * inv_sigma)                # [N, k]
            move = torch.sum(w[..., None] * knn_diff, dim=-2) # [N, nx]

            # project move onto tangent plane
            move -= (move * normals).sum(dim=1, keepdim=True) * normals
            move *= self.surf_pts_uniform_stepsize
            p = p + move

            # Reproject to surface with a short Newton reflow
            ok, p = self._flow_reproject(p,
                                         n_iter_reproj=self.surf_pts_uniform_n_iter_reproj,
                                         filter_thr=self.surf_pts_uniform_prec_eps)
            if not ok:
                return False, None
            if p.shape[0] < self.surf_pts_uniform_min_count:
                return False, None

        return True, p

    def _flow_reproject(self, p: torch.Tensor, n_iter_reproj: int, filter_thr: float):
        old_n_iter = self.surf_pts_n_iter
        old_prec   = self.surf_pts_prec_eps
        self.surf_pts_n_iter = n_iter_reproj
        self.surf_pts_prec_eps = filter_thr
        ok, ret = self._flow(p)
        self.surf_pts_n_iter = old_n_iter
        self.surf_pts_prec_eps = old_prec
        return ok, ret

    # ---------- normals ----------
    def get_normals_nograd(self, pts: torch.Tensor, invert: bool = False) -> torch.Tensor:
        x = pts.detach().requires_grad_(True)
        y = self._f(x).squeeze(1)                           # [N]
        grad = torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y),
            create_graph=False, retain_graph=False, only_inputs=True
        )[0]                                                # [N, nx]
        if invert:
            grad = -grad
        grad = F.normalize(grad, dim=-1)
        return grad.detach()

    # ---------- utilities ----------
    @staticmethod
    def _get_nn_idcs(x, k):
        dist = torch.cdist(x, x, compute_mode='use_mm_for_euclid_dist')
        return dist.argsort(dim=-1)[:, 1:k+1]

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        if y.ndim == 1:
            y = y.unsqueeze(1)
        return y