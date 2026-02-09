import numpy as np
import torch
import cripser #type: ignore  


class PH:
    """PH manager derived from: "https://github.com/ml-jku/GINNs-Geometry-informed-Neural-Networks/tree/main/GINN/ph" """
    def __init__(self,
                 *,
                 nx: int,
                 bounds,
                 model: torch.nn.Module,
                 n_grid_points: int,
                 iso_level: float,
                 target_betti,
                 maxdim: int,
                 is_density: bool,
                 inside_envelope_fn,          
                 group_size_fwd_no_grad: int = -1,
                 add_frame: bool = False,
                 hole_level: float = 0.06,
                 test_case=None):             
        super().__init__()
        self.nx = int(nx)
        self.model = model
        self.ISO = float(iso_level)           
        self.TARGET = list(target_betti)      # e.g. [1,0,0]
        self.MAXDIM = int(maxdim)             # usually 1
        self.is_density = bool(is_density)    # False for SDF
        self.add_frame = bool(add_frame)
        self.hole_level = float(hole_level)
        self.test_case = test_case            

        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        b = torch.as_tensor(bounds, dtype=torch.float32, device=self.device).flatten()
        assert b.numel() == 2 * self.nx
        self.bounds = b.view(self.nx, 2)

        # grid in bounds
        n = int(n_grid_points)
        axes = [torch.linspace(self.bounds[i, 0], self.bounds[i, 1], n, device=self.device)
                for i in range(self.nx)]
        mg   = torch.meshgrid(*axes, indexing='ij')
        xs   = torch.stack([g.reshape(-1) for g in mg], dim=1)  # [N, nx]
        self.n_per_axis = n
        self.grid_shape = (n,) * self.nx
        self.xs_flat = xs
        self.xs      = torch.stack([g for g in mg], dim=-1)     # [n,...,n, nx]

        # envelope-only mask 
        m = inside_envelope_fn(self.xs_flat)
        if isinstance(m, np.ndarray):
            m = torch.from_numpy(m).to(self.device)
        self.mask_in_flat = m.to(dtype=torch.bool).view(-1)
        self.mask_in_grid = self.mask_in_flat.view(*self.grid_shape)
        self.xs_in        = self.xs[self.mask_in_grid]
        self.Y_inf = torch.full(self.grid_shape, float('inf'), device=self.device)

        self.group_size = int(group_size_fwd_no_grad) if group_size_fwd_no_grad is not None else -1
        self._cache = {"fp": None, "PH": None, "Y": None}
        self.iso_eff = self.ISO

    # ---------- cache ----------
    def invalidate_cache(self):
        self._cache = {"fp": None, "PH": None, "Y": None}

    def _fingerprint(self):
        s = 0.0
        n = 0
        for p in self.model.parameters():
            if p.requires_grad and p.numel() > 0:
                s += float(p.detach().sum().item())
                n += p.numel()
        return (n, round(s, 6))


    def _filter_excluded_points(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Remove points that lie in the prohibited region,
        so that holes there do NOT contribute to the holes loss.
        """
        if self.test_case is None or pts.numel() == 0:
            return pts


        inside_points, _ = self.test_case.interfaces.is_inside_prohibited_region(pts)

        # If nothing is inside the prohibited region, keep everything
        if inside_points is None or inside_points.numel() == 0:
            return pts

        inside_points = inside_points.to(pts.device)
        N, D = pts.shape
        M = inside_points.shape[0]
        diff = pts.unsqueeze(1) - inside_points.unsqueeze(0)  # [N, M, D]
        max_abs = diff.abs().max(dim=-1).values               # [N, M]
        inside_any = (max_abs < 1e-6).any(dim=1)              # [N]
        keep_mask = ~inside_any

        if keep_mask.sum() == 0:
            # no points remain -> return empty tensor with correct shape 
            return pts.new_zeros((0, D), dtype=pts.dtype)

        return pts[keep_mask]


    # PH compute 
    def get_PH(self):
        fp = self._fingerprint()
        if self._cache["fp"] == fp and self._cache["PH"] is not None:
            return self._cache["PH"], self._cache["Y"]

        xs_in_flat = self.xs_in.reshape(-1, self.nx)

        with torch.no_grad():
            if self.group_size is None or self.group_size < 0 or xs_in_flat.shape[0] <= self.group_size:
                out = self.model(xs_in_flat)
                if isinstance(out, (tuple, list)):
                    Y_in = out[1].reshape(-1)   # SDF
                else:
                    Y_in = out.reshape(-1)
            else:
                outs = []
                for i in range(0, xs_in_flat.shape[0], self.group_size):
                    out = self.model(xs_in_flat[i:i + self.group_size])
                    if isinstance(out, (tuple, list)):
                        y = out[1]              # SDF
                    else:
                        y = out
                    outs.append(y.reshape(-1))
                Y_in = torch.cat(outs, dim=0)

        Y_full = self.Y_inf.clone()
        Y_full.view(-1)[self.mask_in_flat] = Y_in

        if self.add_frame:
            if self.nx == 2:
                Y_full[0, :]  = float('inf'); Y_full[-1, :] = float('inf')
                Y_full[:, 0]  = float('inf'); Y_full[:, -1] = float('inf')
            elif self.nx == 3:
                Y_full[0, :, :]  = float('inf'); Y_full[-1, :, :] = float('inf')
                Y_full[:, 0, :]  = float('inf'); Y_full[:, -1, :] = float('inf')
                Y_full[:, :, 0]  = float('inf'); Y_full[:, :, -1] = float('inf')

        PH = cripser.computePH(Y_full.detach().cpu().numpy(), maxdim=self.MAXDIM)
        self._cache = {"fp": fp, "PH": PH, "Y": Y_full}
        return PH, Y_full

    # ---------- helpers ----------
    def _gather_points_from_indices(self, idx_numpy: np.ndarray) -> torch.Tensor:
        idx_t = torch.from_numpy(
            np.clip(idx_numpy, 0, self.n_per_axis - 1)
        ).to(self.device, dtype=torch.long)

        if self.nx == 1:
            pts = self.xs[idx_t[:, 0]]
        elif self.nx == 2:
            pts = self.xs[idx_t[:, 0], idx_t[:, 1]]
        elif self.nx == 3:
            pts = self.xs[idx_t[:, 0], idx_t[:, 1], idx_t[:, 2]]
        else:
            slices = [idx_t[:, d] for d in range(self.nx)]
            pts = self.xs[slices]
        return pts.reshape(-1, self.nx).to(self.device)

    def connectedness_loss(self) -> torch.Tensor:
        PH, _ = self.get_PH()
        device = self.device

        PH0 = PH[PH[:, 0] == 0]
        if PH0.size == 0:
            return torch.tensor(0.0, device=device)

        lengths = PH0[:, 2] - PH0[:, 1]
        order = np.argsort(lengths)[::-1]
        PH0 = PH0[order]

        sel = (PH0[:, 1] < -self.iso_eff) & (PH0[:, 2] > self.iso_eff)
        PH_sel = PH0[sel]
        if PH_sel.size == 0:
            return torch.tensor(0.0, device=device)

        tgt = self.TARGET[0] if len(self.TARGET) > 0 else 1
        deaths_idx = PH_sel[:, 6:6 + self.nx].astype(int)
        deaths_idx = deaths_idx[max(0, tgt):]
        if deaths_idx.size == 0:
            return torch.tensor(0.0, device=device)

        x_in = self._gather_points_from_indices(deaths_idx)
        out = self.model(x_in)
        if isinstance(out, (tuple, list)):
            y = out[1]          # SDF
        else:
            y = out
        y = y.reshape(-1)
        return torch.clamp(self.ISO - y, max=0.).pow(2).sum()



    def connectedness_loss_from_density(self,
                                        density_model: torch.nn.Module,
                                        iso_level: float = 0.5) -> torch.Tensor:
        """
        Density-based analogue of connectedness_loss.
        """

        device = self.device

        # Build PH on g(x) = iso_level - ρ(x)  
        xs_in_flat = self.xs_in.reshape(-1, self.nx)  # [N_in, nx]

        with torch.no_grad():
            out = density_model(xs_in_flat)
            if isinstance(out, (tuple, list)):
                rho_in = out[0]
            else:
                rho_in = out
            rho_in = rho_in.reshape(-1)
            g_in = iso_level - rho_in

        Y_full = self.Y_inf.clone()
        Y_full.view(-1)[self.mask_in_flat] = g_in

        if self.add_frame:
            if self.nx == 2:
                Y_full[0, :]  = float('inf'); Y_full[-1, :] = float('inf')
                Y_full[:, 0]  = float('inf'); Y_full[:, -1] = float('inf')
            elif self.nx == 3:
                Y_full[0, :, :]  = float('inf'); Y_full[-1, :, :] = float('inf')
                Y_full[:, 0, :]  = float('inf'); Y_full[:, -1, :] = float('inf')
                Y_full[:, :, 0]  = float('inf'); Y_full[:, :, -1] = float('inf')

        PH = cripser.computePH(Y_full.detach().cpu().numpy(), maxdim=self.MAXDIM)

        PH0 = PH[PH[:, 0] == 0]
        if PH0.size == 0:
            return torch.tensor(0.0, device=device)

        lengths = PH0[:, 2] - PH0[:, 1]
        order = np.argsort(lengths)[::-1]
        PH0 = PH0[order]
        iso_eff = 0.0
        sel = (PH0[:, 1] < -iso_eff) & (PH0[:, 2] > iso_eff)
        PH_sel = PH0[sel]
        if PH_sel.size == 0:
            return torch.tensor(0.0, device=device)

        tgt = self.TARGET[0] if len(self.TARGET) > 0 else 1
        deaths_idx = PH_sel[:, 6:6 + self.nx].astype(int)
        deaths_idx = deaths_idx[max(0, tgt):]  # skip the largest component
        if deaths_idx.size == 0:
            return torch.tensor(0.0, device=device)

        x_in = self._gather_points_from_indices(deaths_idx)  # [M, nx]
        out = density_model(x_in)
        if isinstance(out, (tuple, list)):
            rho = out[0]
        else:
            rho = out
        rho = rho.reshape(-1)
        g_vals = iso_level - rho
        return torch.clamp(-g_vals, max=0.).pow(2).sum()


    # ---------- holes (with prohibited-region filtering) ----------
    def holes_loss(self) -> torch.Tensor:
        if self.MAXDIM < 1:
            return torch.tensor(0.0, device=self.device)

        PH, _ = self.get_PH()
        device = self.device

        PH1 = PH[PH[:, 0] == 1]
        if PH1.size == 0:
            return torch.tensor(0.0, device=device)

        lengths = PH1[:, 2] - PH1[:, 1]
        order = np.argsort(lengths)[::-1]
        PH1 = PH1[order]

        # finite-death only
        fin_mask = np.isfinite(PH1[:, 2])
        PH1_fin = PH1[fin_mask]
        if PH1_fin.size == 0:
            return torch.tensor(0.0, device=device)

        loss = torch.tensor(0.0, device=device)

        keep_k = int(self.TARGET[1]) if len(self.TARGET) > 1 else 0
        if keep_k > 0:
            births_idx_top = PH1_fin[:keep_k, 3:3 + self.nx].astype(int)
            deaths_idx_top = PH1_fin[:keep_k, 6:6 + self.nx].astype(int)

            if births_idx_top.size > 0:
                x_b_full = self._gather_points_from_indices(births_idx_top)
                x_b = self._filter_excluded_points(x_b_full)
                if x_b.numel() > 0:
                    out_b = self.model(x_b)
                    if isinstance(out_b, (tuple, list)):
                        y_b = out_b[1]
                    else:
                        y_b = out_b
                    y_b = y_b.reshape(-1)
                    loss = loss + (-self.hole_level - y_b).pow(2).sum()

            if deaths_idx_top.size > 0:
                x_d_full = self._gather_points_from_indices(deaths_idx_top)
                x_d = self._filter_excluded_points(x_d_full)
                if x_d.numel() > 0:
                    out_d = self.model(x_d)
                    if isinstance(out_d, (tuple, list)):
                        y_d = out_d[1]
                    else:
                        y_d = out_d
                    y_d = y_d.reshape(-1)
                    loss = loss + (self.hole_level - y_d).pow(2).sum()

        rest = PH1_fin[keep_k:]
        if rest.size > 0:
            sel = (rest[:, 1] < -self.iso_eff) & (rest[:, 2] > self.iso_eff)
            rest_sel = rest[sel]
            if rest_sel.size > 0:
                deaths_idx = rest_sel[:, 6:6 + self.nx].astype(int)
                x_full = self._gather_points_from_indices(deaths_idx)
                x_in = self._filter_excluded_points(x_full)
                if x_in.numel() > 0:
                    out = self.model(x_in)
                    if isinstance(out, (tuple, list)):
                        y = out[1]
                    else:
                        y = out
                    y = y.reshape(-1)
                    loss = loss + torch.clamp(self.ISO - y, max=0.).pow(2).sum()

        return loss

    def holes_loss_from_density(self,
                                density_model: torch.nn.Module,
                                iso_level: float = 0.5) -> torch.Tensor:
        """
        Density-based analogue of holes_loss.
        """
        device = self.device

        if self.MAXDIM < 1:
            return torch.tensor(0.0, device=device)
        xs_in_flat = self.xs_in.reshape(-1, self.nx)  # [N_in, nx]

        with torch.no_grad():
            out = density_model(xs_in_flat)
            if isinstance(out, (tuple, list)):
                rho_in = out[0]
            else:
                rho_in = out
            rho_in = rho_in.reshape(-1)

            g_in = iso_level - rho_in  # level set g = 0 <-> rho = iso_level

        Y_full = self.Y_inf.clone()
        Y_full.view(-1)[self.mask_in_flat] = g_in

        if self.add_frame:
            if self.nx == 2:
                Y_full[0, :]  = float('inf'); Y_full[-1, :] = float('inf')
                Y_full[:, 0]  = float('inf'); Y_full[:, -1] = float('inf')
            elif self.nx == 3:
                Y_full[0, :, :]  = float('inf'); Y_full[-1, :, :] = float('inf')
                Y_full[:, 0, :]  = float('inf'); Y_full[:, -1, :] = float('inf')
                Y_full[:, :, 0]  = float('inf'); Y_full[:, :, -1] = float('inf')

        PH = cripser.computePH(Y_full.detach().cpu().numpy(), maxdim=self.MAXDIM)

        PH1 = PH[PH[:, 0] == 1]
        if PH1.size == 0:
            return torch.tensor(0.0, device=device)

        lengths = PH1[:, 2] - PH1[:, 1]
        order = np.argsort(lengths)[::-1]
        PH1 = PH1[order]

        # finite-death only
        fin_mask = np.isfinite(PH1[:, 2])
        PH1_fin = PH1[fin_mask]
        if PH1_fin.size == 0:
            return torch.tensor(0.0, device=device)

        loss = torch.tensor(0.0, device=device)
        keep_k = int(self.TARGET[1]) if len(self.TARGET) > 1 else 0
        if keep_k > 0:
            births_idx_top = PH1_fin[:keep_k, 3:3 + self.nx].astype(int)
            deaths_idx_top = PH1_fin[:keep_k, 6:6 + self.nx].astype(int)
            if births_idx_top.size > 0:
                x_b_full = self._gather_points_from_indices(births_idx_top)
                x_b = self._filter_excluded_points(x_b_full)
                if x_b.numel() > 0:
                    rho_b = density_model(x_b)
                    if isinstance(rho_b, (tuple, list)):
                        rho_b = rho_b[0]
                    rho_b = rho_b.reshape(-1)
                    rho_void_target = 0.0
                    loss = loss + (rho_b - rho_void_target).pow(2).sum()
            if deaths_idx_top.size > 0:
                x_d_full = self._gather_points_from_indices(deaths_idx_top)
                x_d = self._filter_excluded_points(x_d_full)
                if x_d.numel() > 0:
                    rho_d = density_model(x_d)
                    if isinstance(rho_d, (tuple, list)):
                        rho_d = rho_d[0]
                    rho_d = rho_d.reshape(-1)
                    rho_solid_target = 1.0
                    loss = loss + (rho_solid_target - rho_d).pow(2).sum()

        rest = PH1_fin[keep_k:]
        if rest.size > 0:
            iso_eff = 0.0  # on g
            sel = (rest[:, 1] < -iso_eff) & (rest[:, 2] > iso_eff)
            rest_sel = rest[sel]
            if rest_sel.size > 0:
                deaths_idx = rest_sel[:, 6:6 + self.nx].astype(int)
                x_full = self._gather_points_from_indices(deaths_idx)
                x_in = self._filter_excluded_points(x_full)
                if x_in.numel() > 0:
                    rho = density_model(x_in)
                    if isinstance(rho, (tuple, list)):
                        rho = rho[0]
                    rho = rho.reshape(-1)
                    g_vals = iso_level - rho
                    loss = loss + torch.clamp(-g_vals, max=0.).pow(2).sum()

        return loss


def make_inside_envelope_fn_from_domain(domain, nx: int, device=None):
    """
    domain: [x_min, x_max, y_min, y_max, z_min, z_max] (for nx=3) or analogous
    Returns a function that maps (N, nx) points (np.ndarray or torch.Tensor)
    to a torch.BoolTensor mask on the target device.
    """
 
    bounds = np.array(domain, dtype=np.float32).reshape(nx, 2)

    def _fn(x):
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
            if device is not None:
                t = t.to(device)
        elif isinstance(x, torch.Tensor):
            t = x if (device is None or x.device == device) else x.to(device)
        else:
            raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")

        ok = torch.ones(t.shape[0], dtype=torch.bool, device=t.device)
        for d in range(nx):
            lo, hi = bounds[d]
            ok &= (t[:, d] >= lo) & (t[:, d] <= hi)
        return ok

    return _fn