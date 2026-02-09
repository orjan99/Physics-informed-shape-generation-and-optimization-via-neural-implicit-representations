
import torch 
import torch.nn.functional as F 
from typing import Optional, Tuple, Callable,Union 



class Density_Constraints:
    def __init__(
        self,
        constraint_hparams: dict,
        test_case: Optional[object] = None,
        keep_fn: Optional[Callable[[torch.Tensor], Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]] = None,
        prohibit_fn: Optional[Callable[[torch.Tensor], Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]] = None,
    ):
        self.keep_fn = keep_fn          # prescribed-thickness region finder (P)
        self.prohibit_fn = prohibit_fn  # prohibited region finder (R)
        self.enabled = bool(constraint_hparams.get("enabled", True))
        self.priority = str(constraint_hparams.get("priority", "keep_over_prohibit")) 
        self.bridge = test_case

    # ---------- public API ----------
    def apply(
        self,
        sample_positions: torch.Tensor,
        rho: torch.Tensor,
        sens: Optional[torch.Tensor],
        n_cells,  # kept for API compatibility --> unused
        domain,   # kept for API compatibility --> unused
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        rho = rho.view(-1, 1)
        sens = None if sens is None else sens.view(-1, 1)

        if not self.enabled:
            return rho, sens

        keep_fn, prohibit_fn = self._resolve_fns()
        if (keep_fn is None) and (prohibit_fn is None):
            return rho, sens

        N = sample_positions.shape[0]
        device = sample_positions.device

        keep_mask = torch.zeros((N, 1), dtype=torch.bool, device=device)
        if keep_fn is not None:
            keep_mask = self._mask_from_fn(sample_positions, keep_fn)

        prohibit_mask = torch.zeros((N, 1), dtype=torch.bool, device=device)
        if prohibit_fn is not None:
            prohibit_mask = self._mask_from_fn(sample_positions, prohibit_fn)

        # Unrecognized priority => default to keep_over_prohibit 
        pr = (self.priority or "").strip().lower()
        if pr not in ("prohibit_over_keep", "keep_over_prohibit"):
            pr = "keep_over_prohibit"

        # Hard projection 
        if pr == "prohibit_over_keep":
            rho_tmp = torch.where(keep_mask, torch.ones_like(rho), rho)
            rho_out = torch.where(prohibit_mask, torch.zeros_like(rho_tmp), rho_tmp)
        else:
            rho_tmp = torch.where(prohibit_mask, torch.zeros_like(rho), rho)
            rho_out = torch.where(keep_mask, torch.ones_like(rho_tmp), rho_tmp)

        # Sensitivity masking: 
        if sens is None:
            sens_out = None
        else:
            fixed = keep_mask | prohibit_mask
            sens_out = torch.where(fixed, torch.zeros_like(sens), sens)

        return rho_out, sens_out

    # ---------- helpers ----------
    def _resolve_fns(self):
        keep_fn = self.keep_fn
        prohibit_fn = self.prohibit_fn

        # Keep the original 3D fallback behavior (JEB), for compatibility
        bridge = self.bridge if self.bridge is not None else globals().get("JEB", None)
        if bridge is not None:
            if (
                keep_fn is None
                and hasattr(bridge, "interfaces")
                and hasattr(bridge.interfaces, "is_inside_interface_thickness")
            ):
                keep_fn = lambda pts: bridge.interfaces.is_inside_interface_thickness(pts)

            if prohibit_fn is None and hasattr(bridge, "is_inside_prohibited_region"):
                prohibit_fn = lambda pts: bridge.is_inside_prohibited_region(pts)

        return keep_fn, prohibit_fn

    @staticmethod
    def _mask_from_fn(points: torch.Tensor,
        fn: Callable[[torch.Tensor], Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]],
    ) -> torch.Tensor:
        """
        Returns bool mask [N,1] 
        """
        out = fn(points)
        device = points.device
        N = points.shape[0]

        if not (isinstance(out, (tuple, list)) and len(out) == 2):
            raise TypeError("keep_fn/prohibit_fn must return (points_in_region, idx).")

        idx = out[1].to(device)
        if idx.ndim == 0:
            idx = idx.view(1)
        idx = idx.view(-1).long()

        mask = torch.zeros((N,), dtype=torch.bool, device=device)
        if idx.numel() > 0:
            valid = (idx >= 0) & (idx < N)
            if valid.any():
                mask[idx[valid]] = True

        return mask.view(N, 1)



