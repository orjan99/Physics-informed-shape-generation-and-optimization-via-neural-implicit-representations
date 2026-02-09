import os
import csv


def append_csv_row(path, fieldnames, row_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row_dict)  

def log_eval_metrics_csv(self, phase: str, it: int, metrics: dict):
        fields = [
            "phase", "iter",
            "grid_n",
            "vol_frac",
            "vol_outside_env_share",
            "boundary_share",
            "prohibited_share",
            "thickness_share",
            "b0",
            "b0_in_env",
            "dc_share",
            "dc_in_env_share",
            "cd1",
        ] 
        row = {
            "phase": str(phase),
            "iter": int(it),
            "grid_n": int(metrics.get("grid_n", -1)),
            "vol_frac": float(metrics.get("vol_frac", 0.0)),
            "vol_outside_env_share": float(metrics.get("vol_outside_env_share", 0.0)),
            "boundary_share": float(metrics.get("boundary_share", 0.0)),
            "prohibited_share": float(metrics.get("prohibited_share", 0.0)),
            "thickness_share": float(metrics.get("thickness_share", 0.0)),
            "b0": float(metrics.get("b0", 0.0)),
            "b0_in_env": float(metrics.get("b0_in_env", 0.0)), 
            "dc_share": float(metrics.get("dc_share", 0.0)),
            "dc_in_env_share": float(metrics.get("dc_in_env_share", 0.0)),
            "cd1": float(metrics.get("cd1", float("inf"))),
        }
        append_csv_row(self.csv_eval_metrics_path, fields, row)

def log_ginn_losses_csv(self, it, eik, env, connect, hole, intr, norm, thick, prohib, smooth_scaled, total):
    fields = [
        "iter",
        "eikonal", "envelope", "connectivity", "holes", "interface",
        "normals", "thickness", "prohibited", "smoothness_scaled",
        "total"
    ]
    row = {
        "iter": int(it),
        "eikonal": float(eik),
        "envelope": float(env),
        "connectivity": float(connect),
        "holes": float(hole),
        "interface": float(intr),
        "normals": float(norm),
        "thickness": float(thick),
        "prohibited": float(prohib),
        "smoothness_scaled": float(smooth_scaled),
        "total": float(total),
    }
    append_csv_row(self.csv_ginn_loss_path, fields, row)


def log_topo_density_losses_csv(self, step, topo_obj, env_rho, conn_rho, total_raw):
    fields = ["topo_step", "topo_objective", "density_envelope", "density_connectivity", "total_raw"]
    row = {
        "topo_step": int(step),
        "topo_objective": float(topo_obj),
        "density_envelope": float(env_rho),
        "density_connectivity": float(conn_rho),
        "total_raw": float(total_raw),
    }
    append_csv_row(self.csv_topo_loss_path, fields, row)


def log_optimization_metrics_csv(self, it, sigma_metric, sigma_max, current_volume, comp_now):
    vol_frac = float(current_volume) / float(self.test_case.domain_volume + 1e-30) 
    fields = [
        "iter",
        "target_vol_frac",
        "current_vol_frac",
        "current_volume",
        "sigma_metric",
        "sigma_max",
        "compliance"
    ]
    row = {
        "iter": int(it),
        "target_vol_frac": float(self.volume_ratio),
        "current_vol_frac": float(vol_frac),
        "current_volume": float(current_volume),
        "sigma_metric": float(sigma_metric),
        "sigma_max": float(sigma_max),
        "compliance": float(comp_now),
    }
    append_csv_row(self.csv_opt_metrics_path, fields, row)


def log_optimization_losses_csv(self, it, comp_now, sigma_metric, sigma_max, current_volume): 
    fields = ["iter", "compliance", "sigma_metric", "sigma_max", "current_volume", "target_vol_frac"]
    row = {
        "iter": int(it),
        "compliance": float(comp_now),
        "sigma_metric": float(sigma_metric),
        "sigma_max": float(sigma_max),
        "current_volume": float(current_volume),
        "target_vol_frac": float(self.volume_ratio),
    }
    append_csv_row(self.csv_opt_loss_path, fields, row)


def maybe_log_eval_metrics(self, phase: str, it: int, model_kind: str):
        if not getattr(self, "eval_metrics_enable", True):
            return
        try:
            m = self._compute_eval_metrics_3d(phase=phase, model_kind=model_kind)
            self._log_eval_metrics_csv(phase=phase, it=it, metrics=m)
        except Exception as e:
            print(f"[WARN] eval metrics failed at phase={phase}, it={it}: {type(e).__name__}: {e}")