import os 
import torch 
import numpy as np
import csv 


def append_csv_row(path, fieldnames, row_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row_dict)

def log_timing(self, tag, seconds, extra=None):
    fields = ["tag", "seconds"]
    row = {"tag": str(tag), "seconds": float(seconds)}
    if extra:
        for k, v in extra.items():
            if k not in fields:
                fields.append(k)
            row[k] = v
    append_csv_row(self.csv_time_path, fields, row)

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


def record_density_update_time(self, dt):
 
    self.time_density_updates += float(dt)
    self._density_update_calls += 1
    self._density_update_time_buffer.append(float(dt))

    if self._density_update_calls % 50 == 0:
        last50 = self._density_update_time_buffer[-50:]
        block_total = float(np.sum(last50))
        block_avg = block_total / 50.0

        self._log_timing(
            tag="density_update_block_50",
            seconds=block_total,
            extra={
                "topo_step": self._density_update_calls,
                "avg_seconds_per_update": block_avg
            }
        )

def save_models(self, tag):
    ckpt = {
        "tag": tag,
        "u_model": self.u_model.state_dict(),
        "v_model": self.v_model.state_dict(),
        "density_GINN_model": self.density_GINN_model.state_dict(),
        "SDF_GINN_model": self.SDF_GINN_model.state_dict(),
        "volume_ratio": float(getattr(self, "volume_ratio", 0.0)),
        "training_hparams": self.training_hparams,
        "topo_hparams": self.topo_hparams,
        "GINN_hparams": self.GINN_hparams,
        "loss_weight_hparams": self.loss_weight_hparams,
        "seed": self.seed,
    }
    out = os.path.join(self.save_path, f"checkpoint-{str(tag)}.pt")
    torch.save(ckpt, out)