import numpy as np
import random
import torch 
import os 

def set_random_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_domain_volume(domain: np.ndarray) -> float:
    d = np.array(domain, dtype=np.float32)
    lower = d[0:len(d):2]
    upper = d[1:len(d):2]
    length = upper - lower
    length[length == 0.0] = 1.0
    return float(np.prod(length)) 

def save_checkpoint(
    path: str,
    epoch: int,
    u_model,
    v_model,
    w_model,
    optimizer,
    scheduler,
    best_loss: float,
    history: dict,
):
    ckpt = {
        "epoch": epoch,  
        "best_loss": best_loss,
        "u_model_state_dict": u_model.state_dict(),
        "v_model_state_dict": v_model.state_dict(),
        "w_model_state_dict": w_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "history": history,
        "rng_state": get_rng_state(),
    }
    atomic_torch_save(ckpt, path)


def load_checkpoint(
    path: str,
    u_model,
    v_model,
    w_model,
    optimizer,
    scheduler,
    map_location=None,
):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    u_model.load_state_dict(ckpt["u_model_state_dict"])
    v_model.load_state_dict(ckpt["v_model_state_dict"])
    w_model.load_state_dict(ckpt["w_model_state_dict"])

    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict", None) is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    optimizer_to(optimizer, map_location if map_location is not None else device)


    history = ckpt.get("history", {"epoch": [], "loss": [], "L2_u": [], "L2_v": [], "L2_w": [], "L2_s": []})
    best_loss = float(ckpt.get("best_loss", float("inf")))
    last_epoch = int(ckpt.get("epoch", -1))

    if "rng_state" in ckpt:
        set_rng_state(ckpt["rng_state"])

    return last_epoch, best_loss, history

def optimizer_to(optim, device):
    for state in optim.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def get_rng_state():
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    try:
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(state["torch_random_state"])
        if torch.cuda.is_available() and "torch_cuda_random_state_all" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda_random_state_all"])
    except Exception as e:
        print("[WARN] Could not fully restore RNG state:", e)


def atomic_torch_save(obj, path: str):
    """
    Atomic save: write to temp then replace.
    Prevents partially-written checkpoints if the runtime is killed.
    """
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)



class AutoClip:
    """ Auto gradient clipping from: https://github.com/ml-jku/GINNs-Geometry-informed-Neural-Networks/tree/main """
    def __init__(self,
                 grad_clipping_on: bool = True,
                 auto_clip_on: bool = True,
                 grad_clip: float = 0.5,
                 auto_clip_percentile: float = 0.9,
                 auto_clip_hist_len: int = 100,
                 auto_clip_min_len: int = 10):
        self.grad_clip_enabled   = bool(grad_clipping_on)
        self.auto_clip_enabled   = bool(auto_clip_on)
        self.default_clip_value  = float(grad_clip)
        self.percentile          = float(auto_clip_percentile)
        self.min_history_length  = int(auto_clip_min_len)

        self.history_size = 1 if not (self.grad_clip_enabled and self.auto_clip_enabled) else int(auto_clip_hist_len)
        self.gradient_norms = torch.zeros(self.history_size, dtype=torch.float32)
        self.cur_idx = 0
        self.cnt = 0

    def update_gradient_norm_history(self, gradient_norm: float):
        if not self.grad_clip_enabled:
            return
        self.gradient_norms[self.cur_idx] = float(gradient_norm)
        self.cur_idx = (self.cur_idx + 1) % self.history_size
        self.cnt += 1

    def get_clip_value(self) -> float:
        if not self.grad_clip_enabled:
            return np.inf
        if (not self.auto_clip_enabled) or (self.cnt < self.min_history_length):
            return self.default_clip_value
        effective = self.gradient_norms if self.cnt >= self.history_size else self.gradient_norms[:self.cnt]
        return float(torch.quantile(effective, self.percentile))

    def get_last_gradient_norm(self) -> torch.Tensor:
        return self.gradient_norms[self.cur_idx - 1] if self.grad_clip_enabled else torch.tensor(float('nan'))
    



