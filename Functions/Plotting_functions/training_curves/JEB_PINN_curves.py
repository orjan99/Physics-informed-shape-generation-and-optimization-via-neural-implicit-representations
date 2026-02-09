
import os 
import numpy as np
import matplotlib.pyplot as plt 

def save_training_curves_3d(history, filename):
    if len(history["epoch"]) == 0:
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    epochs = np.array(history["epoch"])
    loss = np.array(history["loss"])
    L2_u = np.array(history["L2_u"], dtype=float)
    L2_v = np.array(history["L2_v"], dtype=float)
    L2_w = np.array(history["L2_w"], dtype=float)
    L2_s = np.array(history["L2_s"], dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].plot(epochs, loss, "k-")
    axes[0, 0].set_title("Ritz loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, L2_u, "b-")
    axes[0, 1].set_title("L2 error: u (x_disp)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("L2_u")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs, L2_v, "g-")
    axes[0, 2].set_title("L2 error: v (y_disp)")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("L2_v")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, L2_w, "c-")
    axes[1, 0].set_title("L2 error: w (z_disp)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("L2_w")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, L2_s, "r-")
    axes[1, 1].set_title("L2 error: σ_vM")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("L2_σ")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")

    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def save_training_curves_log_3d(history, filename):
    if len(history["epoch"]) == 0:
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    epochs = np.array(history["epoch"])
    loss = np.array(history["loss"])
    L2_u = np.array(history["L2_u"], dtype=float)
    L2_v = np.array(history["L2_v"], dtype=float)
    L2_w = np.array(history["L2_w"], dtype=float)
    L2_s = np.array(history["L2_s"], dtype=float)

    eps = 1e-12
    loss = np.clip(loss, eps, None)
    L2_u = np.clip(L2_u, eps, None)
    L2_v = np.clip(L2_v, eps, None)
    L2_w = np.clip(L2_w, eps, None)
    L2_s = np.clip(L2_s, eps, None)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].plot(epochs, loss, "k-")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Ritz loss (log scale)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, which="both", alpha=0.3)

    axes[0, 1].plot(epochs, L2_u, "b-")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("L2 error: u (log)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("L2_u")
    axes[0, 1].grid(True, which="both", alpha=0.3)

    axes[0, 2].plot(epochs, L2_v, "g-")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("L2 error: v (log)")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("L2_v")
    axes[0, 2].grid(True, which="both", alpha=0.3)

    axes[1, 0].plot(epochs, L2_w, "c-")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("L2 error: w (log)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("L2_w")
    axes[1, 0].grid(True, which="both", alpha=0.3)

    axes[1, 1].plot(epochs, L2_s, "r-")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("L2 error: σ_vM (log)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("L2_σ")
    axes[1, 1].grid(True, which="both", alpha=0.3)

    axes[1, 2].axis("off")

    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)