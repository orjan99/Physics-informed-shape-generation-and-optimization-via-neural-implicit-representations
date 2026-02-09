import os
import pandas as pd

def save_metrics_csv_2d(history, save_dir, filename="training_metrics.csv"):
    """
    Save the full history (epoch, loss, L2_u, L2_v, L2_s) to CSV.
    """
    if len(history["epoch"]) == 0:
        return

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, filename)

    df = pd.DataFrame(
        {
            "epoch": history["epoch"],
            "ritz_loss": history["loss"],
            "L2_u": history["L2_u"],
            "L2_v": history["L2_v"],
            "L2_sigma": history["L2_s"],
        }
    )
    df.to_csv(csv_path, index=False)