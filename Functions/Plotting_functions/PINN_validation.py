import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import trimesh  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
from Models.PINN_Models.PINN import PINN
from File_Paths.file_paths import twoD_PINN_trained_model_dir, threeD_PINN_trained_model_dir
from Functions.Computations.von_mises_stress import compute_von_mises


def pinn_validation(checkpoint_filename, plot_quantity='x_disp', scale_displacement=True):

    # ----------------------------------------------------------------
    # 1) Load checkpoint, data, and model 
    # ----------------------------------------------------------------
    file_path = os.path.join(threeD_PINN_trained_model_dir, checkpoint_filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Checkpoint file {file_path} does not exist.")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    checkpoint = torch.load(file_path, map_location=device)
    test_case = checkpoint['test_case']
    load_case = checkpoint['load_case']
    FEM_data = checkpoint['FEM_data']
    mesh_object = checkpoint['mesh_object']
    hparams_training = checkpoint['hparams']['training']
    hparams_SIREN = checkpoint['hparams']['model']
    hparams_feature_expansion = checkpoint['hparams']['feature_expansion']
    hparams_point_sampler = checkpoint['hparams']['point_sampler'] 

  
    model = PINN(test_case, feature_expansion = hparams_feature_expansion ,model_hyperparameters = hparams_SIREN) 



    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    x = FEM_data[:, 2]
    y = FEM_data[:, 3]
    z = FEM_data[:, 4]
    FEM_points = np.column_stack((x, y, z))

    center = test_case.domain_center
    scale_factor = test_case.domain_scaling_factor
    points_np = (FEM_points - center[np.newaxis, :]) * scale_factor
    points = torch.tensor(points_np, dtype=torch.float32, device=device)
    densities = torch.ones_like(points[:, 0], dtype=torch.float32, device=device)

    # choose FEM outputs
    if load_case == 'horizontal':
        x_disp = FEM_data[:, 10];  y_disp = FEM_data[:, 11];  z_disp = FEM_data[:, 12]
        disp_magnitude_FEM = FEM_data[:, 13];  stress_FEM = FEM_data[:, 14]
    elif load_case == 'vertical':
        x_disp = FEM_data[:, 5];   y_disp = FEM_data[:, 6];   z_disp = FEM_data[:, 7]
        disp_magnitude_FEM = FEM_data[:, 8];   stress_FEM = FEM_data[:, 9]
    elif load_case == 'diagonal':
        x_disp = FEM_data[:, 15];  y_disp = FEM_data[:, 16];  z_disp = FEM_data[:, 17]
        disp_magnitude_FEM = FEM_data[:, 18];  stress_FEM = FEM_data[:, 19]
    else:
        raise ValueError("Invalid load case. Choose from 'horizontal', 'vertical', or 'diagonal'.")

    # ----------------------------------------------------------------
    # 2) Run model & pick plot_quant / plot_quant_FEM / title 
    # ----------------------------------------------------------------
    if plot_quantity == 'x_disp':
        with torch.no_grad():
            disp_field = model(points, densities)
        disp = disp_field[:, 0] / scale_factor
        plot_quant = disp
        plot_quant_FEM = x_disp
        title = "PINN Validation: X Displacement Field"

    elif plot_quantity == 'y_disp':
        with torch.no_grad():
            disp_field = model(points, densities)
        disp = disp_field[:, 1] / scale_factor
        plot_quant = disp
        plot_quant_FEM = y_disp
        title = "PINN Validation: Y Displacement Field"

    elif plot_quantity == 'z_disp':
        with torch.no_grad():
            disp_field = model(points, densities)
        disp = disp_field[:, 2] / scale_factor
        plot_quant = disp
        plot_quant_FEM = z_disp
        title = "PINN Validation: Z Displacement Field"

    elif plot_quantity == 'disp_magnitude':
        with torch.no_grad():
            disp_field = model(points, densities) / scale_factor
        disp = torch.norm(disp_field, dim=1)
        plot_quant = disp
        plot_quant_FEM = disp_magnitude_FEM
        title = "PINN Validation: Displacement Magnitude Field"

    elif plot_quantity == 'stress':
        points.requires_grad_(True)
        disp_field = model(points, densities)
        stress = compute_von_mises(test_case=test_case,
                                  displacement_field=disp_field,
                                  coords=points,
                                  scale_up=True)
        plot_quant = stress.detach().cpu()
        points = points.detach().cpu()
        plot_quant_FEM = stress_FEM
        title = "PINN Validation: Stress Field"

    else:
        raise ValueError("Invalid plot quantity. Choose from 'x_disp','y_disp','z_disp','disp_magnitude','stress'.")

    # ----------------------------------------------------------------
    # 3) Define ANSYS‐style “jet” colorscale + helper for ticks
    # ----------------------------------------------------------------
    jet_colorscale = [
        [0.0,   "rgb(  0,   0, 128)"],
        [0.15,  "rgb(  0,   0, 255)"],
        [0.33,  "rgb(  0, 255, 255)"],
        [0.50,  "rgb(  0, 255,   0)"],
        [0.67,  "rgb(255, 255,   0)"],
        [0.85,  "rgb(255, 128,   0)"],
        [1.0,   "rgb(255,   0,   0)"]
    ]

    def make_colorbar_dict(data, title_text, x_position):
        cmin, cmax = float(np.min(data)), float(np.max(data))
        tickvals = np.linspace(cmin, cmax, 10)
        ticktext = [f"{v:.6f}" for v in tickvals]
        ticktext[0]  = ticktext[0] + " Min"
        ticktext[-1] = ticktext[-1] + " Max"

        return dict(
            title=dict(text=title_text),
            titleside="top",
            colorscale=jet_colorscale,
            cmin=cmin,
            cmax=cmax,
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            ticks="outside",
            ticklen=5,
            tickfont=dict(size=12),
            lenmode="fraction",
            len=0.8,
            bordercolor="black",
            borderwidth=1,
            x=x_position
        )

    # ----------------------------------------------------------------
    # 4) Build subplots & add traces with custom colorbars
    # ----------------------------------------------------------------
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("NN Displacement Field", "FEM Displacement Field"),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
    )

    # NN panel
    cb_NN = make_colorbar_dict(
        data=plot_quant.cpu().numpy(),
        title_text=("NN (MPa)" if plot_quantity=='stress' else "NN (mm)"),
        x_position=-0.1
    )
    fig.add_trace(
        go.Scatter3d(
            x=points.cpu().numpy()[:, 0],
            y=points.cpu().numpy()[:, 1],
            z=points.cpu().numpy()[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=plot_quant.cpu().numpy(),
                colorscale=jet_colorscale,
                cmin=cb_NN['cmin'],
                cmax=cb_NN['cmax'],
                opacity=1.0,
                showscale=True,
                colorbar=cb_NN
            ),
            name=" "
        ),
        row=1, col=1
    )

    # FEM panel
    cb_FEM = make_colorbar_dict(
        data=plot_quant_FEM,
        title_text=("FEM (MPa)" if plot_quantity=='stress' else "FEM (mm)"),
        x_position=1.1
    )
    fig.add_trace(
        go.Scatter3d(
            x=FEM_points[:, 0],
            y=FEM_points[:, 1],
            z=FEM_points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=plot_quant_FEM,
                colorscale=jet_colorscale,
                cmin=cb_FEM['cmin'],
                cmax=cb_FEM['cmax'],
                opacity=1.0,
                showscale=True,
                colorbar=cb_FEM
            ),
            name=" "
        ),
        row=1, col=2
    )

    # ----------------------------------------------------------------
    # 5) Final layout
    # ----------------------------------------------------------------
    fig.update_layout(
        title=title,
        scene=dict(aspectmode='data'),
        scene2=dict(aspectmode='data')
    )

    return fig