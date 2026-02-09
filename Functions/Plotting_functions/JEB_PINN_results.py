import torch 
import os 
import numpy as np
import matplotlib.pyplot as plt  
import plotly.graph_objects as go #type: ignore 
from plotly.subplots import make_subplots #type:ignore 

def plot_results(material_properties, u_model, v_model, w_model, fem_ref, save_dir, epoch, JEB):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") 

    if fem_ref is None:
        return
    
    s = float(JEB.domain_scaling_factor) 

    coords_scaled = fem_ref["coords_scaled"]
    x_disp = fem_ref["x_disp"]
    y_disp = fem_ref["y_disp"]
    z_disp = fem_ref["z_disp"]
    stress_FEM = fem_ref["sigma_vm"]

    points = torch.tensor(coords_scaled, dtype=torch.float32, device=device, requires_grad=True)

    u = u_model(points).squeeze(-1)
    v = v_model(points).squeeze(-1)
    w = w_model(points).squeeze(-1)

    gu_hat = torch.autograd.grad(u, points, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    gv_hat = torch.autograd.grad(v, points, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    gw_hat = torch.autograd.grad(w, points, grad_outputs=torch.ones_like(w), create_graph=True, retain_graph=True)[0]

    gu = s * gu_hat
    gv = s * gv_hat
    gw = s * gw_hat

    e11 = gu[:, 0]
    e22 = gv[:, 1]
    e33 = gw[:, 2]
    e12 = 0.5 * (gu[:, 1] + gv[:, 0])
    e13 = 0.5 * (gu[:, 2] + gw[:, 0])
    e23 = 0.5 * (gv[:, 2] + gw[:, 1])
    tr = e11 + e22 + e33

    lam = material_properties.lame_lambda.to(device)
    mu  = material_properties.lame_mu.to(device)

    s11 = 2 * mu * e11 + lam * tr
    s22 = 2 * mu * e22 + lam * tr
    s33 = 2 * mu * e33 + lam * tr
    s12 = 2 * mu * e12
    s13 = 2 * mu * e13
    s23 = 2 * mu * e23

    von_mises_stress = torch.sqrt(
        0.5 * (
            (s11 - s22) ** 2 +
            (s22 - s33) ** 2 +
            (s33 - s11) ** 2
        ) + 3 * (s12**2 + s13**2 + s23**2)
    )

    u_np = u.detach().cpu().numpy().reshape(-1)
    v_np = v.detach().cpu().numpy().reshape(-1)
    w_np = w.detach().cpu().numpy().reshape(-1)
    stress_np = von_mises_stress.detach().cpu().numpy().reshape(-1)
    coords_np = points.detach().cpu().numpy()

    u_abs_error = np.abs(u_np - x_disp)
    v_abs_error = np.abs(v_np - y_disp)
    w_abs_error = np.abs(w_np - z_disp)
    stress_abs_error = np.abs(stress_np - stress_FEM)

    os.makedirs(save_dir, exist_ok=True)

    def _scatter_triplet(title1, title2, title3, nn_vals, fem_vals, abs_vals, filename):
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(title1, title2, title3),
            specs=[[{"type": "scatter3d"}] * 3],
        )

        # Panel 1: NN
        vmin, vmax = nn_vals.min(), nn_vals.max()
        ticks = np.linspace(vmin, vmax, 6)
        ticktext = [f"{t:.4f}" for t in ticks]
        if len(ticktext) >= 2:
            ticktext[0] += " Min"
            ticktext[-1] += " Max"

        fig.add_trace(
            go.Scatter3d(
                x=coords_np[:, 0],
                y=coords_np[:, 1],
                z=coords_np[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=nn_vals,
                    colorscale="jet",
                    opacity=1,
                    colorbar=dict(
                        tickvals=ticks,
                        ticktext=ticktext,
                        tickangle=35,
                        orientation="h",
                        x=0.14,
                        y=-0.5,
                        len=0.35,
                        thickness=20,
                    ),
                ),
            ),
            row=1,
            col=1,
        )

        # Panel 2: FEM
        vmin, vmax = fem_vals.min(), fem_vals.max()
        ticks = np.linspace(vmin, vmax, 6)
        ticktext = [f"{t:.4f}" for t in ticks]
        if len(ticktext) >= 2:
            ticktext[0] += " Min"
            ticktext[-1] += " Max"

        fig.add_trace(
            go.Scatter3d(
                x=coords_np[:, 0],
                y=coords_np[:, 1],
                z=coords_np[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=fem_vals,
                    colorscale="jet",
                    opacity=1,
                    colorbar=dict(
                        tickvals=ticks,
                        ticktext=ticktext,
                        tickangle=35,
                        orientation="h",
                        x=0.5,
                        y=-0.5,
                        len=0.35,
                        thickness=20,
                    ),
                ),
            ),
            row=1,
            col=2,
        )

        # Panel 3: abs error
        vmin, vmax = abs_vals.min(), abs_vals.max()
        ticks = np.linspace(vmin, vmax, 6)
        ticktext = [f"{t:.4f}" for t in ticks]
        if len(ticktext) >= 2:
            ticktext[0] += " Min"
            ticktext[-1] += " Max"

        fig.add_trace(
            go.Scatter3d(
                x=coords_np[:, 0],
                y=coords_np[:, 1],
                z=coords_np[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=abs_vals,
                    colorscale="jet",
                    opacity=1,
                    colorbar=dict(
                        tickvals=ticks,
                        ticktext=ticktext,
                        tickangle=35,
                        orientation="h",
                        x=0.86,
                        y=-0.5,
                        len=0.35,
                        thickness=20,
                    ),
                ),
            ),
            row=1,
            col=3,
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(title="X", showgrid=False, zeroline=False, showbackground=False),
                yaxis=dict(title="Y", showgrid=False, zeroline=False, showbackground=False),
                zaxis=dict(title="Z", showgrid=False, zeroline=False, showbackground=False),
                bgcolor="white",
            ),
            scene2=dict(
                xaxis=dict(title="X", showgrid=False, zeroline=False, showbackground=False),
                yaxis=dict(title="Y", showgrid=False, zeroline=False, showbackground=False),
                zaxis=dict(title="Z", showgrid=False, zeroline=False, showbackground=False),
                bgcolor="white",
            ),
            scene3=dict(
                xaxis=dict(title="X", showgrid=False, zeroline=False, showbackground=False),
                yaxis=dict(title="Y", showgrid=False, zeroline=False, showbackground=False),
                zaxis=dict(title="Z", showgrid=False, zeroline=False, showbackground=False),
                bgcolor="white",
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=600,
            width=1200,
            margin=dict(t=50, b=0),
        )

        fig.write_html(filename)

    prefix = os.path.join(save_dir, f"epoch_{epoch:06d}_")

    _scatter_triplet("X Displacement NN", "X Displacement FEM", "Absolute Error",
                    u_np, x_disp, u_abs_error, prefix + "disp_x.html")

    _scatter_triplet("Y Displacement NN", "Y Displacement FEM", "Absolute Error",
                    v_np, y_disp, v_abs_error, prefix + "disp_y.html")

    _scatter_triplet("Z Displacement NN", "Z Displacement FEM", "Absolute Error",
                    w_np, z_disp, w_abs_error, prefix + "disp_z.html")

    _scatter_triplet("Von Mises NN", "Von Mises FEM", "Absolute Error",
                    stress_np, stress_FEM, stress_abs_error, prefix + "von_mises.html")