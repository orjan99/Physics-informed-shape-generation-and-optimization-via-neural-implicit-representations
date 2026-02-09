import torch 
import numpy as np 

def compute_L2_errors_2d(u_model, v_model, material_properties, fem_ref, BRIDGE): 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") 

    if fem_ref is None:
        return None, None, None

    u_model.eval()
    v_model.eval()

    coords_scaled = fem_ref["coords_scaled"]
    x_disp_FEM = fem_ref["x_disp"]
    y_disp_FEM = fem_ref["y_disp"]
    stress_FEM = fem_ref["sigma_vm"]

    coords = torch.tensor(
        coords_scaled, dtype=torch.float32, device=device, requires_grad=True
    )

    u = u_model(coords)  
    v = v_model(coords)  

    grad_u = torch.autograd.grad(
        u,
        coords,
        grad_outputs=torch.ones_like(u),
        create_graph=False,
        retain_graph=True,
    )[0]
    grad_v = torch.autograd.grad(
        v,
        coords,
        grad_outputs=torch.ones_like(v),
        create_graph=False,
        retain_graph=False,
    )[0]

    eps11 = grad_u[:, 0]
    eps22 = grad_v[:, 1]
    eps12 = 0.5 * (grad_u[:, 1] + grad_v[:, 0])
    tr = eps11 + eps22

    lam = material_properties.lame_lambda.to(device)
    mu = material_properties.lame_mu.to(device)

    s11 = 2.0 * mu * eps11 + lam * tr
    s22 = 2.0 * mu * eps22 + lam * tr
    s12 = 2.0 * mu * eps12

    svm2 = s11**2 - s11 * s22 + s22**2 + 3.0 * s12**2
    svm = torch.sqrt(torch.clamp(svm2, min=1e-32)) * BRIDGE.domain_scaling_factor

    u_pred = u.detach().cpu().numpy().reshape(-1)
    v_pred = v.detach().cpu().numpy().reshape(-1)
    s_pred = svm.detach().cpu().numpy().reshape(-1)

    x_disp_FEM = np.asarray(x_disp_FEM)
    y_disp_FEM = np.asarray(y_disp_FEM)
    stress_FEM = np.asarray(stress_FEM)

    def L2(pred, true):
        num = np.sum((pred - true) ** 2)
        den = np.sum(true**2) + 1e-32
        return float(np.sqrt(num / den))

    L2_u = L2(u_pred, x_disp_FEM)
    L2_v = L2(v_pred, y_disp_FEM)
    L2_s = L2(s_pred, stress_FEM)

    u_model.train()
    v_model.train()

    return L2_u, L2_v, L2_s


def compute_L2_errors_3d(u_model, v_model, w_model, material_properties, fem_ref,JEB):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") 
    if fem_ref is None:
        return None, None, None, None
    
    s = float(JEB.domain_scaling_factor)  

    inside_mask = fem_ref.get("inside_mask", None)
    if inside_mask is None:
        inside_mask = np.ones(fem_ref["coords_scaled"].shape[0], dtype=bool)

    # Compute L2 only on FEM nodes
    coords_scaled = fem_ref["coords_scaled"][inside_mask]
    x_disp = fem_ref["x_disp"][inside_mask]
    y_disp = fem_ref["y_disp"][inside_mask]
    z_disp = fem_ref["z_disp"][inside_mask]
    stress_FEM = fem_ref["sigma_vm"][inside_mask]

    # Save/restore training modes
    was_training = (u_model.training, v_model.training, w_model.training)
    u_model.eval(); v_model.eval(); w_model.eval()

    with torch.enable_grad():
        points = torch.tensor(coords_scaled, dtype=torch.float32, device=device, requires_grad=True)

        u = u_model(points).squeeze(-1)
        v = v_model(points).squeeze(-1)
        w = w_model(points).squeeze(-1)

        gu_hat = torch.autograd.grad(u, points, grad_outputs=torch.ones_like(u), create_graph=False, retain_graph=True)[0]
        gv_hat = torch.autograd.grad(v, points, grad_outputs=torch.ones_like(v), create_graph=False, retain_graph=True)[0]
        gw_hat = torch.autograd.grad(w, points, grad_outputs=torch.ones_like(w), create_graph=False, retain_graph=False)[0]

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

    def L2(pred, true):
        num = np.sum((pred - true) ** 2)
        den = np.sum(true ** 2) + 1e-32
        return float(np.sqrt(num / den))

    L2_u = L2(u_np, x_disp)
    L2_v = L2(v_np, y_disp)
    L2_w = L2(w_np, z_disp)
    L2_s = L2(stress_np, stress_FEM)

    # restore training modes
    if was_training[0]: u_model.train()
    if was_training[1]: v_model.train()
    if was_training[2]: w_model.train()

    return L2_u, L2_v, L2_w, L2_s