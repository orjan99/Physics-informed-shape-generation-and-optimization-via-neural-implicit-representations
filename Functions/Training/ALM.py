
import torch 


class ALM:
    def __init__(self, *, loss_keys, objective_key, lambda_dict, alpha, gamma, epsilon, device):
        """
        Adaptive Augmented Lagrangian class - repo version:
        - dervived from https://github.com/ml-jku/GINNs-Geometry-informed-Neural-Networks/tree/main/train/train_utils 
        """
        self.device = device
        self.loss_keys = list(loss_keys)
        self.objective_key = objective_key

        self.lam = {k: torch.tensor(float(lambda_dict.get(k, 1.0)), device=device) for k in self.loss_keys}
        self.mu  = {k: torch.tensor(1.0, device=device) for k in self.loss_keys if k != objective_key}
        self.nu  = {k: torch.tensor(0.0, device=device) for k in self.loss_keys if k != objective_key}

        self.alpha   = float(alpha)
        self.gamma   = float(gamma)
        self.epsilon = float(epsilon)

    def build(self, loss_dict):
        total = torch.tensor(0.0, device=self.device)
        for k in self.loss_keys:
            v = loss_dict[k]
            v = torch.relu(v)
            s = torch.sqrt(v + 1e-12)  
            if k == self.objective_key:
                total = total + self.lam[k] * s
            else:
                total = total + self.lam[k] * s + 0.5 * self.mu[k] * s
        return total

    @torch.no_grad()
    def update(self, loss_dict):
        for k in self.loss_keys:
            if k == self.objective_key:
                continue
            v = torch.relu(loss_dict[k])
            s = torch.sqrt(v + 1e-12)
            self.nu[k] = self.nu[k] * self.alpha + (1.0 - self.alpha) * s
            self.mu[k] = self.gamma / (torch.sqrt(self.nu[k] + 1e-12) + self.epsilon)
            self.lam[k] = self.lam[k] + self.mu[k] * s



# class ALM:
#     def __init__(self, *, loss_keys, objective_key, lambda_dict, alpha, gamma, epsilon, device):
#         """
#         Adaptive Augmented Lagrangian class - paper version:
#         - dervived from https://arxiv.org/pdf/2402.14009 
#         """
#         self.device = device
#         self.loss_keys = list(loss_keys)
#         self.objective_key = objective_key

#         self.lam = {k: torch.tensor(float(lambda_dict.get(k, 1.0)), device=device) for k in self.loss_keys}
#         self.mu  = {k: torch.tensor(1.0, device=device) for k in self.loss_keys if k != objective_key}
#         self.nu  = {k: torch.tensor(0.0, device=device) for k in self.loss_keys if k != objective_key}

#         self.alpha   = float(alpha)
#         self.gamma   = float(gamma)     
#         self.epsilon = float(epsilon)

#     def build(self, loss_dict): 
#         """
#         L = L_E(gamma) + sum_i lam_i * C_i(gamma) + 1/2 * sum_i mu_i * C_i(gamma)^2
#         """
#         total = torch.tensor(0.0, device=self.device)

#         for k in self.loss_keys:
#             v = loss_dict[k]

#             if k == self.objective_key:
#                 total = total + v
#             else:
#                 c = v
#                 total = total + self.lam[k] * c + 0.5 * self.mu[k] * (c * c)

#         return total

#     @torch.no_grad()
#     def update(self, loss_dict):
#         for k in self.loss_keys:
#             if k == self.objective_key:
#                 continue
#             c = loss_dict[k]
#             nu_k = self.nu[k]
#             mu_k = self.mu[k]
#             self.nu[k] = self.alpha * nu_k + (1.0 - self.alpha) * (c * c)
#             self.mu[k] = self.gamma / (torch.sqrt(nu_k + 1e-12) + self.epsilon)
#             self.lam[k] = self.lam[k] + mu_k * c
