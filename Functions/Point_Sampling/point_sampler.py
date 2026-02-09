import torch
import numpy as np  
import math
import random 
from torch.utils.data import IterableDataset, DataLoader

def Point_Sampler(domain, 
                  interfaces = None, 
                  num_points_domain = 1000, 
                  num_points_interface = 0, 
                  num_sample_points_per_cell=1 ,
                  num_workers=0,
                  ): 
    """
    This function will be used to sample points from the given domain using stratified sampling.
    For each epoch of training, the function will generate a new set of points to be used for training --> Does not use the same points for each epoch

    The function can be used to sample points from a 2D or 3D domain.
    The function will divide the domain into cells and sample points in each cell using monte carlo sampling.

    Returns:
      iterator: each next(iterator) is a torch.FloatTensor of shape (N, Dim)
      The iterator will return a new set of points for each epoch of training
    """
    
    if isinstance(domain, torch.Tensor):
        device = domain.device
    else:
        device = None 

    # Generate a dataset of sample points using stratified sampling
    dataset = generate_sample_points(domain, 
                                     interfaces, 
                                     num_points_domain, 
                                     num_points_interface, 
                                     num_sample_points_per_cell,
                                     device=device) 

    # Create a DataLoader to iterate over the dataset
    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers, prefetch_factor=None)
    return iter(loader)


class generate_sample_points(IterableDataset):
    def __init__(self, 
                 domain, 
                 interfaces = None, 
                 num_points_domain = 1000, 
                 num_points_interfaces = 0, 
                 num_sample_points_per_cell = 1,
                 device = None):   
        
        super().__init__()
        self.domain = domain
        self.interfaces = interfaces
        self.num_points_interfaces = num_points_interfaces
        self.num_points_domain = num_points_domain
        self.num_sample_points_per_cell = num_sample_points_per_cell
        self.device = device

        dim = len(domain) // 2 

        if dim == 3:
            nx = int(num_points_domain**(1/3)) 
            ny = int(num_points_domain**(1/3)) 
            nz = int(num_points_domain**(1/3)) 
            self.num_cells = (nx, ny, nz) 
        elif dim == 2:
            nx = int(num_points_domain**(1/2))
            ny = int(num_points_domain**(1/2))
            self.num_cells = (nx, ny) 

    def __iter__(self):
        while True:

            samples_domain = stratified_sampling_1(self.domain, 
                                                 self.num_cells, 
                                                 self.num_sample_points_per_cell,
                                                 device=self.device)  
            samples = samples_domain

            if self.interfaces is not None:
                if isinstance(samples_domain, torch.Tensor): 
                    samples_interfaces = self.interfaces.sample_points_from_all_interfaces(num_points=self.num_points_interfaces,
                                                                                           device=self.device) 
                    if isinstance(samples_interfaces, np.ndarray):
                        samples_interfaces = torch.from_numpy(samples_interfaces)
                    samples = torch.cat((samples_domain, samples_interfaces), dim=0) 
                elif isinstance(samples_domain, np.ndarray):
                    samples_interfaces = self.interfaces.sample_points_from_all_interfaces(num_points = self.num_points_interfaces,
                                                                                           device=self.device)
                    samples = np.concatenate((samples_domain, samples_interfaces), axis=0) 

            # Convert the samples to a PyTorch tensor 
            if isinstance(samples, np.ndarray):
                samples = torch.from_numpy(samples).to(dtype=torch.float32, device=self.device) 
            elif isinstance(samples, torch.Tensor) or isinstance(samples, list): 
                samples = samples.to(dtype=torch.float32, device=self.device)
            else:
                raise ValueError("Samples must be a numpy array or a PyTorch tensor.") 
    
            yield samples 


def stratified_sampling_1(domain, num_cells, num_sample_points_per_cell = 1, device=None): 
    
    if isinstance(domain, torch.Tensor):
        device = domain.device if device is not None else torch.device("cpu")
        domain = domain.to(dtype=torch.float32, device=device)
    elif isinstance(domain, np.ndarray) or isinstance(domain, list):
        domain = np.array(domain, dtype=np.float32)
    else:
        raise ValueError("Domain must be a numpy array,list or PyTorch tensor.") 
    problem_dimensionality = len(domain) // 2  
    all_points = [] 


    if problem_dimensionality == 2:

        # Find the bounds of the domain 
        x_min, x_max = domain[0], domain[1]
        y_min, y_max = domain[2], domain[3]

        # Get the number of cells in each dimension
        num_cells_x, num_cells_y = num_cells[0], num_cells[1]

        # Define the grid cell boundaries
        if isinstance(domain, torch.Tensor):
            cell_x_edges = torch.linspace(x_min, x_max, num_cells_x + 1, device=device)
            cell_y_edges = torch.linspace(y_min, y_max, num_cells_y + 1, device=device)
        elif isinstance(domain, np.ndarray) or isinstance(domain, list):  
            cell_x_edges = np.linspace(x_min, x_max, num_cells_x + 1)
            cell_y_edges = np.linspace(y_min, y_max, num_cells_y + 1) 
        else:
            raise ValueError("Domain must be a numpy array or a PyTorch tensor.") 

        # Loop over each cell
        for i in range(num_cells_x):
            for j in range(num_cells_y):
                if isinstance(domain, torch.Tensor):
                    cell_domain = torch.stack([
                        cell_x_edges[i],   cell_x_edges[i+1],
                        cell_y_edges[j],   cell_y_edges[j+1]
                    ]).to(device)
                elif isinstance(domain, np.ndarray) or isinstance(domain, list): 
                    cell_domain = [
                        cell_x_edges[i],   cell_x_edges[i+1],
                        cell_y_edges[j],   cell_y_edges[j+1]
                ]
                pts, _ = monte_carlo_sampling(cell_domain, num_sample_points_per_cell,device=device) 
                all_points.append(pts)
    elif problem_dimensionality == 3:

        x_min, x_max = domain[0], domain[1]
        y_min, y_max = domain[2], domain[3]
        z_min, z_max = domain[4], domain[5]
        num_cells_x, num_cells_y, num_cells_z = num_cells

        # Define the grid cell boundaries 
        if isinstance(domain, torch.Tensor):
            cell_x_edges = torch.linspace(x_min, x_max, num_cells_x + 1, device=device) 
            cell_y_edges = torch.linspace(y_min, y_max, num_cells_y + 1, device=device)
            cell_z_edges = torch.linspace(z_min, z_max, num_cells_z + 1, device=device)
        elif isinstance(domain, np.ndarray) or isinstance(domain, list): 
            cell_x_edges = np.linspace(x_min, x_max, num_cells_x + 1)
            cell_y_edges = np.linspace(y_min, y_max, num_cells_y + 1)
            cell_z_edges = np.linspace(z_min, z_max, num_cells_z + 1) 
        else:
            raise ValueError("Domain must be a numpy array or a PyTorch tensor.") 

        # Loop over each cell
        for i in range(num_cells_x):
            for j in range(num_cells_y):
                for k in range(num_cells_z):
                    if isinstance(domain, torch.Tensor):
                        cell_domain = torch.stack([
                            cell_x_edges[i],   cell_x_edges[i+1],
                            cell_y_edges[j],   cell_y_edges[j+1],
                            cell_z_edges[k],   cell_z_edges[k+1]
                        ]).to(device) 
                    elif isinstance(domain, np.ndarray) or isinstance(domain, list): 
                        cell_domain = [
                            cell_x_edges[i],   cell_x_edges[i+1],
                            cell_y_edges[j],   cell_y_edges[j+1],
                            cell_z_edges[k],   cell_z_edges[k+1]
                        ]
                    pts, _ = monte_carlo_sampling(cell_domain, num_sample_points_per_cell,device=device) 
                    all_points.append(pts)

    else: 
        raise ValueError("Invalid dimensionality of the problem. Only 2D and 3D are supported.")


    if isinstance(domain, torch.Tensor):
        point_samples = torch.vstack(all_points).to(dtype=torch.float32, device=device) 

    elif isinstance(domain, np.ndarray) or isinstance(domain, list): 
        point_samples = np.vstack(all_points).astype(np.float32) 
    else:
        raise ValueError("Domain must be a numpy array or a PyTorch tensor.")
    
    return point_samples 


def monte_carlo_sampling(domain,num_sample_points,device = None): 
    """
    Function to sample points from the given domain using Monte Carlo sampling - random sampling

    Inputs:
    1. domain: array of domain vertices --> [x_min,x_max,y_min,y_max,...] --> Shape = (1,2*dim)
    2. num_sample_points: number of sample points to be sampled from the domain 

    Outputs:
    1. point_coordinates: array of shape (num_sample_points, dim) --> random points sampled from the domain
    2. weights: array of shape (num_sample_points,) --> quadrature weights (volume element)
    """

    if isinstance(domain, torch.Tensor):
        domain = domain.to(dtype=torch.float32, device=device)  
        problem_dimensionality = len(domain) // 2  

        t = torch.rand(num_sample_points, problem_dimensionality, device=device)

        point_coordinates = domain[0::2] + t * (domain[1::2] - domain[0::2])  
        point_coordinates = point_coordinates.to(dtype=torch.float32, device=device) 

        # Weights = volume of the domain / number of sample points --> quadrature weights
        weights = torch.ones(num_sample_points, device=device) * torch.prod(domain[1::2] - domain[0::2]) / num_sample_points
        weights = weights.to(dtype=torch.float32, device=device)  
        return point_coordinates, weights  
     
    elif isinstance(domain, np.ndarray) or isinstance(domain, list): 
        domain = np.array(domain, dtype=np.float32) 

        problem_dimensionality = len(domain)//2  
        t = np.random.rand(num_sample_points,problem_dimensionality) 
        point_coordinates = domain[0::2] + t*(domain[1::2] - domain[0::2]) 
        point_coordinates = point_coordinates.astype(np.float32) 

        # Weights = volume of the domain / number of sample points --> quadrature weights 
        weights = np.ones(num_sample_points) * np.prod(domain[1::2] - domain[0::2]) / num_sample_points 
        weights = weights.astype(np.float32) 

        return point_coordinates, weights 
    else:
        raise ValueError("Domain must be a numpy array or a PyTorch tensor.")  

    
 # --------------------------------------------------------------------------------------------------------------------------------

""" Sampling helpers derived from NTopo repository: https://github.com/JonasZehn/ntopo/blob/main/ntopo/utils.py :"""

def get_grid_centers(domain: np.ndarray, n_cells, dtype=np.float32):
    d = np.array(domain, dtype=dtype)
    nc = np.array(n_cells, dtype=np.int32)
    assert nc.size in (2, 3)
    if nc.size == 2:
        nx, ny = int(nc[0]), int(nc[1])
        w = d[1] - d[0]
        h = d[3] - d[2]
        xs = np.linspace(d[0] + 0.5*w/nx, d[1] - 0.5*w/nx, nx, dtype=dtype)
        ys = np.linspace(d[2] + 0.5*h/ny, d[3] - 0.5*h/ny, ny, dtype=dtype)
        X, Y = np.meshgrid(xs, ys)
        out = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
        return out
    else: 
        nx, ny, nz = int(nc[0]), int(nc[1]), int(nc[2])
        w = d[1] - d[0]; h = d[3] - d[2]; l = d[5] - d[4]
        cx = 0.5 * w/nx; cy = 0.5 * h/ny; cz = 0.5 * l/nz
        xs = np.linspace(d[0] + cx, d[1] - cx, nx, dtype=dtype)
        ys = np.linspace(d[2] + cy, d[3] - cy, ny, dtype=dtype)
        zs = np.linspace(d[4] + cz, d[5] - cz, nz, dtype=dtype)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        out = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1)
        return out

def stratified_sampling(domain, n_cells, n_points_per_cell=1, dtype=np.float32):
    dom = np.array(domain, dtype=dtype)
    assert isinstance(n_cells, (list, tuple, np.ndarray))
    if len(n_cells) == 2:
        nx, ny = int(n_cells[0]), int(n_cells[1])
        n = nx*ny*n_points_per_cell
        x_lefts = np.linspace(dom[0], dom[1], nx, endpoint=False, dtype=dtype)
        x_rights = x_lefts + (dom[1]-dom[0])/nx
        y_bottoms = np.linspace(dom[2], dom[3], ny, endpoint=False, dtype=dtype)
        y_tops = y_bottoms + (dom[3]-dom[2])/ny
        XL, YB = np.meshgrid(x_lefts, y_bottoms)
        XR, YT = np.meshgrid(x_rights, y_tops)
        XL = XL.reshape(-1,1); XR = XR.reshape(-1,1)
        YB = YB.reshape(-1,1); YT = YT.reshape(-1,1)
        XL = np.tile(XL, (n_points_per_cell,1)); XR = np.tile(XR, (n_points_per_cell,1))
        YB = np.tile(YB, (n_points_per_cell,1)); YT = np.tile(YT, (n_points_per_cell,1))
        l1 = np.random.uniform(0.0, 1.0, (n,1)).astype(dtype)
        l2 = np.random.uniform(0.0, 1.0, (n,1)).astype(dtype)
        xs = (1.0 - l1)*XL + l1*XR
        ys = (1.0 - l2)*YB + l2*YT
        return np.hstack([xs, ys]).astype(dtype)
    elif len(n_cells) == 1:
        nx = int(n_cells[0])
        n = nx*n_points_per_cell
        L = np.linspace(dom[0], dom[1], nx, endpoint=False, dtype=dtype).reshape(-1,1)
        R = L + (dom[1]-dom[0])/nx
        L = np.tile(L, (n_points_per_cell,1)); R = np.tile(R, (n_points_per_cell,1))
        l1 = np.random.uniform(0.0, 1.0, (n,1)).astype(dtype)
        x = (1.0 - l1)*L + l1*R
        return x
    elif len(n_cells) == 3:
        nx, ny, nz = int(n_cells[0]), int(n_cells[1]), int(n_cells[2])
        n = nx*ny*nz*n_points_per_cell
        xL = np.linspace(dom[0], dom[1], nx, endpoint=False, dtype=dtype)
        xR = xL + (dom[1]-dom[0])/nx
        yB = np.linspace(dom[2], dom[3], ny, endpoint=False, dtype=dtype)
        yT = yB + (dom[3]-dom[2])/ny
        zN = np.linspace(dom[4], dom[5], nz, endpoint=False, dtype=dtype)
        zF = zN + (dom[5]-dom[4])/nz
        XL, YB, ZN = np.meshgrid(xL, yB, zN, indexing='ij')
        XR, YT, ZF = np.meshgrid(xR, yT, zF, indexing='ij')
        XL = XL.reshape(-1,1); XR = XR.reshape(-1,1)
        YB = YB.reshape(-1,1); YT = YT.reshape(-1,1)
        ZN = ZN.reshape(-1,1); ZF = ZF.reshape(-1,1)
        # tile per cell
        XL = np.tile(XL, (n_points_per_cell,1)); XR = np.tile(XR, (n_points_per_cell,1))
        YB = np.tile(YB, (n_points_per_cell,1)); YT = np.tile(YT, (n_points_per_cell,1))
        ZN = np.tile(ZN, (n_points_per_cell,1)); ZF = np.tile(ZF, (n_points_per_cell,1))
        l1 = np.random.uniform(0.0, 1.0, (n,1)).astype(dtype)
        l2 = np.random.uniform(0.0, 1.0, (n,1)).astype(dtype)
        l3 = np.random.uniform(0.0, 1.0, (n,1)).astype(dtype)
        xs = (1.0 - l1)*XL + l1*XR
        ys = (1.0 - l2)*YB + l2*YT
        zs = (1.0 - l3)*ZN + l3*ZF
        return np.hstack([xs, ys, zs]).astype(dtype)
    else:
        raise NotImplementedError("unsupported dimensionality for n_cells")

def gen_samples(domain, n_cells):
    while True:
        xs = stratified_sampling(domain, n_cells, n_points_per_cell=1, dtype=np.float32)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") 
        yield torch.tensor(xs, dtype=torch.float32, device=device) 

def get_default_sample_counts(domain: np.ndarray, total_samples: int, even=None):
    """
    2D: returns [nx, ny] matching aspect ratio
    3D: returns [nx, ny, nz] matching domain proportions
    """
    assert len(domain) in (4, 6)
    if len(domain) == 4:
        w = domain[1] - domain[0]
        h = domain[3] - domain[2]
        def best_poss(pos):
            cand_x = pos[0]; cand_y = pos[1]
            if even:
                cand_x = [n for n in cand_x if n % 2 == 0]
                cand_y = [n for n in cand_y if n % 2 == 0]
            pairs = [(nx, ny) for nx in cand_x for ny in cand_y]
            def rel_err(p):
                nx, ny = p
                return abs((nx/ny) - (w/h)) / (w/h)
            idx = min(range(len(pairs)), key=lambda i: rel_err(pairs[i]))
            return list(pairs[idx])
        poss = [[], []]
        ny0 = int(math.floor(math.sqrt(total_samples * h / w)))
        poss[1] += [ny0, ny0+1]
        nx0 = int(math.floor(ny0 * w / h))
        poss[0] += [nx0, nx0+1]
        nx1 = int(math.floor(math.sqrt(total_samples * w / h)))
        poss[0] += [nx1, nx1+1]
        ny1 = int(math.floor(nx1 * h / w))
        poss[1] += [ny1, ny1+1]
        return best_poss(poss)
    else:
        if even:
            raise Exception('get_default_sample_counts: even=True not supported for 3D')
        w = domain[1] - domain[0]
        h = domain[3] - domain[2]
        d = domain[5] - domain[4]
        scale = pow(total_samples / max(1e-30, (w * h * d)), 1.0 / 3.0)
        nx = max(1, int(round(scale * w)))
        ny = max(1, int(round(nx * h / max(1e-30, w))))
        nz = max(1, int(round(nx * d / max(1e-30, w))))
        return [nx, ny, nz]

def get_grid_points(domain, n_cells, dtype=np.float32):
    d = np.array(domain, dtype=dtype)
    nc = np.array(n_cells, dtype=np.int32)
    assert nc.size in (2, 3)
    if nc.size == 2:
        nx, ny = int(nc[0]), int(nc[1])
        x = np.linspace(d[0], d[1], nx, dtype=dtype)
        y = np.linspace(d[2], d[3], ny, dtype=dtype)
        X, Y = np.meshgrid(x, y)
        return np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    else:
        nx, ny, nz = int(nc[0]), int(nc[1]), int(nc[2])
        x = np.linspace(d[0], d[1], nx, dtype=dtype)
        y = np.linspace(d[2], d[3], ny, dtype=dtype)
        z = np.linspace(d[4], d[5], nz, dtype=dtype)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        return np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1)


def get_grid_centers_spacing(domain, n_cells, dtype=np.float32):
    d = np.array(domain, dtype=dtype)
    nc = np.array(n_cells, dtype=np.int32)
    assert nc.size in (2, 3)
    if nc.size == 3:
        w = d[1] - d[0]; h = d[3] - d[2]; l = d[5] - d[4]
        return [0.5 * w/nc[0], 0.5 * h/nc[1], 0.5 * l/nc[2]]
    raise Exception('get_grid_centers_spacing: only supported for 3D domains')