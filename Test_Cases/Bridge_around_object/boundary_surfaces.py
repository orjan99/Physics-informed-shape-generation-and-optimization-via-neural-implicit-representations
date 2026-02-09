
import numpy as np
import torch  

class Bridge_Interfaces:

    def __init__(self, 
                 domain,
                 obstacle_centroid,
                 obstacle_radius,
                 boundary_vertices,
                 prescribed_thickness_obstacle,
                 prescribed_thickness_boundaries,
                 Symmetry = False): 

        self.domain = domain
        self.obstacle_centroid = obstacle_centroid
        self.obstacle_radius = obstacle_radius
        self.obstacle_thickness_radius = obstacle_radius + prescribed_thickness_obstacle 
        self.boundary_vertices = boundary_vertices
        self.prescribed_thickness_obstacle = prescribed_thickness_obstacle
        self.prescribed_thickness_boundaries = prescribed_thickness_boundaries
        self.Symmetry = Symmetry 

    
    #1. Sample points on the dirichlet boundary surface 
    def sample_points_on_dirichlet_boundary(self, 
                                            num_points, 
                                            random_seed = None, # Needed for consistency with other interfaces (not used here)
                                            output_type = 'numpy_array',
                                            device = None,   
                                            ):  
        
        if isinstance(self.boundary_vertices, torch.Tensor):
            x_left = self.boundary_vertices[0][0].item()
            x_right = self.boundary_vertices[2][0].item()
            y_min = self.boundary_vertices[0][1].item()
            y_max = self.boundary_vertices[1][1].item() 
        else:
            x_left = self.boundary_vertices[0][0]
            x_right = self.boundary_vertices[2][0]
            y_min = self.boundary_vertices[0][1]
            y_max = self.boundary_vertices[1][1] 

        y_vals = torch.rand(num_points,device=device) * (y_max - y_min) + y_min
        x_vals_left = torch.full_like(y_vals, x_left)  
        x_vals_right = torch.full_like(y_vals, x_right)  

        points_left = torch.stack([x_vals_left, y_vals], dim=1) 
        points_right = torch.stack([x_vals_right, y_vals], dim=1) 

        # Concatenate the left and right boundary points 
        points = torch.cat([points_left, points_right], dim=0)

        if output_type == 'numpy_array':
            points = points.cpu().numpy()
        elif output_type == 'torch_tensor':
            points = points.to(device)  

        if self.Symmetry == True:
            points = points[points[:, 0] >= 0]  # Filter points to keep only the right side if symmetry is applied
            points = points[points[:, 1] >= 0]  # Filter points to keep only the upper side if symmetry is applied

        return points 
    
    
    #2. Sample points from obstacle boundary 
    def sample_points_from_pinn_interface(self, num_points, random_seed=None, device=None): 
        if device is None:
            device = torch.device('cpu')  
        theta = torch.rand(num_points,device=device) * 2 * np.pi 
        x_vals = self.obstacle_centroid[0] + self.obstacle_radius * torch.cos(theta)
        y_vals = self.obstacle_centroid[1] + self.obstacle_radius * torch.sin(theta)
        points = torch.stack([x_vals, y_vals], dim=1) 
        points = points.to(device)  
        if self.Symmetry == True:
            points = points[points[:, 0] >= 0]  # Filter points to keep only the right side if symmetry is applied
            points = points[points[:, 1] >= 0]  # Filter points to keep only the upper side if symmetry is applied
        return points 
   

    def sample_points_on_neumann_boundary(self, 
                                          num_points, 
                                          load_type: str, 
                                          output_type = 'numpy_array',
                                          device = None, #needed for consistency with other interfaces 
                                          ): 
        
        points = self.sample_points_from_pinn_interface(num_points, device=device) 
        if load_type == 'vertical':
            mask = points[:,1] >= self.obstacle_centroid[1]
        elif load_type == 'horizontal': 
            mask = points[:,0] >= self.obstacle_centroid[0]
        else:
            raise ValueError("Invalid load type. Choose 'vertical' or 'horizontal'.") 

        points_load_surface = points[mask]
        if output_type == 'numpy_array':
            points_load_surface = points_load_surface.cpu().numpy() 
        elif output_type == 'torch_tensor':
            points_load_surface = points_load_surface.to(device)  
        else:
            raise ValueError("Invalid output type. Choose 'numpy_array' or 'torch_tensor'.") 
        
        if self.Symmetry == True:
            points_load_surface = points_load_surface[points_load_surface[:, 0] >= 0]  # Filter points to keep only the right side if symmetry is applied
            points_load_surface = points_load_surface[points_load_surface[:, 1] >= 0]  # Filter points to keep only the upper side if symmetry is applied
    

        return points_load_surface


    #3. Sample points on all interfaces  
    def sample_points_from_all_interfaces(self, 
                                          num_points, 
                                          random_seed = None, 
                                          output_type = 'torch_tensor', 
                                          device = None): 
        num_boundary_pts = int(num_points * (2/3))
        num_obstacle_pts = int(num_points * (1/3))
        dirichlet_points = self.sample_points_on_dirichlet_boundary(num_boundary_pts,
                                                                    random_seed=random_seed,
                                                                    output_type=output_type,
                                                                    device=device)
        neumann_points = self.sample_points_from_pinn_interface(num_obstacle_pts,
                                                              random_seed=random_seed,
                                                              device=device) 
        # Concatenate the points from both interfaces
        points = torch.cat([dirichlet_points, neumann_points], dim=0) 
        if output_type == 'numpy_array':
            points = points.cpu().numpy()
        elif output_type == 'torch_tensor':
            points = torch.tensor(points, device=device)   
        return points 
    
    


    #4. Is on dirichlet boundary surface
    def is_on_dirichlet_boundary(self,points): 

        if isinstance(self.boundary_vertices, torch.Tensor):
            x_left = self.boundary_vertices[0][0].item()
            x_right = self.boundary_vertices[2][0].item()
            y_min = self.boundary_vertices[0][1].item()
            y_max = self.boundary_vertices[1][1].item()
        else:
            x_left = self.boundary_vertices[0][0]
            x_right = self.boundary_vertices[2][0]
            y_min = self.boundary_vertices[0][1]
            y_max = self.boundary_vertices[1][1]
        
        # Check if points are on the left or right boundary
        target_left  = torch.tensor(x_left,  dtype=points.dtype, device=points.device)
        target_right = torch.tensor(x_right, dtype=points.dtype, device=points.device)

        x_match_left  = torch.isclose(points[:, 0], target_left,  atol=1e-6, rtol=0)
        x_match_right = torch.isclose(points[:, 0], target_right, atol=1e-6, rtol=0)
        y_within      = (points[:, 1] >= y_min) & (points[:, 1] <= y_max)

        mask = (x_match_left  & y_within) | (x_match_right & y_within)

        # 4) Extract indices + points
        idx    = torch.nonzero(mask, as_tuple=False).squeeze(1) 
        points = points[idx]
        if isinstance(points, torch.Tensor): 
            points = points.to(points.device) 
            idx = idx.to(points.device) 
        else:
            points = points.cpu().numpy() 
        return points, idx 


    #5. Is on pinn interface surface
    def is_on_pinn_interface(self, points):
        if isinstance(self.obstacle_centroid, torch.Tensor):
            x_centroid = self.obstacle_centroid[0].item()
            y_centroid = self.obstacle_centroid[1].item() 
        else:
            x_centroid = self.obstacle_centroid[0]
            y_centroid = self.obstacle_centroid[1]

        dx = points[:, 0] - x_centroid
        dy = points[:, 1] - y_centroid
        dist_squared = dx**2 + dy**2
        target = torch.tensor(self.obstacle_radius ** 2,
                          dtype=dist_squared.dtype,
                          device=dist_squared.device)

        # boolean mask of points exactly on the circle
        mask = torch.isclose(dist_squared, target, atol=1e-6, rtol=0)

        # convert that mask into integer indices
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)  
        points = points[idx]
        if isinstance(points, torch.Tensor):
            points = points.to(points.device)
            idx = idx.to(points.device)  
        else:
            points = points.cpu().numpy()
        return points, idx 


    #5. Is on neumann boundary surface
    def is_on_neumann_boundary(self,points, load_type = 'vertical'): 
        surface_points, idx = self.is_on_pinn_interface(points)  # Get points on the obstacle boundary 

        if load_type == 'vertical':
            mask = surface_points[:, 1] >= self.obstacle_centroid[1]
        elif load_type == 'horizontal':
            mask = surface_points[:, 0] >= self.obstacle_centroid[0]
        else:
            raise ValueError("Invalid load type. Choose 'vertical' or 'horizontal'.")
        
        points = surface_points[mask]
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)

        if isinstance(points, torch.Tensor):
            points = points.to(points.device)
            idx = idx.to(points.device) 
        else:
            points = points.cpu().numpy()

        return points, idx 

    
    #6. Is on obstacle boundary surface 
    def is_on_obstacle_boundary(self, points):

        if isinstance(points, torch.Tensor):
            device = points.device 
        else:
            device = 'cpu'

        if isinstance(self.obstacle_centroid, torch.Tensor):
            x_centroid = self.obstacle_centroid[0].item()
            y_centroid = self.obstacle_centroid[1].item() 
        else:
            x_centroid = self.obstacle_centroid[0]
            y_centroid = self.obstacle_centroid[1]
        dx = points[:, 0] - x_centroid
        dy = points[:, 1] - y_centroid
        dist_squared = dx**2 + dy**2
        target = torch.tensor(self.obstacle_radius ** 2,
                          dtype=dist_squared.dtype,
                          device=dist_squared.device)

        # boolean mask of points exactly on the circle
        mask = torch.isclose(dist_squared, target, atol=1e-6, rtol=0)

        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)  
        points = points[idx]
        if isinstance(points, torch.Tensor):
            points = points.to(points.device)
            idx = idx.to(points.device) 
        else:
            points = points.cpu().numpy()  
        return points, idx  

    
    #6. Is inside prohibited region 
    def is_inside_prohibited_region(self,points):

        device = points.device if isinstance(points, torch.Tensor) else 'cpu' 

        if isinstance(self.obstacle_centroid, torch.Tensor):
            x_centroid = self.obstacle_centroid[0].item()
            y_centroid = self.obstacle_centroid[1].item() 
        else:
            x_centroid = self.obstacle_centroid[0]
            y_centroid = self.obstacle_centroid[1] 

        dx = points[:, 0] - x_centroid
        dy = points[:, 1] - y_centroid
        dist_squared = dx**2 + dy**2
        obst_points = points[dist_squared <= self.obstacle_radius ** 2]   

        # 2. Points behind the prescribed boundaries of the geometry 
        if isinstance(self.domain, torch.Tensor): 
            x_min_domain = self.domain[0].item()
            x_max_domain = self.domain[1].item()
            y_min_domain = self.domain[2].item()
            y_max_domain = self.domain[3].item()
        else:
            x_min_domain = self.domain[0]
            x_max_domain = self.domain[1]
            y_min_domain = self.domain[2]
            y_max_domain = self.domain[3]

        if isinstance(self.boundary_vertices, torch.Tensor): 
            x_min_left = self.boundary_vertices[0][0].item()
            x_max_right = self.boundary_vertices[2][0].item()
        else:
            x_min_left = self.boundary_vertices[0][0]
            x_max_right = self.boundary_vertices[2][0]

        left_mask = (points[:, 0] >= x_min_domain) & (points[:, 0] <= x_min_left)
        right_mask = (points[:, 0] >= x_max_right) & (points[:, 0] <= x_max_domain)
        behind_boundary_points = points[left_mask | right_mask]

        # 3. Above/below the prescribed thickness region near the vertical bars
        x_min_left_bar = x_min_left
        x_max_left_bar = x_min_left + self.prescribed_thickness_boundaries
        x_max_right_bar = x_max_right
        x_min_right_bar = x_max_right - self.prescribed_thickness_boundaries 

        y_min_top = y_min_domain + self.prescribed_thickness_obstacle
        y_max_top = y_max_domain
        y_min_bottom = y_min_domain

        above_left_mask = (points[:, 0] >= x_min_left_bar) & (points[:, 0] <= x_max_left_bar) & \
                        (points[:, 1] >= y_min_top) & (points[:, 1] <= y_max_top)
        above_right_mask = (points[:, 0] >= x_min_right_bar) & (points[:, 0] <= x_max_right_bar) & \
                        (points[:, 1] >= y_min_top) & (points[:, 1] <= y_max_top)
        above_prescribed_points = points[above_left_mask | above_right_mask]

        below_left_mask = (points[:, 0] >= x_min_left_bar) & (points[:, 0] <= x_max_left_bar) & \
                        (points[:, 1] >= y_min_bottom) & (points[:, 1] < y_min_top)
        below_right_mask = (points[:, 0] >= x_min_right_bar) & (points[:, 0] <= x_max_right_bar) & \
                        (points[:, 1] >= y_min_bottom) & (points[:, 1] < y_min_top)
        below_prescribed_points = points[below_left_mask | below_right_mask]

        # 4. Combine all prohibited zones 
        obst_idx                  = torch.nonzero(dist_squared <= self.obstacle_radius**2,
                                         as_tuple=False).squeeze(1)
        behind_boundary_idx       = torch.nonzero(left_mask | right_mask,
                                                as_tuple=False).squeeze(1)
        above_prescribed_idx      = torch.nonzero(above_left_mask | above_right_mask,
                                                as_tuple=False).squeeze(1)
        below_prescribed_idx      = torch.nonzero(below_left_mask | below_right_mask,
                                                as_tuple=False).squeeze(1)

        # union them into one global index list
        prohibited_idx = torch.unique(torch.cat([
            obst_idx,
            behind_boundary_idx,
            above_prescribed_idx,
            below_prescribed_idx
        ], dim=0))

        # grab the corresponding points once
        prohibited_points = points[prohibited_idx] 

        # temporary fix for issues with overlapping points   
        #-------------------------------------------------------
        thickness_pts,thickness_idx = self.is_inside_interface_thickness(prohibited_points)  
        neu_interface_pts, neu_interface_idx = self.is_on_pinn_interface(prohibited_points)  
        dir_interface_pts, dir_interface_idx = self.is_on_dirichlet_boundary(prohibited_points)  
     
        remove_mask = torch.zeros(prohibited_points.shape[0],
                                  dtype=torch.bool,
                                  device=prohibited_points.device
                                  )
        remove_mask[thickness_idx] = True  
        remove_mask[neu_interface_idx] = True  
        remove_mask[dir_interface_idx] = True  


        #2. Remove shared points from the prohibited points
        prohibited_points = prohibited_points[~remove_mask]
        prohibited_idx    = prohibited_idx[~remove_mask]
        # ----------------------------------------------------------- 

        if isinstance(points, torch.Tensor):
            prohibited_points = prohibited_points.to(device) 
            prohibited_idx = prohibited_idx.to(device) 

        return prohibited_points, prohibited_idx 
    

    #7. Is inside prescribed thickness region
    def is_inside_interface_thickness(self,points):

        # Find the points insde the prescribed thickness region around the obstacle 
        device = points.device if isinstance(points, torch.Tensor) else 'cpu' 
        inner_radius = self.obstacle_radius
        outer_radius = self.obstacle_radius + self.prescribed_thickness_obstacle 
        if isinstance(self.obstacle_centroid, torch.Tensor):
            x_centroid = self.obstacle_centroid[0].item()
            y_centroid = self.obstacle_centroid[1].item() 
        else:
            x_centroid = self.obstacle_centroid[0]
            y_centroid = self.obstacle_centroid[1] 
        dx = points[:, 0] - x_centroid
        dy = points[:, 1] - y_centroid
        dist_squared = dx**2 + dy**2
        idx_obst = torch.nonzero((dist_squared >= inner_radius**2) & 
                                 (dist_squared <= outer_radius**2), as_tuple=False).squeeze(1) 
 

        # Find the points inside the prescribed thickness region around the boundaries
        if isinstance(self.boundary_vertices, torch.Tensor):
            x_min_left = self.boundary_vertices[0][0].item()
            x_max_left = x_min_left + self.prescribed_thickness_boundaries
            x_max_right = self.boundary_vertices[2][0].item()
            x_min_right = x_max_right - self.prescribed_thickness_boundaries
            y_min = self.boundary_vertices[0][1].item()
            y_max = self.boundary_vertices[1][1].item()
        else:
            x_min_left = self.boundary_vertices[0][0]
            x_max_left = x_min_left + self.prescribed_thickness_boundaries
            x_max_right = self.boundary_vertices[2][0]
            x_min_right = x_max_right - self.prescribed_thickness_boundaries
            y_min = self.boundary_vertices[0][1]
            y_max = self.boundary_vertices[1][1]

        x = points[:, 0]
        y = points[:, 1]
        left_mask  = (x >= x_min_left) & (x <= x_max_left) & (y >= y_min) & (y <= y_max)
        right_mask = (x >= x_min_right) & (x <= x_max_right) & (y >= y_min) & (y <= y_max)
        idx_boundaries = torch.nonzero(left_mask | right_mask, as_tuple=False).squeeze(1)
        points_boundaries = points[idx_boundaries] 

  
        thickness_idx = torch.unique(torch.cat([
            idx_obst,
            idx_boundaries
        ], dim=0)) 

        thickness_pts = points[thickness_idx] 

        # temporary fix for issues with overlapping points  
        #----------------------------------------------------------
        

        neu_interface_pts, neu_interface_idx = self.is_on_pinn_interface(thickness_pts)  
        dir_interface_pts, dir_interface_idx = self.is_on_dirichlet_boundary(thickness_pts)  
        local_mask = torch.zeros(thickness_pts.shape[0],
                                 dtype=torch.bool,
                                 device=thickness_pts.device
                                 )
        local_mask[neu_interface_idx] = True  
        local_mask[dir_interface_idx] = True  
        # ----------------------------------------------------------- 

        thickness_pts = thickness_pts[~local_mask]  # Remove points that are also on the interfaces
        thickness_idx = thickness_idx[~local_mask]  # Remove points that are also on the interfaces 

        if isinstance(points, torch.Tensor):
            thickness_pts = thickness_pts.to(device)
            thickness_idx = thickness_idx.to(device) 

        return thickness_pts, thickness_idx 
     
    
    #8. Get Neumann surface normals 
    def get_neumann_surface_normals(self, neumann_points):

        if isinstance(neumann_points, torch.Tensor):
            device = neumann_points.device
            x_centroid = self.obstacle_centroid[0].item()
            y_centroid = self.obstacle_centroid[1].item()
        else:
            device = 'cpu'
            x_centroid = self.obstacle_centroid[0]
            y_centroid = self.obstacle_centroid[1] 

        x_points = neumann_points[:, 0]
        y_points = neumann_points[:, 1]

        dx = x_points - x_centroid
        dy = y_points - y_centroid
        norms = torch.sqrt(dx**2 + dy**2 + 1e-12) 

        dx = dx / norms
        dy = dy / norms

        normals = -1*torch.stack([dx, dy], dim=1)

        if isinstance(neumann_points, torch.Tensor):
            normals = normals.to(device)
        else:
            normals = normals.cpu().numpy()

        if isinstance(normals, torch.Tensor):
            norms = torch.norm(normals, dim=1, keepdim=True)
            normals = normals / norms
        else:
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / norms 

        return normals
    
    #9. Get dirichlet surface normals
    def get_dirichlet_surface_normals(self, dirichlet_points): 

        device = dirichlet_points.device if isinstance(dirichlet_points, torch.Tensor) else 'cpu'  

        if isinstance(self.boundary_vertices, torch.Tensor):
            x_left = self.boundary_vertices[0][0].item()
            x_right = self.boundary_vertices[2][0].item()
        else:
            x_left = self.boundary_vertices[0][0]
            x_right = self.boundary_vertices[2][0]  

        points_left = dirichlet_points[dirichlet_points[:, 0] == x_left] 
        points_right = dirichlet_points[dirichlet_points[:, 0] == x_right]  

        # Calculate normals for left and right boundaries
        if isinstance(dirichlet_points, torch.Tensor):
            normals_left = torch.tensor([-1.0, 0.0], device=device).repeat(points_left.shape[0], 1)
            normals_right = torch.tensor([1.0, 0.0], device=device).repeat(points_right.shape[0], 1)
            normals = torch.cat([normals_left, normals_right], dim=0) 
        else:
            normals_left = np.array([-1.0, 0.0]).reshape(1, 2).repeat(points_left.shape[0], axis=0)
            normals_right = np.array([1.0, 0.0]).reshape(1, 2).repeat(points_right.shape[0], axis=0)
            normals = np.concatenate([normals_left, normals_right], axis=0) 

        if isinstance(dirichlet_points, torch.Tensor):
            normals = normals.to(device) 

        return normals
    

    #10. Get all prescribed surface normals
    def get_all_prescribed_surface_normals(self, 
                                           num_points, 
                                           include_all = False, 
                                           type = 'numpy_array'): 
        if type == 'torch_tensor':
            device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            points = self.sample_points_from_all_interfaces(num_points, device=device, output_type='torch_tensor') 
            dirichlet_points,idx_dir = self.is_on_dirichlet_boundary(points)
            neumann_points,idx_neu = self.is_on_obstacle_boundary(points)   
            normals_dirichlet = self.get_dirichlet_surface_normals(dirichlet_points) 
            normals_neumann = self.get_neumann_surface_normals(neumann_points)  
            normals = torch.cat([normals_dirichlet, normals_neumann], dim=0).to(device)  
        else:
            points = self.sample_points_from_all_interfaces(num_points, output_type='numpy_array')  
            dirichlet_points,idx_dir = self.is_on_dirichlet_boundary(points)
            neumann_points,idx_neu = self.is_on_obstacle_boundary(points)   
            normals_dirichlet = self.get_dirichlet_surface_normals(dirichlet_points)
            normals_neumann = self.get_neumann_surface_normals(neumann_points)  
            normals = np.concatenate([normals_dirichlet, normals_neumann], axis=0)  

        return {
                "points": points,
                "neumann_idx": idx_neu,
                "dirichlet_idx": idx_dir,
                "neumann_normals": normals_neumann,
                "dirichlet_normals": normals_dirichlet
         }

    #11. Calculate SDF 
    def calculate_SDF(self, points):
        
        device = points.device if isinstance(points, torch.Tensor) else 'cpu' 

        # Define the geometry of the obstacle and boundaries 
        radius = self.obstacle_radius 
        if isinstance(self.obstacle_centroid, torch.Tensor):
            x_centroid = self.obstacle_centroid[0].item()
            y_centroid = self.obstacle_centroid[1].item() 
        else:
            x_centroid = self.obstacle_centroid[0]
            y_centroid = self.obstacle_centroid[1]
        if isinstance(self.boundary_vertices, torch.Tensor):
            x_min_left = self.boundary_vertices[0][0].item()
            x_max_right = self.boundary_vertices[2][0].item()
            y_min = self.boundary_vertices[0][1].item()
            y_max = self.boundary_vertices[1][1].item()
        else:
            x_min_left = self.boundary_vertices[0][0]
            x_max_right = self.boundary_vertices[2][0]
            y_min = self.boundary_vertices[0][1]
            y_max = self.boundary_vertices[1][1] 

        Rectangle = torch.tensor([[x_min_left, y_min],
                     [x_min_left, y_max],
                     [x_max_right, y_max],
                     [x_max_right, y_min]],
                     device=device)

        Circle = torch.tensor([x_centroid, y_centroid, radius], device=device, dtype=torch.float32) 

        if isinstance(points,torch.Tensor):
            device = points.device
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") 

        if isinstance(points, torch.Tensor):  
            x = points[:, 0]
            y = points[:, 1]
        else: 
            points = torch.tensor(points, dtype=torch.float32, device=device)
            x = points[:, 0]
            y = points[:, 1] 

        # Rectangle edges
        if isinstance(Rectangle, torch.Tensor): 
            x_min = Rectangle[0][0].item()
            x_max = Rectangle[2][0].item()
            y_min = Rectangle[0][1].item()
            y_max = Rectangle[1][1].item()
        else:
            x_min = Rectangle[0][0]
            x_max = Rectangle[2][0]
            y_min = Rectangle[0][1]
            y_max = Rectangle[1][1] 

        # Distance to rectangle (positive outside, negative inside)
        dx = torch.max(torch.stack([x_min - x, x - x_max, torch.zeros_like(x)]), dim=0).values
        dy = torch.max(torch.stack([y_min - y, y - y_max, torch.zeros_like(y)]), dim=0).values
        outside_rect = torch.sqrt(dx**2 + dy**2)
        inside_dx = torch.min(x - x_min, x_max - x)
        inside_dy = torch.min(y - y_min, y_max - y)
        inside_rect = -torch.min(inside_dx, inside_dy)
        sdf_rect = torch.where((x < x_min) | (x > x_max) | (y < y_min) | (y > y_max), outside_rect, inside_rect)

        # Distance to circular obstacle (positive outside, negative inside)
        cx, cy, r = Circle
        dist_to_center = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        sdf_circle = dist_to_center - r

        sdf_total = torch.max(sdf_rect, -sdf_circle)

        if isinstance(points, torch.Tensor): 
            sdf_total = sdf_total.to(device)
        else:
            sdf_total = sdf_total.cpu().numpy()

        return sdf_total.view(-1, 1) 