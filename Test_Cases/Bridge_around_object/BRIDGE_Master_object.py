import numpy as np
import torch 
from Test_Cases.Bridge_around_object.boundary_surfaces import Bridge_Interfaces


class BRIDGE_Master_Object:
    def __init__(self, Normalize = False, Symmetry = False): 
        self.Normalize = Normalize
        self.Symmetry = Symmetry  
        self.symmetry_axis = ['x', 'y']  
        self.symmetry_location = 0.0  
 
        self.test_case = "Bridge"  
         

        ''' Define the material properties and other physics properties: '''
        # -----------------------------------------------------------------------------------------------
        self.elastic_modulus = 2e5               # MPa(N/mm^2)  
        self.material_density = 7.85e-6          # kg/mm^3 
        self.yield_strength = 903.213            # MPa(N/mm^2)  
        self.poisson_ratio = 0.3                 # dimensionless  
        self.gravitational_acceleration = 9.81    # m/s^2 
        #-----------------------------------------------------------------------------------------------

        ''' Define the permissable design space / design domain / design envelope: ''' 
        # -----------------------------------------------------------------------------------------------
        self.domain = [0,200,0,100]   
        self.domain_volume = (self.domain[1] - self.domain[0]) * (self.domain[3] - self.domain[2]) # Area 
        self.domain_area = self.domain_volume  # Area of the domain 
        self.dim = 2  # 2D problem 

        self.edge_vertices = torch.tensor([
                                        [15, 10],  # x_min_left, y_min
                                        [15, 90],  # x_min_left, y_max
                                        [185, 10],  # x_max_right, y_min
                                        [185, 90],  # x_max_right, y_max    
                                        ])
        
        self.boundary_thickness = 10      #mm
        self.obstacle_radius    = 15      #mm
        self.obstacle_thickness = 10      #mm
        self.obstacle_centroid_x = self.domain[0] + (self.domain[1] - self.domain[0]) / 2  # Center of the obstacle in x
        self.obstacle_centroid_y = self.domain[2] + (self.domain[3] - self.domain[2]) / 2  # Center of the obstacle in y 
        self.obstacle_centroid = np.array([self.obstacle_centroid_x, self.obstacle_centroid_y])  # Center of the obstacle as numpy array 
        #------------------------------------------------------------------------------------------------

        ''' Define the load cases: ''' 
        # -----------------------------------------------------------------------------------------------
        self.applied_point_load = False             # There is no point load for any of the 3 load cases 
        self.applied_distributed_load = True        # Applied loads is a distributed load for each load case
        self.body_forces = False                    # Include body forces --> Gravitational forces on the structure

        # Vertical load case
        self.force_magnitude_vertical = 35600       # Newtons
        self.force_direction_vertical = np.array([0, 1],dtype=float)  # cartesian coordinates 
        self.force_vector_vertical = self.force_magnitude_vertical * self.force_direction_vertical

        # Horizontal load case
        self.force_magnitude_horizontal = 37800       # Newtons
        self.force_direction_horizontal = np.array([1,0],dtype=float) # cartesian coordinates
        self.force_vector_horizontal = self.force_magnitude_horizontal * self.force_direction_horizontal

        # Diagonal load case
        self.force_magnitude_diagonal = 42300       # Newtons
        load_angle = 0.837758                       # Radians -->  = 48 degrees  
        self.force_direction_diagonal = np.array([np.sin(load_angle), np.cos(load_angle)],dtype=float)  
        self.force_vector_diagonal = self.force_magnitude_diagonal * self.force_direction_diagonal 
        # -----------------------------------------------------------------------------------------------
        
        ''' Zero-center and normalize the domain: '''
        #------------------------------------------------------------------------------------------------- 
        
        self.domain_center = np.array([(self.domain[0] + self.domain[1]) / 2,
                                       (self.domain[2] + self.domain[3]) / 2])

        self.domain_extents = [self.domain[1] - self.domain[0],
                               self.domain[3] - self.domain[2]] 
        self.max_extent = max(self.domain_extents)
        self.domain_scaling_factor = 2 / self.max_extent 

        corner_min = np.array([self.domain[0], self.domain[2]])
        corner_max = np.array([self.domain[1], self.domain[3]])
        norm_min = (corner_min - self.domain_center) * self.domain_scaling_factor
        norm_max = (corner_max - self.domain_center) * self.domain_scaling_factor


        if self.Normalize == True: 
            self.domain = np.array([norm_min[0], norm_max[0], norm_min[1], norm_max[1]]) 
            self.domain_volume = self.domain_volume * self.domain_scaling_factor ** 2
            self.domain_area = self.domain_area * self.domain_scaling_factor ** 2 
            self.edge_vertices = (self.edge_vertices - self.domain_center) * self.domain_scaling_factor 
            self.obstacle_centroid = (self.obstacle_centroid - self.domain_center) * self.domain_scaling_factor
            self.obstacle_radius *= self.domain_scaling_factor
            self.boundary_thickness *= self.domain_scaling_factor
            self.obstacle_thickness *= self.domain_scaling_factor 

            if self.Symmetry == True:
                #1. Reduce the domain to the first quadrant
                self.domain[0] = 0
                self.domain[2] = 0 

                #2. Reduce the domain volume to the first quadrant 
                self.domain_volume = self.domain_volume / 4 

                #3. Reduce the applied forces by 2 
                self.force_magnitude_horizontal /= 2
                self.force_magnitude_vertical /= 2
                self.force_magnitude_diagonal /= 2
                self.force_vector_vertical   = self.force_magnitude_vertical * self.force_direction_vertical
                self.force_vector_horizontal = self.force_magnitude_horizontal * self.force_direction_horizontal
                self.force_vector_diagonal   = self.force_magnitude_diagonal * self.force_direction_diagonal  

        # -----------------------------------------------------------------------------------------------


    ''' Function to create the interface object: ''' 
    # ---------------------------------------------------------------------------------------------- 
    def create_interfaces(self):
        self.interfaces = Bridge_Interfaces(
                                            domain=self.domain,
                                            obstacle_centroid=self.obstacle_centroid,
                                            obstacle_radius=self.obstacle_radius,
                                            boundary_vertices=self.edge_vertices,
                                            prescribed_thickness_obstacle=self.obstacle_thickness,
                                            prescribed_thickness_boundaries=self.boundary_thickness,
                                            Symmetry = self.Symmetry
                                            )


    ''' Function to enforce dirichlet boundary conditions: '''
    # ----------------------------------------------------------------------------------------------
    def enforce_dirichlet_boundary_conditions(self, coords):
        device = coords.device if isinstance(coords, torch.Tensor) else None
        if isinstance(self.edge_vertices, torch.Tensor): 
            x_left = self.edge_vertices[0, 0].item()
            x_right = self.edge_vertices[2, 0].item()  
        else:
            x_left = self.edge_vertices[0, 0]
            x_right = self.edge_vertices[2, 0] 

        if isinstance(coords, torch.Tensor):
            distances_left = torch.abs(coords[:, 0] - x_left)
            distances_right = torch.abs(coords[:, 0] - x_right)
            multiplier_left = 1 / (1 + torch.exp(-25 * distances_left))
            multiplier_right = 1 / (1 + torch.exp(-25 * distances_right))
            multiplier = multiplier_left * multiplier_right
            if device is not None:
                 multiplier = multiplier.to(device)
        else:
            distances_left = np.abs(coords[:, 0] - x_left)
            distances_right = np.abs(coords[:, 0] - x_right)
            multiplier_left = 1 / (1 + np.exp(-25 * distances_left))
            multiplier_right = 1 / (1 + np.exp(-25 * distances_right))
            multiplier = multiplier_left * multiplier_right 

        return multiplier