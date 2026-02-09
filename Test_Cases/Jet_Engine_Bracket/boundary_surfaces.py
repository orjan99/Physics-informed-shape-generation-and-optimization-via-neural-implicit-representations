import numpy as np
import torch 
from functools import wraps 

#--------------------- funtion to get all the interfaces for the Jet Engine Bracket -------------------
class Jet_engine_bracket_interfaces:
  """
  This class defines all interfaces for the jet engine bracket design. It includes the following:
  1. Bolt interfaces
  2. PINN interfaces
  3. The function to sample points on the surface of the interfaces --> Used to enforce dirichlet BCs in the PINN model
  4. The function to extract the points inside the interfaces --> Used for geometric constraints in the GINN model 
  5. The function to generate points on the load surface of the interfaces --> Used to enforce Neumann BCs in the PINN model
  6. The function to extract the points inside the interface thickness --> Used for geometric constraints in the GINN model
  7. The function to extract the points inside the counterbore interface thickness --> Used for geometric constraints in the GINN model

  Note: The radius of the interfaces are fixed and cannot be changed for this case --> Determined by the JEB challange  
  """
  def __init__(self, 
           domain,

           # Location of the bolt interfaces
           centroid_bolt_interface_1,
           centroid_bolt_interface_2,
           centroid_bolt_interface_3,
           centroid_bolt_interface_4,
           centroid_pinn_interface_1,
           centroid_pinn_interface_2,

          # Parameters for the bolt interface
           prescribed_bolt_interface_thickness = True, 
           inner_radius_bolt = 10.287/2, # mm
           prescribed_bolt_interface_thickness_value = 6.9431, 
           prescribed_bolt_interface_depth = 10, # mm 
           counterbore = True,
           counterbore_radius = 14.1732/2, # mm
           counterbore_interface_thickness = 5,
           counterbore_depth = 15,

          # Parameters for the PINN interface
           pinn_interface_radius = 19.05/2, # mm
           prescribed_radial_thickness_value = 5,
           prescribed_minimum_width_value = 5, 
           prescribed_sharp_edges = True,
     
          # Symmetry conditioning:
           Symmetry = False): 
           
         
        

     
     # 1. Assign the input parameters to the class attributes 
     self.domain = domain
     self.centroid_bolt_interface_1 = centroid_bolt_interface_1
     self.centroid_bolt_interface_2 = centroid_bolt_interface_2
     self.centroid_bolt_interface_3 = centroid_bolt_interface_3
     self.centroid_bolt_interface_4 = centroid_bolt_interface_4
     self.centroid_pinn_interface_1 = centroid_pinn_interface_1
     self.centroid_pinn_interface_2 = centroid_pinn_interface_2
     self.prescribed_bolt_interface_thickness = prescribed_bolt_interface_thickness
     self.prescribed_bolt_interface_thickness_value = prescribed_bolt_interface_thickness_value
     self.prescribed_bolt_interface_depth = prescribed_bolt_interface_depth
     self.counterbore = counterbore
     self.counterbore_interface_thickness = counterbore_interface_thickness
     self.counterbore_depth = counterbore_depth
     self.prescribed_radial_thickness_value = prescribed_radial_thickness_value 
     self.prescribed_minimum_width_value = prescribed_minimum_width_value
     self.prescribed_sharp_edges = prescribed_sharp_edges
     self.inner_radius_bolt = inner_radius_bolt
     self.counterbore_radius = counterbore_radius
     self.pinn_interface_radius = pinn_interface_radius
     self.Symmetry = Symmetry

     # 2. Create the bolt interface objects
     self.bolt_interface_1 = Bolt_interface_cylindrical_boundary(
          centroid_bolt_interface_1,
          domain,
          prescribed_interface_thickness = prescribed_bolt_interface_thickness,
          prescribed_interface_thickness_value = prescribed_bolt_interface_thickness_value,
          prescribed_interface_depth = prescribed_bolt_interface_depth,
          counterbore = counterbore,
          counterbore_interface_thickness = counterbore_interface_thickness,
          counterbore_depth = counterbore_depth,
          inner_radius_bolt = self.inner_radius_bolt,
          inner_radius_counterbore = self.counterbore_radius,
          )
     
     self.bolt_interface_2 = Bolt_interface_cylindrical_boundary(
            centroid_bolt_interface_2,
            domain,
            prescribed_interface_thickness = prescribed_bolt_interface_thickness,
            prescribed_interface_thickness_value = prescribed_bolt_interface_thickness_value,
            prescribed_interface_depth = prescribed_bolt_interface_depth,
            counterbore = counterbore,
            counterbore_interface_thickness = counterbore_interface_thickness,
            counterbore_depth = counterbore_depth,
            inner_radius_bolt = self.inner_radius_bolt,
            inner_radius_counterbore = self.counterbore_radius,
            )
     
     self.bolt_interface_3 = Bolt_interface_cylindrical_boundary(
          centroid_bolt_interface_3,
          domain,
          prescribed_interface_thickness = prescribed_bolt_interface_thickness,
          prescribed_interface_thickness_value = prescribed_bolt_interface_thickness_value,
          prescribed_interface_depth = prescribed_bolt_interface_depth,
          counterbore = counterbore,
          counterbore_interface_thickness = counterbore_interface_thickness,
          counterbore_depth = counterbore_depth,
          inner_radius_bolt = self.inner_radius_bolt,
          inner_radius_counterbore = self.counterbore_radius,
          )
     
     self.bolt_interface_4 = Bolt_interface_cylindrical_boundary(
          centroid_bolt_interface_4,
          domain,
          prescribed_interface_thickness = prescribed_bolt_interface_thickness,
          prescribed_interface_thickness_value = prescribed_bolt_interface_thickness_value,
          prescribed_interface_depth = prescribed_bolt_interface_depth,
          counterbore = counterbore,
          counterbore_interface_thickness = counterbore_interface_thickness,
          counterbore_depth = counterbore_depth,
          inner_radius_bolt = self.inner_radius_bolt,
          inner_radius_counterbore = self.counterbore_radius,
          )
     
      # 3. Create the PINN interface objects
     self.pinn_interface = Pinn_interface_cylindrical_boundary(
          centroid_pinn_interface_1,
          centroid_pinn_interface_2,
          domain,
          prescribed_radial_thickness_value = prescribed_radial_thickness_value,
          prescribed_minimum_width_value = prescribed_minimum_width_value,
          prescribed_sharp_edges = prescribed_sharp_edges,
          inner_radius = pinn_interface_radius,
          ) 
     
  # Create a function to allow the user to use pytorch tensors as inputs instead of numpy arrays and return outputs as torch tensors
  #----------------------------------------------------------------------------------------------------------------------
  def pytorch_conversion_extract_points(function):
    """
    Converts the input and output of a function to/from torch.Tensor if the input
    is a tensor. Assumes the wrapped function always returns (points_np, idx_list).

    This is needed to use the functions within my PINN and GINN models, which use torch tensors  
    """
    @wraps(function)
    def wrapper(self,points, *args, **kwargs):
        # 1. Check if the input is a torch tensor
        input_is_tensor = isinstance(points, torch.Tensor)

        # 2. If it's a tensor, stash device/dtype and convert to numpy
        if input_is_tensor:
            device = points.device
            dtype  = points.dtype
            points_numpy = points.detach().cpu().numpy() 
        else:
            points_numpy = points

        # 3. Call the original function on the numpy array
        points_numpy_out, idx = function(self,points_numpy, *args, **kwargs)

        # 4. If the input was a tensor, convert outputs back to tensor
        if input_is_tensor:
            # points_out → same device & dtype
            points_output = torch.from_numpy(np.asarray(points_numpy_out)).to(device=device, dtype=dtype)
            # idx → LongTensor on same device
            idx_output = torch.tensor(idx, dtype=torch.long, device=device)
            return points_output, idx_output

        # 5. Otherwise, just return the raw numpy outputs
        return points_numpy_out, idx

    return wrapper   
  # --------------------------------------------------------------------------------------------------------------------- 
  
 
     
  # 4. Function to find points inside the prohibited regions of the design space --> For GINN model
  @pytorch_conversion_extract_points  
  def is_inside_prohibited_region(self,points):
        """
        This function returns all points that are inside the prohibited regions of the design space, and the indices of those points
        The prohibited regions are defined by the bolt interfaces and the PINN interfaces 
        """
        # 1. Extract the points inside the bolt interface
        points_bolt_interface_1, idx1 = self.bolt_interface_1.extract_inside_interface_points(points)
        points_bolt_interface_2, idx2 = self.bolt_interface_2.extract_inside_interface_points(points)
        points_bolt_interface_3, idx3 = self.bolt_interface_3.extract_inside_interface_points(points)
        points_bolt_interface_4, idx4 = self.bolt_interface_4.extract_inside_interface_points(points)

        # 2. Extract the points inside the PINN interface
        points_pinn_interface, idx5 = self.pinn_interface.extract_inside_interface_points(points)

        # 3. Combine all the points inside the prohibited regions
        points_inside_prohibited_region = np.concatenate((points_bolt_interface_1,points_bolt_interface_2,
                                                      points_bolt_interface_3,points_bolt_interface_4,
                                                      points_pinn_interface))
        
        # 4. Combine all the indices of the points inside the prohibited regions
        idx_inside = np.concatenate((idx1,idx2,idx3,idx4,idx5))

        # 5. Remove duplicates 
        unique_idx = np.unique(idx_inside).astype(np.int64)  
        points_inside_prohibited_region = points[unique_idx] 
        idx_inside_prohibited_region = unique_idx  
 

         # temporary fix for issues with overlapping points  
        #----------------------------------------------------------

        #1. Identify points inside the prescribed thickness region around the obstacle and boundaries 
        thickness_pts,thickness_idx = self.is_inside_interface_thickness(points_inside_prohibited_region)  
        neu_interface_pts, neu_interface_idx = self.is_on_pinn_interface(points_inside_prohibited_region)  
        dir_interface_pts, dir_interface_idx = self.is_on_dirichlet_boundary(points_inside_prohibited_region)  
        
        neu_interface_idx = np.array(neu_interface_idx, dtype=np.int64)
        dir_interface_idx = np.array(dir_interface_idx, dtype=np.int64)
        remove_mask = np.zeros(len(points_inside_prohibited_region), dtype=bool)  
        remove_mask[thickness_idx] = True  
        remove_mask[neu_interface_idx] = True  
        remove_mask[dir_interface_idx] = True  

        #2. Remove shared points from the prohibited points
        points_inside_prohibited_region = points_inside_prohibited_region[~remove_mask] 


        #3. Update the indices of the points inside the prohibited region
        idx_inside_prohibited_region = idx_inside_prohibited_region[~remove_mask] 

        # ----------------------------------------------------------- 

        return points_inside_prohibited_region, idx_inside_prohibited_region


     
  # 5. Function to find points within the interface thickness --> For GINN model
  @pytorch_conversion_extract_points 
  def is_inside_interface_thickness(self,points):
        """
        This function returns all points that are inside the prescribed thickness region of the interfaces and the indices of those points
        """
        # 1. Extract the points inside the bolt interface thickness
        points_bolt_interface_1, idx1 = self.bolt_interface_1.extract_inside_interface_thickness_points(points)
        points_bolt_interface_2, idx2 = self.bolt_interface_2.extract_inside_interface_thickness_points(points)
        points_bolt_interface_3, idx3 = self.bolt_interface_3.extract_inside_interface_thickness_points(points)
        points_bolt_interface_4, idx4 = self.bolt_interface_4.extract_inside_interface_thickness_points(points)

        # 2. Extract the points inside the counterbore interface thickness
        if self.counterbore == True:
            points_bolt_interface_1_counterbore, idx_1 = self.bolt_interface_1.extract_counterbore_interface_thickness_points(points)
            points_bolt_interface_2_counterbore, idx_2 = self.bolt_interface_2.extract_counterbore_interface_thickness_points(points)
            points_bolt_interface_3_counterbore, idx_3 = self.bolt_interface_3.extract_counterbore_interface_thickness_points(points)
            points_bolt_interface_4_counterbore, idx_4 = self.bolt_interface_4.extract_counterbore_interface_thickness_points(points)
            points_bolt_interface_1 = np.concatenate((points_bolt_interface_1,points_bolt_interface_1_counterbore))
            points_bolt_interface_2 = np.concatenate((points_bolt_interface_2,points_bolt_interface_2_counterbore))
            points_bolt_interface_3 = np.concatenate((points_bolt_interface_3,points_bolt_interface_3_counterbore))
            points_bolt_interface_4 = np.concatenate((points_bolt_interface_4,points_bolt_interface_4_counterbore))
            idx1 = np.concatenate((idx1,idx_1))
            idx2 = np.concatenate((idx2,idx_2))
            idx3 = np.concatenate((idx3,idx_3))
            idx4 = np.concatenate((idx4,idx_4)) 

        # 3. Extract the points inside the PINN interface thickness
        points_pinn_interface, idx5 = self.pinn_interface.extract_inside_interface_thickness_points(points)

        # 4. Combine all the points inside the interface thickness
        points_inside_interface_thickness = np.concatenate((points_bolt_interface_1,
                                                            points_bolt_interface_2,
                                                            points_bolt_interface_3,
                                                            points_bolt_interface_4,
                                                             points_pinn_interface))
        
        # 5. Combine all the indices of the points inside the interface thickness
        idx_inside = np.concatenate((idx1,idx2,idx3,idx4,idx5))

        # 6. Remove duplicates --> Points could be inside multiple interfaces 
        unique_idx = np.unique(idx_inside).astype(np.int64)  
        points_inside_interface_thickness = points[unique_idx]
        idx_inside_interface_thickness = unique_idx 
  
         # temporary fix for issues with overlapping points  
        #----------------------------------------------------------
        #1. Identify points inside the prescribed thickness region around the obstacle and boundaries 
        neu_interface_pts, neu_interface_idx = self.is_on_pinn_interface(points_inside_interface_thickness) 
        dir_interface_pts, dir_interface_idx = self.is_on_dirichlet_boundary(points_inside_interface_thickness)  
        
        #2. Remove points that are on the PINN interface or Dirichlet boundary
        neu_interface_idx = np.array(neu_interface_idx, dtype=np.int64)
        dir_interface_idx = np.array(dir_interface_idx, dtype=np.int64)
        remove_idx = np.zeros(len(points_inside_interface_thickness), dtype=bool) 
        remove_idx[neu_interface_idx] = True  
        remove_idx[dir_interface_idx] = True  

        #2. Remove shared points from the prohibited points
        points_inside_interface_thickness = points_inside_interface_thickness[~remove_idx]
  

        #3. Update the indices of the points inside the prohibited region
        idx_inside_interface_thickness = idx_inside_interface_thickness[~remove_idx] 
        
        # ----------------------------------------------------------- 

        return points_inside_interface_thickness, idx_inside_interface_thickness
 
     
  # 6. Function to sample points on the bolt interface --> dirichlet boundaries 
  def sample_points_on_dirichlet_boundary(self, num_points, random_seed = None, output_type = 'numpy_array',
                                          device = None, #needed for consistency with other interfaces  
                                          ): 
        """
        This function generates a set of random points on the surface the of the bolt interface 
        """
        # 1. Sample points from each bolt interface
        points_bolt_interface_1 = self.bolt_interface_1.sample_points_from_bolt_interface(num_points, random_seed)
        points_bolt_interface_2 = self.bolt_interface_2.sample_points_from_bolt_interface(num_points, random_seed)
        points_bolt_interface_3 = self.bolt_interface_3.sample_points_from_bolt_interface(num_points, random_seed)
        points_bolt_interface_4 = self.bolt_interface_4.sample_points_from_bolt_interface(num_points, random_seed)

        # 2. Combine all the points from the bolt interfaces
        points_on_bolt_interface = np.concatenate((points_bolt_interface_1,
                                                   points_bolt_interface_2,
                                                   points_bolt_interface_3,
                                                   points_bolt_interface_4))   
        
        if output_type == 'torch_tensor':
            points_on_bolt_interface = torch.from_numpy(points_on_bolt_interface).float() 
            if self.Symmetry == True: 
                points_on_bolt_interface = points_on_bolt_interface[points_on_bolt_interface[:,1] >= 0.0] 
            if device is not None:
                points_on_bolt_interface = points_on_bolt_interface.to(device) 
            return points_on_bolt_interface
        
        elif output_type == 'numpy_array':
            if self.Symmetry == True: 
                points_on_bolt_interface = points_on_bolt_interface[points_on_bolt_interface[:,1] >= 0.0] 
            return points_on_bolt_interface 
        else:
            raise ValueError("Invalid output type. Choose from 'numpy_array' or 'torch_tensor'.")    


     
  # 7. Function to sample points on the load surface of the PINN interface --> Neumann boundaries  
  def sample_points_on_neumann_boundary(self, num_points, load_type: str, output_type = 'numpy_array',
                                         device = None, #needed for consistency with other interfaces 
                                         ): 
      """
      This function generates a set of random points on the portion of the PINN interface that has an applied load. 
      The region of the surface subjected to forces depend on the direction of the applied load 
      """
      # 1. Sample points from the PINN interface 
      points_pinn_interface = self.pinn_interface.sample_points_on_load_surface(num_points, load_type)
      if output_type == 'torch_tensor':
          points_pinn_interface = torch.from_numpy(points_pinn_interface).float() 
          if self.Symmetry == True:  
              points_pinn_interface = points_pinn_interface[points_pinn_interface[:,1] >= 0.0]
          if device is not None:
                points_pinn_interface = points_pinn_interface.to(device) 
          return points_pinn_interface
      elif output_type == 'numpy_array':
          if self.Symmetry == True: 
                points_pinn_interface = points_pinn_interface[points_pinn_interface[:,1] >= 0.0]
          return points_pinn_interface
      else:
          raise ValueError("Invalid output type. Choose from 'numpy_array' or 'torch_tensor'.") 
      
    
  # 8. Function to find points on the surface of the bolt interface
  @pytorch_conversion_extract_points 
  def is_on_bolt_interface(self,points):
        """
        This function returns all points that are on the surface of the bolt interface
        """
        # 1. Extract the points on the surface of the bolt interface
        points_bolt_interface_1, idx1 = self.bolt_interface_1.extract_interface_surface_points(points)
        points_bolt_interface_2, idx2 = self.bolt_interface_2.extract_interface_surface_points(points)
        points_bolt_interface_3, idx3 = self.bolt_interface_3.extract_interface_surface_points(points)
        points_bolt_interface_4, idx4 = self.bolt_interface_4.extract_interface_surface_points(points)

        # 2. Combine all the points on the surface of the bolt interface
        points_on_bolt_interface = np.concatenate((points_bolt_interface_1,
                                                   points_bolt_interface_2,
                                                   points_bolt_interface_3,
                                                   points_bolt_interface_4))  
        # 3. Combine all the indices of the points on the surface of the bolt interface
        idx_on_bolt_interface = np.concatenate((idx1,idx2,idx3,idx4))
        return points_on_bolt_interface, idx_on_bolt_interface
  

  # 9. Function to find points on the surface of the PINN interface
  @pytorch_conversion_extract_points 
  def is_on_pinn_interface(self,points):
        """
        This function returns all points that are on the surface of the PINN interface
        """
        # 1. Extract the points on the surface of the PINN interface
        points_pinn_interface, idx = self.pinn_interface.extract_interface_surface_points(points)


        return points_pinn_interface, idx
  
  # 10. Function to extract the points on the dirichlet boundary
  @pytorch_conversion_extract_points 
  def is_on_dirichlet_boundary(self,points):
           
     points, idx = self.is_on_bolt_interface(points)
     return points, idx 

  # 11. Function to extract the points on the neumann boundary -- based on the load type
  @pytorch_conversion_extract_points 
  def is_on_neumann_boundary(self,points, load_type = 'vertical'): 
      points, idx = self.is_on_pinn_interface(points)
      if load_type == 'vertical':
            mask = (points[:,2] >= self.centroid_pinn_interface_1[2])
      elif load_type == 'horizontal':
            mask = (points[:,0] <= self.centroid_pinn_interface_1[0])
      elif load_type == 'diagonal':
            theta = np.deg2rad(48)
            theta = theta + np.pi/2
            vx, vz = -np.cos(theta), -np.sin(theta)
            mask = ((points[:,0] - self.centroid_pinn_interface_1[0])*vx + (points[:,2] - self.centroid_pinn_interface_1[2])*vz) <= 0
      else:
            raise ValueError("Invalid load type. Choose from 'vertical', 'horizontal', or 'diagonal'.")
      local_idx = np.where(mask)[0]
      points = points[mask]
      idx = np.array(idx)[local_idx].tolist()
      return points, idx
  
  # 12. Function to sample points from all interfaces
  def sample_points_from_all_interfaces(self, num_points, random_seed = None, output_type = 'numpy_array',device = None): 
      num_points_bolts = int(0.67*num_points)
      num_points_pinn = int(0.33*num_points) 
      points_bolt_interface = self.sample_points_on_dirichlet_boundary(num_points_bolts, random_seed)
      points_pinn_interface = self.pinn_interface.sample_points_from_pinn_interface(num_points_pinn, random_seed)
      if self.Symmetry == True:
          points_pinn_interface = points_pinn_interface[points_pinn_interface[:,1] >= 0.0] 
      points = np.concatenate((points_bolt_interface, points_pinn_interface))

      if output_type == 'torch_tensor':
          points = torch.from_numpy(points).float()  
          if device is not None:
              points = points.to(device) 
          return points
      elif output_type == 'numpy_array':
          return points
      else:
          raise ValueError("Invalid output type. Choose from 'numpy_array' or 'torch_tensor'.")
      
# 12. Function to sample points from the inner radius of the counterbore interface  
  def sample_points_from_counterbore_interface(self, num_points, random_seed = None, output_type = 'numpy_array'):
      """
      This function generates a set of random points on the inner radius of the counterbore interface
      """
      # 1. Sample points from each bolt interface
      points_bolt_interface_1 = self.bolt_interface_1.sample_points_from_counterbore_interface(num_points, random_seed)
      points_bolt_interface_2 = self.bolt_interface_2.sample_points_from_counterbore_interface(num_points, random_seed)
      points_bolt_interface_3 = self.bolt_interface_3.sample_points_from_counterbore_interface(num_points, random_seed)
      points_bolt_interface_4 = self.bolt_interface_4.sample_points_from_counterbore_interface(num_points, random_seed)

      # 2. Combine all the points from the bolt interfaces 
      points_on_bolt_interface = np.concatenate((points_bolt_interface_1,
                                                   points_bolt_interface_2,
                                                   points_bolt_interface_3,
                                                   points_bolt_interface_4))   
      if self.Symmetry == True:
          points_on_bolt_interface = points_on_bolt_interface[points_on_bolt_interface[:,1] >= 0.0] 
      # 3. if the output type is a torch tensor, convert the points to a torch tensor
      if output_type == 'torch_tensor':
          points_on_bolt_interface = torch.from_numpy(points_on_bolt_interface).float() 
          return points_on_bolt_interface
        
      elif output_type == 'numpy_array':
          return points_on_bolt_interface
      else:
          raise ValueError("Invalid output type. Choose from 'numpy_array' or 'torch_tensor'.")
      
# 12. Function to extract the points of the cylindracl boundary of the counterbore interface
  @pytorch_conversion_extract_points
  def is_on_counterbore_cylindrical_interface(self, points):
      """
      This function returns all points that are on the cylindrical boundary of the counterbore interface
      """
      # 1. Extract the points on the cylindrical boundary of the bolt interface
      points_bolt_interface_1, idx1 = self.bolt_interface_1.extract_counterbore_cylindrical_surface_points(points)
      points_bolt_interface_2, idx2 = self.bolt_interface_2.extract_counterbore_cylindrical_surface_points(points)
      points_bolt_interface_3, idx3 = self.bolt_interface_3.extract_counterbore_cylindrical_surface_points(points)
      points_bolt_interface_4, idx4 = self.bolt_interface_4.extract_counterbore_cylindrical_surface_points(points)

      # 2. Combine all the points on the cylindrical boundary of the bolt interface
      points_on_bolt_interface = np.concatenate((points_bolt_interface_1, 
                                                   points_bolt_interface_2,
                                                   points_bolt_interface_3,
                                                   points_bolt_interface_4))  
      # 3. Combine all the indices of the points on the cylindrical boundary of the bolt interface
      idx_on_bolt_interface = np.concatenate((idx1, idx2, idx3, idx4))
      return points_on_bolt_interface, idx_on_bolt_interface
  
# 12. Function to extract points from the hollow disc bounary of the counterbore interface
  @pytorch_conversion_extract_points
  def is_on_counterbore_hollow_disc_interface(self, points):
      """
      This function returns all points that are on the hollow disc boundary of the counterbore interface
      """
      # 1. Extract the points on the hollow disc boundary of the bolt interface
      points_bolt_interface_1, idx1 = self.bolt_interface_1.extract_counterbore_disc_surface_points(points)
      points_bolt_interface_2, idx2 = self.bolt_interface_2.extract_counterbore_disc_surface_points(points)
      points_bolt_interface_3, idx3 = self.bolt_interface_3.extract_counterbore_disc_surface_points(points)
      points_bolt_interface_4, idx4 = self.bolt_interface_4.extract_counterbore_disc_surface_points(points)

      # 2. Combine all the points on the hollow disc boundary of the bolt interface
      points_on_bolt_interface = np.concatenate((points_bolt_interface_1,
                                                   points_bolt_interface_2,
                                                   points_bolt_interface_3,
                                                   points_bolt_interface_4))  
      # 3. Combine all the indices of the points on the hollow disc boundary of the bolt interface
      idx_on_bolt_interface = np.concatenate((idx1, idx2, idx3, idx4))
      return points_on_bolt_interface, idx_on_bolt_interface

    
      
      
# 13. Function to get the surface normals for the points on the Neumann boundary of the PINN interface 
  def get_neumann_surface_normals(self, neumann_points):
      """
      This function returns the surface normals for the points on the Neumann boundary of the PINN interface
      """
      # Check if neumann_points is a torch tensor
      if isinstance(neumann_points, torch.Tensor):
          y = neumann_points[:,1]  
          z = self.centroid_pinn_interface_1[2]*torch.ones_like(y) # z coordinate is constant for the PINN interface 
          x = self.centroid_pinn_interface_1[0]*torch.ones_like(y) # x coordinate is constant for the PINN interface

          # project the points onto the central axis of the PINN interface
          points_on_axis = torch.stack((x, y, z), dim=1)  # Stack the coordinates to form points on the axis 

          # Compute the surface normals as the difference between the Neumann points and the points on the axis
          surface_normals_torch = neumann_points - points_on_axis
          surface_normals_torch = -1 * surface_normals_torch  # Reverse the direction of the normals to point outward 

          # Normalize the surface normals
          norms = torch.norm(surface_normals_torch, dim=1, keepdim=True)
          surface_normals = surface_normals_torch / norms  # Normalize the surface normals   
          return surface_normals
      else: 
          x = self.centroid_pinn_interface_1[0] * np.ones_like(neumann_points[:, 1])  # x-coordinates are constant at the centroid's x 
          y = neumann_points[:, 1]  # y-coordinates are constant at the centroid's y
          z = self.centroid_pinn_interface_1[2] * np.ones_like(x)  # z-coordinates are constant at the centroid's z

          points_on_axis = np.column_stack((x, y, z))  # Combine x, y, z into a single array    

          normals = neumann_points - points_on_axis
          normals = -1 * normals 
          


          # Normalize the normals
          norms = np.linalg.norm(normals, axis=1, keepdims=True)
          normals = normals / norms  # Normalize each normal vector  

          surface_normals = normals  

          return surface_normals  
      
# 14: Function to get the surface normals for the points on the Dirichlet boundary of the bolt interfaces  
  def get_dirichlet_surface_normals(self, dirichlet_points):  
      """
      This function returns the surface normals at the points on the Dirichlet boundary of the bolt interfaces
      """

      centroids = [self.centroid_bolt_interface_1,
                   self.centroid_bolt_interface_2,
                   self.centroid_bolt_interface_3,
                   self.centroid_bolt_interface_4]

      # Check if dirichlet_points is a torch tensor
      if isinstance(dirichlet_points, torch.Tensor):
          
          # Calculate the distance to each centroid --> used to determine which bolt interface the point belongs to
          distances = [torch.norm(dirichlet_points - torch.tensor(centroid, dtype=dirichlet_points.dtype, device=dirichlet_points.device), dim=1) for centroid in centroids]
          distances_to_centroid_1 = distances[0]
          distances_to_centroid_2 = distances[1]
          distances_to_centroid_3 = distances[2]
          distances_to_centroid_4 = distances[3]

          # Find the minimum distance to determine the closest centroid
          min_distances, closest_centroid_indices = torch.min(torch.stack((distances_to_centroid_1, 
                                                                 distances_to_centroid_2,
                                                                 distances_to_centroid_3,
                                                                 distances_to_centroid_4)), dim=0)

          centroid = torch.stack([torch.tensor(centroids[i], dtype=dirichlet_points.dtype, device=dirichlet_points.device) for i in closest_centroid_indices])

          z = dirichlet_points[:,2]
          y = centroid[:, 1]
          x = centroid[:, 0]

          # project the points onto the central axis of the bolt interface
          points_on_axis = torch.stack((x, y, z), dim=1)  # Stack the coordinates to form points on the axis

          # Compute the surface normals as the difference between the Dirichlet points and the points on the axis
          surface_normals_torch = dirichlet_points - points_on_axis
          surface_normals_torch = -1 * surface_normals_torch  # Reverse the direction of the normals to point outward
          # Normalize the surface normals
          norms = torch.norm(surface_normals_torch, dim=1, keepdim=True)
          surface_normals = surface_normals_torch / norms  # Normalize the surface normals
          return surface_normals
      else:
          # Calculate the distance to each centroid --> used to determine which bolt interface the point belongs to
          distances = [np.linalg.norm(dirichlet_points - np.array(centroid), axis=1) for centroid in centroids]
          distances_to_centroid_1 = distances[0]
          distances_to_centroid_2 = distances[1]
          distances_to_centroid_3 = distances[2]
          distances_to_centroid_4 = distances[3]

          # Find the minimum distance to determine the closest centroid
          closest_centroid_indices = np.argmin(np.array([distances_to_centroid_1,
                                                         distances_to_centroid_2,
                                                         distances_to_centroid_3,
                                                         distances_to_centroid_4]), axis=0)
          
          centroid = np.array(centroids)[closest_centroid_indices]  

          z = dirichlet_points[:,2]
          y = centroid[:, 1]
          x = centroid[:, 0]

          # project the points onto the central axis of the bolt interface
          points_on_axis = np.column_stack((x, y, z))  # Combine x, y, z into a single array

          # Compute the surface normals as the difference between the Dirichlet points and the points on the axis
          surface_normals = dirichlet_points - points_on_axis
          surface_normals = -1 * surface_normals  # Reverse the direction of the normals to point outward

          # Normalize the surface normals
          norms = np.linalg.norm(surface_normals, axis=1, keepdims=True)
          surface_normals = surface_normals / norms  # Normalize each normal vector
          return surface_normals  # Return the surface normals as a numpy array
      
# 15. Function to get all prescribed surface normals for the counterbore interfaces
  def get_counterbore_surface_normals(self, points): 
      
      surface_normals = np.zeros(points.shape)  # Initialize an array to hold the surface normals

      disc_points, idx_disc = self.is_on_counterbore_hollow_disc_interface(points) 
      cylinder_points, idx_cylinder = self.is_on_counterbore_cylindrical_interface(points) 

      # 1. The points on the disc interface has a a [0,0,1] normal vector
      disc_normals = np.zeros((len(disc_points), 3))
      disc_normals[:, 2] = 1.0  # Set the z-component to 1 for the disc interface normals 
      
      # 2. use the dirichlet function to get the surface normals for the cylindrical interface
      cylinder_normals = self.get_dirichlet_surface_normals(cylinder_points)
      
      # 3. Combine the normals from both interfaces
      disc_normals = surface_normals[idx_disc] 
      cylinder_normals = surface_normals[idx_cylinder]

      return disc_normals, cylinder_normals
  
  def get_pinn_sharp_edges_surface_normals(self, points):
      # Get edge-specific points and indices
      edge_points_list, edge_idx_list = self.pinn_interface.is_on_sharp_edge_interface(points)

      # Flatten all edge points into a single array
      all_edge_points = np.vstack(edge_points_list)
      normals = np.zeros_like(all_edge_points)

      # Assign normal directions based on edge group
      for i, idx in enumerate(edge_idx_list):
          if len(idx) == 0:
              continue
          if i in [0, 2]:  # Edge 1 and 3 → -Y
              normals[idx, 1] = -1.0
          else:  # Edge 2 and 4 → +Y
              normals[idx, 1] = 1.0

      return normals, edge_idx_list

  # 16. Function to get all prescribed surface normals for the interfaces
  def get_all_prescribed_surface_normals(self, num_points, include_all = False, type = 'numpy_array'):
      """
      Returns surface normals for all relevant interfaces.

      Args:
          num_points (int): Number of points to sample in total.
          include_all (bool): Whether to include counterbore and sharp edges.
          type (str): Output format, 'numpy_array' or 'torch_tensor'.

      Returns:
          tuple or dict: Surface normals and associated point data.
      """ 

      if include_all == False:  # Don't include counterbore and pinn sharp edges 
          points = self.sample_points_from_all_interfaces(num_points)
          points_pin_interface, idx_neumann = self.is_on_pinn_interface(points)
          points_bolt_interface, idx_dirichlet = self.is_on_dirichlet_boundary(points)
          surface_normals_neumann = self.get_neumann_surface_normals(points_pin_interface)
          surface_normals_dirichlet = self.get_dirichlet_surface_normals(points_bolt_interface)
          surface_normals = np.concatenate((surface_normals_neumann, surface_normals_dirichlet), axis=0)
          return {
              "surface_normals": surface_normals,
              "points": points,
              "neumann_idx": idx_neumann,
              "dirichlet_idx": idx_dirichlet
          }

      else:  # Include counterbore and pinn sharp edges
          num_points_pin = 0.2 * num_points
          num_points_bolts = 0.4 * num_points
          num_points_counterbore = 0.2 * num_points
          num_points_pinn_sharp_edges = 0.2 * num_points

          points_interfaces = self.sample_points_from_all_interfaces(num_points=int(num_points_pin + num_points_bolts))
          points_counterbore = self.sample_points_from_counterbore_interface(num_points=int(num_points_counterbore))
          points_pinn_sharp_edges = self.pinn_interface.sample_points_from_pinn_interface_sharp_edges(num_points=int(num_points_pinn_sharp_edges))
          points_pin_interface, idx_neumann = self.is_on_pinn_interface(points_interfaces)
          points_bolt_interface, idx_dirichlet = self.is_on_dirichlet_boundary(points_interfaces)
          points_counterbore_disc, idx_counterbore_disc = self.is_on_counterbore_hollow_disc_interface(points_counterbore)
          points_counterbore_cylinder, idx_counterbore_cylinder = self.is_on_counterbore_cylindrical_interface(points_counterbore)

          surface_normals_neumann = self.get_neumann_surface_normals(points_pin_interface)
          surface_normals_dirichlet = self.get_dirichlet_surface_normals(points_bolt_interface)
          surface_normals_counterbore_disc, surface_normals_counterbore_cylinder = self.get_counterbore_surface_normals(points_counterbore)
          edge_normals, edge_idx = self.get_pinn_sharp_edges_surface_normals(points_pinn_sharp_edges)
          edge1_normals = edge_normals[edge_idx[0]]  
          edge2_normals = edge_normals[edge_idx[1]]
          edge3_normals = edge_normals[edge_idx[2]]
          edge4_normals = edge_normals[edge_idx[3]] 
          points = np.vstack((points_pin_interface, points_bolt_interface, points_pinn_sharp_edges, points_counterbore))

          # Shift the indices to match the combined points array
          num_neumann_pts = points_pin_interface.shape[0]
          num_dirichlet_pts = points_bolt_interface.shape[0]
          num_pinn_edges_pts = points_pinn_sharp_edges.shape[0]
          num_counterbore_pts = points_counterbore.shape[0]
          idx_neumann = np.arange(0, num_neumann_pts)
          idx_dirichlet = np.arange(num_neumann_pts, num_neumann_pts + num_dirichlet_pts)
          edge_global_offset = num_neumann_pts + num_dirichlet_pts
          edge_idx = [edge_global_offset + np.array(e_idx) for e_idx in edge_idx]
          idx_counterbore_disc = num_neumann_pts + num_dirichlet_pts + num_pinn_edges_pts + idx_counterbore_disc
          idx_counterbore_cylinder = num_neumann_pts + num_dirichlet_pts + num_pinn_edges_pts + idx_counterbore_cylinder

          if type == 'torch_tensor':
              surface_normals_neumann = torch.from_numpy(surface_normals_neumann).float()
              surface_normals_dirichlet = torch.from_numpy(surface_normals_dirichlet).float()
              surface_normals_counterbore_disc = torch.from_numpy(surface_normals_counterbore_disc).float()
              surface_normals_counterbore_cylinder = torch.from_numpy(surface_normals_counterbore_cylinder).float()
              edge1_normals = torch.from_numpy(edge1_normals).float()
              edge2_normals = torch.from_numpy(edge2_normals).float()
              edge3_normals = torch.from_numpy(edge3_normals).float()
              edge4_normals = torch.from_numpy(edge4_normals).float()
              points = torch.from_numpy(points).float()
              idx_neumann = torch.from_numpy(idx_neumann).long()
              idx_dirichlet = torch.from_numpy(idx_dirichlet).long()
              idx_counterbore_disc = torch.from_numpy(idx_counterbore_disc).long()
              idx_counterbore_cylinder = torch.from_numpy(idx_counterbore_cylinder).long()
              edge_idx = [torch.from_numpy(idx).long() for idx in edge_idx]  # Convert each edge index to torch tensor
          elif type != 'numpy_array':
              raise ValueError("Invalid output type. Choose from 'numpy_array' or 'torch_tensor'.")
 
          return {
                  "points": points,
                  "neumann_idx": idx_neumann,
                  "dirichlet_idx": idx_dirichlet,
                  "counterbore_disc_idx": idx_counterbore_disc,
                  "counterbore_cyl_idx": idx_counterbore_cylinder, 
                  "edge1_idx": edge_idx[0],
                  "edge2_idx": edge_idx[1],
                  "edge3_idx": edge_idx[2],
                  "edge4_idx": edge_idx[3],
                  "neumann_normals": surface_normals_neumann,
                  "dirichlet_normals": surface_normals_dirichlet,
                  "counterbore_disc_normals": surface_normals_counterbore_disc,
                  "counterbore_cyl_normals": surface_normals_counterbore_cylinder,
                  "edge1_normals": edge1_normals,
                  "edge2_normals": edge2_normals,
                  "edge3_normals": edge3_normals,
                  "edge4_normals": edge4_normals
              }
              


# -------------------- Bolt Ineterface - Minimum Prohibited Region -------------------------------------------------
class Bolt_interface_cylindrical_boundary:
  def __init__(self,
          centroid,
          domain, 
          prescribed_interface_thickness = True,
          prescribed_interface_thickness_value = 6.9431, ## To make thickness the same as the counterbore interface
          prescribed_interface_depth = 10, # mm --> This was chosen arbitrarily --> should be updated
          counterbore = True,
          counterbore_interface_thickness = 5,
          counterbore_depth = 15,
          inner_radius_bolt = 10.287/2, # mm
          inner_radius_counterbore = 14.1732/2, # mm
          ): 
    
    """
    This class defines a closed cylinder in 3D space --> Will be used to define the regions of the design space/domain
    that will be prohibited from containing material for the SimJEB model

    """
    
    self.domain = domain  # Array --> [x_min, x_max, y_min, y_max, z_min, z_max]
    self.cx = centroid[0]
    self.cy = centroid[1]
    self.cz = centroid[2]
    self.z_min = domain[4]
    self.z_max = domain[5]

    # Bolt interface dimensions
    self.prescribed_interface_thickness = prescribed_interface_thickness
    self.inner_radius_bolt = inner_radius_bolt
    self.outer_radius_bolt = self.inner_radius_bolt + prescribed_interface_thickness_value # 
    self.bolt_interface_depth = prescribed_interface_depth

    # Counterbore dimensions
    self.counterbore = counterbore
    self.radius_inner_counterbore = inner_radius_counterbore
    self.radius_outer_counterbore = self.radius_inner_counterbore + counterbore_interface_thickness
    self.counterbore_depth = counterbore_depth
    
    

  def extract_inside_interface_points(self, points):
    """
    This function extracts the points that are located inside the bolt interface region 
    - Defines the region that is prohibited from containing material


    Returns True if (x,y,z) is inside the the closed cylinder with 2 regions: 
      1. (x - cx)^2 + (y - cy)^2 <= r^2   AND   z_min <= z <= bolt_interface_depth
      2. (x - cx)^2 + (y - cy)^2 <= r^2   AND   bolt_interface_depth <= z <= z_max
    """
    pts = np.asarray(points)
    x, y, z = pts[:,0], pts[:,1], pts[:,2]

    dx = x - self.cx
    dy = y - self.cy 
    tolerance = 1e-3

    if self.counterbore == False:

      inside_circle = (dx*dx + dy*dy) <= ((self.inner_radius_bolt-tolerance)**2)
      z0 = self.z_min + self.bolt_interface_depth/2
      d  = self.bolt_interface_depth
      inside_height = (z >= self.z_min) & (z <= z0 + 0.5*d)
      mask = inside_circle & inside_height
      inside_points = points[mask]
      inside_idx = np.where(mask)[0].tolist()
      return inside_points, inside_idx

    else: # if counterbore is True


      inside_circle_interface = (dx*dx + dy*dy) <= ((self.inner_radius_bolt-tolerance)**2)
      z0_interface = self.z_min + self.bolt_interface_depth/2
      inside_interface_depth = (z >= self.z_min) & (z <= z0_interface + 0.5*self.bolt_interface_depth)

      inside_circle_counterbore = (dx*dx + dy*dy) <= ((self.radius_inner_counterbore-tolerance)**2)
      z0_counterbore = self.z_min + self.bolt_interface_depth + self.counterbore_depth/2
      inside_counterbore_depth = (z >= z0_counterbore - 0.5*self.counterbore_depth) & (z <= self.z_max) 


      mask = (inside_circle_interface & inside_interface_depth) | (inside_circle_counterbore & inside_counterbore_depth)
      inside_points = points[mask]
      inside_idx = np.where(mask)[0].tolist()
      return inside_points, inside_idx



  def extract_interface_surface_points(self, points):
    """
    Returns True if (x,y,z) is on the surface of the closed cylinder 
      (x - cx)^2 + (y - cy)^2 = r^2   AND   z_min <= z <= z_max
    """
    pts = np.asarray(points)
    x, y, z = pts[:,0], pts[:,1], pts[:,2]

    dx = x - self.cx
    dy = y - self.cy

    on_circle = np.isclose(dx*dx + dy*dy, self.inner_radius_bolt*self.inner_radius_bolt)

    z0 = self.z_min + self.bolt_interface_depth/2
    d  = self.bolt_interface_depth
    on_height = (z >= self.z_min) & (z <= z0 + 0.5*d)

    mask = on_circle & on_height
    surface_points = points[mask]
    surface_idx = np.where(mask)[0].tolist()
    return surface_points, surface_idx

  def extract_counterbore_cylindrical_surface_points(self, points):
     """
     Returns points and indices for points on the cylindrical surface of the counterbore interface.
     """
     pts = np.asarray(points)
     x, y, z = pts[:,0], pts[:,1], pts[:,2]
     dx = x - self.cx
     dy = y - self.cy

     # Points on the cylindrical surface of the counterbore
     on_cylinder = np.isclose(dx*dx + dy*dy, self.radius_inner_counterbore**2)
     z_min = self.z_min + self.bolt_interface_depth
     z_max = self.z_min + self.bolt_interface_depth + self.counterbore_depth
     on_cylinder_height = (z >= z_min) & (z <= z_max)
     mask_cylinder = on_cylinder & on_cylinder_height

     surface_points = pts[mask_cylinder]
     surface_idx = np.where(mask_cylinder)[0].tolist()
     return surface_points, surface_idx

  def extract_counterbore_disc_surface_points(self, points):
     """
     Returns points and indices for points on the bottom disc surface of the counterbore interface (hollow disk in xy plane).
     """
     pts = np.asarray(points)
     x, y, z = pts[:,0], pts[:,1], pts[:,2]
     dx = x - self.cx
     dy = y - self.cy

     # Points on the bottom surface of the counterbore (hollow disk in xy plane)
     z_min = self.z_min + self.bolt_interface_depth
     on_bottom = np.isclose(z, z_min)
     r_squared = dx*dx + dy*dy
     inner_r_sq = self.inner_radius_bolt ** 2
     outer_r_sq = self.radius_inner_counterbore ** 2 
     in_ring = (r_squared >= inner_r_sq) & (r_squared <= outer_r_sq)
     mask_bottom = on_bottom & in_ring

     surface_points = pts[mask_bottom]
     surface_idx = np.where(mask_bottom)[0].tolist()
     return surface_points, surface_idx
     
  
  def extract_inside_interface_thickness_points(self,points):
    """
    Returns true if a point is inside the hollow cylinder that defines the interface thickness
    Inner cylinder: (x - cx)^2 + (y - cy)^2 <= r_inner^2   AND   z_min <= z <= bolt_interface_depth
    Outer cylinder: (x - cx)^2 + (y - cy)^2 <= r_outer^2   AND   z_min <= z <= bolt_interface_depth
    """
    if self.prescribed_interface_thickness == False:
      raise ValueError("Prescribed interface thickness is not defined")
    
    else: # Interface thickness is prescribed
      pts = np.asarray(points)
      x, y, z = pts[:,0], pts[:,1], pts[:,2]

      dx = x - self.cx
      dy = y - self.cy

      inner_circle = (dx*dx + dy*dy) <= (self.inner_radius_bolt*self.inner_radius_bolt)
      outer_circle = (dx*dx + dy*dy) <= (self.outer_radius_bolt*self.outer_radius_bolt)
 
      z0 = self.z_min + self.bolt_interface_depth/2       
      d  = self.bolt_interface_depth
      inside_height = (z >= self.z_min) & (z <= z0 + 0.5*d)
      #inside_height = (z >= self.z_min) & (z <= self.bolt_interface_depth)
      
      mask = (outer_circle & inside_height) & (~inner_circle)
      inside_points = points[mask]
      inside_idx = np.where(mask)[0].tolist()
      return inside_points, inside_idx
    
  def extract_counterbore_interface_thickness_points(self,points):
    """
    Returns true if a point is inside the hollow cylinder that defines the interface thickness
    Inner cylinder: (x - cx)^2 + (y - cy)^2 <= r_inner^2   AND   bolt_interface_depth <= z <= z_max
    Outer cylinder: (x - cx)^2 + (y - cy)^2 <= r_outer^2   AND   bolt_interface_depth <= z <= z_max
    """
    if self.counterbore == False:
      raise ValueError("Counterbore is not defined")
    
    else:
      pts = np.asarray(points)
      x, y, z = pts[:,0], pts[:,1], pts[:,2]

      dx = x - self.cx
      dy = y - self.cy

      inner_circle = (dx*dx + dy*dy) <= (self.radius_inner_counterbore*self.radius_inner_counterbore)
      outer_circle = (dx*dx + dy*dy) <= (self.radius_outer_counterbore*self.radius_outer_counterbore)

      z0 = self.z_min + self.bolt_interface_depth + self.counterbore_depth/2
      inside_height = (z >= z0 - 0.5*self.counterbore_depth) & (z <= z0 + 0.5*self.counterbore_depth)

      mask = (outer_circle & inside_height) & (~inner_circle)
      inside_points = points[mask]
      inside_idx = np.where(mask)[0].tolist()
      return inside_points, inside_idx
    

  def sample_points_from_bolt_interface(self,num_points,random_seed = None):
    """ 
    This function generates a set of random points on the surface the of the bolt interface 
    """
    random_number_generator = np.random.default_rng(random_seed)
    theta = random_number_generator.uniform(0, 2*np.pi,size = num_points)
    z0 = self.z_min + self.bolt_interface_depth/2
    d = self.bolt_interface_depth 
    z_max = self.z_min + self.bolt_interface_depth/2
    z = random_number_generator.uniform(z0 - 0.5*d, z0 + 0.5*d, size = num_points)
    x = self.cx + self.inner_radius_bolt*np.cos(theta)
    y = self.cy + self.inner_radius_bolt*np.sin(theta)
    points = np.column_stack((x,y,z))
    return points 
  
  def sample_points_from_counterbore_interface(self,num_points,random_seed = None): 
    """
    This function generates a set of random points on the surface the of the counterbore interface 
    """
    num_points_cylindrical = int(0.8*num_points)  # 80% of the points will be sampled from the cylindrical surface
    num_points_bottom = num_points - num_points_cylindrical  # Remaining 20% will be sampled from the bottom surface

    # 1. Sample from cylindrical surface of the counterbore interface 
    random_number_generator = np.random.default_rng(random_seed)
    theta = random_number_generator.uniform(0, 2*np.pi, size=num_points_cylindrical)
    z_min = self.z_min + self.bolt_interface_depth
    z_max = self.z_min + self.bolt_interface_depth + self.counterbore_depth
    z = random_number_generator.uniform(z_min, z_max, size=num_points_cylindrical)
    x = self.cx + self.radius_inner_counterbore * np.cos(theta)
    y = self.cy + self.radius_inner_counterbore * np.sin(theta)
    points = np.column_stack((x, y, z))

    # 2. Sample points from the bottom surface of the counterbore interface (hollow cylinder in XY plane at z_bottom)
    z_bottom = self.z_min + self.bolt_interface_depth
    inner_radius = self.inner_radius_bolt
    outer_radius = self.radius_inner_counterbore 
    theta_bottom = random_number_generator.uniform(0, 2*np.pi, size=num_points_bottom)
    r_inner_squared = inner_radius ** 2
    r_outer_squared = outer_radius ** 2
    r_squared = random_number_generator.uniform(r_inner_squared, r_outer_squared, size=num_points_bottom)
    r = np.sqrt(r_squared)
    x_bottom = self.cx + r * np.cos(theta_bottom)
    y_bottom = self.cy + r * np.sin(theta_bottom)
    z_bottom_vals = z_bottom * np.ones_like(x_bottom)
    points_bottom = np.column_stack((x_bottom, y_bottom, z_bottom_vals))

    # 3. Concatenate the cylindrical and bottom points and return
    points = np.vstack((points, points_bottom))
    return points  
  


#----------------------------------------------------------------------------------------------------------------------
  



class Pinn_interface_cylindrical_boundary:
    def __init__(self, 
                 centroid1,
                 centroid2,
                 domain,
                 prescribed_radial_thickness_value=5,
                 prescribed_minimum_width_value=5,
                 prescribed_sharp_edges=False,
                 inner_radius=19.05/2, # mm
                 ):
        """
        This class defines a closed cylinder in 3D space --> Will be used to define the regions of the design space/domain
        that will be prohibited from containing material for the SimJEB model

        The function also allows the user to define the following:
        1. Prescribed radial thickness
        2. Prescribed minimum width
        3. Prescribe sharp edges around the interface holes
        """
        # centroids
        self.cx1, self.cy1, self.cz1 = centroid1
        self.cx2, self.cy2, self.cz2 = centroid2

        # domain Y-bounds
        self.y_min = domain[2] #+ 0.2*(domain[3] - domain[2])
        self.y_max = domain[3] #- 0.2*(domain[3] - domain[2])

        # core geometry
        self.inner_radius = inner_radius
        self.prescribed_radial_thickness = prescribed_radial_thickness_value
        self.outer_radius = self.inner_radius + self.prescribed_radial_thickness

        # width bands
        self.prescribed_minimum_width = prescribed_minimum_width_value
        self.pinn_width = prescribed_minimum_width_value
        self.left_boundary_interface_1  = self.cy1 - self.pinn_width/2
        self.right_boundary_interface_1 = self.cy1 + self.pinn_width/2

        self.left_boundary_interface_2  = self.cy2 - self.pinn_width/2
        self.right_boundary_interface_2 = self.cy2 + self.pinn_width/2

        # sharp edges flag (you can add extra logic later if needed)
        self.prescribed_sharp_edges = prescribed_sharp_edges


    def extract_inside_interface_points(self, points):
        """
  
        """
        pts = np.asarray(points)
        x,y,z = pts[:,0], pts[:,1], pts[:,2]
        dx1, dz1 = x-self.cx1, z-self.cz1


        # Mask for the base case --> Always prohibited region
        tolerance = 1e-3
        mask = (dx1*dx1 + dz1*dz1 <= (self.inner_radius-tolerance)**2) & (y >= self.y_min) & (y <= self.y_max)

        if self.prescribed_sharp_edges:
           
           # Define the y range for each cylindrical region
           y_range_left = (y >= self.y_min) & (y <= self.left_boundary_interface_1) 
           y_range_middle = (y >= self.right_boundary_interface_1) & (y <= self.left_boundary_interface_2)
           y_range_right = (y >= self.right_boundary_interface_2) & (y <= self.y_max)

          # Define the radius of the cylindrical region for each y range
           radius_left = self.inner_radius + self.prescribed_radial_thickness
           radius_middle = self.inner_radius + self.prescribed_radial_thickness
           radius_right = self.inner_radius + self.prescribed_radial_thickness

           # Mask for each cylindrical region
           mask_left = (dx1*dx1 + dz1*dz1 <= radius_left**2) & y_range_left
           mask_middle = (dx1*dx1 + dz1*dz1 <= radius_middle**2) & y_range_middle
           mask_right = (dx1*dx1 + dz1*dz1 <= radius_right**2) & y_range_right

           # add the masks to the base case mask
           mask = mask | mask_left | mask_middle | mask_right

        # Extract the points and indices
        points_inside = pts[mask]
        indices_inside = np.where(mask)[0].tolist()
        return points_inside, indices_inside



    def extract_interface_surface_points(self, points):
        """
  
        """
        pts = np.asarray(points)
        x,y,z = pts[:,0], pts[:,1], pts[:,2]
        dx, dz = x-self.cx1, z-self.cz1

        dx1 = x - self.cx1
        dz1 = z - self.cz1
        cond_rad1 = np.isclose(dx1*dx1 + dz1*dz1, self.inner_radius**2)

        # y-window for interface 1
        cond_y1 = (y >= self.left_boundary_interface_1) & \
                  (y <= self.right_boundary_interface_1)

        # radial test around centroid2
        dx2 = x - self.cx2
        dz2 = z - self.cz2
        cond_rad2 = np.isclose(dx2*dx2 + dz2*dz2, self.inner_radius**2)

        # y-window for interface 2
        cond_y2 = (y >= self.left_boundary_interface_2) & \
                  (y <= self.right_boundary_interface_2)

        # combine: surface1 OR surface2
        mask = (cond_rad1 & cond_y1) | (cond_rad2 & cond_y2)

        surface_points = pts[mask]
        surface_idx    = np.where(mask)[0].tolist()
        return surface_points, surface_idx



    def extract_inside_interface_thickness_points(self, points):
        """
           Returns the points that are inside the hollow cylinder that defines the interface thickness
        """
        pts = np.asarray(points)
        x,y,z = pts[:,0], pts[:,1], pts[:,2]
        dx, dz = x-self.cx1, z-self.cz1
        
        # Define the y range for each cylindrical region
        y_range_left = (y >= self.left_boundary_interface_1) & (y <= self.right_boundary_interface_1)
        y_range_right = (y >= self.left_boundary_interface_2) & (y <= self.right_boundary_interface_2)

        inside_radius_critetion = (dx*dx + dz*dz <= self.outer_radius**2) & \
                                  (dx*dx + dz*dz >= self.inner_radius**2) 
        
        # Mask for each cylindrical region
        mask_left = inside_radius_critetion & y_range_left
        mask_right = inside_radius_critetion & y_range_right
        mask = mask_left | mask_right 
        inside_points = pts[mask]
        inside_idx = np.where(mask)[0].tolist()
        return inside_points, inside_idx 
    

    def sample_points_from_pinn_interface(self, num_points, random_seed=None):
      """
      This function generates a set of random points on the surface the of the bolt interface 
      """
      random_number_generator = np.random.default_rng(random_seed)
      theta = random_number_generator.uniform(0, 2*np.pi, size=num_points)

      # Sample from interface 1
      z = self.cz1 + self.inner_radius * np.sin(theta)
      x = self.cx1 + self.inner_radius * np.cos(theta)
      y = random_number_generator.uniform(self.cy1 - self.pinn_width/2, self.cy1 + self.pinn_width/2, size=num_points)

      # Sample from interface 2
      z2 = self.cz2 + self.inner_radius * np.sin(theta)
      x2 = self.cx2 + self.inner_radius * np.cos(theta)
      y2 = random_number_generator.uniform(self.cy2 - self.pinn_width/2, self.cy2 + self.pinn_width/2, size=num_points)

      # Combine the points from both interfaces
      x = np.hstack((x, x2))
      y = np.hstack((y, y2))
      z = np.hstack((z, z2))

      points = np.column_stack((x,y,z)) 
      return points
    
    def sample_points_on_load_surface(self,num_points, load_type: str ):
      """
      This function generates a set of random points on the portion of the pinn interface that has an applied load. 
      The region of the surface subjected to forces depend on the direction of the applied load 
      """

      num_points = 2*num_points # Increase the number since we filter half the domain

      # 1. Sample points on the surface of the interface
      points = self.sample_points_from_pinn_interface(num_points)
      x = points[:,0]
      y = points[:,1]
      z = points[:,2] 

      if load_type == 'vertical': 
         mask = (z >= self.cz1)
      elif load_type == 'horizontal':
         mask = (x <= self.cx1)
      elif load_type == 'diagonal':
          theta = np.deg2rad(48)
          theta = theta + np.pi/2
          vx, vz = -np.cos(theta), -np.sin(theta)
          mask = ((x - self.cx1)*vx + (z - self.cz1)*vz) <= 0
      else:
          raise ValueError("Invalid load type. Choose from 'vertical', 'horizontal', or 'diagonal'.")
      
      # 2. Apply the mask to filter the points
      points = points[mask]
      return points 
  
    def sample_points_from_pinn_interface_sharp_edges(self, num_points, random_seed=None):
        """
        Function to sample points on the hollow cylinder edges of the PINN interface  
        """

        inner_radius = self.inner_radius 
        outer_radius = self.outer_radius
        width = self.pinn_width 

        y1 = self.cy1 - width/2  # far left edge of interface 
        y2 = self.cy1 + width/2  # middle left edge of interface 
        y3 = self.cy2 - width/2  # middle right edge of interface
        y4 = self.cy2 + width/2  # far right edge of interface
        y_values = [y1, y2, y3, y4]

        num_points_per_edge = int(num_points / 4)  # Divide the number of points equally among the four edges 
        all_points = []

        random_number_generator = np.random.default_rng(random_seed)

        for y_i in y_values:
            theta = random_number_generator.uniform(0, 2*np.pi, size=num_points_per_edge)
            r_squared = random_number_generator.uniform(inner_radius**2, outer_radius**2, size=num_points_per_edge)
            r = np.sqrt(r_squared)
            x_vals = self.cx1 + r * np.cos(theta)
            z_vals = self.cz1 + r * np.sin(theta)
            y_vals = np.full_like(x_vals, y_i)
            points_i = np.column_stack((x_vals, y_vals, z_vals))
            all_points.append(points_i)

        all_points = np.vstack(all_points)  # Combine all edge points into a single array 

        return all_points

    def is_on_sharp_edge_interface(self,points):
        """
        Function to find the points on the sharp edge of each interface of the PINN interface
        """
        inner_radius = self.inner_radius 
        outer_radius = self.outer_radius
        width = self.pinn_width  

        y1 = self.cy1 - width/2  # far left edge of interface 
        y2 = self.cy1 + width/2  # middle left edge of interface 
        y3 = self.cy2 - width/2  # middle right edge of interface
        y4 = self.cy2 + width/2  # far right edge of interface
        y_values = [y1, y2, y3, y4]

        edge_points = [] 
        edge_indices = []
        for i, y_i in enumerate(y_values):
            # Check if the point is on the sharp edge of the interface
            x, y, z = points[:,0], points[:,1], points[:,2]
            dx = x - self.cx1
            dz = z - self.cz1

            # Check if the point is on the sharp edge of the interface
            mask = np.isclose(y, y_i) & (dx*dx + dz*dz >= inner_radius**2) & (dx*dx + dz*dz <= outer_radius**2)

            edge_points_i = points[mask]
            edge_indices_i = np.where(mask)[0].tolist()

            edge_points.append(edge_points_i)
            edge_indices.append(edge_indices_i) 
          
        return edge_points, edge_indices

#----------------------------------------------------------------------------------------------------------------------


