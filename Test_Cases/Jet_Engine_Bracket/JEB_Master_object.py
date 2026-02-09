
# class object containig all the functions and variables for the JEB test case 
# Use this as the input to the PINN loss etc 
import numpy as np
import torch 
from Test_Cases.Jet_Engine_Bracket.boundary_surfaces import Jet_engine_bracket_interfaces  


class JEB_Master_object:
    def __init__(self, Normalize = False, 
                 Symmetry = False,
                 Expand = False,
                 expansion_factor = None):  
        self.Normalize = Normalize

        # Symmetry in the JEB 
        self.Symmetry = Symmetry
        self.symmetry_axis = ['y']  # Symmetry axis is z-axis in this test case 
        self.symmetry_location = 0.0 


        self.test_case = "Jet_Engine_Bracket"  # Name of the test case 



        ''' Define the material properties and other physics properties: '''
        # -----------------------------------------------------------------------------------------------
        self.elastic_modulus = 113800           # MPa(N/mm^2)  
        self.material_density = 4.47e-6          # kg/mm^3 
        self.yield_strength = 903.213            # MPa(N/mm^2)  
        self.poisson_ratio = 0.342               # dimensionless  
        self.gravitational_acceleration = 9.81    # m/s^2 
        #-----------------------------------------------------------------------------------------------
  

        ''' Define the permissable design space / design domain / design envelope: ''' 
        # -----------------------------------------------------------------------------------------------
        # Defined by the bounding box of part 411 from the SimJEB dataset
        self.domain = [ -3.92288017e+01,  6.73299866e+01, -1.63356995e+02,  1.68147602e+01, -1.17950094e-12,  6.25040588e+01]
        
        self.domain_volume = (self.domain[1] - self.domain[0]) * (self.domain[3] - self.domain[2]) * (self.domain[5] - self.domain[4])
        # -------------------------------------------------------------------------------------------------
       

        ''' Define the load cases: '''
        # -----------------------------------------------------------------------------------------------
        self.applied_point_load = False             # There is no point load for any of the 3 load cases 
        self.applied_distributed_load = True        # Applied loads is a distributed load for each load case
        self.body_forces = False                    # Include body forces --> Gravitational forces on the structure

        # Vertical load case
        self.force_magnitude_vertical = 35600       # Newtons
        self.force_direction_vertical = np.array([0, 0, 1],dtype=float)  # cartesian coordinates 
        self.force_vector_vertical = self.force_magnitude_vertical * self.force_direction_vertical

        # Horizontal load case
        self.force_magnitude_horizontal = 37800       # Newtons
        self.force_direction_horizontal = np.array([0, 1, 0],dtype=float) # cartesian coordinates
        self.force_vector_horizontal = self.force_magnitude_horizontal * self.force_direction_horizontal

        # Diagonal load case
        self.force_magnitude_diagonal = 42300       # Newtons
        load_angle = 0.837758                      # Radians -->  = 48 degrees  
        self.force_direction_diagonal = np.array([0, np.sin(load_angle), np.cos(load_angle)],dtype=float)
        self.force_vector_diagonal = self.force_magnitude_diagonal * self.force_direction_diagonal 


        ''' Define the geometrical features of the structure: --> Measurements in mm''' 
        # -----------------------------------------------------------------------------------------------
           

        self.counterbore = True
        self.counterbore_radius = 17/2  
        self.counterbore_interface_thickness = 5
        self.counterbore_depth = 10

        self.prescribed_bolt_interface_thickness = True
        self.bolt_interface_radius = 10.287/2
        self.bolt_interface_thickness = self.counterbore_interface_thickness + (self.counterbore_radius - self.bolt_interface_radius)
        self.bolt_interface_depth = 10 
 
        self.prescribed_pinn_interface_thickness = True
        self.pinn_interface_radius = 19.05/2
        self.pinn_interface_thickness = 5
        self.pinn_interface_depth = 10
        self.pinn_interface_width = 5
        self.pinn1_centroid = np.array([-21.3572765,      -88.9787065,      44.724602], dtype=float)
        self.pinn2_centroid = np.array([-21.3572765,      -60.416866,       44.724602], dtype=float) 
        self.bolt1_centroid = np.array([52.0471405,        1.5804905,        3.9242995], dtype=float) 
        self.bolt2_centroid = np.array([38.0832625,       -146.968384,       3.9242995], dtype=float)
        self.bolt3_centroid = np.array([0,                -148.124817,       3.9242995], dtype=float)
        self.bolt4_centroid = np.array([0,                 0,                3.9242995], dtype=float) 


        # Surface area of the pinn interface / 2 = neumann surface area
        self.neumann_surface_area = (2*np.pi*self.pinn_interface_radius * self.pinn_interface_depth)/2 
        self.dim = 3 # 3D case 
        #----------------------------------------------------------------------------------------------
 

        ''' Zero-center and normalize the domain: '''
        #------------------------------------------------------------------------------------------------- 
        
        #1. Find the center of the domain:
        self.domain_center = np.array([
                                                    0.5*(self.domain[0] + self.domain[1]),
                                                    0.5*(self.domain[2] + self.domain[3]),
                                                    0.5*(self.domain[4] + self.domain[5]),
                                                ])
        
        #2. Find the largest extent of the domain 
        self.domain_extents = np.array([self.domain[1] - self.domain[0],
                                        self.domain[3] - self.domain[2], 
                                        self.domain[5] - self.domain[4]])
        self.max_extent = np.max(self.domain_extents) # The largest extent of the domain is used to normalize the domain

        #3. Set the scaling factor for the domain based on the largest extent of the domain
        self.domain_scaling_factor = 2.0 / self.max_extent

        # Find the new bounding box of the domain after scaling --> This is used to normalize the domain 
        corner_min = np.array([self.domain[0], self.domain[2], self.domain[4]])
        corner_max = np.array([self.domain[1], self.domain[3], self.domain[5]])
        norm_min = (corner_min - self.domain_center) * self.domain_scaling_factor
        norm_max = (corner_max - self.domain_center) * self.domain_scaling_factor 

        
 
        if self.Normalize == True: 
            '''
            Normalize domain such that each axis extends from [-1,1], and zero-center the domain. 
            The geometric features and the load cases are also normalized and shifted to the new domain.
            '''
            scale = self.domain_scaling_factor   

            # Scale the domain: 
            self.domain = np.array([norm_min[0], norm_max[0], norm_min[1], norm_max[1], norm_min[2], norm_max[2]])

            if Expand == True:
                if expansion_factor == None:
                    raise ValueError("Choose an expansion factor between 1 and 1.5")
                elif expansion_factor > 1.5 or expansion_factor < 1:
                    raise ValueError("Choose an expansion factor between 1 and 1.5")
                
                self.domain[0] = self.domain[0]*expansion_factor #x_min
                self.domain[1] = self.domain[1]*expansion_factor #x_max
                self.domain[2] = self.domain[2]*expansion_factor #y_min
                self.domain[3] = self.domain[3]*expansion_factor #y_max
                self.domain[4] = self.domain[4] #z_min = unchanged 
                self.domain[5] = self.domain[5]*expansion_factor #z_max

                self.domain_volume = self.domain_volume = (self.domain[1] - self.domain[0]) * (self.domain[3] - self.domain[2]) * (self.domain[5] - self.domain[4])
            else:
                self.domain_volume = self.domain_volume * (scale**3) # Scale the volume of the domain  

            # Scale the geometric features: 
            scale = self.domain_scaling_factor 
            self.bolt_interface_radius            = self.bolt_interface_radius           * scale
            self.bolt_interface_thickness         = self.bolt_interface_thickness        * scale
            self.bolt_interface_depth             = self.bolt_interface_depth            * scale
            self.counterbore_radius               = self.counterbore_radius              * scale
            self.counterbore_interface_thickness  = self.counterbore_interface_thickness * scale
            self.counterbore_depth                = self.counterbore_depth               * scale
            self.pinn_interface_radius            = self.pinn_interface_radius           * scale
            self.pinn_interface_thickness         = self.pinn_interface_thickness        * scale
            self.pinn_interface_depth             = self.pinn_interface_depth            * scale
            self.pinn_interface_width             = self.pinn_interface_width            * scale
            self.neumann_surface_area             = self.neumann_surface_area            * scale

            # Shift the centroid of the bolt and pinn interfaces:
            self.bolt1_centroid = (self.bolt1_centroid - self.domain_center) * scale
            self.bolt2_centroid = (self.bolt2_centroid - self.domain_center) * scale 
            self.bolt3_centroid = (self.bolt3_centroid - self.domain_center) * scale
            self.bolt4_centroid = (self.bolt4_centroid - self.domain_center) * scale
            self.pinn1_centroid = (self.pinn1_centroid - self.domain_center) * scale
            self.pinn2_centroid = (self.pinn2_centroid - self.domain_center) * scale 

            if self.Symmetry == True:
                #1. Make the centroids of the bolt interfaces symmetric about the y-axis 
                self.bolt2_centroid = np.array([self.bolt1_centroid[0], -1*self.bolt1_centroid[1], self.bolt1_centroid[2]])
                self.bolt3_centroid = np.array([self.bolt4_centroid[0], -1*self.bolt4_centroid[1], self.bolt4_centroid[2]]) 
                self.pinn2_centroid = np.array([self.pinn1_centroid[0], -1*self.pinn1_centroid[1], self.pinn1_centroid[2]])

                # 2. Reduce the domain to half the size in the y-direction
                self.domain[2] = 0.0 

                # 3. Reduce the domain volume to half the size in the y-direction 
                self.domain_volume = self.domain_volume * 0.5 

                # 4. Reduce the force magnitude to half the size 
                self.force_magnitude_vertical /= 2.0
                self.force_magnitude_horizontal /= 2.0
                self.force_magnitude_diagonal /= 2.0
                self.force_vector_vertical   = self.force_magnitude_vertical * self.force_direction_vertical
                self.force_vector_horizontal = self.force_magnitude_horizontal * self.force_direction_horizontal
                self.force_vector_diagonal   = self.force_magnitude_diagonal * self.force_direction_diagonal 
        else:
            scale = 1.0


    # -----------------------------------------------------------------------------------------------

  
    



    ''' Function to create the interface object: ''' 
    # ----------------------------------------------------------------------------------------------- 

    def create_interfaces(self, 
                             domain = None,
                            # Location of the bolt interfaces
                            centroid_bolt_interface_1 = None,
                            centroid_bolt_interface_2 = None,
                            centroid_bolt_interface_3 = None,
                            centroid_bolt_interface_4 = None,
                            # Location of the PINN interfaces
                            centroid_pinn_interface_1 = None,
                            centroid_pinn_interface_2 = None,


                            # Parameters for the bolt interface 
                            prescribed_bolt_interface_thickness = None, 
                            inner_radius_bolt = None,
                            prescribed_bolt_interface_thickness_value = None,
                            prescribed_bolt_interface_depth = None,

                            # Parameters for the counterbore interface  
                            counterbore = True,
                            counterbore_radius = None,
                            counterbore_interface_thickness = None,
                            counterbore_depth = None,

                            # Parameters for the PINN interface
                            pinn_interface_radius = None,
                            prescribed_radial_thickness_value = None,
                            prescribed_minimum_width_value = None,
                            prescribed_sharp_edges = True):
        
        # Use default values if None is passed 
        if domain is not None:
            self.domain = domain
        if centroid_bolt_interface_1 is not None:
            self.bolt1_centroid = centroid_bolt_interface_1
        if centroid_bolt_interface_2 is not None:
            self.bolt2_centroid = centroid_bolt_interface_2
        if centroid_bolt_interface_3 is not None:
            self.bolt3_centroid = centroid_bolt_interface_3
        if centroid_bolt_interface_4 is not None:
            self.bolt4_centroid = centroid_bolt_interface_4
        if centroid_pinn_interface_1 is not None:
            self.pinn1_centroid = centroid_pinn_interface_1
        if centroid_pinn_interface_2 is not None:
            self.pinn2_centroid = centroid_pinn_interface_2
        if prescribed_bolt_interface_thickness is not None:
            self.prescribed_bolt_interface_thickness = prescribed_bolt_interface_thickness
        if inner_radius_bolt is not None:
            self.bolt_interface_radius = inner_radius_bolt
        if prescribed_bolt_interface_thickness_value is not None:
            self.bolt_interface_thickness = prescribed_bolt_interface_thickness_value
        if prescribed_bolt_interface_depth is not None:
            self.bolt_interface_depth = prescribed_bolt_interface_depth
        if counterbore_radius is not None:
            self.counterbore_radius = counterbore_radius
        if counterbore_interface_thickness is not None:
            self.counterbore_interface_thickness = counterbore_interface_thickness
        if counterbore_depth is not None:
            self.counterbore_depth = counterbore_depth
        if pinn_interface_radius is not None:
            self.pinn_interface_radius = pinn_interface_radius
        if prescribed_radial_thickness_value is not None:
            self.pinn_interface_thickness = prescribed_radial_thickness_value
        if prescribed_minimum_width_value is not None:
            self.pinn_interface_width = prescribed_minimum_width_value
        if prescribed_sharp_edges is not None:
            self.pinn_interface_sharp_edges = prescribed_sharp_edges 
        if counterbore is not None:
                self.counterbore = counterbore

 
 
        self.interfaces = Jet_engine_bracket_interfaces(
                                            domain = self.domain,

                                            # Location of the bolt interfaces
                                            centroid_bolt_interface_1 = self.bolt1_centroid,
                                            centroid_bolt_interface_2 = self.bolt2_centroid,
                                            centroid_bolt_interface_3 = self.bolt3_centroid,
                                            centroid_bolt_interface_4 = self.bolt4_centroid,
                                            # Location of the PINN interfaces
                                            centroid_pinn_interface_1 = self.pinn1_centroid,
                                            centroid_pinn_interface_2 = self.pinn2_centroid, 


                                            # Parameters for the bolt interface 
                                            prescribed_bolt_interface_thickness = self.prescribed_bolt_interface_thickness,  
                                            inner_radius_bolt = self.bolt_interface_radius,
                                            prescribed_bolt_interface_thickness_value = self.bolt_interface_thickness,
                                            prescribed_bolt_interface_depth = self.bolt_interface_depth,

                                            # Parameters for the counterbore interface 
                                            counterbore = self.counterbore,
                                            counterbore_radius = self.counterbore_radius,
                                            counterbore_interface_thickness = self.counterbore_interface_thickness,
                                            counterbore_depth = self.counterbore_depth,

                                            # Parameters for the PINN interface
                                            pinn_interface_radius = self.pinn_interface_radius,
                                            prescribed_radial_thickness_value = self.pinn_interface_thickness,
                                            prescribed_minimum_width_value = self.pinn_interface_width,
                                            prescribed_sharp_edges = self.pinn_interface_sharp_edges,
                                            
                                            # Symmetry conditioning:
                                            Symmetry = self.Symmetry) 

        
    def scale_up(self,coords,sdf):
        scale = self.domain_scaling_factor  

        coords = (coords / scale) + self.domain_center
        sdf = sdf/scale 

        return coords, sdf


