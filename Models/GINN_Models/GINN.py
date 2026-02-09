

import torch 
import torch.nn as nn
import numpy as np
from Models.SIREN.SIREN import SIREN 
from Models.PINN_Models.Utils.feature_expansion import expand_input_features_sine # Positional encoding functiom
from Models.PINN_Models.Utils.feature_expansion import expand_input_features_squared # Positional encoding functiom
from Models.PINN_Models.Utils.feature_expansion import generate_fourrier_features # Positional encoding functiom
from Models.WIRE.WIRE import WIRE # WIRE GINN model 
from Models.MLP.MLP import MLP # MLP model for GINN 

class GINN(nn.Module):

  
    # GINN consructor function
    def __init__(self, test_case, feature_expansion: dict ,Model_hyperparameters): 
        
        super(GINN, self).__init__()
        self.domain = np.array(test_case.domain, dtype=np.float32)   # Array of shape (1,6) for 3D or (1,4) for 2D 
        self.Model_hparams = Model_hyperparameters                   # Network hyperparameters for the GINN model 
        self.dim = len(self.domain)/2                                # Dimensionality of the problem - 2D or 3D
        self.feature_expansion_hparams = feature_expansion           # Feature expansion hyperparameters
        self.feature_expansion = feature_expansion['Feature Type']   # Positional encoding technique - sine or squared
        self.num_frequencies = feature_expansion['Num Frequencies']  # Number of frequencies to use for the fourrier features  
        self.test_case = test_case          
                  
 
        # Check if the model will be using expanded input features 
        if self.feature_expansion == 'sine_features': 
            expanded_input_features = True
            self.num_inputs  = 2*self.dim  
        elif self.feature_expansion == 'squared_features':
            expanded_input_features = True
            self.num_inputs  = 2*self.dim
        elif self.feature_expansion == 'fourrier_features':
            expanded_input_features = True
            self.num_inputs = self.dim + 2*self.dim*self.num_frequencies 
        else:
            expanded_input_features = False
            self.num_inputs = self.dim # No feature expansion

        # Initialize the SIREN GINN model - this is the main model that will be used to predict the displacement field
        if self.Model_hparams['Model_type'] == 'SIREN':  
            self.model = SIREN(self.Model_hparams,self.num_inputs,'GINN') # SIREN GINN model 
        elif self.Model_hparams['Model_type'] == 'WIRE':
            self.model = WIRE(self.Model_hparams,self.num_inputs,'GINN') # WIRE GINN model 
        elif self.Model_hparams['Model_type'] == 'MLP':
            self.model = MLP(self.Model_hparams,self.num_inputs,'GINN') 
        else:
            raise ValueError("Invalid model type. Define the model type in the Model_hyperparameters dictionary as 'SIREN','WIRE' or 'MLP'.") 

        if self.feature_expansion == 'sine_features':
            self.feature_object = expand_input_features_sine(self.dim) # = [x, y, z, sin(x), sin(y) ,sin(z)]
        elif self.feature_expansion == 'squared_features':
            self.feature_object = expand_input_features_squared(self.dim) 
        elif self.feature_expansion == 'fourrier_features':
            self.feature_object = generate_fourrier_features(self.dim,self.feature_expansion_hparams)  
        else:
            self.feature_object = None # No feature expansion  


    # GINN forward pass   
    def forward(self, coords: torch.Tensor) -> torch.Tensor: 
        """
        Forward pass of the GINN model:
        Input:
        - coords: coordinates of the input point (x,y,z) --> shape (N,3) for 3D or (N,2) for 2D
        Output:
        - SDF field --> shape (N,1) for 3D or (N,1) for 2D
        """

        #2. Generate the positional encoding for the input coordinates --> (Batch_size, 6)
        # input features = x,y,z coordinates  
        if self.feature_object is not None:
            modified_input_features = self.feature_object(coords) # --> (Batch_size, 2*dim) for sine or squared features
        else:
            modified_input_features = coords 

        #3. Use the SIREN GINN model to predict the displacement field --> (Batch_size, 3) for 3D or (Batch_size, 2) for 2D
        SDF = self.model(modified_input_features)

        if self.test_case.Symmetry == False: 
            return SDF
        
        # If the test case has symmetry - need to mirror the points and SDF values 
        else: 
            points_list = [coords]
            sdf_list    = [SDF]

            # for each symmetry axis, mirror every block we have so far
            for axis in self.test_case.symmetry_axis:
                new_points_blocks = []
                new_sdf_blocks    = []

                for pts, sdf_vals in zip(points_list, sdf_list):
                    if axis == 'x':
                        # reflect x‐coordinate
                        new_x = -pts[:, 0]
                        if pts.shape[1] == 2:
                            new_pts = torch.stack((new_x, pts[:, 1]), dim=1)
                        else:
                            new_pts = torch.stack((new_x, pts[:, 1], pts[:, 2]), dim=1)

                    elif axis == 'y':
                        # reflect y‐coordinate
                        new_y = -pts[:, 1]
                        if pts.shape[1] == 2:
                            new_pts = torch.stack((pts[:, 0], new_y), dim=1)
                        else:
                            new_pts = torch.stack((pts[:, 0], new_y, pts[:, 2]), dim=1)

                    elif axis == 'z' and pts.shape[1] == 3:
                        # reflect z‐coordinate
                        new_z = -pts[:, 2]
                        new_pts = torch.stack((pts[:, 0], pts[:, 1], new_z), dim=1)

                    else:
                        continue 

                    new_points_blocks.append(new_pts)
                    new_sdf_blocks.append(sdf_vals)  # same SDF values for the mirrored points

                # extend our master lists
                points_list.extend(new_points_blocks)
                sdf_list.extend(new_sdf_blocks) 

            # now concatenate everything at once
            new_points = torch.cat(points_list, dim=0)
            new_SDF    = torch.cat(sdf_list,    dim=0)

            return SDF, new_points, new_SDF  # return the original SDF and the mirrored points and SDF values 



                    


  