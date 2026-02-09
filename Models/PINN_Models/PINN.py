
import torch 
import torch.nn as nn
import numpy as np
from Models.SIREN.SIREN import SIREN # SIREN PINN model 
from Models.WIRE.WIRE import WIRE # WIRE PINN model  
from Models.PINN_Models.Utils.feature_expansion import expand_input_features_sine # Positional encoding functiom
from Models.PINN_Models.Utils.feature_expansion import expand_input_features_squared # Positional encoding functiom
from Models.PINN_Models.Utils.feature_expansion import generate_fourrier_features # Positional encoding functiom
from Models.MLP.MLP import MLP



class PINN(nn.Module):

    """
    Physics Informed Neural Network  for calculating the displacement field in a 3D domain.
    - Can be used for both 2D and 3D problems
    
    Inputs:
    ---------------
    - input_domain: Tells us the vertices of the input domain --> Array of shape (1,2*dim) --> [x_min,x_max,y_min,y_max,z_min,z_max]
    - boundary_conditions: The boundary conditions for the problem
    - features: function to generate additional input features for the model - positional encoding
    - nn_model: The neural network model to be used --> SIREN_PINN

    Outputs:
    - displacement field: The displacement field

    Notes: 
    1. The boundary conditions are enforced directly in the forward pass of the model, rather than using a loss function.
    2. The model uses the density to compute the physics loss, but the density is not used in the forward pass of the model.
    3. The geometry of the problem is defined by the input domain and the boundary conditions.
    ----------------

    """

    # PINN consructor function
    def __init__(self, test_case, feature_expansion: dict ,model_hyperparameters):
        
        super(PINN, self).__init__()
        self.domain = np.array(test_case.domain, dtype=np.float32)   # Array of shape (1,6) for 3D or (1,4) for 2D 
        self.model_hparams = model_hyperparameters                   # SIREN hyperparameters - for SIREN_PINN model
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

        # Initialize the SIREN PINN model - this is the main model that will be used to predict the displacement field
        if self.model_hparams['Model_type'] == 'SIREN': 
            self.model = SIREN(model_hyperparameters,self.num_inputs,'PINN') # SIREN PINN model   
        elif self.model_hparams['Model_type'] == 'WIRE':
            self.model = WIRE(model_hyperparameters,self.num_inputs,'PINN') # WIRE PINN model 
        elif self.model_hparams['Model_type'] == 'MLP':
            self.model = MLP(model_hyperparameters,self.num_inputs,'PINN') 
        else:
            raise ValueError("Invalid model type. Choose from 'SIREN', 'WIRE' or 'MLP' in the model_hyperparameters dictionary.") 

        if self.feature_expansion == 'sine_features':
            self.feature_object = expand_input_features_sine(self.dim) # = [x, y, z, sin(x), sin(y) ,sin(z)]
        elif self.feature_expansion == 'squared_features':
            self.feature_object = expand_input_features_squared(self.dim) 
        elif self.feature_expansion == 'fourrier_features':
            self.feature_object = generate_fourrier_features(self.dim,self.feature_expansion_hparams)  
        else:
            self.feature_object = None # No feature expansion  


    # PINN forward pass   
    def forward(self, coords, implicit_density=None): 
        """
        Forward pass of the PINN model:
        Input:
        - coords: coordinates of the input point (x,y,z) --> shape (N,3) for 3D or (N,2) for 2D
        Output:
        - displacement field --> shape (N,3) for 3D or (N,2) for 2D
        """


        #2. Generate the positional encoding for the input coordinates --> (Batch_size, 6)
        # input features = x,y,z coordinates 
        if self.feature_object is not None:
            modified_input_features = self.feature_object(coords) # --> (Batch_size, 2*dim) for sine or squared features
        else:
            modified_input_features = coords 
 
        #3. Use the SIREN PINN model to predict the displacement field --> (Batch_size, 3) for 3D or (Batch_size, 2) for 2D
        # Raw displacement field --> Have not enforced boundary conditions yet
        displacement_field = self.model(modified_input_features)

        #4. Strongly Enforce boundary conditions

        '''Outdated BC enforcement method --> Need to fix this'''
        # if self.dim == 2:
        #     dirichlet_mask = self.test_case.enforce_dirichlet_boundary_conditions(coords) 
        #     displacement_field = displacement_field * dirichlet_mask.view(-1,1) # Enforce Dirichlet BC on the displacement field 
        # elif self.dim == 3:
        #     dirichlet_mask = self.test_case.enforce_dirichlet_boundary_conditions(coords) 
        #     displacement_field = displacement_field * dirichlet_mask.view(-1,1) # Enforce Dirichlet BC on the displacement field
        # else:
        #     raise ValueError("Invalid dimensionality. Only 2D and 3D are supported.")

        if implicit_density is not None:
            # Enforce that displacements are zero outside the geometric surface volume  
            displacement_field = displacement_field * implicit_density.view(-1,1)  
    
        return displacement_field   

  