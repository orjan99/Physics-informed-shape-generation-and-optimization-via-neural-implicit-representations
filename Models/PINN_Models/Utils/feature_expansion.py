
import torch
import torch.nn as nn
import math 

# --------------------------------------------------------------------------------------------------------------------------
class expand_input_features_sine(nn.Module):
    """
    This functions expands the dimensionality of the input features from 3 (x,y,z) to 6 (x,y,z,sin(x),sin(y),sin(z)).
    
    - Inputs:  (Batch Size, 3) --> tensor of (x, y, z) coords for 3D or (x, y) coords for 2D
    - Outputs: (Batch Size, 6) --> tensor [x, y, z, sin(x), sin(y), sin(z)] --> 3D or [x, y, sin(x), sin(y)] --> 2D

    Note: write this as Pytorch module to enable automatic differentiation, and backpropagation 
    """

    def __init__(self,dim):
        super(expand_input_features_sine, self).__init__()
        self.input_dim = dim
        self.output_dim = 2 * dim

    def forward(self, input_coords):

        # Apply sine function to the input coordinates and concatenate with the original coordinates
        new_feature_vector = torch.cat((input_coords, torch.sin(input_coords)), dim=1) # Shape = (Batch_size, 6) for 3D or (Batch_size, 4) for 2D
        
        return new_feature_vector 
# --------------------------------------------------------------------------------------------------------------------------

class generate_fourrier_features(nn.Module):
    """
    - This function uses fourrier features to expand the dimensionality of the input features (x,y,z)
    - Fourrier features are used to represent the input coordinates in a higher dimensional space, similarly to sine features in the function above.
    - Fourrier features maps the input coordinates to a higher dimensional space than the sine features
    - This will help the model capture the high-frequency components of the input data and handle more complex geomettries. 
    - Fourrier features use both sine and cosine functions to represent the input coordinates in a higher dimensional space.
    - This function replaces the original coordinates, instead of adding the sine features to the original coordinates.

    Inputs:  
    - Point Coordinates --> (Batch Size, dim) --> tensor of (x, y, z) coords for 3D 
    
    Outputs:
    - Fourrier features --> (Batch Size, 2 * dim * num_frequencies)
    """
    def __init__(self, dim, fourrier_feature_parameters:dict):
        super(generate_fourrier_features, self).__init__()
        self.num_frequencies = fourrier_feature_parameters['Num Frequencies'] # Number of frequencies to use for the fourrier features
        self.max_frequency = fourrier_feature_parameters['Max Frequency'] # Maximum frequency to use for the fourrier features
        self.input_dim = dim                              # Dimensionality of the input coordinates - 2D or 3D
        self.output_dim = dim + 2 * dim * self.num_frequencies   # Number of output features = 2 * dim * num_frequencies
        
        # Define the frequencies to use for the fourrier features --> Logarithmically spaced frequencies
        #self.frequencies = torch.tensor([2 ** i for i in range(self.num_frequencies)], dtype=torch.float32)

        #frequencies = torch.tensor([2 ** i for i in range(self.num_frequencies)],dtype=torch.float32)

        # Log spaced frequencies from 2^0 to 2^max_frequency 
        frequencies = torch.logspace(
            start=0.0,
            end=math.log2(self.max_frequency),
            steps=self.num_frequencies,
            base=2.0,
            dtype=torch.float32
        )
        self.register_buffer('freqs', frequencies)


    def forward(self, input_coords):
        """
        Forward pass of the fourrier feature expansion module:
        Input:
        - input_coords: coordinates of the input point (x,y,z) --> shape (N,3) for 3D or (N,2) for 2D
        Output:
        - fourrier_features: fourrier features of the input coordinates --> shape (N, 2 * dim * num_frequencies)
        Output features = [sin(2^0*x), cos(2^0*x), sin(2^1*x), cos(2^1*x), sin(2^2*x), cos(2^2*x), ...]
        """
        # expand input_coords to (B, D, 1) then multiply by frequencies (1, 1, L) → (B, D, L)
        coords_expanded = 2*math.pi*input_coords.unsqueeze(-1) * self.freqs.view(1, 1, -1) 

        # compute sin and cos → each (B, D, L)
        sin_features = torch.sin(coords_expanded)
        cos_features = torch.cos(coords_expanded)

        # concatenate along last dim → (B, D, 2L)
        fourrier_features = torch.cat((sin_features, cos_features), dim=-1) 

        # reshape to (B, D*2L)
        fourrier_features = fourrier_features.view(input_coords.shape[0], -1)

        # Concatenate the original coordinates with the fourrier features
        fourrier_features = torch.cat((input_coords, fourrier_features), dim=1) 
        return fourrier_features
    
    
# --------------------------------------------------------------------------------------------------------------------------

class expand_input_features_squared(nn.Module):
    """
    This functions expands the dimensionality of the input features from 3 (x,y,z) to 6 (x, y, z, x^2, y^2, z^2).
    
    - Inputs:  (Batch Size, 3) --> tensor of (x, y, z) coords for 3D case
    - Outputs: (Batch Size, 6) --> tensor [x, y, z, x^2, y^2, z^2] for 3D case

    Note: We write this as Pytorch module to enable automatic differentiation, and backpropagation 
    """

    def __init__(self,dim):
        super(expand_input_features_sine, self).__init__()
        self.input_dim = dim
        self.output_dim = 2 * dim

    def forward(self, input_coords):
        # Square each input coordinate and concatenate with the original coordinates
        new_feature_vector = torch.cat((input_coords, input_coords ** 2), dim=1)
        
        return new_feature_vector 
    

# --------------------------------------------------------------------------------------------------------------------------
