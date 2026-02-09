import math
import torch
import torch.nn as nn
#from Models.SIREN.SIREN_layer import SirenLayer # SIREN layer class 
from Models.SIREN.Utils.SIREN_layer import SirenLayer  

# Later: Add the conditioning input q to the input --> x,y,z coordinates + [z; q] --> z is the latent code
         # Allows us create a fully generative model --> for multiple topology optimization problems 

class Multi_Head_SIREN(nn.Module):
    """
    SIREN model for topology optimization:
    This model builds on the SIREN model by including an additional output for density value
    The density value is used as input for the topology optimization --> implicit density field

    input:  --> shape = (batch_size, 3 + nz)
    1. 3D coordinates(x,y,z) for a single point
    2. latent code (z)

    output: 
    1. signed distance value (SDF) --> (batch_size, 1) 
    2. density value --> in the range [0,1] --> used for TOPOLOGY Optimization --> (batch_size, 1)
 .
    """

    # constructor function - takes in the hyperparameters for the SIREN model as input
    def __init__(self, SIREN_hparms: dict): 
        super().__init__()
        # Assign the hyperparameters to the class variables
        self.layers = SIREN_hparms['layers']         # list of hidden layer widths, e.g. [64,64,64,64]
        self.w0 = SIREN_hparms['w0']                 # omega_0 for hidden layers
        self.w0_initial = SIREN_hparms['w0_initial'] # omega_0 for first layer
        self.nz = SIREN_hparms['latent_dim']         # dimensionality of the latent code

        # build full architecture:
        # network input = x,y,z coordinates for input point + latent code (z) 
        # input_dim = 3 + nz --> 3D coordinates + latent code
        # output_dim = 2 --> signed distance value + density
        dims = [3 + self.nz] + self.layers + [2]
        final_SIREN_dim = dims[-2]  # dimension of the last SIREN layer - after this the networks branch out 


        self.net = nn.ModuleList() 

        # first layer is a SIREN layer with a different omega_0 
        self.net.append(SirenLayer(in_features=dims[0],out_features=dims[1],is_first=True,omega_0=self.w0_initial))

        #  remaining SIREN layers 
        for i in range(1, len(self.layers)):
            self.net.append(SirenLayer(in_features=dims[i],out_features=dims[i+1],is_first=False,omega_0=self.w0))

        # Multi-head output layers
        self.sdf_head = nn.Linear(final_SIREN_dim, 1) # SDF head --> signed distance value
    
        self.density_head = nn.Sequential(
                            nn.Linear(final_SIREN_dim, final_SIREN_dim//2),
                            nn.ReLU(),
                            nn.Linear(final_SIREN_dim//2, 1),
                            nn.Sigmoid()
                         )

        # weight init for final layer per SIREN paper
        bound = math.sqrt(6.0 / dims[-2]) / self.w0 
        nn.init.uniform_(self.final_linear.weight, -bound, bound)
        nn.init.uniform_(self.final_linear.bias,   -bound, bound)

    def forward(self, coords: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SIREN model: --> Shape = (batch_size, 3 + nz)
        - coords: 3D coordinates of the input point 
        - z: latent code

        output: 
        - signed distance value (batch_size, 1)
        - density value (batch_size, 1)

        """
        # Concatenate latent code to coordinates --> Shape = (batch_size, 3 + nz) 
        x = torch.cat([coords, z], dim=-1)

        # Pass through sinusoidal layers
        for layer in self.net:
            x = layer(x)

        # Multi-head outputs
        sdf = self.sdf_head(x)
        density = self.density_head(x)
            
        # Final linear output
        return sdf, density
    
    

  