import math
import torch
import torch.nn as nn
from Models.SIREN.Utils.SIREN_layer import SirenLayer # SIREN layer class 

class SIREN(nn.Module): 
    """
    SIREN model physics informed neural network (PINN): 
    ---------------------------------------------------------------------------------------------------
    -  Can be used for both 2D and 3D problems 
     - Used to predict the displacement field 
     - Will be used as the backbone for the PINN model

    input: 
    1. Point coordinates (x,y) or (x,y,z) 

    output:  
    1. x displacement (u)
    2. y displacement (v)
    3. z displacement (w) --> 3D only 
    --------------------------------------------------------------------------------------------------- 

    SIREN model for geometry informed neural networks (GINN): 
    --------------------------------------------------------------------------------------------------- 
    input: 
    1. Point coordinates (x,y) or (x,y,z) 

    output:  
    1. SDF 
    ---------------------------------------------------------------------------------------------------
    The network has two residual connections:
    The first one goes from the input coordinates to the hidden layer 3, concatenating the input coordinates with the output of hidden layer 2
    The second one goes from the hidden layer 3 to the output layer, concatenating the output of hidden layer 3 with the output of the final hidden layer 
    ---------------------------------------------------------------------------------------------------
    """

    # constructor function - takes in the hyperparameters for the SIREN model as input
    def __init__(self,SIREN_hparms: dict, num_inputs, model_type: str = 'PINN'):   
        super().__init__()
        # Assign the hyperparameters to the class variables
        self.layers = SIREN_hparms['layers']         # list of hidden layer widths, e.g. [64,64,64,64] 
        self.w0 = SIREN_hparms['w0']                 # omega_0 for hidden layers
        self.w0_initial = SIREN_hparms['w0_initial'] # omega_0 for first layer
        self.dim = SIREN_hparms['dimensionality']    # dimensionality of the problem - 2D or 3D
        self.skip_connection = SIREN_hparms['skip_connection'] # boolean to indicate if skip connection is used in the model 

        # Define the input and output dimensions 
        if model_type == 'PINN': 
            output_dim = int(self.dim) # Output dimension is 2 for 2D and 3 for 3D 
        elif model_type == 'GINN':
            output_dim = 1 
        else:
            raise ValueError("Invalid model type. Choose either 'PINN' or 'GINN'.") 
        input_dim  = int(num_inputs)
        self.layers = [int(layer) for layer in self.layers] 

    
        # network input = x,y,z coordinates, positional encoding 
        # input_dim = 6 --> point coordinates + expanded features (x,y,z,sin(x),sin(y),sin(z))
        # output_dim = 3 -->  x,y,z displacements (for 3D) or x,y displacements (for 2D) 
        dims = [input_dim] + self.layers + [output_dim]  

        self.middle_skip_connection_idx = (2 + len(self.layers)) // 2  # Index of the middle layer for the skip connection 

        # Define the SIREN layers in a ModuleList  
        self.net = nn.ModuleList()  

        # Define the first layer as a SIREN layer with a different omega_0 
        self.net.append(SirenLayer(in_features=dims[0],out_features=dims[1],is_first=True,omega_0=self.w0_initial)) 

        for i in range(1, len(self.layers)): 
            if i == self.middle_skip_connection_idx and self.skip_connection == True: # Middle layer has modified dimensionality due to the skip connection
                skip_connection_dim = dims[i] + input_dim  # Concatenate the input coordinates with the output of the previous layer
                self.net.append(SirenLayer(in_features=skip_connection_dim,out_features=dims[i+1],is_first=False,omega_0=self.w0))

            else: # All other layers are SIREN layers with the same omega_0 and same dimensionality
                self.net.append(SirenLayer(in_features=dims[i],out_features=dims[i+1],is_first=False,omega_0=self.w0)) 

        # Final layer is a linear layer that maps the output of the last SIREN layer to 4 outputs - no sine activation
        if self.skip_connection == True: 
            final_layer_input_dim = dims[-2] + dims[self.middle_skip_connection_idx]  # Concatenate the output of the last SIREN layer with the output of the middle layer
        else:
            final_layer_input_dim = dims[-2]  # No skip connection, just the output of the last SIREN layer 
        self.net.append(nn.Linear(final_layer_input_dim, dims[-1]))  

        # weight init for final layer per SIREN paper --> Rest of layer weights are initialized with the SirenLayer class 
        bound = math.sqrt(6.0 / (final_layer_input_dim)) / self.w0
        nn.init.uniform_(self.net[-1].weight, -bound, bound)
        nn.init.uniform_(self.net[-1].bias,   -bound, bound)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SIREN model: 

        Input:
        - coordinates of the input point 

        output: -> Shape = (batch_size, 3)
        - x displacement (u) 
        - y displacement (v) 
        - z displacement (w)  --> 3D only

        The network has two residual connections:
        The first one goes from the input coordinates to the hidden layer 3, concatenating the input coordinates with the output of hidden layer 2
        The second one goes from the hidden layer 3 to the output layer, concatenating the output of hidden layer 3 with the output of the final hidden layer 
        """
    
        layer_input = None
        layer_output_middle = None    
          
        # Pass through SIREN layers with residual block connections for each layer 
        for idx,layer in enumerate(self.net):

            if idx == 0:  
                # First layer, use the input coordinates directly 
                layer_input = coords
            elif idx == self.middle_skip_connection_idx: 
                # This is the middle layer, concatenate the input coordinates with the output of the previous layer
                if self.skip_connection == True:
                    layer_input = torch.cat(([coords, layer_output]), dim=-1)
                else:
                    layer_input = layer_output 
                layer_output_middle = layer_output # Store the output of the layer for the skip connection to the final layer
            elif idx == len(self.net) - 1: 
                # Last layer, concatenate the output of the middle layer with the output of the previous layer
                if self.skip_connection == True:
                    # Skip connection from the middle layer to the final layer 
                    layer_input = torch.cat((layer_output_middle, layer_output), dim=-1)
                else:
                    # No skip connection, just pass the output of the previous layer to the final layer
                    layer_input = layer_output 
                final_layer_output = layer(layer_input)  # Final linear layer to get the displacements  
            else:
                # No skip connection, just pass the output to the next layer   
                layer_input = layer_output 

            layer_output = layer(layer_input)  

        # Final linear output
        model_output = final_layer_output 

        # Note: the derivatives dx, dy, dz are not computed here --> computed in the loss function 
        return model_output

