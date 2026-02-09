import torch
import torch.nn as nn
from Models.WIRE.utils.Complex_Gabor_Layer import ComplexGaborLayer
from Models.WIRE.utils.Real_Gabor_Layer import RealGaborLayer 

class WIRE(nn.Module):
    def __init__(self,WIRE_hparms: dict, num_inputs,model_type:str):  
        super().__init__()
        self.layers = WIRE_hparms['layers']          # list of hidden layer widths, e.g. [64,64,64,64] 
        self.w0 = WIRE_hparms['w0']                  # omega_0 for hidden layers
        self.w0_initial = WIRE_hparms['w0_initial']  # omega_0 for first layer
        self.sigma0 = WIRE_hparms['sigma0']          # sigma_0 for hidden layers
        self.sigma0_initial = WIRE_hparms['sigma0_initial']  # sigma_0 for first layer 
        self.dim = WIRE_hparms['dimensionality']     # dimensionality of the problem - 2D or 3D
        self.layer_type = WIRE_hparms['layer_type']  # Type of layer to use: 'complex_gabor' or 'real_gabor'
        self.trainable = WIRE_hparms['trainable']    # Whether layer parameters are trainable or not -- boolean  
        self.skip_connection = WIRE_hparms['skip_connection']  # Whether to use skip connections or not -- boolean 

        # Define the input and output dimensions 
        if model_type == 'PINN': 
            output_dim = int(self.dim) # Output dimension is 2 for 2D and 3 for 3D 
        elif model_type == 'GINN':
            output_dim = 1 
        else:
            raise ValueError("Invalid model type. Choose either 'PINN' or 'GINN'.") 
        input_dim  = int(num_inputs)
        self.layers = [int(layer) for layer in self.layers] # Convert the layers to integers   
 

        # build full architecture:
        dims = [input_dim] + self.layers + [output_dim] 

        self.middle_skip_connection_idx = (2 + len(self.layers)) // 2

        # Define the WIRE layers in a ModuleList  
        self.net = nn.ModuleList()

        # Defint the first layer 
        if self.layer_type == 'complex_gabor': 
            self.net.append(ComplexGaborLayer(in_features=dims[0], 
                                              out_features=dims[1],
                                              is_first=True, 
                                              omega0=self.w0_initial, 
                                              sigma0=self.sigma0_initial,
                                              trainable=self.trainable))
        elif self.layer_type == 'real_gabor':
            self.net.append(RealGaborLayer(in_features=dims[0], 
                                           out_features=dims[1], 
                                           is_first=True, 
                                           omega0=self.w0_initial, 
                                           sigma0=self.sigma0_initial,
                                           trainable=self.trainable)) 
        else:
            raise ValueError("Invalid layer type. Choose from 'complex_gabor' or 'real_gabor'.") 
        
        # Define the hidden layers
        for i in range(1, len(self.layers)):
            if self.skip_connection and i == self.middle_skip_connection_idx:
                in_features = dims[i] + input_dim
            else:
                in_features = dims[i]
            if self.layer_type == 'complex_gabor':
                self.net.append(ComplexGaborLayer(in_features=in_features,
                                                  out_features=dims[i+1],
                                                  is_first=False, 
                                                  omega0=self.w0, 
                                                  sigma0=self.sigma0,
                                                  trainable=self.trainable))
            elif self.layer_type == 'real_gabor':
                self.net.append(RealGaborLayer(in_features=in_features, 
                                               out_features=dims[i+1], 
                                               is_first=False, 
                                               omega0=self.w0, 
                                               sigma0=self.sigma0,
                                               trainable=self.trainable
                                               ))
            else:
                raise ValueError("Invalid layer type. Choose from 'complex_gabor' or 'real_gabor'.")  
        
        # Define the final linear layer
        if self.skip_connection:
            final_layer_input_dim = dims[-2] + dims[self.middle_skip_connection_idx]
        else:
            final_layer_input_dim = dims[-2]
        self.net.append(nn.Linear(final_layer_input_dim, dims[-1]))

        # Initialize the final layer weights 
        nn.init.xavier_uniform_(self.net[-1].weight)
        if self.net[-1].bias is not None:
            nn.init.zeros_(self.net[-1].bias)  
        
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:  
        x = input_features
        skip_input = input_features  # Save for skip connections

        for idx, layer in enumerate(self.net):
            if self.skip_connection and idx == self.middle_skip_connection_idx:
                x = torch.cat((skip_input, x), dim=-1)
            
            # Final linear layer
            if idx == len(self.net) - 1:
                if self.skip_connection:
                    x = torch.cat((x, skip_output), dim=-1)
                x = layer(x)
            else:
                x = layer(x)
                if self.skip_connection and idx == self.middle_skip_connection_idx:
                    skip_output = x  # Store for final skip

        return x.real if self.layer_type == 'complex_gabor' else x
