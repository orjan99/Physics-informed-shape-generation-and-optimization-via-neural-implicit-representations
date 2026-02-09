import torch
import torch.nn as nn
from Models.MLP.utils.MLP_layer import MLP_Layer

class MLP(nn.Module):
    def __init__(self,hparams_MLP,num_inputs, model_type:str):
        super().__init__()
        self.layers = hparams_MLP['layers']
        self.skip_connection = hparams_MLP['skip_connection'] 
        self.dim = hparams_MLP['dimensionality']  # Dimensionality of the problem - 2D or 3D 

        # Define the input and output dimensions:
        if model_type == 'PINN': 
            output_dim = int(self.dim) # Output dimension is 2 for 2D and 3 for 3D 
        elif model_type == 'GINN':
            output_dim = 1 
        else:
            raise ValueError("Invalid model type. Choose either 'PINN' or 'GINN'.") 
        input_dim  = int(num_inputs)
        self.layers = [int(layer) for layer in self.layers] # Convert the layers to integers
        dims = [input_dim] + self.layers + [output_dim]  
        self.middle_skip_connection_idx = (2 + len(self.layers)) // 2  # Index of the middle layer for the skip connection 

        
        # Initialize the MLP layers 
        self.net = nn.ModuleList()  # Define the MLP layers in a ModuleList

        # Fist layer is a linear layer with input_dim and first hidden layer dimension  

        self.net.append(MLP_Layer(hparams_MLP, in_features=dims[0], out_features=dims[1], bias=True))

        for i in range(1, len(self.layers)):
            if i == self.middle_skip_connection_idx and self.skip_connection == True:  # Middle layer has modified dimensionality due to the skip connection
                skip_connection_dim = dims[i] + input_dim  # Concatenate the input coordinates with the output of the previous layer
                self.net.append(MLP_Layer(hparams_MLP, in_features=skip_connection_dim, out_features=dims[i+1], bias=True))

            else:  # All other layers are MLP layers with the same dimensionality
                self.net.append(MLP_Layer(hparams_MLP, in_features=dims[i], out_features=dims[i+1], bias=True))

        # Final layer is a linear layer that maps the output of the last MLP layer to 4 outputs - no sine activation
        if self.skip_connection == True:
            final_layer_input_dim = dims[-2] + dims[self.middle_skip_connection_idx]  # Concatenate the output of the last MLP layer with the output of the middle layer
        else:
            final_layer_input_dim = dims[-2]  # No skip connection, just the output of the last MLP layer
        self.net.append(nn.Linear(final_layer_input_dim, dims[-1]))  

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        
        layer_input = None
        layer_output_middle = None  # Initialize the variable to store the output of the middle layer   
          
  
        for idx,layer in enumerate(self.net):

            if idx == 0:  
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

        return model_output

