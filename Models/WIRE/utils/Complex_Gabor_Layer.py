import torch
import torch.nn as nn
import math 

class ComplexGaborLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity -> Copied from WIRE Paper!  
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, 
                 in_features,
                 out_features,
                 bias=True,
                 is_first=False, 
                 omega0=10.0, 
                 sigma0=40.0,
                 trainable=False): 
        super().__init__()

        # Save parameters for forward pass 
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        self.in_features = in_features
        
        # Set the data type based on whether it is the first layer or not 
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        # Define the linear layer 
        self.linear = nn.Linear(in_features,out_features,bias=bias,dtype=dtype)
    
    # Forward pass of the complex Gabor layer 
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        return torch.exp(1j*omega - scale.abs().square())
