
import torch 
import torch.nn as nn
import math 


class SirenLayer(nn.Module):
    """
    SIREN layer:
        y = sin(w0 * (Wx + b))
    where:
        W is the weight matrix,
        b is the bias vector,
        w0 is a frequency parameter.
    This layer applies a linear transformation followed by a sine activation function.

    Inputs to constructor:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Whether to include a bias term in the linear transformation.
        is_first: Whether this is the first layer in the network.
        omega_0: Frequency parameter for the sine activation function.

    Outputs:
        x: Output tensor after applying the linear transformation and sine activation function.
    """
    # constructor function 
    def __init__(self,in_features: int, out_features: int, bias: bool = True, is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        # Initialize the linear layer with the given input and output features 
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    # initialize the weights of the linear layer according to the SIREN paper 
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.linear.in_features
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))