

import torch 
import torch.nn as nn

class MLP_Layer(nn.Module):
    """
    MLP Layer: A layer in a multi-layer perceptron (MLP) with optional batch normalization and dropout.

    This function sets up a linear layer and defines the forward pass through the layer. 
    """
    def __init__(self,
                 hparams_MLP: dict, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True, 
                 ): 
        super(MLP_Layer, self).__init__()


        self.activation = hparams_MLP['activation_function']
        self.use_batch_norm = hparams_MLP['use_batch_norm']
        self.use_dropout = hparams_MLP['use_dropout'] 
        self.dropout_rate = hparams_MLP['dropout_rate']  

        # Set up the activation function
        if self.activation == 'relu':
            self.activation_fn = nn.ReLU()
            weight_init = nn.init.kaiming_uniform_
        elif self.activation == 'leaky_relu': 
            self.activation_fn = nn.LeakyReLU()
            weight_init = nn.init.kaiming_uniform_ 
        elif self.activation == 'elu':
            self.activation_fn = nn.ELU()
            weight_init = nn.init.kaiming_uniform_
        elif self.activation == 'tanh':
            self.activation_fn = nn.Tanh()
            weight_init = nn.init.xavier_uniform_ 
        elif self.activation == 'softplus':
            self.activation_fn = nn.Softplus()
            weight_init = nn.init.xavier_uniform_
        elif self.activation == 'softsign':
            self.activation_fn = nn.Softsign()
            weight_init = nn.init.xavier_uniform_
        elif self.activation == 'sigmoid':
            self.activation_fn = nn.Sigmoid()
            weight_init = nn.init.xavier_uniform_
        else:
            raise ValueError("Unsupported activation function. Choose from 'relu', 'leaky_relu', 'elu', 'tanh', 'softplus', 'softsign', or 'sigmoid'.")

        # Initialize the linear layer with the given input and output features
        self.linear = nn.Linear(in_features, out_features, bias=bias) 

        # Initialize the weights of the linear layer
        weight_init(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)  

        # Optional BatchNorm and Dropout layers
        self.batch_norm = nn.BatchNorm1d(out_features) if self.use_batch_norm else None
        self.dropout = nn.Dropout(p=self.dropout_rate) if self.use_dropout else None

        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP layer."""
        x = self.linear(x)
        x = self.activation_fn(x)
        if self.use_batch_norm == True:
            x = self.batch_norm(x)
        if self.use_dropout == True:
            x = self.dropout(x)
        return x



