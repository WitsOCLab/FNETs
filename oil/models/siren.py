import torch
import torch.nn as nn
import numpy as np

class SirenLayer(nn.Module):
    """
    Represents a single layer of a Sinusoidal Representation Network (SIREN).
    It combines a linear transformation with a sine activation function.
    
    The initialisation of weights is critical and follows the scheme from the
    original SIREN paper [1].
    """
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0,
                 append_inputs:bool = False, sin_cos_split:bool = False):
        super().__init__()
        self.omega_0 = float(omega_0)
        self.is_first = is_first

        self.append_inputs = append_inputs
        self.sin_cos_split = sin_cos_split
        
        if sin_cos_split:
            self.linear = nn.Linear(in_features, 2*out_features, bias=False)
        else:
            self.linear = nn.Linear(in_features, out_features)
        
        self.init_weights()

    def init_weights(self):
        """
        Initialise the weights of the linear layer according to the SIREN paper.
        """
        with torch.no_grad():
            if self.is_first:
                # Special initialization for the first layer
                self.linear.weight.uniform_(-1 / self.linear.in_features, 
                                             1 / self.linear.in_features)
            else:
                # Standard initialization for hidden layers
                limit = np.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-limit, limit)

    def forward(self, x):
        """
        Forward pass of the SirenLayer.
        """
        if self.sin_cos_split:
            y = torch.cat([torch.sin(self.omega_0 * self.linear(x)),torch.sin(self.omega_0 * self.linear(x))], dim=-1)
        else:
            y = torch.sin(self.omega_0 * self.linear(x))
        if self.append_inputs:
            y = torch.cat((x,y),dim=-1)

        return y

class SirenNet(nn.Module):
    """
    A complete SIREN network.
    """
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, omega_0=30.0):
        super().__init__()
        
        self.net = nn.ModuleList()

        # First layer
        self.net.append(SirenLayer(in_features, hidden_features, 
                                   is_first=True, omega_0=omega_0))

        # Hidden layers
        for _ in range(hidden_layers):
            self.net.append(SirenLayer(hidden_features, hidden_features, 
                                       is_first=False, omega_0=omega_0))

        # Final layer: a standard linear layer to map to the output range
        self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        """
        Forward pass through the entire network.
        
        Note: The input 'x' is expected to be normalised to the range [-1, 1].
        """
        for layer in self.net:
            x = layer(x)
        
        return self.final_layer(x)

# --- Example Usage ---
# This demonstrates how to instantiate and use the network.
# You would integrate this into your training script.

if __name__ == '__main__':
    # Define network parameters
    input_dim = 2  # e.g., (x, y) coordinates
    hidden_dim = 256
    output_dim = 3 # e.g., (R, G, B) pixel values
    num_hidden_layers = 4
    
    # Create the network
    model = SirenNet(
        in_features=input_dim,
        hidden_features=hidden_dim,
        hidden_layers=num_hidden_layers,
        out_features=output_dim
    )
    
    print("SIREN Model Architecture:")
    print(model)
    
    # Create a dummy input tensor
    # The input should be normalised to [-1, 1] for best performance
    batch_size = 10
    dummy_input = torch.rand(batch_size, input_dim) * 2 - 1 # Normalise to [-1, 1]
    
    # Perform a forward pass
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
