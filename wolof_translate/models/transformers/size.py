
from torch import nn
import torch
class SizePredict(nn.Module):
    
    def __init__(self, input_size: int, target_size: int = 1, n_features: int = 100, n_layers: int = 1, normalization: bool = True, drop_out: float = 0.1):
        super(SizePredict, self).__init__()
        
        self.layers = nn.ModuleList([])
        
        for l in range(n_layers):
            
            # we have to add batch normalization and drop_out if their are specified
            self.layers.append(
                nn.Sequential(
                    nn.Linear(input_size if l == 0 else n_features, n_features),
                    nn.LayerNorm(n_features) if normalization else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(drop_out),
                )
            )
        
        # Initiate the last linear layer
        self.output_layer = nn.Linear(n_features, target_size)
    
    def forward(self, input_: torch.Tensor):
        
        # let's pass the input into the different sequences
        out = input_
        
        for layer in self.layers:
            
            out = layer(out)
        
        # return the final result (you have to take the absolute value of the result to make the number positive)
        return self.output_layer(out)
        
        
