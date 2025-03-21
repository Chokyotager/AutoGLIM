import torch
from torch import nn
import torch.nn.functional as F

from segformer import Segformer

class AutoGolgiSegformer (nn.Module):
    
    def __init__ (self):

        super().__init__()
        self.leaky_relu = nn.LeakyReLU()

        # MiT-B1
        self.segformer = Segformer(
            dims = (8, 16, 32, 64),      # dimensions of each stage
            heads = (1, 2, 4, 8),           # heads of each stage
            ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
            reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
            num_layers = 2,                 # num layers of each stage
            decoder_dim = 32,              # decoder dimension
            final_decoder_dim = 32,          # final decoder dimension,
            channels = 3
        )

        self.deconv = nn.ConvTranspose2d(32, 1, stride=4, kernel_size=4, padding=0)

    def forward (self, x, clamp=True):

        x = self.segformer(x)
        x = self.deconv(x)

        x = torch.squeeze(x, dim=1)

        if clamp:

            x = torch.clamp(x, min=-10, max=10)

        # Return logits
        return x