import torch 
import torch.nn as nn
import torch.nn.functional as F

class STN(nn.Module):
    ''' 
        Implements a spatial transfomer as specified in https://arxiv.org/pdf/1506.02025.pdf
        adapted from https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
        Goal: learn general scale / orientation invariance of user input
    '''
    
    def __init__(self, dim_reg=128):
        super(STN, self).__init__()
        
        # stn loc
        self.localizer = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2))
        
        # stn reg
        self.regressor = nn.Sequential(
            nn.LazyLinear(dim_reg),
            nn.ReLU(True),
            nn.Linear(dim_reg, 6))

        # affine init
        self.regressor[2].weight.data.zero_()
        self.regressor[2].bias.data.copy_(torch.tensor([1,0,0,
                                                        0,1,0], dtype=torch.float))
        
    def forward(self, x):
        x_spatial = self.localizer(x).view(x.shape[0], -1)
        theta = self.regressor(x_spatial).view(x.shape[0], 2, 3)
        grid = F.affine_grid(theta, x.shape)
        x_registered = F.grid_sample(x, grid)
        return x_registered