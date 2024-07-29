from torch import nn
import numpy as np
import torch


class PositionalEncoding(nn.Module):
    def __init__(self, n_poses_max: int = 500, d_model: int = 512):
        super(PositionalEncoding, self).__init__()

        self.n_poses = n_poses_max

        self.n_dims = d_model

        # the angle is calculated as following
        angle = lambda pos, i: pos / 10000 ** (i / self.n_dims)

        # let's initialize the different token positions
        poses = np.arange(0, self.n_poses)

        # let's initialize also the different dimension indexes
        dims = np.arange(0, self.n_dims)

        # let's initialize the index of the different positional vector values
        circle_index = np.arange(0, self.n_dims / 2)

        # let's create the possible combinations between a position and a dimension index
        xv, yv = np.meshgrid(poses, circle_index)

        # let's create a matrix which will contain all the different points initialized
        points = np.zeros((self.n_poses, self.n_dims))

        # let's calculate the circle y axis coordinates
        points[:, ::2] = np.sin(angle(xv.T, yv.T))

        # let's calculate the circle x axis coordinates
        points[:, 1::2] = np.cos(angle(xv.T, yv.T))

        self.register_buffer("pe", torch.from_numpy(points).unsqueeze(0))

    def forward(self, input_: torch.Tensor):

        # let's scale the input
        input_ = input_ * torch.sqrt(torch.tensor(self.n_dims))

        # let's recuperate the result of the sum between the input and the positional encoding vectors
        return input_ + self.pe[:, : input_.size(1), :].type_as(input_)
