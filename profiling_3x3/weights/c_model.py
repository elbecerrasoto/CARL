#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:31:40 2020

@author: ebecerra
"""

import torch
import torch.nn as nn

class DQN(nn.Module):
    fc1 = 256
    fc2 = 512
    fc3 = 256
    fc4 = 64
    def __init__(self, grid_shape, n_actions):
        super(DQN, self).__init__()
        
        grid_size = self._get_size(grid_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(grid_size + 3, self.fc1),
            nn.ReLU(),
            nn.Linear(self.fc1, self.fc2),
            nn.ReLU(),
            nn.Linear(self.fc2, self.fc3 ),
            nn.ReLU(),
            nn.Linear(self.fc3, self.fc4 ),
            nn.ReLU(),
            nn.Linear(self.fc4, n_actions))

    def forward(self, grids, positions, moves):
        # Flatten grid
        x = torch.flatten(grids, start_dim=1)
        
        # Concatenating positions and moves
        x = torch.cat((x, positions, moves), dim=1)
        
        # Feeding to fully connected
        return self.fc(x)

    def _get_size(self, shape):
        size = 1
        for dim_size in shape:
            size *= dim_size
        return size
