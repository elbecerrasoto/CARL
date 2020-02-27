#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:01:57 2020

@author: ebecerra
"""

import helicopter
import numpy as np

# Environment parameters
N_ROW = 8
N_COL = 8
P_FIRE = 0.01
P_TREE = 0.30
# Symbols for cells
TREE = 3
FIRE = 7
EMPTY = 1

# Environment
env = helicopter.Helicopter(n_row = N_ROW, n_col = N_COL,
                            p_fire = P_FIRE, p_tree = P_TREE,
                            tree = TREE, fire = FIRE, empty = EMPTY)
N_ACTIONS = env.actions_cardinality

# Heuristic

# Check all neighbours and current cell
# If fire in any: random to one of them
# If not fire: move to a tree at random
# If not tree: move at random

# Fisrt observation
observation = env.reset()
# Initial lattice
env.reset()

grid, position = observation
p_row = position[0]
p_col = position[1]

# keep everything in a 3,3 array
neighborhood = grid[p_row-1:p_row+2, p_col-1:p_col+2].copy()

fire_cells=[]
tree_cells=[]
empty_cells=[]
for row_idx, row in enumerate(neighborhood):
    for col_idx, col in enumerate(neiborhood):
        if env.fire == neighborhood[row_idx,col_idx]:
            pass


total_reward = 0
for i in range(env.freeze * 100 + 1):
  print('.', end='')
  action = np.random.choice(np.arange(1,10))
  observation, reward, done, info = env.step(action)
  total_reward += reward
  env.render()

print('\nTotal Reward: {}'.format(total_reward))
