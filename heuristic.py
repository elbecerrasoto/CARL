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

# Heuristic

# Check all neighbours and current cell
# If fire in any: random to one of them
# If not fire: move to a tree at random
# If not tree: move at random

def get_neighborhood(grid, position):
    p_row = position[0]
    p_col = position[1]
    up_left_boundary = p_row == 0 and p_col == 0
    up_right_boundary = p_row == 0 and p_col == env.n_col-1
    down_left_boundary = p_row == env.n_col-1 and p_col == 0
    down_right_boundary = p_row == env.n_row-1 and p_col  == env.n_col-1
    up_boundary = p_row == 0 and not (up_left_boundary or up_right_boundary)
    down_boundary = p_row == env.n_row-1 and not (down_left_boundary or down_right_boundary)
    left_boundary = p_col == 0 and p_row > 0 and not (up_left_boundary or down_left_boundary)
    right_boundary = p_col == env.n_col-1 and not (up_right_boundary or down_right_boundary)
    if up_boundary:
        up = grid[p_row:p_row+2, p_col-1:p_col+2]
        dummy = np.repeat(env.empty, 3)[np.newaxis , :]
        neighborhood = np.concatenate((dummy, up), axis=0)
    elif down_boundary:
        down = grid[p_row-1:p_row+1, p_col-1:p_col+2]
        dummy = np.repeat(env.empty, 3)[np.newaxis , :]
        neighborhood = np.concatenate((down, dummy), axis=0)
    elif left_boundary:
        left = grid[p_row-1:p_row+2, p_col:p_col+2]
        dummy = np.repeat(env.empty, 3)[:, np.newaxis]
        neighborhood = np.concatenate((dummy, left), axis=1)
    elif right_boundary:
        right = grid[p_row-1:p_row+2, p_col-1:p_col+1]
        dummy = np.repeat(env.empty, 3)[:, np.newaxis]
        neighborhood = np.concatenate((right, dummy), axis=1)
    elif up_left_boundary:
        up_left = grid[p_row:p_row+2, p_col:p_col+2]
        dummy_up = np.repeat(env.empty, 2)[np.newaxis, :]
        neighborhood = np.concatenate((dummy_up, up_left), axis=0)
        dummy_left = np.repeat(env.empty, 3)[:, np.newaxis]
        neighborhood = np.concatenate((dummy_left, neighborhood), axis=1)
    elif up_right_boundary:
        up_right = grid[p_row:p_row+2, p_col-1:p_col+1]
        dummy_up = np.repeat(env.empty, 2)[np.newaxis, :]
        neighborhood = np.concatenate((dummy_up, up_right), axis=0)
        dummy_right = np.repeat(env.empty, 3)[:, np.newaxis]
        neighborhood = np.concatenate((neighborhood, dummy_right), axis=1)
    elif down_left_boundary:
        down_left = grid[p_row-1:p_row+1, p_col:p_col+2]
        dummy_down = np.repeat(env.empty, 2)[np.newaxis, :]
        neighborhood = np.concatenate((down_left, dummy_down), axis=0)
        dummy_left = np.repeat(env.empty, 3)[:, np.newaxis]
        neighborhood = np.concatenate((dummy_left, neighborhood), axis=1)
    elif down_right_boundary:
        down_right = grid[p_row-1:p_row+1, p_col-1:p_col+1]
        dummy_down = np.repeat(env.empty, 2)[np.newaxis, :]
        neighborhood = np.concatenate((down_right, dummy_down), axis=0)
        dummy_right = np.repeat(env.empty, 3)[:, np.newaxis]
        neighborhood = np.concatenate((neighborhood, dummy_right), axis=1)
    else:
        neighborhood = grid[p_row-1:p_row+2, p_col-1:p_col+2].copy()
    return neighborhood
            
         
def neighborhood_cells_to_actions(grid, position, env):
    position_to_action = ((1,2,3),
                          (4,5,6),
                          (7,8,9))
    neighborhood = get_neighborhood(grid, position)
    fire_actions=[]
    tree_actions=[]
    empty_actions=[]
    for row_idx, row in enumerate(neighborhood):
        for col_idx, col in enumerate(neighborhood):
            cell = neighborhood[row_idx, col_idx]
            if cell == env.fire:
                fire_actions.append(position_to_action[row_idx][col_idx])
            elif cell == env.tree:
                tree_actions.append(position_to_action[row_idx][col_idx])
            elif cell == env.empty:
                empty_actions.append(position_to_action[row_idx][col_idx])
            else:
                raise Exception('Error: Unrecognizable forest cell')
    return fire_actions, tree_actions, empty_actions

def get_action_heuristic(grid, position, env):
    random_idx = lambda obj: np.random.choice(np.arange(0, len(obj)))
    fire_actions, tree_actions, empty_actions = neighborhood_cells_to_actions(grid, position, env)
    if fire_actions:
        idx = random_idx(fire_actions)
        action = fire_actions[idx]
    elif tree_actions:
        idx = random_idx(tree_actions)
        action = tree_actions[idx]
    elif empty_actions:
        idx = random_idx(empty_actions)
        action = empty_actions[idx]
    else:
        raise Exception('Error: Not an action to take')
    return action

if __name__ == '__main__':
    # Environment
    env = helicopter.Helicopter(n_row = N_ROW, n_col = N_COL,
                            p_fire = P_FIRE, p_tree = P_TREE,
                            tree = TREE, fire = FIRE, empty = EMPTY)
    N_ACTIONS = env.actions_cardinality
    
    # First observation
    observation = env.reset()
    grid, position = observation
    env.render()
    
    total_reward = 0
    for i in range(env.freeze * 60):
      print('.', end='')
      action = get_action_heuristic(grid, position, env)
      observation, reward, done, info = env.step(action)
      grid, position = observation
      total_reward += reward
      env.render()
    
    print('\nTotal Reward: {}'.format(total_reward))
