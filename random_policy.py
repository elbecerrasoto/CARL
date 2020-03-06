#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 18:17:54 2020

@author: ebecerra
"""

import helicopter
import numpy as np

env = helicopter.Helicopter(n_row = 8, n_col = 8,
  tree = 3,
  fire = 7,
  empty = 1, p_fire = 0.01, p_tree = 0.3)

# First observation
observation = env.reset()
env.render()

total_reward = 0
for i in range(env.freeze * 100):
  print('.', end='')
  action = np.random.choice(list(env.actions_set))
  observation, reward, done, info = env.step(action)
  total_reward += reward
  env.render()

print('\nTotal Reward: {}'.format(total_reward))

np.random.choice(env.actions_set)
env.step(1)
env.render()

np.random.choice(np.array(env.actions_set))
np.array(list(env.actions_set)).shape

np.random.choice(list(env.actions_set))


# Playing with cross entropy net

# from cross_entropy import Net, observations_to_tensors, get_action
# import pickle

# file = open('cross_entropy_net_v0', 'rb')
# cross_net = pickle.load(file)
# file.close()

# total_reward = 0
# for i in range(env.freeze * 60):
#   print('.', end='')
#   grid, position = observations_to_tensors([observation])
#   action = get_action(grid, position, cross_net)
#   observation, reward, done, info = env.step(action)
#   total_reward += reward
#   env.render()

# print('\nTotal Reward: {}'.format(total_reward))
