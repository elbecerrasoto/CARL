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

# Initial lattice
env.render()

total_reward = 0
for i in range(env.freeze * 100 + 1):
  print('.', end='')
  action = np.random.choice(np.arange(1,10))
  observation, reward, done, info = env.step(action)
  total_reward += reward
  env.render()

print('\nTotal Reward: {}'.format(total_reward))