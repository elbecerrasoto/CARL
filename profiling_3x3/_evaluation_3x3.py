#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:58:13 2020

@author: ebecerra
"""

import os
import re

from lib.helicopter import EnvMakerForestFire
from lib.helpers import Agent, ReplayMemoryNSteps

import torch

WEIGHTS = 'weights'
STEPS_PER_RUN = 20

files = os.listdir(WEIGHTS)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

weights = []
models = []
for file in files:
    if re.match(r'[a-z]\.pytorch', file):
        weights.append(WEIGHTS + '/' + file)

    match_model = re.match(r'([a-z]_model)\.py', file)
    if match_model:
        models.append(match_model.group(1))

weights = sorted(weights)
models = sorted(models)

env = EnvMakerForestFire(observation_mode='one_hot3',
                         n_row=3, n_col=3,
                         p_tree=0.333, p_fire=0.066,
                         moves_before_updating=1,
                         reward_tree=1, reward_fire=-1)

def set_global_shape(env):
    grid, pos, moves = env.reset()
    return grid.shape

shape = set_global_shape(env)
n_actions = len(env.movement_actions)

for run in range(len(weights)):
    module = __import__(WEIGHTS + '.' + models[run])
    run_model = getattr(module, models[run])
    net = run_model.DQN(shape, n_actions)
    net.load_state_dict(torch.load(weights[run], map_location=device))
    
    env = EnvMakerForestFire(observation_mode='one_hot3',
                             n_row=3, n_col=3,
                             p_tree=0.333, p_fire=0.066,
                             moves_before_updating=1,
                             reward_tree=1, reward_fire=-1)
    agent = Agent(env, ReplayMemoryNSteps(42))
    for step in range(STEPS_PER_RUN):
        agent.play_step(net, epsilon=0.0)
    print(f'Experiment: {run}, Return: {env.total_reward} Mean: {env.total_reward/STEPS_PER_RUN}')
    
for policy in ('random', 'heuristic'):
    env = EnvMakerForestFire(observation_mode='one_hot3',
                             n_row=3, n_col=3,
                             p_tree=0.333, p_fire=0.066,
                             moves_before_updating=1,
                             reward_tree=1, reward_fire=-1)
    if policy == 'random':
        for step in range(STEPS_PER_RUN):
            env.reset()
            action = env.random_policy()
            env.step(action)
        print(f'Experiment: {policy}, Return: {env.total_reward} Mean: {env.total_reward/STEPS_PER_RUN}')
    if policy == 'heuristic':
        for step in range(STEPS_PER_RUN):
            env.reset()
            action = env.heuristic_policy()
            env.step(action)
        print(f'Experiment: {policy}, Return: {env.total_reward} Mean: {env.total_reward/STEPS_PER_RUN}')
