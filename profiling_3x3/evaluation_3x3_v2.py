#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:58:13 2020

@author: ebecerra
"""

import os
import string
import re

from lib.helicopter import EnvMakerForestFire
from lib.helpers import Agent, ReplayMemoryNSteps

import numpy as np
import torch

import argparse

parser = argparse.ArgumentParser(description="It produces an evaluation of 3x3 grids.",
                                 epilog="(2020) Any bug report it to: elbecerrasoto@gmail.com")
parser.add_argument("-s", "--steps", default=600, type=int,
                    help=f"Frames to capture.")
parser.add_argument("-o", "--out_file", default='evaluation_3x3.csv', type=str,
                    help=f"Output file.")

args = parser.parse_args()
WEIGHTS = 'weights'
STEPS_PER_RUN = args.steps
OUT = args.out_file

if not os.path.isfile(OUT):
    header = 'run,return,mean,sd\n'
    with open(OUT, 'w') as out_csv:
        out_csv.write(header)

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
      
def evaluate_experiments(epsilon=0.0):
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
        rewards = []
        for step in range(STEPS_PER_RUN):
            rewards.append(agent.play_step(net, epsilon))
            
        if epsilon == 0.0:
            experiment = string.ascii_lowercase[run]
        else:
            experiment = string.ascii_lowercase[run] + '_e' + f'{int(epsilon*100)}'
        Return = agent.env.total_reward
        mean = np.round(np.mean(rewards),4)
        sd = np.round(np.std(rewards),4)
        log = f'{experiment},{Return},{mean},{sd}\n'
        with open(OUT, 'a') as out_csv:
            out_csv.write(log)
if False:
    evaluate_experiments()
    evaluate_experiments(epsilon=0.02)
    evaluate_experiments(epsilon=0.05)
    evaluate_experiments(epsilon=0.10)
        
for policy in ('random', 'heuristic'):
    env = EnvMakerForestFire(observation_mode='one_hot3',
                             n_row=3, n_col=3,
                             p_tree=0.333, p_fire=0.066,
                             moves_before_updating=1,
                             reward_tree=1, reward_fire=-1)
    if policy == 'random':
        rewards = []
        env.reset()
        for step in range(STEPS_PER_RUN):
            action = env.random_policy()
            obs, reward, done, info = env.step(action)
            rewards.append(reward)

        experiment = policy
        Return = env.total_reward
        mean = np.round(np.mean(rewards),4)
        sd = np.round(np.std(rewards),4)
        log = f'{experiment},{Return},{mean},{sd}\n'
        with open(OUT, 'a') as out_csv:
            out_csv.write(log)

    if policy == 'heuristic':
        rewards = []
        env.reset()
        for step in range(STEPS_PER_RUN):            
            action = env.heuristic_policy()
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
        
        experiment = policy
        Return = env.total_reward
        mean = np.round(np.mean(rewards),4)
        sd = np.round(np.std(rewards),4)
        log = f'{experiment},{Return},{mean},{sd}\n'
        with open(OUT, 'a') as out_csv:
            out_csv.write(log)
