#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:02:41 2020

@author: ebecerra
"""

import helicopter
import numpy as np
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

# Global Definitions

SEED = 584105
torch.manual_seed(SEED)

# Environment parameters
N_ROW = 8
N_COL = 8
P_FIRE = 0.01
P_TREE = 0.30
# Symbols for cells
TREE = 3
FIRE = 7
EMPTY = 1

# Test environment to get some important attributes 
env = helicopter.Helicopter(n_row = N_ROW, n_col = N_COL,
                            p_fire = P_FIRE, p_tree = P_TREE,
                            tree = TREE, fire = FIRE, empty = EMPTY)
N_ACTIONS = env.actions_cardinality
FREEZE_FRAMES = env.freeze

FOREST_ITERATIONS = 100
STEPS_PER_EPISODE = FREEZE_FRAMES * FOREST_ITERATIONS
EPOCHS = 2000
ITERATIONS = 4
BATCH_SIZE = 100
PERCENTILE = 70
CPUS = 1

# Take advantage of hardware if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self, n_actions, n_row, n_col):
        super(Net, self).__init__()
        
        # Leaves image size
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 6,
                kernel_size = (3,3),
                stride = 1,
                padding = 1),
            nn.ReLU())
        
        # Halves image size
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 6,
                out_channels = 12,
                kernel_size = (3,3),
                stride = 1,
                padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        # Add Helicopter position
        self.fc12 = nn.Linear(12 * (n_row//2) * (n_col//2) + 2, 60)
        self.fc23 = nn.Linear(60, 20)
        # Output layer
        self.fc34 = nn.Linear(20, n_actions)
        
    def forward(self, batch_grid, batch_position):
        # Conv1
        batch_grid = self.conv1(batch_grid)
        # Conv2
        batch_grid = self.conv2(batch_grid)
        
        # Flatten
        # -1, copies batch dimension
        batch_grid = batch_grid.view(-1, self.num_flat_features(batch_grid))
        
        # Needs to add position info
        # before passing to fully connected
        batch_grid = torch.cat((batch_position, batch_grid), dim=1)
        
        # The start of the fully connected layers
        batch_grid = self.fc12(batch_grid)
        batch_grid = self.fc23(batch_grid)
        
        # Output
        batch_grid = self.fc34(batch_grid)
        return batch_grid
    
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions, except batch
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Observations in an iterable form (in batch)
def observations_to_tensors(observations):
    # Adds Channel and Batch dimensions
    grid = [ [obs[0]] for obs in observations ]
    # Adds Batch dimension
    position = [ pos[1] for pos in observations ]
    grid = torch.FloatTensor(grid)
    position = torch.FloatTensor(position)
    return grid, position

def get_action(grid, position, net):
    # Obtain policy
    sm = nn.Softmax(dim=1)
    act_probs_bat = sm(net.forward(grid, position))
    # To numpy
    policy = act_probs_bat.data.numpy()[0]
    action = np.random.choice(np.arange(1,10), p=policy)
    return action

def play_episode(net, steps_in_episode):
    env = helicopter.Helicopter(n_row = N_ROW, n_col = N_COL,
                            p_fire = P_FIRE, p_tree = P_TREE,
                            tree = TREE, fire = FIRE, empty = EMPTY)
    obs = env.reset()
    total_reward = 0.0
    step_data = {}
    episode_steps = []
    episode_data = {}
    
    for step_num in range(steps_in_episode):
        grid, position = observations_to_tensors([obs])            
        action = get_action(grid, position, net)
        
        next_obs, reward, is_done, info = env.step(action)
        
        total_reward += reward
        step_data['observation'] = obs
        step_data['action'] = action
        step_data['reward'] = reward
        step_data['step_number'] = step_num
        episode_steps.append(step_data)
        obs = next_obs
        step_data = {}
    
    episode_data['total_reward'] = total_reward 
    episode_data['info'] = info
    episode_data['all_steps'] = episode_steps       
    return episode_data

# Harvest batch
def get_batch(net, steps_in_episode, batch_size, cpus):
    pool = mp.Pool(cpus)
    # Get batch in parallel
    batch = pool.starmap(play_episode,
                         [ (net, steps_in_episode) for episode_idx in range(batch_size) ]
                         )
    pool.close()
    return batch

def filter_batch(batch, percentile):
    # Extract the total_reward field
    batch_rewards = [ episode['total_reward'] for episode in batch ]
    cutoff = np.percentile(batch_rewards, percentile, interpolation='higher')
    
    elite_batch = []
    for episode, batch_reward in zip(batch, batch_rewards):
        if batch_reward >= cutoff:
            elite_batch.append(episode)
    return elite_batch, cutoff

def get_train_obs_acts(batch):
    train_obs = []
    train_acts = []
    for episode in batch:
        eps_obs = []
        eps_acts = []
        for step in episode['all_steps']:
            eps_obs.append(step['observation'])
            # -1 to get range [0,8]
            eps_acts.append(step['action'] - 1)
        train_obs.extend(eps_obs)
        train_acts.extend(eps_acts)
    return train_obs, train_acts

if __name__ == "__main__":
    # Initializations
    writer = SummaryWriter(comment='cross_helicopter')
    net = Net(N_ACTIONS, N_ROW, N_COL)    
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.001)
    top_episodes = []
    # Main Loop
    for epoch in range(EPOCHS):
        batch = get_batch(net, STEPS_PER_EPISODE, BATCH_SIZE, CPUS)
        batch_reward_mean = np.mean([episode['total_reward'] for episode in batch])
        batch_reward_mean = np.round(batch_reward_mean, 2)
        batch_cutted_down, cutoff = filter_batch(batch, percentile = PERCENTILE)
        
        top_episodes += batch_cutted_down
        obs, acts = get_train_obs_acts(top_episodes)
        # Iterations to help convergence
        for iteration in range(ITERATIONS):
            # Getting the inputs and targets
            grids, positions = observations_to_tensors(obs)
            # Training
            optimizer.zero_grad()
            predictions = net.forward(grids, positions)
            targets = torch.LongTensor(acts)
            loss_v = objective(predictions, targets)
            loss_v.backward()
            optimizer.step()
            
        top_current_batch, __ = filter_batch(batch_cutted_down, percentile = 90)
        top_episodes += top_current_batch
        
        # Flush top episodes if more than n acummulated
        if len(top_episodes) == 128:
           top_episodes, __ = filter_batch(top_episodes, percentile = 90)            
            
        print(f'\n------------ {epoch + 1} ------------')
        print('reward_mean: {} loss: {}, cutoff: {}, top_length: {}'.format(
                                                        batch_reward_mean,
                                                        np.round(loss_v.item(), 4),
                                                        cutoff,
                                                        len(top_episodes)))
        # Writing to tensorboard
        writer.add_scalar("loss", loss_v.item(), epoch)
        writer.add_scalar("reward_mean", batch_reward_mean, epoch)
        writer.add_scalar("reward_bound", cutoff, epoch)
