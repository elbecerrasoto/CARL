#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:02:41 2020

@author: ebecerra
"""

import helicopter

import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import multiprocessing as mp

# Global Definitions

SEED = 584105
torch.manual_seed(SEED)

writer = SummaryWriter(comment='cross_helicopter')

STEPS_PER_EPISODE = 4 * 100
EPOCHS = 2000
ITERATIONS = 4
BATCH_SIZE = 100
PERCENTILE = 70
CPUS = 2

# For storing step data in all steps per episode
# obs -> action -> reward
EpisodeStep = namedtuple('EpisodeStep', ('observation',
                                         'action',
                                         'reward',
                                         'policy',
                                         'step'))

# For storing global episode data
Episode = namedtuple('Episode', ('total_reward',
                                 'info',
                                 'steps'))

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

def get_action_policy(grid, position, net):
    # Obtain policy
    sm = nn.Softmax(dim=1)
    act_probs_bat = sm(net.forward(grid, position))
    # To numpy
    policy = act_probs_bat.data.numpy()[0]
    action = np.random.choice(np.arange(1,10), p=policy)
    return action, policy

def play_episode(env, net, steps_in_episode):
    obs = env.reset()
    total_reward = 0.0
    episode_steps = []
    
    for step_num in range(steps_in_episode):
        grid, position = observations_to_tensors([obs])            
        action, policy = get_action_policy(grid, position, net)
        
        next_obs, reward, is_done, info = env.step(action)
        
        total_reward += reward
        step_data = EpisodeStep(observation = obs,
                                action      = action,
                                reward      = reward,
                                policy      = policy,
                                step        = step_num)
        episode_steps.append(step_data)
        obs = next_obs
        
    episode_data = Episode(total_reward = total_reward,
                           info         = info,
                           steps        = episode_steps)      
    return episode_data


# Test the the neural network
# net_test = Net(n_actions=9, n_row=env.n_row, n_col=env.n_col)
# obs=env.reset()0].total_reward

# grid, position = observations_to_tensors([obs])
# position
# net_test.forward(grid, position)

# Test play episode
# test_net = Net(9, env.n_row, env.n_col)
# some_episode = play_episode(env, test_net, STEPS_PER_EPISODE)

# some_episode?
# some_episode.total_reward
# some_episode.info
# some_episode.steps[0].action

env = helicopter.Helicopter(n_row = 8, n_col = 8,
                      tree = 3,
                      fire = 7,
                      empty = 1)

# Harvest batch
def get_batch(net, steps_in_episode, batch_size, cpus):
    def initialize_environments(how_many):
        all_envs = []
        for i_env in range(how_many):
            env = helicopter.Helicopter(n_row = 8, n_col = 8,
              tree = 3,
              fire = 7,
              empty = 1)
            all_envs.append(env)
        return all_envs
    envs = initialize_environments(batch_size)
    pool = mp.Pool(cpus)
    # Get batch in parallel
    batch = pool.starmap_async(play_episode,
                         [ (env, net, steps_in_episode) for env in envs ]
                         ).get()
    pool.close()
    pool.join()
    return batch

# Test get batch
net_test = Net(n_actions=9, n_row=env.n_row, n_col=env.n_col)
# # batch = get_batch(net_test, STEPS_PER_EPISODE, BATCH_SIZE, CPUS)

# # Testing get batch in a loop

# for i in range(3):
#     batch = get_batch(env, net_test, , BATCH_SIZE, CPUS)
#     print(batch[0].total_reward)
    
batch1 = get_batch(net_test, 100, 4, 2)
batch2 = get_batch(net_test, 100, 4, 2)
batch3 = get_batch(net_test, 100, 4, 2)

len(batch1)

batch1[0].total_reward
batch2[0].total_reward
batch3[0].total_reward

batch1[0].steps[0]
batch2[0].steps[0]
batch3[0].steps[0]


list(map(lambda r: r.total_reward, batch1))


def get_batch_non_parallel(env, net, steps_in_episode, batch_size):
    batch = []
    for episode in range(batch_size):
        batch.append( play_episode(env, net, steps_in_episode) )
    return batch
    
# m_bat1 = get_batch_non_parallel(env, net_test, 100, 10)
# m_bat2 = get_batch_non_parallel(env, net_test, 100, 10)
# m_bat3 = get_batch_non_parallel(env, net_test, 100, 10)

if False:    
    def filter_batch(batch, percentile):
        # Extract the total_reward field
        batch_rewards = list(map(lambda eps: eps.total_reward, batch))
        cutoff = np.percentile(batch_rewards, percentile, interpolation='higher')
        train_obs = []
        train_acts = []
        elite_batch = []
        
        for episode, batch_reward in zip(batch, batch_rewards):
            if batch_reward >= cutoff:
                # Extract elite observations
                # Coerces map to list, (Type propagation)
                train_obs.extend(map(lambda step: step.observation, episode.steps))
                # Extract elite actions
                train_acts.extend(map(lambda step: step.action - 1, episode.steps))
                elite_batch.append(episode)
        return elite_batch, train_obs, train_acts, cutoff
    
    if False:    
        if __name__ == "__main__":
            # Initializations
            net = Net(n_actions=9, n_row=env.n_row, n_col=env.n_col)    
            objective = nn.CrossEntropyLoss()
            optimizer = optim.Adam(params=net.parameters(), lr=0.001)
            top_episodes = []
            # Main Loop
            for epoch in range(EPOCHS):
                batch = get_batch(env, net, STEPS_PER_EPISODE, BATCH_SIZE, CPUS)
                batch_reward_mean = np.round(np.mean(list(map(lambda eps: eps.total_reward, batch))), 2)
                batch_cutted_down, obs, acts, cutoff = filter_batch(batch, percentile = PERCENTILE)
                # Keep all of them
                __, obs, acts, cutoff2 = filter_batch(batch_cutted_down + top_episodes, percentile = 0)
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
                    
                top_tmp, obs, acts, cutoff = filter_batch(batch_cutted_down, percentile = 90)
                top_episodes += top_tmp
                
                # Flush top episodes if more than 100 acummulated
                if len(top_episodes) == 100:
                   top_episodes, obs, acts, cutoff = filter_batch(top_episodes, percentile = 90)            
                    
                print(f'\n------------ {epoch + 1} ------------')
                print('reward_mean: {} loss: {}, cutoff: {}, batch_length: {}'.format(
                                                                batch_reward_mean,
                                                                np.round(loss_v.item(), 4),
                                                                cutoff,
                                                                len(obs)))
                # Writing to tensorboard
                writer.add_scalar("loss", loss_v.item(), epoch)
                writer.add_scalar("reward_mean", batch_reward_mean, epoch)
                writer.add_scalar("reward_bound", cutoff, epoch)
