#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 12:36:42 2020

@author: ebecerra
"""

import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

import helicopter

N_ROW = 8; N_COL=8; P_FIRE=0.01; P_TREE=0.30
ENV = helicopter.Helicopter(n_row = N_ROW, n_col = N_COL,
                            p_fire = P_FIRE, p_tree = P_TREE)
REPLAY_SIZE = 10000
# Start learning at this size of the replay memory
REPLAY_START_SIZE = 1000
BATCH_SIZE = 32

GAMMA = 0.95

LEARNING_RATE = 0.001

# Epsilon decreses linearly by epoch
EPSILON_START = 1.00
EPSILON_END = 0.00

EPOCHS = 100
SYNC_TARGET = 100

Experience = collections.namedtuple('Experience',
                                    field_names=('state', 'action', 'reward', 'done', 'new_state'))

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

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    def __len__(self):
        return len(self.buffer)
    def append(self, experience):
        self.buffer.append(experience)
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return states, actions, rewards, dones, next_states

def observations_to_tensors(observations):
    # Observations in an iterable form (in batch)
    # Adds Channel and Batch dimensions
    grid = [ [obs[0]] for obs in observations ]
    # Adds Batch dimension
    position = [ pos[1] for pos in observations ]
    grid = torch.FloatTensor(grid)
    position = torch.FloatTensor(position)
    return grid, position

class Agent:
    """Implements interaction with the environment &
    Policy"""
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0):
        if np.random.random() < epsilon:
            actions_set = list(self.env.actions_set)
            action = np.random.choice(actions_set)
        else:
            grid, position = observations_to_tensors([self.state])
            q_vals_v = net(grid, position)
            __, act_v = torch.max(q_vals_v, dim=1)
            # Actions are from 1 to 9
            action = int(act_v.item() + 1)

        # do step in the environment
        new_state, reward, is_done, info = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            print('Episode is done')
            self._reset()
        return self.total_reward

def calc_loss(batch, net, tgt_net):
    states, actions, rewards, dones, next_states = batch
    
    grids, positions = observations_to_tensors(states)
    grids_next, positions_next = observations_to_tensors(next_states)
    
    actions_v = torch.tensor(actions) - 1
    rewards_v = torch.tensor(rewards)
    done_mask = torch.BoolTensor(dones)

    # Quality of the taken actions
    state_action_values = net(grids, positions).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    
    # Next state max Q(s,a)
    next_state_values = tgt_net(grids_next, positions_next).max(1)[0]
    # The value of ended episode is 0
    next_state_values[done_mask] = 0.0
    
    # Calculate target
    expected_state_action_values = rewards_v + GAMMA * next_state_values
    return nn.MSELoss()(state_action_values, expected_state_action_values)

# if __name__ == 'main':
# Initializations
net = Net(n_actions=ENV.actions_cardinality, n_row=ENV.n_row, n_col=ENV.n_col)
tgt_net = Net(n_actions=ENV.actions_cardinality, n_row=ENV.n_row, n_col=ENV.n_col)

buffer = ReplayMemory(REPLAY_SIZE)
agent = Agent(ENV, buffer)

optimizer = optim.Adam(net.parameters(), lr = LEARNING_RATE)

# epsilon decreseases linearly by epoch
epsilon_schedule = np.linspace(EPSILON_START, EPSILON_END, EPOCHS)

# Fill the replay memory
for filling_step in range(REPLAY_START_SIZE):
    agent.play_step(net, epsilon = 1.00)
    print('.', end='')
    print('Filled Replay Memory')
    print('Starting to learn now', end='\n\n')

# Start training
for epoch, epsilon in enumerate(epsilon_schedule):
    print(f'epoch: #{epoch}')
    # Syncronize net and target net
    if epoch % SYNC_TARGET == 0:
        tgt_net.load_state_dict(net.state_dict())
    
    # Play a step
    agent.env.render()
    agent.play_step(net, epsilon)
    
    # Network Optimization
    optimizer.zero_grad()
    batch = buffer.sample(BATCH_SIZE)
    loss_t = calc_loss(batch, net, tgt_net)
    loss_t.backward()
    optimizer.step()


    

