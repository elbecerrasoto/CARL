#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 12:36:42 2020

@author: ebecerra
"""

import numpy as np
np.set_printoptions(precision=4)
import collections
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/DQN_helicopter')

import helicopter

N_ROW = 1; N_COL=8; P_FIRE=0.01; P_TREE=0.30
ENV = helicopter.Helicopter(n_row = N_ROW, n_col = N_COL,
                            p_fire = P_FIRE, p_tree = P_TREE)
REPLAY_SIZE = 1000
# Start learning at this size of the replay memory
REPLAY_START_SIZE = 200
BATCH_SIZE = 8
GAMMA = 0.75
LEARNING_RATE = 0.0001
EPSILON = 0.10

EPOCHS = 5001
SYNC_TARGET = 100

Experience = collections.namedtuple('Experience',
                                    field_names=('state', 'action', 'reward', 'done', 'new_state'))

HIDDEN_12 = 64
HIDDEN_23 = 32

class Net(nn.Module):
    def __init__(self, n_actions, n_row, n_col, hidden_12, hidden_23):
        super(Net, self).__init__()
        self.hidden_12 = hidden_12
        self.hidden_23 = hidden_23
        # Flatten Lattice and Add Helicopter position
        self.fc12 = nn.Linear(n_row * n_col + 2, self.hidden_12)
        self.fc23 = nn.Linear(self.hidden_12, self.hidden_23)
        # Output layer
        self.fc34 = nn.Linear(self.hidden_23, n_actions)
    def forward(self, batch_grid, batch_position):
        # Flatten
        # -1, copies batch dimension
        batch_grid = batch_grid.view(-1, self.num_flat_features(batch_grid))
        # Adding position info
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
        return reward

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


def get_epsilon_schedule(epochs, epsilon=0.05, exploring_time=0.15, epsilon_time=0.84):
    assert exploring_time + epsilon_time <= 1.0
    exploiting_time = 1.0 - (exploring_time + epsilon_time)
    exploring_steps = np.floor(epochs*exploring_time)
    epsilon_steps = np.floor(epochs*epsilon_time)
    exploiting_steps = np.floor(epochs*exploiting_time)
    exploring_v = np.repeat(1.0, exploring_steps)
    epsilon_v = np.repeat(epsilon, epsilon_steps)
    exploiting_v = np.repeat(0.0, exploiting_steps)
    schedule = np.concatenate((exploring_v, epsilon_v, exploiting_v))
    tail_len = epochs - len(schedule)
    tail = np.repeat(0.0, tail_len)
    schedule = np.concatenate((schedule, tail))
    return schedule

if __name__ == '__main__':
    # Initializations
    net = Net(ENV.actions_cardinality, ENV.n_row, ENV.n_col, HIDDEN_12, HIDDEN_23)
    tgt_net = Net(ENV.actions_cardinality, ENV.n_row, ENV.n_col, HIDDEN_12, HIDDEN_23)
    
    buffer = ReplayMemory(REPLAY_SIZE)
    agent = Agent(ENV, buffer)
    
    optimizer = optim.Adam(net.parameters(), lr = LEARNING_RATE)
    
    # epsilon decreseases linearly by epoch
    epsilon_schedule = get_epsilon_schedule(EPOCHS, EPSILON)
    
    # Fill the replay memory
    for filling_step in range(REPLAY_START_SIZE):
        agent.play_step(net, epsilon = 1.00)
        # print('.', end='')    
    print(f'\nFilled Replay Memory')
    
    rewards_x_steps = collections.deque()
    # sample_renders = collections.deque()
    
    print('STARTING TO LEARN NOW', end='\n\n')
    # Start training
    for epoch, epsilon in enumerate(epsilon_schedule):
        writer.add_scalar('epsilon', epsilon, epoch)
        # print(f'.{epoch}',end='')
        # Syncronize net and target net
        if epoch % SYNC_TARGET == 0 and epoch != 0:
            # sample_renders.append(agent.env.render())
            # print(f'\n\n')
            print(f'#### EPOCH: {epoch} ####')
            tgt_net.load_state_dict(net.state_dict())
            rewards_x_steps_v = np.array(rewards_x_steps, dtype='float32')
            total_reward = rewards_x_steps_v.sum()
            mean_reward = rewards_x_steps_v.mean()
            writer.add_scalar(f'mean_reward_per_step', mean_reward, epoch)
            writer.add_scalar(f'return_{SYNC_TARGET}_steps', total_reward, epoch)
            print(f'TOTAL REWARD per {SYNC_TARGET} steps: {total_reward}\n')
            rewards_x_steps = collections.deque()
            # torch.save(net.state_dict(), 'DQN_last.dat')
            # with open('renders.dat', 'wb') as file:
            #     pickle.dump(sample_renders, file)
        # Play a step
        # agent.env.render()
        reward = agent.play_step(net, epsilon)
        rewards_x_steps.append(reward)
        writer.add_scalar('rewards', reward, epoch)
        
        # Network Optimization
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net)
        writer.add_scalar('loss', loss_t, epoch)
        loss_t.backward()
        optimizer.step()
        
    states, __, __, __, __ = agent.exp_buffer.sample(BATCH_SIZE)
    grids, positions = observations_to_tensors(states)
    writer.add_graph(net, (grids, positions))
    writer.close()
    with open('net.dat', 'wb') as file:
        pickle.dump(net, file)
    torch.save(net.state_dict(), 'DQN_params.dat')
