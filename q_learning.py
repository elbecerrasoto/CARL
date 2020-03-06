#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:15:46 2020

@author: ebecerra
"""

# Importing the environment
import helicopter

from collections import defaultdict
import numpy as np

# Environment parameters
N_ROW = 8
N_COL = 8
P_FIRE = 0.01
P_TREE = 0.30

# Train Loops 
TRAIN_LOOPS = 100
GAMMA = 0.9
ALPHA = 0.2

# Test Episodes Parameters
TEST_EPISODES = 20
STEPS_PER_EPISODE = 60 * 4

class Agent:
    def __init__(self, env):
        self.env = env
        self.state = self.env.reset()
        self.values = defaultdict(float)
    def sample_env(self):
        action = np.random.choice(list(self.env.actions_set))
        old_state = self.state 
        new_state, reward, is_done, info = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return old_state, action, reward, new_state
    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in self.env.actions_set:
            # Default value of state-action pair is 0
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action
    def value_update(self, s, a, r, next_s):
        # Best q of the next state
        best_v, __ = self.best_value_and_action(next_s)
        new_val = r + GAMMA * best_v
        old_val = self.values[(s,a)]
        self.values[(s,a)] = old_val * (1-ALPHA) + new_val * ALPHA
    def play_episode(self, env):  # Play a sample episode 
        total_reward = 0.0
        state = env.reset()
        for step in range(STEPS_PER_EPISODE):
            __, action = self.best_value_and_action(state)
            new_state, reward, is_done, info = env.step(action)
            total_reward += reward
            env.render()
            state = new_state
        return total_reward
          
if __name__ == "__main__":
    ENV = helicopter.Helicopter(n_row = N_ROW, n_col = N_COL,
                                p_fire = P_FIRE, p_tree = P_TREE)
    agent = Agent(ENV)
    best_reward = 0.0
    for iter_no in range(TRAIN_LOOPS):
        iter_no += 1
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s)
        print(f'.{iter_no}', end='')

list_str = ['Hola', 'Mundo']
zeros_locos = (( (1,2,(3,4)), 10 ), np.zeros((2,2,2,2)), 'Hola', {42,43,{44,45}} )

def to_list_recursively(nested_iterable):
    new_list = []
    for elem in nested_iterable:
        if not hasattr(elem, '__iter__'):
            new_list.append(elem)
        elif isinstance(elem, str) and len(elem) == 1:
            new_list.append(elem)
        else:
            new_list.append(to_list_recursively(elem))
    return new_list

{ 42,43, 44,45 }

to_list_recursively(zeros_locos)

dict_loco = {1: 'a', 2:{3:'c', [{1''}, ]}}

to_dict_recursively()

hasattr('Hola', '__iter__')
hasattr(1, '__iter__')
list(1)

test_dict = {}

{(1,2): 'abc'}








from hashlib import sha1

from numpy import all, array, uint8


class hashable(object):
    r'''Hashable wrapper for ndarray objects.

        Instances of ndarray are not hashable, meaning they cannot be added to
        sets, nor used as keys in dictionaries. This is by design - ndarray
        objects are mutable, and therefore cannot reliably implement the
        __hash__() method.

        The hashable class allows a way around this limitation. It implements
        the required methods for hashable objects in terms of an encapsulated
        ndarray object. This can be either a copied instance (which is safer)
        or the original object (which requires the user to be careful enough
        not to modify it).
    '''
    def __init__(self, wrapped, tight=False):
        r'''Creates a new hashable object encapsulating an ndarray.

            wrapped
                The wrapped ndarray.

            tight
                Optional. If True, a copy of the input ndaray is created.
                Defaults to False.
        '''
        self.__tight = tight
        self.__wrapped = array(wrapped) if tight else wrapped
        self.__hash = int(sha1(wrapped.view(uint8)).hexdigest(), 16)

    def __eq__(self, other):
        return all(self.__wrapped == other.__wrapped)

    def __hash__(self):
        return self.__hash

    def unwrap(self):
        r'''Returns the encapsulated ndarray.

            If the wrapper is "tight", a copy of the encapsulated ndarray is
            returned. Otherwise, the encapsulated ndarray itself is returned.
        '''
        if self.__tight:
            return array(self.__wrapped)

        return self.__wrapped
Using the wrapper class is simple enough:
from numpy import arange
a = arange(0, 1024)
d = {}
d[a] = 'foo'
# TypeError: unhashable type: 'numpy.ndarray'
b = hashable(a)
d[b] = 'bar'
d[b]
# 'bar'



