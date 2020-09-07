#!/usr/bin/python3

from helicopter import EnvMakerForestFire
import pickle

ENV_FILE =  'dqn_env.pickle'

env = EnvMakerForestFire(observation_mode='one_hot3',
                         n_row=3, n_col=4,
                         p_fire=0.066, p_tree=0.333,
                         moves_before_updating=2,
                         reward_fire=-1.00, reward_tree=0.30, reward_hit=0.60, reward_empty=-0.10, reward_type='both')

with open(ENV_FILE, 'wb') as env_data:
	pickle.dump(env, env_data)

print('Done.')
