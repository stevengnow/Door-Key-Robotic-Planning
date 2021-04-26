import numpy as np
import gym
from utils import *
from example import example_use_of_gym_env
import itertools
import copy
import time

MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

def doorkey_problem_random(env, info):
    # Create State Space
    T = 100
    action_list = [MF, TL, TR, PK, UD]
    coordinates = list(itertools.product(np.arange(info['height']), repeat = 2))
    goal_loc = (info['goal_pos'][0],info['goal_pos'][1])
    door_loc = (info['door_pos'][0],info['door_pos'][1])
    door1_loc = tuple(door_loc[0])
    door2_loc = tuple(door_loc[1])
    door1 = env.grid.get(info['door_pos'][0][0], info['door_pos'][0][1])
    door2 = env.grid.get(info['door_pos'][1][0], info['door_pos'][1][1])
    
    key_loc = (info['key_pos'][0],info['key_pos'][1])
    start_loc = (info['init_agent_pos'][0],info['init_agent_pos'][1])
    start_dir = [info['init_agent_dir'][0],info['init_agent_dir'][1]]

    start_dict = {'position': start_loc, 'direction': start_dir, 'carry_key': 0, 'door1_open': door1.is_open, 'door2_open':door2.is_open, 'door_loc1': door1_loc, 'door_loc2': door2_loc, 'goal_loc': goal_loc, 'key_loc': key_loc}
    total_space = {'position' : coordinates, 'direction' : [[-1,0],[1,0],[0,-1],[0,1]], 'carry_key' : [0,1], 'door1_open' : [0,1], 'door2_open': [0,1], 'door_loc1' : [door1_loc], 'door_loc2': [door2_loc], 'goal_loc' : [goal_loc], 'key_loc' : [key_loc]}
    keys = total_space.keys()
    values = (total_space[key] for key in keys)
    state_space = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    for state in range(len(state_space)):
        if isinstance(env.grid.get(state_space[state]['position'][0],state_space[state]['position'][1]),gym_minigrid.minigrid.Wall):
            state_space[state]=0
    state_space = [i for i in state_space if i != 0]  

    # Set Initial V, Q, and Pi Matrix
    V = np.ones((T,len(state_space)))*np.inf
    Q = np.ones((T,len(state_space), 5))*np.inf
    idx_goal = [item for item in range(len(state_space)) if state_space[item]["position"] == goal_loc]
    V[0, idx_goal] = 0 
    pi = np.zeros((T,len(state_space)))
    start_idx = state_space.index(start_dict)

    # Dynamic Programming
    for t in range(T-1):
        for s, state in enumerate(state_space):
            for action in action_list:
                new_state = next_state_random(state,action)
                new_state_idx = [item for item in range(len(state_space)) if state_space[item] == new_state]
                if new_state_idx != []:
                    Q[t,s,action] = step_cost(action) + V[t,new_state_idx]    
            if V[t,s] == np.inf:
                V[t+1,s] = np.min(Q[t][s])
            else:
                V[t+1,s] = V[t,s]
            pi[t+1,s] = np.argmin(Q[t][s])
        if np.array_equal(V[t+1],V[t]):  # terminate early
            break

    # Find terminating time step
    for i in range(V.shape[0]-1):
        if np.all(V[i]==V[i+1]):
            same_row = i
            break
    
    # Find action sequence    
    act_seq = []
    s_STATE = copy.deepcopy(start_dict)
    for t in range(same_row,0,-1):       
        a_action = pi[t,state_space.index(s_STATE)]
        act_seq.append(a_action)
        s_STATE = next_state_random(s_STATE,a_action)

    # Find Optimal action sequence
    optim_act_seq = []
    s_STATE = copy.deepcopy(start_dict)
    for true_action in act_seq:
        optim_act_seq.append(true_action)
        s_STATE = next_state_random(s_STATE,true_action)
        if s_STATE['position'] == goal_loc:
            optim_act_seq.append(0.0) # to end up on goal cell
            break
    return optim_act_seq

def next_state_random(state, action):
    '''
    '''
    new_state = copy.deepcopy(state)
    cell_front = tuple(np.array(new_state['position']) + np.array(new_state['direction']))
    
    if action == 0: 
        # check to see if we are in front of a door
        if (cell_front == tuple(new_state['door_loc1']) and new_state['door1_open'] == 0) or (cell_front == tuple(new_state['door_loc2']) and new_state['door2_open'] == 0):
            new_state['position'] = [-10,-10]  # trash state
        else:
            new_state['position'] = cell_front  
    elif action == 1:
        if new_state['direction'] == [0,1]:
            new_state['direction'] = [1,0]
        elif new_state['direction'] == [1,0]:
            new_state['direction'] = [0,-1]
        elif new_state['direction'] == [0,-1]:
            new_state['direction'] = [-1,0]
        else:
            new_state['direction'] = [0,1]
    elif action == 2:
        if new_state['direction'] == [0,1]:
            new_state['direction'] = [-1,0]
        elif new_state['direction'] == [-1,0]:
            new_state['direction'] = [0,-1]
        elif new_state['direction'] == [0,-1]:
            new_state['direction'] = [1,0]
        else:
            new_state['direction'] = [0,1]
    elif action == 3:
        # check to see if there is a key in front of us
        if cell_front == new_state['key_loc']:
            new_state['carry_key'] = 1
        else:
            new_state['carry_key'] = -10  # trash state
    else:
        # check to see if we can unlock one of the doors (whichever is in front if any)
        if (cell_front == tuple(new_state['door_loc1']) and new_state['carry_key'] == 1):
            new_state['door1_open'] = 1
        elif (cell_front == tuple(new_state['door_loc2']) and new_state['carry_key'] == 1):
            new_state['door2_open'] = 1
        else:
            new_state['door1_open'] = -10  # trash state
            new_state['door2_open'] = -10  # trash state
    return new_state



