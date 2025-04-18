#!/usr/bin/env python3
"""
teach.py

UNSW COMP3411/9814 Artificial Intelligence

You may modify this file however you wish, including creating additional
variables, functions, classes, etc.
"""
import random
import pygame
import matplotlib
import numpy as np
from env import StaticGridEnv
from utils import plot_rewards, plot_heatmap
import os
import time

env = StaticGridEnv(seed = 42)
############################################################################
######                Train agent using Q-learning                    ######
############################################################################

state_space_size = 10 * 10       # 100 states
action_space_size = 4            # 4 actions (up, down, left, right)

q_table = np.zeros((state_space_size, action_space_size))
q_table2 = np.zeros((10,10))

def state_to_index(state):
    x, y = state
    return x * 10 + y

def neighbours(state):
    y, x = state
    arr = []
    if y <= 10 and x < 10: arr.append((y-1, x))
    if y <= 8 and x < 10: arr.append((y+1, x))
    if y < 10 and x <= 10: arr.append((y, x-1))
    if y < 10 and x <= 8: arr.append((y, x+1))
    print(f"arr in neighbours {arr}")
    return arr

def neighbours_max(arr):
    i = 0
    while i < len(arr):
        arr[i] = q_table2[arr[i]]
        print(f"arr[i]:", arr[i])
        i += 1
    action = np.argmax(arr)
    print(f"action: {action}")
    return action

def match_action(action, state):
    y, x = state
    next_state = (y, x)
    if action == 0:
        if y <= 10 and x < 10: 
            next_state = (y-1, x)
    elif action == 1:
        if y <= 8 and x < 10: 
            next_state = (y+1, x)
    elif action == 2:
        if y < 10 and x <= 10: 
            next_state = (y, x-1)
    elif action == 3:
        if y < 10 and x <= 8: 
            next_state = (y, x+1)
        
    print(f"next: {next_state}")
    return next_state


# NEXT FIX: when we reach the botom (end of map), the agent just cant move futher
# currently it updates the field its on to -1: line 71 but it should instead move right
# in output arrneighbours are printed, and when there are 3 of them then argmax returns the
# index of max. but because in neighbours we do not clear up which neighbour is missing
# which results in going wrong direction. so next step is to change that to avoid the agent
# being stuck in place 
def train_agent(max_total_steps=50000):
    reward_per_episode = []
    success_rate = 0
    avg_steps_per_episode = 0

    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 100:
        # Take an action
        #state_idx = state_to_index(state)
        #action = np.argmax(q_table[state_idx][0])

        action = neighbours_max(neighbours(state))
        # Execute the action
        next_state, reward, done, _ = env.step(action)

        print(f"next_state: {next_state}")
        print(f"reward: {reward}")
        print(f"done: {done}")
        #print(f"value at: {state_idx} and {action}: {q_table[state_idx][action]}")
        # Update the reward
        if reward < 0:
            q_table2[match_action(action, state)] += reward

        print(f"whole qtable: {q_table2}")
        state = next_state
        total_reward += reward
        steps += 1

    # Render the environment
    env.render (delay = 2 , episode =1 , learning_type = "Q - learning ", availability =0.8 , accuracy = 0.9)
   
    #env.close()
    # Fill in tracking values
    reward_per_episode.append(total_reward)
    success_rate = 1 if reward > 0 else 0
    avg_steps_per_episode = steps

    return q_table, reward_per_episode, success_rate, avg_steps_per_episode


############################################################################
######               Evaluate agent after training                    ######
############################################################################
def evaluate_agent(
        q_table,
        max_total_steps
):
    # YOU MAY ADD ADDITIONAL PARAMETERS IF YOU WISH
    
    # MODIFY CODE HERE
    
    avg_reward_per_episode = 0
    success_rate = 0
    avg_steps_per_episode = 0

    # return avg_reward_per_episode, success_rate, avg_steps_per_episode


############################################################################
######        Train agent using Q-learning with Teacher Advice        ######
############################################################################
def train_agent_with_teacher(
        teacher_q_table,
        max_total_steps,
        availability,
        accuracy
):
    # YOU MAY ADD ADDITIONAL PARAMETERS IF YOU WISH
    
    # MODIFY CODE HERE

    agent_q_table = None
    avg_reward_per_episode = 0
    success_rate = 0
    avg_steps_per_episode = 0

    # return agent_q_table, avg_reward_per_episode, success_rate, avg_steps_per_episode


############################################################################
######                        Main Function                           ######
############################################################################
def main():

    TEACHER_AVAILABILITY = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    TEACHER_ACCURACY = [0.2, 0.4, 0.6, 0.8, 1.0]

    q_table, reward_per_episode, success_rate, avg_steps_per_episode = \
        train_agent(50000)

    plot_rewards(reward_per_episode, "Reward per Episode for Q-learning")

   # avg_reward_per_episode, success_rate, avg_steps_per_episode = \
    #    evaluate_agent(q_table,10000)

    # print("Average Reward per Episode:", avg_reward_per_episode)
    print("Success Rate:", success_rate)
    print("Steps per Episode:", avg_steps_per_episode)
    
    avg_rewards_train = np.zeros((6,5))

    #agent_q_table, reward_per_episode, success_rate, avg_steps_per_episode = \
     #   train_agent_with_teacher(q_table,10000,1,1)

    plot_heatmap(
        avg_rewards_train,
        TEACHER_ACCURACY,      # X-axis: Teacher accuracy
        TEACHER_AVAILABILITY,  # Y-axis: Teacher availability
        "Average Reward for Different Teacher Availability and Accuracy\n (Q-learning Teacher)",  # Title of the heatmap
        "Accuracy",            # X-axis label
        "Availability",        # Y-axis label
    )

if __name__ == '__main__':
    main()
