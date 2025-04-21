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

env = StaticGridEnv(seed = 54)
############################################################################
######                Train agent using Q-learning                    ######
############################################################################

# INITIALIZE: Field initialization of a q_table holding necessary info about the learning
q_table = np.full((100,4), 0, dtype=float)


# state_to_index
# Translates the elements from 2D Matrix to 1D
def state_to_index(state):
    y, x = state
    return y * 10 + x

def index_to_state(index):
    y = index // 10
    x = index % 10
    return (y, x)

# trackbackReward
# If goal is reached, the reward is spread all across the states in q_table that the agent
# followed en route. Prioritizes higher reward for elements closer to the goal.
def trackbackReward(arr):
    arr.reverse()
    arr = list(dict.fromkeys(arr))
    lr = 0.01
    reward = 20
    decay = 0.95
    for i, j in arr:
        q_table[i, j] += lr * (reward - q_table[i, j])
        reward *= decay

# updateWithRewards
# Update the q_table at every regular step (not 'winning' step)
# Uses changable learning rate, and discount factor
def updateWithRewards(reward, action, state, next_state):
    dis_fact = 0.95
    lr = 0.01
    q_table[state, action] += lr * (reward + dis_fact * np.max(q_table[next_state]) 
                                                - q_table[state, action])
        
# episodeValueUpdate
# Update the elements that are refreshed at the start of every episode 
def episodeValueUpdate(steps, state, total_episode_num, total_steps):
    total_episode_num += 1
    episode_reward = 0
    trackbackArr = []
    total_steps += steps
    steps = 0
    return total_steps, state, total_episode_num, episode_reward, trackbackArr, steps


# stepValueUpdate
# Update the elements that are refreshed at every step of an episode
def stepValueUpdate(episode_reward, reward, next_state, state, steps):
    state = next_state  
    steps += 1
    episode_reward += reward
    return episode_reward, state, steps


# train_agent
# Agent training algorithm that uses q-learning algorithm
def train_agent(max_total_steps=10000):

    # INITIALIZE: Return variables and helpful operational elements
    reward_per_episode = []
    exploration_prob = 0.5
    total_episode_num = 0
    total_steps = 0
    steps = 0
    success_num = 0
    decay = 0.95

    # TOTAL EPISODES LOOP: Iterates over all episodes, as long as max step # isnt reached
    while total_steps < max_total_steps:
        state = env.reset()
        state = state_to_index(state)
        total_steps, state, total_episode_num, episode_reward, trackbackArr, steps = episodeValueUpdate(steps, state, total_episode_num, total_steps)
       
        # EPISODE LOOP: every episode has max 100 steps to suceed
        while steps < 100:
            exploration_prob = exploration_prob * decay

            # EXPLORATION vs EXPLOTIATION: with every iteration gain confidence in q_table
            if np.random.rand() < exploration_prob:
                action = np.random.randint(0, 3)  
            else:
                action = np.argmax(q_table[state])  

            # ACTION: Perform action, update the statistics and update q_table with action rewards
            next_state, reward, done, _ = env.step(action)
            next_state = state_to_index(next_state)
            trackbackArr.append((state, action))
            updateWithRewards(reward, action, state, next_state)

            # IF SUCCESS: Reward leading steps, update stats and break from loop
            if done: 
                trackbackReward(trackbackArr) 
                success_num += 1 
                break

            # STEP UPDATE: Update the episode with the statistics
            episode_reward, state, steps = stepValueUpdate(episode_reward, reward, next_state, state, steps)

        reward_per_episode.append(episode_reward)
        
    # TOTAL STATS: calculate total-model statistics
    avg_steps_per_episode = total_steps / total_episode_num
    success_rate = (success_num / total_episode_num) * 100
    # RENDER: rendering last episode
    #env.render (delay = 2 , episode = total_episode_num , learning_type = "Q - learning", 
    #           availability = 0.8 , accuracy = 0.9)

    return q_table, reward_per_episode, success_rate, avg_steps_per_episode


############################################################################
######               Evaluate agent after training                    ######
############################################################################

# evaluate_agent
# Perform agent evaluation based on previously calculated q_table
def evaluate_agent(q_table, max_total_steps):

    # INITIALIZE: The variables needed to evaluate the returning variables
    nr_of_steps = 0
    nr_success = 0
    nr_episodes = 0
    total_reward = 0
    total_steps = 0

    # TOTAL EPISODES LOOP: As long as max step celling is not reached
    while max_total_steps > total_steps:

        # STEP UPDATE: Update the episode with the statistics
        total_steps += nr_of_steps
        current_state = env.reset()
        current_state = state_to_index(current_state)
        nr_episodes += 1
        nr_of_steps = 0

        # EPISODE LOOP: Perform the episode within 100 steps
        while nr_of_steps < 100:

            # ACTION: Perform based on q-table, and update statistics
            action = np.argmax(q_table[current_state]) 
            next_state, reward, done, _ = env.step(action)
            next_state = state_to_index(next_state)
            total_reward += reward
            current_state = next_state
            nr_of_steps += 1

            # IF SUCCESS: Mark success and break from loop
            if done:  
                nr_success += 1 
                break 

    # TOTAL STATS: Calculate total statistics using data from loops
    avg_reward_per_episode = total_reward /  nr_episodes
    success_rate = (nr_success / nr_episodes) * 100
    avg_steps_per_episode = total_steps / nr_episodes

    return avg_reward_per_episode, success_rate, avg_steps_per_episode


############################################################################
######        Train agent using Q-learning with Teacher Advice        ######
############################################################################

# EGreedy algorithm
# Uses epsilon learning to create agent's own learning algorithm that it works on 
# alongside learning from the teacher
class EGreedy():
    agent_q_table = np.full((100,4), 0, dtype=float)

    def _init_ (self):
        pass

    # tracebackReward
    # Creates EGreedy's own function to reward tiles on path that led to positive reward
    def trackbackReward(arr):
        arr.reverse()
        arr = list(dict.fromkeys(arr))
        lr = 0.2
        reward = 20
        decay = 0.95
        for i, j in arr:
            EGreedy.agent_q_table[(i, j)] += lr * (reward - EGreedy.agent_q_table[(i, j)])
            reward *= decay

# update_epsilon_greedy
# On every iteration update the agent q-table with new learnings to improve the model
def update_epsilon_greedy (action, current_state, reward):
    lr = 0.2
    old = EGreedy.agent_q_table[(current_state, action)]
    EGreedy.agent_q_table[(current_state, action)] = old + lr * (reward - old)

# teach_epsilon_greedy
# When agent is learning from the teacher, empower the learning 
def teach_epsilon_greedy (table, action, current_state, accuracy):
    lr = 0.8
    delta = table[current_state, action] - EGreedy.agent_q_table[current_state, action]
    EGreedy.agent_q_table[current_state, action] += lr * delta


# action_decision
# Perform decision of what action should be taken based on teachers accuracy and availbility
# If teaching does not happen - agent works on its own epsilon algorithm
def action_decision (teacher_q_table, availability, accuracy, current_state, epsilon):
    rand1 = np.random.rand()
    rand2 = np.random.rand()
    
    if rand1 < availability:
        if rand2 < accuracy:
            action = np.argmax(teacher_q_table[current_state])
            teach_epsilon_greedy(teacher_q_table, action, current_state, accuracy)
        else:
            worst_action = np.argmax(teacher_q_table[current_state])
            choices = [0, 1, 2, 3]
            choices.remove(worst_action)
            action = np.random.choice(choices)
            #teach_epsilon_greedy(teacher_q_table, action, current_state, accuracy)
    else:
        if np.random.rand() < epsilon:
            action = np.random.randint(0,4)
        else:
            action = np.argmax(EGreedy.agent_q_table[current_state])

    return action, epsilon

# train_agent_with_teacher
# Given q-table of the teacher, if possible, agent learns from the teacher
def train_agent_with_teacher(teacher_q_table, max_total_steps, availability, accuracy):
    # INITIALIZE: The variables needed to evaluate the returning variables
    nr_of_steps, nr_success = 0, 0
    nr_success = 0
    nr_episodes = 0
    total_reward = 0
    total_steps = 0
    decay_rate = 0.95
    epsilon = 0.2

    # TOTAL EPISODES LOOP: As long as max step celling is not reached
    while max_total_steps > total_steps:

        # STEP UPDATE: Update the episode with the statistics
        total_steps += nr_of_steps
        current_state = env.reset()
        current_state = state_to_index(current_state)
        nr_episodes += 1
        nr_of_steps = 0
        trackbackArr = []
        epsilon = max(0.01, epsilon * decay_rate)

        while nr_of_steps < 200 :
            # CALCULATE ACTON: based on teacher stats - either learn or do own algorithm
            action, epsilon = action_decision(teacher_q_table, availability, accuracy, current_state, epsilon)
            
            # EPISODE UPDATE: variables get new values
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            nr_of_steps += 1
            next_state = state_to_index(next_state)
            trackbackArr.append((current_state, action))
        
            # CALCULATE: the new cell values for the q-table
            update_epsilon_greedy(action, current_state, reward)
            
            #  IF SUCCESS: Mark success and break from loop          
            if done:  
                nr_success += 1 
                break
            
            # NEXT STEP: pass to next step
            current_state = next_state

        # IF SUCCEDED EPISODE: share the reward across all cells visitied in the episode
        if done: EGreedy.trackbackReward(trackbackArr) 

    # FINAL STATS: calculate to return
    avg_reward_per_episode = total_reward /  nr_episodes
    success_rate = (nr_success / nr_episodes) * 100
    avg_steps_per_episode = total_steps / nr_episodes

    return EGreedy.agent_q_table, avg_reward_per_episode, success_rate, avg_steps_per_episode


############################################################################
######                        Main Function                           ######
############################################################################
def main():

    TEACHER_AVAILABILITY = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    TEACHER_ACCURACY = [0.2, 0.4, 0.6, 0.8, 1.0]

    avg_rewards_train = np.zeros((6,5))
    suc = 0
    i = 0
    ac = 0
    av = 0
    lowest_suc = 100
    while i < 1:
        for av in TEACHER_AVAILABILITY:
            for ac in TEACHER_ACCURACY:
                q_table, reward_per_episode, success_rate_b, avg_steps_per_episode = train_agent(50000)

            #plot_rewards(reward_per_episode, "Reward per Episode for Q-learning")

                avg_reward_per_episode, success_rate_t, avg_steps_per_episode = evaluate_agent(q_table, 50000)

            #print("Average Reward per Episode:", avg_reward_per_episode)
            #print("Success Rate:", success_rate)
            #print("Steps per Episode:", avg_steps_per_episode)
                #agent_q_table, reward_per_episode, success_rate, avg_steps_per_episode = train_agent_with_teacher(q_table,10000, random.choice(TEACHER_AVAILABILITY),random.choice(TEACHER_ACCURACY))
                print(f"AVAILABILITY {av},          ACCURACY: {ac}")
                agent_q_table, reward_per_episode, success_rate, avg_steps_per_episode = train_agent_with_teacher(q_table,10000, av, ac)
                suc = ((suc * i) + success_rate) / (i + 1)
                print("Average Reward per Episode:", reward_per_episode)
                print("Success Rate:", success_rate)
                print("Steps per Episode:", avg_steps_per_episode)
                print("Running average success rate:", suc)
                if suc < lowest_suc: lowest_suc = suc

        print(f"lowest success: {lowest_suc}")
        i += 1
    #plot_heatmap(
    #    avg_rewards_train,
    #    TEACHER_ACCURACY,      # X-axis: Teacher accuracy
    #    TEACHER_AVAILABILITY,  # Y-axis: Teacher availability
    #    "Average Reward for Different Teacher Availability and Accuracy\n (Q-learning Teacher)",  # Title of the heatmap
    #    "Accuracy",            # X-axis label
    #    "Availability",        # Y-axis label
    #)

if __name__ == '__main__':
    main()
