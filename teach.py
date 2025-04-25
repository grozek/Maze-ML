#!/usr/bin/env python3
"""
teach.py

UNSW COMP3411/9814 Artificial Intelligence

Gabriela Roznawska 
COMP-3411: Artificial Intelligence
Term 1, 2025

"""
import random
import pygame
import matplotlib
import numpy as np
import pandas as pd
from env import StaticGridEnv
from utils import plot_rewards, plot_heatmap


# INITIALIZE ENV: Setting up the enviornment with custom seed 65
env = StaticGridEnv(seed = 65)
############################################################################
######                Train agent using Q-learning                    ######
############################################################################
# state_to_index
# Translates the states from 2D Matrix to 1D index values
def state_to_index(state):
    y, x = state
    return y * 10 + x

# index_to_state
# Translates the states from 1D index values to 2D Matrix
def index_to_state(index):
    y = index // 10
    x = index % 10
    return (y, x)

# stepValueUpdate
# Update the elements that are refreshed at every step of an episode
def stepValueUpdate(episode_reward, reward, next_state, state, steps):
    state = next_state  
    steps += 1
    episode_reward += reward
    return episode_reward, state, steps


# train_agent
# Agent training algorithm that uses q-learning algorithm
def train_agent(max_total_steps):

    # INITIALIZE: Return variables and helpful operational elements
    t_table = np.zeros((100, 4), dtype=float)
    lr = 0.2
    discount_factor = 0.95
    exploration_prob = 1  
    min_epsilon = 0.05
    decay = 0.995
    total_episode_num = 0
    total_steps = 0
    success_num = 0
    reward_per_episode = []

    # TOTAL EPISODES LOOP: Iterates over all episodes, as long as max step # isnt reached
    while total_steps < max_total_steps:
        state = env.reset()
        state = state_to_index(state)
        episode_reward = 0
        steps = 0
        total_episode_num += 1

        # EPISODE LOOP: every episode has max 100 steps to suceed
        while steps < 100:

            # EXPLORATION vs EXPLOTIATION: with every iteration gain confidence in q_table
            if np.random.rand() < exploration_prob:
                action = np.random.randint(0, 4)
            else:
                action = np.argmax(t_table[state])

            # ACTION: Perform action, update the statistics and update q_table with action rewards
            next_state, reward, done, _ = env.step(action)
            next_state = state_to_index(next_state)
            
            t_table[state, action] += lr * (
                reward + discount_factor * np.max(t_table[next_state]) - t_table[state, action]
            )
            # EPISODE UPDATE: Update the episode with the statistics
            episode_reward, state, steps = stepValueUpdate(episode_reward, reward, next_state, state, steps)
            

            # IF SUCCESS: Reward leading steps, update stats and break from loop
            if done:
                success_num += 1
                break

        # STEP UPDATE: Update the episode with the statistics
        reward_per_episode.append(episode_reward)
        exploration_prob = max(min_epsilon, exploration_prob * decay)
        total_steps += steps

    # STATS: calculate final stats
    avg_steps_per_episode = total_steps / total_episode_num
    success_rate = (success_num / total_episode_num) * 100

    return t_table, reward_per_episode, success_rate, avg_steps_per_episode



############################################################################
######               Evaluate agent after training                    ######
############################################################################

# evaluate_agent
# Perform agent evaluation based on previously calculated q_table
def evaluate_agent(input_table, max_total_steps):

    # INITIALIZE: The variables needed to evaluate the returning variables
    nr_of_steps = 0
    nr_success = 0
    nr_episodes = 0
    total_reward = 0
    total_steps = 0

    # TOTAL EPISODES LOOP: As long as max step celling is not reached
    while max_total_steps > total_steps:
        current_state = env.reset()
        current_state = state_to_index(current_state)
        nr_episodes += 1
        nr_of_steps = 0

        # STEP UPDATE: Update the episode with the statistics
        while nr_of_steps < 100:

            # ACTION: Perform based on q-table, and update statistics
            action = np.argmax(input_table[current_state]) 
            next_state, reward, done, _ = env.step(action)
            next_state = state_to_index(next_state)
            total_reward += reward
            current_state = next_state
            nr_of_steps += 1

            # IF SUCCESS: Mark success and break from loop
            if done:  
                nr_success += 1 
                break 
        
        total_steps += nr_of_steps

    # TOTAL STATS: Calculate total statistics using data from loops
    avg_reward_per_episode = total_reward /  nr_episodes
    success_rate = (nr_success / nr_episodes) * 100
    avg_steps_per_episode = total_steps / nr_episodes

    return avg_reward_per_episode, success_rate, avg_steps_per_episode


############################################################################
######        Train agent using Q-learning with Teacher Advice        ######
############################################################################

# EGreedy algorithm
# Stores the agent table. Agent implements elements of both q-learning and greedy learning
class EGreedy():
    agent_q_table = np.full((100,4), 0, dtype=float)


# action_decision
# Perform decision of what action should be taken based on teachers accuracy and availbility
# If teaching does not happen - agent works on its own epsilon algorithm. Uses tMode variable
# to mark whether learning was directed by accurate teacher
def action_decision (teacher_q_table, availability, accuracy, current_state, epsilon):
    tMode = False

    # TEACHER: For high availability and accuracy - teacher chooses the best next action,
    # but if its available but not accurate - computes random 'non-wrong' action
    if np.random.rand() < availability:
        if np.random.rand() < accuracy:
            action = np.argmax(teacher_q_table[current_state])
            tMode = True
        else:
            wrong_action = np.argmax(teacher_q_table[current_state])
            while True:
                action = np.random.randint(0, 4)
                if action != wrong_action:
                    break
    # AGENT: If teacher is not available then randomly explore an action (probability decays
    # over time) or choose the best move based on q-table
    else:
        if np.random.rand() < epsilon:
           action = np.random.randint(0, 4)
        else:
            action = np.argmax(EGreedy.agent_q_table[current_state])

    return action, epsilon, tMode


# train_agent_with_teacher
# Given q-table of the teacher, if possible, agent learns from the teacher
# If not, agent learns on its own creating its table using a different algorithm
def train_agent_with_teacher(teacher_q_table, max_total_steps, availability, accuracy):
    # INITIALIZE: The variables needed to evaluate the returning variables
    nr_of_steps = 0
    nr_success = 0
    nr_episodes = 0
    total_reward = 0
    total_steps = 0
    decay_rate = 0.99
    epsilon = 0.2
    discount_factor = 0.95
    lr = 0.1

    # TOTAL EPISODES LOOP: As long as max step celling is not reached
    while max_total_steps > total_steps:

        # STEP UPDATE: Update the episode with the statistics
        current_state = env.reset()
        current_state = state_to_index(current_state)
        nr_episodes += 1
        nr_of_steps = 0
        epsilon = max(0.05, epsilon * decay_rate)
        
        while nr_of_steps < 100:
            # CALCULATE ACTON: based on teacher stats - either learn or do own algorithm
            action, epsilon, tMode = action_decision(teacher_q_table, availability, accuracy, current_state, epsilon)

            # EPISODE UPDATE: variables get new values
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            nr_of_steps += 1
            next_state = state_to_index(next_state)

            # CALCULATE: the new cell values for the q-table
            if tMode == True:
                EGreedy.agent_q_table[current_state, action] = teacher_q_table[current_state, action]
            else:
                highest_next_state = np.max(EGreedy.agent_q_table[next_state])
                EGreedy.agent_q_table[current_state, action] = (1 - lr) * EGreedy.agent_q_table[current_state, action] + lr * (reward + discount_factor * highest_next_state )
            
            #  IF SUCCESS: Mark success and break from loop          
            if done:  
                nr_success += 1 
                break

            # NEXT STEP: pass to next step
            current_state = next_state

        total_steps += nr_of_steps

    # FINAL STATS: calculate to return
    avg_reward_per_episode = total_reward /  nr_episodes
    success_rate = (nr_success / nr_episodes) * 100
    avg_steps_per_episode = total_steps / nr_episodes

    return EGreedy.agent_q_table, avg_reward_per_episode, success_rate, avg_steps_per_episode



############################################################################
######                        Main Function                           ######
############################################################################


# main
# Computes the q-table of teacher, trains the student with teacher, evaluates its 
# performance and displays the results in two heat maps - one for training and 
# one for evaluation
def main():

    # VARIABLE SETUP: data used to contain information about our model
    TEACHER_AVAILABILITY = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    TEACHER_ACCURACY = [0.2, 0.4, 0.6, 0.8, 1.0]
    avg_reward_eval = np.zeros((6,5))
    avg_reward_train = np.zeros((6,5))

    # train_agent: returns fully trained teacher with q-table stored at t_table
    t_table, reward_per_episode, success_rate_train, avg_steps_per_episode = train_agent(50000)

    # COMMENTED OUT: uncomment to generate data for Task 2 report submission
    #avg_reward_per_episode_eval, success_rate, avg_steps_per_episode = evaluate_agent(t_table, 10000)
    #print("Success Rate:", success_rate)
    #print("Reward per Episode:", avg_reward_per_episode_eval)
    #print("Steps per Episode:", avg_steps_per_episode)

    # COMMENTED OUT: uncomment to generate data for Task 1 report submission
    #plot_rewards(reward_per_episode, "Reward per Episode for Q-learning during training")

    # LOOP: Compute the performance of student that learns with teacher on different values
    # of availability and accuracy of the teacher
    for av in TEACHER_AVAILABILITY:
        for ac in TEACHER_ACCURACY:

            # clear up the agent table at every iteration to avoid "too accurate" learning
            #EGreedy.agent_q_table = np.zeros((100,4), dtype = float)

            # TRAIN STUDENT: training student with q-table stored in EGreedy.agent_q_table
            # with help of the teacher with its q-table at t_table. Uses only 10000 epochs
            EGreedy.agent_q_table, avg_reward_per_episode, success_rate, avg_steps_per_episode = train_agent_with_teacher(t_table,12000, av, ac)
            avg_reward_train[(TEACHER_AVAILABILITY.index(av), TEACHER_ACCURACY.index(ac))] = avg_reward_per_episode
            
            # EVALUATE STUDENT: testing student's performance on test set with set availability
            # and accuracy of the teacher. Storing the average reward in a cell of 2d array, so
            # that once plotted to heatmap - each cell represents one testing cycle
            avg_reward_per_episode, success_rate, avg_steps_per_episode = evaluate_agent(EGreedy.agent_q_table, 10000)
            avg_reward_eval[(TEACHER_AVAILABILITY.index(av), TEACHER_ACCURACY.index(ac))] = avg_reward_per_episode


    # DATA FRAMES: data frames store the average training and testing rewards
    df_reward_train = pd.DataFrame(avg_reward_train)
    df_reward_eval = pd.DataFrame(avg_reward_eval)

    # TRAIN HEATMAP : heatmap illustrating the average rewards during the train cycle
    plot_heatmap(
        avg_reward_train,
        TEACHER_ACCURACY,      # X-axis: Teacher accuracy
        TEACHER_AVAILABILITY,  # Y-axis: Teacher availability
        "Average Reward for Different Teacher Availability and Accuracy\n (Student Q-learning From Teacher)",  # Title of the heatmap
        "Accuracy",            # X-axis label
        "Availability",        # Y-axis label
    )

    # EVALUATION HEATMAP : heatmap illustrating the average rewards during the test cycle
    plot_heatmap(
        avg_reward_eval,
        TEACHER_ACCURACY,      # X-axis: Teacher accuracy
        TEACHER_AVAILABILITY,  # Y-axis: Teacher availability
        "Average Reward for Different Teacher Availability and Accuracy\n (Testing Student's Performance)",  # Title of the heatmap
        "Accuracy",            # X-axis label
        "Availability",        # Y-axis label
    )
    

if __name__ == '__main__':
    main()