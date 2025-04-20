env = StaticGridEnv(seed = 42)
############################################################################
######                Train agent using Q-learning                    ######
############################################################################
q_table2 = np.full((10,10), 0.5)
q_table2[(9,9)] = 100
q_table = [(100, 4)]

def state_to_index(state):
    x, y = state
    return x * 10 + y

def neighbours(state):
    y, x = state
    arr = np.empty(4, dtype=object)
    if 0 < y < 10 and x < 10: arr[0] = ((y-1, x))
    if y <= 8 and x < 10: arr[1] = ((y+1, x))
    if y < 10 and 0 < x < 10: arr[2] = ((y, x-1))
    if y < 10 and x <= 8: arr[3] = ((y, x+1))
    #print(f"arr in neighbours {arr}")
    return neighbours_max(arr)

def neighbours_max(arr):
    q_values = []
    # valid actions store the indexes of elements in arr[] - aka actions for agent
    valid_actions = []
    for i in range(len(arr)):
        if arr[i] is not None:
            q_val = q_table2[arr[i]]
            q_values.append(q_val)
            valid_actions.append(i)
            #print(f"q_table2[{arr[i]}] = {q_val}")

    max_index = np.argmax(q_values)
    action = valid_actions[max_index]
    #print(f"Chosen action index: {action}")
    return action



# if the next move was to bump into the wall, then calculate the coordinates of the wall,
# since the next-move value will not have this information (as agent stays in place, so
# next state in this scenario is the current state)
def match_action(action, state):
    y, x = state
    next_state = (y, x)
    if action == 0:
        if 0 < y < 10 and x < 10: 
            next_state = (y-1, x)
    elif action == 1:
        if y <= 8 and x < 10: 
            next_state = (y+1, x)
    elif action == 2:
        if y < 10 and 0 < x < 10:
            next_state = (y, x-1)
    elif action == 3:
        if y < 10 and x <= 8: 
            next_state = (y, x+1)
        
    # print(f"next: {next_state}")
    return next_state


def trackbackReward(arr):
    arr.reverse()
    arr = list(dict.fromkeys(arr))
    decay = 0.95
    target = 1.0
    alpha = 0.2
    for state in arr:
        q_table2[state] += alpha * (target - q_table2[state])
        target *= decay  # decay the value for earlier steps
    print(f"trackbackArr: after reward {arr}")


def trackbackPunishment(arr):
    arr.reverse()
    arr = list(dict.fromkeys(arr))
    decay = 0.95
    target = 0.0
    alpha = 0.2
    for state in arr:
        q_table2[state] += alpha * (target - q_table2[state])
        target *= decay  # optional: soften punishment the further back we go
    print(f"trackbackArr: after punishment {arr}")


def updateWithRewards(reward, action, state, next_state, trackbackArr):
    if reward == -5:
        q_table2[match_action(action, state)] = q_table2[next_state] * 0.8
    elif reward == -1:
        #q_table2[next_state] += reward
        q_table2[next_state] = q_table2[next_state] * 0.96
        #print(f"next state: {next_state}")
        trackbackArr.append(next_state)
    return trackbackArr
        
# NEXT FIX: when we reach the botom (end of map), the agent just cant move futher
# currently it updates the field its on to -1: line 71 but it should instead move right
# in output arrneighbours are printed, and when there are 3 of them then argmax returns the
# index of max. but because in neighbours we do not clear up which neighbour is missing
# which results in going wrong direction. so next step is to change that to avoid the agent
# being stuck in place 
def train_agent(max_total_steps=1000):
    # initialize the return values
    reward_per_episode = []
    success_rate = 0
    avg_steps_per_episode = 0.0
    #state_space_size = 10 * 10       # 100 states
    #action_space_size = 4            # 4 actions (up, down, left, right)
    max_total_steps= 10000
    episode_reward = 0
    # first state initialization
    total_episode_num = 0
    success_num = 0
    steps = 0
    total_steps = 0

    while max_total_steps > total_steps :
        total_episode_num += 1
        state = env.reset()
        trackbackArr = []
        trackbackArr.append(state)
        while steps < 100:
            # Look at neighbouts and find the most suitable move based on rewards of neighbours
            action = (neighbours(state))
            
            # Execute the action
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1
            #print(f"next_state: {next_state}")
            #print(f"reward: {reward}")
            #print(f"done: {done}")
            if done: 
                success_num += 1 
                break
            else: trackbackArr = updateWithRewards(reward, action, state, next_state, trackbackArr)

            state = next_state
            #print(f"whole qtable: {q_table2}")
            #print(f"BEFORE TOTAL: {max_total_steps}")
            #print(f"successs: {success_num}")
        trackbackArr.append(next_state)
        if done: trackbackReward(trackbackArr) 
        else: trackbackPunishment(trackbackArr)
        
        print(f"whole qtable: {q_table2}")
        print(f"successs: {success_num}")
        print(f"num of episodes: {total_episode_num}")
        #if total_episode_num == 3: break
        total_steps += steps
        reward_per_episode.append(episode_reward)
        # continue generating new states, so prepare the variables
        if (max_total_steps > 0):
            steps = 0
            episode_reward = 0
    # Render the environment

    print(f"whole qtable: {q_table2}")

    avg_steps_per_episode = total_steps / total_episode_num
    success_rate = success_num / total_episode_num
    env.render (delay = 2 , episode = total_episode_num , learning_type = "Q - learning ", availability =0.8 , accuracy = 0.9)
   
    # Final update of return values

    return q_table2, reward_per_episode, success_rate, avg_steps_per_episode

