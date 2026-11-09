"""
Q-Learning on FrozenLake from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

def init_q_table(num_states, num_actions):
    matrix = np.zeros((num_states, num_actions))
    return matrix

def max_q_value(q_table, state):
    q_table=np.array(q_table,dtype=float)
    value=np.max(q_table[state])
    return value

def greedy_action(q_table, state):
    return int(np.argmax(q_table[state]))

def sample_random_action(action_space):
    return int(action_space.sample())

def should_explore(epsilon, rng):
    return bool(rng.random() < epsilon)

def epsilon_greedy_action(q_table, state, epsilon, action_space, rng):
    if should_explore(epsilon,rng):
        return int(sample_random_action(action_space))
    q_values=q_table[state]
    greed_value=np.max(q_values)

    best_actions = np.flatnonzero(q_values == greed_value)

    return int(rng.choice(best_actions))

def decay_epsilon(epsilon, decay_rate, min_epsilon):
    return max(min_epsilon, epsilon * decay_rate)

def td_target(reward, gamma, q_table, next_state, done):
    if done:
        return float(reward)
    td_target =reward + gamma * max_q_value(q_table,next_state)
    return float(td_target)

def td_error(target, q_table, state, action):
    return float(target - q_table[state, action])

def q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma):
    target=td_target(reward,gamma,q_table,next_state,done)
    error=td_error(target,q_table,state,action)

    q_table[state,action]+=alpha*error

    return float(q_table[state,action])

def interaction_step(env, q_table, state, epsilon, alpha, gamma, rng):
    
    greedy_action=epsilon_greedy_action(q_table,state,epsilon,env.action_space,rng)

    next_state, reward, terminated, truncated, info = env.step(greedy_action)

    done = bool(terminated or truncated)
    next_state = int(next_state)
    reward = float(reward)

    q_learning_update(q_table,state,greedy_action,reward,next_state,done,alpha,gamma)

    return (next_state,reward,done)

def run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps=200):

    state,info=env.reset()
    state=int(state)

    total_reward=0.0

    for _ in range(max_steps):
        next_state,reward,done=interaction_step(env,q_table,state,epsilon,alpha,gamma,rng)

        total_reward+=reward

        state=next_state

        if done:
            break
    return float(total_reward)

def train_q_learning(env,num_episodes,alpha=0.8,gamma=0.95,epsilon_start=1.0,epsilon_min=0.01,epsilon_decay=0.99,seed=0,max_steps=200):
    
    rng = np.random.default_rng(seed)
    env.action_space.seed(seed)
    env.reset(seed=seed)

    states = env.observation_space.n
    actions = env.action_space.n

    q_table = init_q_table(states, actions)

    epsilon = epsilon_start
    returns = []

    for episode in range(num_episodes):

        episode_reward = run_training_episode(env,q_table,epsilon,alpha,gamma,rng,max_steps)

        returns.append(float(episode_reward))

        if episode < num_episodes - 1:
            epsilon = decay_epsilon(epsilon,epsilon_decay,epsilon_min)

    return q_table, returns

def extract_greedy_policy(q_table):
    policy = np.argmax(q_table, axis=1).astype(np.int64)
    return policy

def run_greedy_episode(env, policy, seed=None, max_steps=200):
    if seed is not None:
        state, info = env.reset(seed=seed)
    else:
        state, info = env.reset()

    final_reward = 0.0

    for _ in range(max_steps):
        action=int(policy[state])

        next_state,reward,terminated,truncated,info=env.step(action)

        final_reward=float(reward)
        state=int(next_state)

        if terminated or truncated:
            break    
    return bool(final_reward>0.0)

def evaluate_success_rate(env, policy, num_episodes, seed=0, max_steps=200):
    successes=0
    for episode in range(num_episodes):
        episode_seed=seed+episode

        success=run_greedy_episode(env,policy,episode_seed,max_steps)

        if success:
            successes+=1
    return float(successes/num_episodes)

