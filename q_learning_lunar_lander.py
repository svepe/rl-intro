import gym
import numpy as np

# Learning rate
alpha = 1e-4
# Discounting factor
gamma = 0.9999
# Exploration rate
epsilon = 0.01

state_size = (9, 9, 9, 9, 9, 9, 2, 2)
min_state = [-1.2, -0.3, -2.4, -2.4, -3.5, -6.0, 0., 0.]
max_state = [ 1.2,  1.2,  2.4,  0.7,  3.8,  7.9, 1., 1.]
bins = [np.arange(min, max, (max - min) / sz)[1:]
        for sz, min, max in zip(state_size, min_state, max_state)]


def discretise(state):
    res = []
    for s, b in zip(state, bins):
        res.append(np.digitize(s, b).item())
    return tuple(res)


def rollout(env, Q, train=False):
    episode_reward = 0

    state = env.reset()
    state = discretise(state)
    for t in range(env.unwrapped.spec.max_episode_steps):
        # Choose action
        if train and np.random.uniform(0., 1.) < epsilon:
            # Select a random action
            action = np.random.choice(env.action_space.n)
        else:
            # Select the greedy action
            action = np.argmax(Q[state])

        next_state, reward, done, _ = env.step(action)
        next_state = discretise(next_state)

        episode_reward += reward

        if train:
            # Read the current state-action value
            Q_sa = Q[state + (action,)]

            # Find the value of the next state
            next_Q_s = np.max(Q[next_state])

            # Calculate the updated state action value
            new_Q_sa = Q_sa + alpha * (reward + gamma * next_Q_s - Q_sa)

            # Write the updated value
            Q[state + (action,)] = new_Q_sa

        state = next_state

        if done:
            break

    return episode_reward


def main():

    env = gym.make('LunarLander-v2')

    Q = np.ones(state_size + (env.action_space.n,))

    episodes = int(1e4)
    average_reward = 0.
    for e in range(episodes):
        # Train
        rollout(env, Q, train=True)

        # Test
        episode_reward = rollout(env, Q, train=False)
        average_reward = 0.99 * average_reward + 0.01 * episode_reward

        if average_reward >= env.spec.reward_threshold:
            print('Solved!')
            break

        print("{}: {}".format(e, average_reward))


if __name__ == "__main__":
    main()
