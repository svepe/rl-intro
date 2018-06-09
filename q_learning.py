import gym
import numpy as np
from matplotlib import pyplot as plt


# Learning rate
alpha = 0.05
# Discounting factor
gamma = 0.999
# Exploration rate
epsilon = 0.1


def rollout(env, Q, train=False):
    episode_reward = 0

    # Get initial state
    state = env.reset()
    for t in range(env.unwrapped.spec.max_episode_steps):
        # Choose action
        if train and np.random.uniform(0., 1.) < epsilon:
            # Select a random action
            action = np.random.choice(env.action_space.n)
        else:
            # Select the greedy action
            action = np.argmax(Q[state])

        next_state, reward, done, _ = env.step(action)

        episode_reward += reward

        if train:
            # Read the current state-action value
            Q_sa = Q[state, action]

            # Find the value of the next state
            next_Q_s = np.max(Q[next_state])

            # Calculate the updated state action value
            new_Q_sa = Q_sa + alpha * (reward + gamma * next_Q_s - Q_sa)

            # Write the updated value
            Q[state, action] = new_Q_sa

        state = next_state

        if done:
            break

    return episode_reward


def plot(Q):
    V = np.max(Q, 1).reshape(4, 4)
    # Visualise resulting values
    plt.imshow(V, interpolation='none', aspect='auto', cmap='RdYlGn')
    plt.xticks([0, 1, 2, 3])
    plt.yticks([0, 1, 2], ['2', '1', '0'])
    plt.colorbar()

    arrows = ['\u25c0', '\u25bc', '\u25b6', '\u25b2']
    for (i, j), v in np.ndenumerate(np.around(V, 2)):
        a = np.argmax(Q[i])
        label = arrows[a] + "\n" + str(v)
        plt.gca().text(j, i, label, ha='center', va='center')

    plt.show()


def main():
    env = gym.make('FrozenLake-v0')

    Q = np.ones((env.observation_space.n, env.action_space.n))

    episodes = int(1e4)
    average_reward = 0.
    for e in range(episodes):
        # Train with e-greedy policy
        rollout(env, Q, train=True)

        # Test with greedy policy
        episode_reward = rollout(env, Q, train=False)

        # Keep an estimate of the average reward for the past 100 episodes
        average_reward = 0.99 * average_reward + 0.01 * episode_reward

        # Reward threshold is 0.78 (theoretical optimum is 0.81)
        if average_reward >= env.spec.reward_threshold:
            print('Solved!')
            break

        print("{}: {}".format(e, average_reward))

    plot(Q)


if __name__ == "__main__":
    main()
