from copy import deepcopy
from matplotlib import pyplot as plt
from numpy import ndenumerate, around, argmax

# Terminal states have nonzero reward
world = [
    [0.,   0.,   0.,   1.],
    [0., None,   0.,  -1.],
    [0.,   0.,   0.,   0.],
]

# Actions: Up, Down, Left, Right
actions = [0, 1, 2, 3]

# Actions succeed with probability (1. - noise)
noise = 0.2

# Discount factor
gamma = 0.9


def move_up(state):
    i, j = state

    if i > 0 and world[i - 1][j] is not None:
        return (i - 1, j)
    else:
        return state


def move_down(state):
    i, j = state

    if i < len(world) - 1 and world[i + 1][j] is not None:
        return (i + 1, j)
    else:
        return state


def move_left(state):
    i, j = state

    if j > 0 and world[i][j - 1] is not None:
        return (i, j - 1)
    else:
        return state


def move_right(state):
    i, j = state

    if j < len(world[0]) - 1 and world[i][j + 1] is not None:
        return (i, j + 1)
    else:
        return state


def step(state, action):
    # Up
    if action == 0:
        next_states = [move_up(state), move_left(state), move_right(state)]

    # Down
    if action == 1:
        next_states = [move_down(state), move_right(state), move_left(state)]

    # Left
    if action == 2:
        next_states = [move_left(state), move_down(state), move_up(state)]

    # Right
    if action == 3:
        next_states = [move_right(state), move_up(state), move_down(state)]

    # Probability of every possible next state
    probs = [1. - noise, noise / 2., noise / 2.]

    return probs, next_states


def stateValue(state, V):
    i, j = state

    state_values = [0.] * len(actions)
    for a in actions:
        # Determine all possible next states and their probabilities
        probs, next_states = step(state, a)

        for p, (next_i, next_j) in zip(probs, next_states):

            # Accumulate current reward
            state_values[a] += p * world[i][j]

            # Accumulate future reward if state not terminal
            if world[i][j] == 0.:
                state_values[a] += p * gamma * V[next_i][next_j]

    return max(state_values), argmax(state_values)


def valueIteration(V):
    next_V = deepcopy(V)
    policy = deepcopy(world)

    for i in range(len(V)):
        for j in range(len(V[i])):

            # Skip obstacles
            if world[i][j] is None:
                continue

            # Calculate value of the state and find the best action
            next_V[i][j], policy[i][j] = stateValue((i, j), V)

    return next_V, policy


if __name__ == "__main__":
    # Initialise the value for each state to 0.
    h = len(world)
    w = len(world[0])
    V = [[0.] * w for _ in range(h)]

    # Set horizon
    H = 100

    # Run value iteration
    for t in range(H):
        V, policy = valueIteration(V)

    # Visualise resulting values
    plt.imshow(V, interpolation='none', aspect='auto', cmap='RdYlGn')
    plt.xticks([0, 1, 2, 3])
    plt.yticks([0, 1, 2], ['2', '1', '0'])
    plt.colorbar()

    arrows = ['\u25b2', '\u25bc', '\u25c0', '\u25b6']
    for (i, j), v in ndenumerate(around(V, 2)):
        a = policy[i][j]

        label = ""
        if a is not None:
            label += "\n" + str(v)
            if world[i][j] == 0.:
                label = arrows[a] + label

        plt.gca().text(j, i, label, ha='center', va='center')

    plt.show()
