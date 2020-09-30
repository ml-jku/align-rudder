import numpy as np


class learn_sr():
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

    def update(self, _state, action, reward, state, done):
        return NotImplementedError


def onehot(value, max_value):
    vec = np.zeros(max_value)
    vec[value] = 1
    return vec


class SuccessorRepresentation(learn_sr):
    def __init__(self, env, rep_lr=0.05, rep_gamma=0.99):
        super().__init__(env.observation_space, env.action_space)

        self.env = env
        self.rep_lr = rep_lr
        self.rep_gamma = rep_gamma
        self.env.reset()
        self.maze = self.env.unwrapped.maze.to_value()
        self.maze[self.maze == 2] = 0
        self.maze[self.maze == 3] = 0
        self.maze[self.maze == 4] = 0
        _state_space = self.state_space.high[0]
        self.state_size = int(_state_space[0]) * int(_state_space[1]) * int(_state_space[2]) - np.sum(self.maze)
        self.maze[self.maze == 1] = -1
        m = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[2]):
                    if self.maze[i, j, k] != -1:
                        self.maze[i, j, k] = m
                        m += 1
        self.sr_table = np.zeros([self.state_size, self.state_size], dtype=float)

    def update(self, _state, action, reward, state, done):
        _state = self.transform_state(_state)
        state = self.transform_state(state)

        if done:
            prev_value = self.sr_table[_state, :]
            visit_indicator = onehot(_state, self.state_size)
            next_state_value = visit_indicator
            self.sr_table[_state, :] = (1 - self.rep_lr) * prev_value + self.rep_lr * next_state_value
        else:
            prev_value = self.sr_table[_state, :]
            visit_indicator = onehot(_state, self.state_size)
            next_state_value = visit_indicator + self.rep_gamma * self.sr_table[state, :]
            self.sr_table[_state, :] = (1 - self.rep_lr) * prev_value + self.rep_lr * next_state_value

    def transform_state(self, state):
        # transforms state to index values in the sr table
        index = self.maze[state[0], state[1], state[2]]
        return index

    def transform_back(self, index):
        # transform state index to state to be used with env
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[2]):
                    if self.maze[i, j, k] == index:
                        state = [i, j, k]
                        break
        return state

    def give_sr(self, _state, state):
        return self.sr_table[np.int(_state), np.int(state)]
