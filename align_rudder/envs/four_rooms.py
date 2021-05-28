import numpy as np
from gym.spaces import Box
from gym.spaces import Discrete
from mazelab import BaseEnv
from align_rudder.envs.mazes.maze import BaseMaze
from align_rudder.envs.mazes.maze import Object
from align_rudder.envs.mazes.four_rooms import rooms_maze
from mazelab import DeepMindColor as color
from mazelab import VonNeumannMotion
import random

WIDTH = 6
ROOMS = 23
x, doors, paired_doors, pi = rooms_maze(WIDTH, ROOMS)


class Maze(BaseMaze):
    @property
    def size(self):
        return x.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(x == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(x == 1), axis=1))
        agent = Object('agent', 2, color.agent, False, [])
        goal = Object('goal', 3, color.goal, False, [])
        door = Object('door', 4, color.button, False, np.stack(np.where(x == 4), axis=1))

        return free, obstacle, agent, goal, door


class FourRooms(BaseEnv):
    def __init__(self):
        super(FourRooms, self).__init__()

        self.maze = Maze()
        self.motions = VonNeumannMotion()
        self.start_idx = []
        for i in range(ROOMS - 3):
            self.start_idx.append([[3, 3, i]])
        self.goal_idx = [[WIDTH - 2, WIDTH - 2, ROOMS - 1]]
        self.width = WIDTH
        self.rooms = ROOMS
        self.observation_space = Box(low=0, high=np.max([WIDTH, ROOMS]), shape=np.array(self.start_idx[0]).shape,
                                     dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))
        self.steps = 0

    def step(self, action):
        self.steps += 1
        random_action = np.random.choice([0, 1, 2, 3])
        if np.random.choice([0, 1], p=[0.01, 0.99]) == 0:
            action = random_action
        current_position = self.maze.objects.agent.positions[0]
        # write a new function which takes care of the new functionalities
        new_position = self.get_new_position(current_position, action)

        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        if self._is_goal(new_position):
            reward = +3 - self.steps / 200 + 53.32 / 200
            done = True
            self.steps = 0
        elif not valid and self.steps == 200:
            reward = 1
            done = False
        elif not valid:
            reward = 0
            done = False
        elif self.steps == 200:
            reward = 1
            done = False
        else:
            reward = 0
            done = False

        return np.array(self.maze.objects.agent.positions[0]), reward, done, {}

    def reset(self):
        self.steps = 0
        self.maze.objects.agent.positions = random.sample(self.start_idx, k=1)[0]
        self.maze.objects.goal.positions = self.goal_idx
        return np.array(self.maze.objects.agent.positions[0])

    def get_new_position(self, position, action):
        motion = self.motions[action]

        door_present, door = self._is_door(position)
        new_position = [position[0] + motion[0], position[1] + motion[1], position[2]]
        if door_present:
            # check if the action is same as the action required to pass through the door
            # if yes, select the position of the next door
            for d, dn, a in paired_doors:
                if door[0] == d[0] and door[1] == d[1] and door[2] == d[2] and action == a:
                    next_door = random.sample(dn, k=1)[0]
                    new_position = next_door
        return new_position

    def _is_door(self, position):
        out = False
        door = 0
        for pos in self.maze.objects.door.positions:
            if position[0] == pos[0] and position[1] == pos[1] and position[2] == pos[2]:
                out = True
                door = pos
                break
        return out, door

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0 and position[2] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1] and position[2] < \
                      self.maze.size[2]
        passable = False
        if within_edge:
            passable = not self.maze.to_impassable()[position[0]][position[1]][position[2]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1] and position[2] == pos[2]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()


class FourRoomsv1(BaseEnv):
    def __init__(self):
        super(FourRoomsv1, self).__init__()

        self.maze = Maze()
        self.motions = VonNeumannMotion()
        self.start_idx = []
        for i in range(ROOMS - 3):
            self.start_idx.append([[3, 3, i]])
        self.goal_idx = [[WIDTH - 2, WIDTH - 2, ROOMS - 1]]
        self.width = WIDTH
        self.rooms = ROOMS
        self.observation_space = Box(low=0, high=np.max([WIDTH, ROOMS]), shape=np.array(self.start_idx[0]).shape,
                                     dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))
        self.steps = 0

    def step(self, action):
        self.steps += 1
        random_action = np.random.choice([0, 1, 2, 3])
        if np.random.choice([0, 1], p=[0.05, 0.95]) == 0:
            action = random_action
        current_position = self.maze.objects.agent.positions[0]
        # write a new function which takes care of the new functionalities
        new_position = self.get_new_position(current_position, action)

        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        if self._is_goal(new_position):
            reward = +3 - self.steps / 200 + 53.32 / 200
            done = True
            self.steps = 0
        elif not valid and self.steps == 200:
            reward = 1
            done = False
        elif not valid:
            reward = 0
            done = False
        elif self.steps == 200:
            reward = 1
            done = False
        else:
            reward = 0
            done = False

        return np.array(self.maze.objects.agent.positions[0]), reward, done, {}

    def reset(self):
        self.steps = 0
        self.maze.objects.agent.positions = random.sample(self.start_idx, k=1)[0]
        self.maze.objects.goal.positions = self.goal_idx
        return np.array(self.maze.objects.agent.positions[0])

    def get_new_position(self, position, action):
        motion = self.motions[action]

        door_present, door = self._is_door(position)
        new_position = [position[0] + motion[0], position[1] + motion[1], position[2]]
        if door_present:
            # check if the action is same as the action required to pass through the door
            # if yes, select the position of the next door
            for d, dn, a in paired_doors:
                if door[0] == d[0] and door[1] == d[1] and door[2] == d[2] and action == a:
                    next_door = random.sample(dn, k=1)[0]
                    new_position = next_door
        return new_position

    def _is_door(self, position):
        out = False
        door = 0
        for pos in self.maze.objects.door.positions:
            if position[0] == pos[0] and position[1] == pos[1] and position[2] == pos[2]:
                out = True
                door = pos
                break
        return out, door

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0 and position[2] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1] and position[2] < \
                      self.maze.size[2]
        passable = False
        if within_edge:
            passable = not self.maze.to_impassable()[position[0]][position[1]][position[2]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1] and position[2] == pos[2]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()


class FourRoomsv2(BaseEnv):
    def __init__(self):
        super(FourRoomsv2, self).__init__()

        self.maze = Maze()
        self.motions = VonNeumannMotion()
        self.start_idx = []
        for i in range(ROOMS - 3):
            self.start_idx.append([[3, 3, i]])
        self.goal_idx = [[WIDTH - 2, WIDTH - 2, ROOMS - 1]]
        self.width = WIDTH
        self.rooms = ROOMS
        self.observation_space = Box(low=0, high=np.max([WIDTH, ROOMS]), shape=np.array(self.start_idx[0]).shape,
                                     dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))
        self.steps = 0

    def step(self, action):
        self.steps += 1
        random_action = np.random.choice([0, 1, 2, 3])
        if np.random.choice([0, 1], p=[0.1, 0.9]) == 0:
            action = random_action
        current_position = self.maze.objects.agent.positions[0]
        # write a new function which takes care of the new functionalities
        new_position = self.get_new_position(current_position, action)

        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        if self._is_goal(new_position):
            reward = +3 - self.steps / 200 + 53.32 / 200
            done = True
            self.steps = 0
        elif not valid and self.steps == 200:
            reward = 1
            done = False
        elif not valid:
            reward = 0
            done = False
        elif self.steps == 200:
            reward = 1
            done = False
        else:
            reward = 0
            done = False

        return np.array(self.maze.objects.agent.positions[0]), reward, done, {}

    def reset(self):
        self.steps = 0
        self.maze.objects.agent.positions = random.sample(self.start_idx, k=1)[0]
        self.maze.objects.goal.positions = self.goal_idx
        return np.array(self.maze.objects.agent.positions[0])

    def get_new_position(self, position, action):
        motion = self.motions[action]

        door_present, door = self._is_door(position)
        new_position = [position[0] + motion[0], position[1] + motion[1], position[2]]
        if door_present:
            # check if the action is same as the action required to pass through the door
            # if yes, select the position of the next door
            for d, dn, a in paired_doors:
                if door[0] == d[0] and door[1] == d[1] and door[2] == d[2] and action == a:
                    next_door = random.sample(dn, k=1)[0]
                    new_position = next_door
        return new_position

    def _is_door(self, position):
        out = False
        door = 0
        for pos in self.maze.objects.door.positions:
            if position[0] == pos[0] and position[1] == pos[1] and position[2] == pos[2]:
                out = True
                door = pos
                break
        return out, door

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0 and position[2] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1] and position[2] < \
                      self.maze.size[2]
        passable = False
        if within_edge:
            passable = not self.maze.to_impassable()[position[0]][position[1]][position[2]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1] and position[2] == pos[2]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()


class FourRoomsv3(BaseEnv):
    def __init__(self):
        super(FourRoomsv3, self).__init__()

        self.maze = Maze()
        self.motions = VonNeumannMotion()
        self.start_idx = []
        for i in range(ROOMS - 3):
            self.start_idx.append([[3, 3, i]])
        self.goal_idx = [[WIDTH - 2, WIDTH - 2, ROOMS - 1]]
        self.width = WIDTH
        self.rooms = ROOMS
        self.observation_space = Box(low=0, high=np.max([WIDTH, ROOMS]), shape=np.array(self.start_idx[0]).shape,
                                     dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))
        self.steps = 0

    def step(self, action):
        self.steps += 1
        random_action = np.random.choice([0, 1, 2, 3])
        if np.random.choice([0, 1], p=[0.15, 0.85]) == 0:
            action = random_action
        current_position = self.maze.objects.agent.positions[0]
        # write a new function which takes care of the new functionalities
        new_position = self.get_new_position(current_position, action)

        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        if self._is_goal(new_position):
            reward = +3 - self.steps / 200 + 53.32 / 200
            done = True
            self.steps = 0
        elif not valid and self.steps == 200:
            reward = 1
            done = False
        elif not valid:
            reward = 0
            done = False
        elif self.steps == 200:
            reward = 1
            done = False
        else:
            reward = 0
            done = False

        return np.array(self.maze.objects.agent.positions[0]), reward, done, {}

    def reset(self):
        self.steps = 0
        self.maze.objects.agent.positions = random.sample(self.start_idx, k=1)[0]
        self.maze.objects.goal.positions = self.goal_idx
        return np.array(self.maze.objects.agent.positions[0])

    def get_new_position(self, position, action):
        motion = self.motions[action]

        door_present, door = self._is_door(position)
        new_position = [position[0] + motion[0], position[1] + motion[1], position[2]]
        if door_present:
            # check if the action is same as the action required to pass through the door
            # if yes, select the position of the next door
            for d, dn, a in paired_doors:
                if door[0] == d[0] and door[1] == d[1] and door[2] == d[2] and action == a:
                    next_door = random.sample(dn, k=1)[0]
                    new_position = next_door
        return new_position

    def _is_door(self, position):
        out = False
        door = 0
        for pos in self.maze.objects.door.positions:
            if position[0] == pos[0] and position[1] == pos[1] and position[2] == pos[2]:
                out = True
                door = pos
                break
        return out, door

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0 and position[2] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1] and position[2] < \
                      self.maze.size[2]
        passable = False
        if within_edge:
            passable = not self.maze.to_impassable()[position[0]][position[1]][position[2]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1] and position[2] == pos[2]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()


class FourRoomsv4(BaseEnv):
    def __init__(self):
        super(FourRoomsv4, self).__init__()

        self.maze = Maze()
        self.motions = VonNeumannMotion()
        self.start_idx = []
        for i in range(ROOMS - 3):
            self.start_idx.append([[3, 3, i]])
        self.goal_idx = [[WIDTH - 2, WIDTH - 2, ROOMS - 1]]
        self.width = WIDTH
        self.rooms = ROOMS
        self.observation_space = Box(low=0, high=np.max([WIDTH, ROOMS]), shape=np.array(self.start_idx[0]).shape,
                                     dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))
        self.steps = 0

    def step(self, action):
        self.steps += 1
        random_action = np.random.choice([0, 1, 2, 3])
        if np.random.choice([0, 1], p=[0.25, 0.75]) == 0:
            action = random_action
        current_position = self.maze.objects.agent.positions[0]
        # write a new function which takes care of the new functionalities
        new_position = self.get_new_position(current_position, action)

        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        if self._is_goal(new_position):
            reward = +3 - self.steps / 200 + 53.32 / 200
            done = True
            self.steps = 0
        elif not valid and self.steps == 200:
            reward = 1
            done = False
        elif not valid:
            reward = 0
            done = False
        elif self.steps == 200:
            reward = 1
            done = False
        else:
            reward = 0
            done = False

        return np.array(self.maze.objects.agent.positions[0]), reward, done, {}

    def reset(self):
        self.steps = 0
        self.maze.objects.agent.positions = random.sample(self.start_idx, k=1)[0]
        self.maze.objects.goal.positions = self.goal_idx
        return np.array(self.maze.objects.agent.positions[0])

    def get_new_position(self, position, action):
        motion = self.motions[action]

        door_present, door = self._is_door(position)
        new_position = [position[0] + motion[0], position[1] + motion[1], position[2]]
        if door_present:
            # check if the action is same as the action required to pass through the door
            # if yes, select the position of the next door
            for d, dn, a in paired_doors:
                if door[0] == d[0] and door[1] == d[1] and door[2] == d[2] and action == a:
                    next_door = random.sample(dn, k=1)[0]
                    new_position = next_door
        return new_position

    def _is_door(self, position):
        out = False
        door = 0
        for pos in self.maze.objects.door.positions:
            if position[0] == pos[0] and position[1] == pos[1] and position[2] == pos[2]:
                out = True
                door = pos
                break
        return out, door

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0 and position[2] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1] and position[2] < \
                      self.maze.size[2]
        passable = False
        if within_edge:
            passable = not self.maze.to_impassable()[position[0]][position[1]][position[2]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1] and position[2] == pos[2]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()


class FourRoomsv5(BaseEnv):
    def __init__(self):
        super(FourRoomsv5, self).__init__()

        self.maze = Maze()
        self.motions = VonNeumannMotion()
        self.start_idx = []
        for i in range(ROOMS - 3):
            self.start_idx.append([[3, 3, i]])
        self.goal_idx = [[WIDTH - 2, WIDTH - 2, ROOMS - 1]]
        self.width = WIDTH
        self.rooms = ROOMS
        self.observation_space = Box(low=0, high=np.max([WIDTH, ROOMS]), shape=np.array(self.start_idx[0]).shape,
                                     dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))
        self.steps = 0

    def step(self, action):
        self.steps += 1
        random_action = np.random.choice([0, 1, 2, 3])
        if np.random.choice([0, 1], p=[0.4, 0.6]) == 0:
            action = random_action
        current_position = self.maze.objects.agent.positions[0]
        # write a new function which takes care of the new functionalities
        new_position = self.get_new_position(current_position, action)

        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        if self._is_goal(new_position):
            reward = +3 - self.steps / 200 + 53.32 / 200
            done = True
            self.steps = 0
        elif not valid and self.steps == 200:
            reward = 1
            done = False
        elif not valid:
            reward = 0
            done = False
        elif self.steps == 200:
            reward = 1
            done = False
        else:
            reward = 0
            done = False

        return np.array(self.maze.objects.agent.positions[0]), reward, done, {}

    def reset(self):
        self.steps = 0
        self.maze.objects.agent.positions = random.sample(self.start_idx, k=1)[0]
        self.maze.objects.goal.positions = self.goal_idx
        return np.array(self.maze.objects.agent.positions[0])

    def get_new_position(self, position, action):
        motion = self.motions[action]

        door_present, door = self._is_door(position)
        new_position = [position[0] + motion[0], position[1] + motion[1], position[2]]
        if door_present:
            # check if the action is same as the action required to pass through the door
            # if yes, select the position of the next door
            for d, dn, a in paired_doors:
                if door[0] == d[0] and door[1] == d[1] and door[2] == d[2] and action == a:
                    next_door = random.sample(dn, k=1)[0]
                    new_position = next_door
        return new_position

    def _is_door(self, position):
        out = False
        door = 0
        for pos in self.maze.objects.door.positions:
            if position[0] == pos[0] and position[1] == pos[1] and position[2] == pos[2]:
                out = True
                door = pos
                break
        return out, door

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0 and position[2] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1] and position[2] < \
                      self.maze.size[2]
        passable = False
        if within_edge:
            passable = not self.maze.to_impassable()[position[0]][position[1]][position[2]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1] and position[2] == pos[2]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()
