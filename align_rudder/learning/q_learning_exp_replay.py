import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random
from collections import namedtuple
from align_rudder import Transition, TransitionNstep


class ReplayBuffer():
    def __init__(self, memory_len=20000, demo_transitions=140):
        self.memory_len = memory_len
        self.transition = []
        self.prob_alpha = 0.4
        self.demos_per_const = 0.001
        self.exp_per_const = 1.0
        self.priorities = np.zeros((self.memory_len,), dtype=np.float32)
        self.remove_pos = demo_transitions + 1
        self.pos = 0

    def add_demo_transition(self, transition, td_error):
        self.transition.append(transition)
        self.priorities[self.pos] = td_error
        self.pos = (self.pos + 1)

    def add(self, transition):
        if self.length() < self.memory_len:
            self.transition.append(transition)
        else:
            self.transition[self.pos] = transition
        max_prio = self.priorities.max() if self.transition else 1.0
        self.priorities[self.pos] = max_prio
        if self.pos + 1 % self.memory_len == 0:
            self.pos = (self.pos + 1) % self.memory_len + self.remove_pos
        else:
            self.pos = (self.pos + 1) % self.memory_len

    def sample(self, batch_size, beta=0.6):
        if self.length() == self.memory_len:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(self.length(), batch_size, p=probs)
        samples = [self.transition[idx] for idx in indices]

        total = self.length()
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def length(self):
        return len(self.transition)


class learn():
    # class for learning algorithms
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

    def learn(self):
        return NotImplementedError

    def select_action(self, state, eps):
        return NotImplementedError

    def update(self, _state, action, reward, state, done, nstep_ret, expert_loss, weight):
        return NotImplementedError


class QlearningExpReplay(learn):
    def __init__(self, env, eps=0.1, alpha=0.1, gamma=0.99, total_timesteps=10000, rudder=False,
                 num_store_seq=30, enough_seq=2, num_clusters=5, top_n=12, consensus_type="all",
                 consensus_thresh=0.9, cluster_type="default", run_path="runs", anneal_eps=0.9999999,
                 eps_lb=0.2, rr_thresh=0.005, log_every=10000, normalise_rr_by_max=True, normalisation_scale=2,
                 use_succ=True, use_demo=True, eval=20, demo_path="dataset/tabular_case/dataset_100.npy",
                 num_demo_use=3,
                 max_episodes=100, max_reward=1, update_alignment=True, memory_len=5000, margin=0.1, nstep=10,
                 pre_training_iterations=500, batch=5, a1=1, a2=1, a3=1, a4=1, seed=0, init_mean=True):
        super().__init__(env.observation_space, env.action_space)

        self.env = env
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.total_timesteps = total_timesteps
        self.max_episodes = max_episodes
        self.eval_epi = eval
        self.num_updates = 0
        self.run_path = run_path
        self.max_reward = max_reward
        self.history = []

        #### loss hyper param
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4

        self.writer = SummaryWriter(log_dir=run_path)
        self.width = self.env.unwrapped.width
        self.rooms = self.env.unwrapped.rooms
        self.q_table = np.zeros([self.width, self.width,
                                 self.rooms, self.action_space.n], dtype=float)
        self.target_q_table = np.zeros([self.width, self.width,
                                        self.rooms, self.action_space.n],
                                       dtype=float)
        self.anneal_eps = 1  # anneal_eps
        self.eps_lb = eps_lb
        self.log_every = log_every
        # Rudder hyperparameters
        self.rudder = rudder
        self.update_alignment = update_alignment

        self.use_demo = use_demo
        self.demo_path = demo_path
        self.num_demo_use = num_demo_use
        self.memory_len = memory_len

        self.buffer = ReplayBuffer(memory_len=memory_len)
        self.margin = margin
        self.nstep = nstep
        self.pre_training_iterations = pre_training_iterations
        self.batch = batch

        self.optimal_return = 3
        if self.use_demo:
            # load demos
            traj_all = np.load(demo_path, allow_pickle=True)
            sample_index = random.sample(list(np.arange(len(traj_all))), self.num_demo_use)
            self.demos = list(traj_all[sample_index])

            # Behavior Clone
            demos_return = []
            mean_return = 0
            for traj in self.demos:
                mean_return = 0
                for trans in traj:
                    mean_return += trans.reward
                demos_return.append(mean_return)

            self.mean_opt_reward = np.mean(np.array(demos_return))
            self.train_till = self.mean_opt_reward * 0.8

            self.std_opt_reward = np.std(demos_return)
            print("Mean Return Demos:", mean_return)
            if init_mean == True:
                self.q_table = np.random.normal(size=(self.width, self.width,
                                                      self.rooms,
                                                      self.action_space.n), loc=self.mean_opt_reward,
                                                scale=self.std_opt_reward)
            else:
                self.q_table = np.random.normal(size=(self.width, self.width,
                                                      self.rooms,
                                                      self.action_space.n), loc=1,
                                                scale=0.5)
            # self.q_table[20, 20, :] = 0

        self.pre_training()

    def get_optimal_return(self):
        return self.eval(100, optimal=1)

    def initialize_q_table(self):
        # Initialization based on the demos
        self.q_table = np.random.normal(size=(self.width, self.width,
                                              self.rooms,
                                              self.action_space.n), loc=self.mean_opt_reward,
                                        scale=self.std_opt_reward)

    def learn(self):
        training_complete = False
        steps = 0
        num_episodes = 0
        total_return = 0
        _state = self.env.reset()
        trajectory = []
        epsteps = 0
        nstep_list = []
        n_good_seq = 0
        timesteps = 0
        self.a1 = 1
        self.a2 = 1
        self.a3 = 0.01
        while not training_complete:

            action = self.select_action(_state, self.eps)
            state, reward, done, info = self.env.step(action)
            total_return += reward
            epsteps += 1
            nstep_list.append(Transition(_state, action, reward, state, done))
            # Sample and update
            if num_episodes > 10:
                self.sample_update(batch=self.batch)

            if epsteps >= self.nstep:
                nstep_return = 0
                for trans in nstep_list[:-1]:
                    nstep_return += trans.reward
                self.buffer.add(TransitionNstep(nstep_list[0].state, nstep_list[0].action, nstep_list[0].reward,
                                                nstep_list[0].next_state, nstep_list[0].done, nstep_return,
                                                nstep_list[-1].state))
                # remove the first transition
                nstep_list.pop(0)

            steps += 1
            timesteps += 1
            if done:
                timesteps = 0

                state = self.env.reset()
                nstep_return = [0 for i in range(len(nstep_list))]
                for i, trans in enumerate(nstep_list):
                    nstep_return[i] = trans.reward
                for i in range(len(nstep_return)):
                    nstep_return[i] = np.sum(nstep_return[i:])

                for trans, nstep_ret in zip(nstep_list, nstep_return):
                    self.buffer.add(TransitionNstep(trans.state, trans.action, trans.reward, trans.next_state,
                                                    trans.done, nstep_ret, self.env.unwrapped.goal_idx[0]))

                nstep_list = []
                epsteps = 0
                num_episodes += 1
                if total_return > self.train_till:
                    n_good_seq += 1

                total_return = 0
                if num_episodes % self.log_every == 0:
                    state, mean_return = self.eval(num_episodes=self.eval_epi)
                    self.writer.add_scalar('train/evaluation/return_steps', mean_return, steps)
                    self.writer.add_scalar('train/eps_steps', self.eps, steps)
                    print("Num Episodes: {} Mean return {} Good sequences {}".format(num_episodes, mean_return,
                                                                                     n_good_seq))
                    # print("Mean Return:", mean_return)
                    if mean_return >= self.train_till:
                        training_complete = True
                    self.writer.add_scalar('train/evaluation/return', mean_return, num_episodes)
                    self.writer.add_scalar('train/eps', self.eps, num_episodes)

                    if num_episodes % 100 == 0:
                        self.plot()

            _state = state

            if num_episodes > self.max_episodes:
                training_complete = True

        self.plot()
        print(num_episodes, total_return)
        self.writer.add_scalar('final_num_episodes', num_episodes)
        self.writer.add_scalar('final_mean_return', mean_return)
        np.save(self.run_path + '/num_episodes_' + str(num_episodes) + '_' + str(mean_return), num_episodes)

    def eval(self, num_episodes=40, optimal=0):
        state_ = self.env.reset()
        total_return_ = 0

        if optimal:
            for _ in range(num_episodes):
                done_ = False
                while not done_:
                    action_ = np.argmax(self.env.optimal_policy[state_[0], state_[1], state_[2]])
                    state_, reward_, done_, info_ = self.env.step(action_)
                    total_return_ += reward_

                    if done_:
                        state_ = self.env.reset()
        else:
            for _ in range(num_episodes):
                done_ = False
                while not done_:
                    action_ = self.select_action(state_, 0.1)
                    state_, reward_, done_, info_ = self.env.step(action_)
                    total_return_ += reward_

                    if done_:
                        state_ = self.env.reset()

        state = self.env.reset()
        mean_return = total_return_ / num_episodes

        return state, mean_return

    def sample_update(self, batch):
        samples, indices, weights = self.buffer.sample(batch)
        demos = []
        for i in indices:
            if i < self.buffer.remove_pos:
                demos.append(True)
            else:
                demos.append(False)
        td_error = []
        self.target_q_table = self.q_table.copy()
        for transition, demo, weight in zip(samples, demos, weights):
            if demo:
                state = transition.state
                action = transition.action
                reward = transition.reward
                next_state = transition.next_state
                done = transition.done
                nstep_return = transition.nstep_return
                nstep_state = transition.nstep_state
                nstep_return = nstep_return + np.max(self.q_table[nstep_state[0], nstep_state[1], nstep_state[2]])
                expert_loss = np.max(self.q_table[state[0], state[1], state[2]] + self.margin_loss(action)) - \
                              self.q_table[
                                  state[0], state[1], state[2], action]
                error = self.update(state, action, reward, next_state, done, nstep_return, expert_loss, weight)
                td_error.append(error)
            else:
                state = transition.state
                action = transition.action
                reward = transition.reward
                next_state = transition.next_state
                done = transition.done
                nstep_return = transition.nstep_return
                nstep_state = transition.nstep_state
                nstep_return = nstep_return + np.max(self.q_table[nstep_state[0], nstep_state[1], nstep_state[2]])
                error = self.update(state, action, reward, next_state, done, nstep_return, 0, weight)
                td_error.append(error)

        self.buffer.update_priorities(indices, td_error)

    def update(self, _state, action, reward, state, done, nstep_ret, expert_loss, weight=1):
        if done:
            prev_value = self.q_table[_state[0], _state[1], _state[2], action]
            new_value = reward
            j1 = new_value - prev_value
            j3 = expert_loss
            loss = np.mean(self.a1 * j1 + self.a3 * j3)
            self.q_table[_state[0], _state[1], _state[2], action] = prev_value + self.alpha * weight * loss
            return abs(loss)
        else:
            prev_value = self.q_table[_state[0], _state[1], _state[2], action]
            new_value = reward + self.gamma * np.max(self.target_q_table[state[0], state[1], state[2]])
            j1 = new_value - prev_value
            j2 = nstep_ret - prev_value
            j4 = np.sqrt(prev_value)
            j3 = expert_loss
            loss = np.mean(self.a1 * j1 + self.a2 * j2 + self.a3 * j3)
            self.q_table[_state[0], _state[1], _state[2], action] = prev_value + self.alpha * weight * loss

            return abs(loss)

    def pre_training(self):
        # Compute the nstep return
        nstep_returns = []
        nstep_state = []
        for traj in self.demos:
            ret = [0 for i in range(len(traj))]
            _nstep_state = []
            for i, trans in enumerate(traj):
                reward = trans.reward
                ret[i] = reward
            # we use gamma = 1, so no discounting for nstep return
            for i, trans in enumerate(traj):
                if i + self.nstep >= len(traj):
                    n_state = self.env.unwrapped.goal_idx[0]
                else:
                    n_state = traj[i + self.nstep].state
                ret[i] = np.sum(ret[i:i + self.nstep - 1])
                _nstep_state.append(n_state)
            nstep_returns.append(ret)
            nstep_state.append(_nstep_state)

        # go over all transitions and update q table
        demo_trans = 0
        track_loss = 0
        loss_ = []
        for i in range(self.pre_training_iterations):
            self.target_q_table = self.q_table.copy()
            for traj, nstep_ret, nth_state in zip(self.demos, nstep_returns, nstep_state):
                for trans, _nstep_ret, _nstep_state in zip(traj, nstep_ret, nth_state):
                    state = trans.state
                    next_state = trans.next_state
                    done = trans.done
                    reward = trans.reward
                    action = trans.action
                    expert_loss = np.max(self.q_table[state[0], state[1], state[2]] + self.margin_loss(action)) - \
                                  self.q_table[
                                      state[0], state[1], state[2], action]
                    _nstep_ret_ = _nstep_ret + np.max(
                        self.target_q_table[_nstep_state[0], _nstep_state[1], _nstep_state[2]])
                    loss = self.update(state, action, reward, next_state, done, _nstep_ret_, expert_loss)
                    if i == self.pre_training_iterations - 1:
                        self.buffer.add_demo_transition(
                            TransitionNstep(state, action, reward, next_state, done, _nstep_ret, _nstep_state), loss)
                        demo_trans += 1

        self.buffer.remove_pos = demo_trans + 1

    def margin_loss(self, action):
        margin_loss = np.zeros(self.env.action_space.n)
        margin_loss[:] = self.margin
        margin_loss[action] = 0
        return margin_loss

    def select_action(self, state, eps):

        if self.rudder:
            if np.random.rand() > eps:
                index, self.history = self.return_index(state, self.history, None)
                q_action = np.argmax(self.q_table[tuple(index)])
                action = q_action
            else:
                action = np.random.choice(self.env.valid_actions(state))
        else:
            if np.random.rand() > eps:
                q_action = np.argmax(self.q_table[state[0], state[1], state[2]])
                action = q_action
            else:
                action = np.random.choice(self.env.action_space.n)

        return action

    def return_index(self, state, history, action):
        history = 0
        if action != None:
            index = [state[0], state[1], state[2]]
            index.append(action)
        else:
            index = [state[0], state[1], state[2]]

        return index, history

    def get_optimal_policy(self):
        maze = self.env.unwrapped.maze.to_value()
        maze = maze.astype(str)
        maze[maze == '1'] = '#'
        maze[maze == '3'] = '$'

        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                for k in range(maze.shape[2]):
                    if maze[i, j, k] != '#':
                        if maze[i, j, k] != '$':
                            maze[i, j, k] = np.argmax(self.q_table[i, j, k])

        return maze

    def save_optimal_policy(self):
        maze = self.get_optimal_policy()
        np.save(self.run_path + '/optimal_policy_' + str(self.num_updates), maze)
        np.save(self.run_path + '/optimal_policy_' + str(self.num_updates), self.q_table)

    def plot(self):
        maze = self.get_optimal_policy()

        # print(maze)
        # print(self.eps)

        # self.save_optimal_policy()

    def plot_visitation_demos(self):

        visit_table = np.zeros_like(self.q_table)
        for traj in self.demos:
            for trans in traj:
                state = trans.state
                action = trans.action
                visit_table[state[0], state[1], state[2], action] += 1

    def plot_policy(self):
        walls = self.env.maze.objects.obstacle.positions
        policy = self.q_table.copy()
        plot_policy = np.argmax(policy, axis=-1)
        for wall in walls:
            plot_policy[wall[0], wall[1]] = -1
