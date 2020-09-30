import numpy as np
from torch.utils.tensorboard import SummaryWriter
from align_rudder import Transition
from align_rudder.representation.successor import SuccessorRepresentation
from align_rudder.alignment.rudder_alignment import RudderAlign
import random
import copy


class learn():
    # class for learning algorithms
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

    def learn(self):
        return NotImplementedError

    def select_action(self, state, eps):
        return NotImplementedError

    def update(self, _state, action, reward, state, done):
        return NotImplementedError


class Qlearning(learn):
    def __init__(self, env, eps=0.1, alpha=0.1, gamma=0.99, total_timesteps=10000, rudder=False,
                 num_store_seq=30, enough_seq=2, num_clusters=5, top_n=12, consensus_type="all",
                 consensus_thresh=0.9, cluster_type="default", run_path="runs", anneal_eps=0.9999999,
                 eps_lb=0.4, rr_thresh=0.005, log_every=10000, normalise_rr_by_max=True, normalisation_scale=2,
                 use_succ=True, use_demo=True, eval=20, demo_path="dataset/tabular_case/dataset_100.npy", num_demo_use=3,
                 max_episodes=1000, max_reward=1, update_alignment=False, mode='log', stop_criteria='good_sequences',
                 seed=0, init_mean=True, use_new_form=True):
        super().__init__(env.observation_space, env.action_space)
        self.env = env
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.total_timesteps = total_timesteps
        self.max_episodes = max_episodes
        self.num_updates = 0
        self.run_path = run_path
        self.eval_epi = eval
        self.max_reward = max_reward
        self.initialization = 7.5
        self.writer = SummaryWriter(log_dir=run_path)
        self.use_new_form = use_new_form

        self.anneal_eps = anneal_eps
        self.eps_lb = eps_lb
        self.log_every = log_every
        # Rudder hyperparameters
        self.rudder = rudder
        self.stop_criteria = stop_criteria

        self.update_alignment = update_alignment

        self.sr = SuccessorRepresentation(self.env, rep_lr=0.1, rep_gamma=0.99)
        self.use_succ = use_succ
        if self.rudder:
            self.rudder_model = RudderAlign(env=self.env, state_space=self.state_space, num_store_seq=num_store_seq,
                                            succ_r=self.sr, enough_seq=enough_seq, num_clusters=num_clusters,
                                            top_n=top_n,
                                            consensus_type=consensus_type, consensus_thresh=consensus_thresh,
                                            cluster_type=cluster_type, run_path=run_path, rr_thresh=rr_thresh,
                                            normalise_rr_by_max=normalise_rr_by_max,
                                            normalisation_scale=normalisation_scale,
                                            mode=mode)

            self.num_episode_updates = 0

        self.use_demo = use_demo
        self.demo_path = demo_path
        self.num_demo_use = num_demo_use

        if self.use_demo:
            # load demos
            traj_all = np.load(demo_path, allow_pickle=True)
            sample_index = random.sample(list(np.arange(len(traj_all))), self.num_demo_use)
            self.demos = list(traj_all[sample_index])

            demos_return = []
            for traj in self.demos:
                ret_traj = 0
                for t in traj:
                    ret_traj += t.reward
                demos_return.append(ret_traj)

            # train till average reward from evaluation is 0.9
            self.mean_opt_reward = np.mean(demos_return)
            self.train_till = self.mean_opt_reward * 0.8
            self.width = self.env.unwrapped.width
            self.rooms = self.env.unwrapped.rooms
            # Initialization based on the demos

            if init_mean == True:
                self.q_table = np.random.normal(size=(self.width, self.width,
                                                      self.rooms,
                                                      self.action_space.n), loc=self.mean_opt_reward,
                                                scale=np.std(demos_return))
            else:
                self.q_table = np.random.normal(size=(self.width, self.width,
                                                      self.rooms,
                                                      self.action_space.n), loc=1,
                                                scale=0.5)

            # update sr representation from demos
            if self.use_succ:
                for i in range(10):
                    for traj in self.demos:
                        for trans in traj:
                            state = trans.state
                            action = trans.action
                            reward = trans.reward
                            next_state = trans.next_state
                            done = trans.done

                            self.sr.update(state, action, reward, next_state, False)
                            if i == 0 and self.rudder:
                                self.rudder_model.update_visitation_frequency(state, next_state, done)
                            if done:
                                self.sr.update(next_state, action, reward, next_state, done)

            if self.rudder:
                self.rudder_model.demos = self.demos
                cluster_model = self.rudder_model.cluster_states()

                self.max_clusters = np.max(cluster_model) + 1

                # Cluster sequences and align them
                clus_demos = self.rudder_model.assign_cluster_demo(self.rudder_model.cluster_model, self.demos, False)

                alignment, score_file = self.rudder_model.align(clus_demos)
                self.history = []
                # update q table using behavior cloning
                self.behavior_clone()
                self.temp_q_table = copy.deepcopy(self.q_table)
                for i in range(self.max_clusters):
                    self.q_table = np.repeat(np.expand_dims(self.q_table, axis=3), 2, axis=3)
            else:
                self.behavior_clone()

            self.plot()

    def learn(self):
        training_complete = False
        steps = 0
        num_episodes = 0
        total_return = 0
        _state = self.env.reset()
        trajectory = []
        _ret = 0
        num_good_seq = 0
        while not training_complete:
            action = self.select_action(_state, self.eps)
            state, reward, done, info = self.env.step(action)
            total_return += reward
            _ret += reward
            # update representation
            self.sr.update(_state, action, reward, state, False)
            if done:
                self.sr.update(state, action, reward, state, done)

            if self.rudder:
                # Store trajectory till completion
                trajectory.append(Transition(_state, action, reward, state, done))
                # update visitation frequency
                self.rudder_model.update_visitation_frequency(_state, state, done)

                # align with the profile
                if done:
                    self.history = []
                    # update q table
                    redist_reward = self.rudder_model.redistribute_reward(trajectory)
                    ret = self.update_trajectory(trajectory, redist_reward, True)

                    if _ret > 1:
                        self.rudder_model.demos.append(copy.deepcopy(trajectory))
                        # realign demos
                        if self.update_alignment:
                            # Cluster sequences and align them
                            clus_demos = self.rudder_model.assign_cluster_demo(self.rudder_model.cluster_model,
                                                                               self.rudder_model.demos,
                                                                               False)
                            alignment, score_file = self.rudder_model.align(clus_demos)

                    trajectory = []
            else:
                trajectory.append(Transition(_state, action, reward, state, done))
                # Normal Q learning update
                self.update(_state, action, reward, state, done)

            steps += 1
            if done:
                trajectory = []
                state = self.env.reset()
                num_episodes += 1
                if _ret > self.train_till:
                    num_good_seq += 1

                _ret = 0
                if num_episodes % self.log_every == 0:
                    state, mean_return = self.eval(self.eval_epi)
                    self.writer.add_scalar('train/evaluation/return_steps', mean_return, steps)
                    self.writer.add_scalar('train/eps_steps', self.eps, steps)
                    print("Num Episodes: {} Mean Return: {} Num good seq {}".format(num_episodes, mean_return,
                                                                                    num_good_seq))

                    if self.stop_criteria == 'good_sequences':
                        if num_good_seq >= 100:
                            training_complete = True
                    else:
                        if mean_return >= self.train_till:
                            training_complete = True
                    self.writer.add_scalar('train/evaluation/return', mean_return, num_episodes)
                    self.writer.add_scalar('train/eps', self.eps, num_episodes)

                self.history = []

            _state = state

            if num_episodes > self.max_episodes:
                training_complete = True

        self.plot()
        self.writer.add_scalar('final_num_episodes', num_episodes)
        self.writer.add_scalar('final_mean_return', mean_return)
        np.save(self.run_path + '/num_episodes_' + str(num_episodes) + '_' + str(mean_return), num_episodes)

    def eval(self, num_episodes):

        if self.rudder:
            state_ = self.env.reset()
            total_return_ = 0
            for _ in range(num_episodes):
                done_ = False
                self.history = []
                while not done_:
                    action_ = self.select_action(state_, 0.1)
                    state_, reward_, done_, info_ = self.env.step(action_)
                    total_return_ += reward_

                    if done_:
                        state_ = self.env.reset()
        else:

            state_ = self.env.reset()
            total_return_ = 0
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

    def update(self, _state, action, reward, state, done):
        if done:
            prev_value = self.q_table[_state[0], _state[1], _state[2], action]
            new_value = reward
            self.q_table[_state[0], _state[1], _state[2], action] = (1 - self.alpha) * prev_value + self.alpha * new_value
        else:
            prev_value = self.q_table[_state[0], _state[1], _state[2], action]
            new_value = reward + self.gamma * np.max(self.q_table[state[0], state[1], state[2]])
            self.q_table[_state[0], _state[1], _state[2], action] = (1 - self.alpha) * prev_value + self.alpha * new_value

        self.num_updates += 1

    def update_index(self, index_state_action, reward, index_next_state_q_val, done, rr_kappa_):
        if done:
            prev_value = self.q_table[tuple(index_state_action)]
            new_value = rr_kappa_
            self.q_table[tuple(index_state_action)] = (1 - self.alpha) * prev_value + self.alpha * new_value
        else:
            prev_value = self.q_table[tuple(index_state_action)]
            new_value = rr_kappa_
            self.q_table[tuple(index_state_action)] = (1 - self.alpha) * prev_value + self.alpha * new_value

        self.num_updates += 1

    def update_trajectory(self, seq, redist_reward=None, use_rudder=False):
        if use_rudder and self.use_new_form:
            ret = [0 for i in range(len(seq))]
            rr_kappa = [0 for i in range(len(seq))]
            for i, trans in enumerate(seq):
                reward = trans.reward
                ret[i] = reward
            for i in range(len(ret)):
                ret[i] = np.sum(ret[i:])
            G_0 = ret[0]
            history = []
            # Compute reward redistribution
            RR = [i * self.mean_opt_reward for i in redist_reward]
            # compute kappa
            N = 20
            for i in range(len(RR)):
                rr_kappa[i] = np.sum(RR[i:i + N])

            lenght_ = [i for i in range(len(seq))]

            for reward, trans, ret_, rr_kappa_, l in zip(redist_reward, seq, ret, rr_kappa, lenght_):
                state = trans.state
                next_state = trans.next_state
                done = trans.done
                original_reward = trans.reward
                action = trans.action
                new_reward = self.mean_opt_reward * reward
                index_state_action, history = self.return_index(state, history, action)
                index_next_state_q_val, history = self.return_index(next_state, history, None)
                self.update_index(index_state_action, new_reward, index_next_state_q_val, done, rr_kappa_)
                if l == lenght_:
                    # Correct reward redistribution
                    correction = G_0 - np.sum(RR)
                    self.update_index(index_state_action, correction, index_next_state_q_val, done, correction)

            return ret[0]
        elif use_rudder:
            ret = [0 for i in range(len(seq))]
            rr_kappa = [0 for i in range(len(seq))]
            for i, trans in enumerate(seq):
                reward = trans.reward
                ret[i] = reward
            for i in range(len(ret)):
                ret[i] = np.sum(ret[i:])
            G_0 = ret[0]
            history = []
            # Compute reward redistribution
            RR = [i * G_0 for i in redist_reward]
            # compute kappa
            N = 20
            for i in range(len(RR)):
                rr_kappa[i] = np.sum(RR[i:i + N])

            for reward, trans, ret_, rr_kappa_ in zip(redist_reward, seq, ret, rr_kappa):
                state = trans.state
                next_state = trans.next_state
                done = trans.done
                original_reward = trans.reward
                action = trans.action
                new_reward = G_0 * reward
                index_state_action, history = self.return_index(state, history, action)
                index_next_state_q_val, history = self.return_index(next_state, history, None)
                self.update_index(index_state_action, new_reward, index_next_state_q_val, done, rr_kappa_)

            return ret[0]
        else:
            ret = 0
            history = []
            for trans in seq:
                state = trans.state
                next_state = trans.next_state
                done = trans.done
                reward = trans.reward
                action = trans.action
                ret += reward
                index_state_action, history = self.return_index(state, history, action)
                index_next_state_q_val, history = self.return_index(state, history, None)
                self.update_index(index_state_action, reward, index_next_state_q_val, done)

            return ret

    def behavior_clone(self):
        temp_q_table = np.zeros([self.width, self.width,
                                self.rooms, self.action_space.n], dtype=float)
        original_q_table = copy.deepcopy(self.q_table)

        returns = []
        for traj in self.demos:
            ret = []
            for trans in traj:
                reward = trans.reward
                ret.append(reward)
            for i in range(len(ret)):
                ret[i] = np.sum(ret[i:])

            returns.append(copy.deepcopy(ret))

        for traj, ret in zip(self.demos, returns):
            for trans, _ret in zip(traj, ret):
                state = trans.state
                action = trans.action
                next_state = trans.next_state

                self.q_table[state[0], state[1], state[2], action] = self.q_table[state[0], state[1], state[2], action] + _ret
                temp_q_table[state[0], state[1], state[2], action] = temp_q_table[state[0], state[1], state[2], action] + 1

        for i in range(self.width):
            for j in range(self.width):
                for k in range(self.rooms):
                    for l in range(self.action_space.n):
                        if temp_q_table[i, j, k, l] != 0:
                            self.q_table[i, j, k, l] = self.q_table[i, j, k, l] - original_q_table[i, j, k, l]
                            self.q_table[i, j, k, l] = self.q_table[i, j, k, l] / temp_q_table[i, j, k, l]
                            self.q_table[i, j, k, l] = self.q_table[i, j, k, l] * (
                                    temp_q_table[i, j, k, l] / np.sum(temp_q_table[i, j, k, :]))

    def select_action(self, state, eps):
        if self.rudder:

            if np.random.rand() > eps:
                index, self.history = self.return_index(state, self.history, None)
                q_action = np.argmax(self.q_table[tuple(index)])
                action = q_action
            else:
                action = np.random.choice(self.env.action_space.n)
        else:
            if np.random.rand() > eps:
                q_action = np.argmax(self.q_table[state[0], state[1], state[2]])
                action = q_action
            else:
                action = np.random.choice(self.env.action_space.n)
        return action

    def return_index(self, state, history, action):
        if action != None:
            cluster = self.rudder_model.cluster_model[state[0], state[1], state[2]]
            if cluster not in history:
                history.append(cluster)
            index = [state[0], state[1], state[2]]
            hist_index = [0 for i in range(self.max_clusters)]
            for i in history:
                hist_index[i] = 1
            index.extend(hist_index)
            index.append(action)
        else:
            cluster = self.rudder_model.cluster_model[state[0], state[1], state[2]]
            if cluster not in history:
                history.append(cluster)
            index = [state[0], state[1], state[2]]
            hist_index = [0 for i in range(self.max_clusters)]
            for i in history:
                hist_index[i] = 1
            index.extend(hist_index)

        return index, history

    def get_optimal_policy(self):
        maze = self.env.unwrapped.maze.to_value()
        maze[maze == 1] = -1

        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                for k in range(maze.shape[2]):
                    if maze[i, j, k] != -1:
                        maze[i, j, k] = np.argmax(self.q_table[i, j, k])

        return maze

    def save_optimal_policy(self):
        maze = self.get_optimal_policy()
        np.save(self.run_path + '/optimal_policy_' + str(self.num_updates), maze)
        np.save(self.run_path + '/optimal_policy_' + str(self.num_updates), self.q_table)

        # save sequence alignment consensus
        if self.rudder:
            np.save(self.run_path + '/optimal_policy_' + str(self.num_updates), self.rudder_model.redist_reward)
            np.save(self.run_path + '/redist_reward_count_' + str(self.num_updates), self.rudder_model.env_maze)

    def plot(self):
        pass
