import numpy as np
from torch.utils.tensorboard import SummaryWriter
from align_rudder import Transition
import random
import copy
from torch.utils.data import Dataset
import torch
from widis_lstm_tools.nn import LSTMLayer


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


class Environment(Dataset):
    def __init__(self, demos_return, demos_sa_traj, max_traj_length: int, rnd_gen: np.random.RandomState):
        """Our simple 1D environment as PyTorch Dataset"""
        super(Environment, self).__init__()

        state_actions = []
        length = []
        returns = []

        for i in range(len(demos_return)):
            ret = demos_return[i]
            sa_traj = demos_sa_traj[i]
            sa_traj = np.stack(sa_traj)
            sa_traj = np.pad(sa_traj, pad_width=((max_traj_length - sa_traj.shape[0], 0), (0, 0)))
            state_actions.append(sa_traj)
            returns.append(ret)

        self.observations = state_actions
        self.returns = returns

    def __len__(self):
        return len(self.returns)

    def __getitem__(self, idx):
        return self.observations[idx], self.returns[idx]


class Net(torch.nn.Module):
    def __init__(self, n_positions, n_actions, n_lstm):
        super(Net, self).__init__()

        # This will create an LSTM layer where we will feed the concatenate
        self.lstm1 = LSTMLayer(
            in_features=n_positions + n_actions, out_features=n_lstm, inputformat='NLC',
            # cell input: initialize weights to forward inputs with xavier, disable connections to recurrent inputs
            w_ci=(torch.nn.init.xavier_normal_, False),
            # input gate: disable connections to forward inputs, initialize weights to recurrent inputs with xavier
            w_ig=(False, torch.nn.init.xavier_normal_),
            # output gate: disable all connection (=no forget gate) and disable bias
            w_og=False, b_og=False,
            # forget gate: disable all connection (=no forget gate) and disable bias
            w_fg=False, b_fg=False,
            # LSTM output activation is set to identity function
            a_out=lambda x: x
        )

        # After the LSTM layer, we add a fully connected output layer
        self.fc_out = torch.nn.Linear(n_lstm, 1)

    def forward(self, observations):
        # Process input sequence by LSTM
        lstm_out, *_ = self.lstm1(observations,
                                  return_all_seq_pos=True  # return predictions for all sequence positions
                                  )
        net_out = self.fc_out(lstm_out)
        return net_out


class RudderLearn(learn):
    # class for learning algorithms
    def __init__(self, env, eps=0.2, alpha=0.01, gamma=0.99, total_timesteps=10000000, rudder=False,
                 run_path="runs", anneal_eps=0.9999999, eps_lb=0.4, log_every=10, use_demo=True, eval=40,
                 demo_path="dataset/tabular_case/dataset_100.npy", num_demo_use=3,
                 max_episodes=100000, max_reward=1, stop_criteria='good_sequences',
                 init_mean=True, use_new_form=True):
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

        self.use_demo = use_demo
        self.demo_path = demo_path
        self.num_demo_use = num_demo_use

        # load demos
        traj_all = np.load(demo_path, allow_pickle=True)
        sample_index = random.sample(list(np.arange(len(traj_all))), self.num_demo_use)
        self.demos = list(traj_all[sample_index])
        n_actions = self.action_space.n
        self.rooms = self.env.unwrapped.rooms
        demos_return = []
        demos_sa_traj = []
        for traj in self.demos:
            ret_traj = 0
            s_a_traj = []
            for t in traj:
                ret_traj += t.reward
                s_a = self.process_obs(t.state, t.action, n_actions, self.rooms)
                s_a_traj.append(s_a)
            demos_return.append(ret_traj)
            demos_sa_traj.append(s_a_traj)

        random_ret, random_seq = self.get_sequences_random(self.env, self.num_demo_use)

        # balance the data with unsuccessful sequences
        demos_return.extend(random_ret)
        demos_sa_traj.extend(random_seq)

        rnd_gen = np.random.RandomState(seed=123)
        env_data = Environment(demos_return, demos_sa_traj, max_traj_length=200, rnd_gen=rnd_gen)
        env_loader = torch.utils.data.DataLoader(env_data, batch_size=int(self.num_demo_use / 2), num_workers=4)

        # train till average reward from evaluation is 0.9
        self.mean_opt_reward = np.mean(demos_return)
        self.train_till = self.mean_opt_reward * 0.8
        self.width = self.env.unwrapped.width

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

        device = 'cpu'
        n_positions = self.width * 2 + self.rooms
        print(n_positions)
        n_lstms = 16
        net = Net(n_positions=n_positions, n_actions=n_actions, n_lstm=n_lstms)
        _ = net.to(device)

        self.net = self.train_rudder_model(env_loader, net, 1e-3, 1e-5, 1000)

        self.behavior_clone()

    def learn(self):
        training_complete = False
        steps = 0
        num_episodes = 0
        total_return = 0
        _state = self.env.reset()
        trajectory = []
        sa_traj = []
        _ret = 0
        num_good_seq = 0
        while not training_complete:
            action = self.select_action(_state, self.eps)
            state, reward, done, info = self.env.step(action)
            total_return += reward
            _ret += reward

            # Store trajectory till completion
            trajectory.append(Transition(_state, action, reward, state, done))
            sa_obs = self.process_obs(_state, action, self.env.action_space.n, self.rooms)
            sa_traj.append(sa_obs)

            if done:
                self.history = []
                # update q table
                redist_reward = self.redistribute_reward(sa_traj, _ret)
                ret = self.update_trajectory(trajectory, redist_reward)

                trajectory = []
                sa_traj = []

            steps += 1
            if done:
                trajectory = []
                state = self.env.reset()
                num_episodes += 1

                _ret = 0
                if num_episodes % self.log_every == 0:
                    state, mean_return = self.eval(self.eval_epi)
                    self.writer.add_scalar('train/evaluation/return_steps', mean_return, steps)
                    self.writer.add_scalar('train/eps_steps', self.eps, steps)
                    print("Num Episodes: {} Mean Return: {} Num good seq {}".format(num_episodes, mean_return,
                                                                                    num_good_seq))

                    if mean_return >= self.train_till:
                        training_complete = True
                    self.writer.add_scalar('train/evaluation/return', mean_return, num_episodes)
                    self.writer.add_scalar('train/eps', self.eps, num_episodes)

                self.history = []

            _state = state

            if num_episodes > self.max_episodes:
                training_complete = True

        self.writer.add_scalar('final_num_episodes', num_episodes)
        self.writer.add_scalar('final_mean_return', mean_return)
        np.save(self.run_path + '/num_episodes_' + str(num_episodes) + '_' + str(mean_return), num_episodes)

    def eval(self, num_episodes):

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

    def update_trajectory(self, seq, redist_reward=None):
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
        RR = [i for i in redist_reward]
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
            new_reward = 0.2 * reward + 0.8 * original_reward

            self.update(state, action, new_reward, next_state, done)

        return ret[0]

    def redistribute_reward(self, sa_traj, returns):
        with torch.no_grad():
            device = 'cpu'
            sa_traj = np.stack(sa_traj)
            sa_traj = torch.FloatTensor(sa_traj)
            sa_traj = torch.unsqueeze(sa_traj, dim=0)
            predictions = self.net(observations=sa_traj.to(device))[..., 0]

            # Use the differences of predictions as redistributed reward
            redistributed_reward = predictions[:, 1:] - predictions[:, :-1]

            # For the first timestep we will take (0-predictions[:, :1]) as redistributed reward
            redistributed_reward = torch.cat([predictions[:, :1], redistributed_reward], dim=1)

            returns = torch.FloatTensor([returns]).to(device)

            predicted_returns = redistributed_reward.sum(dim=1)
            prediction_error = returns - predicted_returns

            # Distribute correction for prediction error equally over all sequence positions
            redistributed_reward += prediction_error[:, None] / redistributed_reward.shape[1]

            redistributed_reward = redistributed_reward.detach().numpy()

        return np.squeeze(redistributed_reward)

    def select_action(self, state, eps):
        if np.random.rand() > eps:
            q_action = np.argmax(self.q_table[state[0], state[1], state[2]])
            action = q_action
        else:
            action = np.random.choice(self.env.action_space.n)
        return action

    def update(self, _state, action, reward, state, done):
        if done:
            prev_value = self.q_table[_state[0], _state[1], _state[2], action]
            new_value = reward
            self.q_table[_state[0], _state[1], _state[2], action] = (
                                                                            1 - self.alpha) * prev_value + self.alpha * new_value
        else:
            prev_value = self.q_table[_state[0], _state[1], _state[2], action]
            new_value = reward + self.gamma * np.max(self.q_table[state[0], state[1], state[2]])
            self.q_table[_state[0], _state[1], _state[2], action] = (
                                                                            1 - self.alpha) * prev_value + self.alpha * new_value

        self.num_updates += 1

    def process_obs(self, state, action, num_actions, num_rooms):
        a = np.zeros(num_actions + num_rooms + 12)
        a[state[0]] = 1
        a[6 + state[1]] = 1
        a[12 + state[2]] = 1
        a[12 + num_rooms + action] = 1
        a = a / np.linalg.norm(a)

        return a

    def get_sequences_random(self, env, num_seq=2):
        demos_return_ = []
        demos_sa_traj_ = []
        for i in range(num_seq):
            done = False
            s = env.reset()
            sa_traj = []
            ret_traj = 0
            while not done:
                random_action = env.action_space.sample()
                sa_obs = self.process_obs(s, random_action, env.action_space.n, env.unwrapped.rooms)
                sa_traj.append(sa_obs)
                s, reward, done, info = env.step(random_action)
                ret_traj += reward

            demos_return_.append(ret_traj)
            demos_sa_traj_.append(sa_traj)

        return demos_return_, demos_sa_traj_

    def train_rudder_model(self, env_loader, net, lr=1e-3, weight_decay=1e-5, n_updates=1000):

        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        device = 'cpu'
        update = 0
        n_updates = n_updates
        running_loss = 100.
        while update < n_updates:
            for data in env_loader:
                # Get samples
                observations, returns = data
                observations, returns = observations.to(device), returns.to(device)
                observations = observations.type(torch.FloatTensor)
                returns = returns.type(torch.FloatTensor)
                # Reset gradients
                optimizer.zero_grad()

                # Get outputs for network
                outputs = net(observations=observations)

                # Calculate loss, do backward pass, and update
                loss = self.lossfunction(outputs[..., 0], returns)
                loss.backward()
                running_loss = running_loss * 0.99 + loss * 0.01
                optimizer.step()
                update += 1

        return net

    def lossfunction(self, predictions, returns):
        # Main task: predicting return at last timestep
        main_loss = torch.mean(predictions[:, -1] - returns) ** 2
        # Auxiliary task: predicting final return at every timestep ([..., None] is for correct broadcasting)
        aux_loss = torch.mean(predictions[:, :] - returns[..., None]) ** 2
        # Combine losses
        loss = main_loss + aux_loss * 0.5
        return loss

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

                self.q_table[state[0], state[1], state[2], action] = self.q_table[
                                                                         state[0], state[1], state[2], action] + _ret
                temp_q_table[state[0], state[1], state[2], action] = temp_q_table[
                                                                         state[0], state[1], state[2], action] + 1

        for i in range(self.width):
            for j in range(self.width):
                for k in range(self.rooms):
                    for l in range(self.action_space.n):
                        if temp_q_table[i, j, k, l] != 0:
                            self.q_table[i, j, k, l] = self.q_table[i, j, k, l] - original_q_table[i, j, k, l]
                            self.q_table[i, j, k, l] = self.q_table[i, j, k, l] / temp_q_table[i, j, k, l]
                            self.q_table[i, j, k, l] = self.q_table[i, j, k, l] * (
                                    temp_q_table[i, j, k, l] / np.sum(temp_q_table[i, j, k, :]))
