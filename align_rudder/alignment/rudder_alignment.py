import numpy as np
from sklearn.cluster import AffinityPropagation, SpectralClustering
from alignment.align import aa_dict, create_fasta_sequences, create_scoring_matrix
from align_rudder.alignment.align import get_alignment, get_pssm
import copy
from Bio.Align.Applications import ClustalwCommandline
from collections import OrderedDict


class AbstractRudderAlign:
    # Base class for reward redistribution using alignment
    def store_trajectory(self, state_traj, r_return, reward_traj, action_traj, done_traj):
        # take a trajectory and store
        # If return is greater > constant then store
        # if num_stored_sequences > constant then see if seq length is small, then store
        raise NotImplementedError

    def generate_sequences(self, num_seq, iter):
        # if return
        raise NotImplementedError

    def cluster_states(self):
        # returns cluster centers, and reduced sequences
        raise NotImplementedError

    def align(self, sequences):
        raise NotImplementedError

    def redistribute_reward(self, new_sequence):
        raise NotImplementedError


class RudderAlign(AbstractRudderAlign):
    def __init__(self, env, state_space, num_store_seq, succ_r, enough_seq=2, num_clusters=5, top_n=12,
                 consensus_type="all",
                 consensus_thresh=0.9, cluster_type="cluster_type", run_path="run_path", rr_thresh=0.005,
                 normalise_rr_by_max=True, normalisation_scale=2, mode='log'):
        super().__init__()

        self.max_store_seq = num_store_seq
        self.max_return = 1
        self.sequences = []
        self.sequences_for_k_means = []
        self.state_trajectory = []
        self.reward_trajectory = []
        self.action_trajectory = []
        self.done_trajectory = []
        self.r_return = 0
        self.redistribute = False
        self.enough = enough_seq
        self.num_store_seq = 0
        self.num_clusters = num_clusters
        self.env = env
        self.top_n = top_n
        self.consensus_type = consensus_type
        self.consensus_thresh = consensus_thresh
        self.num_succ_redist_seq = 0
        self.num_succ_seq_used = 0
        self.cluster_type = cluster_type
        self.visitation_frequencies = [0 for i in range(20)]
        self.num_sequences = 0
        self.rr_thresh = rr_thresh
        self.consensus = []
        self.redist_reward = []
        self.aligned = False
        self.run_path = run_path
        self.state_space = state_space
        self.width = self.env.unwrapped.width
        self.rooms = self.env.unwrapped.rooms
        self.env_maze = np.zeros([self.width, self.width,
                                  self.rooms])
        self.env_maze_visitation = np.zeros([self.width, self.width,
                                             self.rooms])
        self.cluster = True
        self.normalise_rr_by_max = normalise_rr_by_max
        self.normalisation_scale = normalisation_scale
        self.sr = succ_r
        self.cluster_maze = []
        self.demos = []
        self.score_file = ""
        self.alignment_file = ""
        self.mode = mode

    def align(self, sequences):
        self.update_vf_cluster(self.cluster_model)
        exclude = None
        top_n_seq, all_seq, seq_file = create_fasta_sequences(sequences, outdir=self.run_path, top_n=self.top_n,
                                                              exclude=exclude)

        scoring_matrix, score_file = create_scoring_matrix(outdir=self.run_path,
                                                           outfile="scoring_matrix",
                                                           fasta_sequences=all_seq,
                                                           offdiag=-1.0,
                                                           main_diag_factor=0.1,
                                                           scaling="log")

        alignment = get_alignment(seq_file, self.top_n, score_file, self.run_path, "demo_profile")
        self.score_file = score_file
        self.aligment_file = "msa_demo_profile.aln"

        return alignment, score_file

    def redistribute_reward(self, new_sequence):
        self.update_vf_cluster(self.cluster_model)
        exclude = None

        clus_demos_database = self.assign_cluster_demo(self.cluster_model, [new_sequence], False)

        _, _, seq_file_2 = create_fasta_sequences(clus_demos_database, outdir=self.run_path,
                                                  top_n=5, exclude=exclude)

        with open(seq_file_2, "r") as f:
            lines = f.readlines()

        with open(self.run_path + "/" + "query.fasta", "w") as f:
            for line in lines:
                cleaned_line = line.replace(",", "")
                cleaned_line = cleaned_line.replace("0", str(len(self.demos)))
                cleaned_line = cleaned_line.replace("[", "")
                cleaned_line = cleaned_line.replace("]", "")
                cleaned_line = cleaned_line.replace("'", "")
                cleaned_line = cleaned_line.replace(" ", "")
                f.write(cleaned_line)
        inprofile = self.run_path + "/" + self.aligment_file
        query_seq = self.run_path + "/" + "query.fasta"
        score_file = self.score_file
        outfile = self.run_path + "/" + "output"
        gap_open = 0
        gap_ext = 0
        clustalw_cline = ClustalwCommandline("clustalw2", profile1=inprofile, profile2=query_seq,
                                             outfile=outfile, gapopen=gap_open, gapext=gap_ext,
                                             pwmatrix=score_file, matrix=score_file, pwgapopen=gap_open,
                                             pwgapext=gap_ext, case="UPPER", type="protein", output="gde")
        stdout, stderr = clustalw_cline()

        # redistribute reward according to the pssm
        # get the aligned sequence from the alignment
        with open(outfile) as f:
            lines = f.readlines()

        copy_lines = False
        traj_test_aligned = ""
        for line in lines:
            if copy_lines:
                traj_test_aligned = traj_test_aligned + line.strip()
            if line.startswith("%" + str(len(self.demos))):
                copy_lines = True

        # Process it in to the right format
        sequences = OrderedDict()
        curseqname = None
        curseq = ""
        skip_next = False
        # compute the PSSM:
        for line in lines:
            if skip_next:
                continue
            if line.startswith("%"):
                if line.startswith("%" + str(len(self.demos))):
                    skip_next = True
                    continue
                if curseqname is not None:
                    sequences[curseqname] = curseq
                    curseqname = None
                    curseq = ""
                curseqname = line[1:].strip()
            else:
                curseq += line.strip()
        sequences[curseqname] = curseq

        pssm = get_pssm(sequences, self.visitation_frequencies)

        # assign rewards using pssm to clustered states in the query sequence
        traj_test_redist = []
        for i in range(len(traj_test_aligned)):
            char = traj_test_aligned[i]
            if char != '-':
                reward = pssm[i][char]
                if self.mode == 'log':
                    traj_test_redist.append(reward)
                else:
                    traj_test_redist.append(np.exp(reward))

        # Remove "-" gaps
        traj_test_aligned = traj_test_aligned.replace("-", "")

        # scale, threshold and normalize reward
        traj_test_redist = [float(i) - min(traj_test_redist) for i in traj_test_redist]
        traj_test_redist = [float(i) / sum(traj_test_redist) for i in traj_test_redist]
        # assign reward to actual sequence
        # reward is assigned for the last occurrence of the state in case of consecutive similar clustered states
        traj_test_comp = self.assign_cluster_demo(self.cluster_model, [new_sequence], True)

        redist_reward = [0 for i in range(len(traj_test_comp[0]['fasta']))]
        j = len(traj_test_redist) - 1
        for seq in traj_test_comp:
            for i, state in reversed(list(enumerate(seq['fasta']))):
                if j < 0:
                    break
                if state == traj_test_aligned[j]:
                    redist_reward[i] = traj_test_redist[j]
                    j -= 1
        redist_reward.pop(0)

        return redist_reward

    def store_trajectory(self, state_traj, r_return, reward_traj, action_traj, done_traj):
        successful = False
        if r_return >= self.max_return:
            self.sequences.append({"state_traj": state_traj,
                                   "r_return": r_return,
                                   "reward_traj": reward_traj,
                                   "action_traj": action_traj,
                                   "done_traj": done_traj})
            self.num_store_seq += 1
            successful = True

        if self.num_store_seq >= self.enough:
            self.redistribute = True

        return successful

    def cluster_states(self):

        if self.cluster:
            if self.cluster_type == 'SC':
                # Use spectral Clustering
                sc = SpectralClustering(self.num_clusters, affinity='precomputed', assign_labels='discretize')
                clustering = sc.fit_predict(self.sr.sr_table)

                maze = self.env.unwrapped.maze.to_value()
                maze[maze == 1] = -1

                for i, val in enumerate(clustering):
                    state = self.sr.transform_back(i)
                    if maze[state[0], state[1]] != -1:
                        maze[state[0], state[1]] = val
                self.cluster_model = copy.deepcopy(maze)

            elif self.cluster_type == 'AP':
                # Use Affinity Propogation
                preferences = np.zeros(self.sr.state_size)
                preferences[:] = -0.5
                preferences[-1] = 1
                sc = AffinityPropagation(damping=0.5, max_iter=1000, affinity='precomputed', convergence_iter=30,
                                         copy=True, preference=preferences)
                clustering = sc.fit_predict(self.sr.sr_table)
                cluster_model = sc.fit(self.sr.sr_table)

                maze = self.env.unwrapped.maze.to_value()
                maze[maze == 1] = -1
                maze[maze == 2] = -2
                maze[maze == 3] = -2
                maze[maze == 0] = -2
                maze[maze == 4] = -2

                visitation_mask = copy.deepcopy(self.env_maze_visitation)
                visitation_mask[visitation_mask > 0] = 1

                for i in range(maze.shape[0]):
                    for j in range(maze.shape[1]):
                        for k in range(maze.shape[2]):
                            index = self.sr.transform_state([i, j, k])
                            val = clustering[index]
                            if maze[i, j, k] != -1 and visitation_mask[i, j, k] != 0:
                                maze[i, j, k] = val
                # add cluster for states which were never visited
                maze[maze == -2] = np.max(maze) + 1
                # Merge clusters whose exemplars are most similar to each other
                # only if the num of clusters is above some fixed limit
                if np.max(maze) >= self.num_clusters:
                    exemplars = cluster_model.cluster_centers_indices_
                    similarity = []
                    for center in exemplars:
                        for c_center in exemplars:
                            if center != c_center:
                                sim_c_c = self.sr.sr_table[center, c_center]
                                similarity.append({'similarity': sim_c_c, 'index_1': center, 'index_2': c_center})

                    exemplar_sim_list = sorted(similarity, key=lambda i: i['similarity'], reverse=True)
                    # combine the clusters with most similar exemplars:
                    combined_list = []
                    num_clust_combine = np.max(maze) - self.num_clusters
                    clust_combined = 0
                    i = 0

                    while clust_combined < num_clust_combine:
                        most_sim_exemplars = exemplar_sim_list[i]
                        cluster_1 = maze[self.sr.transform_back(most_sim_exemplars['index_1'])[0],
                                         self.sr.transform_back(most_sim_exemplars['index_1'])[1],
                                        self.sr.transform_back(most_sim_exemplars['index_1'])[2]]

                        cluster_2 = maze[
                            self.sr.transform_back(most_sim_exemplars['index_2'])[0],
                            self.sr.transform_back(most_sim_exemplars['index_2'])[1],
                            self.sr.transform_back(most_sim_exemplars['index_2'])[2]]

                        maze[maze == cluster_1] = cluster_2
                        i += 1
                        clust_combined += 1
                    clusters = []
                    for i in range(maze.shape[0]):
                        for j in range(maze.shape[1]):
                            for k in range(maze.shape[2]):
                                if maze[i, j, k] != -1:
                                    clusters.append(maze[i, j, k])

                    clusters = list(set(clusters))
                    for i, cluster in enumerate(clusters):
                        maze[maze == cluster] = i

                max_cluster = np.max(maze)
                maze[-2, -2] = max_cluster + 1
                self.cluster_model = copy.deepcopy(maze)
        return self.cluster_model

    def get_states(self, sequences):
        states = []
        for seq in sequences:
            states.extend(seq['state_traj'])
        return np.stack(states).squeeze()

    def assign_cluster_demo(self, maze, sequences, new_cluster):
        # maze is the cluster model, sequences are the stored trajectories
        _sequences = []
        for seq in sequences:
            cluster_state_traj = []
            reward_traj = []
            _s = -1
            ret = 0
            for trans in seq:
                state = trans.state
                next_state = trans.next_state
                reward = trans.reward
                done = trans.done
                ret += reward
                s = maze[state[0], state[1], state[2]]
                if _s != s:
                    # store only if new cluster is entered
                    cluster_state_traj.append(aa_dict[s])
                    reward_traj.append(reward)
                elif new_cluster:
                    cluster_state_traj.append(aa_dict[s])
                    reward_traj.append(reward)
                _s = s

                if done:
                    s = maze[next_state[0], next_state[1], next_state[2]]
                    if _s != s:
                        # store only if new cluster is entered
                        cluster_state_traj.append(aa_dict[s])
                    elif new_cluster:
                        cluster_state_traj.append(aa_dict[s])

            _sequences.append({"fasta": cluster_state_traj,
                               "r_return": ret,
                               "reward_traj": reward_traj})

        return _sequences

    def assign_cluster_sequences(self, maze, sequences):
        _sequences = []
        for seq in sequences:
            cluster_state_traj = []
            _s = -1
            for i, state in enumerate(seq['state_traj']):
                s = maze[state[0], state[1], state[2]]
                if _s != s:
                    # store only if new cluster is entered
                    cluster_state_traj.append(aa_dict[s])
                _s = s
            _sequences.append({"fasta": cluster_state_traj,
                               "r_return": seq['r_return'],
                               "reward_traj": seq['reward_traj']})
        return _sequences

    def print_cluster_assignments(self, cluster_model, maze):
        maze = maze.astype(str)

        for i in range(cluster_model.shape[0]):
            for j in range(cluster_model.shape[1]):
                s = cluster_model[i, j]
                if s != -1:
                    maze[i, j] = aa_dict[s]

        print("Cluster Assignments:")
        print(maze)

    def update_visitation_frequency(self, state, next_state, done):
        if done:
            self.env_maze_visitation[state[0], state[1], state[2]] = self.env_maze_visitation[state[0], state[1], state[2]] + 1
            self.env_maze_visitation[next_state[0], next_state[1], next_state[2]] = self.env_maze_visitation[
                                                                         next_state[0], next_state[1], next_state[2]] + 1
        else:
            self.env_maze_visitation[state[0], state[1], state[2]] = self.env_maze_visitation[state[0], state[1], state[2]] + 1

    def update_vf_cluster(self, cluster_model):
        self.visitation_frequencies = [0 for i in range(20)]
        for i in range(cluster_model.shape[0]):
            for j in range(cluster_model.shape[1]):
                for k in range(cluster_model.shape[2]):
                    s = cluster_model[i, j, k]
                    if s != -1:
                        num_visit = self.env_maze_visitation[i, j, k]
                        self.visitation_frequencies[s] += num_visit
