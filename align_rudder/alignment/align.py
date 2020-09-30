from collections import OrderedDict
import pandas as pd
from os import path
import os
import numpy as np
import subprocess
from collections import Counter
import copy

# Amino Acid Letter Code
aa_dict = OrderedDict({
    0: "G", 1: "A", 2: "L", 3: "M",
    4: "F", 5: "W", 6: "K", 7: "Q", 8: "E", 9: "S",
    10: "P", 11: "V", 12: "I", 13: "C",
    14: "Y", 15: "H", 16: "R", 17: "N", 18: "D", 19: "T"
})

aa_dict_2 = OrderedDict({
    "G": 0, "A": 1, "L": 2, "M": 3,
    "F": 4, "W": 5, "K": 6, "Q": 7, "E": 8, "S": 9,
    "P": 10, "V": 11, "I": 12, "C": 13,
    "Y": 14, "H": 15, "R": 16, "N": 17, "D": 18, "T": 19
})

aa_dict_3 = OrderedDict({
    "G": 0, "A": 0, "L": 0, "M": 0,
    "F": 0, "W": 0, "K": 0, "Q": 0, "E": 0, "S": 0,
    "P": 0, "V": 0, "I": 0, "C": 0,
    "Y": 0, "H": 0, "R": 0, "N": 0, "D": 0, "T": 0
})


# create FASTA encoded file from sequences

def create_fasta_sequences(sequences, outdir="runs/alignment",
                           top_n=10, exclude=None):
    file_suffix = ""
    outfile = None
    fasta_sequences = pd.DataFrame(sequences)
    top_n_seq = fasta_sequences.sort_values(["r_return"], ascending=[False]).iloc[:top_n]
    # write
    if outdir is not None:
        outfile = path.join(outdir, "top_{}{}.fasta".format(top_n, file_suffix))
        with open(outfile, "w") as f:
            for file, seq in top_n_seq.iterrows():
                f.write(">{}|rew={}|\n{}\n".format(path.basename(str(file)), seq.r_return, seq.fasta))

    return top_n_seq, fasta_sequences, outfile


# Create scoring matrix based on state frequencies
def create_scoring_matrix(
        outdir='runs/alignment',
        outfile='scoring_matrix',
        fasta_sequences=None,
        offdiag=-10.0,
        main_diag_factor=0.1,
        scaling=None,
        reward_op="mul", freq_scaling=None):
    # count AAs
    sym_dict = OrderedDict({k: 0 for k in aa_dict.values()})
    print("Sym Dict:", sym_dict)
    n_total = 0
    _num_states = 0
    for traj in fasta_sequences.fasta:
        n_total += len(traj)
        for sym in traj:
            sym_dict[sym] += 1
            _num_states += 1
    assert n_total == _num_states
    sym_dict['T'] = 1

    # log frequency
    for sym, cnt in sym_dict.items():
        if cnt > 0:
            sym_dict[sym] = -2 * np.log(cnt / n_total)
            if freq_scaling == "square":
                sym_dict[sym] = sym_dict[sym] ** 2

    # scores
    scoring = np.full(shape=(len(sym_dict), len(sym_dict)),
                      fill_value=offdiag,
                      dtype=np.float32)
    # main_diag and reward scaling
    for i in range(len(scoring)):
        scoring[i, i] = sym_dict[list(sym_dict.keys())[i]] * main_diag_factor

    # Write
    outfile += "_m{}_o{}".format(main_diag_factor, offdiag)
    if scaling is not None:
        outfile += "_{}{}".format(reward_op, scaling)
    if freq_scaling is not None:
        outfile += "_f{}".format(freq_scaling)
    outfile = path.join(outdir, outfile)
    offdiag_gap = offdiag / 10
    with open(outfile, "w") as f:
        f.write("   " + " ".join(list(aa_dict.values())) + " *\n")
        for i in range(len(scoring)):
            f.write(list(sym_dict.keys())[i] + " ")
            f.write(" ".join([str(x) for x in scoring[i]]))
            f.write(" " + str(offdiag))
            f.write("\n")
        f.write("*" + (" " + str(offdiag_gap)) * (len(scoring) + 1))
    scoring = 0
    return scoring, outfile

def get_alignment(seq_file, top_n, score_file, path_curr_dir, file_name_profile):
    indir = path_curr_dir
    outdir = path_curr_dir
    infile = seq_file
    outfile = "{}/msa_{}.aln".format(outdir, file_name_profile)
    msatree = "{}/msa_{}.dnd".format(outdir, top_n)

    gap_open = 0
    gap_ext = 0

    if os.path.exists(outfile):
        os.remove(outfile)
    if os.path.exists(msatree):
        os.remove(msatree)

    cmd = "clustalw2 -ALIGN -CLUSTERING=UPGMA -NEGATIVE " \
          "-INFILE={infile} " \
          "-OUTFILE={outfile} " \
          "-PWMATRIX={scores} -PWGAPOPEN={gapopen} -PWGAPEXT={gapext} " \
          "-MATRIX={scores} -GAPOPEN={gapopen} -GAPEXT={gapext} -CASE=UPPER " \
          "-NOPGAP -NOHGAP -MAXDIV=0 -ENDGAPS -NOVGAP " \
          "-NEWTREE={tree} -TYPE=PROTEIN -OUTPUT=GDE".format(
        infile=infile, outfile=outfile, scores=score_file, gapopen=gap_open, gapext=gap_ext,
        tree=msatree
    )

    output = subprocess.run(cmd.split(" "), stdout=subprocess.PIPE).stdout.decode("utf-8")
    # read output
    with open(outfile, "r") as f:
        lines = f.readlines()
    sequences = OrderedDict()
    curseqname = None
    curseq = ""
    for line in lines:
        if line.startswith("%"):
            if curseqname is not None:
                sequences[curseqname] = curseq
                curseqname = None
                curseq = ""
            curseqname = line[1:].strip()
        else:
            curseq += line.strip()
    sequences[curseqname] = curseq
    alignment = np.array([[c for c in seq] for seq in sequences.values()])
    return alignment


def get_pssm(sequences, visitation_frequencies):
    for k in sequences.keys():
        alignment_length = len(sequences[k])

    dict_list = []
    for i in range(alignment_length):
        dict_list.append(copy.deepcopy(aa_dict_3))

    pssm = np.array(copy.deepcopy(dict_list))

    # get counts per column
    for i in range(alignment_length):
        for key in sequences.keys():
            char = sequences[key][i]
            if char != '-':
                pssm[i][char] += 1
    # normalize the counts to sum upto 1
    for i in range(alignment_length):
        sum_count = sum(pssm[i].values()) + 1e-8
        pssm[i] = {k: v / sum_count for k, v in pssm[i].items()}

    # divide by visitation frequency of each state and compute log
    for i in range(alignment_length):
        for k in pssm[i].keys():
            pssm[i][k] = (1e-8 + pssm[i][k]) * np.sum(visitation_frequencies) / (
                    1e-8 + visitation_frequencies[aa_dict_2[k]])
            pssm[i][k] = np.log(pssm[i][k])

    return pssm


def get_consensus(aligned_sequences, type, thresh):
    l = len(aligned_sequences[0])
    seq = np.stack(aligned_sequences)
    num_sequences = seq.shape[0]
    consensus = []
    redistributed_reward = []
    if type == "most_common":
        # selecting for every comparison, select most common symbol amongst sequences
        for i in range(l):
            sym = Counter(seq[:, i]).most_common(1)[0][0]
            if sym != '-':
                consensus.append(sym)
    elif type == "all":
        # select the cluster if it is present in all the sequences
        # or atleast greater than a certain threshold of sequences
        consensus_thr = thresh
        for i in range(l):
            elements, counts = np.unique(seq[:, i], return_counts=True)
            max_element = elements[np.argmax(counts)]
            if max_element != "-":
                agree = counts[np.argmax(counts)] / num_sequences >= consensus_thr
                if agree:
                    consensus += max_element
                    redistributed_reward.append(counts[np.argmax(counts)] / num_sequences)

    # Give reward only for attaining clusters
    # i.e remove recurring clusters
    consensus_reward = {"consensus": consensus, "reward": redistributed_reward}

    return consensus, consensus_reward
