"""Parses phosphoELM data"""

import numpy as np
import pandas as pd
import random
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB import PDBParser
import os
#from scipy.signal import find_peaks

### Parameters
train_portion = 0.7
test_portion = 0.2
#psite_padding = 50 # Include only +- psite_padding AAs around +P site
input_seqlen = 100 # Length of input sequence
min_psite_padding = 5 # In case of cutting sequence and randomly placing the +P site, minimum padding around +P site position
psite_group_closeness = 5 # group psites together if they are <= psite_group_closeness positions apart
psite_window = 3 # calculate a psite score for each position, taking into account the number of psites in += psite_window AAs around position
psite_binsize = 20 # Length of each bin
alphafold_dir = './alphafold'
alphafold_tar = 'UP000005640_9606_HUMAN.tar'
ires, ichem, icharge, ihbond, ipolarity, ivol, ihydropathy, ipKa, ipKb, ipI, icacoord, ircoord = np.arange(12) # Properties indices

class ProtStructure:
    structure_dict = {} # {acc: ProtStructure}

    def __init__(self, acc, pdb_file):
        self.acc = acc
        self.path = os.path.join(alphafold_dir, pdb_file)
        self.sequence = list(SeqIO.parse(self.path, 'pdb-seqres'))[0]
        parser = PDBParser()
        self.structure = parser.get_structure(acc, self.path)
        residues = self.structure.get_residues()
        self.residues = []
        self.r_coords = []
        self.ca_coords = []
        for residue in residues:
            self.residues.append(residue)
            atoms = residue.get_atoms()
            r_group = []
            for atom in atoms:
                if atom.name == 'CA':
                    self.ca_coords.append(atom.coord)
                elif atom.name != 'C' and atom.name != 'N':
                    r_group.append(atom.coord)
            if len(r_group) == 0: # Glycine does not have R group
                self.r_coords.append(self.ca_coords[-1]) # Use Ca coordinates
            else:
                x = sum(coord[0] for coord in r_group) / len(r_group)
                y = sum(coord[1] for coord in r_group) / len(r_group)
                z = sum(coord[2] for coord in r_group) / len(r_group)
                self.r_coords.append(np.array([x, y, z]))
        if len(self.residues) != len(self.sequence):
            print('Missing residues for {}'.format(acc))
        self.structure_dict[acc] = self

    def get_coord_ndarray(self, seq_range): # range[1] exclusive, range[0] inclusive
        arr = np.zeros((seq_range[1] - seq_range[0], 2, 20), dtype=np.int)
        arr[i, :, 0:3] = np.array([self.ca_coords[seq_range[0]:seq_range[1]], self.r_coords[seq_range[0]:seq_range[1]]])
        return arr

class AminoAcid:
    chemical_properties = ['aliphatic', 'aromatic', 'sulfur', 'hydroxyl', 'basic', 'acidic', 'amide', 'glycine', 'proline']
    charges = ['positive', 'neutral', 'negative']
    hydrogen_bonds = ['donor', 'acceptor']
    polarity = ['polar', 'nonpolar']
    hydropathy_range = (-4.5, 4.5)
    hydropathy_binsize = (hydropathy_range[1] - hydropathy_range[0]) / 20
    volume_range = (60, 228)
    volume_binsize = (volume_range[1] - volume_range[0]) / 20
    pKa_range = (1.8, 2.9)
    pKa_binsize = (pKa_range[1] - pKa_range[0]) / 20
    pKb_range = (8.7, 10.7)
    pKb_binsize = (pKb_range[1] - pKb_range[0]) / 20
    pI_range = (2.7, 10.8)
    pI_binsize = (pI_range[1] - pI_range[0]) / 20

    def __init__(self, name, chem, charge, h_bonds, polarity, volume, hydropathy, pKa, pKb, pI):
        self.name = name
        self.chemical_property = chem
        self.charge = charge
        self.h_bonds = h_bonds
        self.polarity = polarity
        self.volume = volume
        self.hydropathy = hydropathy
        self.pKa = pKa
        self.pKb = pKb
        self.pI = pI

    def chem_v(self):
        v = np.zeros(20, dtype=np.int)
        v[self.chemical_properties.index(self.chemical_property)] = 1
        return v

    def charge_v(self):
        v = np.zeros(20, dtype=np.int)
        v[self.charges.index(self.charge)] = 1
        return v

    def hbond_v(self):
        v = np.zeros(20, dtype=np.int)
        for bond in self.h_bonds:
            v[self.hydrogen_bonds.index(bond)] = 1
        return v

    def polarity_v(self):
        v = np.zeros(20, dtype=np.int)
        v[self.charges.index(self.charge)] = 1
        return v

    def volume_v(self):
        v = np.zeros(20, dtype=np.int)
        v[int((self.volume - self.volume_range[0]) // self.volume_binsize)] = 1
        return v

    def hydropathy_v(self):
        v = np.zeros(20, dtype=np.int)
        v[int((self.hydropathy - self.hydropathy_range[0]) // self.hydropathy_binsize)] = 1
        return v

    def pKa_v(self):
        v = np.zeros(20, dtype=np.int)
        v[int((self.pKa - self.pKa_range[0]) // self.pKa_binsize)] = 1
        return v

    def pKb_v(self):
        v = np.zeros(20, dtype=np.int)
        v[int((self.pKb - self.pKb_range[0]) // self.pKb_binsize)] = 1
        return v

    def pI_v(self):
        v = np.zeros(20, dtype=np.int)
        v[int((self.pI - self.pI_range[0]) // self.pI_binsize)] = 1
        return v

### Make amino acids
A = AminoAcid(name='A', chem='aliphatic', charge='neutral', h_bonds=[], polarity='nonpolar', volume=88.6, hydropathy=1.8, pKa=2.34, pKb=9.69, pI=6.00)
C = AminoAcid(name='C', chem='sulfur', charge='neutral', h_bonds=[], polarity='nonpolar', volume=108.5, hydropathy=2.5, pKa=1.96, pKb=10.28, pI=5.07)
D = AminoAcid(name='D', chem='acidic', charge='negative', h_bonds=['acceptor'], polarity='polar', volume=111.1, hydropathy=-3.5, pKa=1.88, pKb=9.60, pI=2.77)
E = AminoAcid(name='E', chem='acidic', charge='negative', h_bonds=['acceptor'], polarity='polar', volume=138.4, hydropathy=-3.5, pKa=2.19, pKb=9.67, pI=3.22)
F = AminoAcid(name='F', chem='aromatic', charge='neutral', h_bonds=[], polarity='nonpolar', volume=189.9, hydropathy=2.8, pKa=1.83, pKb=9.13, pI=5.48)
G = AminoAcid(name='G', chem='glycine', charge='neutral', h_bonds=[], polarity='nonpolar', volume=60.1, hydropathy=-0.4, pKa=2.34, pKb=9.60, pI=5.97)
H = AminoAcid(name='H', chem='basic', charge='positive', h_bonds=['donor', 'acceptor'], polarity='polar', volume=153.2, hydropathy=-3.2, pKa=1.82, pKb=9.17, pI=7.59)
I = AminoAcid(name='I', chem='aliphatic', charge='neutral', h_bonds=[], polarity='nonpolar', volume=166.7, hydropathy=4.5, pKa=2.36, pKb=9.60, pI=6.02)
K = AminoAcid(name='K', chem='basic', charge='positive', h_bonds=['donor'], polarity='polar', volume=168.6, hydropathy=-3.9, pKa=2.18, pKb=8.95, pI=9.74)
L = AminoAcid(name='L', chem='aliphatic', charge='neutral', h_bonds=[], polarity='nonpolar', volume=166.7, hydropathy=3.8, pKa=2.36, pKb=9.60, pI=5.98)
M = AminoAcid(name='M', chem='sulfur', charge='neutral', h_bonds=[], polarity='nonpolar', volume=162.9, hydropathy=1.9, pKa=2.28, pKb=9.21, pI=5.74)
N = AminoAcid(name='N', chem='amide', charge='neutral', h_bonds=['donor', 'acceptor'], polarity='polar', volume=114.1, hydropathy=-3.5, pKa=2.02, pKb=8.80, pI=5.41)
P = AminoAcid(name='P', chem='proline', charge='neutral', h_bonds=[], polarity='nonpolar', volume=112.7, hydropathy=-1.6, pKa=1.99, pKb=10.60, pI=6.30)
Q = AminoAcid(name='Q', chem='amide', charge='neutral', h_bonds=['donor', 'acceptor'], polarity='polar', volume=143.8, hydropathy=-3.5, pKa=2.17, pKb=9.13, pI=5.65)
R = AminoAcid(name='R', chem='basic', charge='positive', h_bonds=['donor'], polarity='polar', volume=173.4, hydropathy=-4.5, pKa=2.17, pKb=9.04, pI=10.76)
S = AminoAcid(name='S', chem='hydroxyl', charge='neutral', h_bonds=['donor', 'acceptor'], polarity='polar', volume=89.0, hydropathy=-0.8, pKa=2.21, pKb=9.15, pI=5.68)
T = AminoAcid(name='T', chem='hydroxyl', charge='neutral', h_bonds=['donor', 'acceptor'], polarity='polar', volume=116.1, hydropathy=-0.7, pKa=2.09, pKb=9.10, pI=5.60)
V = AminoAcid(name='V', chem='aliphatic', charge='neutral', h_bonds=[], polarity='nonpolar', volume=140.0, hydropathy=4.2, pKa=2.32, pKb=9.62, pI=5.96)
W = AminoAcid(name='W', chem='aromatic', charge='neutral', h_bonds=['donor'], polarity='nonpolar', volume=227.8, hydropathy=-0.9, pKa=2.83, pKb=9.39, pI=5.89)
Y = AminoAcid(name='Y', chem='aromatic', charge='neutral', h_bonds=['donor', 'acceptor'], polarity='polar', volume=193.6, hydropathy=-1.3, pKa=2.20, pKb=9.11, pI=5.66)
amino_acids = [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]
aa_dict = {amino_acids[i].name:(amino_acids[i], i) for i in range(len(amino_acids))}

### Generate a random list of ints between given interval
def get_random_list(len, start, stop):
    rlist = []
    for i in range(len):
        n = random.randint(start, stop)
        rlist.append(n)
    return rlist

### Convert sequence input to one-hot ndarray
def onehot_seq_input(seq_dataset):
    onehot = np.zeros((len(seq_dataset), len(seq_dataset[0]), 12, 20), dtype=np.int) # 1st dimension: different sequence records; 2nd dimension: residue positions in a sequence; 3rd dimension: number of properties; 4th dimension: vector length
    """for acc in acc_to_pdb: # Build ProtStructures
        ProtStructure(acc, acc_to_pdb[acc])"""
    for iseq in range(len(seq_dataset)):
        for ipos in range(len(seq_dataset[iseq])):
            residue = seq_dataset[iseq][ipos]
            if residue not in aa_dict:
                continue
            aa = aa_dict[residue][0]
            onehot[iseq, ipos, ires, aa_dict[residue][1]] = 1
            onehot[iseq, ipos, ichem] = aa.chem_v()
            onehot[iseq, ipos, icharge] = aa.charge_v()
            onehot[iseq, ipos, ihbond] = aa.hbond_v()
            onehot[iseq, ipos, ipolarity] = aa.polarity_v()
            onehot[iseq, ipos, ivol] = aa.volume_v()
            onehot[iseq, ipos, ihydropathy] = aa.hydropathy_v()
            onehot[iseq, ipos, ipKa] = aa.pKa_v()
            onehot[iseq, ipos, ipKb] = aa.pKb_v()
            onehot[iseq, ipos, ipI] = aa.pI_v()
        """seq = seq_dataset[iseq]
        seq_no_padding = seq.strip('0')
        acc = seq_to_acc[seq]
        pstructure = ProtStructure.structure_dict[acc]
        istart = pstructure.sequence.find(seq_no_padding)
        if istart == -1:
            print('No matching structure for {}'.format(seq_no_padding))
            continue
        istop = istart + len(seq_no_padding)
        front_padding_len = seq.find(seq_no_padding)
        onehot[iseq, front_padding_len:front_padding_len + len(seq_no_padding), icacoord:ircoord + 1] = pstructure.get_coord_ndarray([istart, istop])"""
    return onehot

def bin_ytrain(pos_input):
    binned_ytrain = np.zeros((len(pos_input), input_seqlen // psite_binsize, psite_binsize), dtype=np.int) # Assumes input_seqlen is divisible by psite_binsize
    for i in range(len(pos_input)):
        for pos in pos_input[i]:
            i_bin = (pos - 1) // psite_binsize
            i_inbin = (pos - 1) % psite_binsize
            binned_ytrain[i, i_bin, i_inbin] = 1
    return binned_ytrain

### Use sliding window to make a distribution of p sites
def get_sliding_values(positions, seq_len):
    values = []
    for i in range(seq_len):
        window_start = i - psite_window + 1
        window_stop = i + psite_window + 1
        if window_start < 1:
            window_start = 1
        if window_stop > seq_len:
            window_stop = seq_len
        value = len([pos for pos in positions if pos >= window_start and pos <= window_stop])
        values.append(value)
    return values

### Merge positions and keep only unique sequences
def merge_seq_pos(df, group_psites):
    seqs = []
    pos = []
    for index, row in df.iterrows():
        if row['sequence'] in seqs:
            pos[seqs.index(row['sequence'])].append(row['position'])
            pos[seqs.index(row['sequence'])].sort()
        else:
            seqs.append(row['sequence'])
            pos.append([row['position']])
    if group_psites:
        print('window size = {}'.format(psite_window * 2 + 1))
        max_grouplen = 0
        max_numgroups = 0
        for i in range(len(pos)):
            """sliding_values = get_sliding_values(pos[i], len(seqs[i]))
            peaks, properties = find_peaks(sliding_values)
            peaks = [peak + 1 for peak in peaks]
            pos[i]"""
            filtered_sites = []
            group = []
            for position in pos[i]:
                if group == []:
                    group.append(position)
                elif position - curr > psite_group_closeness:
                    filtered_sites.append(group[0])
                    if len(group) > max_grouplen:
                        max_grouplen = len(group)
                    group = [position]
                else:
                    group.append(position)
                curr = position
            if len(filtered_sites) > max_numgroups:
                max_numgroups = len(filtered_sites)
            pos[i] = filtered_sites
        print('Maximum number of sites in each group: {}; Maximum number of groups in each sequence: {}'.format(max_grouplen, max_numgroups))
    df = pd.DataFrame([seqs, pos]).transpose()
    df.columns = ['sequence', 'position']
    return df

### Return a dict {acc: pdb_filename in alphafold_dir}
def get_acc_to_pdb(accs):
    files = [file for file in os.listdir(alphafold_dir) if file.endswith('.pdb')]
    pdbs = {}
    accs_no_pdb = []
    for acc in accs:
        matches = [file for file in files if acc in file]
        if len(matches) > 1:
            print('Multiple structure files found for {}: {}'.format(acc, ','.join(matches)))
        elif len(matches) == 0:
            accs_no_pdb.append(acc)
        else:
            pdbs[acc] = matches[0]
    print('{} accessions do not have a matching structure'.format(len(accs_no_pdb)))
    return pdbs

def parse(input_file, verbose=False, cut_seqs=False, random_psite=False, group_psites=False, bin_seqs=False):
    df = pd.read_table(input_file)
    df.drop(['pmids', 'source', 'entry_date', 'kinases', 'code'], axis=1, inplace=True)
    df = df[df['species'] == 'Homo sapiens']
    accs = df[['acc']].drop_duplicates()['acc'].tolist()
    acc_to_pdb = get_acc_to_pdb(accs)
    df = df[df['acc'].isin(acc_to_pdb.keys())]
    df.drop(['species'], axis=1, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    df = df.drop_duplicates()
    df['position'] = pd.to_numeric(df['position'])
    merged_df = merge_seq_pos(df, group_psites)
    seq_to_acc = {} # {sequence: acc}
    ### Cut sequence to just the amino acids around +P site and add padding if needed
    if cut_seqs:
        pre_psite_padding = input_seqlen / 2
        post_psite_padding = input_seqlen / 2 - 1
        if random_psite:
            psite_positions = get_random_list(len(df), min_psite_padding, input_seqlen - 1 - min_psite_padding)
            i = 0
        cut_data = []
        for index, row in df.iterrows():
            all_positions = merged_df[merged_df['sequence'] == row['sequence']].iloc[0]['position']
            target_pos = row['position']
            if target_pos not in all_positions:
                continue
            if random_psite:
                pre_psite_padding = psite_positions[i]
                post_psite_padding = input_seqlen - 1 - psite_positions[i]
                i += 1
            if row['position'] > pre_psite_padding:
                seq_start = row['position'] - 1 - pre_psite_padding
                row['sequence'] = row['sequence'][seq_start:]
            else:
                padding = pre_psite_padding - row['position']
                row['sequence'] = padding * '0' + row['sequence']
            row['position'] = pre_psite_padding + 1
            if len(row['sequence']) - row['position'] > post_psite_padding:
                seq_end = row['position'] + post_psite_padding
                row['sequence'] = row['sequence'][:seq_end]
            else:
                padding = post_psite_padding - (len(row['sequence']) - row['position'])
                row['sequence'] = row['sequence'] + padding * '0'
            ### Add all other P site positions in this sequence to position column
            pos_shift = row['position'] - target_pos
            new_positions = [pos + pos_shift for pos in all_positions if pos + pos_shift <= input_seqlen and pos + pos_shift >= 1]
            cut_data.append([row['sequence'], new_positions])
            seq_to_acc[row['sequence']] = row['acc']
        df = pd.DataFrame(cut_data, columns=['sequence', 'position'])
        print('Maximum number of positions in cut sequence: {}'.format(max(len(cut_data_row[1]) for cut_data_row in cut_data)))
    else:
        ### Add paddings
        for index, row in df.iterrows:
            seq_to_acc[row['sequence']] = row['acc']
        df = merged_df
        max_seqlen = max(len(row['sequence']) for index, row in df.iterrows())
        for index, row in df.iterrows():
            seqlen = len(row['sequence'])
            padding = max_seqlen - seqlen
            row['sequence'] = row['sequence'] + padding * '0'
            df.loc[index] = row.values
    df = df.iloc[np.random.permutation(len(df))] # Shuffle
    ### Extract input to CNN
    x = df['sequence'].tolist()
    y = df['position'].tolist()
    if bin_seqs:
        z = []
        for positions in y:
            bins = []
            for i in range((input_seqlen - 1) // psite_binsize + 1):
                if any(pos > i * psite_binsize and pos <= (i + 1) * psite_binsize for pos in positions):
                    bins.append(1)
                else:
                    bins.append(0)
            z.append(bins)
    train_size = int(len(df) * train_portion)
    test_size = int(len(df) * test_portion)
    validation_size = len(df) - train_size - test_size
    ### Print data sizes
    if verbose:
        print('Training set: {}'.format(train_size))
        print('Test set: {}'.format(test_size))
        print('Validation set: {}'.format(validation_size))
        print('Total: {}'.format(len(df)))
    ### Divide data to training set, test set, validation set
    x_train = x[:train_size]
    x_test = x[train_size:train_size + test_size]
    x_valid = x[train_size + test_size:]
    y_train = y[:train_size]
    y_test = y[train_size:train_size + test_size]
    y_valid = y[train_size + test_size:]
    if bin_seqs:
        y_train = bin_ytrain(y_train)
    rets = {'x_train': x_train, 'x_test': x_test, 'x_valid': x_valid,
    'y_train': y_train, 'y_test': y_test, 'y_valid': y_valid,
    'acc_to_pdb': acc_to_pdb, 'seq_to_acc': seq_to_acc}
    if bin_seqs:
        rets['z_train'] = z[:train_size]
        rets['z_test'] = z[train_size:train_size + test_size]
        rets['z_valid'] = z[train_size + test_size:]
    return rets

### Only unzip structures in alphafold_tar that matches an accession in phosphorylation records
def process_structures(input_file):
    df = pd.read_table(input_file)
    df.drop(['pmids', 'source', 'entry_date', 'kinases', 'code'], axis=1, inplace=True)
    df = df[df['species'] == 'Homo sapiens']
    accs = df[['acc']].drop_duplicates()['acc'].tolist()
    no_pdb = []
    multi_chains = []
    for acc in accs:
        wildcards = '\'' + '*' + acc + '*pdb*' + '\''
        args = ['tar', '-C', alphafold_dir, '-xvf', alphafold_tar, '--wildcards', wildcards]
        argstr = ' '.join(args)
        try:
            output = subprocess.check_output(argstr, shell=True)
        except subprocess.CalledProcessError:
            no_pdb.append(acc)
            continue
        extracted = str(output, 'utf-8')[:str(output, 'utf-8').find('\n')] # Only use first matching pdb file found
        args = ['gzip', '-d', alphafold_dir + '/' + extracted]
        subprocess.call(args)
        pdb = extracted.strip('.gz')
        pdb_path = os.path.join(alphafold_dir, pdb)
        recs = list(SeqIO.parse(pdb_path, "pdb-seqres"))
        if len(recs) > 1:
            multi_chains.append(acc)
            continue
        """else:
            target_seq = df[df['acc'] == acc].iloc[0]['sequence']
            if target_seq == recs[0].seq: # No need to make aln if seqs are the same
                continue
            target_rec = SeqRecord(Seq(target_seq), id=acc)
            fasta_name = acc + '.fasta'
            SeqIO.write([target_rec, recs[0]], fasta_name, 'fasta')
            aln_name = acc + '_aln.fasta'
            args = ['mafft', '--maxiterate', '1000', '--genafpair', fasta_name, '>', aln_name]
            argstr = ' '.join(args)
            subprocess.run(argstr, shell=True)"""
    for acc in no_pdb:
        print('Failed to extract pdb file for {}'.format(acc))
    for acc in multi_chains:
        print('Multiple chains in {}'.format(acc))


def main():
    datasets = parse('phosphoELM_all_data.txt',
    verbose=True,
    cut_seqs=True,
    random_psite=True,
    group_psites=False,
    bin_seqs=True)
    x_train = datasets['x_train']
    y_train = bin_ytrain(datasets['y_train'])
    z_train = datasets['z_train']
    x_test = datasets['x_test']
    y_test = bin_ytrain(datasets['y_test'])
    z_test = datasets['z_test']
    x_valid = datasets['x_valid']
    y_valid = bin_ytrain(datasets['y_valid'])
    z_valid = datasets['z_valid']

    return x_train, y_train, z_train, x_test, y_test, z_test, x_valid, y_valid, z_valid


# main()