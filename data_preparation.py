#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import json
import time
import math
import copy
import requests
import numpy as np
import scipy as sp
import gudhi as gd
import gudhi.representations
import networkx as nx
from prody import *
from pylab import *
from argparse import Namespace
from matplotlib import pyplot as plt
from matplotlib import colors as mplcolor
from IPython.display import clear_output
from tqdm.auto import tqdm
import os
import json
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_networkx
import csv
from argparse import Namespace

from tqdm.auto import tqdm
import py3Dmol


# In[7]:


table = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 
         'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
         'LEU':10, 'LYS':11, 'MET':12, 'PHE':13, 'PRO':14,
         'SER':15, 'THR':16, 'TRP':17, 'TYR':18, 'VAL':19, 
         'ASX': 3, 'GLX': 6, 'CSO': 4, 'HIP': 8, 'HSD': 8,
         'HSE': 8, 'HSP': 8, 'MSE':12, 'SEC': 4, 'SEP':15,
         'TPO':16, 'PTR':18, 'XLE':10, 'XAA':20}
os.chdir('C:/Users/moham/Downloads/dat')
target_property_file = 'trg.csv'
with open(target_property_file) as f:
        reader = csv.reader(f)
        print(reader)
        target_data = [row for row in reader]
target_data = target_data[1:]


##Process structure files and create structure graphs
data_list = []
for index in range(0, len(target_data)):

    structure_id = target_data[index][0]
    data = Data()

    ##Read in structure file using ase
    
    pdb = parsePDB(structure_id + ".pdb")
    anm = ANM(pdb)
    if 'A' in pdb.getChids():
        # Select only chain A atoms
        c_atoms = pdb.select('chain A and name CA')
    else:
        # Select only chain B atoms
        c_atoms = pdb.select('name CA') 
    anm.buildHessian(c_atoms, cutoff=10, gamma=1, n_cpu=4, norm=True)

    """ Kirchhoff matrix """
    K = anm.getKirchhoff()

    D = np.diag(np.diag(K) + 1.)

    """ Contact map """
    cont = -(K - D)

    """ Mode calculation """

    anm.calcModes(n_modes=15, zeros=False, turbo=True)

    flu = calcSqFlucts(anm)

    freqs = []
    for mode in anm:
        freqs.append(math.sqrt(mode.getEigval()))



    """ Correlation map """
    corr = calcCrossCorr(anm)

    corr_abs = np.abs(corr)
    corr_abs[corr_abs < 0.5] = 0
    diff = cont - corr_abs
    diff[diff < 0] = -1
    diff[diff >= 0] = 0
    pdbcontcorredges = [int(np.sum(cont)), int(-np.sum(diff))]


    """ Adjacency matrix """

    comb = cont + diff # 1: contact edge / -1: correlation edge
    Adj = np.abs(comb)
    g = nx.from_numpy_array(Adj)

    attrs = {}
    for i, resname in enumerate(c_atoms.getResnames()):
        attrs[i] = {}
        attrs[i]["resname"] = resname

    nx.set_node_attributes(g, attrs)

    """ edge attributes """

    for edge in g.edges:
        node_i, node_j = edge        
        if comb[node_i][node_j] == 1:
            g.edges[edge]["weight"] = 1   # contact map
        elif comb[node_i][node_j] == -1:
            g.edges[edge]["weight"] = -1  # correlation map

    ''' map from serial id to residue id '''
    mapping = dict(zip(g, c_atoms.getResnums().tolist()))
    g = nx.relabel.relabel_nodes(g, mapping)

    data = from_networkx(g)

    # ========
    # features
    # ========

    uniques, counts = np.unique(list(table.values()), return_counts=True)

    x = np.zeros((len(data.resname), len(uniques)))

    for j, residue in enumerate(data.resname):
        if residue not in table:
            residue = 'XAA'
        x[j,table[residue]] = 1
    x = torch.tensor(x)

    flu = torch.tensor(flu).unsqueeze(1)
    x = torch.cat((x,flu),1).float()
    target = target_data[index][1:]
    y = torch.Tensor(np.array([target], dtype=np.float32))
    data.y = y
    data.x = x.float()
    del data.resname, data.weight
    data_list.append(data)


# In[22]:



# Read data
df_train = pd.read_pickle("train_seq.p")
df_test = pd.read_pickle("test_seq.p")
dt = pd.read_pickle("train_biological.p")
dk = pd.read_pickle("test_biological.p")

# Convert sequences to DataFrame
k = df_train['Seq']
k.to_csv('seq.csv')
del dt["Seq"]
del dk["Seq"]
del dk['3Str']
del dk['8Str']
del dt['3Str']
del dt['8Str']
del dk['Acc_Sol']
del dt['Acc_Sol']
# Normalize data
scaler = MinMaxScaler()
dk = scaler.fit_transform(dk)
dt = scaler.fit_transform(dt)

X = dt
X1 = dk


# In[66]:


codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
char_dict = {code: index + 1 for index, code in enumerate(codes)}

def integer_encoding(data):
    encode_list = []
    for row in data:
        row_encode = [char_dict.get(code, 0) for code in row]
        encode_list.append(np.array(row_encode))
    return encode_list

train_encode = integer_encoding(df_train['Seq'].values)
test_encode = integer_encoding(df_test['Seq'].values)

# Padding sequences
max_length = 800
train_pad = nn.utils.rnn.pad_sequence([torch.tensor(row) for row in train_encode], batch_first=True)
test_pad = nn.utils.rnn.pad_sequence([torch.tensor(row) for row in test_encode], batch_first=True)
train_pad_index = train_pad.long()
test_pad_index = test_pad.long()
# One-hot encoding
train_ohe = F.one_hot(train_pad_index, num_classes=21)
test_ohe = F.one_hot(test_pad_index, num_classes=21 )
# One-hot encoding

# Label encoding
le = LabelEncoder()
y_train_le = le.fit_transform(df_train['solubility'])
y_test_le = le.transform(df_test['solubility'])

# Define PyTorch data loaders
train_dataset = TensorDataset(train_ohe, torch.tensor(X), torch.tensor(y_train_le))
test_dataset = TensorDataset(test_ohe, torch.tensor(X1), torch.tensor(y_test_le))
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# In[74]:


for i in range(len(data_list)):
    data_list[i].seq = train_dataset[i][0]
    data_list[i].struc = train_dataset[i][1]
    data_list[i].Y = train_dataset[i][2]

    
    


# In[ ]:




