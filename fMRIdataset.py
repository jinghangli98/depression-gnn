import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np 
import os
from tqdm import tqdm
import pdb
import networkx as nx
from itertools import chain
import glob
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx
print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

"""
!!!
NOTE: This file was replaced by dataset_featurizer.py
but is kept to illustrate how to build a custom dataset in PyG.
!!!
"""


class fMRI(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        super(fMRI, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        IDs = self.data['BID']
        for index, ID in enumerate(IDs):
            data = self.buildData(ID, self.data, 0)
            # Create data object
            # data = Data(x=node_feats, 
            #             edge_index=edge_index,
            #             y=label) 
            data.sex = self.data[self.data['BID'] == ID]['sex'].item()
            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                f'data_test_{index}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                f'data_{index}.pt'))


    def buildData(self, ID, df, threshold):
        regions = []
        nii_columns = [col for col in df.columns if '.nii' in col]
        for connection in nii_columns:
            parts = connection.split('--')
            regions.append([part.split('.')[0].strip('r') for part in parts])
        
        regions = list(set(list(chain.from_iterable(regions))))

        connectivity_matrix = np.zeros((len(regions), len(regions)))
        for connection in nii_columns:
            parts = connection.split('--')
            parts = [part.split('.')[0].strip('r') for part in parts]
            parts_location = [regions.index(part) for part in parts]
            connectivity_matrix[parts_location[0], parts_location[1]] = df[df['BID']==ID][connection].values
            connectivity_matrix[parts_location[1], parts_location[0]] = df[df['BID']==ID][connection].values
        
        adj_mtx = (connectivity_matrix>threshold).astype(int)
        adj_mtx = adj_mtx + np.eye(adj_mtx.shape[0])
        corr_matrix_nx = nx.Graph(connectivity_matrix)
        corr_matrix_data = from_networkx(corr_matrix_nx)
        
        edge_index = torch.tensor(np.int16(adj_mtx)).nonzero().t().contiguous()
        # label=df[df['BID']==ID]['age'].item()
        label=df[df['BID']==ID]['madrs'].item()
        # label=df[df['BID']==ID]['sex'].item()
        
        corr_matrix_data.x = connectivity_matrix + np.eye(connectivity_matrix.shape[0])
        corr_matrix_data.y = label

        G = nx.Graph()
        G.add_edges_from([(i, j) for i in range(len(adj_mtx)) for j in range(len(adj_mtx[i])) if adj_mtx[i][j] == 1])
        degree_centrality = nx.degree_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter = 1000)
        katz_centrality = nx.katz_centrality_numpy(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        for idx in range(0, adj_mtx.shape[1]):
            eigenvector_centrality.setdefault(idx, 0)
            degree_centrality.setdefault(idx, 0)
            katz_centrality.setdefault(idx, 0)
            betweenness_centrality.setdefault(idx, 0)
            closeness_centrality.setdefault(idx, 0)

        list_of_dicts = [degree_centrality, eigenvector_centrality, katz_centrality, betweenness_centrality, closeness_centrality]
        feature_vector = np.array([[d[key] for key in sorted(d)] for d in list_of_dicts]).T
        if edge_index.max().detach().item()>=feature_vector.shape[0]:
            pdb.set_trace()
            
        if corr_matrix_data.edge_index.max().detach().item()>=(corr_matrix_data.x.shape[0] * corr_matrix_data.x.shape[1]):
            pdb.set_trace()        
        # return torch.tensor(feature_vector), torch.tensor(edge_index), torch.tensor(label)
        return corr_matrix_data


    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))
        return data
