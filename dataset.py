import os.path as osp

from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB, Amazon, Actor, Twitch, LastFMAsia
import torch_geometric.transforms as T
import pandas as pd
import networkx as nx
from torch_geometric.data.data import Data
import torch
import numpy as np

def get_label(data, box_num):
    label = data.y
    temp = sorted(label)
    box = len(label)//box_num
    
    new_label= []
    for i in label:
        if i <= temp[box]:
            new_label.append(0)
        elif i <= temp[box*2]:
            new_label.append(1)
        elif i <= temp[box*3]:
            new_label.append(2)
        elif i <= temp[box*4]:
            new_label.append(3)
        else:
            new_label.append(4)
    
    data.y = torch.from_numpy(np.array(new_label))
    return data

def get_dataset(path, name):
    assert name in ['Cora', 'Citeseer', 'Pubmed', 'Cornell', 'Texas', 'Wisconsin', 'Chameleon', 'Crocodile', 'Squirrel', 'Photo', 'Computers', 'Actor', 'EN', 'ES', 'LastFMAsia']
    
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        return Planetoid(osp.join(path, 'pyg-format', 'Planetoid'), name=name, transform=T.NormalizeFeatures())[0]
    elif name in ['Chameleon', 'Squirrel', 'Crocodile']:
        # return read_data(path, name)
        data = WikipediaNetwork(osp.join(path, 'pyg-format', 'wikipedia'), geom_gcn_preprocess =False, name=name, transform=T.NormalizeFeatures())[0]
        data = get_label(data, 5)
        return data
    elif name in ['Cornell', 'Texas', 'Wisconsin']:
        return WebKB(osp.join(path, 'pyg-format', 'Webkb'), name=name, transform=T.NormalizeFeatures())[0]
    elif name in ['Photo', 'Computers']:
        return Amazon(osp.join(path, 'pyg-format', 'Amazon'), name=name, transform=T.NormalizeFeatures())[0]
    elif name in ['Actor']:
        return Actor(osp.join(path, 'pyg-format', 'Actor'), transform=T.NormalizeFeatures())[0]
    elif name in ['EN', 'ES']:
        return Twitch(osp.join(path, 'pyg-format', 'Twitch'), name=name, transform=T.NormalizeFeatures())[0]
    elif name in ['LastFMAsia']:
        return LastFMAsia(osp.join(path, 'pyg-format', 'LastFMAsia'), transform=T.NormalizeFeatures())[0]
    

def read_data(path, datasetName):
    G = nx.read_adjlist(osp.join(path, 'ntx-format', datasetName, "{}_adjlist.txt".format(datasetName)),
                        delimiter=' ',
                        nodetype=int,
                        create_using=nx.DiGraph())
    G_label = pd.read_pickle(osp.join(path, 'ntx-format', datasetName, "{}_label.pickle".format(datasetName)))
    G_attr = pd.read_pickle(osp.join(path, 'ntx-format', datasetName, "{}_attr.pickle".format(datasetName)))

    edge_index = pd.DataFrame(G.edges(),columns = ['u','v']).values.T
    data = Data(x=torch.tensor(G_attr.drop('nodes',axis=1).values, dtype=torch.float), edge_index=torch.tensor(edge_index, dtype=torch.long), y=torch.tensor(list(G_label['label'])))
    return data

def get_path(base_path, name):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        return base_path
    else:
        return osp.join(base_path, name)
