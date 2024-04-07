import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import yaml
from yaml import SafeLoader
from tqdm import tqdm

import random
import argparse
import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse, degree, to_undirected
from model import Encoder, GRACE
from functional import drop_feature, drop_feature_weighted_2, feature_drop_weights
from utils import get_base_model, get_activation
from dataset import get_dataset
import ot
import torch.nn.functional as F
from torch_sparse import SparseTensor
import os
import wandb
from GCL.eval import get_split, LREvaluator
import numpy as np

def sparse_permute(M, Pi):
    M = SparseTensor.from_dense(M)
    Pi = SparseTensor.from_dense(Pi)
    inv_Pi = Pi.t()
    A_shuffle = inv_Pi @ M @ Pi
    return A_shuffle.to_dense()

def get_cost_matrix(cost_type='cos'):
    if cost_type == 'cos':
        num_nodes = data.x.shape[0]
        normalized_x = F.normalize(data.x, dim=1)
        cost = 1 - torch.mm(normalized_x, normalized_x.t())
        eye = torch.eye(num_nodes).to(device)
        mask = 1 - eye
        cost = cost * mask + 2 * eye
    elif cost_type == '1d':
        cost = torch.cdist(data.x, data.x, p=1)
    elif cost_type == '2d':
        cost = torch.cdist(data.x, data.x, p=2)
    return cost

def get_same_degree_matrix():
    degree_vec = data.degree_y.unsqueeze(0)
    eye = torch.eye(degree_vec.shape[1]).to(device)
    mask = (1 - eye).int()
    return mask * (degree_vec == degree_vec.T).int()

def train(model, optimizer):
    model.train()
    optimizer.zero_grad()

    def ot_augmentation(data, cost, keep_prob=0.5, distance='emd', reg_lambda=1e-3):
        # adj decomposition
        A = to_dense_adj(data.edge_index).squeeze(0)
        num_nodes = A.shape[0]
        num_ot = int(num_nodes * (1 - keep_prob))

        permutation_vector = torch.randperm(num_nodes)
        eye = torch.eye(num_nodes).to(device)
        permutation_matrix = eye[permutation_vector]
        inv_permutation_matrix = permutation_matrix.t()
        A_shuffle = inv_permutation_matrix @ A @ permutation_matrix
        cost_shuffle = inv_permutation_matrix @ cost @ permutation_matrix

        # optimal transport
        A_ot = A_shuffle[:num_ot, :num_ot]
        total_weight = torch.sum(A_ot)
        A_ot_nodes = A_ot.shape[0]
        A_ot = A_ot / total_weight
        ones = torch.ones(A_ot_nodes, 1).to(device)
        C = cost_shuffle[:num_ot, :num_ot]
        alpha = (A_ot @ ones).squeeze(-1)
        beta = (A_ot.t() @ ones).squeeze(-1)

        if distance == 'emd':
            P = ot.emd(alpha, beta, C, numItermax=100000, check_marginals=False)
        elif distance == 'sinkhorn':
            P = ot.sinkhorn(alpha, beta, C, reg=reg_lambda, numItermax=1000, method='sinkhorn')
        P = P * total_weight

        A_shuffle[:num_ot, :num_ot] = P
        A_hat = permutation_matrix @ A_shuffle @ inv_permutation_matrix
        
        return dense_to_sparse(A_hat)


    edge_index_1, edge_index_weight_1 = ot_augmentation(data, cost, param["keep_prob_1"], param["distance"], param["reg_lambda"] if param["distance"] == "sinkhorn" else None)
    edge_index_2, edge_index_weight_2 = ot_augmentation(data, cost, param["keep_prob_2"], param["distance"], param["reg_lambda"] if param["distance"] == "sinkhorn" else None)
    
    x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
    x_2 = drop_feature(data.x, param['drop_feature_rate_2'])
    if param['drop_scheme'] == 'degree':
        x_1 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_1'])
        x_2 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_2'])


       
    z1 = model(x_1, edge_index_1, edge_index_weight_1)
    z2 = model(x_2, edge_index_2, edge_index_weight_2)

    loss = model.loss(z1, z2)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data, split):
    model.eval()
    edge_index_weight = torch.ones_like(data.edge_index[0]).float()
    z = model(data.x, data.edge_index, edge_index_weight)

    result = LREvaluator()(z, data.y, split)
    return result


if __name__ == '__main__':
    print(os.path.basename(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    param = vars(args)
    config = yaml.load(open(param['config']), Loader=SafeLoader)[param['dataset']]
    for key in config:
        param[key] = config[key]

    print(param)
    torch.manual_seed(param['seed'])
    random.seed(param['seed'])
    device = torch.device(param['device'])
    wandb.init(config=args)

    micros = []
    macros = []
    for i in range(0, 5):
        data = get_dataset(path = "../data" ,name = param["dataset"])
        data = data.to(device)
        cost = get_cost_matrix(param["cost_type"])
        edge_index_ = to_undirected(data.edge_index)
        data.degree_y = degree(edge_index_[1])
        same_degree_pairs = get_same_degree_matrix()

        encoder = Encoder(data.x.shape[1], param['num_hidden'], get_activation(param['activation']),
                        base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
        model = GRACE(encoder, same_degree_pairs, param['num_hidden'], param['num_proj_hidden'], param['tau'], lambda_2=param['p']).to(device)
        
        if param['drop_scheme'] == 'degree':
            edge_index_ = to_undirected(data.edge_index)
            node_deg = degree(edge_index_[1], num_nodes=data.x.shape[0])
            feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=param['learning_rate'],
            weight_decay=param['weight_decay']
        )
        with tqdm(range(1, param['num_epochs'] + 1)) as pbar:
            for epoch in pbar:
                loss = train(model, optimizer)
                wandb.log({"loss":loss})
                pbar.set_description(f"Loss: {loss:.4f}")
                
        split = get_split(num_samples=data.x.shape[0], train_ratio=0.1, test_ratio=0.1)
        res = test(model, data, split)
        print("----micro_f1:{}----macro_f1:{}----".format(res['micro_f1'], res['macro_f1']))
        micros.append(res['micro_f1'])
        macros.append(res['macro_f1'])

    micros = np.array(micros)
    macros = np.array(macros)
    wandb.run.summary["micro_f1_mean"] = np.mean(micros)
    wandb.run.summary["micro_f1_std"] = np.std(micros)
    wandb.run.summary["macro_f1_mean"] = np.mean(macros)
    wandb.run.summary["macro_f1_std"] = np.std(macros)
