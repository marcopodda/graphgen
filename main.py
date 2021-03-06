import argparse
from pathlib import Path
import random
import time
import pickle
from torch.utils.data import DataLoader

from args import Args
from utils import create_dirs
from datasets.process_dataset import create_graphs
from datasets.preprocess import calc_max_prev_node, dfscodes_weights
from baselines.dgmg.data import DGMG_Dataset_from_file
from baselines.graph_rnn.data import Graph_Adj_Matrix_from_file
from graphgen.data import Graph_DFS_code_from_file
from model import create_model
from train import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--preprocess", action="store_true", default=False)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    epochs = args.epochs
    preprocess = args.preprocess


    args = Args(dataset_name, epochs)
    args = args.update_args()

    create_dirs(args)

    random.seed(123)

    dataset_path = Path('datasets') / dataset_name

    if preprocess:
        graphs = create_graphs(args)
        exit()

    graphs = pickle.load(open(dataset_path / "graphs" / "graphs.dat", 'rb'))
    splits = pickle.load(open(dataset_path / "graphs" / "splits.dat", 'rb'))
    tensors = pickle.load(open(dataset_path / "min_dfscode_tensors" / "min_dfscode_tensors.dat", 'rb'))

    # show graphs statistics
    print('Model:', args.note)
    print('Device:', args.device)
    print('Graph type:', args.graph_type)

    # Loading the feature map
    with open(dataset_path / "graphs" / 'map.dict', 'rb') as f:
        feature_map = pickle.load(f)

    print('Max number of nodes: {}'.format(feature_map['max_nodes']))
    print('Max number of edges: {}'.format(feature_map['max_edges']))
    print('Min number of nodes: {}'.format(feature_map['min_nodes']))
    print('Min number of edges: {}'.format(feature_map['min_edges']))
    print('Max degree of a node: {}'.format(feature_map['max_degree']))
    print('No. of node labels: {}'.format(len(feature_map['node_forward'])))
    print('No. of edge labels: {}'.format(len(feature_map['edge_forward'])))

    dataset_train = Graph_DFS_code_from_file(tensors, splits['train'])
    dataset_validate = Graph_DFS_code_from_file(tensors, splits['val'])

    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers)
    dataloader_validate = DataLoader(
        dataset_validate, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers)

    model = create_model(args, feature_map)

    train(args, dataloader_train, model, feature_map, dataloader_validate)
