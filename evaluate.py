import os
from pathlib import Path
import shutil
from statistics import mean
import torch
import argparse

from graphgen.train import predict_graphs as gen_graphs_dfscode_rnn
from baselines.graph_rnn.train import predict_graphs as gen_graphs_graph_rnn
from baselines.dgmg.train import predict_graphs as gen_graphs_dgmg
from utils import get_model_attribute, load_graphs, save_graphs

import metrics.stats

LINE_BREAK = '----------------------------------------------------------------------\n'


class ArgsEvaluate():
    def __init__(self, path):
        # Can manually select the device too
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model_path = path

        self.num_epochs = int(path.stem.split("_")[-1])

        # Whether to generate networkx format graphs for real datasets
        self.generate_graphs = True

        self.count = 2560
        self.batch_size = 32  # Must be a factor of count

        self.metric_eval_batch_size = 256

        # Specific DFScodeRNN
        self.max_num_edges = 50

        # Specific to GraphRNN
        self.min_num_node = 0
        self.max_num_node = 40

        self.train_args = get_model_attribute('saved_args', self.model_path, self.device)
        name = self.train_args.fname
        t = self.train_args.time

        self.graphs_save_path = Path('graphs')
        self.current_graphs_save_path = self.graphs_save_path / f"{name}_{t}" / f"{self.num_epochs}"


def patch_graph(graph):
    for u in graph.nodes():
        graph.nodes[u]['label'] = graph.nodes[u]['label'].split('-')[0]

    return graph


def generate_graphs(eval_args):
    """
    Generate graphs (networkx format) given a trained generative model
    and save them to a directory
    :param eval_args: ArgsEvaluate object
    """

    train_args = eval_args.train_args

    if train_args.note == 'GraphRNN':
        gen_graphs = gen_graphs_graph_rnn(eval_args)
    elif train_args.note == 'DFScodeRNN':
        gen_graphs = gen_graphs_dfscode_rnn(eval_args)
    elif train_args.note == 'DGMG':
        gen_graphs = gen_graphs_dgmg(eval_args)

    if os.path.isdir(eval_args.current_graphs_save_path):
        shutil.rmtree(eval_args.current_graphs_save_path)

    os.makedirs(eval_args.current_graphs_save_path)

    save_graphs(eval_args.current_graphs_save_path, gen_graphs)


def print_stats(
    node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref,
    edge_count_avg_pred, degree_mmd, clustering_mmd, orbit_mmd,
    nspdk_mmd, node_label_mmd, edge_label_mmd, node_label_and_degree
):
    print('Node count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(node_count_avg_ref), mean(node_count_avg_pred)))
    print('Edge count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(edge_count_avg_ref), mean(edge_count_avg_pred)))

    print('MMD Degree - {:.6f}, MMD Clustering - {:.6f}, MMD Orbits - {:.6f}'.format(
        mean(degree_mmd), mean(clustering_mmd), mean(orbit_mmd)))
    print('MMD NSPDK - {:.6f}'.format(mean(nspdk_mmd)))
    print('MMD Node label - {:.6f}, MMD Edge label - {:.6f}'.format(
        mean(node_label_mmd), mean(edge_label_mmd)
    ))
    print('MMD Joint Node label and degree - {:.6f}'.format(
        mean(node_label_and_degree)
    ))
    print(LINE_BREAK)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    eval_args = ArgsEvaluate(Path(args.path))
    train_args = eval_args.train_args

    graphs = generate_graphs(eval_args)
