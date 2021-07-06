import networkx as nx
import pandas as pd

NODES_NUMBER = 450


def add_nodes(graph):
    nodes_list = range(450)
    graph.add_nodes_from(nodes_list)
    return graph


def add_edges(edges_file):
    edges_df = pd.read_csv(edges_file, header=None)
    print()


def create_atn_network(nodes_file, edges_file):
    graphs_list = []
    for i in range(37):
        cur_graph = nx.Graph()
        cur_graph = add_nodes(cur_graph)
        graphs_list.append(cur_graph)


if __name__ == '__main__':
    edges_file = 'network.csv'
    add_edges(edges_file)
