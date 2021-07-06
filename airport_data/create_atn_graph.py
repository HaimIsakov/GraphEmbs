import networkx as nx
import numpy as np
import pandas as pd

NODES_NUMBER = 450


def add_nodes(graph):
    nodes_list = range(1, 451)
    graph.add_nodes_from(nodes_list)
    return graph


def add_edges(graph, edges_file_reader):
    parse_layer(graph, edges_file_reader)
    return graph


def parse_layer(graph, edges_file_reader):
    number_of_active_nodes = edges_file_reader.readline().replace('\n', '')
    buffer = edges_file_reader.readline().replace('\n', '')  # for empty line
    buffer = edges_file_reader.readline().replace('\n', '')
    while buffer != '':
        split_buffer = [int(i) for i in buffer.split('\t')]
        node_id = split_buffer[0]
        degree = split_buffer[1]
        neighbors = split_buffer[2:]
        buffer = edges_file_reader.readline().replace('\n', '')
        edges_to_add = [(node_id, neighbor) for neighbor in neighbors]
        graph.add_edges_from(edges_to_add)
    return graph


def create_atn_network(nodes_file, edges_file):
    graphs_list = []
    with open(edges_file, "r") as edges_file_reader:
        for i in range(37):
            print(f"graph {i}")
            cur_graph = nx.Graph()
            cur_graph = add_nodes(cur_graph)
            cur_graph = add_edges(cur_graph, edges_file_reader)
            graphs_list.append(cur_graph)
    return graphs_list

#
# if __name__ == '__main__':
#     edges_file = 'network.csv'
#     nodes_file = 'airports.txt'
#     create_atn_network(nodes_file, edges_file)
