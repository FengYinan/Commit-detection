import numpy as np
from java_keywords import keywords
import re
from remove_comments import *
from process_java3 import *

# generate the graph
"""
After we obtain the node of new and old java file, we can generate the graphs.
For each node, we assign the name a hash value, and we can generate:
node_hash = {key = hash: value = node_name}
graph_node: a matrix, which node[i,:] represents the feature vector of node[i]
node_list: a list contains the order of node's hash value and name
node_graph[i,j] means whether node[i] calls node[j]
"""

# first, generate node_hash and node_list


def generate_node_hash_list(node_hash_old, node_hash_new):
    # obtain the union of node_hash_new and node_hash old
    node_hash = {**node_hash_old, **node_hash_new}
    hash_list = sorted(node_hash.keys())
    node_list = []
    for hash_value in hash_list:
        node_list.append(node_hash[hash_value])
    return node_hash, node_list

# now we have node_hash and node_list, we can generate the node_feature


def generate_graph_node(node_list_old, node_list_new, node_feature_old, node_feature_new):
    #len_feature = node_feature_new[0].shape
    graph_node_old = list(node_feature_old.values())
    graph_node_new = list(node_feature_new.values())

    graph_node_old = np.array(graph_node_old)
    # convert it to numpy format
    graph_node_new = np.array(graph_node_new)

    # if graph_node_old.shape[0] == 0:
    #     graph_node_old = np.zeros(len_feature, dtype=float)
    # if graph_node_new.shape[0] == 0:
    #     graph_node_new = np.zeros(len_feature, dtype=float)
    return graph_node_old, graph_node_new


"""
graph_edge[i,j] means whether node[i] calls node[j].
Input is the node_list, node_line and node_feature
Output is the graph_edge
"""


def generate_graph_edge(node_list, node_line, node_feature, codes):
    node_length = len(node_list)
    graph_edge = np.zeros((node_length, node_length), dtype=float)
    # generate the graph_edge matrix
    for i in range(node_length):
        # find the node_name
        # node_i = node_list[i]
        for line in codes[node_line[i][1]: node_line[i][2]]:
            # enter the data segment of method/class
            for j in range(node_length):
                node_j = node_list[j]
                if node_j in line:
                    graph_edge[i, j] = 1
    return graph_edge


def generate_two_graph(path_old, path_new):
    # path_old = "E:/1-Android/Graph-Generated/java_source/old_AuthorizationClient.java"
    # path_new = "E:/1-Android/Graph-Generated/java_source/AuthorizationClient.java"
    # define the file path, this can be improved by following work

    # load the codes from old file path

    codes_old = remove_comment(path_old)  # remove the comments in the java file
    # find the methods
    print(path_old)

    node_line_old, node_list_old = find_node_name_location(codes_old)
    # determine the node features
    node_feature_old = find_node_feature(codes_old, node_line_old)

    # load the codes from new file path
    codes_new = remove_comment(path_new)  # remove the comments in the java file
    # find the methods
    node_line_new, node_list_new = find_node_name_location(codes_new)
    # determine the node features
    node_feature_new = find_node_feature(codes_new, node_line_new)

    """
    After we obtain the node of new and old java file, we can generate the graphs.
    For each node, we assign the name a hash value, and we can generate:
    node_hash = {key = hash: value = node_name}
    node_list: a list contains the order of node's hash value and name
    graph_edge[i,j] means whether node[i] calls node[j]
    """
    graph_node_old, graph_node_new = generate_graph_node(node_list_old, node_list_new, node_feature_old, node_feature_new)
    # generate the graph edge for old file
    graph_edge_old = generate_graph_edge(node_list_old, node_line_old, node_feature_old, codes_old)
    # generate the graph edge for new file
    graph_edge_new = generate_graph_edge(node_list_new, node_line_new, node_feature_new, codes_new)
    return node_list_old, node_list_new, graph_node_old, graph_node_new, graph_edge_old, graph_edge_new


if __name__ == '__main__':
    path_old = "E:/1-Android/Graph-Generated/java_source/old_AuthorizationClient.java"
    path_new = "E:/1-Android/Graph-Generated/java_source/AuthorizationClient.java"
    node_hash, node_list, graph_node_old, graph_node_new, graph_edge_old, graph_edge_new = generate_two_graph(path_old, path_new)
