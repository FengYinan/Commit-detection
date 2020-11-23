import numpy as np
from glob import glob
import os


def load_npz(npz_path):
    npz_feature = ['node_list', 'graph_node_old', 'graph_node_new', 'graph_edge_old', 'graph_edge_new']
    file = np.load(npz_path)
    old_node = file[npz_feature[1]]
    new_node = file[npz_feature[2]]

    return old_node, new_node

def normanddel():
    data_list_new = glob('C:/Users/yinan/Desktop/data/old_feature/Commits/commit/Commits_Graph/*')
    data_list_susp = glob('C:/Users/yinan/Desktop/data/old_feature/GoogleDoc/Very_Suspicious/Commits_Graph/*')
    data_list_draft = glob('C:/Users/yinan/Desktop/data/old_feature/GoogleDoc/Commits_Draft/Commits_Graph/*')
    data_list_old = glob('C:/Users/yinan/Desktop/data/old_feature/Diffs_old_result/*/Commits_Graph/*')

    list = data_list_new + data_list_draft + data_list_old + data_list_susp
    o_data = np.zeros((1, 195))
    n_data = np.zeros((1, 195))
    for i in list:
        # print(i)
        o, n = load_npz(i)
        # print(o.shape)
        if np.sum(n) == np.sum(o) == 0:
            os.remove(i)
            continue
        if len(o.shape) != 2 or len(n.shape) != 2:
            os.remove(i)
        else:
            if o.shape[1] != 195:
                os.remove(i)
            else:
                o_data = np.concatenate((o_data, o), axis=0)
                n_data = np.concatenate((n_data, n), axis=0)

    o_data = o_data[1:]
    n_data = n_data[1:]

    o_mean = np.mean(o_data, axis=0)
    n_mean = np.mean(n_data, axis=0)
    o_std = np.std(o_data, axis=0)
    n_std = np.std(n_data, axis=0)

    print(o_mean.shape)

    np.savez_compressed('norm.npz', old_mean=o_mean, new_mean=n_mean, old_std=o_std, new_std=n_std)

