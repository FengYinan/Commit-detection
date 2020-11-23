import numpy as np
import os
from graph_generation2 import *
from generate_java_file import *
from glob import glob
import random

def generate_graph_overall(path_file, path_graph):
    # define the dir name
    dirname = []
    files = os.listdir(path_file)
    for file in files:
        m = os.path.join(path_file, file)
        if (os.path.isdir(m)):
            h = os.path.split(m)
            dirname.append(h[1])

    folder = os.path.exists(path_graph)
    if not folder:
        os.makedirs(path_graph)

    #random.shuffle(dirname)

    for dir_path in dirname:
        try:
            if os.path.exists(path_graph + dir_path + '.npz'):
                print('existing')
                continue
            path_java_old = path_file + dir_path + '/old.cpp'
            path_java_new = path_file + dir_path + '/new.cpp'
            old = os.path.exists(path_java_old)
            new = os.path.exists(path_java_new)
            if not old or not new:
                print('one of them does not existing')
                continue
            fsize = os.path.getsize(path_java_old)
            fsize2 = os.path.getsize(path_java_new)
            if fsize < 1024 or fsize2 < 1024:
                f = open(path_java_old, "r")
                lines = f.readlines()
                f.close()
                f = open(path_java_new, "r")
                lines2 = f.readlines()
                f.close()
                if '404: Not Found' in lines or '404: not found' in lines2:
                    print('404: not found #' + dir_path)
                    continue
            if fsize > 1024 * 1024 or fsize2 > 1024 * 1024:
                continue
            else:
                node_list_old, node_list_new, graph_node_old, graph_node_new, graph_edge_old, graph_edge_new = generate_two_graph(
                    path_java_old, path_java_new)
                print("Sucess")
                np.savez(path_graph + dir_path + '.npz', node_list_old=node_list_old, node_list_new=node_list_new,
                         graph_node_old=graph_node_old,
                         graph_node_new=graph_node_new, graph_edge_old=graph_edge_old,
                         graph_edge_new=graph_edge_new)
                print("successfully build graph #" + dir_path)
        except:
            print('404')





def download_java_code_generate_graph():
    path_list = ["C:/Users/yinan/Desktop/data/research_project_summary-master/Diffs_old/",
                 "C:/Users/yinan/Desktop/data/research_project_summary-master/Diffs_new/"]
    for path_original in path_list:
        main_test(path_original) ########已生成一部分
        path_dir = os.listdir(path_original)
        for path in path_dir:
            path_file = 'C:/Users/yinan/Desktop/python_code/' + path_original.split('/')[
                -2] + "_result_graph/" + path + '/Commits_File/'
            path_graph = 'C:/Users/yinan/Desktop/python_code/' + path_original.split('/')[
                -2] + "_result_graph/" + path + "/Commits_Graph/"
            generate_graph_overall(path_file, path_graph)


def generate_graph():
    path_list = ["D:/UVA_RESEARCH/COMMIT/data/old_feature/old2_feature/",
                 "D:/UVA_RESEARCH/COMMIT/data/old_feature/Commits/",
                 "D:/UVA_RESEARCH/COMMIT/data/old_feature/GoogleDoc/"] #
    for path_original in path_list:
        path_dir = os.listdir(path_original)
        for path in path_dir:
            if '.' in path:
                continue
            if 'mopub-android-sdk' in path:
                print(path)
            path_file = path_original + path + '/Commits_File/'
            path_graph = path_original.replace('old', 'new', 1) + path + "/Commits_Graph/"
            generate_graph_overall(path_file, path_graph)


def generate_graph_from_malicious():
    path_list = ["D:/UVA_RESEARCH/COMMIT/data/old_feature/malicious/malware_generated"] #
    for path_original in path_list:
        path_file = path_original + '/Commits_File/'
        path_graph = path_original + "/Commits_Graph/"
        generate_graph_overall(path_file, path_graph)

if __name__ == '__main__':
    generate_graph()