#import tensorflow as tf
#from tensorflow.contrib import slim
from scipy import misc
import os, random
import numpy as np
import torch
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

##################################################################################
# Loss function
##################################################################################


def l1_loss(x, y, graph=True):
    if graph:
        loss = torch.mean(torch.abs(x - y))
    else:
        loss = torch.mean(torch.abs(x - y), dim=-1)

    return loss.to(DEVICE)


def l2_loss(x, y, graph=True, c=0):
    if graph:
        loss = torch.clamp(torch.sum(torch.pow(x-y, 2)) - c, min=0.)
    else:
        loss = torch.clamp(torch.sum(torch.pow(x-y, 2), dim=-1) - c, min=0.)

    return loss.to(DEVICE)


def l21_loss(x, y, graph=True):
    if graph:
        loss = torch.sum(torch.sqrt(torch.sum(torch.pow(x-y, 2),dim=-1)))
    else:
        loss = torch.sqrt(torch.sum(torch.pow(x-y, 2),dim=-1))

    return loss.to(DEVICE)


def cross_entropy(output, lable, graph=True):
    if graph:
        loss = torch.mean(-1 * lable * torch.log(output + 1e-15) - (1-lable) * torch.log(1-output + 1e-15))
    else:
        loss = torch.mean(-1 * lable * torch.log(output + 1e-15) - (1-lable) * torch.log(1-output + 1e-15), dim=-1)

    return  loss.to(DEVICE)


def pca_analysis(x, task_name, c_names, label=None):
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)

    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    if label is None:
        db = KMeans(3).fit(principalComponents)

        k = [0, 0, 0]
        for i in range(len(c_names)):
            if db.labels_[i] == 0:
                #print("{}:  {}".format(c_names[i], db.labels_[i]))
                k[0] += 1
            if db.labels_[i] == 1:
                #print("{}:  {}".format(c_names[i], db.labels_[i]))
                k[1] += 1
            if db.labels_[i] == 2:
                #print("{}:  {}".format(c_names[i], db.labels_[i]))
                k[2] += 1
        print(k)
        label = map(int, db.labels_)


    finalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])
    #print(finalDf)
    commit_name = pd.DataFrame(data=c_names
                         , columns=['name'])
    cluster_name = pd.DataFrame(data=label
                               , columns=['label'])
    SaveDF = pd.concat([finalDf, commit_name, cluster_name], axis=1)
    SaveDF.to_excel("{}_output.xlsx".format(task_name))

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_xlabel('Principal Component 1', fontsize=15)
    # ax.set_ylabel('Principal Component 2', fontsize=15)
    # ax.set_title('2 Component PCA: {}'.format(task_name), fontsize=20)
    #
    # ax.scatter(finalDf.loc[:, 'principal component 1']
    #            , finalDf.loc[:, 'principal component 2']
    #            , c='r'
    #            , s=50)
    #
    # ax.grid()
    # plt.show()


    fig = px.scatter(SaveDF, x='principal component 1', y='principal component 2', color='label',
                     hover_name='name')
    fig.update_layout(title={'text': '2 Component PCA: {}'.format(task_name), 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                      xaxis_title="Principal Component 1", yaxis_title="Principal Component 2")
    fig.show()


def softhd(v1, sz1, v2, sz2):
    byy = v2.unsqueeze(1).expand((v2.size(0), v1.size(1), v2.size(1), v2.size(2))).transpose(1, 2)
    bxx = v1.unsqueeze(1).expand_as(byy)

    bdxy = torch.sqrt(torch.sum((bxx - byy) ** 2, 3))

    # Create a mask for nodes
    node_mask2 = torch.arange(0, bdxy.size(1)).unsqueeze(0).unsqueeze(-1).expand(bdxy.size(0),
                                                                                 bdxy.size(1),
                                                                                 bdxy.size(2)).long()
    node_mask1 = torch.arange(0, bdxy.size(2)).unsqueeze(0).unsqueeze(0).expand(bdxy.size(0),
                                                                                bdxy.size(1),
                                                                                bdxy.size(2)).long()

    if v1.is_cuda:
        node_mask1 = node_mask1.cuda()
        node_mask2 = node_mask2.cuda()
    node_mask1 = Variable(node_mask1, requires_grad=False)
    node_mask2 = Variable(node_mask2, requires_grad=False)
    node_mask1 = (node_mask1 >= sz1.unsqueeze(-1).unsqueeze(-1).expand_as(node_mask1))
    node_mask2 = (node_mask2 >= sz2.unsqueeze(-1).unsqueeze(-1).expand_as(node_mask2))

    node_mask = node_mask1 | node_mask2

    maximum = bdxy.max()

    bdxy.masked_fill_(node_mask, float(maximum))

    bm1, _ = bdxy.min(dim=2)
    bm2, _ = bdxy.min(dim=1)

    bm1.masked_fill_(node_mask.prod(dim=2), 0)
    bm2.masked_fill_(node_mask.prod(dim=1), 0)

    d = bm1.sum(dim=1) + bm2.sum(dim=1)

    return d / (sz1.float() + sz2.float())


def adj_to_bias(adj, nhood=1):
    mt = torch.eye(adj.shape[0]).to(DEVICE)
    for _ in range(nhood):
        mt = torch.matmul(mt, (adj + torch.eye(adj.shape[0]).to(DEVICE)))
    mt = torch.clamp(mt, max=1.0)

    return mt

def tsne_analysis(x, task_name, c_names, label=None):
    x = StandardScaler().fit_transform(x)

    pca = TSNE(n_components=2, verbose=0, perplexity=50, n_iter=1000, init='pca', learning_rate=13, random_state=0)
    principalComponents = pca.fit_transform(x)


    if label is None:
        db = DBSCAN(eps=1, min_samples=50).fit(principalComponents)
        label = map(int, db.labels_)


    finalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])
    #print(finalDf)
    commit_name = pd.DataFrame(data=c_names
                         , columns=['name'])
    cluster_name = pd.DataFrame(data=label
                               , columns=['label'])
    SaveDF = pd.concat([finalDf, commit_name, cluster_name], axis=1)

    fig = px.scatter(SaveDF, x='principal component 1', y='principal component 2', color='label',
                     hover_name='name')
    fig.update_layout(title={'text': '2 Component T-sne: {}'.format(task_name), 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                      xaxis_title="Component 1", yaxis_title="Component 2")
    fig.show()


def h_embedding(a, b):
    tem_attention = torch.matmul(a - a.mean(), (b - b.mean()).T) / torch.clamp(
        torch.sqrt(torch.sum(torch.pow(a - a.mean(), 2))) * torch.sqrt(torch.sum(torch.pow(a - a.mean(), 2))), min=1e-8)
    tem_attention = tem_attention.T

    value, indicate = torch.min(torch.abs(tem_attention), dim=-1)
    value2, indicate2 = torch.max(value, dim=0)
    return value2, b[indicate[indicate2]]

if __name__ == '__main__':
    data = np.load('new_embed_1.npz')
    x = data['ed']
    cl = data['cl']
    label = data['label']
    pca_analysis(x, '1', cl)
    # data = np.load('new_embed_3.npz')
    # x = data['ed']
    # cl = data['cl']
    # l = data['label']
    # tsne_analysis(x, '2', cl, l)
    # x = StandardScaler().fit_transform(x)
    # db = DBSCAN(eps=13, min_samples=20).fit(x)
    # a = 0
    # for i in range(len(cl)):
    #     if db.labels_[i] != -1:
    #         continue
    #     print(cl[i])
    #     print(db.labels_[i])
    #     a+=1
    # print(a)