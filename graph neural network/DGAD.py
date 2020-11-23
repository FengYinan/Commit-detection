from net import *
from utils import *
from Logger import *
from trans_graph import *

import time
import datetime
import sys
import gc
import math
import os
import pandas as pd
from glob import glob
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import DBSCAN


class DGAD(object) :
    def __init__(self, args):
        self.model_name = 'JAVA_Graph_Conv_AE'
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir

        self.dataset_name = args.dataset

        self.new_start = args.new_start

        self.epoch = args.epoch
        self.ftepoch = args.ftepoch
        #self.iteration = args.iteration##
        self.resume_iters = args.resume_iters
        self.denoising = args.denoising
        self.dropout = args.dropout
        self.alpha = args.alpha

        self.loss_function = eval(args.loss_function)
        self.ax_w = args.ax_w
        self.decay_flag = args.decay_flag
        self.decay_epoch = args.decay_epoch
        self.init_lr = args.lr

        self.print_net = args.print_net

        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.use_tensorboard = args.use_tensorboard

        self.batch_size = args.batch_size
        self.graph_ch = args.dataset_setting[self.dataset_name][0]
        self.conv_ch = args.dataset_setting[self.dataset_name][1] if args.conv_ch ==0 else args.conv_ch

        self.npz_feature = ['node_list_new', 'graph_node_old', 'graph_node_new', 'graph_edge_old', 'graph_edge_new']
        if self.dataset_name == 'vccfinder':
            self.train_list = glob("D:/UVA_RESEARCH/COMMIT/data/VCCFinder/*/Commits_Graph/*")
            self.test_list = glob("D:/UVA_RESEARCH/COMMIT/data/VCCFinder/*/Commits_Graph/*")
            self.ft_list_bad = []
            self.ft_list_good = []
        else:
            self.data_list_new = glob('D:/UVA_RESEARCH/COMMIT/data/new_feature/Commits/commit/Commits_Graph/*')
            self.data_list_susp = glob('D:/UVA_RESEARCH/COMMIT/data/new_feature/GoogleDoc/new_su_file/Commits_Graph/*')
            self.data_list_draft = glob(
                'D:/UVA_RESEARCH/COMMIT/data/new_feature/GoogleDoc/Commits_Draft/Commits_Graph/*')
            self.data_list_old = glob('D:/UVA_RESEARCH/COMMIT/data/new_feature/old2_feature/*/Commits_Graph/*')
            self.data_malware_generated = glob(
                'D:/UVA_RESEARCH/COMMIT/data/new_feature/malicious/malware_generated/Commits_Graph/*')
            self.data_bug = glob("D:/UVA_RESEARCH/COMMIT/data/bug/Commits_Graph/*")
            self.train_list, self.test_list, self.ft_list_bad, self.ft_list_good = self.generate_train_test_list()
        self.train_len = len(self.train_list)
        self.test_len = len(self.test_list)
        #self.load_norm()


        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # build graph
        print(" [*] Buliding model!")
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        print("##### Information #####")
        print("# loss function: ", args.loss_function)
        print("# dataset : ", self.dataset_name)
        #print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# train length : ", self.train_len)
        print("# test length : ", self.test_len)

        #torch.autograd.set_detect_anomaly(True)

    def generate_train_test_list(self):
        ft_list_bad = self.data_list_susp.copy()
        ft_list_bad.append(self.data_list_draft[-1])

        good_len = -3 * len(ft_list_bad)

        train_list = self.data_list_new[:150] + self.data_list_new[350:] + self.data_list_old[:good_len] + self.data_list_draft[:-1]
        ft_list_good = self.data_list_old[good_len:]

        test_list = self.data_list_new + self.data_list_susp + self.data_list_draft + self.data_list_old[:6000] + self.data_malware_generated + self.data_bug

        return train_list, test_list, ft_list_bad, ft_list_good

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        if self.resume_iters:
            checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
            print('Loading the trained models from step {}...'.format(resume_iters))
            G_path = os.path.join(checkpoint_dir, '{}-G.ckpt'.format(resume_iters))
            self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

    def from_pretrain(self, resume_iters):
        self.E1 = encoder(self.graph_ch, self.conv_ch, dropout=self.dropout, alpha=self.alpha)
        self.E2 = encoder(self.graph_ch, self.conv_ch, dropout=self.dropout, alpha=self.alpha)

        self.E1_optimizer = torch.optim.Adam(self.E1.parameters(), self.init_lr)
        self.E2_optimizer = torch.optim.Adam(self.E2.parameters(), self.init_lr)

        self.E1.to(self.device)
        self.E2.to(self.device)

        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(checkpoint_dir, '{}-G.ckpt'.format(resume_iters))

        pretrained_dict = torch.load(G_path, map_location=lambda storage, loc: storage)
        model_dict = self.E1.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.E1.load_state_dict(model_dict)

        model_dict = self.E2.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.E2.load_state_dict(model_dict)

    def save(self, save_dir, counter):
        self.model_save_dir = os.path.join(save_dir, self.model_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(counter + 1))
        torch.save(self.G.state_dict(), G_path)

        print('Saved model {} checkpoints into {}...'.format(counter+1, self.model_save_dir))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = Logger(self.log_dir)

    def update_lr(self, lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def build_model(self):
        self.G = autoencoder(self.graph_ch, self.conv_ch, dropout=self.dropout, alpha=self.alpha)

        if self.print_net:
            self.print_network(self.G, 'G')

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.init_lr)

        self.G.to(self.device)

    def load_norm(self):
        file = np.load('D:/UVA_RESEARCH/COMMIT/data/old_feature/norm.npz')
        self.old_mean = file['old_mean']
        self.new_mean = file['new_mean']
        self.old_std = file['old_std']
        self.new_std = file['new_std']

    def load_npz(self, npz_path):
        file = np.load(npz_path)
        old_node = file[self.npz_feature[1]] #- self.old_mean) / (self.old_std + 1e-6)
        new_node = file[self.npz_feature[2]] #- self.new_mean) / (self.new_std + 1e-6)
        old_edge = file[self.npz_feature[3]]
        new_edge = file[self.npz_feature[4]]

        return torch.from_numpy(old_node).float().to(self.device), torch.from_numpy(old_edge).float().to(
            self.device), torch.from_numpy(
            new_node).float().to(self.device), torch.from_numpy(new_edge).float().to(self.device)

    def train(self):
        start_iters = self.resume_iters if not self.new_start else 0
        self.restore_model(self.resume_iters)

        self.iteration = self.train_len

        start_epoch = (int)(start_iters / self.iteration)
        start_batch_id = start_iters - start_epoch * self.iteration

        # loop for epoch
        start_time = time.time()
        lr = self.init_lr

        self.set_requires_grad([self.G], True)

        self.G.train()

        loss = {}
        loss['Edge_reconstruction_error_old'] = 0
        loss['Feature_reconstruction_error_old'] = 0
        loss['Reconstruction_error_old'] = 0
        loss['Edge_reconstruction_error_new'] = 0
        loss['Feature_reconstruction_error_new'] = 0
        loss['Reconstruction_error_new'] = 0
        loss['Reconstruction_error'] = 0

        for epoch in range(start_epoch, self.epoch):
            random.shuffle(self.train_list)
            if self.decay_flag and epoch > self.decay_epoch:
                lr /= 10#self.init_lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch) # linear decay
                self.update_lr(lr)

            for idx in range(start_batch_id, self.iteration):
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                commit_name = self.train_list[idx]
                old_node, old_edge, new_node, new_edge = self.load_npz(commit_name)
                if old_node.shape[0] == 0 :#or torch.sum(new_node) == 0
                    start_iters += 1
                    continue

                old_edge += torch.eye(old_edge.shape[0]).to(self.device)
                new_edge += torch.eye(new_edge.shape[0]).to(self.device)

                # old_edge = adj_to_bias(old_edge, 2)
                # new_edge = adj_to_bias(new_edge, 2)

                # =================================================================================== #
                #                             2. Train the Auto-encoder                              #
                # =================================================================================== #
                try:
                    old_node = F.dropout(old_node, self.denoising)
                    old_edge = F.dropout(old_edge, self.denoising)
                    old_recon_a, old_recon_x, _ = self.G(old_node, old_edge)

                    recon_a_error_old = self.loss_function(old_recon_a, old_edge)
                    recon_x_error_old = self.loss_function(old_recon_x, old_node)
                    Reconstruction_error_old = (self.ax_w * recon_a_error_old
                                                + (1 - self.ax_w) * recon_x_error_old)

                    # Logging.
                    loss['Edge_reconstruction_error_old'] += recon_a_error_old.item()
                    loss['Feature_reconstruction_error_old'] += recon_x_error_old.item()
                    loss['Reconstruction_error_old'] += Reconstruction_error_old.item()

                    # new graph
                    new_node = F.dropout(new_node, self.denoising)
                    new_edge = F.dropout(new_edge, self.denoising)
                    new_recon_a, new_recon_x, _ = self.G(new_node, new_edge)

                    recon_a_error_new = self.loss_function(new_recon_a, new_edge)
                    recon_x_error_new = self.loss_function(new_recon_x, new_node)
                    Reconstruction_error_new = (self.ax_w * recon_a_error_new
                                                + (1 - self.ax_w) * recon_x_error_new)

                    # Logging.
                    loss['Edge_reconstruction_error_new'] += recon_a_error_new.item()
                    loss['Feature_reconstruction_error_new'] += recon_x_error_new.item()
                    # loss['G/loss_cycle'] = self.cycle_loss.item()
                    loss['Reconstruction_error_new'] += Reconstruction_error_new.item()

                    Reconstruction_Error = 0.5 * (Reconstruction_error_old + Reconstruction_error_new)
                    loss['Reconstruction_error'] += Reconstruction_Error.item()

                    if (Reconstruction_error_new > 100 or Reconstruction_error_old > 100) and epoch > 0:
                        print(commit_name)

                    self.reset_grad()
                    Reconstruction_Error.backward()
                    self.g_optimizer.step()

                    del recon_a_error_new
                    del recon_x_error_new
                    del recon_a_error_old
                    del recon_x_error_old
                    del Reconstruction_error_new
                    del Reconstruction_error_old
                    del Reconstruction_Error
                    torch.cuda.empty_cache()

                    # =================================================================================== #
                    #                                 4. Miscellaneous                                    #
                    # =================================================================================== #

                    # Print out training information.
                    if start_iters % self.print_freq == 0 and start_iters != 0:
                        et = time.time() - start_time
                        et = str(datetime.timedelta(seconds=et))[:-7]
                        log = "Elapsed [{}], Epoch [{}/{}], Iteration [{}/{}]".format(et, epoch + 1, self.epoch,
                                                                                      idx + 1, self.iteration)
                        for tag, value in loss.items():
                            if 'error' in tag:  # != 'G/lable' and tag !='O/lable':
                                log += ", {}: {:.4f}".format(tag, value / (start_iters + 1))
                                if self.use_tensorboard:
                                    self.logger.scalar_summary(tag, value / (start_iters + 1), start_iters)
                        print(log)

                    start_iters += 1

                    # Save model checkpoints.
                    if start_iters % self.save_freq == 0 and start_iters != 0:
                        self.save(self.checkpoint_dir, start_iters)
                        torch.cuda.empty_cache()
                except:
                    start_iters += 1
                    print('CUDA out of memmory')
                    continue


            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, start_iters)
            self.resume_iters = start_iters + 1


            torch.cuda.empty_cache()

    def fintuning(self):
        self.from_pretrain(self.resume_iters)

        self.ft_list = []
        for g in self.ft_list_good:
            self.ft_list.append([g,1.])
        for b in self.ft_list_bad:
            self.ft_list.append([b,-1.])

        self.iteration = len(self.ft_list)

        # loop for epoch
        start_time = time.time()
        lr = self.init_lr

        self.set_requires_grad([self.E1], True)
        self.E1.train()
        self.set_requires_grad([self.E2], True)
        self.E2.train()

        for epoch in range(self.ftepoch):
            #random.shuffle(self.ft_list)
            if self.decay_flag and epoch > self.decay_epoch:
                lr = self.init_lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch)  # linear decay
                self.update_lr(lr)

            for idx in range(self.iteration):
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                commit_name, label = self.ft_list[idx]
                old_node, old_edge, new_node, new_edge = self.load_npz(commit_name)
                if old_node.shape[0] == 0: #or torch.sum(new_node) ==
                    continue

                loss = {}
                # =================================================================================== #
                #                             2. Train the Auto-encoder                              #
                # =================================================================================== #
                # old graph
                old_z, _ = self.E1(old_node, old_edge)
                new_z, _ = self.E2(new_node, new_edge)

                old_z = torch.mean(old_z, dim=0).unsqueeze(0)
                new_z = torch.mean(new_z, dim=0).unsqueeze(0)

                #similarity = (F.cosine_similarity(old_z,new_z) - label) ** 2
                similarity = F.cosine_embedding_loss(old_z,new_z,target=torch.tensor(label).to(self.device))

                # Logging.
                loss['Similarity'] = similarity.item()

                self.E1_optimizer.zero_grad()
                self.E2_optimizer.zero_grad()
                similarity.backward()
                self.E1_optimizer.step()
                self.E2_optimizer.step()

                del similarity
                torch.cuda.empty_cache()

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #
                # Print out training information.
                if idx % self.print_freq == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Epoch [{}/{}], Iteration [{}/{}]".format(et, epoch + 1, self.epoch, idx + 1,
                                                                                  self.iteration)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model

            torch.cuda.empty_cache()


        self.E1.eval()
        self.E2.eval()
        self.set_requires_grad([self.E1], False)
        self.set_requires_grad([self.E2], False)

        with torch.no_grad():
            similarity_list = []
            commit_list = []

            for idx in range(self.test_len):
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                commit_name = self.test_list[idx]
                old_node, old_edge, new_node, new_edge = self.load_npz(commit_name)
                if old_node.shape[0] == 0 or torch.sum(new_node) == 0:
                    continue

                # =================================================================================== #
                #                             2. Train the Auto-encoder                              #
                # =================================================================================== #
                old_z, _ = self.E1(old_node, old_edge)
                new_z, _ = self.E2(new_node, new_edge)

                old_z = torch.mean(old_z, dim=0).unsqueeze(0)
                new_z = torch.mean(new_z, dim=0).unsqueeze(0)

                similarity = 1 - F.cosine_similarity(old_z, new_z)

                similarity_list.append(similarity.cpu().detach())
                commit_list.append(commit_name)

            similarity_list = torch.tensor(similarity_list)
            _, indicates = torch.topk(similarity_list, k=5, dim=-1)
            print("Similarity Top K")
            for i in indicates:
                print("{}:  {}".format(commit_list[i], similarity_list[i]))
            print("Finish! \n")

    @property
    def model_dir(self):
        return "{}_{}".format(self.model_name, self.dataset_name)

    def test(self):
        self.restore_model(self.resume_iters)

        self.G.eval()
        self.set_requires_grad(self.G, False)

        #self.device = torch.device('cpu')
        #self.G.to(self.device)

        self.iteration = self.test_len
        self.model_save_dir = os.path.join(self.checkpoint_dir, self.model_dir)

        with torch.no_grad():
            loss = []
            new_embedding = []
            embedding_different = []
            dissimilarity = []
            commit_list = []
            label_list = []
            cosin_mean_different = []


            for idx in range(self.iteration):
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                commit_name = self.test_list[idx]

                old_node, old_edge, new_node, new_edge = self.load_npz(commit_name)
                if old_node.shape[0] == 0 or torch.sum(new_node) == 0:
                    continue

                # if old_edge.shape[0] < 100 and new_edge.shape[0] < 100:
                #     continue

                #old_edge = torch.clamp(old_edge + old_edge.T, max=1) + torch.eye(old_edge.shape[0]).to(self.device)
                #new_edge = torch.clamp(new_edge + new_edge.T, max=1) + torch.eye(new_edge.shape[1]).to(self.device)



                old_edge += torch.eye(old_edge.shape[0]).to(self.device)
                new_edge += torch.eye(new_edge.shape[0]).to(self.device)

                # old_edge = adj_to_bias(old_edge, 2)
                # new_edge = adj_to_bias(new_edge, 2)

                try:
                    # =================================================================================== #
                    #                             2. Train the Auto-encoder                              #
                    # =================================================================================== #
                    # old graph
                    old_node = F.dropout(old_node, self.denoising)
                    old_edge = F.dropout(old_edge, self.denoising)
                    _, _, old_z = self.G(old_node, old_edge)

                    # new graph
                    new_node = F.dropout(new_node, self.denoising)
                    new_edge = F.dropout(new_edge, self.denoising)
                    new_recon_a, new_recon_x, new_z = self.G(new_node, new_edge)

                    recon_a_error_new = self.loss_function(new_recon_a, new_edge)
                    recon_x_error_new = self.loss_function(new_recon_x, new_node)
                    Reconstruction_error_new = (self.ax_w * recon_a_error_new
                                                + (1 - self.ax_w) * recon_x_error_new)

                    #h_different_value, h_embedding_different = h_embedding(old_z, new_z)

                    old_z, _ = torch.max(old_z, dim=0)
                    new_z, _ = torch.max(new_z, dim=0)

                    loss.append(Reconstruction_error_new.cpu().detach())
                    dissimilarity.append(F.cosine_embedding_loss(old_z.cpu().unsqueeze(0), new_z.cpu().unsqueeze(0),
                                                                 target=torch.tensor(1).float().cpu().detach()))
                    embedding_different.append((new_z - old_z).cpu().detach().numpy())
                    # dissimilarity.append(h_different_value.float().cpu().detach())
                    # embedding_different.append(h_embedding_different.cpu().detach().numpy())
                    new_embedding.append(new_z.cpu().detach().numpy())
                    commit_list.append(commit_name)
                    if self.dataset_name == 'vccfinder':
                        if 'blamed' in commit_name:
                            label_list.append(0)
                        elif 'fixing' in commit_name:
                            label_list.append(2)
                        else:
                            label_list.append(1)
                    else:
                        if 'malware_generated' in commit_name:
                            label_list.append(0)
                        elif 'bug' in commit_name:
                            label_list.append(2)
                        else:
                            label_list.append(1)
                except:
                    print('out of memory')
                    continue

            loss = torch.tensor(loss)
            dissimilarity = torch.mean(torch.tensor(embedding_different), dim=-1)

            new_embedding = np.asarray(new_embedding)
            embedding_different = np.asarray(embedding_different)
            commit_list = np.asarray(commit_list)

            # Reconstruction error Top K
            _, indicates = torch.topk(loss, k=25, dim=-1)
            print("Reconstruction error Top K")
            for i in indicates:
                print("{}:  {}".format(commit_list[i], loss[i]))
            print("Finish! \n")

            # for i in range(len(self.data_malware_generated)):
            #     print("{}:  {}".format(commit_list[-(i+1)], loss[-(i+1)]))
            # print("Finish! \n")

            # Dissimilarity Top K
            _, indicates = torch.topk(dissimilarity, k=25, dim=-1)
            print("Dissimilarity Top K")
            for i in indicates:
                print("{}:  {}".format(commit_list[i], dissimilarity[i]))
            print("Finish! \n")

            # for i in range(len(self.data_malware_generated)):
            #     print("{}:  {}".format(commit_list[-(i+1)], dissimilarity[-(i+1)]))
            # print("Finish! \n")

            np.savez('new_embed_2',ed = embedding_different, cl=commit_list)


            # db = DBSCAN(eps=0.001, min_samples=5).fit(embedding_different)
            # for i in range(len(commit_list)):
            #     if db.labels_[i] != 0:
            #         print("{}:  {}".format(commit_list[i], db.labels_[i]))
            print("Finish! \n")

            # New Emending PCA
            pca_analysis(new_embedding, 'New Embedding', commit_list, label_list)

            # Emending different PCA
            pca_analysis(embedding_different, 'Embedding Different', commit_list, label_list)

            torch.cuda.empty_cache()

        # with torch.no_grad():
        #     loss = []
        #     new_embedding = []
        #     embedding_different = []
        #     dissimilarity = []
        #     commit_list = []
        #     label_list = []
        #     cosin_mean_different = []
        #
        #     for idx in range(self.iteration):
        #         # =================================================================================== #
        #         #                             1. Preprocess input data                                #
        #         # =================================================================================== #
        #         commit_name = self.test_list[idx]
        #
        #         old_node, old_edge, new_node, new_edge = self.load_npz(commit_name)
        #         if old_node.shape[0] == 0 or torch.sum(new_node) == 0:
        #             continue
        #
        #
        #         if old_edge.shape[0] > 100 or new_edge.shape[0] > 100:
        #             continue
        #
        #         # old_edge = torch.clamp(old_edge + old_edge.T, max=1) + torch.eye(old_edge.shape[0]).to(self.device)
        #         # new_edge = torch.clamp(new_edge + new_edge.T, max=1) + torch.eye(new_edge.shape[1]).to(self.device)
        #
        #         old_edge += torch.eye(old_edge.shape[0]).to(self.device)
        #         new_edge += torch.eye(new_edge.shape[0]).to(self.device)
        #
        #         # old_edge = adj_to_bias(old_edge, 2)
        #         # new_edge = adj_to_bias(new_edge, 2)
        #
        #         try:
        #             # =================================================================================== #
        #             #                             2. Train the Auto-encoder                              #
        #             # =================================================================================== #
        #             # old graph
        #             old_node = F.dropout(old_node, self.denoising)
        #             old_edge = F.dropout(old_edge, self.denoising)
        #             _, _, old_z = self.G(old_node, old_edge)
        #
        #             # new graph
        #             new_node = F.dropout(new_node, self.denoising)
        #             new_edge = F.dropout(new_edge, self.denoising)
        #             new_recon_a, new_recon_x, new_z = self.G(new_node, new_edge)
        #
        #             recon_a_error_new = self.loss_function(new_recon_a, new_edge)
        #             recon_x_error_new = self.loss_function(new_recon_x, new_node)
        #             Reconstruction_error_new = (self.ax_w * recon_a_error_new
        #                                         + (1 - self.ax_w) * recon_x_error_new)
        #
        #             # h_different_value, h_embedding_different = h_embedding(old_z, new_z)
        #
        #             old_z, _ = torch.max(old_z, dim=0)
        #             new_z, _ = torch.max(new_z, dim=0)
        #
        #             loss.append(Reconstruction_error_new.cpu().detach())
        #             dissimilarity.append(F.cosine_embedding_loss(old_z.cpu().unsqueeze(0), new_z.cpu().unsqueeze(0),
        #                                                          target=torch.tensor(1).float().cpu().detach()))
        #             embedding_different.append((new_z - old_z).cpu().detach().numpy())
        #             # dissimilarity.append(h_different_value.float().cpu().detach())
        #             # embedding_different.append(h_embedding_different.cpu().detach().numpy())
        #             new_embedding.append(new_z.cpu().detach().numpy())
        #             commit_list.append(commit_name)
        #             if 'malware_generated' in commit_name:
        #                 label_list.append(0)
        #             else:
        #                 label_list.append(1)
        #         except:
        #             print('out of memory')
        #             continue
        #
        #     loss = torch.tensor(loss)
        #     dissimilarity = torch.mean(torch.tensor(embedding_different), dim=-1)
        #
        #     new_embedding = np.asarray(new_embedding)
        #     embedding_different = np.asarray(embedding_different)
        #     commit_list = np.asarray(commit_list)
        #
        #     # Reconstruction error Top K
        #     _, indicates = torch.topk(loss, k=25, dim=-1)
        #     print("Reconstruction error Top K")
        #     for i in indicates:
        #         print("{}:  {}".format(commit_list[i], loss[i]))
        #     print("Finish! \n")
        #
        #     for i in range(len(self.data_malware_generated)):
        #         print("{}:  {}".format(commit_list[-(i + 1)], loss[-(i + 1)]))
        #     print("Finish! \n")
        #
        #     # Dissimilarity Top K
        #     _, indicates = torch.topk(dissimilarity, k=25, dim=-1)
        #     print("Dissimilarity Top K")
        #     for i in indicates:
        #         print("{}:  {}".format(commit_list[i], dissimilarity[i]))
        #     print("Finish! \n")
        #
        #     for i in range(len(self.data_malware_generated)):
        #         print("{}:  {}".format(commit_list[-(i + 1)], dissimilarity[-(i + 1)]))
        #     print("Finish! \n")
        #
        #     np.savez('new_embed_2', ed=embedding_different, cl=commit_list)
        #
        #     # db = DBSCAN(eps=0.001, min_samples=5).fit(embedding_different)
        #     # for i in range(len(commit_list)):
        #     #     if db.labels_[i] != 0:
        #     #         print("{}:  {}".format(commit_list[i], db.labels_[i]))
        #     print("Finish! \n")
        #
        #     # New Emending PCA
        #     pca_analysis(new_embedding, 'New Embedding', commit_list, label_list)
        #
        #     # Emending different PCA
        #     pca_analysis(embedding_different, 'Embedding Different', commit_list, label_list)
        #
        #     torch.cuda.empty_cache()

