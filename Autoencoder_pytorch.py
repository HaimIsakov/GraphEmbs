from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib as mpl
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AutoencoderPytorch(nn.Module):
    def __init__(self, nets):
        super(AutoencoderPytorch, self).__init__()
        self.nb_epoch = 15
        self.batch_size = 128
        self.h1_dim = 800
        self.lr = 0.001
        self.noise = 0.05
        self.X = self.triangular_adjacency_matrix(nets, 3)
        self.embs = 0
        self.sim_mat = 0
        nb_visible = self.X.shape[1]
        self.encoder_layer = nn.Linear(nb_visible, self.h1_dim)
        self.decoder_layer = nn.Linear(self.h1_dim, nb_visible)

    def forward(self, data):
        data = self.encoder(data)
        data = self.decoder(data)
        return data

    def similarity_matrix(self):
        dist = metrics.pairwise.euclidean_distances(self.embs)
        dist_sc = (dist - dist.min())/(dist.max()-dist.min())
        emb_sim = 1 - dist_sc
        return emb_sim

    @staticmethod
    def triangular_adjacency_matrix(nets, power=1):
        X = []
        for g in nets.keys():
            A = nx.adjacency_matrix(nets[g]['network'])
            A = np.linalg.matrix_power(A.toarray(), power)
            # print(A.shape)
            indices = np.triu_indices_from(A)
            X.append(A[indices])
        #X.append(A.toarray().flatten())
        X = np.array(X)
        return X

    @staticmethod
    def get_noise(data, p):
        res = []
        for i in range(data.shape[0]):
            res.append(np.random.binomial(1, p, data.shape[1])*-1)
        return np.array(res)

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def encoder(self, data):
        data = F.tanh(self.encoder_layer(data))
        return data

    def decoder(self, data):
        data = F.sigmoid(self.decoder_layer(data))
        return data

    def _train(self):
        train_index = range(0, self.X.shape[0])
        nb_visible = self.X.shape[1]
        x_train = self.X[train_index]
        noise_matrix = self.get_noise(x_train, self.noise)
        x_train_noisy = x_train + noise_matrix
        x_train_noisy[x_train_noisy < 0] = 0

        print("Training shape: ", x_train.shape)
        print("Hidden dimension: ", self.h1_dim)
        print("Learning rate: ", self.lr)
        print("Batch size: ", self.batch_size)


        sc = preprocessing.MinMaxScaler().fit(x_train)
        x_train = torch.Tensor(sc.transform(x_train))
        x_train_noisy = torch.Tensor(sc.transform(x_train_noisy))

        optimizer = self.get_optimizer()
        loss_fn = nn.MSELoss(reduction='mean')
        for epoch in range(self.nb_epoch):
            optimizer.zero_grad()
            net_out = self(x_train_noisy)
            loss = loss_fn(net_out, x_train)
            loss.backward()
            optimizer.step()

        self.embs = self.encoder(x_train).detach().numpy()
        self.sim_mat = self.similarity_matrix()

