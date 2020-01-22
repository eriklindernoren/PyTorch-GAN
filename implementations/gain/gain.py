"""

paper implementation in pytorch <GAIN: Missing Data Imputation using Generative Adversarial Nets ICML2018>

"""


import torch
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--missing_rate", type=float, default=0.2, help="missing rate")
parser.add_argument("--hint_rate", type=float, default=0.9, help="hint rate")
parser.add_argument("--train_rate", type=float, default=0.8, help="training rate")
parser.add_argument("--alpha", type=float, default=0.2, help="loss penalty")


opt = parser.parse_args()


def generate_samples(norws, ncol):
    return np.random.rand(n_rows, n_cols)


def scale(data_array):
    feature_dim = data_array.shape[1]
    min_val_record = np.zeros(feature_dim)
    max_val_record = np.zeros(feature_dim)

    for col_index in range(feature_dim):
        min_val_record[col_index] = np.min(data_array[:, col_index])
        data_array[:, col_index] = data_array[:, col_index] - min_val_record[col_index]
        max_val_record[col_index] = np.max(data_array[:, col_index])
        data_array[:, col_index] = data_array[:, col_index] / (max_val_record[col_index] + 1e-6)

    return data_array


def generate_mask(n_rows, n_cols, missing_rate):
    """
    @n_rows: number of rows to generate missing matrix
    @n_cols: number of columns to generate missing matrix
    """


    random_data = np.random.uniform(0., 1., size=[n_rows, n_cols])
    tmp = random_data > missing_rate
    missing_mat = 1. * tmp

    return missing_mat


def generate_noise(n_rows, n_cols):
    """
    generate noise matrix
    """
    return np.random.uniform(0., 1., size=[n_rows, n_cols])


def collect_data(n_rows, n_cols):
    """

    :param n_rows:
    :param n_cols:
    :return: train set , test set and their masks, all np array
    """
    data = generate_samples(n_rows, n_cols)

    data = scale(data)

    missing_mask = generate_mask(n_rows, n_cols, opt.missing_rate)

    idx = np.random.permutation(n_rows)
    train_nums = int(n_rows * opt.train_rate)
    test_nums = n_rows - train_nums

    train_samples = data[idx[:train_nums], :]
    test_samples = data[idx[train_nums:], :]
    train_mask = missing_mask[idx[:train_nums], :]
    test_mask = missing_mask[idx[train_nums:], :]

    return train_samples, test_samples, train_mask, test_mask


class SimpleDataLoader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, specify_data, mask):
        """
        """
        self.specify_data = specify_data
        self.mask = mask

    def __len__(self):
        return len(self.specify_data)

    def __getitem__(self, idx):
        data = self.specify_data[idx]
        mask = self.mask[idx]

        return data, mask


class NetD(torch.nn.Module):
    def __init__(self, feature_dim):
        """

        :param feature_dim:
        """
        super(NetD, self).__init__()
        self.fc1 = torch.nn.Linear(feature_dim * 2, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, feature_dim)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        layers = [self.fc1, self.fc2, self.fc3]
        [torch.nn.init.xavier_normal_(layer.weight) for layer in layers]

    def forward(self, x, m, g, h):
        """
        reference equation(4) in paper

        :param x: original data
        :param m: missing mask
        :param g: generated data by Generator
        :param h: hint, see paper
        :return: as a prob matrix, denote where is missing or not
        """
        self.init_weight()
        inp = m * x + (1 - m) * g
        inp = torch.cat((inp, h), dim=1)
        out = self.relu(self.fc1(inp))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))

        return out


class NetG(torch.nn.Module):
    def __init__(self,feature_dim):
        """

        :param feature_dim:
        """
        super(NetG, self).__init__()
        self.fc1 = torch.nn.Linear(feature_dim * 2, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, feature_dim)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        layers = [self.fc1, self.fc2, self.fc3]
        [torch.nn.init.xavier_normal_(layer.weight) for layer in layers]

    def forward(self, x, z, m):
        """

        reference equation(2,3) in paper

        :param x: mising data
        :param z: noise
        :param m: missing mask, used to replace missing part bu noise
        :return: generated data, size same as original data
        """
        self.init_weight()
        inp = m * x + (1 - m) * z
        inp = torch.cat((inp, m), dim=1)
        out = self.relu(self.fc1(inp))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))

        return out


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

bce_loss = torch.nn.BCEWithLogitsLoss(reduction="elementwise_mean")
mse_loss = torch.nn.MSELoss(reduction="elementwise_mean")

n_rows, n_cols = 500,40

netD = NetD(feature_dim=n_cols).to(device)
netG = NetG(feature_dim=n_cols).to(device)

optimD = torch.optim.RMSprop(netD.parameters(), lr=opt.lr)
optimG = torch.optim.RMSprop(netG.parameters(), lr=opt.lr)

train_samples, test_samples, train_mask, test_mask = collect_data(n_rows, n_cols)
train_dset = SimpleDataLoader(train_samples, train_mask)
train_loder = DataLoader(train_dset,
                         batch_size=opt.batch_size,
                         num_workers=1)

if __name__ == "__main__":
    epochs = opt.epochs
    missing_rate = opt.missing_rate
    alpha = opt.alpha

    for epoch in range(opt.epochs):
        for idx, (x, mask) in enumerate(train_loder):
            noise = generate_noise(x.shape[0], x.shape[1])
            hint = generate_mask(x.shape[0], x.shape[1], 1 - missing_rate)

            x = torch.tensor(x).float().to(device)
            noise = torch.tensor(noise).float().to(device)
            mask = torch.tensor(mask).float().to(device)
            hint = torch.tensor(hint).float().to(device)

            hint = mask * hint + 0.5 * (1 - hint)

            # train D
            optimD.zero_grad()
            G_sample = netG(x, noise, mask)
            D_prob = netD(x, mask, G_sample, hint)
            D_loss = bce_loss(D_prob, mask)

            D_loss.backward()
            optimD.step()

            # train G
            optimG.zero_grad()
            G_sample = netG(x, noise, mask)
            D_prob = netD(x, mask, G_sample, hint)
            D_prob.detach_()

            G_loss = ((1 - mask) * (torch.sigmoid(D_prob) + 1e-8).log()).mean() / (1 - mask).sum()
            G_mse_loss = mse_loss(mask * x, mask * G_sample) / mask.sum()
            G_loss = G_loss + alpha * G_mse_loss

            G_loss.backward()
            optimG.step()

            G_mse_test = mse_loss((1 - mask) * x, (1 - mask) * G_sample) / (1 - mask).sum()
            G_mse_train = mse_loss((mask) * x, (mask) * G_sample) / (mask).sum()

            if epoch % 2 == 0:
                print('Iter:{}\tD_loss: {:.4f}\tG_loss: {:.4f}\tTeloss: {:.4f}\tTrLoss:{:.4f}'. \
                      format(epoch, D_loss, G_loss, np.sqrt(G_mse_test.data.cpu().numpy()),
                             np.sqrt(G_mse_train.data.cpu().numpy())))

    mse_all = 0
    count = 0
    test_noise = generate_noise(test_samples.shape[0], test_samples.shape[1])
    test_samples = torch.tensor(test_samples).float().to(device)
    test_noise = torch.tensor(test_noise).float().to(device)
    test_mask = torch.tensor(test_mask).float().to(device)

    G_sample = netG(test_samples, test_noise, test_mask)

    MSE_final = mse_loss((1 - test_mask) * test_samples, (1 - test_mask) * G_sample) / (1 - test_mask).sum()
    print('total rmse:', np.sqrt(MSE_final.data.cpu().numpy()))
