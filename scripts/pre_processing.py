import numpy as np
from numpy import genfromtxt
import math


class LoadDataset():

    def __init__(self, path_train):
        my_dict = {'s': 1, 'b': -1}
        self.y = np.loadtxt(path_train, delimiter=",",
                            dtype=np.str_, skiprows=1, usecols=1)
        self.y = np.array([my_dict[i] for i in self.y])
        self.y = self.y.reshape(self.y.shape[0], 1)
        self.x = np.loadtxt(path_train, delimiter=",",
                            skiprows=1, usecols=range(2, 32))
        self.ids = np.loadtxt(path_train, delimiter=",", skiprows=1, usecols=0)
        self.ids = self.ids.reshape(self.ids.shape[0], 1)
        # Standardize Data Manually for each separate group of data after Distribution manipulation
        #self.x = (self.x - np.mean(self.x, axis=0))/np.std(self.x, axis=0)
        self.n_samples = self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.ids[i]

    def __len__(self):
        return self.n_samples

    def get_data(self):
        return self.x, self.y, self.ids


def separate_data(tx, y, ids):

    array_1 = np.where((tx[:, 22] != 0))[0]
    tx_1 = np.delete(tx, array_1, 0)
    y_1 = np.delete(y, array_1, 0)
    ids_1 = np.delete(ids, array_1, 0)

    array_2 = np.where((tx[:, 22] != 1))[0]
    tx_2 = np.delete(tx, array_2, 0)
    y_2 = np.delete(y, array_2, 0)
    ids_2 = np.delete(ids, array_2, 0)

    array_3 = np.nonzero(np.logical_or((tx[:, 22] == 0), (tx[:, 22] == 1)))[0]
    tx_3 = np.delete(tx, array_3, 0)
    y_3 = np.delete(y, array_3, 0)
    ids_3 = np.delete(ids, array_3, 0)

    return (tx_1, y_1, ids_1), (tx_2, y_2, ids_2), (tx_3, y_3, ids_3)


def adapt_features(data_n):

    tx_n = data_n[0]
    to_delete = np.zeros(tx_n.shape[0])-999
    indexes = []
    for i in range(tx_n.shape[1]):
        # if(sum(tx_n[:,i]-to_delete)==0 or sum(tx_n[:,i])==0 ):
        # indexes.append(i)
        # Find unique values in column along with their length
        # if len is == 1 then it contains same values i.e -999 so features to drop or zeros
        if len(np.unique(tx_n[:, i])) == 1:
            indexes.append(i)

    tx_n = np.delete(tx_n, indexes, 1)

    return (tx_n, data_n[1], data_n[2])


def add_median(data_n):

    tx_n = data_n[0]
    for i in range(tx_n.shape[1]):
        column = tx_n[:, i]
        m = np.median(column[column != -999])
        column[column == -999] = m
        tx_n[:, i] = column

    return (tx_n, data_n[1], data_n[2])


def dimensionality_reduction_corr(data_n):
    tx_n = data_n[0]
    corr = np.corrcoef(tx_n.T)
    pairs = np.argwhere(np.triu(np.isclose(corr, 1, rtol=2e-01), 1))
    tx_n = np.delete(tx_n, pairs[:, 1], axis=1)
    return (tx_n, data_n[1], data_n[2])


def scale_transform(data_n):
    tx_n = data_n[0]
    # Scaling Data to apply Log function
    dividor = np.max(tx_n, axis=0)-np.min(tx_n, axis=0)
    tx_n = (tx_n - np.min(tx_n, axis=0))/dividor
    # Transforming Data
    tx_n = np.log(1+tx_n)
    # Normalizing Data
    centered_data = tx_n - np.mean(tx_n, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    return (std_data, data_n[1], data_n[2])


def get_processed_data(path):
    train_data = LoadDataset(path)
    tx, y, ids = train_data.get_data()
    datas = separate_data(tx, y, ids)

    for d in datas:
        d = adapt_features(d)
        d = add_median(d)
        d = dimensionality_reduction_corr(d)
        d = scale_transform(d)

    return datas
