import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class StackDataset(Dataset):
    def __init__(self, file_path):
        self.data = np.load(file_path)
        self.X = self.data['feats']
        self.Y = self.data['target']
        self.B = self.data['bounds']
        print (self.X.shape)
        print (self.Y.shape)
        print (self.B.shape)

    def __getitem__(self, index):
        features = self.X[index]
        boundries = self.B[index]
        labels = self.Y[index]

        features = features.T

        assingments = []
        for j in range(len(boundries)):
            if (j == len(boundries) - 1):
                assingments.extend([j+1] * (features.shape[1] - boundries[j]))
            else:
                assingments.extend([j+1] * (boundries[j+1] - boundries[j]))
        assingments = np.array(assingments)

        label_mask = np.tile(1, labels.shape[0])

        return (torch.from_numpy(features), torch.from_numpy(assingments),
                torch.from_numpy(labels), torch.from_numpy(label_mask))

    def __len__(self):
        return self.X.shape[0]


def collate_stack(data):

    maxtime = 0
    maxphon = 0

    batch_feat = []
    batch_assign = []
    batch_label = []
    batch_labelm = []

    for i in range(len(data)):
        f, a, l, lm = data[i]

        if int(f.shape[1]) > maxtime:
            maxtime = int(f.shape[1])
        if int(l.shape[0]) > maxphon:
            maxphon = int(l.shape[0])

    for i in range(len(data)):
        f, a, l, lm = data[i]

        batch_feat.append(np.pad(f, ((0, 0), (0, maxtime - int(f.shape[1]))), 'constant'))
        batch_assign.append(np.pad(a, (0, maxtime - int(a.shape[0])), 'constant'))
        batch_label.append(np.pad(l, (0, maxphon - int(l.shape[0])), 'constant'))
        batch_labelm.append(np.pad(lm, (0, maxphon - int(lm.shape[0])), 'constant'))

    batch_feat = np.array(batch_feat)
    batch_assign = np.array(batch_assign)
    batch_label = np.array(batch_label)
    batch_labelm = np.array(batch_labelm)

    return (batch_feat, batch_assign,
            batch_label, batch_labelm)



def get_DataLoader(path, batch_size):
    training_Dataset = StackDataset(path)
    training_DataLoader = DataLoader(training_Dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     collate_fn=collate_stack)
    return training_DataLoader

if __name__ == "__main__":
    loader = get_DataLoader('valid5.npz', 32)
    for batch_idx, batch_data in enumerate(loader):
        batch_feat, batch_assignment, batch_label, batch_labelm = batch_data

        if batch_idx == 5:
            break;

        print (batch_idx, batch_feat.shape)
