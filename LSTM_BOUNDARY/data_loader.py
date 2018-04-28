import sys, os, time
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

GPU = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class CustomDataset(Dataset):
    def __init__(self, audio_file_path):
        audio_data = np.load(audio_file_path)

        self.X = audio_data['feats']
        self.Y = audio_data['target']
        self.bound = audio_data['bounds']

    def __getitem__(self, index):
        return (self.X[index], self.Y[index], self.bound[index])

    def __len__(self):
        return self.X.shape[0]


def collate_stack(data):
    utterance, target, bounds = zip(*data)

    print len(utterance)
    print utterance[0].shape
    # print max(utterance_length)
    utterance_length = map(lambda x: x.shape[0], utterance)
    print utterance_length

    max_len = max(utterance_length)

    utterance_length = Variable(torch.from_numpy(np.array(utterance_length))).long()
    utterance_tensor = Variable(torch.zeros((len(utterance_length),
                                             max_len, 225))).float()
    target_tensor = Variable(torch.zeros((len(utterance_length),
                                          max_len))).long()

    for idx, (seq, seqlen) in enumerate(zip(utterance, utterance_length)):
        utterance_tensor[idx, :seqlen] = torch.FloatTensor(seq)


    for idx in range(target_tensor.shape[0]):
        for i in range(bounds[idx].shape[0]):
            if i == (bounds[idx].shape[0] - 1):
                target_tensor[idx][bounds[idx][i]:utterance[idx].shape[0]] = target[idx][i]
                continue
            target_tensor[idx][bounds[idx][i]:bounds[idx][i+1]] = target[idx][i]

    utterance_length, perm_idx = utterance_length.sort(0, descending=True)

    utterance_tensor = utterance_tensor[perm_idx]
    target_tensor = target_tensor[perm_idx]

    # utterance_tensor = utterance_tensor.transpose(0, 1)
    # target_tensor = target_tensor.transpose(0, 1)

    return (utterance_tensor, target_tensor, utterance_length)

def get_DataLoader(audio_file_path, batch_size):
    data_set = CustomDataset(audio_file_path)
    data_loader = DataLoader(data_set,
                             batch_size=batch_size,
                             shuffle=True,
                             collate_fn=collate_stack)
    return data_loader

if __name__ == "__main__":

    DATA_DIRECTORY = sys.argv[1]

    utter_path = os.path.join(DATA_DIRECTORY)

    loader = get_DataLoader(utter_path, 32)

    for idx, data in enumerate(loader):
        (utter, targ, utter_len) = data
        print utter_len
        print targ
        break