from torch.utils.data import*
import numpy as np

class dataClass (Dataset):


    def __init__ (self, feats, label):
        self.feats = feats
        self.labels = label


    def __getitem__ (self, index):
        return (self.feats[index], self.labels[index])

    def __len__(self):
        return self.feats.shape[0]


## replaces default collate function
def my_collate(batch):
    feats = np.array([item[0] for item in batch])
    labels = np.array([item[1] for item in batch])

    return (feats, labels)

def conver_label(l):
    if 'classical' in

def label_to_int(labels):
    print ("LABELS", labels)
    a = list (map (lambda x : 1 if ("blues" in x[0]) else 0, labels))
    print (a)
    return np.array(a)


def load_data(batch_size, shuffle):
    ## loads training, validation and testing data
    train_data = np.load("../../data/train.npz")
    valid_data = np.load("../../data/valid.npz")
    test_data = np.load("../../data/test.npz")

    ## creates dataset for the training, validation, test data
    train_data = dataClass(train_data["feat"], label_to_int(train_data["target"]))
    valid_data = dataClass(valid_data["feat"], label_to_int(valid_data["target"]))
    test_data = dataClass(test_data["feat"], label_to_int(test_data["target"]))

    ## data loaders for training, validation and testing data
    train_loader = DataLoader(dataset = train_data, collate_fn = my_collate, shuffle = shuffle, batch_size = batch_size)
    valid_loader = DataLoader(dataset = valid_data, collate_fn = my_collate, shuffle = shuffle, batch_size = batch_size)
    test_loader = DataLoader(dataset = test_data, collate_fn = my_collate, shuffle = shuffle, batch_size = batch_size)

    return (train_loader, valid_loader, test_loader)
