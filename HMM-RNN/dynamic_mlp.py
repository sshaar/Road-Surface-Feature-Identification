import os,sys
import time
import numpy as np
import math
from collections import Counter
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

ACCUM_GRAD = 16
GPU = 0
SHOW = 100

class Flatten(nn.Module):
    """
    Implement a simple custom module that reshapes (n, m, x) tensors to (n*m, x).
    """
    def forward(self, x):
        x = x.view(x.shape[0] * x.shape[1], x.shape[2])
        return x


class LockedDropout(nn.Module):

    def forward(self, data, is_training=True, dropout=0.5):
        if not is_training or not dropout:
            return data

        mask = data.data.new(1, data.size(1), data.size(2)).bernoulli_(1 - dropout)
        mask = Variable(mask, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(data)
        if GPU:
            mask = mask.cuda()

        return mask * data

class DynamicMLP(nn.Module):
    def __init__(self, input_size, output_size, num_layers=7, hidden_size=128):
        super(DynamicMLP, self).__init__()
        # All the layers in the states of the HMM.
        self.layers = []

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = num_layers


        self.i_dropout = 0.1
        self.h_dropout = 0.3
        self.o_dropout = 0.4

        for i in range(num_layers):
            layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=1, dropout=0.4)
            self.layers.append(layer)
        self.rnns = torch.nn.ModuleList(self.layers)

        self.decoder = nn.Linear(hidden_size, output_size)

        self.flatten = Flatten()

        self.dropout = LockedDropout()

        self.initializer()

    def initializer(self):

        for rnn in self.layers:
            for name, param in rnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant(param, 0.0)
                elif 'weight' in name:
                    nn.init.uniform(param, -1/math.sqrt(self.hidden_size), 1/math.sqrt(self.hidden_size))

        self.decoder.weight.data.uniform_(-.1, .1)
        self.decoder.bias.data.fill_(0)

    def forward(self, input, layer_seq, states=[], is_training=True):
        # states = self.initialize_states(1)
        result = []
        for idx in range(input.shape[1]):
            time_step = input[0][idx]
            time_step = time_step.view((1, 1, self.input_size))

            layer = self.layers[layer_seq[idx]]


            if idx == 0:
                if len(states) == self.n_layers:
                    output, state = layer(time_step, states[layer_seq[idx]])
                    # state = states[layer_seq[idx]]
                else:
                    output, state = layer(time_step)
            else:
                output, state = layer(time_step, state)
            # states[layer_seq[idx]] = state

            output = self.dropout(output, is_training=is_training, dropout=self.h_dropout)
            output = self.decoder(output)
            result.append(output)

        if GPU:
            output = output.cuda()
        output = torch.cat(result, dim=1)
        return self.flatten(output)

    def initialize_states(self, bsz):

        weight = next(self.parameters()).data
        states = []
        for i in range(self.n_layers):
            if i == self.n_layers - 1:
                new_state = (Variable(weight.new(1, bsz, self.hidden_size).zero_()),
                            Variable(weight.new(1, bsz, self.hidden_size).zero_()))
            else:
                new_state = (Variable(weight.new(1, bsz, self.hidden_size).zero_()),
                            Variable(weight.new(1, bsz, self.hidden_size).zero_()))

            states.append(new_state)
        return states

class DataClass (Dataset):
    def __init__ (self, feats, label, seq, bounds):
        self.feats = feats
        self.labels = label
        self.seq = seq
        self.bounds = bounds
        print 'Feats', feats.shape
        print 'Labels', label.shape
        print 'Seq', seq.shape

    def __getitem__ (self, index):
        return (self.feats[index], self.labels[index], self.seq[index], self.bounds[index])

    def __len__(self):
        return self.feats.shape[0]

def my_collate(batch):
    # feats = np.array([item[0] for item in batch])
    # labels = np.array([item[1] for item in batch])
    # seqs = np.array([item[2] for item in batch])
    (feats, labels, seqs, bounds) = zip(*batch)

    feats = np.array(feats)
    labels = np.array(labels)
    seqs = np.array(seqs)
    bounds = np.array(bounds)

    target_tensor = Variable(torch.zeros((seqs.shape[1]))).long()

    idx = 0

    for i in range(bounds[idx].shape[0]):
        if i == (bounds[idx].shape[0] - 1):
            target_tensor[bounds[idx][i]:feats[idx].shape[1]] = labels[idx][i]
            # print bounds[idx][i], utterance[idx].shape[0], target_tensor[idx][bounds[idx][i]:utterance[idx].shape[0]]
        else:
            target_tensor[bounds[idx][i]:bounds[idx][i+1]] = labels[idx][i]
            # print bounds[idx][i], bounds[idx][i+1], target_tensor[idx][bounds[idx][i]:bounds[idx][i+1]]

    return (feats, target_tensor, seqs)


def label_to_int(labels):
    # print ("LABELS", labels)
    # print labels.shape
    a = list (map (lambda x : change_label(x[0]), labels))
    # print (a)
    return np.array(a)

def log_data(model, optimizer, lr, weight_decay, training_loss, validation_loss,
             training_accuracy, validation_accuracy, confusion_matrix_valid,
             confusion_matrix_test):

    results = {}

    results['model'] = str(model)
    results['optimizer'] = str(optimizer.defaults)
    results['lr'] = lr
    results['weight_decay'] = weight_decay
    results['training_loss'] = training_loss
    results['validation_loss'] = validation_loss
    results['training_accuracy'] = training_accuracy
    results['validation_accuracy'] = validation_accuracy
    results['confusion_matrix_valid'] = confusion_matrix_valid
    results['confusion_matrix_test'] = confusion_matrix_test

    pickle.dump(results, open('results/hybrid6.pkl', 'w'))

def load_data(batch_size, shuffle):
    ## loads training, validation and testing data
    train_data = np.load("/home/sshaar/hmm-rnn/road/train6.npz")
    train_seq = np.load("/home/sshaar/hmm-rnn/road/train_seqs6.npy")
    valid_data = np.load("/home/sshaar/hmm-rnn/road/valid6.npz")
    valid_seq = np.load("/home/sshaar/hmm-rnn/road/valid_seqs6.npy")
    test_data = np.load("/home/sshaar/hmm-rnn/road/test6.npz")
    test_seq = np.load("/home/sshaar/hmm-rnn/road/test_seqs6.npy")

    ## creates dataset for the training, validation, test data
    train_data = DataClass(train_data["feats"], train_data["target"], train_seq, train_data['bounds'])
    valid_data = DataClass(valid_data["feats"], valid_data["target"], valid_seq, valid_data['bounds'])
    test_data = DataClass(test_data["feats"], test_data["target"], test_seq, test_data['bounds'])

    ## data loaders for training, validation and testing data
    train_loader = DataLoader(dataset = train_data, collate_fn = my_collate, shuffle = shuffle, batch_size = batch_size)
    valid_loader = DataLoader(dataset = valid_data, collate_fn = my_collate, shuffle = shuffle, batch_size = batch_size)
    test_loader = DataLoader(dataset = test_data, collate_fn = my_collate, shuffle = shuffle, batch_size = batch_size)

    return (train_loader, valid_loader, test_loader)

def evaluate(model, criterion, dataloader, epoch, output_size=5, batch_size=1):
    model.eval()
    start_time = time.time()
    total_loss = 0.0
    correct = 0
    total = 0
    matrix = np.zeros((output_size, output_size))
    for i, (data, labels, hmm_seq) in enumerate(dataloader):
        data = Variable(torch.from_numpy(data), requires_grad=True).float()
        # labels = Variable(torch.from_numpy(labels)).long()
        hmm_seq = hmm_seq.reshape(hmm_seq.shape[1])
        total += data.shape[1]

        if GPU:
            data.cuda()
            labels.cuda()

        states = model.initialize_states(batch_size)
        output = model(data, hmm_seq, states, is_training=False).long()

        _, prediction = torch.max(output, 1)
        matches = (prediction.cpu().long() == labels.long()).long()

        for j, instance in enumerate(labels.data):
            predict_label = int(prediction[j].data[0])
            true_label = int(instance.data[0])

            matrix[true_label][predict_label] += 1

        correct += matches.sum()
        # loss = criterion(output, labels)

        # total_loss += loss.data[0]
        if (i+1)%SHOW == 0:
            elapsed = time.time() - start_time
            s = 'Validation Epoch: {} [{}]\tLoss: {:.6f}\tCorrect: {}\tTime: {:5.2f} '.format(epoch+1, i+1, total_loss/(i+1), float(correct)/total, elapsed)
            f = open('results/log6', 'a+')
            f.write(s+'\n')
            print s

    return total_loss/(i+1), (float(correct)/total), matrix

def train(epochs=10, learning_rate=1e-3, batch_size=1, input_size=15, hidden_size=350, output_size=5, num_layers=7, shuffle=True, weight_decay=0.001):

    train_loader, valid_loader, test_loader = load_data(batch_size, shuffle)

    model = DynamicMLP(input_size, output_size, num_layers=num_layers, hidden_size=hidden_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if GPU:
        criterion.cuda()
        model.cuda()

    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []
    confusion_matrix_valid = []

    confusion_matrix_test = None

    for epoch in range(epochs):

        model.train()
        start_time = time.time()
        total_loss = 0.0
        correct = 0.0
        total = 0
        for i, (data, labels, hmm_seq) in enumerate(train_loader):
            data = Variable(torch.from_numpy(data), requires_grad=True).float()
            # labels = Variable(torch.from_numpy(labels)).long()
            hmm_seq = hmm_seq.reshape(hmm_seq.shape[1])
            total += data.shape[1]
            if GPU:
                data.cuda()
                labels.cuda()

            states = model.initialize_states(batch_size)
            output = model(data, hmm_seq, states)

            _, prediction = torch.max(output, 1)
            # print labels
            correct += (prediction.cpu().long() == labels.long()).long().sum()
            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()

            if (i+1) % ACCUM_GRAD == 0:
                optimizer.zero_grad()

            total_loss += loss.data[0]
            if (i+1)%SHOW == 0:
                elapsed = time.time() - start_time
                s = 'Training Epoch: {} [{}]\tLoss: {:.6f}\tCorrect: {}\tTime: {:5.2f}'.format( epoch+1, i+1, total_loss/(i+1), float(correct)/total, elapsed)
                print s
                f = open('results/log6', 'a+')
                f.write(s+'\n')
        training_loss.append(total_loss/(i+1))
        training_accuracy.append(float(correct)/total)

        (eval_loss, acc, matrix) = evaluate(model, criterion, valid_loader, epoch, output_size=output_size)
        validation_loss.append(eval_loss)
        validation_accuracy.append(acc)
        confusion_matrix_valid.append(matrix)

    _, _, confusion_matrix_test = evaluate(model, criterion, test_loader, epoch, output_size=output_size)

    log_data(model, optimizer, learning_rate, weight_decay, training_loss, validation_loss,
             training_accuracy, validation_accuracy, confusion_matrix_valid,
             confusion_matrix_test)

train()
