import numpy as np
import torch
import shutil
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
import torch.optim as optim
from collections import Counter

import pickle, sys, random, os, csv, time

from loader import *
from model import *

TRAINING_PATH = '/home/sshaar/hmm-rnn/road/train6.npz'
VALID_PATH = '/home/sshaar/hmm-rnn/road/valid6.npz'
TEST_PATH = '/home/sshaar/hmm-rnn/road/test6.npz'


BATCH_SIZE = 20
EPOCHS = 30
DEEPNET = 1
DISPLAY_BATCH = 100
DISPLAY_EPOCH = 1
RESET_GRAD = 1

input_size = 15
output_size = 5

lr = 0.0005
weight_decay = 0.0005

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
LOG_FILE = open('results/log6', 'w+')


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

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

    pickle.dump(results, open('results/CNN_'+str(sys.argv[1])+'.pkl', 'w'))



def evaluate(model, loss, dataloader):
    model.eval()

    train_acc = 0
    n_correct = 0
    n_total = 0
    total_loss = 0
    matrix = np.zeros((output_size, output_size))

    for batch_idx, batch_data in enumerate(dataloader):

        batch_feat, batch_assignment, batch_label, batch_labelm = batch_data

        predicted_scores = model(batch_feat, batch_assignment, DEEPNET)
        predicted_scores = predicted_scores.view(predicted_scores.shape[0]*predicted_scores.shape[1], output_size)

        _, predicted_labels = torch.max(predicted_scores, 1)

        batch_label = Variable(torch.from_numpy(batch_label.reshape(batch_label.size))).long()
        batch_labelm = Variable(torch.from_numpy(batch_labelm.reshape(batch_labelm.size))).double()
        if (predicted_scores.shape[0] != batch_label.shape[0]):
            continue

        if DEEPNET:
            batch_label = batch_label.cuda()
            batch_labelm = batch_labelm.cuda()

        matches = (predicted_labels == batch_label).long() * batch_labelm.long()

        j = 0
        for i, instance in enumerate(batch_label.data):
            if int(batch_labelm[i].data[0]) == 0:
                continue
            predict_label = int(predicted_labels[i].data[0])
            true_label = int(instance.data[0])

            matrix[true_label][predict_label] += 1


        n_correct += float(matches.sum().data[0])
        n_total += float(torch.sum(batch_labelm).data[0])

        output = loss.forward(predicted_scores, batch_label)
        summed_weight = torch.sum(output * batch_labelm)
        current_loss = float(summed_weight.data[0]/torch.sum(batch_labelm).data[0])
        total_loss += float(current_loss)

    accuracy = float(n_correct)/n_total

    return total_loss, accuracy, matrix


def main():
    global LOG_FILE
    start_epoch = 0

    training_DataLoader = get_DataLoader(TRAINING_PATH, BATCH_SIZE)
    valid_DataLoader = get_DataLoader(VALID_PATH, BATCH_SIZE)
    test_DataLoader = get_DataLoader(TEST_PATH, BATCH_SIZE)
    print 'training_DataLoader', len(training_DataLoader)
    print 'valid_DataLoader', len(valid_DataLoader)
    print 'test_DataLoader', len(test_DataLoader)
    batch_training = len(training_DataLoader)

    model = StackNet(input_size, output_size)
    print model

    loss = nn.CrossEntropyLoss(reduce=False)
    if DEEPNET:
        loss = loss.cuda()
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []
    confusion_matrix_valid = []

    confusion_matrix_test = None

    print model
    print str(optimizer.defaults)

    for epoch in range(start_epoch, EPOCHS):
        model.train()

        total_training_loss = 0.0
        n_correct = 0
        n_total = 0

        start_time = time.time()

        for batch_idx, batch_data in enumerate(training_DataLoader):
            batch_feat, batch_assignment, batch_label, batch_labelm = batch_data

            # Reset gradient
            if (batch_idx+1) % RESET_GRAD:
                optimizer.zero_grad()


            fx = model.forward(batch_feat, batch_assignment, DEEPNET)
            fx = fx.view(fx.shape[0]*fx.shape[1], output_size)

            batch_label = Variable(torch.from_numpy(batch_label.reshape(batch_label.size))).long()
            batch_labelm = Variable(torch.from_numpy(batch_labelm.reshape(batch_labelm.size))).double()

            if DEEPNET:
                batch_label = batch_label.cuda()
                batch_labelm = batch_labelm.cuda()

            if (fx.shape[0] != batch_label.shape[0]):
                continue

            output = loss.forward(fx, batch_label)
            summed_weight = torch.sum(output * batch_labelm)

            # Backward
            summed_weight.backward()

            # Update parameters
            optimizer.step()

            current_loss = float(summed_weight.data[0]/torch.sum(batch_labelm).data[0])
            total_training_loss += float(current_loss)

            _, predicted_labels = torch.max(fx, 1)

            matches = (predicted_labels == batch_label).long() * batch_labelm.long()
            n_correct += float(matches.sum().data[0])
            n_total += float(torch.sum(batch_labelm).data[0])

            if (batch_idx+1)%50 == 0:
                s = ('Epoch %d/%d Batch %d/%d\t' %(epoch+1, EPOCHS, batch_idx+1, batch_training))
                s += ('Loss %f \tAccuracy %f' %(current_loss, float(n_correct)/n_total))
                print s
                LOG_FILE = open('results/log6', 'a+')
                LOG_FILE.write(s+'\n')


        training_loss.append(total_training_loss/float(batch_idx))
        training_accuracy.append(float(n_correct)/n_total)

        (eval_loss, acc, matrix) = evaluate(model, loss, valid_DataLoader)
        validation_loss.append(eval_loss)
        validation_accuracy.append(acc)
        confusion_matrix_valid.append(matrix)

        s = ('Training loss: %f \tTraining Accuracy: %f \tValidloss: %f \tValidAccuracy: %f'%(total_training_loss, float(n_correct)/n_total, eval_loss, acc))
        print ('Epoch %d/%d \t\tTime %f' %(epoch+1, EPOCHS, time.time()-start_time))
        print (s)

        LOG_FILE = open('results/log6', 'a+')
        LOG_FILE.write(s+'\n')

        for i in range(output_size):
            print 'Class', i
            print matrix[i]
            print ""
        print '______________________________________________________'

    _, _, confusion_matrix_test = evaluate(model, loss, test_DataLoader)

    log_data(model, optimizer, lr, weight_decay, training_loss, validation_loss,
             training_accuracy, validation_accuracy, confusion_matrix_valid,
             confusion_matrix_test)

main()
