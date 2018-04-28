import numpy as np
import pickle
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.utils.data as dt
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import*
import shutil
import os
import numpy as np
import random
from model import *
from data_loader import *
import Levenshtein as L
from torch.nn.utils.rnn import PackedSequence
import pickle 

def save_checkpoint(state, filename='models/model.tar'):
    torch.save(state, filename)

###############
##PARAMETERS###
###############

epochs = 30
hidden_size = 300
em_size = 300
batch_size = 64
inp_size = 225
dropouts = [0.4,0.3,0.4]
model = Model(inp_size,em_size,hidden_size, dropouts,5, True)

# m = torch.load("models/2_1.tar")
# model.load_state_dict(m['state_dict'])
lr = 0.0005
weight_decay = 0.0001

optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
criterion = nn.CrossEntropyLoss(reduce = False)
model.cuda()

train_accuracies = []
valid_accuracies = []
train_losses = []
valid_losses = []
valid_mats = []


for e in range (epochs):
    print ("STARTING EPOCH:", e)
    
    train_loader = get_DataLoader("/home/sshaar/hmm-rnn/road/train7.npz", batch_size)
    valid_loader = get_DataLoader("/home/sshaar/hmm-rnn/road/valid7.npz", batch_size)
    test_loader = get_DataLoader("/home/sshaar/hmm-rnn/road/test7.npz", batch_size)
    sum_loss = 0.0
    total = 0
    correct = 0
    model.train()
    model.cuda()


    for batch, (batch_x, batch_y, lengths) in enumerate (train_loader):
        if (batch_x.shape[0] < batch_size):
            break

        # print (lengths)
        # break
        # print ("BATCH X SHAPE:", batch_x.shape)
        init_states = []
        for j in range (3):
            if (j == 2):
                init_state = model.init_out(batch_x.shape[0])
            else:
                init_state = model.init_hidden(batch_x.shape[0])
            init_states.append(init_state)

        waste = np.array(batch_y)
        # lengths = np.asarray([s.shape[0] for s in batch_y], dtype=np.int64)
        max_len = torch.max(lengths).numpy()
        # print ("MAX LEN", int(max_len))

        max_len = int(max_len)
        all_lengths = np.array([i for i in range (max_len)])
        # lens = torch.from_numpy(lengths)
        all_lens = torch.from_numpy(all_lengths)

        all_lens = all_lens.view(max_len,1)

        lens = lengths.view(1,batch_size)

        mask = all_lens < lens
        # print ("BATCH SHAPE", batch_y.shape)
        # print ("MASK SHAPE",mask.shape)


        batch_x = pad_sequence (batch_x).float().cuda()
        batch_y = pad_sequence (batch_y).long().cuda()

        # batch_x = batch_x.transpose(0,1)
        # print ("BATCH X SHAPE", batch_x.shape)


        optimizer.zero_grad()
        out = model(batch_x,init_states)
        # print ("OUT SHAPE:", out.shape)
        # print ("BATCH SHAPE",batch_y.shape)

        # mask = mask.transpose(0,1).contiguous()
        # batch_y = batch_y.transpose(1,0).contiguous()
        
        batch_y = batch_y.view(batch_y.shape[0]*batch_y.shape[1])
        mask = mask.view (mask.shape[0]*mask.shape[1]).float().cuda()
        # print ("TRAINING")
        # print ("MASK SHAPE", mask.shape[0])
        # print ("MASK SUM", sum(mask))
        # print ("SUM LENGTHS", lengths.sum())

        # print ("AFTER BATCH SHAPE", batch_y.shape)

        loss = criterion(out,batch_y)
        loss = loss*mask
        loss = torch.sum(loss)
        

        loss.backward()
        ss = batch_y.cpu().numpy()
        mask = mask.cpu().numpy()
        tot = 0
        cor = 0
        zeros = 0
        predicted = torch.max(out, 1)[1].cpu().numpy()
        
        for x in range (mask.shape[0]):
            if mask[x] == 0:
                continue
            else:
                tot += 1
                if (ss[x] == 0):
                    zeros += 1
                # print (predicted[x],ss[x])
                if (predicted[x] == ss[x]):
                    cor += 1

        total += tot
        correct += cor

        # print ("lengths", lengths)
        # print ("length total", lengths.sum())
        # print ("MASK TOTAL", tot)
        # print ("zeros", zeros)
        # # acc = (predicted == ss).sum().data.numpy()/ss.shape[0]
        # print ("TRAIN BATCH ACCURACY:", cor*1.0/tot)

        print (predicted.shape)
        batch_loss = loss.data[0]/sum(mask)
        # print ("TRAIN BATCH LOSS:",batch_loss)

        torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
        optimizer.step()
        sum_loss += loss.data[0]
        ave_loss = sum_loss/(i+1)
        

        # if (not (i % 10)):
        print ("EPOCH: ", e, " BATCH: ",batch, " LOSS: ", loss, "AVERAGE LOSS:", ave_loss, "Batch loss:", batch_loss)
        # if (batch == 200):
        #     break

    train_loss = sum_loss*1.0/total
    train_accuracy = correct*1.0/total
    train_accuracies.append(train_accuracy)
    train_losses.append(train_loss)

    print ("TRAIN ACCURACY", train_accuracy)
    print ("TRAIN LOSS", e, ": ", train_loss)

    sum_loss = 0.0
    total = 0
    correct = 0 
    model.eval()

    valmat = {0: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 1: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 2: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
            3 : {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 4: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 4: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}}


    for batch, (batch_x, batch_y, lengths) in enumerate (valid_loader):

        if (batch_x.shape[0] < batch_size):
            break
        # print (lengths)
        # break
        # print ("BATCH X SHAPE:", batch_x.shape)
        init_states = []
        for j in range (3):
            if (j == 2):
                init_state = model.init_out(batch_x.shape[0])
            else:
                init_state = model.init_hidden(batch_x.shape[0])
            init_states.append(init_state)

        waste = np.array(batch_y)
        # lengths = np.asarray([s.shape[0] for s in batch_y], dtype=np.int64)
        max_len = torch.max(lengths).numpy()
        # print ("MAX LEN", int(max_len))

        max_len = int(max_len)
        all_lengths = np.array([i for i in range (max_len)])
        # lens = torch.from_numpy(lengths)
        all_lens = torch.from_numpy(all_lengths)

        all_lens = all_lens.view(max_len,1)

        lens = lengths.view(1,batch_size)

        mask = all_lens < lens
        # print ("BATCH SHAPE", batch_y.shape)
        # print ("MASK SHAPE",mask.shape)


        batch_x = pad_sequence (batch_x).float().cuda()
        batch_y = pad_sequence (batch_y).long().cuda()



        optimizer.zero_grad()
        out = model(batch_x,init_states)
        # print ("OUT SHAPE:", out.shape)
        # print ("BATCH SHAPE",batch_y.shape)
        # mask = mask.transpose(0,1).contiguous()
        # batch_y = batch_y.transpose(1,0).contiguous()

        batch_y = batch_y.view(batch_y.shape[0]*batch_y.shape[1])
        mask = mask.view (mask.shape[0]*mask.shape[1]).float().cuda()
        # print ("AFTER BATCH SHAPE", batch_y.shape)



        loss = criterion(out,batch_y)
        loss = loss*mask
        loss = torch.sum(loss)
        

        # loss.backward()
        ss = batch_y.cpu().numpy()
        mask = mask.cpu().numpy()
        tot = 0
        cor = 0
        zeros = 0
        predicted = torch.max(out, 1)[1].cpu().numpy()
        
        for x in range (mask.shape[0]):
            if mask[x] == 0:
                continue
            else:
                tot += 1
                # if (ss[x] == 0):
                #     zeros += 1
                valmat[ss[x]][predicted[x]] += 1
                # print (predicted[x],ss[x])
                if (predicted[x] == ss[x]):
                    cor += 1

        total += tot
        correct += cor


        # print ("lengths", lengths)
        # print ("length total", lengths.sum())
        # print ("MASK TOTAL", tot)
        # print ("zeros", zeros)
        # acc = (predicted == ss).sum().data.numpy()/ss.shape[0]
        # print ("VALID BATCH ACCURACY:", cor*1.0/tot)

        print (predicted.shape)
        batch_loss = loss.data[0]/sum(mask)
        # print ("VALID LOSS:",batch_loss)

        # torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
        # optimizer.step()
        sum_loss += loss.data[0]
        ave_loss = sum_loss/(i+1)
        # break
        

        # if (not (i % 10)):
        print ("EPOCH: ", e, " BATCH: ",batch, " LOSS: ", loss, "AVERAGE LOSS:", ave_loss, "BATCH LOSS", batch_loss)

    valid_loss = sum_loss*1.0/total
    valid_accuracy = correct*1.0/total
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)
    valid_mats.append(valmat)

    print ("VALID ACCURACY", valid_accuracy)
    print ("Valid loss", valid_loss)

    print ("VALID MAT", valmat)

    


    test_accuracy = correct*1.0/total
    print ("TEST ACCURACY", test_accuracy)
    # print ("TEST MAT", testmat)
    # print ("VALID MAT", valmat)



    save_checkpoint({
        'epoch': e + 1,
        # 'REAL_BOUNDS,': real_bounds,
        # 'predicted_bounds': predicted_bounds,
        # 'REAL_LABELS': real_labels,
        # 'predicted_labels': predicted_labels,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        # 'train_accuracy': train_accuracy,
        # 'train_loss': train_loss,
        # 'valid_accuracy': valid_accuracy,
        # 'valid_loss' : valid_loss,
        # 'test_accuracy' : test_accuracy,
        # 'test_matrix': testmat,
        # 'valid_matrix': valmat,
    },  'models/5_'+str(e+1)+'.tar')

    model.train()


testmat = {0: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 1: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 2: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
            3 : {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 4: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 4: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}}

sum_loss = 0.0
total = 0
correct = 0 
model.eval()

real_bounds = []
predicted_bounds = []

real_labels = []
predicted_labels = []


for batch, (batch_x, batch_y, lengths) in enumerate (test_loader):
    if (batch_x.shape[0] < batch_size):
        break
    # print (lengths)
    # break
    # print ("BATCH X SHAPE:", batch_x.shape)
    init_states = []
    for j in range (3):
        if (j == 2):
            init_state = model.init_out(batch_x.shape[0])
        else:
            init_state = model.init_hidden(batch_x.shape[0])
        init_states.append(init_state)

    waste = np.array(batch_y)
    # lengths = np.asarray([s.shape[0] for s in batch_y], dtype=np.int64)
    max_len = torch.max(lengths).numpy()
    # print ("MAX LEN", int(max_len))

    max_len = int(max_len)
    all_lengths = np.array([i for i in range (max_len)])
    # lens = torch.from_numpy(lengths)
    all_lens = torch.from_numpy(all_lengths)

    all_lens = all_lens.view(max_len,1)

    lens = lengths.view(1,batch_size)

    mask = all_lens < lens
    # print ("MASK SHAPE", mask.shape)
    # print ("BATCH SHAPE", batch_y.shape)
    # print ("MASK", mask)
    # print ("BATCH, ", batch_y)
    # print ("BATCH SHAPE", batch_y.shape)
    # print ("MASK SHAPE",mask.shape)



    batch_x = pad_sequence (batch_x).float().cuda()
    batch_y = pad_sequence (batch_y).long().cuda()

    

    optimizer.zero_grad()
    out = model(batch_x,init_states)
    # print ("OUT SHAPE:", out.shape)
    # print ("BATCH SHAPE",batch_y.shape)
    a = batch_y.shape[0]
    b = batch_y.shape[1]
    mask = mask.transpose(0,1).contiguous()
    batch_y = batch_y.transpose(1,0).contiguous()
    batch_y = batch_y.view(batch_y.shape[0]*batch_y.shape[1]).cuda()
    mask = mask.view (mask.shape[0]*mask.shape[1]).float().cuda()
    # print ("AFTER BATCH SHAPE", batch_y.shape)

    # loss = criterion(out,batch_y)
    # loss = loss*mask
    # loss = torch.sum(loss)
    

    # loss.backward()
    ss = batch_y.cpu().numpy()
    mask = mask.cpu().numpy()
    tot = 0
    cor = 0
    zeros = 0
    predicted = torch.max(out, 1)[1].contiguous()
    predicted = predicted.view(a,b)
    predicted = predicted.transpose(0,1)
    predicted = predicted.contiguous()
    predicted = predicted.view(predicted.shape[0]*predicted.shape[1]).cpu().numpy()
    prevPred = -1
    prevReal = -1
    current = 0
    time_step = 0

    r_bounds = []
    bounds = []
    labels = []
    r_labels = []
    # print ("BATCH Y", batch_y)
    # print ("MASK SHAPE", mask.shape[0])
    # print ("MASK SUM", sum(mask))
    # print ("SUM LENGTHS", lengths.sum())
    
    lengths = lengths.numpy()
    count = 0
    # print ("LENGTHS,", lengths)
    for x in range (mask.shape[0]):
        # print (mask[x])
        if mask[x] == 0:
            # print ("HYE HOO")
            if (current == 1 ):
                # # print ("WOOLOOLO")
                # real_bounds.append(r_bounds)
                # predicted_bounds.append(bounds)
                # real_labels.append(r_labels)
                # predicted_labels.append(labels)

                current = 0
                time_step = 0
            
            continue
        else:
            # if (current == 0):
            #     r_bounds = []
            #     bounds = []
            #     labels = []
            #     r_labels = []
            #     current = 1

            


            time_step += 1
            if (predicted[x] != prevPred):
                labels.append(predicted[x])
                bounds.append(time_step)
                # print ("PREDICTED BOUDNS", bounds)
            
            if (ss[x] != prevReal):
                r_labels.append(ss[x])
                r_bounds.append(time_step)
                # print ("TRUE BOUNDS", r_bounds)
            prevReal = ss[x]
            prevPred = predicted[x]
            tot += 1
            # if (ss[x] == 0):
            #     zeros += 1
            # print (predicted[x],ss[x])
            testmat[ss[x]][predicted[x]] += 1
            if (predicted[x] == ss[x]):
                cor += 1

            if (time_step == lengths[count]):
                prevReal = -1
                prevPred = -1
                count += 1
                time_step = 0
                real_bounds.append(r_bounds)
                predicted_bounds.append(bounds)
                real_labels.append(r_labels)
                predicted_labels.append(labels)
                r_bounds = []
                bounds = []
                labels = []
                r_labels = []


    if (current == 1):
        real_bounds.append(r_bounds)
        predicted_bounds.append(bounds)
        real_labels.append(r_labels)
        predicted_labels.append(labels)

    total += tot
    correct += cor

    # print ("lengths", lengths)
    # print ("length total", lengths.sum())
    # print ("MASK TOTAL", tot)
    # print ("zeros", zeros)
    # acc = (predicted == ss).sum().data.numpy()/ss.shape[0]
    # print ("TEST BATCH ACCURACY:", cor*1.0/tot)
    # print ("REAL BOUNDS,", real_bounds)
    # print ("predicted_bounds", predicted_bounds)
    # print ("REAL LABELS,", real_labels)
    # print ("predicted_LABELS", predicted_labels)
    # break

    # print (predicted.shape)
    # batch_loss = loss.data[0]/sum(mask)
    # print ("VALID LOSS:",batch_loss)

    # torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
    # optimizer.step()
    # sum_loss += batch_loss
    # ave_loss = sum_loss/(i+1)
    

    # if (not (i % 10)):
    print ("TEST BATCH: ",batch)
test_accuracy = correct*1.0/total
# valid_loss = ave_loss
real_bounds = np.array(real_bounds)
predicted_bounds = np.array(predicted_bounds)
real_labels = np.array(real_labels)
predicted_labels = np.array(predicted_labels)

results = {}

results['lr'] = lr
results['weight_decay'] = weight_decay
results['training_loss'] = train_losses
results['validation_loss'] = valid_losses
results['training_accuracy'] = train_accuracies
results['validation_accuracy'] = valid_accuracies
results['confusion_matrix_valid'] = valid_mats
results['confusion_matrix_test'] = testmat
results['real_bounds'] = real_bounds
results['predicted_bounds'] = predicted_bounds
results['real_labels'] = real_labels
results['predicted_labels'] = predicted_labels
results['test_accuracy'] = test_accuracy
pickle.dump(results, open("model6.pkl",'w'))


print ("NUM BOUNDS", real_bounds.shape)
print ("NUM BOUNDS PRED", predicted_bounds.shape)
print ("NUM LABELS", real_labels.shape)
print ("NUM LABELS PRED", predicted_labels.shape)