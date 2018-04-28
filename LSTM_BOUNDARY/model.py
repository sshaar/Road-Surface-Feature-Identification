
import numpy as np
# import pickle
# import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch
# import torch.utils.data as dt
# from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import*
# import shutil
import os
# import numpy as np
# import random

def sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))

def init_hidden(batch_size, hidden_size):
    
    return Variable(torch.zeros(1,batch_size, hidden_size)),Variable(torch.zeros(1,batch_size, hidden_size))

def init_out (batch_size, embed_size):
    return Variable(torch.zeros(1,batch_size, embed_size)),Variable(torch.zeros(1,batch_size, embed_size))        

class LockedDropout(nn.Module):

    def forward(self, data, is_training=True, dropout=0.5):
        if not is_training or not dropout:
            return data

        mask = data.data.new(1, data.size(1), data.size(2)).bernoulli_(1 - dropout)
        mask = Variable(mask, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(data).cuda()

        return mask * data

class Model(nn.Module):

    def __init__(self, inp_size, linear_size, hidden_size,  dropouts, n_classes, training):
        super(Model, self).__init__()

        
        # super(Model, self).__init__()
        self.training = training
        self.inp_size = inp_size
        # self.em_size = em_size
        self.hidden_size = hidden_size
        self.linear_size = linear_size
        # self.unigram = unigram

        # self.word_dropout = dropouts[0]
        self.lock_dropout = dropouts[1]
        self.out_dropout = dropouts[2]
        # self.em_dropout = dropouts[3]


        # self.embedding = nn.Embedding(inp_size, em_size)

        # self.rnns = nn.ModuleList([
        #     nn.LSTM(input_size=em_size, hidden_size=hidden_size,batch_first = True),
        #     nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,batch_first = True),
        #     nn.LSTM(input_size=hidden_size, hidden_size=em_size,batch_first =True)])

        self.rnns = nn.ModuleList([
            nn.LSTM(input_size=inp_size, hidden_size=hidden_size, batch_first = False),
            nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first = False),
            nn.LSTM(input_size=hidden_size, hidden_size=linear_size, batch_first = False)])

        self.linear = nn.Linear(linear_size, linear_size)
        self.projection = nn.Linear(linear_size,n_classes)
        self.lock_drop = LockedDropout()


        ## internal layers initialization
        weight_1 = -(1/(hidden_size**0.5))
        weight_2 = 1/(hidden_size**0.5)
        for layer in self.rnns:
            ## snippet from: https://discuss.pytorch.org/t/initializing-parameters-of-a-multi-layer-lstm/5791
            for name, param in layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant(param, 0.0)
                elif 'weight' in name:
                    nn.init.uniform(param, weight_1, weight_2)

        # ## EMBEDDING INITIALIZATION
  #       for name, param in self.embedding.named_parameters():
  #           if 'bias' in name:
  #               nn.init.constant(param, 0.0)
  #           elif 'weight' in name:
  #               nn.init.uniform(param, weight_1, weight_2)

  #       self.embedding.weight = self.projection.weight


        ## PROJECTION INITIALIZATION
        for name, param in self.projection.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
                # if (len(unigram) != 0):
                #     self.projection.bias.data = torch.from_numpy(unigram)
            elif 'weight' in name:
                nn.init.uniform(param, -0.1, 0.1)


    ## snippet from: https://discuss.pytorch.org/t/when-to-initialize-lstm-hidden-state/2323/3
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(1, bsz, self.hidden_size).zero_()),
                Variable(weight.new(1, bsz, self.hidden_size).zero_()))
   
    def init_out (self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(1, bsz, self.linear_size).zero_()),
                Variable(weight.new(1, bsz, self.linear_size).zero_()))
    
    def init_all (self, batch_size, hidden_size, em_size):
        weight = next(self.parameters()).data
        states = []
        for i in range(3):
            if i <2:
                new_state = (Variable(weight.new(1, batch_size, hidden_size).zero_()),
                            Variable(weight.new(1, batch_size, hidden_size).zero_()))
            else:
                new_state = (Variable(weight.new(1, batch_size, linear_size).zero_()),
                            Variable(weight.new(1, batch_size, linear_size).zero_()))
                
            states.append(new_state)
        return states


    def forward(self, input, init_states, forward=0, stochastic=False):

        h = input  # (n, t)
        # h = self.lock_drop(input, self.training, self.word_dropout)
        # h = self.embedding(input)
        # h = self.embedding(input)
        # h = self.lock_drop(h, self.training, self.em_dropout) 
        states = []
        i = 0
        for rnn in self.rnns:
            init_state = init_states[i]    
            h, state = rnn(h, init_state)
            states.append(state)
            if (i == 2):
                h = self.lock_drop(h, self.training, self.lock_dropout)
            else:
                h = self.lock_drop(h, self.training, self.out_dropout)
            i += 1
        
        h = self.linear(h)
        h = self.lock_drop(h, self.training, self.out_dropout)
        h = self.projection(h)
        print ("projection SHAPE", h.shape)


        n = h.shape[0]
        m = h.shape[1]
        o = h.shape[2]
        flat = h.view(n*m,o)
        print (flat.shape)

        # if stochastic:
        #     gumbel = Variable(sample_gumbel(shape=h.size(), out=h.data.new()))
        #     h += gumbel
        # logits = h
        # if forward > 0:
        #     outputs = []
        #     h = torch.max(logits[:, -1:, :], dim=2)[1] 
        #     for i in range(forward):
        #         h = self.embedding(h)
        #         for j, rnn in enumerate(self.rnns):
        #             h, state = rnn(h, states[j])
        #             states[j] = state
        #         h = self.projection(h)
        #         if stochastic:
        #             gumbel = Variable(sample_gumbel(shape=h.size(), out=h.data.new()))
        #             h += gumbel
        #         outputs.append(h)
        #         h = torch.max(h, dim=2)[1]
        #     logits = torch.cat([logits] + outputs, dim=1)
            
        #     flat = logits
    
        return flat



