import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

class StackNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(StackNet, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.layer1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(input_size, 768, kernel_size=5, padding=2),
            nn.BatchNorm1d(768),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv1d(768, 768, kernel_size=5, padding=2),
            nn.BatchNorm1d(768),
            nn.ReLU())

        self.layer3 = nn.Sequential(
            nn.Conv1d(768, 768, kernel_size=5, padding=2),
            nn.BatchNorm1d(768),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv1d(768, 768, kernel_size=3, padding=1),
            nn.BatchNorm1d(768),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv1d(768, 768, kernel_size=3, padding=1),
            nn.BatchNorm1d(768),
            nn.ReLU())

        self.layer6 = nn.Sequential(
            nn.Conv1d(768, 768, kernel_size=3, padding=1),
            nn.BatchNorm1d(768),
            nn.ReLU())

        self.layer7 = nn.Sequential(
            nn.Conv1d(768, 768, kernel_size=3, padding=1),
            nn.BatchNorm1d(768),
            nn.ReLU())

        self.layer8 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv1d(768, 768, kernel_size=1, padding=0),
            nn.BatchNorm1d(768),
            nn.ReLU())

        self.layer9 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv1d(768, output_size, kernel_size=1, padding=0),
            nn.BatchNorm1d(output_size),
            nn.ReLU())

        self.layer1 = self.layer1.cuda().double()
        self.layer2 = self.layer2.cuda().double()
        self.layer3 = self.layer3.cuda().double()
        self.layer4 = self.layer4.cuda().double()
        self.layer5 = self.layer5.cuda().double()
        self.layer6 = self.layer6.cuda().double()
        self.layer7 = self.layer7.cuda().double()
        self.layer8 = self.layer8.cuda().double()
        self.layer9 = self.layer9.cuda().double()

        self.layer_initialize(self.layer1)
        self.layer_initialize(self.layer2)
        self.layer_initialize(self.layer3)
        self.layer_initialize(self.layer4)
        self.layer_initialize(self.layer5)
        self.layer_initialize(self.layer6)
        self.layer_initialize(self.layer7)
        self.layer_initialize(self.layer8)
        self.layer_initialize(self.layer9)

    def layer_initialize(self, layer):
        for l in layer:
            if isinstance(l, nn.Conv1d):
                nn.init.xavier_normal(l.weight.double().cuda())
                l.bias.data.fill_(0)
                # print ("Weight", l.weight)

    def forward(self, feat, assign, DEEPNET):
        feat = Variable(torch.from_numpy(feat), requires_grad=True).double()
        if DEEPNET:
            feat = feat.cuda()

        output = self.layer1(feat)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        output = self.layer6(output)
        output = self.layer7(output)
        output = self.layer8(output)
        output = self.layer9(output) #(batch_size, 5, maxtime)
        # Pooling
        output = output.permute(0, 2, 1) #(batch_size, maxtime, 5)

        # print (output)

        hot = Variable(torch.from_numpy(np.delete(onehot_2dto3d(assign), 0, axis=1)), requires_grad=True) #(batch_size, maxphon, maxtime)
        if DEEPNET:
            hot = hot.cuda().double()


        sum_activ = torch.bmm(hot, output)
        count_activ = torch.sum(hot, dim=2)

        all_ones = Variable(torch.from_numpy(np.ones(count_activ.shape)), requires_grad=True).double()
        if DEEPNET:
            all_ones = all_ones.cuda()

        pooled_output = sum_activ/torch.max(count_activ, all_ones).view(count_activ.shape[0], count_activ.shape[1], 1)

        # return pooled_output.view(pooled_output.shape[0]*pooled_output.shape[1], 46)
        return pooled_output

def onehot_2dto3d(a):

    # the 3d array that will be the one-hot representation
    # a.max() + 1 is the number of labels we have
    b = np.zeros((a.shape[0], a.max() + 1, a.shape[1]), dtype=np.float32)

    # if you visualize this as a stack of layers,
    # where each layer is a sample,
    # this first index selects each layer separately
    layer_idx = np.arange(a.shape[0]).reshape(a.shape[0], 1)

    # this index selects each component separately
    component_idx = np.tile(np.arange(a.shape[1]), (a.shape[0], 1))

    # then we use `a` to select indices according to category label
    b[layer_idx, a, component_idx] = 1.

    return b
