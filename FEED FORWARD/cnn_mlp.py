#!/usr/bin/env python
"""
Example of using Keras to implement a 1D convolutional neural network (CNN) for timeseries prediction.
"""

from __future__ import print_function, division
import numpy as np
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, Activation
from keras.models import Sequential
from keras import optimizers
import keras
import pickle


def create_model():
    rate = 0.8
    model=Sequential()

    # model.add(Dense(1024, input_dim = 225))
    # model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='valid',input_dim=15, input_length = 1))  
    # model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

    # model.add(Activation('relu'))
    # # model.add(keras.layers.Dropout(rate, noise_shape=None, seed=None))

    # # model.add(MaxPooling1D(pool_length=2))  

    # model.add(Convolution1D(nb_filter=64, filter_length=3, border_mode='valid'))  

    # model.add(Activation('relu'))  
    # # model.add(keras.layers.Dropout(rate, noise_shape=None, seed=None))

    # model.add(Convolution1D(nb_filter=32, filter_length=1, border_mode='valid'))  

    # model.add(Activation('relu'))  
    # model.add(keras.layers.Dropout(rate, noise_shape=None, seed=None))
    
    # # model.add(MaxPooling1D(pool_length=2))  

    # model.add(Flatten())  
    # model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

    model.add(Dense(256, input_dim = 150) )

    model.add((Activation('relu')))  
    model.add(keras.layers.Dropout(rate, noise_shape=None, seed=None))
    # model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))


    model.add(Dense(256))  
    model.add((Activation('relu')))
    model.add(keras.layers.Dropout(rate, noise_shape=None, seed=None))
    # model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))


    model.add(Dense(512))  
    model.add((Activation('relu')))
    model.add(keras.layers.Dropout(rate, noise_shape=None, seed=None))

    model.add(Dense(512))  
    model.add((Activation('relu')))
    # model.add(keras.layers.Dropout(0, noise_shape=None, seed=None))
    # model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))



    model.add(Dense(512))  
    model.add((Activation('relu')))
    # model.add(keras.layers.Dropout(rate, noise_shape=None, seed=None))
    # model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))


    # # model.add(keras.layers.Dropout(rate, noise_shape=None, seed=None))

    # model.add(Dense(512))  
    # model.add((Activation('relu')))
    # model.add(keras.layers.Dropout(rate, noise_shape=None, seed=None))
    # # model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

    

    model.add(Dense(5))  
    model.add(Activation('softmax'))  
    opt = optimizers.Adam(lr = 0.001, decay = 0.00001)
    # sgd = SGD(lr=0.1, momentum=0.9, nesterov=True, decay=1e-6)  
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']) 
    return model



# print ("ALL DATA SHAPE", all_data.shape)
# print ("TEST data shape", test_data.shape)

# feats = np.concatenate([all_data['feat'], test_data['feat']], axis = 0)
# labels = np.concatenate([all_data['target'].astype(int), test_data['target'].astype(int)], axis = 0)
# boundary= 10000

# feats = all_data["feat"]
# labels = all_data["target"].astype(int)
# indices = np.array([i for i in range (labels.shape[0])])
# np.random.shuffle(indices)
# feats = feats[indices]
# labels = labels[indices]
boundary= 5000
n_classes = 5

all_data = np.load("../train_data/train.npz")
valid_data = np.load("../train_data/valid.npz")
test_data = np.load("../train_data/test.npz")

# feats = feats.reshape(feats.shape[0],5,30)
# labels = labels.reshape()
train_feats = all_data["feat"]
train_labels = all_data["target"].astype(int)

print ("TRAIN FEAT SHAPE,", train_feats.shape)
print ("TRAIN LABEL SHAPE, ", train_labels.shape) 

test_feats = test_data["feat"]
test_labels = test_data["target"].astype(int)



valid_feats = valid_data["feat"]
valid_labels = valid_data["target"].astype(int)

print ("VALID FEAT SHAPE,", valid_feats.shape)
print ("VALID LABEL SHAPE, ", valid_labels.shape) 

print ("TEST FEAT SHAPE,", test_feats.shape)
print ("TEST LABEL SHAPE, ", test_labels.shape) 



learning_rate = 0.01
training_epochs = 30
batch_size = 64
display_step = 1


train_losses = []
valid_losses = []
valid_accs = []
train_accs = []
val_confusions = []
model = create_model()

for epoch in range(training_epochs):
    print ("RUNNING Epoch", epoch + 1)

    # feats = all_data["feat"]
    # labels = all_data["target"]
    n = train_labels.shape[0]
    indices = np.array([i for i in range (n)])
    np.random.shuffle(indices)
    train_feats = train_feats[indices]
    train_labels = train_labels[indices]

    m = valid_labels.shape[0]
    indices = np.array([i for i in range (m)])
    np.random.shuffle(indices)
    valid_feats = valid_feats[indices]
    valid_labels = valid_labels[indices]         


    avg_cost = 0.
    cost = 0
    allAcc = []
    allLoss = []

    val_cost = 0

    num_batches_v = int(n/batch_size)
    num_batches_d = int(m/batch_size)
    # Loop over all batches
    for batch in range(num_batches_v):
        if (batch == num_batches_v - 1):
            indexes = [i for i in range (batch*batch_size, n)]
        else:
            indexes = [i for i in range(batch * batch_size, (batch + 1) * batch_size)]

        batch_x = train_feats[indexes]
            # print ("train_labels[indexes]",train_labels[indexes])
        batch_y = np.eye(n_classes)[train_labels[indexes]]


        model.train_on_batch(batch_x, batch_y) 
        a = model.evaluate(batch_x, batch_y, verbose = 0)
        allLoss.append(a[0])
        # print (a)
        allAcc.append(a[1])

    print ("train losses:", np.mean(allLoss))
    print ("TRAIN ACCURACY", np.mean(allAcc))
    train_accs.append(np.mean(allAcc))
    train_losses.append(np.mean(allLoss))

    mat = {0: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 1: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 2: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
            3 : {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 4: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 4: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 
            5 : {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 6 : {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}}


    allAcc = []
    allLoss = []
    for batch in range(num_batches_d):
        if (batch == num_batches_d - 1):
            indexes = [i for i in range (batch*batch_size, m)]
        else:
            indexes = [i for i in range(batch * batch_size, (batch + 1) * batch_size)]

        batch_x = valid_feats[indexes]
        batch_s = valid_labels[indexes]
        batch_y = np.eye(n_classes)[valid_labels[indexes]]
        a = model.evaluate(batch_x, batch_y, verbose = 0)
        p = model.predict_classes(batch_x)
        for h in range (len(batch_s)):
                # if (p[h] == 4)
            mat[batch_s[h]][p[h]] += 1
        # print ("PREDICTED SHAPE", b.shape)

        # break
        allLoss.append(a[0])
        # print (a)
        allAcc.append(a[1])
    # print(allAcc)
    # print (valid_losses)
    print ("valid losses:", np.mean(allLoss))
    print ("VALID ACCURACY", np.mean(allAcc))
    print ("______________________________________________")
    valid_accs.append(np.mean(allAcc))
    valid_losses.append(np.mean(allLoss))
    val_confusions.append(mat)





m = test_labels.shape[0]
# indices = np.array([i for i in range (m)])
# np.random.shuffle(indices)
# test_feats = test_feats[indices]
# test_labels = test_labels[indices]         


avg_cost = 0.
cost = 0
allAcc = []
allLoss = []

test_cost = 0

num_batches_t = int(m/batch_size)
# num_batches_d = int(m/batch_size)
# Loop over all batches

testmat = {0: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 1: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 2: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
        3 : {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 4: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 4: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 
        5 : {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, 6 : {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}}


# allAcc = []
# allLoss = []
for batch in range(num_batches_t):
    if (batch == num_batches_t - 1):
        indexes = [i for i in range (batch*batch_size, m)]
    else:
        indexes = [i for i in range(batch * batch_size, (batch + 1) * batch_size)]

    batch_x = test_feats[indexes]
    batch_s = test_labels[indexes]
    batch_y = np.eye(n_classes)[test_labels[indexes]]
    a = model.evaluate(batch_x, batch_y, verbose = 0)
    p = model.predict_classes(batch_x)
    for h in range (len(batch_s)):
            # if (p[h] == 4)
        testmat[batch_s[h]][p[h]] += 1
    # print ("PREDICTED SHAPE", b.shape)

    # break
    # allLoss.append(a[0])
    # # print (a)
    allAcc.append(a[1])
# print(allAcc)
# print (valid_losses)
# print (" losses:", np.mean(allLoss))
test_accuracy = np.mean(allAcc)
print ("TEST ACCURACY", np.mean(allAcc))
print ("______________________________________________")
# valid_accs.append(np.mean(allAcc))
# valid_losses.append(np.mean(allLoss))
# val_confusions.append(mat)

results = {}
results['lr'] = 0.001
results['weight_decay'] = 0.001
results['training_loss'] = train_losses
results['validation_loss'] = valid_losses
results['training_accuracy'] = train_accs
results['validation_accuracy'] = valid_accs
results['confusion_matrix_valid'] = val_confusions
results['confusion_matrix_test'] = testmat
# results['real_bounds'] = real_bounds
# results['predicted_bounds'] = predicted_bounds
# results['real_labels'] = real_labels
# results['predicted_labels'] = predicted_labels
results['test_accuracy'] = test_accuracy
pickle.dump(results, open("3mlp10.pkl",'wb'))





# np.save("vallosses1.npy", valid_losses)
# np.save("valaccs1.npy", valid_accs)
# np.save("valConfs1.npy", val_confusions)
# np.save("train_losses1.npy", valid_losses)
# np.save("train_accs1.npy", valid_accs)
# np.save("valConfs.npy", val_confusions)

        