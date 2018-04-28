import csv
import sys

import numpy as np
import random


INDICES = np.array(range(1, 16))
FILE_LABELS = sys.argv[1]
FILE_LABELS_DEV = sys.argv[2]
OUTPUT_FILE = 'data.npz'
DATA_PROCESSED = 'processed.npz'
WINDOW_SIZE = 15

TRAIN_N = 5000
VALID_N = 1000
TEST_N = 2000

TRAIN_FILE = 'train7.npz'
VALID_FILE = 'valid7.npz'
TEST_FILE = 'test7.npz'

def get_data_from_file(file_name):

    csvfile = open('/home/sshaar/hmm-rnn/road/data/'+file_name)
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

    wanted_data_all = []

    for i, row in enumerate(spamreader):
        if i == 0:
            continue
        row = list(filter(lambda x: len(x)>1, row))
        # print (row)
        row = list(map(eval, row))
        if (len (row) < len(INDICES)):
            continue
        wanted_data = np.array(row)
        wanted_data_all.append(wanted_data[INDICES])

    return np.array(wanted_data_all)

def get_all_data(file_labels, output_file):

    csvfile = open(file_labels)
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

    feats = []
    targets = []

    for row in spamreader:
        # print (row)
        instance_data = get_data_from_file(row[0])
        instance_label = int(row[1])

        feats.append(instance_data)
        targets.append(instance_label)
        # break

    feats = np.array(feats)
    targets = np.array(targets)

    return (feats, targets)

def collect_data(feats, targets):

    dic = {}

    for i in range(feats.shape[0]):
        f = feats[i]
        l = targets[i]
        if l in dic.keys():
            dic[l].append(np.array(f))
        else:
            dic[l] = [np.array(f)]

    for i in dic.keys():
        dic[i] = np.array(dic[i])
    return dic

def get_average_size(llist):
    cpy = []
    for i in range(llist.shape[0]):
        cpy.append(llist[i].shape[0])
    print cpy
    return np.mean(np.array(cpy))


def split_normal_offroad(dic):

    all_normal = np.concatenate(np.array(dic[0]))
    print all_normal.shape

    segment_length = int(0.75*get_average_size(dic[3]))
    print segment_length

    all_normal = all_normal[:(all_normal.shape[0]//segment_length) * segment_length]
    all_normal = all_normal.reshape(all_normal.shape[0]//segment_length, segment_length, all_normal.shape[-1])

    all_offroad = np.concatenate(np.array(dic[1]))

    all_offroad = all_offroad[:all_offroad.shape[0]//segment_length * segment_length]
    all_offroad = all_offroad.reshape(all_offroad.shape[0]//segment_length, segment_length, all_offroad.shape[-1])
    print all_normal.shape
    print all_offroad.shape
    dic[0] = all_normal
    dic[1] = all_offroad
    return dic

def generate_data(dic, N, f):

    feats = []
    all_boundries = []
    targets = []
    for i in range(N):

        if (i+1)%1000 == 0:
            print 'Dont with', i+1, '/', N

        segment_length = random.choice(range(4,6))
        sequence = []
        labels = []
        boundries = [0]
        for j in range(segment_length):
            if j%6 == 0:
                c = 0
                # labels.append(c)

            else:
                # c = random.choice([1,2,3,4,5,6])
                c = random.choice([1,2,3,4])
                # c = random.choice([2,3,4, 5, 6])
                # labels.append(c-1)

            labels.append(c)

            ii = random.choice(range(dic[c].shape[0]))
            # print i, j, ii, c, dic[c].shape, (dic[c][ii].shape), boundries[j]
            instance = dic[c][ii]
            noised_instance = instance
            # noised_instance = instance + np.random.gumbel(0, 1e-2, instance.shape)
            sequence.append(noised_instance)

            if j < (segment_length - 1):
                boundries.append(sequence[j].shape[0] + boundries[-1])

        # print(len(boundries), boundries[-1])
        feats.append(np.concatenate(np.array(sequence), axis=0))
        # print(np.array(sequence).shape, feats[i].shape,  boundries[-1])
        all_boundries.append(np.array(boundries))
        targets.append(np.array(labels))

    print feats[0].shape
    print targets[0]
    # print feats[0][1].shape

    feats, all_boundries, targets = (np.array(feats), np.array(all_boundries), np.array(targets))
    np.savez(f, feats=feats, target=targets, bounds=all_boundries)
    print feats.shape, targets[-1]
    return (np.array(feats), np.array(all_boundries), np.array(targets))

def segregate_data(feats):

    n_files = feats.shape[0]
    all_feats = []
    for i in range(n_files):
        # if i < 2:
        #     continue
        n_rows = feats[i].shape[0] ## number of rows in current file
        rows = feats[i] ## rows of current file

        # print ("BEFORE WINDOW", rows.shape)
        ## concats into windows of WINDOW_SIZE
        windowed = np.array([rows[i:i + WINDOW_SIZE] for i in range(0, len(rows)-WINDOW_SIZE, WINDOW_SIZE)])
        # print windowed.shape

        ## reshapes into sizes of windows. each element has shape WINDOW_SIZE*len(INDICES)
        windowed = np.reshape(windowed, (windowed.shape[0], WINDOW_SIZE*len(INDICES)))
        # print ("AFTER WINDOW", windowed.shape)
        window_rows = windowed.shape[0]

        all_feats.append(windowed)
        # if (i == 1):
        #     break

    all_feats = np.array(all_feats)
    # print ("FEAT SHAPE:", all_feats.shape)
    return (all_feats)

def split_dictionary(d):

    PERC = 0.75

    train = d.copy()
    valid = d.copy()
    test = d.copy()
    for i in d:
        data = d[i]
        random.shuffle(data)

        train[i] = data[:int(len(data)*PERC)]
        valid[i] = data[int(len(data)*PERC):]
        test[i] = data[int(len(data)*PERC):]

    return (train, valid, test)

def split_dictionary_2(d1,d2):

    PERC = 0.75

    train = d1.copy()
    valid = d1.copy()
    test = d1.copy()
    for i in d1:
        data = d1[i]
        data = np.concatenate((d1[i], d2[i]), axis=0)
        random.shuffle(data)

        train[i] = data[:int(len(data)*PERC)]
        valid[i] = data[int(len(data)*PERC):]
        test[i] = data[int(len(data)*PERC):]

    return (train, valid, test)

# def context(feats):
#
#     for i in range(feats.shape[0]):
#         instance = feats[i]
#         for j in range(instance.shape[0]):
#             start_index = j-WINDOW_SIZE
#             last_index = j+WINDOW_SIZE+1
#             new_instance = np.zeros((2*WINDOW_SIZE+1, WINDOW_SIZE*instance.shape[1]), dtype=np.float64)
#
#             cur = start_index
#             for k in (range(2*WINDOW_SIZE+1)):
#                 if k < j:
#
#                 elif k == j:
#
#                 else:
#
#                 new_instance[k]
#
#
#             instance[j] =

feats, targets = get_all_data('/home/sshaar/hmm-rnn/road/'+FILE_LABELS, OUTPUT_FILE)
feats = segregate_data(feats)
print feats.shape
print targets.shape
dic = collect_data(feats, targets)
dic = split_normal_offroad(dic)
for i in dic:
    print 'class', i, len(dic[i])

# feats, targets = get_all_data('/home/sshaar/hmm-rnn/road/'+FILE_LABELS_DEV, OUTPUT_FILE)
# feats = segregate_data(feats)
# print feats.shape
# print targets.shape
# test_dic = collect_data(feats, targets)
# test_dic = split_normal_offroad(test_dic)

train_dic, valid_dic, test_dic = split_dictionary(dic)
# train_dic, valid_dic, test_dic = split_dictionary_2(dic, test_dic)

for i in train_dic:
    print 'Train class', i, len(train_dic[i])

for i in test_dic:
    print 'Test class', i, len(test_dic[i])

generate_data(train_dic, TRAIN_N, TRAIN_FILE)
generate_data(valid_dic, VALID_N, VALID_FILE)
generate_data(test_dic, TEST_N, TEST_FILE)
