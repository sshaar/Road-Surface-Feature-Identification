from hmmlearn import hmm
import numpy as np
import warnings
warnings.filterwarnings("ignore")



data = np.load('../data/train7.npz', encoding = "bytes")
test = np.load('../data/test7.npz', encoding = "bytes")
valid = np.load('../data/valid7.npz', encoding = "bytes")

print (valid.keys())


test_labels = test['target']
valid_labels = valid['target']
train_labels = data['target']
train_bounds = data["bounds"]


test = test['feats']
valid = valid['feats']
feat = data['feats']

print 'test', test.shape
print 'valid', valid.shape
print 'feat', feat.shape

##print ("FEAT", feat.shape[0])
train_feat = np.random.choice(data['feats'], 2500)
# np.random.shuffle(train_feat)
# train_feat = data['feats'][:500]
print ("SHAPE", train_feat.shape)

num = train_feat.shape[0]
lengths = []
for x in range (train_feat.shape[0]):
	lengths.append(train_feat[x].shape[0])

# seq_len = feat.shape[1]
print ("feat shape", feat.shape)
print ("seqlen", len(lengths))


cat_feat = np.concatenate(train_feat, axis=0)

print ('Loaded data')

print ("TRAINING HMM")
GMMHMM = hmm.GMMHMM(n_components=7, n_mix=8, n_iter=20,
                            verbose=True)

GMMHMM.fit(cat_feat, lengths)

print ("PREDICTED SEQUENCE")


print ("TRAINING_SEQS")
train_seqs = []
for i in range (feat.shape[0]):
    opt_seq = GMMHMM.predict(feat[i])
    train_seqs.append(opt_seq)


print ("TESTING_SEQS")
test_seqs = []
for i in range (test.shape[0]):
    opt_seq = GMMHMM.predict(test[i])
    test_seqs.append(opt_seq)


print ("VALID_SEQS")
valid_seqs = []
for i in range (valid.shape[0]):
    opt_seq = GMMHMM.predict(valid[i])
    valid_seqs.append(opt_seq)



train_seqs = np.array(train_seqs)
test_seqs = np.array(test_seqs)
valid_seqs = np.array(valid_seqs)


print ("TRAIN SHAPE", train_seqs.shape)
print ("TEST SHAPE", test_seqs.shape)
print ("VALID SHAPE", valid_seqs.shape)


np.save("../data/train_seqs7", train_seqs)
np.save("../data/test_seqs7", test_seqs)
np.save("../data/valid_seqs7", valid_seqs)

# loaded_seqs = np.load("../../data/train_seqs4.npy")
# print ("loaded shape", loaded_seqs.shape)


##from hmm.continuous.GMHMM import GMHMM
