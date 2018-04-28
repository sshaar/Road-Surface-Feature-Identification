import matplotlib.pyplot as plt
import pickle
import numpy

results = pickle.load( open( "model6.pkl", "r" ) )

lr = results['lr'] 
weight_decay = results['weight_decay'] 
train_losses = results['training_loss'] 
valid_losses = results['validation_loss']
train_accuracies = results['training_accuracy'] 
valid_accuracies = results['validation_accuracy'] 
valid_mats = results['confusion_matrix_valid'] 
testmat = results['confusion_matrix_test'] 
real_bounds = results['real_bounds'] 
predicted_bounds = results['predicted_bounds'] 
real_labels = results['real_labels'] 
predicted_labels = results['predicted_labels'] 
test_acc = results['test_accuracy']


for x in range (len (train_losses)):
	train_losses[x] = float(train_losses[x].cpu().data[0])
	valid_losses[x] = float(valid_losses[x].cpu().data[0])
	train_accuracies[x] = train_accuracies[x]*100
	valid_accuracies[x] = valid_accuracies[x]*100

epochs = [i+1 for i in range (30)]
# print (epochs)
print ("LEN TRAINIGN", len(train_accuracies))


# print ("train_losses", train_losses)
print ("lr", lr)
print ("weight_decay", weight_decay)
print ("train_losses", train_losses)
print ("train_accuracies", train_accuracies)
print ("valid_losses", valid_losses)
print ("valid_accuracies", valid_accuracies)
print ("testmat", testmat)
print ("test_acc", test_acc)

fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
ax.plot(epochs, train_accuracies)
# fig.savefig('model6_train_accuracies.png')   # save the figure to file
ax.set_xlabel('Number of epochs')
ax.set_ylabel('Accuracy (%)')
ax.set_title("Training accuracy against number of epochs")
plt.savefig('images/model6_train_accuracies.png')
plt.close(fig)

fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
ax.plot(epochs, train_losses)
# fig.savefig('model6_train_accuracies.png')   # save the figure to file
ax.set_xlabel('Number of epochs')
ax.set_ylabel('Loss')
ax.set_title("Training loss against number of epochs")
plt.savefig('images/model6_train_losses.png')
plt.close(fig)   


fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
ax.plot(epochs, valid_accuracies)
# fig.savefig('model6_train_accuracies.png')   # save the figure to file
ax.set_xlabel('Number of epochs')
ax.set_ylabel('Accuracy (%)')
ax.set_title("Valid accuracy against number of epochs")
plt.savefig('images/model6_valid_accuracies.png')
plt.close(fig)   


fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
ax.plot(epochs, valid_losses)
# fig.savefig('model6_train_accuracies.png')   # save the figure to file
ax.set_xlabel('Number of epochs')
ax.set_ylabel('Loss')
ax.set_title("Valid loss against number of epochs")
plt.savefig('images/model6_valid_losses.png')
plt.close(fig)   




