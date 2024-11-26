from AlexNet import Model

import torch
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim import SGD

# step1: Load the test set of CIFAR10
# # change the size of CIFAR10 into 227*227*3
# transform_data = transforms.Compose([transforms.Resize((227,227)),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


dataset = CIFAR10('./test', train=False, transform=ToTensor(), download=True)
print(dataset)
print(f"the number of data entries: {len(dataset)}")
print(f"the number of classes: {len(dataset.classes)}")
print(dataset.classes)
print(Counter(dataset.targets))


# randomly split them into 7000 training_set, 2000 val_set and 1000 test_set
train_dataset, val_dataset, test_dataset = random_split(dataset=dataset, lengths=[7000, 2000, 1000])
print(f"Length of train dataset:{len(train_dataset)}")
print(f"Length of validation dataset:{len(val_dataset)}")
print(f"Length of test dataset:{len(test_dataset)}")

epoch = 50
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# to show the image size
train_features, train_labels= next(iter(train_loader))
print(f"image size: {train_features.size()}")

# load the model and prepare for training
model = Model()

load_model = True
cost = CrossEntropyLoss()
sgd = SGD(model.parameters(), lr=0.005, momentum=0.9)
# use adam for the second optimizor
# adam = optim.Adam(model.parameters(), lr=0.005)

# prepare some lists to contain all the history values for train and val
train_losses = []
val_losses = []
train_accs = []
val_accs = []

# start training the model
for epoch in range(epoch):
    model.train()

    print(f"round of epoch: {epoch+1}")

    loss_epoch = 0
    acc_epoch = 0

    val_loss_epoch = 0
    val_acc_epoch = 0

    # perform model training
    for idx, (train_x, train_labels) in enumerate(train_loader):

        label_np = np.zeros((train_labels.shape[0], 10))

        # Forward Pass
        sgd.zero_grad()
        # use adam as the optimizor
        # adam.zero_grad()

        # forward, backward pass with parameter update
        predict_y = model(train_x)

        # calculate the train loss
        train_loss = cost(predict_y, train_labels)

        # calculate the accuracy
        predict_ys = np.argmax(predict_y.data, axis=-1)
        predict_result = predict_ys == train_labels
        train_acc = predict_result.sum().item() / predict_result.shape[0]

        if idx % 10 == 0:
            print('batch idx: {}, loss: {}, acc: {}%'.format(idx, train_loss.sum().item(), train_acc * 100))

        # backward propagation
        train_loss.backward()

        # update 'W', 'b'
        sgd.step()
        # use adam as the optimizor
        # adam.step()

        # calculate the total loss and accuracy
        loss_epoch += train_loss.item()
        acc_epoch += train_acc

    # calculate the average acc and loss, and load them into the list
    avg_loss_epoch = loss_epoch/len(train_loader)
    train_losses.append(avg_loss_epoch)

    avg_acc_epoch = acc_epoch/len(train_loader)
    train_accs.append(avg_acc_epoch)

    # perform model validation
    model.eval()

    correct = 0
    _sum = 0

    for idx, (val_x, val_labels) in enumerate(val_loader):

        # calculate the accuracy
        predict_y = model(val_x)
        _, predict_ys = predict_y.max(1)
        correct += (predict_ys == val_labels).sum()
        _sum += predict_ys.size(0)

        # calculate the loss
        val_loss = cost(predict_y, val_labels)
        # load the total loss
        val_loss_epoch += val_loss.item()

    # calculate the average loss and accuracy and load them into the list
    avg_loss_epoch = val_loss_epoch/len(val_loader)
    val_losses.append(avg_loss_epoch)

    avg_acc_epoch = correct/_sum
    val_accs.append(avg_acc_epoch)

    print(f"In validation datasets: {correct} / {_sum} with accuracy {correct/_sum}")

num_corrects = 0
num_samples = 0

# load the true label and predict label into list
Y_true = []
Y_predict = []

for idx, (test_x, test_labels) in enumerate(test_loader):
    predict_y = model(test_x)
    _, predict_ys = predict_y.max(1)
    num_corrects += (predict_ys == test_labels).sum()
    num_samples += predict_ys.size(0)

    # load the labels into the list
    Y_true.extend(test_labels.view(-1).numpy())
    Y_predict.extend(predict_ys.view(-1).numpy())

# show the accuracy for test datasets
print(f"In test datasets: {num_corrects} / {num_samples} with accuracy {num_corrects/num_samples} ")
# show the confusion matrix
cm = confusion_matrix(Y_true, Y_predict)
print(cm)

# show the history value of acc and loss
print(f"train_losses: {train_losses}")
print(f"val_losses: {val_losses}")
print(f"train_accs: {train_accs}")
print(f"val_accs: {val_accs}")

# plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.title("Training and Validation Loss")
plt.plot(val_losses, label="val_losses")
plt.plot(train_losses, label="train_losses")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# plot the training and validation accuracy
plt.figure(figsize=(10, 5))
plt.title("Training and Validation Accuracy")
plt.plot(val_accs, label="val_accs")
plt.plot(train_accs, label="train_accs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()