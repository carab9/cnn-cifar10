#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sagemaker import get_execution_role

role = get_execution_role()
print(role)


# # **Training a CNN to classify the CIFAR-10 dataset**
# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The accuracy of the small model from scratch is around 88.67% by 30 training epochs. This model has a VGG architecture and uses techniques such as L2 regularization, reducing learning rate, early stopping, and data augumentation to increase the accuracy. Data loading time is around 1 second.

# In[2]:


# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import time
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
torch.manual_seed(17)


# In[3]:


if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('Running on GPU')
    print(torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print('Running on CPU')


# ## Data processing

# In[4]:


# for data augmentation
transform = torchvision.transforms.Compose([
    torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.CIFAR10),
    torchvision.transforms.ToTensor()])


# In[5]:


# load the CIFAR 10 training and testing data sets from a directory
root_dir = 'cifar-10-datasets'
t1 = time.time()
train_dataset = torchvision.datasets.CIFAR10(root=root_dir, transform=transform, download=False)
test_dataset = torchvision.datasets.CIFAR10(root=root_dir, train=False, transform=torchvision.transforms.ToTensor(), download=False)
print("Loading time:", time.time()-t1)


# In[6]:


# unpickle data function
import pickle
def unpickle(file):
  with open(file, "rb") as infile:
    data = pickle.load(infile, encoding="latin1")
  return data

# raw data before reshaped and transposed
batch1 = unpickle(root_dir + '/cifar-10-batches-py/data_batch_1')
print(batch1.keys())
print(batch1['data'])
print(batch1['data'].shape)    # (N, CxHxW)
print(batch1['data'][0].shape) # (CxHxW,)

# dataset details
classes = train_dataset.classes
print("classes:", classes)
print("class to index:", train_dataset.class_to_idx)
# data after reshaped and transposed from (N,CxHxW) to (N,H,W,C)
print("training data shape:", train_dataset.data.shape)
print("test data shape:", test_dataset.data.shape)


# In[7]:


# create training and testing loaders
train_loader = torch.utils.data.DataLoader(train_dataset, 128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, 128, shuffle=True)


# In[8]:


# visualizing a sample from train dataset
image, label = train_dataset[99]
print(label)
print(image.shape)
plt.imshow(image.permute(1,2,0))
plt.show()


# In[9]:


# visualizing a sample from train loader
# torchvision.transforms.ToTensor() converts a PIL Image or numpy.ndarray (H x W x C)
# in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
train_iter = iter(train_loader)
batch_images, batch_labels = next(train_iter)
image, label = batch_images[0], batch_labels[0]
print(label)
print(image.shape)
plt.imshow(image.permute(1,2,0))
plt.show()


# ## Building the model

# CNN's have a typical VGG architecture that involves CONV -> CONV-> Maxpool -> .... -> FC -> ... Output

# In[10]:


# construct the architecture of the CNN model
class CNN(nn.Module):
  # CIFAR-10 is 3x32x32, b is the batch size
  def __init__(self):
    super().__init__()

    # VGG MODEL
    # CONV->CONV->POOL

    # 3x32x32 -> 64x32x32
    self.layer1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())

    # 64x32x32 -> 64x16x16
    self.layer2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride = 2))

    # 64x16x16 -> 128x16x16
    self.layer3 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU())

    # 128x16x16 -> 128x8x8
    self.layer4 =  nn.Sequential(
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride = 2))

    # 128x8x8 -> 256x8x8
    self.layer5 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU())

    # 256x8x8 -> 256x4x4
    self.layer6 =  nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride = 2))

    # bx256x4x4 ->bx(256x4x4)
    self.fc = nn.Sequential(
        nn.Flatten())

    # bx(256*4*4) -> bx256
    self.fc1 = nn.Sequential(
        nn.Linear(256*4*4, 512),
        nn.ReLU())

    # bx256 -> bx10
    self.fc2 = nn.Sequential(
        nn.Linear(512, 10))

  def forward(self, x):
    # input 3x32x32, output 32x32x32
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = self.layer6(out)
    out = self.fc(out)
    out = self.fc1(out)
    out = self.fc2(out)
    return out


# In[11]:


# Sanity check
cnn = CNN()
print(cnn(torch.randn(128, 3, 32, 32)).shape)
del cnn


# In[12]:


# create an instance of CNN class
model = CNN()
model.to(device) # specify that this model will be stored on the device you chose earlier (GPU or CPU)


# In[13]:


# loss function
criterion = nn.CrossEntropyLoss()


# In[14]:


# optimizer: adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


# In[15]:


# reduce learning rate when a metric has stopped improving
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, min_lr=0.0001, patience=5, verbose=1)


# ## Training

# In[16]:


# training function
def train_one_epoch(model, train_loader, optimizer, criterion, device):
  # sets the module in training mode
  model.train()

  correct = 0
  for i, batch in tqdm(enumerate(train_loader)):  # looping through
    inputs, labels = batch
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    # bx10, want to do argmax on the "10" dimension
    predictions = outputs.argmax(dim=1)
    correct += (predictions == labels).sum().item()
    # Computes the gradients and stores it in the model parameters' .grad
    # attribute (this is backprop or autodiff)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  accuracy = 100 * correct / len(train_dataset)
  print('End of epoch train loss:', round(loss.item(), 3))
  print(f"End of epoch val accuracy: {accuracy}%")

  return loss, accuracy


# ## Testing

# In[17]:


# validation/testing function
def test(model, test_loader, device):
  # the classes - these are from the cifar-10 dataset
  # classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  # sets the module in evaluation mode
  model.eval()

  correct = 0
  for i, batch in tqdm(enumerate(test_loader)):
    # print("i:", i)
    inputs, labels = batch
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    # bx10, want to do argmax on the "10" dimension
    predictions = outputs.argmax(dim=1)
    correct += (predictions == labels).sum().item()

  accuracy = 100 * correct / len(test_dataset)
  print('End of epoch val loss:', round(loss.item(), 3))
  print(f"End of epoch val accuracy: {accuracy}%")

  # visualizing the current model's performance
  for i in range(5):
    print("Guess:", predictions[i], "Label:", batch_labels[i])
    print('Guess:', classes[predictions[i]], '| Label:', classes[batch_labels[i]])
    plt.imshow(inputs[i].cpu().permute(1,2,0))
    plt.show()

  return loss, accuracy


# # Running the train-test loop

# In[ ]:


# run a loop which calls the training and testing functions
NUM_EPOCHS = 20

# parameters for early stopping
start_from_epoch = 10
patience = 5
counter = 0

# best val loss
best_loss = float('inf')
best_epoch = -1

# save the loss and accuracy for plotting later
df_loss = pd.DataFrame(columns=['train', 'validation'])
df_acc = pd.DataFrame(columns=['train', 'validation'])

t0 = time.time()
for epoch in range(NUM_EPOCHS):
  print("Epoch: ", epoch + 1)

  # train the model
  loss1, accuracy1 = train_one_epoch(model, train_loader, optimizer, criterion, device)

  # test the model
  loss2, accuracy2 = test(model, test_loader, device)

  # save the loss and accuracy of the epoch
  df_loss.loc[len(df_loss.index)] = [loss1.item(), loss2.item()]
  df_acc.loc[len(df_acc.index)] = [accuracy1, accuracy2]

  # reduce learning rate if val_loss has stopped improving
  scheduler.step(loss2)

  # save the best (val_loss) model
  val_loss = round(loss2.item(), 3)
  if epoch == 0 or val_loss < best_loss:
    # val_loss improved from the best val loss
    print("Epoch:", str(epoch + 1), "val_loss improved from", str(best_loss),
          "to", str(val_loss), ", saving model to" +
          "./best_model_local.pth")
    torch.save(model.state_dict(), "./best_model_local.pth")
    best_loss = val_loss
    best_epoch = epoch;
    counter = 0
  else:
    # val_loss did not improve from the best val_loss
    print("Epoch:", str(epoch + 1), "val_loss did not improve from", str(best_loss))
    counter += 1
    # early stopping the tran-test loop
    if epoch >= start_from_epoch and counter > patience:
      print("Epoch:", str(epoch + 1), "early stopping", "best epoch:", best_epoch)
      break

print("Training time:", time.time()-t0)


# In[ ]:


# plot train and validation loss and accuracy graph
df_loss.plot(title='Model loss',figsize=(6,4)).set(xlabel='Epoch',ylabel='Loss')
df_acc.plot(title='Model Accuracy',figsize=(6,4)).set(xlabel='Epoch',ylabel='Accuracy')


# In[ ]:


size = 0
for param in model.parameters():
  size += np.prod(param.shape)
print(f"Number of parameters: {size}")


# ## Saving the weights

# In[ ]:


# save the weights of the model
torch.save(model.state_dict(), "./best_model_local.pth")


# ## Loading the weights

# In[ ]:


# reload the weights previously saved
model_new = CNN()
model_new.load_state_dict(torch.load("./best_model_local.pth"))
model.to(device)
model.eval()


# In[ ]:


# test the model
test(model, test_loader, device)

