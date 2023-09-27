#!/usr/bin/env python
# coding: utf-8

# # **Training a CNN from scratch to classify the CIFAR-10 dataset**
# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The accuracy of the small model from scratch is around 88.67% by 30 training epochs. This model has a VGG architecture and uses techniques such as L2 regularization, reducing learning rate, early stopping, and data augumentation to increase the accuracy. Run time on a GPU is around 1268 seconds. Data loading time is around 5 seconds.

# In[1]:


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
import s3fs
torch.manual_seed(17)


# In[2]:


if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('Running on GPU')
    print(torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print('Running on CPU')


# ## Data processing

# In[3]:


s3 = s3fs.S3FileSystem()


# In[19]:


from PIL import Image
from torchvision.datasets import VisionDataset
import os
import pickle

# CIFAR10 Class modified to work with S3 Bucket
class My_CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with s3.open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        with s3.open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


# In[20]:


# for data augmentation
transform = torchvision.transforms.Compose([
    torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.CIFAR10),
    torchvision.transforms.ToTensor()])


# In[45]:


# load the CIFAR 10 training and testing data sets from a directory
root_dir = 's3://cburgess-bucket/CIFAR-10-Model'
#train_dataset = torchvision.datasets.CIFAR10(root=root_dir, transform=transform, download=False)
#test_dataset = torchvision.datasets.CIFAR10(root=root_dir, train=False, transform=torchvision.transforms.ToTensor(), download=False)
t1 = time.time()
train_dataset = My_CIFAR10(root=root_dir, transform=transform)
test_dataset = My_CIFAR10(root=root_dir, train=False, transform=torchvision.transforms.ToTensor())
print("Loading time:", time.time()-t1)


# In[35]:


# unpickle data function
def unpickle(file):
  with s3.open(file, "rb") as infile:
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


# In[22]:


# create training and testing loaders
train_loader = torch.utils.data.DataLoader(train_dataset, 128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, 128, shuffle=True)


# In[23]:


# visualizing a sample from train dataset
image, label = train_dataset[99]
print(label)
print(image.shape)
plt.imshow(image.permute(1,2,0))
plt.show()


# In[24]:


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

# In[25]:


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


# In[26]:


# Sanity check
cnn = CNN()
print(cnn(torch.randn(128, 3, 32, 32)).shape)
del cnn


# In[28]:


# create an instance of CNN class
model = CNN()
model.to(device) # specify that this model will be stored on the device you chose earlier (GPU or CPU)


# In[29]:


# loss function
criterion = nn.CrossEntropyLoss()


# In[30]:


# optimizer: adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


# In[31]:


# reduce learning rate when a metric has stopped improving
scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, min_lr=0.0001, patience=3, verbose=1)


# ## Training

# In[32]:


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
  print(f"End of epoch train accuracy: {accuracy}%")

  return loss.item(), accuracy


# ## Testing

# In[33]:


# validation/testing function
def test(model, test_loader, device):
  # the classes - these are from the cifar-10 dataset
  # classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  # sets the module in evaluation mode
  model.eval()

  # average the loss of batches
  running_loss = 0.0
  correct = 0
  for i, batch in tqdm(enumerate(test_loader)):
    # print("i:", i)
    inputs, labels = batch
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    running_loss += loss.item() * inputs.size(0)
    # bx10, want to do argmax on the "10" dimension
    predictions = outputs.argmax(dim=1)
    correct += (predictions == labels).sum().item()

  epoch_loss = running_loss / len(test_dataset)
  epoch_accu = 100 * correct / len(test_dataset)
  print('End of epoch val loss:', round(epoch_loss, 3))
  print(f"End of epoch val accuracy: {epoch_accu}%")

  # visualizing the current model's performance
  for i in range(2):
    print("Guess:", predictions[i], "Label:", labels[i])
    print('Guess:', classes[predictions[i]], '| Label:', classes[labels[i]])
    plt.imshow(inputs[i].cpu().permute(1,2,0))
    plt.show()

  return epoch_loss, epoch_accu


# # Running the train-test loop

# Save the model weights of the best epoch (highest val_accuracy)

# In[38]:


# run a loop which calls the training and testing functions
NUM_EPOCHS = 30

# parameters for early stopping
start_from_epoch = 10
patience = 5
counter = 0

# best val loss
best_accuracy = float('inf')
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
  df_loss.loc[len(df_loss.index)] = [loss1, loss2]
  df_acc.loc[len(df_acc.index)] = [accuracy1, accuracy2]

  # reduce learning rate if val_accuracy has stopped improving
  scheduler.step(accuracy2)

  # save the weights of the best model (highest val_accuracy)
  val_accuracy = accuracy2
  if epoch == 0 or val_accuracy > best_accuracy:
    # val_accuracy improved from the best val loss
    print("Epoch:", str(epoch + 1), "val_accuracy improved from", str(best_accuracy),
          "to", str(val_accuracy) + ", saving model to", "./best_model_s3.pth")
    torch.save(model.state_dict(), "./best_model_s3.pth")
    best_accuracy = val_accuracy
    best_epoch = epoch;
    counter = 0
  else:
    # val_accuracy did not improve from the best val_accuracy
    print("Epoch:", str(epoch + 1), "val_accuracy did not improve from", str(best_accuracy))
    counter += 1
    # early stopping the tran-test loop
    if epoch >= start_from_epoch and counter > patience and (epoch + 1) != NUM_EPOCHS:
      print("Epoch:", str(epoch + 1), "early stopping")
      break

print("Best epoch:", str(best_epoch + 1) + ", val_accuracy:", str(best_accuracy))
print("Training time:", time.time()-t0)


# Plot train and validation loss and accuracy graphs

# In[40]:


df_loss.plot(title='Model loss',figsize=(6,4)).set(xlabel='Epoch',ylabel='Loss')
df_acc.plot(title='Model Accuracy',figsize=(6,4)).set(xlabel='Epoch',ylabel='Accuracy')


# In[41]:


size = 0
for param in model.parameters():
  size += np.prod(param.shape)
print(f"Number of parameters: {size}")


# ## Loading the weights

# In[42]:


# reload the weights previously saved
model_new = CNN()
model_new.load_state_dict(torch.load("./best_model_s3.pth"))
model_new.to(device)
model_new.eval()


# In[43]:


# test the model
test(model_new, test_loader, device)


# In[ ]:




