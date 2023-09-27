# cnn-cifar10

Training a CNN from scratch to classify the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The accuracy of the small model from scratch is around 88.67% by 30 training epochs. This model has a VGG architecture and uses techniques such as L2 regularization, reducing learning rate, early stopping, and data augumentation to increase the accuracy. Run time on a GPU is around 1268 seconds. Data loading time is around 5 seconds.

## Requirements
Python, Pytorch, Jupyter Notebook.

## Technical Skills
Pytorch APIs, CNN architecture, regularization techniques such as L2 regularization, reducing learning rate, early stopping, and data augumentation.

## Results
![image](https://github.com/carab9/cnn-cifar10/blob/main/cifar10_loss.png?raw=true)

![image](https://github.com/carab9/cnn-cifar10/blob/main/cifar10_accuracy.png?raw=true)
