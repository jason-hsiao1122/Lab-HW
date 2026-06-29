# Image Classification with PyTorch
This is **the second** homework of COAV group member.

## Instructions
Implement a Convolutional Neural Network (CNN) model by yourself to classify the CIFAR-10 dataset. You **CAN** use deep learning packages such as **Pytorch or Tensorflow (unfortunately, Keras is still prohibited in this and the rest of homeworks)** for this homework (unless you ***really*** want a challenge and do it from scratch). We strongly recommend you choose **Pytorch** due to some unreasonable reasons.

Please design the model by yourself. You are free to choose arbitrary numbers of convolutional and fully connected layers in your model.

1. Plot the learning curves and the accuracy of classification versus the number of epochs until convergence for training data as well as test data.  
2. Repeat 1 with different batch sizes.
3. Repeat 1 with different learning rates  
4. Randomly choose 10 images from your testing data and use your model to predict their class. You can show us the result on the subtitle of each subplots.  

## Load and normalize CIFAR-10
Download and load CIFAR-10 with `torchvision`, an example code is provided below: 
```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

Learn the basics of `torchvisions.transforms` and modify as you see fit.

## Requirements
* PyTorch
* torchvision
* matplotlib  
* Pillow (recommended) or OpenCV

## Hand-in procedure
1. Add your info to the file `README.md` in the `main` branch.
2. Revise your info to the file `README.md` when you accomplish this homework.

## Resources
Here are some resources to help you get started.

[PyTorch tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)  

