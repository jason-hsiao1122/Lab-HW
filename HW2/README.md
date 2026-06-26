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

#### Who has done this homework
|Name|Started Date|Finished Date|Note|
|:--:|:--:|:--:|:--:|
|GaGaSu|2016.01.01|2016.01.01|Too easy|
|黃子原|2022.01.11|2022.01.21|Newbie in Pytorch|
|蔡沛珈|2022.01.??|2022.02.11|potato|
|林峻潁|2022.01.11|2022.02.23|......|
|吳恆宇|2023.08.02|----------|A for Average|
|郭羿萱|2023.08.14|2023.08.--|哥哥嚇弟弟&emsp;<地下道>|
|林子睿|2023.10.16|----------|南京人為甚麼身體好? 他們是建康人|
|鍾旻諠|2023.11.07|2023.11.30|(=ↀωↀ=)✧|
|林彥君|2024.01.15|2024.01.19|寒假!|
|楊適豪|2024.02.18|2024.03.04|接不到家教;;|
|褚亭妏|2024.04.15|2024.05.01||
|陳柏蓁|2024.04.29|2024.05.27|電腦怪怪TT|
|林琪雅|2024.05.08|---||
|陳亭羽|2024.07.30|2024.08.15|交映池|
|王翔郁|2024.08.01|---||
|洪宜均|2024.07.21|2024.08.01|大家好捲窩被捲鼠|
|賴竑均|2025.02.20|---||
|賴偲旻|2025.03.10|2025.04.15|ω・´)|
|蕭博勁|2026.06.26|---||
