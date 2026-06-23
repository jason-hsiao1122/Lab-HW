# Feedforward Classification Network from Scratch
This is **the first** homework of COAV group member.

## Instructions
Implement a feedforward neural network (FNN) model by yourself to classify the MNIST dataset. Deep learning packages such as Pytorch and Tensorflow **are strickly PROHIBITED** in this homework.

Design a FNN model architecture with **2 hidden layers** and perform the **random initialization for model weights**. **The first hidden layer contains 100 neurons** and **the second one contains 150 neurons**. Run backpropagation algorithm and use mini-batch SGD (stochastic gradient descent) to optimize the parameters. Use **cross-entropy loss** as your loss function.
1. Plot the learning curves and the accuracy of classification versus the number of iterations until convergence for training data as well as test data.  
2. Repeat 1 with different batch sizes.  
3. Repeat 1 by performing zero initialization for the model weights.
4. Randomly plot 10 images and use your model to predict which digit it is (you can show your result on the title of each subplot).  

## Dataset
[MNIST dataset source](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz)

What does `mnist.npz` have?  
|Name|Dimension|Description|
|--|--|--|
|x_train|(60000,28,28,1)|60000 training images with shape (28,28,1)|
|y_train|(60000,)|corresponding labels of training images|
|x_test|(10000,28,28,1)|10000 testing images with shape (28,28,1)|
|y_test|(10000,)|corresponding labels of testing images|

Sample code for data loading:  

```python
import numpy as np
import os

path = os.getcwd()

data = np.load(os.path.join(path,"mnist.npz"))

x_train, y_train = data["x_train"], data["y_train"]
x_test, y_test = data["x_test"], data["y_test"]

data.close()
```

#### Requirements
* numpy  
* matplotlib  
* Pillow or OpenCV

## Hand-in procedure
1. Add your info to the file `README.md` in the `main` branch.
2. Revise your info to the file `README.md` when you accomplish this homework.

## Resources
Here are some videos help you understand the fundamentals of neural networks.

[3blue1brown Youtube channel](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw)  
[Neural Network Playlist](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)  

This document provides you some equations of neural networks.  

[NCTUEEML2016_HW3](./ML_hw3_2016.pdf)  


