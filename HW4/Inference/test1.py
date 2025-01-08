from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import mnist
import torch
from Lenet5 import Lenet5, Tanh
import numpy as np

import torchvision

 

 

def test(dataloader,model):
    #please implement your test code#                                                                                                                                                                      
    ###################### #####   
    test_accuracy = 0.0
    for data, target in dataloader:
        y_pred = model(data)
        test_accuracy += (torch.argmin(y_pred,dim=1)==target).sum().item()
    test_accuracy /= len(dataloader.dataset)
    print("test accuracy:", test_accuracy)

 

def main():

    pad=torchvision.transforms.Pad(2,fill=0,padding_mode='constant')
    mnist_test=mnist.MNIST(split="test",transform=pad)
    test_dataloader= DataLoader(mnist_test,batch_size=1,shuffle=False)

    model = torch.load("problem1.pth")

    test(test_dataloader,model)

 

if __name__=="__main__":
    main()

