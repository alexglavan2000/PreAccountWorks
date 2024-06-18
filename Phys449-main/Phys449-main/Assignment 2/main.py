# import libraries
import pandas as pd
import torch
import tensorflow as tf
import torch.nn as nn
import matplotlib.pyplot as plt

def network (data_x1):
    
    #convert the data into an array to remove the odd numbers
    x=array(split)
    L=len(split)
    i=0
    while i<L:
        if x[i]%2==0:
            x.remove(x[i])
            i=+1
        else:
            i+=1
    X=ToTensor(x)
    
    #initialize the variables for the optimizer
    n_input = tf.size(X)
    n_hidden = 0
    n_out = 1
    batch_size = 100
    learning_rate = 0.01  
    
    #generate the model and keep track of every trial
    model = nn.Sequential(nn.Linear(n_input, n_hidden),nn.ReLU(),nn.Linear(n_hidden, n_out),nn.Sigmoid())
    losses = []
    i=0
    for epoch in range(5000):
        pred_y = model(X)
        loss = loss_function(pred_y, len(X))
        losses.append(loss.item())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        print(i)
        i+=1

    #plot the losses to see how good or bad the model is
    plt.plot(losses)
    plt.show()

#download the data from the csv file
df = pd.read_csv("even_mnist.csv")

#confert dataframe to tensor
x = ToTensor(df)
# split up the test data
split=tf.split(data_x,3000,1)

network(split)
network(x)
