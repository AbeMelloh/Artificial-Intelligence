

# THIS IS THE CODE FOR A NEURAL NETWORK

import numpy as np
import pandas as pd
import random
import datetime # import datetime
import math #importing this so you can use use the sigmoid function in line 14
random.seed(2016)
sample_size=50
sample=pd.Series(random.sample(range(-10000,10000),sample_size))
x=sample/10000
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
  #Change this to the sigmoid, Tanh, or Relu Function.  #You Chose Sigmoid.  #You got this code off of stack overflow
#print x.head(10)
#print y.head(10)
#print x.describe()
count=0
dataSet=[([x.ix[count]],[y.ix[count]])]
count=1
while (count < sample_size):
    #print "Working on data item:  ",count
    dataSet=(dataSet+[([x.ix[count]],[y.ix[count]])])
    count=count+1
    
import neuralpy
fit=neuralpy.Network(1,3,7,1)#  Each number is how many nodes are in the layer.  So for example the first layer has one node, the second layer has 3 nodes, the third layer has 7 nodes, etc.

#Below you can most likely implement a loop.  

epochsA=1000#Epoch is the amount of time given for the neural network to run
learning_rateA1=1  # Learning rate is the amount of space given to the algorithm to learn.  Too strict (a very small number) and it takes forever to run. Too loose and the prediction isn't very good. 
learning_rateA2=2
learning_rateA3=3

epochsB=2000
learning_rateB1=2
learning_rateB2=4
learning_rateB3=6

epochsC=3000
learning_rateC1=9
learning_rateC2=18
learning_rateC3=27

#Be sure to take the MSE of each of the nine combinations of epochs and learning rate.  
#Record the training time for each combination. (#Use the training code from the assignment page) (already pasted in this code)


print "fitting model right now"
start_time = datetime.datetime.now()
fit.train(dataSet,epochs,learning_rate)   #This line is where we are training the weights
print "Done fitting model"
stop_time = datetime.datetime.now()
count=0
pred=[]
while (count< sample_size):
    out=fit.forward(x[count])
    print ("Obs: ", count+1, " x= ",round(x[count],4), " y= ",round(y[count],4), "prediction = ", round(pd.Series(out),4))
    pred.append(out)
    count=count+1
print "Time required for optimization:",stop_time - start_time
