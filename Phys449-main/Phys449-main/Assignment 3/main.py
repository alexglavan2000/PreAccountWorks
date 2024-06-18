# Write your assignment here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping

print(tf.__version__)

#loading sets a and b
set1 = np.array(np.loadtxt('a.txt'))
set2 = np.array(np.loadtxt('b.txt'))

#fusing sets a and b, called set 3
setnew= []
j=0
x=8
while j<x:
    setnew.append(set1[0][j]*set2[0][j])
    j+=1
j=0

while j<x:
    setnew.append(set1[1][j]*set2[1][j])
    j+=1
j=0
while j<x:
    setnew.append(set1[2][j]*set2[2][j])
    j+=1
j=0

while j<x:
    setnew.append(set1[3][j]*set2[3][j])
    j+=1
j=0
while j<x:
    setnew.append(set1[4][j]*set2[4][j])
    j+=1
j=0
while j<x:
    setnew.append(set1[5][j]*set2[5][j])
    j+=1
j=0
while j<x:
    setnew.append(set1[6][j]*set2[6][j])
    j+=1
j=0
while j<x:
    setnew.append(set1[7][j]*set2[7][j])
    j+=1
j=0
while j<x:
    setnew.append(set1[8][j]*set2[8][j])
    j+=1
j=0
while j<x:
    setnew.append(set1[9][j]*set2[9][j])
    j+=1
    
#loading set c
set3 = np.array(np.loadtxt('c.txt'))
new=set3.flatten()

#inputing data of set c into new array so it cqn be later modified
L=len(new)
i=0
y=[]
while i<L:
    y.append(float(new[i]))
    i+=1

#changing the 1 and 0 into true and false for sets 3 and c
L=len(setnew)
i=0
while i<L:
    if setnew[i]==0.0:
        setnew[i]=False
    else:
        setnew[i]=True
    i+=1


L=len(y)
i=0
while i<L:
    if y[i]==0.0:
        y[i]=False
    else:
        y[i]=True
    i+=1

x_train = np.array([[1, 3], [5, 7], [9, 11], [13, 15], [np.nan, np.nan]])
y_train = np.array([[np.nan, np.nan], [81, 82], [83, 84], [85, 86], [87, 88]])
n_input = 1
generator = TimeseriesGenerator(x_train, y_train, length=n_input, batch_size=1)

x_test = setnew
y_test = y
n_input_val = 1
validation_generator = TimeseriesGenerator(x_test, y_test, length=n_input_val, batch_size=1)

print("Training generator:")
for i in range(len(generator)):
    a, b = generator[i]
    print('%s => %s' % (a, b))

print("Validation generator:")
for i in range(len(validation_generator)):
    a, b = validation_generator[i]
    print('%s => %s' % (a, b))

# define model
model = Sequential()
model.add(SimpleRNN(50,input_shape=(1, 2)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss',patience=2)

model.fit(generator,epochs=1000,validation_data=validation_generator,callbacks=[early_stop])

losses = pd.DataFrame(model.history.history)

losses.plot()
plt.show()


