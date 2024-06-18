# Write your assignment here

#for the 1st set
import numpy as np
import matplotlib.pyplot as plt

def compare (x,y):
    from numpy import mean
    L=len(x)
    i=0
    ans=[]
    while i<L:
        ans.append(1-abs(x[i]-y[i])/(x[i]+y[i]))
        i+=1
    return mean(ans)

def approx(x,y,j,n):
    i=0
    ans=[]
    while i<n:
        ans.append((x[i]+y[i])**(j))
        i+=1
    return ans

data=np.loadtxt("1.in")


#separating the data into 3 separate arrays for easier access
L=len(data)
X1=[]
X2=[]
Y=[]
i=0
while i<L:
    X1.append(data[i][0])
    X2.append(data[i][1])
    Y.append(data[i][2])
    i+=1

#Assuming that the equation is in form y=(x1+x2)^T for some T
#also just hard coded the learning rate and number iteration
i=0
j=0
comp=[0]
while j<1000:
    comp.append(compare(approx(X1,X2,j*0.0001,L),Y))
    if comp[j-1]<comp[j]:
        j+=1
    else:
        i=j
        j=1000



#plotting the data


fig = plt.figure(figsize=(4,4))

ax = fig.add_subplot(111, projection='3d')

ax.plot(X1,X2,approx(X1,X2,i,L),'red')

ax.scatter(X1,X2,Y)
plt.show()


#for the 2nd set
import numpy as np
import matplotlib.pyplot as plt


data=np.loadtxt("2.in")


#separating the data into 3 separate arrays for easier access
L=len(data)
X1=[]
X2=[]
Y=[]
i=0
while i<L:
    X1.append(data[i][0])
    X2.append(data[i][1])
    Y.append(data[i][2])
    i+=1

#Assuming that the equation is in form y=(x1+x2)^T for some T
#also just hard coded the learning rate and number iteration
i=0
j=0
comp=[0]
while j<1000:
    comp.append(compare(approx(X1,X2,j*0.01,L),Y))
    if comp[j-1]<comp[j]:
        j+=1
    else:
        i=j
        j=1000



#plotting the data


fig = plt.figure(figsize=(4,4))

ax = fig.add_subplot(111, projection='3d')

ax.plot(X1,X2,approx(X1,X2,i,L),'red')

ax.scatter(X1,X2,Y)
plt.show()
