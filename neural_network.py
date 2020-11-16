import numpy as np
import torch
import matplotlib.pyplot as plt

m = 8  # number of examples
l1, l2, l3 = 8, 3, 8   # number of neuros in layer0, layer1 and layer2
alpha = 4  # learning rate

# derivate of sigmoid function
def sigmoid_derivate(x): 
    return torch.mul(torch.sigmoid(x), 1-torch.sigmoid(x))

# derivate of tanh function
def tanh_derivate(x):
    return 1-torch.tanh(x)**2

data= torch.eye(l1)
y = torch.eye(l3)

# initial weights and biases
w1 = torch.randn(l2,l1)* 0.01
w2 = torch.randn(l3, l2)* 0.01
b1 = torch.randn(1)*0.01
b2 = torch.randn(1)*0.01

losses = []
epochs = 1000
for epoch in range(epochs):
    
    # forward propagation
    z1 = torch.mm(w1, data).add(b1)
    a1 = torch.tanh(z1) # activate in layer1 (hidden layer)

    z2 = torch.mm(w2, a1).add(b2)
    a2 = torch.sigmoid(z2) # activate in layer2(output layer)

    # loss function
    loss = torch.sum((a2-data)**2)
    losses.append(loss.item())

    # back propagation
    # layer2 (output layer)
    da2 = a2.sub(y).mul(2)   # derivative of a2
    dz2 = torch.mul(da2, sigmoid_derivate(z2)) # derivative of z2
    dw2 = torch.mm(dz2, a1.T).div(m)
    db2 = torch.sum(dz2).div(m)  
   
    # layer1 (hidden layer)
    da1 = torch.mm(w2.T, dz2)
    dz1 = torch.mul(da1, tanh_derivate(z1))
    dw1 = torch.mm(dz1, data.T).div(m)
    db1 = torch.sum(dz1).div(m)  

    # update weights and biases
    w1 -= alpha*dw1
    b1 -= alpha*db1 
    w2 -= alpha*dw2
    b2 -= alpha*db2

    print("epoch: "+ str(epoch) + ", loss: "+ str(loss.item()))

# test
z1 = torch.mm(w1, data).add(b1)
a1 = torch.tanh(z1)
z2 = torch.mm(w2, a1).add(b2)
a2 = torch.sigmoid(z2) 

# print the output
np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)
print("\n The output data is: \n", a2.numpy())
# print(a2.argmax(dim=0)) # print the index of max value in each line

plt.xlabel("epoch") 
plt.ylabel("loss") 
plt.plot(np.arange(epochs), losses)
plt.show()


