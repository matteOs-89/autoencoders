
# Imports

import torch
import torchvision
import torch.nn as nn
import  torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import os
import time

import matplotlib.pyplot as plt

batch_size = 128


train_data = torchvision.datasets.MNIST(root="../../data",
                                          train=True,
                                          transform=transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.MNIST(root="../../data",
                                          train=False,
                                          transform=transforms.ToTensor())


train_loader = DataLoader(train_data,
                          shuffle=True,
                          batch_size=batch_size,)

test_loader = DataLoader(test_data,
                          shuffle=False,
                          batch_size=batch_size, )



def CNN_AENet(print_shape=False):

  class Cnn_Autoencodernet(nn.Module):

    def __init__(self,printtoggle):
      super().__init__()
      
      self.print = print_shape

      # Encoder layers
      
      self.encoderconv1  = nn.Conv2d(1,32,3,padding=1,stride=2)
  
      self.encoderconv2  = nn.Conv2d(32,64,3,padding=1,stride=2)

    
      # Decoder layers 
      self.decoderconv1  = nn.ConvTranspose2d(64,32,4,padding=1,stride=2)

      self.decoderconv2  = nn.ConvTranspose2d(32,1,4,padding=1,stride=2)
      
      

    def forward(self,x):
      
   
      x = F.leaky_relu(self.encoderconv1(x))
      if self.print: print(f'First encoder layer: {list(x.shape)}')
 
      x = F.leaky_relu(self.encoderconv2(x))
      if self.print: print(f'Second encoder layer: {list(x.shape)}')

      x = F.leaky_relu(self.decoderconv1(x))
      if self.print: print(f'First decoder layer: {list(x.shape)}')

      x = F.leaky_relu(self.decoderconv2(x))
      if self.print: print(f'Second decoder layer: {list(x.shape)}')

      return x


  model = Cnn_Autoencodernet(print_shape)
  
  loss_function = nn.MSELoss()

  optimizer = torch.optim.Adam(model.parameters(),lr=.001)

  return model,loss_function,optimizer

model, loss_function, optimizer = CNN_AENet()

x,y = next(iter(train_loader))

yhat = model(x)
print(yhat.shape)

if not os.path.exists("resultsCNN_AUE"):
  os.makedirs("resultsCNN_AUE")

num_epochs = 10

def train_and_eval(epoch):

  model, loss_function, optimizer = CNN_AENet()

  start_time = time.time()
  train_loss = []
  output = []
  test_loss = []

  model.train()

  for i, (images,_) in enumerate(train_loader):

    images = images

    reconstucted_image = model(images)

    loss = loss_function(reconstucted_image, images)

    train_loss.append(loss.item())
    average_loss = sum(train_loss)/len(train_loader)

    # Back propagation

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
    end_time = time.time() - start_time

  model.eval()
  
  images,_ = next(iter(test_loader))

  with torch.no_grad():

    Generated_IMG = model(images)

    loss = loss_function(Generated_IMG, images)

    test_loss.append(loss.item())

    average_loss = sum(test_loss)/len(test_loader)

  output.append((epoch, images, reconstucted_image))

  if epoch % 5 == 0:

    comparison_img = torch.cat([images[:10], reconstucted_image[:10], Generated_IMG[:10]])
    save_image(comparison_img, "resultsCNN_AUE/reconstruct_" + str(epoch) +".png", nrow = 5)


    print(f"Train Epoch: {epoch} Train Loss: {train_loss[i]:.3f}, Average Train Loss: {average_loss}, Time: {end_time}")
    

  return model, train_loss, test_loss, output

for epoch in range(1, num_epochs +1):
  
  model, train_loss, test_loss, output = train_and_eval(epoch)

fig = plt.figure(figsize=(8,4))

plt.plot(train_loss,'s-',label='Train')
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.title('Model loss')
plt.legend()

plt.show()
