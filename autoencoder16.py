
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
                          batch_size=batch_size, 
                          )

def createAutoEncoder():

  class Autoencodernet(nn.Module):

    def __init__(self):
      super().__init__()

      self.input   = nn.Linear(28*28, 128)

      self.fc1     = nn.Linear(128, 64)

      self.encoder = nn.Linear(64,32)

      self.code    = nn.Linear(32, 64)

      self.fc2     = nn.Linear(64, 128)

      self.decoder = nn.Linear(128, 28*28 )
      

    def forward(self, x):
    
      x = F.relu(self.input(x))

      x = F.relu(self.fc1(x))

      x = F.relu(self.encoder(x))
      
      x = F.relu(self.code(x))

      x = F.relu(self.fc2(x))

      output = torch.sigmoid(self.decoder(x))

      return output

  model = Autoencodernet()

 
  loss_function = nn.MSELoss()
  

  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  return model, loss_function, optimizer

model, loss_function, optimizer = createAutoEncoder()

x,y = next(iter(train_loader))

yhat = model(x.view(-1, 28*28))
print(yhat.shape)

if not os.path.exists("resultsAUE"):
  os.makedirs("resultsAUE")

num_epochs = 25

def train(epoch):

  model, loss_function, optimizer = createAutoEncoder()

  model.train()

  start_time = time.time()
  train_loss = []
  output = []

  for i, (images,_) in enumerate(train_loader):

    images = images.view(-1, 28*28)

    reconstucted_image = model(images)

    loss = loss_function(reconstucted_image, images)

    train_loss.append(loss.item())
    average_loss = sum(train_loss)/len(train_loader)

    # Back propagation

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
    end_time = time.time() - start_time

  output.append((epoch, images, reconstucted_image))

  if epoch % 5 == 0:

    comparison_img = torch.cat([images.view(-1,1,28,28)[:10], reconstucted_image.view(-1,1,28,28)[:10]])
    save_image(comparison_img, "resultsAUE/reconstruct_" + str(epoch) +".png", nrow = 5)


    print(f"Train Epoch: {epoch} Train Loss: {train_loss[i]:.3f}, Average Train Loss: {average_loss}, Time: {end_time:.2f}")
    

  return train_loss, output

for epoch in range(1, num_epochs +1):

  train_loss, output = train(epoch)

plt.plot(train_loss, ".-")
plt.xlabel("Iterations")
plt.ylabel("Model Traing loss")
plt.title("Loss Performance")
plt.show()
