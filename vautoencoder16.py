
# Imports

import torch
import torchvision
import torch.nn as nn
import  torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import sys
import os
import time

import matplotlib.pyplot as plt

batch_size = 128
num_epochs = 20


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

class VariationalAutoencoder(nn.Module):

  def __init__(self, image_size=784, hidden_dim=400,latent_dim=20):
    super(VariationalAutoencoder, self).__init__()

    self.image_size = image_size

    self.input = nn.Linear(image_size, hidden_dim)
    
    self.fc1_mean = nn.Linear(hidden_dim, latent_dim)
    self.fc1_var = nn.Linear(hidden_dim, latent_dim) # Variance
    
    self.fc2 = nn.Linear(latent_dim, hidden_dim) 
    self.output = nn.Linear(hidden_dim, image_size)

  def encoder(self, x):

    h = F.relu(self.input(x))
    meanUnit = self.fc1_mean(h)
    VarUnit = self.fc1_var(h)

    return meanUnit, VarUnit

  
  def reparameterize(self, meanUnit, VarUnit):

    """
    Regularizing the latent space, and drawing
    sample from normal distribution, which in turn helps
    during back propagation to perform better.
    """
    std = torch.exp(VarUnit/2)
    eps = torch.randn_like(std)            
    return meanUnit + eps * std

  def decoder(self, z):

    sample = F.relu(self.fc2(z))
    
    output = torch.sigmoid(self.output(sample))
    
    
    return output

 
  def forward(self, x):

    meanUnit, VarUnit = self.encoder(x.view(-1, self.image_size))
    z = self.reparameterize(meanUnit, VarUnit)
    Generated_IMG = self.decoder(z)
    #print(Generated_IMG.shape)

    return Generated_IMG, meanUnit, VarUnit


model = VariationalAutoencoder()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Ensuring model is correctly constructed

x,y = next(iter(train_loader))

Generated_IMG, meanUnit, VarUnit = model(x)

def loss_function(Generated_IMG, original_image, meanUnit=meanUnit, VarUnit=VarUnit):
  bce = F.binary_cross_entropy(Generated_IMG, original_image.view(-1, 784), reduction="sum")
  kld = 0.5 * torch.sum(-1 - VarUnit + meanUnit.pow(2) + VarUnit.exp())
  return bce + kld

if not os.path.exists("results"):
  os.makedirs("results")

def train(epoch):

  start_time = time.time()
  model.train()
  train_loss = []
  for i, (images,_) in enumerate(train_loader):

    Generated_IMG, meanUnit, VarUnit = model(images)

    loss = loss_function(Generated_IMG, images, meanUnit, VarUnit)
    
    train_loss.append(loss.item())

    average_loss = sum(train_loss)/len(train_loader)

    # Back propagation

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    end_time = start_time - time.time()

  if i % 50 == 0:

    msg = f"Train Epoch: {epoch}  Batch: {i}/{len(train_loader)} Train Loss: {train_loss[i]:.3f}, Average Train Loss: {average_loss}, Time: {end_time}"
    sys.stdout.write("\r" + msg)

def test(epoch):
  
  model.eval()
  
  test_loss = []

  with torch.no_grad():

    for idx, (images,_) in enumerate(test_loader):

      Generated_IMG, meanUnit, VarUnit = model(images)

      loss = loss_function(Generated_IMG, images, meanUnit, VarUnit)

      test_loss.append(loss.item())
      average_loss = sum(test_loss)/len(test_loader)

      if idx == 0:

        """
        Save original and generated images 
        """

        comparison_img = torch.cat([images[:10], Generated_IMG.view(batch_size,1,28,28)[:10]])
        save_image(comparison_img, "results/reconstructions_" + str(epoch) +".png", nrow = 5)

  print(f"Average Test Loss: {average_loss:.3f}")

for epoch in range(1, num_epochs +1):

  train(epoch)

  test(epoch)

  with torch.no_grad():
    
    # Generating new images
    sample_img = torch.randn(64, 20)  #[ sample(64), latent space(20)]
    generated = model.decoder(sample_img)
    save_image(generated.view(64,1,28,28), "results/sample_"+str(epoch) +".png")
