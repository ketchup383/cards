import torch
import torch.nn as nn # Specific functions related to neural networks
import torch.optim as optim #optimzer
from torch.utils.data import Dataset, DataLoader
# Makes working with image files easier:
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm # library for loading in architecture specific for image classification

# Mainly for data:
import matplotlib.pyplot as plt # for data visualization
import pandas as pd
import numpy as np

# PyTorch DATASETS and DATALOADERS
class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data = ImageFolder(data_dir, transform = transform) 
        # ImageFolder class assumes all sub folders of folders have class name for image --> handles label creation for us

    def __len__(self): # need len, because data loader needs to know sample size
        return len(self.data)

    def __getitem__(self, idx): # takes an index location and returns an item
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes
    
data_dir = "cards-dataset"
dataset = PlayingCardDataset(data_dir)
target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()} # Creates dictionary that associates each index to correct value

# from torch vision; taking image and making sure that they are always 128 x 128 and converting into pyTorch tensor
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Wrap with a pytorch Dataloader (handles processing and reading each image)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True) # typically kept at shuffled; trains faster when taking in batches at a time

# PYTORCH MODEL
# use timm package (great for image classification)
class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifier, self).__init__()

        self.base_model = timm.create_model('efficientnet_b0', pretrained=True) # use training model effcientnet; pretrained true means weights have already been trained
        self.features = nn.Sequential(*list(self.base_model.children())[:-1]) # for this context, trim off last layer of model
        enet_out_size = 1280 # default feature size for efficientnet is 1280
        # Make classifier
        self.classifier = nn.Linear(enet_out_size, num_classes) # change to make it 53
        pass

    def forward(self, x): #Connect batches and return outputs
        x = self.features(x)
        output = self.classifier(x)
        return output
    
