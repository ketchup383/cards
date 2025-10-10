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
from tqdm import tqdm

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

model = SimpleCardClassifier(num_classes=53)

# Loss function
criterion = nn.CrossEntropyLoss()
# Optimize
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_folder = r'C:\Users\vince\Desktop\VSCode\cards-pytorch-model\cards-dataset\train'
valid_folder = r'C:\Users\vince\Desktop\VSCode\cards-pytorch-model\cards-dataset\valid'
test_folder = r'C:\Users\vince\Desktop\VSCode\cards-pytorch-model\cards-dataset\test'

train_dataset = PlayingCardDataset(train_folder, transform=transform)
valid_dataset = PlayingCardDataset(valid_folder, transform=transform)
test_dataset = PlayingCardDataset(test_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # keep the training shuffled
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_dataset_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_epoch = 5 # epoch is one run through entire training set
best_val_acc = 0.0 # track best validation accuracy
train_losses, val_losses = [], []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epoch):
    # Set the model to train
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc='Training loop'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward() # back propagation which will update model weight in every step of the way
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation phase
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc='Validation loop'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
    val_loss = running_loss / len(valid_loader.dataset)
    val_losses.append(val_losses)

    print(f"Epoch {epoch+1}/{num_epoch} - Train loss: {train_loss}, Validation loss: {val_loss}")