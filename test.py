from torchvision import models
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import transforms, datasets
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn as nn
from torch.utils.model_zoo import load_url
import torch.nn.functional as F
import os
from efficientnet_pytorch import EfficientNet

torch.manual_seed(0)
torch.manual_seed(torch.initial_seed())
# In[43]:

test_path = '/home/umair/Desktop/umair/gender/our_data_cropped'

data_transforms = transforms.Compose([      
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_data = torchvision.datasets.ImageFolder(
    root = test_path,
    transform = data_transforms
)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=2,
    num_workers=0,
    shuffle=True
)

def test_model(model, dataloader):
    device = "cuda:0"
    model.eval()
    model.to(device)
    val_acc_history = []
    running_corrects = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs)
        _, preds = torch.max(output, 1)
        running_corrects += torch.sum(preds == labels.data)
    acc = running_corrects.double() / len(dataloader.dataset)
    print("Model accuracy: {:.4f}".format(acc))
    
model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=2)
print(model)
model.load_state_dict(torch.load("/home/umair/Desktop/umair/genderClassification/merged_CEFL.pt"))
test_model(model, test_loader)
