from __future__ import print_function, division
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
from train import data_transforms

class Resnet():
    def __init__(self, cuda_num=0):
        self.device = torch.device("cuda:{}".format(cuda_num) if torch.cuda.is_available() else "cpu")		
		# state dick VERSION
        self.model_ft = torchvision.models.resnet18(pretrained=False)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 2)
        self.path_to_state_dict = './model/state_dict_v1.pt'
        self.model_ft.load_state_dict(torch.load(self.path_to_state_dict, map_location=self.device))
        self.model_ft.to(self.device)
        self.model_ft.eval()
        self.model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def get_weights(self, image, test_transforms=data_transforms["val"]):
        image = Image.fromarray(image)
        image_tensor = test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = image_tensor.to(self.device)
        output = self.model_ft(input)
        weights=(torch.nn.functional.softmax(output, dim=1).data.cpu().numpy())
        #weights = output.data.cpu().numpy()
        print (weights[0])
        return weights[0]


