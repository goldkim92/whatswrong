import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1'

import torchvision as tv
from os.path import join
from torchvision.models import googlenet

import torch
import torch.nn as nn
import torch.nn.functional as F

from run import Run_Imagenet

class GoogleNet_Drop(nn.Module):
    def __init__(self):
        super(GoogleNet_Drop, self).__init__()

        # get the pretrained ResNet50SE network
        model = googlenet(pretrained=True)

        # dissect the network
        self.features_conv = nn.Sequential(*list(model.children())[:-6])

        # additional layers
        self.added_conv = nn.Sequential(
            nn.Conv2d(832,1024, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#         self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(1024*7*7, 4096, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000, bias=True),
        )

        # delete the origin model
        del model


    def forward(self, x):
        self.features = self.features_conv(x)
        self.added_features = self.added_conv(self.features)
        x = self.max_pool(self.added_features)
        x = x.view((x.size(0), -1))
        x = self.classifier(x)
        return x

    def freeze_params(self, freeze=True):
        if freeze:
            for param in self.features_conv.parameters():
                param.requires_grad = False

    def get_activations(self):
        return self.features

    def get_activations2(self):
        return self.added_features


class Arguments():
    def __init__(self):
        self.phase = 'train'
        self.freeze = True
        self.model = GoogleNet_Drop()
        self.lr = 1e-4
        self.bs = 64
        self.epochs = 10
        self.log_step = 5000
        self.img_step = 2500
        self.img_dir = 'img'
        self.save_path = join('model','new_googlenet.pth')
        self.load_path = None
        
        self.make_dirs()

    def make_dirs(self):
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        if not os.path.exists('model'):
            os.makedirs('model')


if __name__ == "__main__":
    args = Arguments()
    run = Run_Imagenet(args)
    run.train()

    
