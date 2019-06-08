import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .vgg import VGG, VGG11
from .text_net import BrainLSTM


class EarlyFusion(nn.Module):
    def __init__(self, combine_dim):
        super().__init__()
        self.vgg = VGG(combine_dim)
        self.lstm = BrainLSTM(768, 256, 1, 2, combine_dim)

        self.fusion = nn.Sequential(
            nn.Linear(combine_dim*2, combine_dim, bias=True),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(combine_dim, 2, bias=True),
        )
    
    def forward(self, embeddings, images):
        # images = images.view(-1, 1, 
        #                      images.size(-2), images.size(-1))
        # print(images.shape)
        embs_repr = self.lstm(embeddings)
        imgs_repr = self.vgg(images)

        x = torch.cat((embs_repr, imgs_repr), dim=1)

        x = self.fusion(x)
        return x