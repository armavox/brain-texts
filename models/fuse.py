import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .vgg import VGG, VGG11
from .text_net import BrainLSTM


class EarlyFusion(nn.Module):
    def __init__(self, combine_dim):
        super().__init__()
        self.vgg = VGG11(combine_dim=combine_dim)
        self.lstm = BrainLSTM(embed_dim=768, hidden_dim=1024, num_layers=2,
                              context_size=2, combine_dim=combine_dim,
                              dropout=0)

        self.fusion = nn.Sequential(
            nn.Linear(combine_dim * 2, combine_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            # nn.Linear(combine_dim, combine_dim),
            # nn.LeakyReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Linear(combine_dim, combine_dim // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(combine_dim // 2, 2)
        )
    
    def forward(self, embeddings, images):
        embs_repr = self.lstm(embeddings)
        imgs_repr = self.vgg(images)

        x = torch.cat((embs_repr, imgs_repr), dim=1)
        x = self.fusion(x)
        return x