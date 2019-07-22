import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.resnet as ResNet
from torch.utils import model_zoo as zoo

class Encoder(nn.Module):

    def __init__(self, **kwargs):
        super(Pretrain, self).__init__()

        # dimension of word embedding vectors
        embed_size = kwargs['embed_size'] if 'embed_size' in kwargs else 256


        # load pretrained resent 152 model
        # delete the last fc layer
        # resnet = models.resnet152(pretrained=True) # there seems to be an https error in downloading
        resnet = models.resnet152(pretrained = True) # there seems to be an https error in downloading
        modules = list(resnet.children())[:-1] # remove last layer, make custom layer

        # in features
        self.in_features = resnet.fc.in_features

        # create torch model sequence
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(self.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def update_layer(self, embed_size):
        self.linear = nn.Linear(self.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features
