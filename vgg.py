import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class VGGFeatureExtractor(nn.Module):
    def __init__(self, device):
        super(VGGFeatureExtractor, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

        self.required = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '17': 'relu3_4',
            '26': 'relu4_4'
        }

        self.layers = nn.Sequential()
        self.selected = {}

        for name, layer in vgg._modules.items():
            self.layers.add_module(name, layer)
            if name in self.required:
                self.selected[name] = self.required[name]
            if len(self.selected) == len(self.required):
                break

    def forward(self, x):
        features = {}
        for name, layer in self.layers._modules.items():
            x = layer(x)
            if name in self.selected:
                features[self.selected[name]] = x
        return features
