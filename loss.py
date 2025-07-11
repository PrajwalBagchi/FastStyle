import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import gram_matrix

class StyleTransferLoss(nn.Module):
    def __init__(self, vgg, style_grams, content_weight, style_weight, tv_weight, device):
        super(StyleTransferLoss, self).__init__()
        self.vgg = vgg.to(device).eval()
        self.style_grams = style_grams  # precomputed in trainer.py
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight

        for param in self.vgg.parameters():
            param.requires_grad = False

    def total_variation_loss(self, img):
        x_diff = img[:, :, :, 1:] - img[:, :, :, :-1]
        y_diff = img[:, :, 1:, :] - img[:, :, :-1, :]
        return torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))

    def forward(self, content_img, stylized_img):
        with torch.no_grad():
            content_features = self.vgg(content_img)
        stylized_features = self.vgg(stylized_img)


        # Content Loss (e.g., relu2_2)
        content_loss = F.mse_loss(
            stylized_features["relu2_2"],
            content_features["relu2_2"]
        ) * self.content_weight

        # Style Loss
        style_loss = 0.0
        for layer in self.style_grams:
            target_gram = gram_matrix(stylized_features[layer])
            style_gram = self.style_grams[layer]
            style_loss += F.mse_loss(target_gram, style_gram.expand_as(target_gram))
        style_loss *= self.style_weight

        # Total Variation Loss
        tv_loss = self.total_variation_loss(stylized_img) * self.tv_weight

        total = content_loss + style_loss + tv_loss
        return total, content_loss, style_loss, tv_loss
