import torch.nn as nn
import torchvision
import torch


class ResNet50WithDropout(nn.Module):
    """ResNet implementation.

    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
        num_classes:
            Number of classes in final softmax layer.
    """

    def __init__(self, pretrained: bool, num_classes: int = 101):

        super(ResNet50WithDropout, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model = nn.Sequential(*(list(self.model.children())[:-1]))
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(in_features, 101)

    def forward(self, x: torch.Tensor):
        """Forward pass through ResNet.

        Args:
            x:
                Tensor of shape bsz x channels x W x H

        Returns:
            Output tensor of shape bsz x num_classes

        """

        out = self.model(x)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def logits(self, x: torch.Tensor):
        out = self.model(x)
        out = out.view(out.size(0), -1)
        return out
