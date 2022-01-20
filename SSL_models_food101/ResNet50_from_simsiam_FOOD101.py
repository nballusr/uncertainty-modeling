import torch.nn as nn
import torch


class ResNet50FromSimSiamWithDropout(nn.Module):

    def __init__(self, simsiam_model: nn.Module, num_classes: int = 101):

        super(ResNet50FromSimSiamWithDropout, self).__init__()
        self.model = simsiam_model
        # Continuar a aqui
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
