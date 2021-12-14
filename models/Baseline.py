from pytorch_lightning.core.lightning import LightningModule
from models import ResNet18


class Baseline(LightningModule):
    def __init__(self):
        super().__init__()

        self.model = ResNet18()

    def forward(self, x):
        return self.model(x)
