from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.nn as nn

from models import ResNet18


class Baseline(LightningModule):
    def __init__(self):
        super().__init__()

        self.model = ResNet18()
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = 0.1
        self.scheduler_length = 200

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = batch_idx #self.criterion(outputs, labels)
        #self.log('train_loss', loss)
        #computar els que estan be i tambe fer un log
        #posar tant loss com accuracy en un array
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.scheduler_length)
        return [optimizer], [scheduler]
