from pytorch_lightning.core.lightning import LightningModule
from models import ResNet18


class Baseline(LightningModule):
    def __init__(self):
        super().__init__()

        self.model = ResNet18()

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
