from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.nn as nn

from models import ResNet18


def correct_predict(feature, labels):
    return (feature.argmax(axis=1) == labels).sum()


class Baseline(LightningModule):
    def __init__(self):
        super().__init__()

        self.model = ResNet18()
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = 0.1
        self.scheduler_length = 200
        self.train_loss = []
        self.train_accuracy = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        num_correct = correct_predict(outputs.squeeze(), labels)
        return {'loss': loss, 'num_correct': num_correct, 'num_total': images.size(0)}

    def training_epoch_end(self, outputs):
        num_total = 0
        num_correct = 0.
        epoch_loss = 0.
        for element in outputs:
            epoch_loss += element['loss']*element['num_total']
            num_correct += element['num_correct']
            num_total += element['num_total']
        epoch_accuracy = 100. * num_correct / num_total
        epoch_loss = epoch_loss / num_total

        self.log('train_accuracy', epoch_accuracy, prog_bar=True)
        self.log('train_loss', epoch_loss, prog_bar=True)
        self.train_accuracy.append(epoch_accuracy.cpu())
        self.train_loss.append(epoch_loss.cpu())


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.scheduler_length)
        return [optimizer], [scheduler]
