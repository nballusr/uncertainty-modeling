from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.nn as nn
import numpy as np
import torchvision


def correct_predict(feature, labels):
    return (feature.argmax(axis=1) == labels).sum()


class Food101Baseline(LightningModule):
    def __init__(self, learning_rate, scheduler_length):
        super().__init__()

        self.model = torchvision.models.resnet50(pretrained=True)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.scheduler_length = scheduler_length
        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        num_correct = correct_predict(outputs.squeeze(), labels)
        return {'loss': loss, 'num_correct': num_correct, 'num_total': images.size(0)}

    def training_epoch_end(self, outputs):
        num_total = 0
        num_correct = 0.
        epoch_loss = 0.
        for element in outputs:
            epoch_loss += element['loss'] * element['num_total']
            num_correct += element['num_correct']
            num_total += element['num_total']
        epoch_accuracy = 100. * num_correct / num_total
        epoch_loss = epoch_loss / num_total

        self.log('train_accuracy', epoch_accuracy, prog_bar=True)
        self.log('train_loss', epoch_loss, prog_bar=True)
        self.train_accuracy.append(float(epoch_accuracy.cpu()))
        self.train_loss.append(float(epoch_loss.cpu()))

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        num_correct = correct_predict(outputs.squeeze(), labels)
        return {'val_loss': loss, 'val_num_correct': num_correct, 'val_num_total': images.size(0)}

    def validation_epoch_end(self, outputs):
        num_total = 0
        num_correct = 0.
        val_loss = 0.
        for element in outputs:
            val_loss += element['val_loss'] * element['val_num_total']
            num_correct += element['val_num_correct']
            num_total += element['val_num_total']
        val_accuracy = 100. * num_correct / num_total
        val_loss = val_loss / num_total

        self.log('val_accuracy', val_accuracy, prog_bar=True)
        self.log('val_loss', val_loss, prog_bar=True)
        self.val_accuracy.append(float(val_accuracy.cpu()))
        self.val_loss.append(float(val_loss.cpu()))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.scheduler_length)
        return [optimizer], [scheduler]

    def save_metrics(self):
        np.save("train_loss", self.train_loss)
        np.save("train_accuracy", self.train_accuracy)
        np.save("val_loss", self.val_loss)
        np.save("val_accuracy", self.val_accuracy)
