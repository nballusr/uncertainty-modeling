import lightly
import torch
import torch.nn as nn

from models.benchmark import BenchmarkModule
from models import ResNet18
from loss import BarlowTwinsLoss


class BarlowTwinsModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, gpus, knn_k, knn_t, cosine_schedule_length, num_mlp_layers, initial_lr,
                 weight_decay):
        super().__init__(dataloader_kNN, gpus, 10, knn_k, knn_t)
        # create a ResNet backbone and remove the classification head
        resnet = ResNet18()
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a simsiam model based on ResNet
        # note that barlowtwins has the same architecture
        self.resnet_simsiam = \
            lightly.models.SimSiam(self.backbone, num_ftrs=512, num_mlp_layers=num_mlp_layers)
        device = 'cuda' if gpus else 'cpu'
        self.criterion = BarlowTwinsLoss(device=device)
        self.initial_lr = initial_lr
        self.weight_decay = weight_decay
        self.cosine_schedule_length = cosine_schedule_length

    def forward(self, x):
        return self.resnet_simsiam(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        num_total = len(x0)
        x0, x1 = self.resnet_simsiam(x0, x1)
        # our simsiam model returns both (features + projection head)
        z_a, _ = x0
        z_b, _ = x1
        loss = self.criterion(z_a, z_b)
        self.log('train_loss_ssl', loss)
        return {'loss': loss, 'num_total': num_total}

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_simsiam.parameters(), lr=self.initial_lr,
                                momentum=0.9, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.cosine_schedule_length)
        return [optim], [scheduler]
