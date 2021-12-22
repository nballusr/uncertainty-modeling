from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
import lightly
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint

from models import BarlowTwinsModel

parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training and validation")
parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for the dataloaders")
parser.add_argument("--epochs", type=int, default=800, help="Number of training epochs")
parser.add_argument("--continue_from_checkpoint", type=str, help="Checkpoint to continue training from")
parser.add_argument("--num_mlp_layers", type=int, default=3, help="Number of MLP layers. It can be 2 or 3")
args = parser.parse_args()

knn_k = 200
knn_t = 0.1
seed = 1
pl.seed_everything(seed)

# use a GPU if available
gpus = 1 if torch.cuda.is_available() else 0
device = 'cuda' if gpus else 'cpu'

# Data
print('==> Preparing data..')
# Use SimCLR augmentations, additionally, disable blur
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=32,
    gaussian_blur=0.,
    normalize={'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]}
)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

dataset_train_ssl = lightly.data.LightlyDataset.from_torch_dataset(
    torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        download=True))
dataset_train_kNN = lightly.data.LightlyDataset.from_torch_dataset(torchvision.datasets.CIFAR10(
    root='data',
    train=True,
    transform=test_transforms,
    download=True))
dataset_test = lightly.data.LightlyDataset.from_torch_dataset(torchvision.datasets.CIFAR10(
    root='data',
    train=False,
    transform=test_transforms,
    download=True))

dataloader_train_ssl = torch.utils.data.DataLoader(
    dataset_train_ssl,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=args.num_workers
)
dataloader_train_kNN = torch.utils.data.DataLoader(
    dataset_train_kNN,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=args.num_workers
)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=args.num_workers
)

if args.num_mlp_layers == 2:
    print("Not implemented yet")
    exit()
elif args.num_mlp_layers == 3:
    num_mlp_layers = 3
    initial_lr = 1e-3
    weight_decay = 5e-4
else:
    print("Not available number for MLPs")
    exit()

model = BarlowTwinsModel(dataloader_train_kNN, gpus, knn_k, knn_t, args.epochs, num_mlp_layers, initial_lr,
                         weight_decay)

checkpoint_callback = ModelCheckpoint(save_top_k=-1)
trainer = pl.Trainer(max_epochs=args.epochs, gpus=gpus, progress_bar_refresh_rate=100, callbacks=[checkpoint_callback])

if args.continue_from_checkpoint:
    trainer.fit(
        model,
        train_dataloader=dataloader_train_ssl,
        val_dataloaders=dataloader_test,
        ckpt_path=args.continue_from_checkpoint
    )
else:
    trainer.fit(
        model,
        train_dataloader=dataloader_train_ssl,
        val_dataloaders=dataloader_test
    )

model.save_metrics()

print(f'Highest test accuracy: {model.max_accuracy:.4f}')

print("finished")
