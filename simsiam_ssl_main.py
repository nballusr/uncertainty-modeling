from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as transforms

from models import SimSiamModel
from util import TwoCropsTransform

parser = ArgumentParser()
parser.add_argument("--num_mlp_layers", type=int, required=True, help="Number of MLP layers (2 or 3)")
parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
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
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=TwoCropsTransform(transforms.Compose(transform_train)))

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=512, shuffle=True, drop_last=True, num_workers=2)

trainset_knn = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_val)

train_knn_loader = torch.utils.data.DataLoader(
    trainset_knn, batch_size=512, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_val)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=512, shuffle=False, num_workers=2)

if args.num_mlp_layers == 2:
    num_mlp_layers = 2
    initial_lr = 0.03
    weight_decay = 0.0005
elif args.num_mlp_layers == 3:
    print("TO TO")
    exit()
else:
    print("Not correct number of MLPs")
    exit()

model = SimSiamModel(train_knn_loader, gpus, 10, knn_k, knn_t, args.epochs, num_mlp_layers,
                     initial_lr, weight_decay)

trainer = pl.Trainer(max_epochs=args.epochs, gpus=gpus, progress_bar_refresh_rate=100)

trainer.fit(
    model,
    train_dataloader=trainloader,
    val_dataloaders=testloader
)

print(f'Highest test accuracy: {model.max_accuracy:.4f}')

print("finished")


