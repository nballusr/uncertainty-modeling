import argparse
import pytorch_lightning as pl
import torch.cuda
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from models import Baseline

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training and validation")
# parser.add_argument("--num_workers", type=int, required=True, help="Number of workers for the dataloaders")
parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--rand-aug', default=False, type=bool, help='Whether to use rand aug or not')
parser.add_argument('--resume', required=False, type=str, help='checkpoint of the model to restore the training from')
parser.add_argument('--warm-restart', required=False, default=-1, type=int, help='After how many epochs restart the'
                                                                                 'cosine annealing. If it is -1, no'
                                                                                 'warm restarts.')

args = parser.parse_args()

seed = 1
pl.seed_everything(seed)

gpus = 1 if torch.cuda.is_available() else 0
device = 'cuda' if gpus else 'cpu'

# Data
print('==> Preparing data..')
if args.rand_aug:
    transform_train = transforms.Compose([
        transforms.RandAugment(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
else:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

baseline = Baseline(learning_rate=args.lr, scheduler_length=args.epochs)

checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=-1, mode="min")
trainer = pl.Trainer(max_epochs=args.epochs, gpus=gpus, callbacks=[checkpoint_callback])

if not args.resume:
    trainer.fit(baseline, train_dataloaders=trainloader, val_dataloaders=testloader)
else:
    trainer.fit(baseline, train_dataloaders=trainloader, val_dataloaders=testloader, ckpt_path=args.resume)

# retrieve the best checkpoint after training
print("Best model checkpoint:", checkpoint_callback.best_model_path)

baseline.save_metrics()

print("finished")
