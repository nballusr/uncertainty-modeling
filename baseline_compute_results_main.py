import argparse
import pytorch_lightning as pl
import torch.cuda
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np

from models import Baseline

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
#parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training and validation")
#parser.add_argument("--num_workers", type=int, required=True, help="Number of workers for the dataloaders")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint of the baseline trained")
parser.add_argument("--num_tests", type=int, required=True, help="Number of tests to run")
parser.add_argument("--train_results_path", type=str, required=True, help="File where to save training dataset results")
parser.add_argument("--val_results_path", type=str, required=True, help="File where to save validation dataset results")

args = parser.parse_args()

seed = 1
pl.seed_everything(seed)

gpus = 1 if torch.cuda.is_available() else 0
device = 'cuda' if gpus else 'cpu'

# Data
print('==> Preparing data..')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

baseline = Baseline.load_from_checkpoint(checkpoint_path=args.checkpoint, learning_rate=0, scheduler_length=0)

if gpus > 0:
    baseline = baseline.cuda()

train_results = baseline.compute_results(trainloader, 50000, 10, args.num_tests, gpus)

val_results = baseline.compute_results(testloader, 10000, 10, args.num_tests, gpus)

np.save(args.train_results_path, train_results)
np.save(args.val_results_path, val_results)

print("finished")
