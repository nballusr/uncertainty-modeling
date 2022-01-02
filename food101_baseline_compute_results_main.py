import argparse
import pytorch_lightning as pl
import torchvision.transforms as transforms
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.datasets as datasets
import os
import torch
import numpy as np

from models import Food101Baseline

parser = argparse.ArgumentParser(description='PyTorch FOOD101 Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
# parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training and validation")
# parser.add_argument("--num_workers", type=int, required=True, help="Number of workers for the dataloaders")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint of the baseline trained")
parser.add_argument("--num_tests", type=int, required=True, help="Number of tests to run")
parser.add_argument("--train_results_path", type=str, required=True, help="File where to save training dataset results")
parser.add_argument("--val_results_path", type=str, required=True, help="File where to save validation dataset results")
parser.add_argument("--train_logits_path", type=str, required=True, help="File where to save training dataset logits")
parser.add_argument("--val_logits_path", type=str, required=True, help="File where to save validation dataset logits")

args = parser.parse_args()

seed = 1
pl.seed_everything(seed)

gpus = 1 if torch.cuda.is_available() else 0
device = 'cuda' if gpus else 'cpu'

# Data
print('==> Preparing data..')
traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_set = datasets.ImageFolder(
    traindir,
    transform_train
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=False, num_workers=8)

val_set = datasets.ImageFolder(
    valdir,
    transform_val
)

val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, num_workers=8)

baseline = Food101Baseline.load_from_checkpoint(checkpoint_path=args.checkpoint, learning_rate=0, scheduler_length=0)

if gpus > 0:
    baseline = baseline.cuda()

train_results = baseline.compute_results(train_loader, 75750, 101, args.num_tests, gpus)
train_logits = baseline.compute_logits(train_loader, 75750, 101, gpus)

val_results = baseline.compute_results(val_loader, 25250, 101, args.num_tests, gpus)
val_logits = baseline.compute_logits(val_loader, 25250, 101, gpus)

np.save(args.train_results_path, train_results)
np.save(args.val_results_path, val_results)
np.save(args.train_logits_path, train_logits)
np.save(args.val_logits_path, val_logits)

print("finished")
