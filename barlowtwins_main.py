import argparse
import pytorch_lightning as pl
import torch.cuda
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import os

from models import Baseline

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training and validation")
# parser.add_argument("--num_workers", type=int, required=True, help="Number of workers for the dataloaders")
parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--checkpoint', required=True, type=str, help='checkpoint of the pretrained model')

args = parser.parse_args()

seed = 1
pl.seed_everything(seed)

gpus = 1 if torch.cuda.is_available() else 0
device = 'cuda' if gpus else 'cpu'

# Data
print('==> Preparing data..')
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

if os.path.isfile(args.checkpoint):
    print("=> loading checkpoint '{}'".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    state_dict = checkpoint['state_dict']
    new_state_dict = dict()

    for old_key, value in state_dict.items():
        if old_key.startswith('resnet_simsiam.backbone.0'):
            new_key = old_key.replace('resnet_simsiam.backbone.0', 'model.conv1')
            new_state_dict[new_key] = value

        elif old_key.startswith('resnet_simsiam.backbone.1'):
            new_key = old_key.replace('resnet_simsiam.backbone.1', 'model.bn1')
            new_state_dict[new_key] = value

        elif old_key.startswith('resnet_simsiam.backbone.2'):
            new_key = old_key.replace('resnet_simsiam.backbone.2', 'model.layer1')
            new_state_dict[new_key] = value

        elif old_key.startswith('resnet_simsiam.backbone.3'):
            new_key = old_key.replace('resnet_simsiam.backbone.3', 'model.layer2')
            new_state_dict[new_key] = value

        elif old_key.startswith('resnet_simsiam.backbone.4'):
            new_key = old_key.replace('resnet_simsiam.backbone.4', 'model.layer3')
            new_state_dict[new_key] = value

        elif old_key.startswith('resnet_simsiam.backbone.5'):
            new_key = old_key.replace('resnet_simsiam.backbone.5', 'model.layer4')
            new_state_dict[new_key] = value

    msg = baseline.load_state_dict(new_state_dict, strict=False)
    print(msg)
    assert set(msg.missing_keys) == {"model.linear.weight", "model.linear.bias"}
    print("=> loaded pre-trained model '{}'".format(args.checkpoint))
else:
    print("=> no checkpoint found at '{}'".format(args.checkpoint))
    exit()

checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=-1, mode="min")
trainer = pl.Trainer(max_epochs=args.epochs, gpus=gpus, callbacks=[checkpoint_callback])

trainer.fit(baseline, train_dataloaders=trainloader, val_dataloaders=testloader)

# retrieve the best checkpoint after training
print("Best model checkpoint:", checkpoint_callback.best_model_path)

baseline.save_metrics()

print("finished")
