import argparse
import pytorch_lightning as pl
import torchvision.transforms as transforms
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.datasets as datasets
import torch
import os

from models import Food101Baseline

parser = argparse.ArgumentParser(description='PyTorch FOOD101 Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
# parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training and validation")
# parser.add_argument("--num_workers", type=int, required=True, help="Number of workers for the dataloaders")
parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--checkpoint', required=False, type=str, help='checkpoint of the pretrained model')
parser.add_argument('--rand-aug', default=False, type=bool, help='Whether to use rand aug or not')
parser.add_argument('--resume', required=False, type=str, help='checkpoint of the model to restore the training from')
parser.add_argument('--warm-restart', required=False, default=-1, type=int, help='After how many epochs restart the '
                                                                                 'cosine annealing. If it is -1, no '
                                                                                 'warm restarts.')
parser.add_argument('--auto-lr-find', default=False, type=bool, help='Whether to auto find the best lr. It it is set '
                                                                     'to true, --lr is ignored')
parser.add_argument('--checkpoint-linear', required=False, type=str, help='checkpoint of the linear model already '
                                                                          'trained for some epochs')
parser.add_argument('--early-stopping', default=-1, type=int, help='patience for early stopping. It it is -1, no '
                                                                   'early stopping is used.')
parser.add_argument('--eta-min', default=0, type=float, help='eta min for cosine scheduler')
parser.add_argument('--factor-warm-restart', required=False, default=1, type=int, help='A factor increases the number '
                                                                                       'of epochs of a restart when'
                                                                                       'using warm-restart.')

args = parser.parse_args()

# Assert that only one checkpoint is passed
assert (args.checkpoint and not args.resume and not args.checkpoint_linear) or \
       (not args.checkpoint and args.resume and not args.checkpoint_linear) or \
       (not args.checkpoint and not args.resume and args.checkpoint_linear)

seed = 1
pl.seed_everything(seed)

gpus = 1 if torch.cuda.is_available() else 0
device = 'cuda' if gpus else 'cpu'

# Data
print('==> Preparing data..')
traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')

if args.rand_aug:
    transform_train = transforms.Compose([
        transforms.RandAugment(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
else:
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

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=8)

val_set = datasets.ImageFolder(
    valdir,
    transform_val
)

val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, num_workers=8)

if not args.checkpoint_linear:
    baseline = Food101Baseline(learning_rate=args.lr, scheduler_length=args.epochs, warm_restart=args.warm_restart,
                               eta_min=args.eta_min, t_mult=args.factor_warm_restart)
else:
    baseline = Food101Baseline.load_from_checkpoint(checkpoint_path=args.checkpoint_linear, learning_rate=args.lr,
                                                    scheduler_length=args.epochs, warm_restart=args.warm_restart,
                                                    eta_min=args.eta_min, t_mult=args.factor_warm_restart)
    print("=> loaded linear trained model '{}'".format(args.checkpoint_linear))

if not args.checkpoint:
    print("Not resuming from a pre-trained model")
else:
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        state_dict = checkpoint['model']
        new_state_dict = dict()

        for old_key, value in state_dict.items():
            if old_key.startswith('module.backbone.conv1'):
                new_key = old_key.replace('module.backbone.conv1', 'model.model.0')
                new_state_dict[new_key] = value

            elif old_key.startswith('module.backbone.bn1'):
                new_key = old_key.replace('module.backbone.bn1', 'model.model.1')
                new_state_dict[new_key] = value

            elif old_key.startswith('module.backbone.layer1'):
                new_key = old_key.replace('module.backbone.layer1', 'model.model.4')
                new_state_dict[new_key] = value

            elif old_key.startswith('module.backbone.layer2'):
                new_key = old_key.replace('module.backbone.layer2', 'model.model.5')
                new_state_dict[new_key] = value

            elif old_key.startswith('module.backbone.layer3'):
                new_key = old_key.replace('module.backbone.layer3', 'model.model.6')
                new_state_dict[new_key] = value

            elif old_key.startswith('module.backbone.layer4'):
                new_key = old_key.replace('module.backbone.layer4', 'model.model.7')
                new_state_dict[new_key] = value

        msg = baseline.load_state_dict(new_state_dict, strict=False)
        print(msg)
        assert set(msg.missing_keys) == {"model.linear.weight", "model.linear.bias"}
        print("=> loaded pre-trained model '{}'".format(args.checkpoint))
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
        exit()

checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=-1, mode="min")

training_callbacks = [checkpoint_callback]
if args.early_stopping != -1:
    training_callbacks.append(EarlyStopping(monitor="val_loss", patience=args.early_stopping))

if not args.resume:
    if args.auto_lr_rate:
        trainer = pl.Trainer(max_epochs=args.epochs, gpus=gpus, callbacks=training_callbacks, auto_lr_find=True)
        trainer.tune(baseline, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        trainer = pl.Trainer(max_epochs=args.epochs, gpus=gpus, callbacks=training_callbacks)
    trainer.fit(baseline, train_dataloaders=train_loader, val_dataloaders=val_loader)
else:
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=gpus, callbacks=training_callbacks)
    trainer.fit(baseline, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume)

# retrieve the best checkpoint after training
print("Best model checkpoint:", checkpoint_callback.best_model_path)

baseline.save_metrics()

print("finished")
