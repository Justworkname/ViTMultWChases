import sys
import os
import argparse
import logging
import json
import time
import subprocess
from shutil import copyfile

import numpy as np
from sklearn import metrics
from easydict import EasyDict as edict
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import DataParallel
from mit_ub.model.backbone.vit import ViTConfig
from mit_ub.model.layers.pool import PoolType, get_global_pooling_layer
from soap import soapy as soapy
from torch.optim import AdamW
from tensorboardX import SummaryWriter
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

import wandb
from data.dataset import ImageDataset  # noqa
from model.classifier import Classifier  # noqa
from utils.misc import lr_schedule  # noqa
from model.utils import get_optimizer  # noqa
from mit_ub.model.backbone import ViT,AdaptiveViT
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score

torch.set_float32_matmul_precision('high')
import torch._dynamo
torch._dynamo.config.suppress_errors = True

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help="Path to the config file in yaml format")
parser.add_argument('save_path', default=None, metavar='SAVE_PATH', type=str,
                    help="Path to the saved models")
parser.add_argument('--num_workers', default=8, type=int, help="Number of "
                    "workers for each data loader")
parser.add_argument('--device_ids', default='0,1,2,3', type=str,
                    help="GPU indices ""comma separated, e.g. '0,1' ")
parser.add_argument('--pre_train', default=None, type=str, help="If get"
                    "parameters from pretrained model")
parser.add_argument('--resume', default=0, type=int, help="If resume from "
                    "previous run")
parser.add_argument('--logtofile', default=False, type=bool, help="Save log "
                    "in save_path/log.txt if set True")
parser.add_argument('--verbose', default=False, type=bool, help="Detail info")

from data.dataset import ImageDataset
wandb.init(
    # set the wandb project where this run will be logged
    project="Chase_project",

    # track hyperparameters and run metadata
    config={
    "backbone": "ViT",
    "width": 512,
    "height": 512,
    "dim": 512, 
    "pretrained": "False",
    "dim_feedforward": 2048,
    "in_channels": 1,
    "stochastic_depth": 0.1,
    "batch_size": 16,
    "bias": "False",
    "qk_norm": "True",
    "activation": "relu2",
    "gate_activation": "None",
    "patch_size" : [16,16],
    "num_classes" : 5,
    "dropout" : 0.1,
    "depth": 15,
    "enhance_index": [2,6],
    "enhance_times": 1,
    "nhead" : 8,
    "share_layers": "False",
    "target_shape": [512, 512]
    }
)
def load_data(cfg, num_workers=8, mode='train'):
    # Get label paths from the configuration
    label_path = cfg.get(f'{mode}_csv', None)  # Train/val CSV paths from the JSON config
    if not label_path:
        raise ValueError(f"Label path for {mode} is not specified in the config.")

    # Initialize the ImageDataset for the specified mode
    dataset = ImageDataset(csv_file=label_path, cfg=cfg, mode=mode)

    # Create DataLoader for the specified dataset
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.get('batch_size', 16),
        num_workers=num_workers,
        shuffle=True if mode == 'train' else False,
        pin_memory=True
    )

    return data_loader

def run(args):
    wandb.init(project="your_project_name", entity="your_wandb_account_name", config=args)
    # Load configuration from JSON file
    with open(args.cfg_path, 'r') as f:
        cfg = json.load(f)

    # Load train and validation data
    train_loader = load_data(cfg, num_workers=args.num_workers, mode='train')
    val_loader = load_data(cfg, num_workers=args.num_workers, mode='dev')

    vit_config = ViTConfig(
        in_channels=cfg['in_channels'],
        dim=cfg['dim'],
        patch_size=cfg['patch_size'],
        dropout=cfg['dropout'],
        depth=cfg['depth'],
        checkpoint=cfg['pretrained'],
        dim_feedforward=cfg['dim_feedforward'],
        nhead=cfg['nhead'],
        pretrained=cfg['pretrained'],
        num_classes=cfg['num_classes'],
        share_layers=cfg['share_layers'],
        pool_type=cfg['pool_type'],
        target_shape=cfg['target_shape']
    )
    class AdaptiveViTWithGAP(nn.Module):
        def __init__(self, vit_model, num_classes):
            super().__init__()
            self.vit_model = vit_model  # AdaptiveViT backbone
            self.fc1 = nn.Linear(512, 256)  # First fully connected layer
            self.bn1 = nn.BatchNorm1d(256)  # BatchNorm after the first FC layer
            self.relu = nn.ReLU()  # ReLU activation after BatchNorm
            self.dropout = nn.Dropout(0.1)  # Dropout with a probability of 0.1
            self.fc2 = nn.Linear(256, 128)  # Second fully connected layer
            self.bn2 = nn.BatchNorm1d(128)  # BatchNorm after the second FC layer
            self.fc3 = nn.Linear(128, num_classes)  # Final classification layer

        def forward(self, x):
            feature_map, cls_tokens = self.vit_model(x)  # Extract cls_tokens from ViT

            # Use cls_tokens instead of feature_map
            x = self.fc1(cls_tokens)  # First FC layer
            x = self.bn1(x)  # BatchNorm
            x = self.relu(x)  # ReLU activation
            x = self.dropout(x)  # Dropout

            x = self.fc2(x)  # Second FC layer
            x = self.bn2(x)  # BatchNorm
            x = self.relu(x)  # ReLU activation

            output = self.fc3(x)  # Final output layer
            return output, cls_tokens

    # Initialize model
    #vit_model = AdaptiveViT(vit_config)  # Ensure vit_config is defined
    vit_model = ViT(vit_config)
    if vit_config.pretrained:
        # Load the checkpoint
        checkpoint = load_file('pretrained/checkpoint.safetensors')
        
        # Check the checkpoint keys
        print("Checkpoint keys:", checkpoint.keys())

        # Attempt to load the state dict
        try:
            vit_model.load_state_dict(checkpoint, strict=False)
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading state_dict: {e}")
        
    model = AdaptiveViTWithGAP(vit_model, num_classes=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Optimizer and loss function
    #optimizer = soapy.SOAP(model.parameters(), lr=0.001)
    optimizer = AdamW(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_outputs = []

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float()  # Ensure labels are float

            optimizer.zero_grad()

            # Forward pass
            outputs, cls_tokens = model(inputs)
            cls_tokens = cls_tokens.detach()  # Detach cls_tokens from the computation graph

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backpropagate the loss
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

            # Apply sigmoid to the output logits to get probabilities for each class
            outputs = torch.sigmoid(outputs)

            # Convert probabilities to binary predictions (0 or 1) using a threshold of 0.5
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().detach().numpy())
            predicted = (outputs > 0.5).float()


            # Calculate accuracy (count the number of correct predictions)
            correct += (predicted == labels).sum().item()
            total += labels.size(0) * labels.size(1)  # Multiply by number of labels per sample

            # Print actual vs predicted labels every 100 batches
            if i % 50 == 2:  # Print every 100 batches
                avg_loss = running_loss / 100
                accuracy = 100 * correct / total
                all_labels_batch = np.concatenate(all_labels, axis=0)
                all_outputs_batch = np.concatenate(all_outputs, axis=0)
                # Compute AUC (use average AUC for all classes)
                auc = roc_auc_score(all_labels_batch, all_outputs_batch, average='macro')
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, AUC: {auc:.2f}")
                # Log AUC to wandB
                wandb.log({
                    "Loss": avg_loss,
                    "Accuracy": accuracy,
                    "Batch AUC": auc,
                    "Epoch": epoch + 1,
                    "Batch": i + 1
                })
                running_loss = 0.0

        # Optionally, save model checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(args.save_path, f"checkpoint_epoch_{epoch+1}.pt"))

        # Log epoch-level metrics to wandB
        avg_loss_epoch = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        wandb.log({
            "Epoch Loss": avg_loss_epoch,
            "Epoch Accuracy": epoch_accuracy,
            "Epoch": epoch + 1
        })
    wandb.finish()

def main():
    args = parser.parse_args()
    if args.verbose is True:
        print('Using the specified args:')
        print(args)

    run(args)

if __name__ == '__main__':
    main()
