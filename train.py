import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

from models.model import SimpleCNN
from dataset.dataset import get_tiny_imagenet_loaders
from utils.utils import save_checkpoint
from eval import evaluate_model

def train(args):
    # Initialize wandb
    wandb.init(project="mldl_lab3", config=args)
    config = wandb.config

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataloaders
    train_loader, test_loader = get_tiny_imagenet_loaders(config.data_dir, config.batch_size)

    # Model
    model = SimpleCNN(num_classes=200).to(device)
    wandb.watch(model, log='all')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)

    # Best accuracy tracking
    best_val_accuracy = 0.0

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        
        # Calculate accuracy
        train_accuracy = evaluate_model(model, train_loader, device)
        val_accuracy = evaluate_model(model, test_loader, device)

        wandb.log({
            "epoch": epoch + 1, 
            "loss": epoch_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy
        })
        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")

        # Save checkpoint
        if not os.path.exists(config.checkpoint_dir):
            os.makedirs(config.checkpoint_dir)
        
        # Check if this is the best model
        is_best = val_accuracy > best_val_accuracy
        if is_best:
            best_val_accuracy = val_accuracy
            print(f"New best model! Val Accuracy: {val_accuracy:.2f}%")
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_val_accuracy
        }, is_best=is_best, checkpoint_dir=config.checkpoint_dir)


    print('Finished Training')
    print(f'Best Validation Accuracy: {best_val_accuracy:.2f}%')
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=10, type=int, help='number of epochs to train')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--data_dir', default='./data', type=str, help='directory for dataset')
    parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str, help='directory for checkpoints')
    args = parser.parse_args()
    train(args)

