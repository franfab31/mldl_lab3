import argparse
import torch
from tqdm import tqdm

from models.model import SimpleCNN
from dataset.dataset import get_tiny_imagenet_loaders
from utils.utils import load_checkpoint

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def evaluate(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloaders
    _, test_loader = get_tiny_imagenet_loaders(args.data_dir, args.batch_size)

    # Model
    model = SimpleCNN(num_classes=200).to(device)

    # Load checkpoint
    if args.checkpoint_path:
        load_checkpoint(args.checkpoint_path, model)

    accuracy = evaluate_model(model, test_loader, device)
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet Evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--data_dir', default='./data', type=str, help='directory for dataset')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='path to checkpoint for evaluation')
    args = parser.parse_args()
    evaluate(args)
