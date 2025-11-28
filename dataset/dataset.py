import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import requests
import zipfile
import shutil
from torchvision.datasets import ImageFolder

def get_cifar10_loaders(data_dir, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root=data_dir, train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root=data_dir, train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader


# --- Tiny ImageNet Code ---

def download_and_unzip_tiny_imagenet(data_dir):
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_path = os.path.join(data_dir, 'tiny-imagenet-200.zip')
    dataset_path = os.path.join(data_dir, 'tiny-imagenet-200')

    if os.path.exists(dataset_path):
        print('Tiny ImageNet dataset already exists.')
        return dataset_path

    os.makedirs(data_dir, exist_ok=True)

    print('Downloading Tiny ImageNet...')
    r = requests.get(url, stream=True)
    with open(zip_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    
    print('Unzipping Tiny ImageNet...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    os.remove(zip_path)
    print('Tiny ImageNet downloaded and unzipped.')
    return dataset_path

def reorganize_val_folder(val_dir):
    val_annotations_path = os.path.join(val_dir, 'val_annotations.txt')
    images_dir = os.path.join(val_dir, 'images')

    if not os.path.exists(images_dir):
        print("Validation folder already reorganized.")
        return

    with open(val_annotations_path) as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            class_dir = os.path.join(val_dir, cls)
            os.makedirs(class_dir, exist_ok=True)
            
            src_path = os.path.join(images_dir, fn)
            dst_path = os.path.join(class_dir, fn)
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)

    shutil.rmtree(images_dir)
    os.remove(val_annotations_path)
    print("Validation folder reorganized.")


def get_tiny_imagenet_loaders(data_dir, batch_size):
    dataset_root = download_and_unzip_tiny_imagenet(data_dir)
    train_dir = os.path.join(dataset_root, 'train')
    val_dir = os.path.join(dataset_root, 'val')

    # Reorganize val folder to be compatible with ImageFolder
    reorganize_val_folder(val_dir)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(root=train_dir, transform=transform)
    val_dataset = ImageFolder(root=val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader
