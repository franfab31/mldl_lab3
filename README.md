# MLDL Lab 3: Structuring a Deep Learning Project from Scratch

My solution to Lab 3 of the *Machine Learning and Deep Learning* course at Politecnico di Torino.
The goal of the lab is to turn notebook-style code into a clean, modular PyTorch project: a training script and an evaluation script with command-line arguments, reusable dataset and model modules, checkpointing, and experiment tracking with Weights & Biases.

The task is image classification on **Tiny ImageNet** (200 classes) with a simple CNN.

## Project structure

```text
mldl_lab3/
├── train.py            # Training loop, W&B logging, best-model checkpointing
├── eval.py             # Evaluates a saved checkpoint on the validation set
├── models/model.py     # SimpleCNN
├── dataset/dataset.py  # Tiny ImageNet download + DataLoaders (also a CIFAR-10 loader)
├── utils/utils.py      # save_checkpoint / load_checkpoint
├── data/               # Dataset is downloaded here
├── checkpoints/        # Saved checkpoints
└── requirements.txt
```

## Model

`SimpleCNN` has five 3×3 convolutions with stride 2 and ReLU (64 → 128 → 256 → 512 → 1024 channels).
They are followed by global average pooling and a linear layer with 200 outputs.
Input images are resized to 224×224 and normalised with ImageNet statistics.

## Setup

```bash
git clone https://github.com/franfab31/mldl_lab3.git
cd mldl_lab3
pip install -r requirements.txt
wandb login   # or set WANDB_MODE=offline to skip online logging
```

On the first run, Tiny ImageNet (~240 MB) is downloaded into `./data`.
Its validation folder is then reorganised so that `torchvision.datasets.ImageFolder` can read it.

## Usage

Train (default values shown):

```bash
python train.py --lr 0.001 --epochs 10 --batch_size 128 --data_dir ./data --checkpoint_dir ./checkpoints
```

Training uses SGD with momentum 0.9. Loss, train accuracy and validation accuracy are logged to the `mldl_lab3` W&B project after each epoch.
Two checkpoints are written to `checkpoints/`: `checkpoint.pth.tar` (last epoch) and `model_best.pth.tar` (best validation accuracy).

Evaluate a checkpoint:

```bash
python eval.py --checkpoint_path checkpoints/model_best.pth.tar
```

## Credits

The course provided the empty project skeleton (folder layout and `.gitignore`).
I wrote the data pipeline, the model, the training and evaluation scripts, the checkpoint utilities and `requirements.txt`.
