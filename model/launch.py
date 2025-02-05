import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from eval import evaluate
from model import make_model
from train import train

# Add the parent directory of the dataset module to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.dataset import CharacterDataset

parser = argparse.ArgumentParser(description='SEAL')
parser.add_argument('--checkpoint_path', type=str, default='',
                    help='Path to the checkpoint file ("ignore" to force no checkpoint, otherwise latest checkpoint found will be used)')
parser.add_argument('--eval', type=str, default='',
                    help='Which dataset to evaluate (train|test), if empty the model trains on the train dataset')
parser.add_argument('--force_cpu', type=bool, default=False, help='Force to use the CPU instead of CUDA')
parser.add_argument('--discard_optim', type=bool, default=False, help='Discard optim state dict')
parser.add_argument('--disable_hc', type=bool, default=False, help='Disable hierachical classification')
parser.add_argument('--visual_inspect', type=bool, default=True, help='Visualize model predictions')
parser.add_argument('--calc_metrics', type=bool, default=True, help='Calculate performance metrics')

args = parser.parse_args()

checkpoint_path = args.checkpoint_path
eval = args.eval
force_cpu = args.force_cpu
discard_optim = args.discard_optim
disable_hc = args.disable_hc
evaluate_what = []
if args.visual_inspect:
    evaluate_what.append('visual')
if args.calc_metrics:
    evaluate_what.append('metrics')

device = torch.device('cuda') if torch.cuda.is_available() and not force_cpu else torch.device('cpu')
print(f'Using device: {device}')

# Define the dataset and dataloader
data_folder = '../dataset/output'
batch_size = 2
dataset = CharacterDataset(data_folder, split=('train' if not eval else eval), transform=T.ToTensor())
num_classes = len(dataset.classes)
num_superclasses = len(dataset.radical_counts)
superclasses_groups = dataset.radical_groups

model, multiscale_roi_align = make_model(device, dataset.mean, dataset.std, disable_hc, num_superclasses, superclasses_groups)

print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')

if not eval:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    train(device, model, multiscale_roi_align, dataset, dataloader, batch_size, checkpoint_path, discard_optim)
    quit()

print(f"Evaluating on {eval} set")
dataloader = DataLoader(dataset, batch_size=1, shuffle=(True if eval == 'train' else False), collate_fn=lambda x: tuple(zip(*x)))
evaluate(evaluate_what, device, model, multiscale_roi_align, dataset, dataloader, checkpoint_path, discard_optim, max_images=(10 if eval == 'train' else None))
