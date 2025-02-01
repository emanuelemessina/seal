import glob
import re

import torch


def load_checkpoint(checkpoint_path, discard_optim, model, optimizer=None, scheduler=None):
    try:
        if checkpoint_path == 'ignore':
            print("Forced no checkpoint")
            raise Exception
        if not checkpoint_path:
            checkpoint_files = glob.glob(f'checkpoint_*.pth')
            if len(checkpoint_files) == 0:
                print("No checkpoint, training from scratch.")
                raise Exception  # to break out
            checkpoint_files.sort(key=lambda x: re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', x).group(), reverse=True)
            checkpoint_path = checkpoint_files[0]
        print(f'Using checkpoint {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        if not discard_optim:
            if optimizer:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    except Exception as e:
        if e is FileNotFoundError:
            print(f"No checkpoint found at '{checkpoint_path}'.")
            quit()
