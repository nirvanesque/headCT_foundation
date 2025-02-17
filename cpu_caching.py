from monai import data

import torch
from torch.utils.data import Dataset

from src.data.transforms import loading_transforms

import argparse
import pandas as pd
from tqdm import tqdm


class HeadDatasetCache(Dataset):
    def __init__(self, roi, in_channels, csv_file, cache_dir=None):
        self.data = pd.read_csv(csv_file)
        self.load = loading_transforms(roi, in_channels)

        self.cache_dir = cache_dir
        self.cache_dataset = data.PersistentDataset(
            data=list([{"image": d} for d in self.data['img_path']]), 
            transform=self.load, 
            cache_dir=self.cache_dir,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            image = self.cache_dataset.__getitem__(idx)
            print(f"image: {image['image'].shape}")
            return image
        except:
            print("Error: {}".format(idx))


parser = argparse.ArgumentParser(description='Example of a command-line argument parser')

# Positional argument
parser.add_argument('--start_idx', type=int, help='Path to the input file')

# Optional argument
parser.add_argument('--end_idx', type=int, help='Path to the input file')

# Flag argument (boolean)
args = parser.parse_args()

device = torch.device("cuda")

roi = [96, 96, 96]
csv_file = '<path-to>/datasets/dataset.csv'
cache_dir = '<path-to>/embedding_cache'

train_ds = HeadDatasetCache(
    roi, 
    in_channels=3, 
    csv_file=csv_file, 
    cache_dir=cache_dir, 
)

for idx in tqdm(range(args.start_idx, args.end_idx)):
    try:
        b = train_ds.__getitem__(idx)
    except:
        print("Error: {}".format(idx))