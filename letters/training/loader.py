from glob import glob
import os
import random

import numpy as np
import imageio
from torch.utils.data import Dataset

def generate_datasets(lang_dir_path):
    char_paths = glob(f'{lang_dir_path}/*/*.png')
    random.shuffle(char_paths)

    classes = sorted(list(set(map(lambda f: f.split('/')[-2], char_paths))))

    split_index = int(len(char_paths) * 0.85)
    train_paths = char_paths[:split_index]
    val_paths = char_paths[split_index:]

    train_set = CharDataset(train_paths, classes)
    val_set = CharDataset(val_paths, classes)

    return classes, train_set, val_set

def generate_all_datasets(lang_dir_path):
    char_paths = glob(f'{lang_dir_path}/*/*.png')
    random.shuffle(char_paths)
    classes = sorted(list(set(map(lambda f: f.split('/')[-2], char_paths))))
    total_set = CharDataset(char_paths, classes)
    return classes, total_set

class CharDataset(Dataset):
    def __init__(self, char_paths, classes):
        self.char_paths = char_paths
        self.dataset = []
        for char_path in char_paths:
            char = char_path.split('/')[-2]
            class_index = classes.index(char)

            img = imageio.imread(char_path)
            img = np.expand_dims(np.asarray((255-img[:,:,0])/255.0, dtype=np.float32), 0)
            self.dataset.append((img, class_index))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
