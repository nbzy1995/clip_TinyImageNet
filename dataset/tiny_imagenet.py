import os
import torch
from torch.utils.data import SubsetRandomSampler
import numpy as np

from .common import ImageFolderWithPaths


def load_persistent_indices(data_location):
    """
    Load persistent train/val split indices.
    
    ⚠️ These indices should be generated ONCE using generate_train_val_indices.py
    and then used consistently across all experiments.
    """
    idx_file = os.path.join(data_location, 'tiny_imagenet_train_val_indices.npy')
    
    if not os.path.exists(idx_file):
        raise FileNotFoundError(
            f"Persistent indices file not found: {idx_file}\n"
            f"Run generate_train_val_indices.py ONCE to create the indices."
        )
    
    val_indices = np.load(idx_file).astype(bool)
    return val_indices


class TinyImageNetData:
    """
    TinyImageNet dataset class for training, validation, and test splits.
    """
    def __init__(self,
                 preprocess,
                 location=None,
                 batch_size=32,
                 num_workers=2,
                 distributed=False):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed = distributed
        
        self.classname_map = self.get_classname_map()

        self.populate_train()
        self.populate_val()
        self.populate_test()

    def get_classname_map(self):
        """
        Get the word net map, restricted to available classes in dataset. For example, classname_map['n01443537'] gives 'fish'.
        """
        tiny_imagenet_path = os.path.join(self.location, self.name())

        # Use directories as class names the dataset has
        train_dir = os.path.join(tiny_imagenet_path, 'train')
        class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)) and d.startswith('n')]

        # Read words (the mapping from wnid to names)
        with open(os.path.join(tiny_imagenet_path, 'words.txt'), 'r') as f:
            words_dict = {}
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    words_dict[parts[0]] = parts[1]

        classname_map = {}
        for wnid in class_dirs:
            if wnid in words_dict:
                classname_map[wnid] = words_dict[wnid]
            else:
                raise ValueError(f"Class {wnid} not found in words.txt")

        return classname_map

    def populate_train(self):
        """
        This is 90% subset of the original train set, we use as train split.
        Uses persistent indices for consistent train/val split.
        """
        # The original train dataset
        # ImageFolder loads classes in alphabetical order (deterministic across platforms)
        self.original_train_dataset = ImageFolderWithPaths(
            os.path.join(self.location, self.name(), 'train'),
            transform=self.preprocess,
            )
        
        # Get training split indices (where val_indices is False)
        val_indices = load_persistent_indices(self.location)
        train_indices = np.where(~val_indices)[0]
        self.train_sampler = SubsetRandomSampler(train_indices)
        assert self.train_sampler is not None, "Train sampler is None."

        self.train_loader = torch.utils.data.DataLoader(
            self.original_train_dataset,
            sampler=self.train_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def populate_val(self):
        """
        This is 10% subset of the original train set, we use as val split.
        Uses persistent indices for consistent train/val split.
        """
        if self.original_train_dataset is None:
            raise ValueError("Original train dataset is not populated.")
        
        # Get validation split indices (where val_indices is True)
        val_indices = load_persistent_indices(self.location)
        val_indices = np.where(val_indices)[0]
        self.val_sampler = SubsetRandomSampler(val_indices)
        assert self.val_sampler is not None, "Val sampler is None."

        self.val_loader = torch.utils.data.DataLoader(
            self.original_train_dataset,
            sampler=self.val_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    def populate_test(self):
        """
         The original validation set is used by us as test split
        """
        self.test_dataset = TinyImageNetValFolder(
            os.path.join(self.location, self.name(), 'val'), 
            transform=self.preprocess
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=None
        )

    def name(self):
        return 'tiny-imagenet-200'



class TinyImageNetValFolder(torch.utils.data.Dataset):
    """
        This is the validation folder from original TinyImageNetTrainDataset dataset. But we will use it as a test split.
    """
    def __init__(self, val_dir, transform=None):
        self.val_dir = val_dir
        self.transform = transform
        
        # Read validation annotations
        annotations_file = os.path.join(val_dir, 'val_annotations.txt')
        self.image_paths = []
        self.labels = []
        
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name = parts[0]
                class_id = parts[1]
                
                self.image_paths.append(os.path.join(val_dir, 'images', img_name))
                self.labels.append(class_id)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        from PIL import Image
        
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'images': image,
            'labels': self.labels[index],
            'image_paths': image_path
        }

