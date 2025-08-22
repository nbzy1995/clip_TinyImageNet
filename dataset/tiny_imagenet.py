import os
import torch
from torch.utils.data import SubsetRandomSampler
import numpy as np

from .common import ImageFolderWithPaths, SubsetSampler


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


class TinyImageNetTrainDataset:
    """
        This is the training folder from original TinyImageNet dataset. But we will split into training and val split, using TinyImageNetTrain90p, TinyImageNetTrain10p
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
        
        # Load class names from wnids.txt. 
        # self.classnames = self.get_tiny_imagenet_train_dir_classnames()

        self.populate_train()
        # self.populate_test()
    
    # def get_tiny_imagenet_train_dir_classnames(self):
    #     tiny_imagenet_path = os.path.join(self.location, self.name())
        
        # Get class directories in alphabetical order (same as ImageFolder's default)
        # train_dir = os.path.join(tiny_imagenet_path, 'train')
        # class_dirs = sorted([d for d in os.listdir(train_dir) 
                        #    if os.path.isdir(os.path.join(train_dir, d)) and d.startswith('n')])
        
        # Read words (the mapping from wnid to names)  
        # with open(os.path.join(tiny_imagenet_path, 'words.txt'), 'r') as f:
        #     words_dict = {}
        #     for line in f:
        #         parts = line.strip().split('\t', 1)
        #         if len(parts) == 2:
        #             words_dict[parts[0]] = parts[1]

        # # Create classnames in alphabetical order (consistent with ImageFolder)
        # classnames = []
        # for wnid in class_dirs:
        #     if wnid in words_dict:
        #         classnames.append(words_dict[wnid])
        #     else:
        #         raise ValueError(f"Class {wnid} not found in words.txt")
        
        # Store the wnid to alphabetical index mapping
        # self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(class_dirs)}
        
        # return classnames
    
    def populate_train(self):
        # The original train dataset will be split into our train and val splits by its sampler
        traindir = os.path.join(self.location, self.name(), 'train')
        # ImageFolder loads classes in alphabetical order (deterministic across platforms)
        self.original_train_dataset = ImageFolderWithPaths(
            traindir,
            transform=self.preprocess,
            )
        # Here is how train and val split are handled
        sampler = self.get_train_sampler()
        self.sampler = sampler
        assert sampler is not None, "Train sampler must be implemented in Train90p, Train10p subclasses."

        self.ds_loader = torch.utils.data.DataLoader(
            self.original_train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    # def populate_test(self):
    #     # The original validation set is used by us as test split
    #     self.test_dataset = self.get_test_dataset()
    #     self.test_loader = torch.utils.data.DataLoader(
    #         self.test_dataset,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         pin_memory=True,
    #         sampler=self.get_test_sampler()
    #     )

    # def get_test_dataset(self):
    #     return TinyImageNetValDataset(
    #         os.path.join(self.location, self.name(), 'val'), 
    #         transform=self.preprocess,
    #         wnid_to_idx=self.wnid_to_idx
    #     )

    def get_train_sampler(self):
        raise NotImplementedError(
            "Train sampler must be implemented in Train90p, Train10p."
        )

    # def get_test_sampler(self):
    #     return None

    def name(self):
        return 'tiny-imagenet-200'



class TinyImageNetValDataset(torch.utils.data.Dataset):
    """
        This is the validation folder from original TinyImageNetTrainDataset dataset. But we will use it as a test split.
    """
    def __init__(self, val_dir, transform=None, wnid_to_idx=None):
        self.val_dir = val_dir
        self.transform = transform
        
        # Read validation annotations
        annotations_file = os.path.join(val_dir, 'val_annotations.txt')
        self.image_paths = []
        self.labels = []
        
        # Create wnid to alphabetical index mapping (consistent with train dataset)
        if wnid_to_idx is None:
            # Generate alphabetical ordering from train directory
            train_dir = os.path.join(os.path.dirname(val_dir), 'train')
            class_dirs = sorted([d for d in os.listdir(train_dir) 
                               if os.path.isdir(os.path.join(train_dir, d)) and d.startswith('n')])
            wnid_to_idx = {wnid: idx for idx, wnid in enumerate(class_dirs)}
        
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name = parts[0]
                class_id = parts[1]
                
                self.image_paths.append(os.path.join(val_dir, 'images', img_name))
                self.labels.append(wnid_to_idx[class_id])
    
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


class TinyImageNetTrain90p(TinyImageNetTrainDataset):
    """
        This is 90% subset of the original train set, we use as train split.
        Uses persistent indices for consistent train/val split.
    """
    def get_train_sampler(self):
        # Load persistent train/val split indices
        val_indices = load_persistent_indices(self.location)
        
        # Get training indices (where val_indices is False)
        train_indices = np.where(~val_indices)[0]
        sampler = SubsetRandomSampler(train_indices)
        
        return sampler


class TinyImageNetTrain10p(TinyImageNetTrainDataset):
    """
        This is 10% subset of the original train set, we use as val split.
        Uses persistent indices for consistent train/val split.
    """
    def get_train_sampler(self):
        # Load persistent train/val split indices
        val_indices = load_persistent_indices(self.location)
        
        # Get validation indices (where val_indices is True)
        val_indices_list = np.where(val_indices)[0]
        sampler = SubsetSampler(val_indices_list)
        
        return sampler