import os
import torch
from torch.utils.data import SubsetRandomSampler
import numpy as np

from .common import ImageFolderWithPaths, SubsetSampler


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
        self.classnames = self.get_tiny_imagenet_train_dir_classnames()

        self.populate_train()
        self.populate_test()
    
    def get_tiny_imagenet_train_dir_classnames(self):
        tiny_imagenet_path = os.path.join(self.location, self.name())
        
        # Read wnids (class IDs)
        with open(os.path.join(tiny_imagenet_path, 'wnids.txt'), 'r') as f:
            wnids = [line.strip() for line in f.readlines()]

        # Read words (class names)  
        with open(os.path.join(tiny_imagenet_path, 'words.txt'), 'r') as f:
            words_dict = {}
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    words_dict[parts[0]] = parts[1]

        # Create mapping for TinyImageNetTrainDataset classes
        classnames = []
        for wnid in wnids:
            if wnid in words_dict:
                classnames.append(words_dict[wnid])
            else:
                classnames.append(wnid)
        
        return classnames
    
    def populate_train(self):
        # The original train dataset will be split into our train and val splits.
        traindir = os.path.join(self.location, self.name(), 'train')
        # TODO: check here.
        self.train_dataset = ImageFolderWithPaths(
            traindir,
            transform=self.preprocess,
            )
        sampler = self.get_train_sampler()
        self.sampler = sampler
        kwargs = {'shuffle' : True} if sampler is None else {}
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            **kwargs,
        )

    def populate_test(self):
        # The original validation set is used by us as test split
        self.test_dataset = self.get_test_dataset()
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=self.get_test_sampler()
        )

    def get_test_dataset(self):
        return TinyImageNetValDataset(
            os.path.join(self.location, self.name(), 'val'), 
            transform=self.preprocess
        )

    def get_train_sampler(self):
        return torch.utils.data.distributed.DistributedSampler(self.train_dataset) if self.distributed else None

    def get_test_sampler(self):
        return None

    def name(self):
        return 'tiny-imagenet-200'



class TinyImageNetValDataset(torch.utils.data.Dataset):
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
        
        # Load wnids to get class index mapping
        wnids_file = os.path.join(os.path.dirname(val_dir), 'wnids.txt')
        with open(wnids_file, 'r') as f:
            wnids = [line.strip() for line in f.readlines()]
        
        wnid_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
        
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
    """
    def get_train_sampler(self):
        idx_file = os.path.join(self.location, 'tiny_imagenet_90_idxs.npy') # TODO:  also check this index file
        assert os.path.exists(idx_file)
        with open(idx_file, 'rb') as f:
            idxs = np.load(f)

        idxs = (1 - idxs).astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])

        return sampler


class TinyImageNetTrain10p(TinyImageNetTrainDataset):
    """
        This is 10% subset of the original train set, we use as val split.
    """
    def get_train_sampler(self):
        idx_file = os.path.join(self.location, 'tiny_imagenet_90_idxs.npy')
        assert os.path.exists(idx_file)
        with open(idx_file, 'rb') as f:
            idxs = np.load(f)

        idxs = idxs.astype('int')
        sampler = SubsetSampler(np.where(idxs)[0])
        return sampler