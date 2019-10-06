import os
from os.path import join
from PIL import Image
import json

import torch
import torchvision as tv
from torch.utils.data import Dataset, DataLoader


def get_class_dict(base_dir):
    '''
    Returns:
        class_dict_idx2ctgr:  ex) {0: 'tench', ...} where ctgr is category
        class_dict_name2idx: ex) {'n01440764': 0, ...}
    '''
    with open(join(base_dir, 'imagenet_class_index.json'),'r') as json_file:
        class_dict = json.load(json_file)

    class_dict_idx2ctgr = {int(k):v[1] for k,v in class_dict.items()}
    class_dict_name2idx = {v[0]:int(k) for k,v in class_dict.items()}

    return class_dict_idx2ctgr, class_dict_name2idx


class FolderDataset(Dataset):
    def __init__(self, t_input):
        self.t_input = t_input
        self.base_dir = join('/','mnt','nas','shared','data','imagenet')
        self.train_image_dir = join(self.base_dir,'train','image')

        _, self.class_name2idx = get_class_dict(self.base_dir)

        self.load_files()
    
    def load_files(self):
        self.train_files = os.listdir(self.train_image_dir)
        self.train_files = [file for file in self.train_files 
                            if file.endswith('.JPEG')]
        
    def __getitem__(self, index):
        file = self.train_files[index]
        path = join(self.train_image_dir, file)
        
        input = Image.open(path).convert('RGB')
        input = self.t_input(input)
        
        target = file.split('_')[0] # name, ex) n01440764
        target = self.class_name2idx[target] # idx
        
        return input, target

    def __len__(self):
        return len(self.train_files)


def imagenet_loader(bs=32):
    t_input = tv.transforms.Compose([
        tv.transforms.RandomResizedCrop(224),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])

    imagenet_dataset = FolderDataset(t_input)

    loader = DataLoader(imagenet_dataset, batch_size=bs,
                    shuffle=True, num_workers=16)

    return loader
