import os
import glob
import time

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, path, shuffle_pairs=True, augment=False):
        '''
        Create an iterable dataset from a directory containing sub-directories of 
        entities with their images contained inside each sub-directory.

            Parameters:
                    path (str):                 Path to directory containing the dataset.
                    shuffle_pairs (boolean):    Pass True when training, False otherwise. When set to false, the image pair generation will be deterministic
                    augment (boolean):          When True, images will be augmented using a standard set of transformations.

            where b = batch size

            Returns:
                    output (torch.Tensor): shape=[b, 1], Similarity of each pair of images
        '''
        self.path = path

        self.feed_shape = [3, 224, 224]
        self.shuffle_pairs = shuffle_pairs

        self.augment = augment

        if self.augment:
            # If images are to be augmented, add extra operations for it (first two).
            self.transform = transforms.Compose([
                transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize(self.feed_shape[1:])
            ])
        else:
            # If no augmentation is needed then apply only the normalization and resizing operations.
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize(self.feed_shape[1:])
            ])

        # self.image_paths = [*glob.glob(os.path.join(self.path, "*/*.JPG")), *glob.glob(os.path.join(self.path, "*/*.jpg"))]
        self.image_paths = glob.glob(os.path.join(self.path, "*/*.pt"))

        self.image_classes = []
        self.class_indices = {}
        self.rel_indices = {}

        for image_path in self.image_paths:
            image_class = image_path.split(os.path.sep)[-2]
            image_rel = image_path.split(os.path.sep)[-1].split('_')[0]

            self.image_classes.append(image_class)

            if image_class not in self.class_indices:
                self.class_indices[image_class] = []
            self.class_indices[image_class].append(self.image_paths.index(image_path))

            if image_rel not in self.rel_indices:
                self.rel_indices[image_rel] = []
            self.rel_indices[image_rel].append(self.image_paths.index(image_path))

        self.create_pairs()

    def create_pairs(self):
        '''
        Creates two lists of indices that will form the pairs, to be fed for training or evaluation.
        '''

        self.indices_C = np.copy(self.rel_indices['C'])

        if self.shuffle_pairs:
            np.random.seed(int(time.time()))
            np.random.shuffle(self.indices_C)
        else:
            # If shuffling is set to off, set the random seed to 1, to make it deterministic.
            np.random.seed(1)

        select_pos_pair_M = np.random.rand(len(self.rel_indices['C'])) < 0.5
        select_pos_pair_F = np.random.rand(len(self.rel_indices['C'])) < 0.5

        self.indices_M = []
        self.indices_F = []

        for i, pos_M, pos_F in zip(self.indices_C, select_pos_pair_M, select_pos_pair_F):
            class_C = self.image_classes[i]
            if pos_M:
                class_M = class_C
            else:
                class_M = np.random.choice(list(set(self.class_indices.keys()) - {class_C}))
            if pos_F:
                class_F = class_C
            else:
                class_F = np.random.choice(list(set(self.class_indices.keys()) - {class_C}))

            idx_M = np.random.choice(self.class_indices[class_M])

            while idx_M not in self.rel_indices['M']:
                idx_M = np.random.choice(self.class_indices[class_M])

            idx_F = np.random.choice(self.class_indices[class_F])

            while idx_F not in self.rel_indices['F']:
                idx_F = np.random.choice(self.class_indices[class_F])

            self.indices_M.append(idx_M)
            self.indices_F.append(idx_F)

        self.indices_M = np.array(self.indices_M)
        self.indices_F = np.array(self.indices_F)

    def __iter__(self):
        self.create_pairs()

        for idx1, idx2, idx3 in zip(self.indices_C, self.indices_M, self.indices_F):

            image_path1 = self.image_paths[idx1]
            image_path2 = self.image_paths[idx2]
            image_path3 = self.image_paths[idx3]

            class1 = self.image_classes[idx1]
            class2 = self.image_classes[idx2]
            class3 = self.image_classes[idx3]

            image1 = torch.load(image_path1)
            image2 = torch.load(image_path2)
            image3 = torch.load(image_path3)

            # image1 = Image.open(image_path1).convert("RGB")
            # image2 = Image.open(image_path2).convert("RGB")
            # image3 = Image.open(image_path3).convert("RGB")

            # if self.transform:
            #     image1 = self.transform(image1).float()
            #     image2 = self.transform(image2).float()
            #     image3 = self.transform(image3).float()

            yield (image1, image2, image3), (torch.FloatTensor([class1==class2]), torch.FloatTensor([class1==class3])), (class1, class2, class3)
        
    def __len__(self):
        return len(self.rel_indices['C'])
