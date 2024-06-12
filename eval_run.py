import os
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from tripleNet.tripleNet3 import *
from libs.dataset_fam import Dataset

if __name__ == "__main__":

    os.makedirs('./results_triple', exist_ok=True)

    # print('cuda: ', torch.cuda.is_available())
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    val_dataset     = Dataset('./dataset/test/families_pt', shuffle_pairs=False, augment=False)
    val_dataloader   = DataLoader(val_dataset, batch_size=1)

    # checkpoint = torch.load('./out_triple/best.pth')

    # model = TripleSiameseNetwork(layer='conv').to(device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # criterion = ContrastiveLoss()

    # losses = []
    # correct = 0
    # total = 0

    inv_transform = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                         std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                         std = [ 1., 1., 1. ]),
                                   ])

    # img = inv_transform(torch.load('./dataset/test/families_pt/0/M_01.pt')).cpu().numpy()[0]
    # print(img)
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.axis("off")
    # plt.savefig('./results_triple/test.png')

    for i, ((img1, img2, img3), (label1, label2), (class1, class2, class3)) in enumerate(val_dataloader):
        print("Epoch [{} / {}]".format(i, len(val_dataloader)))


        fig = plt.figure("label1={}\tlabel2={}".format(label1, label2), figsize=(4, 2))
        plt.suptitle("cls1={} cls2={} cls3={}".format(class1, class2, class3))

        # Apply inverse transform (denormalization) on the images to retrieve original images.
        img1 = inv_transform(img1).cpu().numpy()[0]
        img2 = inv_transform(img2).cpu().numpy()[0]
        img3 = inv_transform(img3).cpu().numpy()[0]

        # show first image
        ax = fig.add_subplot(1, 3, 1)
        plt.imshow(img1[0], cmap=plt.cm.gray)
        plt.axis("off")

        # show the second image
        ax = fig.add_subplot(1, 3, 2)
        plt.imshow(img2[0], cmap=plt.cm.gray)
        plt.axis("off")

        # show third image
        ax = fig.add_subplot(1, 3, 3)
        plt.imshow(img3[0], cmap=plt.cm.gray)
        plt.axis("off")

        # show the plot
        plt.savefig(os.path.join('./results_triple', '{}.png').format(i))
