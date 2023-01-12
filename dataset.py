################################################################################
#                                  Dataset                                     #
################################################################################

import os
import logging
import random
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image

import matplotlib.pyplot as plt

# to convert masks to uint8
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
convert_tensor = transforms.ToTensor()

logging.basicConfig(level=logging.INFO, format='[DATASET] %(levelname)s: %(message)s')

class SaltDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.paths = os.listdir(img_dir)

        logging.info(f"Loading images from {img_dir} into memory")
        self.images = [ \
            convert_tensor( \
                Image.open(os.path.join(self.img_dir, fn))) for fn in self.paths]
        
        logging.info(f"Loading masks from {mask_dir} into memory")
        self.masks = []
        for fn in self.paths:
            mask_path = os.path.join(self.mask_dir, fn)
        
            mask_pil = Image.open(mask_path)
            mask_tensor = convert_tensor(mask_pil)
            mask_tensor[mask_tensor==65535] = 1
            mask = mask_tensor.to(torch.uint8)   
            self.masks.append(mask)

        # ---- save for viewing ----
        #for i in range(100):
        #    save_image(self.images[i], f"./del/{i}-i.png")
        #    save_image(self.masks[i] / 255.0, f"./del/{i}-m.png")

        logging.info(f"Done loading images and masks.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]

        if self.transform != None:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask

################################################################################
# NOTE this part is just for VIEWING the images and masks.. maybe these images #
# will come in handy later                                                     #
################################################################################
if __name__ == "__main__":
    sds=SaltDataset(
        "/content/drive/MyDrive/deep_learning/salt_identification_challenge/train/images",
        "/content/drive/MyDrive/deep_learning/salt_identification_challenge/train/masks"
    )

    print(f"The SaltDataset contains {len(sds)} items")
    
    indices=[7, 8, 9, 10, 11, 12, 13, 14]
    cols, rows = len(indices), 2
    figure = plt.figure(figsize=(cols, rows))
    for i in range(0, len(indices)):
        img, label = sds[indices[i]]
        
        figure.add_subplot(rows, cols, i + 1)
        plt.axis("off")
        img = torch.transpose(img, 0, 2)
        plt.imshow(img.squeeze(), cmap="gray")

        figure.add_subplot(rows, cols, len(indices) + i + 1)
        plt.axis("off")
        label[label==0.0] = 1.0
        label = torch.transpose(label, 0, 2)
        plt.imshow(label.squeeze(), cmap="gray")    

    plt.show()
