import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class BrainTumorSegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        # it is already single channel greyscale image but if we need
        # we can with .convert("L")
        image = np.array(Image.open(image_path)) 
        mask = np.array(Image.open(mask_path))

        # we have only 2 classes with segmoid for last activation
        mask[mask == 255.0] = 1.0 

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask




