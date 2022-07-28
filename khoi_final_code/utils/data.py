import torch
import torchvision
from torchvision.io import read_image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import os
import random
import numpy as np

class ETZ(Dataset):
    def __init__(self, path_to_imgs, path_to_masks, path_to_labels, transform=False, seed = 42):
        """
        path_to_imgs: Path to where you save the images.
        path_to_mask: Path to where you save the masks.
        path_to_label: Path to where you save the target files, which contain bounding boxes information.
        transform: Whether you want to apply augmentation or not. 
        """
        self.img_dir = path_to_imgs
        self.mask_dir = path_to_masks
        self.label_dir = path_to_labels
        self.transform = transform
        self.seed = seed

    def __len__(self):
        return len(os.list_dir(self.img_dir))

    def __getitem__(self, idx):
        """
        idx: Key to index a file in the format 'CTP[0-9]{2,3}_00[1|2]_[0-9]{4}'.
        """
        img_path = os.path.join(self.img_dir, f"{idx}.png")
        mask_path = os.path.join(self.mask_dir, f"{idx}.png")
        #label_path = os.path.join(self.img_dir, f"{idx}.txt")
        image = read_image(img_path, mode = torchvision.io.ImageReadMode.GRAY)
        mask = read_image(mask_path, mode = torchvision.io.ImageReadMode.GRAY)
        
        if self.transform:
            if self.seed:
                torch.manual_seed(self.seed)
                random.seed(self.seed)
            else:
                seed = np.random.randint(2147483647)
                torch.manual_seed(seed)
                random.seed(seed)

            # Random translation
            width_translate = random.uniform(-200, 200)
            height_translate = random.uniform(-200, 200)
            image_aug = TF.affine(image,angle = 0, translate = [width_translate,height_translate], scale = 1, shear = 0) 
            mask_aug = TF.affine(mask,angle = 0, translate = [width_translate,height_translate], scale = 1, shear = 0) 
            
            # Left to right flipping. Also 50% chance. 
            # First we need to squeeze because fliplr expect a 2D array. We then unsqueeze to get a (1, width, height) array again.
            if random.random() > 0.5:
                image_aug = torch.fliplr(image_aug.squeeze()).unsqueeze(0)
                mask_aug = torch.fliplr(mask_aug.squeeze()).unsqueeze(0)
            # Up and down flipping.
            else:
                image_aug = torch.flipud(image_aug.squeeze()).unsqueeze(0)
                mask_aug = torch.flipud(mask_aug.squeeze()).unsqueeze(0)

            # Rotating a random degree between -30 and 30. There is a 50% chance this transformation is applied. 
            if random.random() > 0.5:
                rand_deg = random.uniform(-30, 30)
                image_aug, mask_aug = TF.rotate(image_aug, rand_deg), TF.rotate(mask_aug, rand_deg)  

            return image_aug, mask_aug
        return image, mask