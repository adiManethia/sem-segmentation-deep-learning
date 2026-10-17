# data_loader.py
# It loads and prepares the synthetic images and masks
# for training, validation, and testing. 
# our dataset into neat batches for the model to learn from.

import os
import glob
import numpy as np
import cv2
import torch 
from torch.utils.data import Dataset, DataLoader, random_split
import shutil

from config import DATASET_IMG_DIR, DATASET_MSK_DIR, IMG_SIZE, TRAIN_RATIO, VAL_RATIO, BATCH_SIZE
from config import TRAIN_IMG_DIR, TRAIN_MSK_DIR, VAL_IMG_DIR, VAL_MSK_DIR, TEST_IMG_DIR, TEST_MSK_DIR

class GrainDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        """Dataset class for loading grain images and their masks.
        
        Args:
            img_dir (str): Directory containing image files.
            mask_dir (str): Directory containing mask files.
        """
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))  # List all images
        self.mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))  # List all masks
        assert len(self.img_files) == len(self.mask_files), "Oops! Mismatch between images and masks"

    def __len__(self):
        return len(self.img_files)  # Total number of image-mask pairs

    def __getitem__(self, idx):
        """Get an image-mask pair at the given index.
        
        Args:
            idx (int): Index of the pair to retrieve.
        
        Returns:
            tuple: (image_tensor, mask_tensor) both as [1, H, W] tensors.
        """
        img = cv2.imread(self.img_files[idx], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask = cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)

def build_loaders():
    """Build data loaders for training, validation, and test sets.
    
    This function splits the dataset according to the configured ratios and creates
    DataLoader instances for efficient batching.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader) DataLoader objects.
    """
    dataset = GrainDataset(DATASET_IMG_DIR, DATASET_MSK_DIR)
    total_size = len(dataset)
    train_size = int(total_size * TRAIN_RATIO)
    val_size = int(total_size * VAL_RATIO)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

def ensure_directories_populated():
    """Ensure train, val, and test directories are populated with data."""
    # Define the splits
    splits = {
        "train": (TRAIN_IMG_DIR, TRAIN_MSK_DIR),
        "val": (VAL_IMG_DIR, VAL_MSK_DIR),
        "test": (TEST_IMG_DIR, TEST_MSK_DIR),
    }

    # Ensure directories exist
    for split, (img_dir, msk_dir) in splits.items():
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)

    # Check if directories are empty and populate them
    dataset_images = sorted(os.listdir(DATASET_IMG_DIR))
    dataset_masks = sorted(os.listdir(DATASET_MSK_DIR))

    if not dataset_images or not dataset_masks:
        raise FileNotFoundError("Dataset image or mask directories are empty. Please generate the dataset first.")

    total_size = len(dataset_images)
    train_size = int(total_size * 0.7)  # 70% for training
    val_size = int(total_size * 0.2)    # 20% for validation
    test_size = total_size - train_size - val_size  # Remaining for testing

    splits_indices = {
        "train": (0, train_size),
        "val": (train_size, train_size + val_size),
        "test": (train_size + val_size, total_size),
    }

    for split, (img_dir, msk_dir) in splits.items():
        start, end = splits_indices[split]
        for img_file, msk_file in zip(dataset_images[start:end], dataset_masks[start:end]):
            shutil.copy(os.path.join(DATASET_IMG_DIR, img_file), os.path.join(img_dir, img_file))
            shutil.copy(os.path.join(DATASET_MSK_DIR, msk_file), os.path.join(msk_dir, msk_file))

    print("Train, validation, and test directories have been populated.")

if __name__ == "__main__":
    ensure_directories_populated()