import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2
import os
from torch.utils.data import Dataset
from PIL import Image
from data.imgaug import GetTransforms
from data.utils import transform
from torchvision import transforms
np.random.seed(0)

class ImageDataset(Dataset):
    def __init__(self, csv_file, cfg, mode='train'):
        """
        Args:
            csv_file (string): Path to the csv file with image paths and labels.
            cfg (dict): Configuration dictionary for transformations and other parameters.
            mode (string): Mode could be 'train', 'dev', or 'test'.
        """
        self.cfg = cfg
        self.mode = mode
        self.data = pd.read_csv(csv_file)  # Read the CSV file
        
        # Select only the relevant columns for multi-label classification
        relevant_columns = ["Cardiomegaly", "Edema", "Pneumonia", "Atelectasis", "Pleural_Effusion"]

        # Convert NaN values to 0
        self.data[relevant_columns] = self.data[relevant_columns].fillna(0)

        self.data = self.data[self.data[relevant_columns].ne(-1).all(axis=1)]

        self.data = self.data[self.data[relevant_columns].sum(axis=1) > 0]
        
        self.image_paths = self.data['Path'].values
        self.labels = self.data[relevant_columns].values  # Extract only relevant columns for labels
        self._num_samples = len(self.image_paths)
        
        # Define transforms
        self.transforms = transforms.Compose([
            transforms.Resize((cfg['height'], cfg['width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Example normalization
        ])

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        assert os.path.exists(img_path), f"Image path {img_path} does not exist."
        
        # Read image as grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale (1 channel)
        
        # Convert to PIL Image
        image = Image.fromarray(image)
        # Apply transformations (resize, normalize, etc.)
        image = self.transforms(image)
        # Ensure shape is [C, H, W] for PyTorch
        if len(image.shape) == 2:  
            image = image.unsqueeze(0)  # Add channel dimension: [H, W] -> [1, H, W]

        # Get labels for the image
        labels = self.labels[idx]
        labels = np.array(labels).astype(np.float32)

        if self.mode == 'test':
            return image, img_path  # If in test mode, return image and path
        else:
            return image, labels