import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class SARColorizationDataset(Dataset):
    def __init__(self, sar_dir, optical_dir, transform=None):
        if not os.path.exists(sar_dir) or not os.path.exists(optical_dir):
            raise FileNotFoundError(f"SAR or Optical dataset folder not found!\nSAR: {sar_dir}\nOptical: {optical_dir}")

        valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')

        self.sar_images = sorted([f for f in os.listdir(sar_dir) if f.endswith(valid_extensions)])
        self.optical_images = sorted([f for f in os.listdir(optical_dir) if f.endswith(valid_extensions)])

        if not self.sar_images or not self.optical_images:
            raise FileNotFoundError("One or both dataset folders are empty!")

        if len(self.sar_images) != len(self.optical_images):
            raise ValueError(f"Mismatch: {len(self.sar_images)} SAR images vs {len(self.optical_images)} Optical images!")

        self.sar_dir = sar_dir
        self.optical_dir = optical_dir
        self.transform = transform

    def __len__(self):
        return len(self.sar_images)

    def __getitem__(self, idx):
        sar_image_path = os.path.join(self.sar_dir, self.sar_images[idx])
        optical_image_path = os.path.join(self.optical_dir, self.optical_images[idx])

        try:
            sar_image = Image.open(sar_image_path).convert('L')
            optical_image = Image.open(optical_image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading images:\nSAR: {sar_image_path}\nOptical: {optical_image_path}\nError: {e}")

        if self.transform:
            sar_image = self.transform(sar_image)
            optical_image = self.transform(optical_image)

        return sar_image, optical_image
