import torch
import os
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from dataset_loader import SARColorizationDataset
from unet_model import UNet
import torchvision.transforms as transforms

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model_path = os.path.abspath("../models/sar_colorization_model.pth")
print("Looking for model file at:", model_path)

model.load_state_dict(torch.load(model_path))
model.eval()

# Load Dataset
sar_dir = "../datasets/sar"
optical_dir = "../datasets/optical"
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
dataset = SARColorizationDataset(sar_dir, optical_dir, transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

# Evaluation
def evaluate_model(model, dataloader, device):
    model.eval()
    total_psnr, total_ssim, count = 0, 0, 0

    with torch.no_grad():
        for sar, optical in dataloader:
            sar, optical = sar.to(device), optical.to(device)
            output = model(sar)

            for i in range(sar.size(0)):
                pred_img = output[i].cpu().numpy().transpose(1, 2, 0)
                gt_img = optical[i].cpu().numpy().transpose(1, 2, 0)

                psnr_val = psnr(gt_img, pred_img, data_range=1.0)
                ssim_val = ssim(gt_img, pred_img, multichannel=True, data_range=1.0)

                total_psnr += psnr_val
                total_ssim += ssim_val
                count += 1

    print(f"Average PSNR: {total_psnr / count:.2f}, Average SSIM: {total_ssim / count:.4f}")

# Run evaluation
evaluate_model(model, dataloader, device)
