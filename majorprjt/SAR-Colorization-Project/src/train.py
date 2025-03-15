import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_loader import SARColorizationDataset
from unet_model import UNet
import torchvision.transforms as transforms

# Paths
sar_dir = "./datasets/sar"
optical_dir = "./datasets/optical"
checkpoint_path = "./models/sar_colorization_checkpoint.pth"

# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load Dataset
dataset = SARColorizationDataset(sar_dir, optical_dir, transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load Checkpoint if Exists
start_epoch = 0
start_step = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    start_step = checkpoint['step']
    print(f"Resuming training from Epoch {start_epoch}, Step {start_step}")

# Training Loop
epochs = 20
for epoch in range(start_epoch, epochs):  
    for i, (sar, optical) in enumerate(dataloader, start=start_step):
        sar, optical = sar.to(device), optical.to(device)

        # Ensure tensors are not None
        if sar is None or optical is None:
            print(f"Warning: Found None in data at Step {i+1}")
            continue  

        # Fix Shape Mismatch
        if sar.shape[1] == 1:  # If SAR is grayscale
            sar = sar.repeat(1, 3, 1, 1)  # Convert SAR to 3-channel

        if optical.shape[1] == 3:  # Convert Optical RGB to grayscale
            optical = optical.mean(dim=1, keepdim=True)

        print(f"Processed SAR shape: {sar.shape}, Optical shape: {optical.shape}")

        optimizer.zero_grad()
        output = model(sar)

        # Debugging: Check if model output is None
        if output is None or output.shape != optical.shape:
            print(f"Warning: Model returned None or incorrect shape at Step {i+1}")
            continue

        loss = criterion(output, optical)
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'step': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, checkpoint_path)
        print(f"Checkpoint saved at Epoch {epoch+1}, Step {i+1}")

    start_step = 0  # Reset step count for next epoch

# Save Final Model
torch.save(model.state_dict(), "./models/sar_colorization_model.pth")
print("Training complete!")

