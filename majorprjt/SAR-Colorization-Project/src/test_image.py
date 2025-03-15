import torch
from PIL import Image
import torchvision.transforms as transforms
from unet_model import UNet

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("../models/sar_colorization_model.pth"))
model.eval()

# Image Processing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def colorize_sar(image_path):
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image).squeeze(0).cpu()

    output_image = transforms.ToPILImage()(output)
    output_image.show()

# Test
colorize_sar("../datasets/sar/sample_sar_image.jpg")  # Change this path
