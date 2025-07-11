import torch
from torchvision import transforms
from PIL import Image
from src.model import TransformerNet
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image

def tensor_to_image(tensor):
    tensor = tensor.clone().detach().cpu().squeeze(0)
    tensor = tensor.permute(1, 2, 0)  # CHW -> HWC
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
    tensor = tensor.clamp(0, 1)
    tensor = (tensor.numpy() * 255).astype("uint8")
    return Image.fromarray(tensor)



# Paths
image_path = "A:/Re-implementing real time transfer and super resolution/Data/Test/content.jpeg"
model_path = "A:/Re-implementing real time transfer and super resolution/Models/Colab_Final/fast_style_model.pth"
output_path = "A:/Re-implementing real time transfer and super resolution/Data/Output/stylized_image.png"

# Load image (MUST match training size, which was 256 here)
content_image = load_image(image_path, image_size=256).to(device)

# Load model safely
model = TransformerNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Stylize
with torch.no_grad():
    output = model(content_image)

# Convert to image and save
output_image = tensor_to_image(output)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Detect format from extension
ext = os.path.splitext(output_path)[-1].lower()
if ext == "":
    output_path += ".png"

output_image.save(output_path)

# Debug stats
print(f"Output Tensor Min: {output.min().item()}, Max: {output.max().item()}, Mean: {output.mean().item()}")
print(f"Stylized image saved to: {output_path}")
