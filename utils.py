import os
import torch
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms

def save_sample_output(content_img, stylized_img, epoch, step, output_dir):
    """Save side-by-side comparison of content and stylized output."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Clamp and scale to [0, 1]
    stylized_img = stylized_img.clone().detach().cpu()
    content_img = content_img.clone().detach().cpu()

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# De-normalize
    stylized_img = stylized_img * imagenet_std + imagenet_mean
    content_img = content_img * imagenet_std + imagenet_mean

    stylized_img = stylized_img.clamp(0, 1)
    content_img = content_img.clamp(0, 1)

    # Take only the first image in the batch
    side_by_side = torch.cat((content_img[0], stylized_img[0]), dim=2)  # concat width-wise
    file_path = os.path.join(output_dir, f"epoch{epoch}_step{step}.png")
    save_image(side_by_side, file_path)

def save_model(model, path):
    """Save the model weights to the given path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


def tensor_to_image(tensor):
    """Convert a tensor to a PIL Image."""
    tensor = tensor.clone().detach().cpu()
    tensor = tensor.squeeze(0)
    tensor = tensor / 255.0
    return transforms.ToPILImage()(tensor.clamp(0, 1))

def gram_matrix(tensor):
    """Normalized Gram matrix to prevent exploding style loss."""
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    features = features - features.mean(dim=2, keepdim=True)  # Center features
    gram = features.bmm(features.transpose(1, 2)) / (b * c * h * w)  # Normalize
    return gram

