import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import subprocess  # For GPU temperature monitoring
import time

from src.model import TransformerNet
from src.vgg import VGGFeatureExtractor
from src.loss import StyleTransferLoss
from src import utils
from src.config import Config
from src.utils import load_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Style image loading
#style_image_path = "A:/Re-implementing real time transfer and super resolution/Data/Style/starry_night.jpg"
style_image_path = "fast-style-transfer/Data/Style/starry_night.jpg"
image_size = 256
style_image_tensor = load_image(style_image_path, image_size).to(device)

def train(device):
    # Data transforms

    transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.CenterCrop(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])

    # Dataset and loader
    train_dataset = datasets.ImageFolder(Config.TRAIN_DATA_DIR, transform)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # Model
    model = TransformerNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # VGG for loss
    vgg = VGGFeatureExtractor(device).to(device)
    with torch.no_grad():
        style_features = vgg(style_image_tensor)
    style_grams = {layer: utils.gram_matrix(style_features[layer]) for layer in style_features}

    # Loss module
    loss_fn = StyleTransferLoss(
        vgg=vgg,
        style_grams=style_grams,
        content_weight=Config.CONTENT_WEIGHT,
        style_weight=Config.STYLE_WEIGHT,
        tv_weight=Config.TV_WEIGHT,
        device=device  
    )

    # Training loop
    model.train()
    for epoch in range(Config.EPOCHS):
        for batch_id, (x, _) in enumerate(train_loader):
            x = x.to(device)
            y_hat = model(x)
            y_hat = y_hat.clamp(0, 1)
            loss, c_loss, s_loss, tv_loss = loss_fn(x, y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_id + 1) % 50 == 0:
                print(f"[{epoch+1}/{Config.EPOCHS}][{batch_id+1}/{len(train_loader)}] "
                      f"Total: {loss.item():.2f} | Content: {c_loss.item():.2f} "
                      f"| Style: {s_loss.item():.2f} | TV: {tv_loss.item():.2f}")
            if (batch_id + 1) % 50 == 0:
                 print(f"[{epoch+1}/{Config.EPOCHS}][{batch_id+1}/{len(train_loader)}] "
                       f"Total: {loss.item():.2f} | Content: {c_loss.item():.2f} "
                       f"| Style: {s_loss.item():.2f} ({s_loss.item() / c_loss.item():.2f}x) | TV: {tv_loss.item():.2f}")


            # 🔥 GPU Temperature Logging every 100 batches
            if (batch_id + 1) % 100 == 0:
                try:
                    temp = subprocess.check_output([
                        'nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'
                    ])
                    print(f"GPU Temp: {temp.decode().strip()} C")
                except Exception as e:
                    print("Could not read GPU temperature:", e)

        # Save checkpoint
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved at: {ckpt_path}")

    if (batch_id + 1) % 25 == 0: 
            time.sleep(2)
            
    # Final model save
    os.makedirs(os.path.dirname(Config.FINAL_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), Config.FINAL_MODEL_PATH)
    print(f"Model saved to: {Config.FINAL_MODEL_PATH}")

