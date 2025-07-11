import cv2
import os
import torch
import numpy as np
from src.model import TransformerNet
from src.utils import tensor_to_image, load_image
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

def stylize_frame(frame, model, device, image_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    tensor = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
    image = tensor_to_image(output)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def stylize_video(video_path, model_path, output_path, image_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = TransformerNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Read video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (image_size, image_size))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(frame_count), desc="Stylizing video"):
        ret, frame = cap.read()
        if not ret:
            break
        styled_frame = stylize_frame(frame, model, device, image_size)
        out.write(styled_frame)

    cap.release()
    out.release()
    print(f"Stylized video saved to: {output_path}")
