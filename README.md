# 🎨 FastStyle

FastStyle+ is a real-time neural style transfer model re-implemented in PyTorch, based on the ECCV 2016 paper *"Perceptual Losses for Real-Time Style Transfer and Super-Resolution"*. It allows users to apply artistic styles (e.g., Van Gogh's *Starry Night*) to both images and videos in real time using GPU-accelerated inference.

---

## 🧠 Key Features

- 🚀 **Real-time style transfer** using a lightweight TransformerNet architecture
- 🖼️ **Supports both image and video stylization**
- 🎯 Trained on the **Flickr8k dataset** with a perceptual loss from a pre-trained **VGG19** network
- 🔍 **Style, Content, and Total Variation loss** customization
- 📦 Modular and reproducible implementation using PyTorch
- ⚙️ Supports **checkpointing and resuming training**
- 🌐 Streamlit UI for demoing and showcasing results

---

## 📁 Directory Structure

fast-style-transfer/
├── Data/
│ ├── Train/ # Training images (from Flickr8k)
│ ├── Style/ # Style image (e.g., starry_night.jpg)
│ └── Output/ # Stylized output samples
├── Models/
│ ├── Final/ # Final saved .pth model
│ └── Checkpoints/ # Epoch-wise model checkpoints
├── src/
│ ├── config.py # Configuration file
│ ├── loss.py # Style/content/TV loss
│ ├── model.py # TransformerNet definition
│ ├── trainer.py # Training loop
│ ├── utils.py # Helper functions
│ └── vgg.py # VGG19 perceptual feature extractor
├── inference.py # Image stylization script
├── video_stylizer.py # Video stylization script
├── README.md # You're here!
└── requirements.txt # All dependencies

yaml
Copy
Edit

---

## 🧪 How to Use

### 🔧 Setup
```bash
# Clone the repo
git clone https://github.com/yourusername/faststyle-plus.git
cd faststyle-plus

pip install -r requirements.txt

```
🏋️ Train the Model

```bash
# Place style image in /Data/Style/
# Place training images in /Data/Train/images/

python train.py
```

🖼️ Stylize an Image

```python
python inference.py
```

🎞️ Stylize a Video

```bash
python video_stylizer.py --video your_video.mp4 --output styled_video.mp4
```
🔬 Model Details
Component	Configuration
Base Model	TransformerNet
Loss Backbone	VGG19 (ImageNet pre-trained)
Style Weight	Tuned (e.g., 750)
Content Weight	1e5
Training Data	Flickr8k (8,000 real-world images)
Output Size	256×256
Inference Speed	Real-time with T4 GPU

📷 Sample Outputs

Content Image	Style Image	Stylized Output
		

🧠 Citation
If you're referencing the underlying paper:

```mathematica
@inproceedings{johnson2016perceptual,
  title={Perceptual Losses for Real-Time Style Transfer and Super-Resolution},
  author={Johnson, Justin and Alahi, Alexandre and Fei-Fei, Li},
  booktitle={ECCV},
  year={2016}
}
```

🛡️ License
This project is licensed under the MIT License.

✨ Credits
Based on Justin Johnson's paper

Trained using Flickr8k dataset

Style examples from Van Gogh’s Starry Night

PyTorch + Streamlit implementation by [Your Name]

yaml
Copy
Edit
