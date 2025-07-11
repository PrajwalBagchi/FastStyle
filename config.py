import os

class Config:
    # Data paths
    BASE_DIR = "/content/fast-style-transfer"
    #TRAIN_DATA_DIR = os.path.join("Data", "Train")
    #VAL_DATA_DIR = os.path.join("Data", "Val")
    #STYLE_IMAGE_PATH = os.path.join("Data", "Style", "starry_night.jpg")
    #OUTPUT_DIR = os.path.join("Data", "Output")
    TRAIN_DATA_DIR = os.path.join(BASE_DIR, "Data", "Train")
    VAL_DATA_DIR = os.path.join(BASE_DIR, "Data", "Val")
    STYLE_IMAGE_PATH = os.path.join(BASE_DIR, "Data", "Style", "starry_night.jpg")
    OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "Output")


    # Training parameters
    EPOCHS = 10
    BATCH_SIZE = 4
    IMAGE_SIZE = 256
    LEARNING_RATE = 1e-3
    STYLE_WEIGHT = 750
    CONTENT_WEIGHT = 1e5
    TV_WEIGHT = 1e-6



    # Model saving
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "Models", "Checkpoints")
    FINAL_MODEL_PATH = os.path.join(BASE_DIR, "Models", "Final", "fast_style_model.pth")
