# config.py
#It holds all the settings and paths we need to run the SEM grain segmentation pipeline.

# All paths are set relative to a root directory, and we create them if they don’t exist—keeping things organized!

import os
import torch

# Set the main folder where everything happens—adjust this to your Google Drive or local path!
ROOT_DIR = 'grain_segmentation'
os.makedirs(ROOT_DIR, exist_ok=True)  # Create it if it’s not there yet

# Decide if we use the GPU 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[config] using {DEVICE} ")

# Directories to store all our data, models, and stats
DATA_DIR = os.path.join(ROOT_DIR, "data")                    # Where raw and generated data live
MODELS_DIR = os.path.join(ROOT_DIR, "models")                # Home for our trained model files
STATS_OUT_DIR = os.path.join(ROOT_DIR, "stats_outputs")      # Where we save numbers and plots

# Real SEM image and its patches
REAL_SEM_DIR = os.path.join(DATA_DIR, "real_sem")            # Folder for the original SEM image
REAL_PATCH_DIR = os.path.join(DATA_DIR, "real_patches")      # Where we store chopped-up SEM pieces

SEM_FILENAME = "Au_3_BSD_50kx_A_woscale.tif"                 # The name of our star SEM image
SEM_PATH = os.path.join(REAL_SEM_DIR, SEM_FILENAME)          # Full path to find it

# Synthetic data folders—our artificial SEM world!
DATASET_RENDER_DIR = os.path.join(DATA_DIR, "synthetic-render")  # Pre-GAN synthetic images
DATASET_LABEL_DIR = os.path.join(DATA_DIR, "labels")             # Labels from Voronoi
DATASET_LABEL_GRAY_DIR = os.path.join(DATA_DIR, "labels_gray")   # Grayscale versions of labels
DATASET_LABEL_VIZ_DIR = os.path.join(DATA_DIR, "labels_viz")     # Visualized labels for fun

DATASET_ROOT_DIR = os.path.join(DATA_DIR, "dataset")             # Main spot for synthetic dataset
DATASET_IMG_DIR = os.path.join(DATASET_ROOT_DIR, "images")       # Synthetic SEM-like images
DATASET_MSK_DIR = os.path.join(DATASET_ROOT_DIR, "masks")        # Corresponding boundary masks

# Data splits for training, validation, and testing
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "images")
TRAIN_MSK_DIR = os.path.join(DATA_DIR, "train", "masks")
VAL_IMG_DIR = os.path.join(DATA_DIR, "val", "images")           # Validation images
VAL_MSK_DIR = os.path.join(DATA_DIR, "val", "masks")            # Validation masks
TEST_IMG_DIR = os.path.join(DATA_DIR, "test", "images")
TEST_MSK_DIR = os.path.join(DATA_DIR, "test", "masks")

# Inference and visualization spots
INFERENCE_OUT_DIR = os.path.join(DATA_DIR, "inference_outputs")  # Where inference results go

# Model checkpoints—our saved progress!
REFINER_CKPT = os.path.join(MODELS_DIR, "refiner_ckpt.pth")      # GAN refiner's checkpoint
UNET_CKPT = os.path.join(MODELS_DIR, "unet_ckpt.pth")            # U-Net checkpoint
ATTN_UNET_CKPT = os.path.join(MODELS_DIR, "attnunet_ckpt.pth")   # Attention U-Net checkpoint

# GAN output directory
GAN_OUT_DIR = os.path.join(DATA_DIR, "gan_outputs")              # GAN results root
GAN_SAMPLES_DIR = os.path.join(GAN_OUT_DIR, "samples")           # GAN-refined sample images

# Create all directories to avoid "oops, where’s my folder?" moments
for d in [
    REAL_SEM_DIR, REAL_PATCH_DIR,
    DATASET_ROOT_DIR, DATASET_IMG_DIR, DATASET_MSK_DIR,
    DATASET_RENDER_DIR, DATASET_LABEL_GRAY_DIR,
    TRAIN_IMG_DIR, TRAIN_MSK_DIR, VAL_IMG_DIR, VAL_MSK_DIR, TEST_IMG_DIR, TEST_MSK_DIR,
    INFERENCE_OUT_DIR, MODELS_DIR, STATS_OUT_DIR, GAN_OUT_DIR, GAN_SAMPLES_DIR
]:
    os.makedirs(d, exist_ok=True)

# Hyperparameters—tweak these to fine-tune our magic!
IMG_SIZE = 192              # Size of our square images (pixels)
SYNTH_COUNT = 100           # Number of synthetic images to generate initially
TRAIN_RATIO, VAL_RATIO = 0.7, 0.15  # How we split data: 70% train, 15% val, 15% test
VOR_SEED_RANGE = (40, 50)   # Range for number of grains in Voronoi image
BOUNDARY_THICKNESS = 2      # Width of boundaries in pixels
BOUNDARY_JITTER_P = 0.2     # Chance to jitter boundaries for realism

PATCH_SIZE = 192            # Size of patches from real SEM
STRIDE = 32                 # Step size when cutting patches
BATCH_GAN = 1               # Batch size for GAN training 
EPOCHS_GAN = 10             # How many rounds to train the GAN
LR_G = 2e-4                 # Learning rate for GAN generator
LR_D = 2e-4                 # Learning rate for GAN discriminator
LAMBDA_E = 10.0              # Weight for edge loss in GAN (adjusted for better balance)
LAMBDA_ID = 5.0             # Weight for identity loss in GAN
SYNTH_TRAIN_SAMPLES = 100   # Synthetic samples for GAN training
FINAL_PAIRS = 500           # Final number of synthetic image-mask pairs
USE_GAN = True

BATCH_SIZE = 8              # Batch size for U-Net training
EPOCHS_UNET = 100           # Training rounds for U-Net
LR_UNET = 1e-4              # Learning rate for U-Net
BASE_CHANNELS = 32          # Starting number of channels in U-Net
POS_WEIGHT = 5.0            # Weight for boundary class 
AUG_FLIP = True             # Flip images for variety
AUG_ROTATE = True           # Rotate images for more angles
AUG_INTENSITY_JITTER = True # Add some brightness tweaks

USE_COSINE_LR = True        # Use a smooth learning rate schedule
FINAL_LR = 1e-6             # Final learning rate value

TILE = 512                  # Size of tiles for inference on big images
OVERLAP = 128               # Overlap between tiles to avoid edges
THR = 0.30                  # Threshold for turning probabilities into boundaries (adjusted)
MIN_BOUNDARY_PIX = 2      # Minimum boundary pixels
MIN_GRAIN_AREA = 10         # Minimum grain area in pixels 
#THR_CANDIDATES = (0.30, 0.35, 0.40, 0.45, 0.50)  # Options to test thresholds

SCALE = 2.801              # Scaling factor for physical units (e.g., nm/pixel)