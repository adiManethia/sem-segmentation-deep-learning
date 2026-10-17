# Use trained model to segment a new real SEM image and visualize results (no saving, just show plots)

# Import necessary libraries and modules
import os
import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt
from config import UNET_CKPT, ATTN_UNET_CKPT, DEVICE, THR, TILE, OVERLAP
from models import get_model
from skimage.measure import label
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set the path to the new SEM image to be processed
# This image will be segmented using the trained models
NEW_SEM_PATH = "Au_3_BSD_50kx_B_woscale.tif"  # Example SEM image path

# Utility functions for image preprocessing and postprocessing

def load_gray01(path):
    # Load an image, convert to grayscale if needed, and normalize pixel values to [0, 1]
    img = np.array(Image.open(path))
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)

def tile_image(img, tile=TILE, overlap=OVERLAP):
    # Divide the image into overlapping tiles for processing
    H, W = img.shape
    step = tile - overlap
    tiles, coords = [], []
    for y in range(0, H, step):
        for x in range(0, W, step):
            y0, x0 = y, x
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            patch = np.zeros((tile, tile), dtype=np.float32)
            patch[:(y1-y0), :(x1-x0)] = img[y0:y1, x0:x1]
            tiles.append(patch)
            coords.append((y0, x0, y1, x1))
    return np.stack(tiles), coords, (H, W)

def merge_tiles(prob_tiles, coords, shape):
    # Merge the processed tiles back into a single image
    H, W = shape
    acc = np.zeros((H, W), dtype=np.float32)
    wgt = np.zeros((H, W), dtype=np.float32)
    for i, (y0, x0, y1, x1) in enumerate(coords):
        ph, pw = y1-y0, x1-x0
        acc[y0:y1, x0:x1] += prob_tiles[i][:ph, :pw]
        wgt[y0:y1, x0:x1] += 1.0
    wgt[wgt == 0] = 1.0
    return acc / wgt

def main():
    # Prompt the user to select which model to use for segmentation
    print("Select the model to use:")
    print("1. U-Net")
    print("2. Attention U-Net")
    model_choice = input("Enter the number corresponding to your choice: ").strip()

    # Determine the model and checkpoint path based on user input
    if model_choice == "1":
        model_name = "unet"
        ckpt_path = UNET_CKPT
    elif model_choice == "2":
        model_name = "attnunet"
        ckpt_path = ATTN_UNET_CKPT
    else:
        print("Invalid choice. Exiting.")
        return

    # Create a directory to save evaluation results for the selected model
    output_dir = os.path.join("evaluation", model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load the selected model and its trained weights
    model = get_model(model_name)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    # Load the new SEM image to be segmented
    sem = load_gray01(NEW_SEM_PATH)
    H, W = sem.shape
    print(f"[test] SEM: {os.path.basename(NEW_SEM_PATH)}  size: {H}x{W}")

    # Divide the image into tiles and predict segmentation for each tile
    tiles, coords, shape = tile_image(sem)
    pred_logits_tiles = []
    bs = 1  # Batch size for processing tiles
    with torch.no_grad():
        for i in range(0, len(tiles), bs):
            batch = tiles[i:i+bs]
            t = torch.from_numpy(batch[:, None, ...]).float().to(DEVICE)
            logits = model(t)
            pred_logits_tiles.append(logits.cpu().numpy()[:, 0])
    pred_logits_tiles = np.concatenate(pred_logits_tiles, axis=0)
    prob_tiles = 1.0 / (1.0 + np.exp(-pred_logits_tiles))  # Apply sigmoid to get probabilities
    prob_full = merge_tiles(prob_tiles, coords, shape)

    # Save the probability map as an image
    prob_map_path = os.path.join(output_dir, "probability_map.png")
    plt.figure(figsize=(10, 8))
    plt.imshow(prob_full, cmap='viridis')
    plt.colorbar(label='Boundary Probability')
    plt.title('Predicted Boundary Probability Map (New SEM)')
    plt.axis('off')
    plt.savefig(prob_map_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Apply thresholding and morphological operations to refine the segmentation
    pred_boundary = (prob_full >= THR).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    pred_boundary = cv2.morphologyEx(pred_boundary, cv2.MORPH_CLOSE, kernel)
    pred_boundary = cv2.dilate(pred_boundary, kernel, iterations=1)

    # Save the input SEM image and the predicted boundary mask
    input_path = os.path.join(output_dir, "input_sem.png")
    mask_path = os.path.join(output_dir, "predicted_boundary_mask.png")
    cv2.imwrite(input_path, (sem * 255).astype(np.uint8))
    cv2.imwrite(mask_path, pred_boundary * 255)

    # Analyze the segmentation results: count grains and visualize labeled grains
    labeled_grains = label(1 - pred_boundary)  # Grains are the non-boundary regions
    num_grains = labeled_grains.max()
    print(f"[test] Number of grains detected: {num_grains}")

    # Save the labeled grains visualization as an image
    labeled_grains_path = os.path.join(output_dir, "labeled_grains.png")
    plt.figure(figsize=(8, 6))
    plt.imshow(labeled_grains, cmap='nipy_spectral')
    plt.title(f'Labeled Grains (N={num_grains})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(labeled_grains_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Notify the user where the results have been saved
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    main()
