# inference.py
# Performs inference on the real SEM image using the trained model.
# Outputs the boundary mask and visualizes it.

import os
import numpy as np
import cv2
from PIL import Image
import torch
from config import UNET_CKPT
from config import ATTN_UNET_CKPT
from config import *
from config import DEVICE, INFERENCE_OUT_DIR, SEM_PATH, SEM_FILENAME, THR, TILE, OVERLAP
from models import get_model
import matplotlib.pyplot as plt
import time
import json

# Workaround for OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def load_gray01(path):
    img = np.array(Image.open(path))
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)

def tile_image(img, tile=TILE, overlap=OVERLAP):
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
    H, W = shape
    acc = np.zeros((H, W), dtype=np.float32)
    wgt = np.zeros((H, W), dtype=np.float32)
    for i, (y0, x0, y1, x1) in enumerate(coords):
        ph, pw = y1-y0, x1-x0
        acc[y0:y1, x0:x1] += prob_tiles[i][:ph, :pw]
        wgt[y0:y1, x0:x1] += 1.0
    wgt[wgt == 0] = 1.0
    return acc / wgt

def save_with_axes(image, save_path, x_label="Width (pixels)", y_label="Height (pixels)"):
    """Save an image with x and y axes labeled.

    Args:
        image (np.ndarray): The image to save.
        save_path (str): Path to save the image.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
    """
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def segment_image(sem_path, model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Determine model name based on the weight file
    if "unet_ckpt" in model_path.lower():
        model_name = "unet"
    elif "attnunet_ckpt" in model_path.lower():
        model_name = "attnunet"
    else:
        raise ValueError("Unsupported model type. Ensure the model name is in the weight file.")

    model_out_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_out_dir, exist_ok=True)

    start_time = time.time()

    model = get_model(model_name)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except RuntimeError as e:
        raise RuntimeError(f"Error loading weights for {model_name}: {e}")

    model.eval()

    sem = load_gray01(sem_path)
    H, W = sem.shape
    print(f"[inference] SEM: {os.path.basename(sem_path)}  size: {H}x{W}")

    tiles, coords, shape = tile_image(sem)
    pred_logits_tiles = []
    bs = 1
    with torch.no_grad():
        for i in range(0, len(tiles), bs):
            batch = tiles[i:i+bs]
            t = torch.from_numpy(batch[:, None, ...]).float().to(DEVICE)
            logits = model(t)
            pred_logits_tiles.append(logits.cpu().numpy()[:, 0])
    pred_logits_tiles = np.concatenate(pred_logits_tiles, axis=0)
    prob_tiles = 1.0 / (1.0 + np.exp(-pred_logits_tiles))
    prob_full = merge_tiles(prob_tiles, coords, shape)

    pred_boundary = (prob_full >= THR).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    pred_boundary = cv2.morphologyEx(pred_boundary, cv2.MORPH_CLOSE, kernel)
    pred_boundary = cv2.dilate(pred_boundary, kernel, iterations=1)

    input_path = os.path.join(model_out_dir, "sem_input.png")
    mask_path = os.path.join(model_out_dir, f"{model_name}_pred_boundary_mask.png")
    cv2.imwrite(input_path, (sem*255).astype(np.uint8))
    cv2.imwrite(mask_path, pred_boundary*255)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[inference] Total time taken for inference: {elapsed_time:.2f} seconds")

    return pred_boundary

def main():
    os.makedirs(INFERENCE_OUT_DIR, exist_ok=True)

    # Prompt user for model selection
    print("Select the trained model for inference:")
    print("1. U-Net")
    print("2. Attention U-Net")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        model_name = "unet"
        ckpt_path = UNET_CKPT
    elif choice == "2":
        model_name = "attnunet"
        ckpt_path = ATTN_UNET_CKPT
    else:
        raise ValueError("Invalid choice. Please enter 1 for U-Net or 2 for Attention U-Net.")

    # Create a subdirectory for the selected model's results
    model_out_dir = os.path.join(INFERENCE_OUT_DIR, model_name)
    os.makedirs(model_out_dir, exist_ok=True)

    # Start timing
    start_time = time.time()

    # Load model
    model = get_model(model_name)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # Load SEM
    sem = load_gray01(SEM_PATH)
    H, W = sem.shape
    print(f"[inference] SEM: {SEM_FILENAME}  size: {H}x{W}")

    # Tile + predict
    tiles, coords, shape = tile_image(sem)
    pred_logits_tiles = []
    bs = 1
    with torch.no_grad():
        for i in range(0, len(tiles), bs):
            batch = tiles[i:i+bs]
            t = torch.from_numpy(batch[:, None, ...]).float().to(DEVICE)
            logits = model(t)
            pred_logits_tiles.append(logits.cpu().numpy()[:, 0])
    pred_logits_tiles = np.concatenate(pred_logits_tiles, axis=0)
    prob_tiles = 1.0 / (1.0 + np.exp(-pred_logits_tiles))
    prob_full = merge_tiles(prob_tiles, coords, shape)

    # Visualize the probability map
    plt.figure(figsize=(10, 8))
    plt.imshow(prob_full, cmap='viridis') 
    plt.colorbar(label='Boundary Probability')
    plt.title('Predicted Boundary Probability Map')
    plt.axis('off')
    prob_map_viz_path = os.path.join(model_out_dir, f"{model_name}_prob_map.png")
    plt.savefig(prob_map_viz_path)
    plt.show()
    print(f"[inference] Saved probability map visualization -> {prob_map_viz_path}")

    # Threshold and simple morphology
    pred_boundary = (prob_full >= THR).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    pred_boundary = cv2.morphologyEx(pred_boundary, cv2.MORPH_CLOSE, kernel)
    pred_boundary = cv2.dilate(pred_boundary, kernel, iterations=1)

    # Save outputs
    input_path = os.path.join(model_out_dir, "sem_input.png")
    mask_path = os.path.join(model_out_dir, f"{model_name}_pred_boundary_mask.png")
    cv2.imwrite(input_path, (sem*255).astype(np.uint8))
    cv2.imwrite(mask_path, pred_boundary*255)
    print("[inference] Saved:")
    print(" -", input_path)
    print(" -", mask_path)

    # Save with axes
    input_path_labeled = os.path.join(model_out_dir, "sem_input_labeled.png")
    mask_path_labeled = os.path.join(model_out_dir, f"{model_name}_pred_boundary_mask_labeled.png")
    save_with_axes(sem, input_path_labeled)
    save_with_axes(pred_boundary, mask_path_labeled)
    print("[inference] Saved with axes:")
    print(" -", input_path_labeled)
    print(" -", mask_path_labeled)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[inference] Total time taken for inference: {elapsed_time:.2f} seconds")

    # Save elapsed time to a file
    time_info = {"elapsed_time_seconds": elapsed_time}
    time_info_path = os.path.join(model_out_dir, "inference_time.json")
    with open(time_info_path, "w") as f:
        json.dump(time_info, f, indent=4)
    print(f"[inference] Saved inference time -> {time_info_path}")

if __name__ == "__main__":
    main()