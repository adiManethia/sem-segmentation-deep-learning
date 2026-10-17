# statistics and segmentation 
# Computes per-grain statistics from  boundary mask.
# Saves CSV, JSON summary, histogram, and labeled visualization.

import os
import json
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops_table
from skimage.morphology import remove_small_holes, remove_small_objects
from config import INFERENCE_OUT_DIR, MIN_GRAIN_AREA, SCALE, STATS_OUT_DIR, DEVICE

# Workaround for OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def find_mask_path():
    """Find the predicted mask file based on user selection.

    Returns:
        str: Path to the selected mask file.

    Raises:
        FileNotFoundError: If no mask file is found.
    """
    print("Select the predicted mask to analyze:")
    print("1. U-Net predicted mask")
    print("2. Attention U-Net predicted mask")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        mask_name = "unet_pred_boundary_mask.png"
    elif choice == "2":
        mask_name = "attnunet_pred_boundary_mask.png"
    else:
        raise ValueError("Invalid choice. Please enter 1 for U-Net or 2 for Attention U-Net.")

    mask_path = os.path.join(INFERENCE_OUT_DIR, "unet" if choice == "1" else "attnunet", mask_name)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Predicted mask not found: {mask_path}")
    return mask_path

# Visualize labeled grains with a colorbar to interpret the colors
def visualize_labeled_grains(labels_img, stats_out_dir, model_name):
    """Visualize labeled grains with a colorbar to map colors to grain label IDs.

    Args:
        labels_img (np.ndarray): Labeled image where each grain has a unique ID.
        stats_out_dir (str): Directory to save the visualization.
        model_name (str): Name of the model (e.g., 'unet' or 'attnunet').
    """
    # Normalize labels to [0, 1] for colormap (avoid division by zero)
    max_label = max(labels_img.max(), 1)
    label_norm = labels_img.astype(np.float32) / max_label

    # Use a perceptually uniform colormap (e.g., 'viridis')
    plt.figure(figsize=(10, 8))
    im = plt.imshow(label_norm, cmap='viridis')

    # Add a colorbar to map colors to grain label IDs
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Grain Label ID (Normalized)', rotation=270, labelpad=15)

    # Get image size and number of grains
    img_h, img_w = labels_img.shape
    n_grains = int(labels_img.max())

    plt.xlabel(f'X (pixels, width={img_w})')
    plt.ylabel(f'Y (pixels, height={img_h})')
    plt.title(f'Labeled Grains (Size: {img_w}x{img_h})\nNumber of Grains: {n_grains}')
    plt.xticks(np.linspace(0, img_w, num=11, dtype=int))
    plt.yticks(np.linspace(0, img_h, num=11, dtype=int))
    plt.tight_layout()

    label_viz_path = os.path.join(stats_out_dir, f"labeled_grains_{model_name}.png")
    plt.savefig(label_viz_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"[stats] Saved labeled visualization for {model_name} -> {label_viz_path}")

def generate_statistics(pred_boundary, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    grain_mask = (pred_boundary == 0).astype(np.uint8)
    grain_bool = grain_mask.astype(bool)
    grain_bool = remove_small_holes(grain_bool, area_threshold=5)
    grain_bool = remove_small_objects(grain_bool, MIN_GRAIN_AREA)
    labels_img = label(grain_bool, connectivity=2)
    n_grains = int(labels_img.max())
    print(f"[stats] Detected grains: {n_grains}")

    props = regionprops_table(
        labels_img,
        properties=("label", "area", "equivalent_diameter"),
    )
    df = pd.DataFrame(props)

    summary = {
        "n_grains": int(n_grains),
        "eq_diam_mean_phys": float(df["equivalent_diameter"].mean()) if len(df) else 0.0,
        "eq_diam_std_phys": float(df["equivalent_diameter"].std()) if len(df) else 0.0,
    }
    for k, v in summary.items():
        print(f"[stats] {k}: {v}")

    csv_path = os.path.join(output_dir, "grain_stats.csv")
    df.to_csv(csv_path, index=False)
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"[stats] Saved CSV -> {csv_path}")
    print(f"[stats] Saved summary -> {summary_path}")

    if len(df):
        plt.figure(figsize=(8, 6))
        plt.hist(df["equivalent_diameter"], bins=50, color="#4e79a7", edgecolor="white")
        plt.title(f"Equivalent Diameter (pixels), n={n_grains}\nScale Factor: {SCALE}")
        plt.xlabel("Equivalent diameter (pixels)")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        hist_path = os.path.join(output_dir, "eq_diam_hist.png")
        plt.savefig(hist_path, dpi=300)
        plt.show()
        print(f"[stats] Saved histogram -> {hist_path}")

    visualize_labeled_grains(labels_img, output_dir, "model")
    return summary

def main():
    """Analyze the predicted mask or probability map to label grains and compute statistics."""
    os.makedirs(STATS_OUT_DIR, exist_ok=True)

    mask_path = find_mask_path()
    model_name = os.path.basename(mask_path).split('_')[0]  # Extract model name (e.g., 'unet' or 'attnunet')
    print(f"[stats] Using predicted mask: {mask_path} ")

    # Read the probability map
    prob_map = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    assert prob_map is not None, f"Could not read: {mask_path}"

    # Normalize probability map to [0, 1]
    prob_map = prob_map.astype(np.float32) / 255.0

    # Threshold the probability map to create a binary mask
    threshold = 0.5  # You can adjust this threshold based on your requirements
    pred_boundary = (prob_map > threshold).astype(np.uint8)

    generate_statistics(pred_boundary, STATS_OUT_DIR)

if __name__ == "__main__":
    main()