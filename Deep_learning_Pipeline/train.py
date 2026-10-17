# train.py
# This is where the real learning happens! We train our U-Net or Attention U-Net on synthetic data,
# validate it to check progress, and test it—saving the best model and some cool visualizations along the way.
# The validation results are saved in the 'val' folder for later review.

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import torch
import torch.nn as nn

from config import (
    DEVICE, MODELS_DIR, STATS_OUT_DIR,  # Removed TEST_VIZ_DIR
    UNET_CKPT, ATTN_UNET_CKPT, EPOCHS_UNET, LR_UNET,
    VAL_IMG_DIR, VAL_MSK_DIR, POS_WEIGHT
)
from data_loader import build_loaders
from models import get_model

# workaround for OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ----- Losses & Metrics ----- #

class BCEDiceLoss(nn.Module):
    """Combine Binary Cross-Entropy (BCE) and Dice loss for balanced boundary detection.
    
    This custom loss blends BCE (which cares about each pixel's probability) with Dice
    (which measures how well boundaries overlap). It helps handle the imbalance between
    rare boundaries and common grains.
    
    Args:
        w_bce (float): Weight for BCE loss (default: 0.5).
        w_dice (float): Weight for Dice loss (default: 0.5).
    """
    def __init__(self, w_bce=0.5, w_dice=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(DEVICE))
        self.wb = w_bce; self.wd = w_dice

    def forward(self, logits, target):
        """Compute the combined loss for a batch.
        
        Args:
            logits (torch.Tensor): Model output before sigmoid (shape: [batch, 1, H, W]).
            target (torch.Tensor): Ground truth mask (shape: [batch, 1, H, W]).
        
        Returns:
            torch.Tensor: Weighted sum of BCE and Dice losses.
        """
        bce = self.bce(logits, target)  # Pixel-wise loss
        prob = torch.sigmoid(logits)    # Convert logits to probabilities
        eps = 1e-7  # Small epsilon to avoid division by zero
        intersection = 2.0 * torch.sum(prob * target) + eps
        union = torch.sum(prob) + torch.sum(target) + eps
        dice = 1 - (intersection / union)  # Dice loss (1 - coefficient)
        return self.wb * bce + self.wd * dice

def dice_coef(pred, target, eps=1e-7):
    """Calculate the Dice coefficient to measure segmentation overlap.
    
    The Dice coefficient is a metric that shows how well the predicted mask matches the ground truth,
    ranging from 0 (no overlap) to 1 (perfect overlap).
    
    Args:
        pred (torch.Tensor): Predicted probability mask (shape: [batch, 1, H, W]).
        target (torch.Tensor): Ground truth mask (shape: [batch, 1, H, W]).
        eps (float): Small value to prevent division by zero.
    
    Returns:
        float: Dice coefficient value.
    """
    pred = (pred > 0.5).float()  # Threshold to binary
    num = 2.0 * torch.sum(pred * target) + eps
    den = torch.sum(pred) + torch.sum(target) + eps
    return (num / den).item()

def iou_score(pred, target, eps=1e-7):
    """Calculate the Intersection over Union (IoU) score.
    
    IoU measures the overlap between prediction and ground truth divided by their union,
    ranging from 0 (no overlap) to 1 (perfect match).
    
    Args:
        pred (torch.Tensor): Predicted probability mask.
        target (torch.Tensor): Ground truth mask.
        eps (float): Small value to prevent division by zero.
    
    Returns:
        float: IoU score.
    """
    pred = (pred > 0.5).float()  # Threshold to binary
    inter = torch.sum(pred * target) + eps
    union = torch.sum(pred) + torch.sum(target) - inter + eps
    return (inter / union).item()

# ----- Utils ----- #

def ensure_dir(p):
    """Create a directory if it doesn’t exist—keeps our files tidy!
    
    Args:
        p (str): Path to the directory.
    """
    os.makedirs(p, exist_ok=True)

def plot_and_save_curves(train_losses, val_losses, out_png):
    """Plot and save training and validation loss curves for a visual check.
    
    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        out_png (str): Output path for the plot.
    """
    plt.figure()
    plt.plot(train_losses, label="Training Loss"); plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_and_save_model_curves(model_name, train_losses, val_losses, output_dir):
    """Plot and save training and validation loss curves for a specific model.

    Args:
        model_name (str): Name of the model (e.g., 'UNet', 'Attention UNet').
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        output_dir (str): Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, f"{model_name}_loss_curve.png")

    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"Loss Curve for {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def save_json(obj, path):
    """Save a Python object as a JSON file for easy reference.

    Args:
        obj (dict): Object to save (e.g., loss curves or metrics).
        path (str): Output file path.
    """
    def numpy_serializer(o):
        if isinstance(o, (np.integer, np.int64)):
            return int(o)
        elif isinstance(o, (np.floating, np.float64)):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=4, default=numpy_serializer)

def visualize_comparison(img, gt_mask, pred_mask, save_path, index):
    """Visualize ground truth and predicted grains side by side for validation.
    
    Args:
        img (np.ndarray): Input image.
        gt_mask (np.ndarray): Ground truth mask.
        pred_mask (np.ndarray): Predicted mask.
        save_path (str): Template for save path with {index}.
        index (int): Index for filename.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Ground Truth
    ax1.imshow(gt_mask, cmap='gray')
    ax1.set_title('Ground Truth Grains')
    ax1.axis('off')
    
    # Predicted
    ax2.imshow(pred_mask, cmap='gray')
    ax2.set_title('Predicted Grains')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path.format(index))
    plt.close()

def main():
    """Train the segmentation model and evaluate on validation and test sets with visualizations."""
    start_time = time.time()  # Start timing

    # Load configuration variables once
    config = {
        "DEVICE": DEVICE,
        "MODELS_DIR": MODELS_DIR,
        "STATS_OUT_DIR": STATS_OUT_DIR,
        "UNET_CKPT": UNET_CKPT,
        "ATTN_UNET_CKPT": ATTN_UNET_CKPT,
        "EPOCHS_UNET": EPOCHS_UNET,
        "LR_UNET": LR_UNET,
        "VAL_IMG_DIR": VAL_IMG_DIR,
        "VAL_MSK_DIR": VAL_MSK_DIR,
        "POS_WEIGHT": POS_WEIGHT
    }

    ensure_dir(config["MODELS_DIR"])
    ensure_dir(config["STATS_OUT_DIR"])
    ensure_dir(config["VAL_IMG_DIR"])
    ensure_dir(config["VAL_MSK_DIR"])
    ensure_dir(os.path.join(config["VAL_IMG_DIR"], "../viz"))

    # Build data loaders from the synthetic dataset
    train_loader, val_loader, test_loader = build_loaders()

    # Prompt user for model selection
    print("Select the model to train:")
    print("1. U-Net")
    print("2. Attention U-Net")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        model_name = "unet"
        ckpt_path = config["UNET_CKPT"]
    elif choice == "2":
        model_name = "attnunet"
        ckpt_path = config["ATTN_UNET_CKPT"]
    else:
        raise ValueError("Invalid choice. Please enter 1 for U-Net or 2 for Attention U-Net.")

    print(f"[train] Using {model_name} model on {config['DEVICE']}")
    model = get_model(model_name)

    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LR_UNET"])
    criterion = BCEDiceLoss(0.5, 0.5)  # Balanced mix of BCE and Dice

    # Train from scratch and save the best checkpoint
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    for epoch in range(1, config["EPOCHS_UNET"] + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        for imgs, msks in tqdm(train_loader, desc=f"Training {model_name} Epoch {epoch}/{config['EPOCHS_UNET']}"):
            imgs = imgs.to(config["DEVICE"])
            msks = msks.to(config["DEVICE"])
            logits = model(imgs)
            loss = criterion(logits, msks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_losses.append(running_loss / len(train_loader.dataset))

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice_scores = []
        val_iou_scores = []
        val_viz_saved = 0
        with torch.no_grad():
            for i, (imgs, msks) in enumerate(val_loader):
                imgs = imgs.to(config["DEVICE"])
                msks = msks.to(config["DEVICE"])
                logits = model(imgs)
                loss = criterion(logits, msks)
                val_loss += loss.item() * imgs.size(0)

                prob = torch.sigmoid(logits)
                val_dice_scores.append(dice_coef(prob, msks))
                val_iou_scores.append(iou_score(prob, msks))

                if val_viz_saved < 5:
                    for j in range(imgs.size(0)):
                        img = imgs[j].cpu().numpy().squeeze()
                        gt_mask = (msks[j].cpu().numpy().squeeze() * 255).astype(np.uint8)
                        pred_mask = (prob[j].cpu().numpy().squeeze() * 255).astype(np.uint8)
                        cv2.imwrite(os.path.join(config["VAL_IMG_DIR"], f"val_img_{epoch:03d}_{i*imgs.size(0)+j:03d}.png"), 
                                   (img * 255).astype(np.uint8))
                        cv2.imwrite(os.path.join(config["VAL_MSK_DIR"], f"val_mask_{epoch:03d}_{i*imgs.size(0)+j:03d}.png"), gt_mask)
                        visualize_comparison(img, gt_mask, pred_mask, 
                                            os.path.join(config["VAL_IMG_DIR"], "../viz/val_comp_{:03d}.png"), 
                                            i*imgs.size(0)+j)
                        val_viz_saved += 1

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        avg_dice = np.mean(val_dice_scores)
        avg_iou = np.mean(val_iou_scores)
        print(f"[train] Epoch {epoch}/{config['EPOCHS_UNET']}: Train Loss = {train_losses[-1]:.4f}, "
              f"Val Loss = {val_loss:.4f}, Dice = {avg_dice:.4f}, IoU = {avg_iou:.4f}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"[train] Saved best model to {ckpt_path} with Val Loss = {val_loss:.4f}")

    # Save loss curves
    plot_and_save_curves(train_losses, val_losses, os.path.join(config["STATS_OUT_DIR"], "loss_curves.png"))
    plot_and_save_model_curves(model_name, train_losses, val_losses, config["STATS_OUT_DIR"])
    save_json({"train_losses": train_losses, "val_losses": val_losses}, 
              os.path.join(config["STATS_OUT_DIR"], "loss_history.json"))
    print(f"[train] Training complete! Saved loss curves and history.")

    # Save validation results
    val_results = {
        "epoch": list(range(1, config["EPOCHS_UNET"] + 1)),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_dice_scores": val_dice_scores,
        "val_iou_scores": val_iou_scores
    }
    val_results_path = os.path.join(config["STATS_OUT_DIR"], f"{model_name}_validation_results.json")
    save_json(val_results, val_results_path)
    print(f"[train] Validation results saved -> {val_results_path}")

    # Test phase
    model.eval()
    test_dice_scores = []
    test_iou_scores = []
    test_viz_saved = 0
    with torch.no_grad():
        from skimage.measure import label
        gt_counts = []
        pred_counts = []
        for i, (imgs, msks) in enumerate(test_loader):
            imgs = imgs.to(config["DEVICE"])
            msks = msks.to(config["DEVICE"])
            logits = model(imgs)
            prob = torch.sigmoid(logits)
            test_dice_scores.append(dice_coef(prob, msks))
            test_iou_scores.append(iou_score(prob, msks))
            # Count grains for each image in batch
            for j in range(imgs.size(0)):
                gt_mask = (msks[j].cpu().numpy().squeeze() > 0.5).astype(np.uint8)
                pred_mask = (prob[j].cpu().numpy().squeeze() > 0.5).astype(np.uint8)
                # Masks represent boundaries (1), not grain interiors.
                # Label interior regions to estimate grain counts.
                gt_grain_count = label(gt_mask == 0, connectivity=2).max()
                pred_grain_count = label(pred_mask == 0, connectivity=2).max()
                gt_counts.append(gt_grain_count)
                pred_counts.append(pred_grain_count)
                if test_viz_saved < 5:
                    img = imgs[j].cpu().numpy().squeeze()
                    visualize_comparison(img, gt_mask * 255, pred_mask * 255,
                                        os.path.join(config["STATS_OUT_DIR"], f"{model_name}_test_comp_{i*imgs.size(0)+j:03d}.png"),
                                        i*imgs.size(0)+j)
                    test_viz_saved += 1

    avg_test_dice = np.mean(test_dice_scores)
    avg_test_iou = np.mean(test_iou_scores)
    print(f"[train] Test Results: Avg Dice = {avg_test_dice:.4f}, Avg IoU = {avg_test_iou:.4f}")

    # Save test results
    test_results = {
        "test_dice_scores": test_dice_scores,
        "test_iou_scores": test_iou_scores,
        "avg_test_dice": avg_test_dice,
        "avg_test_iou": avg_test_iou,
        "gt_counts": gt_counts,
        "pred_counts": pred_counts
    }
    test_results_path = os.path.join(config["STATS_OUT_DIR"], f"{model_name}_test_results.json")
    save_json(test_results, test_results_path)
    print(f"[train] Test results saved -> {test_results_path}")

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[train] Total time taken for {model_name} training and evaluation: {elapsed_time:.2f} seconds")

    # Save elapsed time
    time_info = {
        "model_name": model_name,
        "elapsed_time_seconds": elapsed_time
    }
    time_info_path = os.path.join(config["STATS_OUT_DIR"], f"{model_name}_training_time.json")
    save_json(time_info, time_info_path)
    print(f"[train] Training time saved -> {time_info_path}")

    # Plot predicted vs ground truth grain counts
    plt.figure(figsize=(8,6))
    plt.scatter(gt_counts, pred_counts, alpha=0.7, color='royalblue')
    plt.plot([min(gt_counts), max(gt_counts)], [min(gt_counts), max(gt_counts)], 'r--', label='Ideal')
    plt.xlabel('Ground Truth Grain Count')
    plt.ylabel('Predicted Grain Count')
    plt.title('Test Set: Predicted vs Ground Truth Grain Counts')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(config["STATS_OUT_DIR"], f"{model_name}_grain_count_scatter.png"))
    plt.close()

if __name__ == "__main__":
    main()