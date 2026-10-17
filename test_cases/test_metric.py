import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq
import os
import math

# Get user inputs
image_size = int(input("Enter image size (square image, e.g., 100): "))
num_squares = int(input("Enter number of squares: "))
output_folder = "metric_outputs"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Calculate layout parameters
height, width = image_size, image_size
padding = image_size // 10
square_size = image_size // 5

# Calculate grid for nice placement
rows = math.ceil(math.sqrt(num_squares))
cols = math.ceil(num_squares / rows)
cell_height = (height - 2 * padding) // rows
cell_width = (width - 2 * padding) // cols

# Adjust square_size to fit in cells
square_size = min(square_size, min(cell_height, cell_width) // 2)

# Generate ground truth image
gt = np.zeros((height, width), dtype=int)
for i in range(num_squares):
    row_idx = i // cols
    col_idx = i % cols
    y = padding + row_idx * cell_height + (cell_height - square_size) // 2
    x = padding + col_idx * cell_width + (cell_width - square_size) // 2
    gt[y:y+square_size, x:x+square_size] = 1

# Use ground truth directly for segmentation (no noise)
image = gt.astype(float)

# Apply K-means segmentation
flat_image = image.flatten().reshape(-1, 1)
centroids, _ = kmeans(flat_image, 2)
labels, _ = vq(flat_image, centroids)
pred = labels.reshape(height, width)

# Match labels: assume lower centroid is background (0)
if centroids[0] > centroids[1]:
    pred = 1 - pred

# Flatten for metrics
gt_flat = gt.flatten()
pred_flat = pred.flatten()

# Compute IoU
intersection = np.sum(gt_flat * pred_flat)
union = np.sum(gt_flat) + np.sum(pred_flat) - intersection
iou = intersection / union if union > 0 else 0.0

# Compute Dice
dice = 2 * intersection / (np.sum(gt_flat) + np.sum(pred_flat)) if (np.sum(gt_flat) + np.sum(pred_flat)) > 0 else 0.0

# Create overlay image for visualization
overlay = np.zeros((height, width, 3))
overlay[gt == 1] = [1, 0, 0]  # GT in red
overlay[pred == 1] = [0, 0, 1]  # Pred in blue
overlap_mask = (gt == 1) & (pred == 1)
overlay[overlap_mask] = [1, 0, 1]  # Overlap in magenta

# Set plot style for better aesthetics
plt.style.use('ggplot')  # Use ggplot style for clean look
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'figure.dpi': 100
})

# Plot image visualizations
fig1, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
axs[0].imshow(gt, cmap='gray')
axs[0].set_title('Ground Truth Image')
axs[0].axis('off')

axs[1].imshow(pred, cmap='gray')
axs[1].set_title('K-means Segmentation')
axs[1].axis('off')

axs[2].imshow(overlay)
axs[2].set_title('Overlay (GT: Red, Pred: Blue, Overlap: Magenta)')
axs[2].axis('off')

fig1.suptitle('Segmentation Visualization', fontsize=16, fontweight='bold')
fig1.savefig(os.path.join(output_folder, 'segmentation_visualization.png'))
plt.show()

# Plot metrics separately
metrics = {'IoU': iou, 'Dice': dice}
metric_names = list(metrics.keys())
metric_values = list(metrics.values())

fig2, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
bars = ax.bar(metric_names, metric_values, color=['#66b3ff', '#99ff99'], edgecolor='black')
ax.set_title('Segmentation Metrics', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score', fontsize=12)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
for bar, v in zip(bars, metric_values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')

fig2.savefig(os.path.join(output_folder, 'segmentation_metrics.png'))
plt.show()

# Print metrics
print(f'Intersection over Union (IoU): {iou:.4f}')
print(f'Dice Coefficient: {dice:.4f}')