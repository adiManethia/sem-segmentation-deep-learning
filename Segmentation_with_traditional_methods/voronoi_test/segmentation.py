import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.segmentation import mark_boundaries, slic
from scipy.signal import find_peaks
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter
import os
import csv
import time

# Function to load Voronoi data (grayscale image and label map)
def load_voronoi_data(output_dir, prefix):
    """
    Load the grayscale image and label map for Voronoi segmentation.

    Args:
        output_dir (str): Directory containing the data.
        prefix (str): Prefix for the file names.

    Returns:
        tuple: Grayscale image and label map as numpy arrays.
    """
    grayscale_img = cv2.imread(f"{output_dir}/{prefix}_grains_gray.png", cv2.IMREAD_GRAYSCALE)
    label_map = np.load(f"{output_dir}/{prefix}_labels.npy")
    return grayscale_img, label_map

# Function to estimate the number of clusters in an image
def estimate_n_clusters(img, min_peak_height=10, min_peak_distance=2, plot_hist=False, output_dir=None, prefix=""):
    """
    Estimate the number of clusters in an image using histogram peak detection.

    Args:
        img (np.ndarray): Input grayscale image.
        min_peak_height (int): Minimum height of peaks to detect.
        min_peak_distance (int): Minimum distance between peaks.
        plot_hist (bool): Whether to plot the histogram and detected peaks.
        output_dir (str): Directory to save the histogram plot (if enabled).
        prefix (str): Prefix for the saved plot file name.

    Returns:
        int: Estimated number of clusters.
    """
    # Compute histogram
    hist, bin_edges = np.histogram(img, bins=256, range=(0, 255))

    # Smooth the histogram to reduce noise
    smoothed_hist = gaussian_filter(hist, sigma=1)

    # Dynamically adjust peak detection parameters
    dynamic_min_peak_height = max(min_peak_height, smoothed_hist.mean() + smoothed_hist.std())
    dynamic_min_peak_distance = max(min_peak_distance, len(smoothed_hist) // 50)  # Adjust based on histogram length

    # Detect peaks
    peaks, _ = find_peaks(smoothed_hist, height=dynamic_min_peak_height, distance=dynamic_min_peak_distance)
    n_clusters = len(peaks)

    # Validate n_clusters
    n_clusters = max(2, min(n_clusters, 100))  # Ensure clusters are between 2 and 100

    # Plot histogram and detected peaks (optional)
    if plot_hist and output_dir:
        plt.figure(figsize=(8, 4))
        plt.plot(hist, label='Original Histogram')
        plt.plot(smoothed_hist, label='Smoothed Histogram')
        plt.plot(peaks, smoothed_hist[peaks], "x", label='Detected Peaks')
        plt.title("Histogram and Detected Peaks")
        plt.xlabel("Intensity")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_histogram_peaks.png"))
        plt.close()

    return n_clusters

# Function to perform K-means segmentation
def segment_kmeans(img, n_clusters):
    """
    Segment an image using K-means clustering.

    Args:
        img (np.ndarray): Input grayscale image.
        n_clusters (int): Number of clusters.

    Returns:
        np.ndarray: Segmentation map.
    """
    pixels = img.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, labels, centers = cv2.kmeans(pixels, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return labels.reshape(img.shape)

# Function to perform ISODATA segmentation
def segment_isodata(img, k_init=6, max_iter=30, merge_dist=10, split_std=10, min_cluster_size=5):
    """
    Segment an image using the ISODATA algorithm.

    Args:
        img (np.ndarray): Input grayscale image.
        k_init (int): Initial number of clusters.
        max_iter (int): Maximum number of iterations.
        merge_dist (float): Distance threshold for merging clusters.
        split_std (float): Standard deviation threshold for splitting clusters.
        min_cluster_size (int): Minimum size of a cluster.

    Returns:
        np.ndarray: Segmentation map.
    """
    data = img.flatten()
    centers = np.random.choice(data, k_init, replace=False)
    for it in range(max_iter):
        dists = np.abs(data[:, None] - centers[None, :])
        labels = np.argmin(dists, axis=1)
        new_centers = []
        for i in range(len(centers)):
            members = data[labels == i]
            if len(members) > 0:
                new_centers.append(np.mean(members))
            else:
                new_centers.append(centers[i])
        centers = np.array(new_centers)
        # Merge clusters that are close
        to_merge = []
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                if np.abs(centers[i] - centers[j]) < merge_dist:
                    to_merge.append((i, j))
        merged = set()
        for i, j in to_merge:
            if i not in merged and j not in merged:
                centers[i] = (centers[i] + centers[j]) / 2
                centers[j] = np.nan
                merged.add(j)
        centers = centers[~np.isnan(centers)]
        # Split clusters with high std
        new_centers = []
        for i, c in enumerate(centers):
            members = data[labels == i]
            if len(members) > min_cluster_size and np.std(members) > split_std:
                new_centers.extend([c - np.std(members)/2, c + np.std(members)/2])
            else:
                new_centers.append(c)
        centers = np.array(new_centers)
        if len(centers) == 0:
            break
    dists = np.abs(data[:, None] - centers[None, :])
    labels = np.argmin(dists, axis=1)
    return labels.reshape(img.shape)

def segment_slic(img, n_segments):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return slic(img_rgb, n_segments=n_segments, compactness=10, sigma=1, start_label=1)

def label_connected_components(seg_img):
    return label(seg_img, connectivity=1)

def get_grain_stats(labels):
    props = regionprops(labels)
    eqd = [2 * np.sqrt(p.area / np.pi) for p in props]
    return eqd

def plot_histogram_with_peaks(img, output_dir, prefix):
    hist, _ = np.histogram(img, bins=256, range=(0,255))
    peaks, _ = find_peaks(hist, height=10, distance=2)
    plt.figure(figsize=(8,4))
    plt.plot(hist, label='Histogram')
    plt.plot(peaks, hist[peaks], "x", label='Detected Peaks')
    plt.title("Histogram and Detected Peaks")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_histogram_peaks.png"))
    plt.close()

def plot_grain_size_histogram(eqd, name, output_dir):
    plt.figure()
    plt.hist(eqd, bins=30, alpha=0.8)
    plt.title(f"Grain Size Histogram ({name})")
    plt.xlabel("Equivalent Diameter (px)")
    plt.ylabel("Number of Grains")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_grain_size_hist.png"))
    plt.close()

def save_overlay(gt_label_map, seg, name, output_dir):
    # Overlay segmentation boundaries on ground truth label map
    gt_img = gt_label_map.astype(np.float32) / gt_label_map.max()
    overlay = mark_boundaries(gt_img, seg, color=(1,0,0), mode='thick')
    plt.figure(figsize=(6,6))
    plt.imshow(overlay, cmap='gray')
    plt.title(f"{name} Segmentation Overlay (on GT)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_overlay_on_gt.png"), dpi=150)
    plt.close()

# Modify save_segmentation_map to save colorful segmentation results
def save_segmentation_map(seg, name, output_dir):
    plt.figure(figsize=(6,6))
    plt.imshow(seg, cmap='nipy_spectral')  # Use a colorful colormap
    plt.axis('off')  # Remove axis
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_segmentation_map.png"), dpi=150, bbox_inches='tight', pad_inches=0)  # Save without border
    plt.close()

def adjusted_rand_index(seg, gt):
    return adjusted_rand_score(gt.ravel(), seg.ravel())

# Function to compute the Intersection over Union (IoU) between two binary masks
def compute_iou(seg, gt):
    """
    Compute the Intersection over Union (IoU) metric between two binary masks.

    Args:
        seg (np.ndarray): Segmentation result (binary mask).
        gt (np.ndarray): Ground truth mask (binary mask).

    Returns:
        float: IoU value, which is the ratio of the intersection area to the union area.
               Returns 0 if the union area is zero.
    """
    seg_bin = seg > 0
    gt_bin = gt > 0
    intersection = np.logical_and(seg_bin, gt_bin).sum()
    union = np.logical_or(seg_bin, gt_bin).sum()
    return intersection / union if union > 0 else 0

# Function to compute the Dice coefficient between two binary masks
def compute_dice(seg, gt):
    """
    Compute the Dice coefficient between two binary masks.

    Args:
        seg (np.ndarray): Segmentation result (binary mask).
        gt (np.ndarray): Ground truth mask (binary mask).

    Returns:
        float: Dice coefficient, which is a measure of overlap between the two masks.
               Returns 0 if the sum of the masks is zero.
    """
    seg_bin = seg > 0
    gt_bin = gt > 0
    intersection = np.logical_and(seg_bin, gt_bin).sum()
    return 2 * intersection / (seg_bin.sum() + gt_bin.sum()) if (seg_bin.sum() + gt_bin.sum()) > 0 else 0

def compute_per_grain_iou_and_dice(gt_labels, pred_labels):
    gt_ids = np.unique(gt_labels)
    pred_ids = np.unique(pred_labels)
    gt_ids = gt_ids[gt_ids != 0]  # Exclude background
    pred_ids = pred_ids[pred_ids != 0]  # Exclude background

    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))

    for i, gt_id in enumerate(gt_ids):
        gt_mask = (gt_labels == gt_id)
        for j, pred_id in enumerate(pred_ids):
            pred_mask = (pred_labels == pred_id)
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            iou_matrix[i, j] = intersection / union if union > 0 else 0

    # Hungarian matching to maximize IoU
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_ious = iou_matrix[row_ind, col_ind]

    # Compute Dice for matched pairs
    matched_dices = []
    for i, j in zip(row_ind, col_ind):
        gt_mask = (gt_labels == gt_ids[i])
        pred_mask = (pred_labels == pred_ids[j])
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        dice = 2 * intersection / (gt_mask.sum() + pred_mask.sum()) if (gt_mask.sum() + pred_mask.sum()) > 0 else 0
        matched_dices.append(dice)

    mean_iou = matched_ious.mean() if len(matched_ious) > 0 else 0
    mean_dice = np.mean(matched_dices) if len(matched_dices) > 0 else 0

    return mean_iou, mean_dice

# Function to save results to a CSV file
def save_results_to_csv(results, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Method", "Time Taken (s)", "IoU", "Dice", "Segmented Grains"])
        writer.writerows(results)

# --- MAIN PIPELINE ---

if __name__ == "__main__":
    output_dir = "voronoi_generation"
    prefix = "periodic"
    results_csv = os.path.join(output_dir, f"{prefix}_segmentation_results.csv")

    # 1. Load data
    grayscale_img, label_map = load_voronoi_data(output_dir, prefix)

    # 2. Estimate number of clusters and plot histogram with peaks
    n_clusters = estimate_n_clusters(grayscale_img, min_peak_height=10, min_peak_distance=2, plot_hist=True, output_dir=output_dir, prefix=prefix)
    print(f"Estimated number of clusters: {n_clusters}")
    plot_histogram_with_peaks(grayscale_img, output_dir, prefix)

    # 3. Segmentation
    results = []
    for segment_func, name in zip(
        [segment_kmeans, segment_isodata, segment_slic],
        ["K-means", "ISODATA", "SLIC"]
    ):
        start_time = time.time()  # Start timing

        # Perform segmentation and all related operations
        seg = segment_func(grayscale_img, n_clusters)
        seg_labeled = label_connected_components(seg)
        iou, dice = compute_per_grain_iou_and_dice(label_map, seg_labeled)
        segmented_grains = len(np.unique(seg_labeled)) - 1

        elapsed_time = time.time() - start_time  # Calculate elapsed time

        print(f"{name}: Time taken = {elapsed_time:.3f} seconds")
        print(f"{name}: Mean IoU = {iou:.3f}")
        print(f"{name}: Mean Dice = {dice:.3f}")
        print(f"{name}: Segmented grains: {segmented_grains}")

        # Append results for CSV
        results.append([name, elapsed_time, iou, dice, segmented_grains])

        # Save segmentation map
        save_segmentation_map(seg_labeled, f"{prefix}_{name.lower()}", output_dir)

    # Save all results to CSV
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Method", "Time Taken (s)", "IoU", "Dice", "Segmented Grains"])
        writer.writerows(results)

    print(f"Results saved to {results_csv}")

    # Also report ground truth
   #print(f"GT grains: {len(np.unique(label_map))}")