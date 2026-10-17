import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, morphology, measure, color
from skimage.measure import label, regionprops
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
import os
import time
import csv

np.random.seed(42)

def segment_isodata(img, k_init=5, max_iter=30, merge_dist=10, split_std=10, min_cluster_size=5):
    """
    Perform ISODATA clustering on the input image.

    Parameters:
        img (ndarray): Grayscale input image.
        k_init (int): Initial number of clusters.
        max_iter (int): Maximum number of iterations.
        merge_dist (float): Distance threshold for merging clusters.
        split_std (float): Standard deviation threshold for splitting clusters.
        min_cluster_size (int): Minimum size of a cluster to be retained.

    Returns:
        ndarray: Segmentation labels for the input image.
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

def estimate_n_clusters(img, min_peak_height=10, min_peak_distance=2, plot_hist=False, output_dir=None, prefix=""):
    """
    Estimate the number of clusters in the image histogram using peak detection.

    Parameters:
        img (ndarray): Grayscale input image.
        min_peak_height (int): Minimum height of peaks to be considered.
        min_peak_distance (int): Minimum distance between peaks.
        plot_hist (bool): Whether to save a plot of the histogram and detected peaks.
        output_dir (str): Directory to save the histogram plot.
        prefix (str): Prefix for the saved plot filename.

    Returns:
        int: Estimated number of clusters.
    """
    hist, bin_edges = np.histogram(img, bins=256, range=(0,255))
    peaks, _ = find_peaks(hist, height=min_peak_height, distance=min_peak_distance)
    n_clusters = len(peaks)
    if plot_hist and output_dir:
        fig = plt.figure(figsize=(8,4))
        plt.plot(hist, label='Histogram')
        plt.plot(peaks, hist[peaks], "x", label='Detected Peaks')
        plt.title("Histogram and Detected Peaks")
        plt.xlabel("Intensity")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_histogram_peaks.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
    return n_clusters

def save_image_with_axes(img, path, title="", img_shape=None):
    """
    Save an image with X/Y axes showing ONLY pixel dimensions.
    
    Parameters:
        img (ndarray): Image to save.
        path (str): Path to save the image.
        title (str): Title for the plot.
        img_shape (tuple): Shape of the image (height, width) for pixel dimensions.
    """
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    ax.set_title(f"{title}\nSize: {img_shape[0]}x{img_shape[1]} pixels")
    ax.set_xlabel("Width (pixels)")
    ax.set_ylabel("Height (pixels)")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_colormap_with_axes(data, path, title="", img_shape=None, cmap='nipy_spectral'):
    """
    Save a colormap image with X/Y axes showing ONLY pixel dimensions.
    
    Parameters:
        data (ndarray): Data to visualize as colormap.
        path (str): Path to save the image.
        title (str): Title for the plot.
        img_shape (tuple): Shape of the image (height, width) for pixel dimensions.
        cmap (str): Colormap to use.
    """
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(data, cmap=cmap)
    ax.set_title(f"{title}\nSize: {img_shape[0]}x{img_shape[1]} pixels")
    ax.set_xlabel("Width (pixels)")
    ax.set_ylabel("Height (pixels)")
    plt.colorbar(im, ax=ax, label="Cluster Label")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_colorful_regions_with_axes(segmentation_mask, output_path, img_shape, title_prefix=""):
    """
    Generate and save colorful contiguous regions with ONLY pixel dimensions.
    
    Parameters:
        segmentation_mask (ndarray): Segmentation mask.
        output_path (str): Path to save.
        img_shape (tuple): Image shape for dimensions.
        title_prefix (str): Prefix for title.
    """
    labeled = measure.label(segmentation_mask, connectivity=2)
    colorful = color.label2rgb(labeled, bg_label=0, kind='overlay', alpha=0.8)
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(colorful)
    ax.set_title(f"{title_prefix} Colorful Grains\nSize: {img_shape[0]}x{img_shape[1]} pixels")
    ax.set_xlabel("Width (pixels)")
    ax.set_ylabel("Height (pixels)")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_grain_size_histogram(eqd, output_path, img_shape):
    """
    Generate and save a histogram of grain sizes with ONLY image dimensions.
    
    Parameters:
        eqd (list): List of equivalent diameters.
        output_path (str): Path to save.
        img_shape (tuple): Image shape.
    """
    fig = plt.figure(figsize=(8,6))
    plt.hist(eqd, bins=30, alpha=0.85, color='#1f77b4', edgecolor='black')
    plt.title(f"Grain Size Distribution\nImage Size: {img_shape[0]}x{img_shape[1]} pixels", fontsize=16)
    plt.xlabel("Equivalent Diameter (pixels)", fontsize=14)
    plt.ylabel("Number of Grains", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main_pipeline(sem_path, output_dir, scale_factor=2.981):
    """
    Main pipeline for processing and segmenting an SEM image.

    Parameters:
        sem_path (str): Path to the SEM image file.
        output_dir (str): Directory to save the results.
        scale_factor (float): Scale factor for converting pixels to physical units.

    Returns:
        None
    """
    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load and crop
    img = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Error: Unable to read '{sem_path}'. Check file path.")
    img_shape = img.shape
    save_image_with_axes(img, f"{output_dir}/01_cropped.png", "Original Cropped SEM", img_shape)

    # 2. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)
    save_image_with_axes(img_clahe, f"{output_dir}/02_clahe.png", "CLAHE Enhanced", img_shape)

    # 3. Bilateral filter
    img_bilateral = cv2.bilateralFilter(img_clahe, d=9, sigmaColor=75, sigmaSpace=75)
    save_image_with_axes(img_bilateral, f"{output_dir}/03_bilateral.png", "Bilateral Filtered", img_shape)

    # 4. Sobel edge detection
    sobelx = cv2.Sobel(img_bilateral, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_bilateral, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.hypot(sobelx, sobely)
    sobel = (sobel / sobel.max() * 255).astype(np.uint8)
    save_image_with_axes(sobel, f"{output_dir}/04_sobel.png", "Sobel Edges", img_shape)

    # 5. Global thresholding
    thresh_val = np.percentile(sobel, 90)
    binary = (sobel > thresh_val).astype(np.uint8) * 255
    save_image_with_axes(binary, f"{output_dir}/05_thresholded.png", "Thresholded", img_shape)

    # 6. Median filtering
    median = median_filter(binary, size=3)
    save_image_with_axes(median, f"{output_dir}/06_median.png", "Median Filtered", img_shape)

    # 7. Morphological closing
    selem = morphology.disk(3)
    closed = morphology.closing(median, selem)
    closed = (closed * 255).astype(np.uint8)
    save_image_with_axes(closed, f"{output_dir}/07_morph_closing.png", "Morphological Closing", img_shape)

    # 8. Connected component filtering
    labeled = measure.label(closed, connectivity=2)
    props = measure.regionprops(labeled)
    min_area = 50
    filtered = np.zeros_like(labeled)
    for p in props:
        if p.area >= min_area:
            filtered[labeled == p.label] = p.label
    filtered_binary = (filtered > 0).astype(np.uint8)*255
    save_image_with_axes(filtered_binary, f"{output_dir}/08_connected_filtered.png", "Filtered Components", img_shape)

    # 9. Estimate clusters and ISODATA clustering
    n_clusters = estimate_n_clusters(img_bilateral, plot_hist=True, output_dir=output_dir, prefix="isodata")
    isodata_labels = segment_isodata(img_bilateral, k_init=n_clusters)
    
    # Save ISODATA colormap - ONLY SIZE
    save_colormap_with_axes(isodata_labels, f"{output_dir}/09_isodata_colormap.png", 
                           "ISODATA Clustering", img_shape, cmap='nipy_spectral')
    
    # Save ISODATA segmentation - ONLY SIZE
    save_colormap_with_axes(isodata_labels, f"{output_dir}/09_isodata_segmentation.png", 
                           "ISODATA Segmentation", img_shape, cmap='nipy_spectral')

    # 10. Overlay edges
    edges = cv2.Canny(img_bilateral, 100, 200)
    overlay = cv2.addWeighted(img_bilateral, 0.7, edges, 0.3, 0)
    save_image_with_axes(overlay, f"{output_dir}/10_overlay.png", "Edge Overlay", img_shape)

    # 11. Statistics
    stats_labels = measure.label(isodata_labels, connectivity=2)
    props = measure.regionprops(stats_labels)
    areas = [p.area for p in props if p.area >= min_area]
    eqd = [2 * np.sqrt(a / np.pi) for a in areas]
    num_grains = len(eqd)

    # Save colorful grains - ONLY SIZE
    save_colorful_regions_with_axes(isodata_labels, f"{output_dir}/12_colorful_grains.png", img_shape, "ISODATA")

    # Save grain size histogram - ONLY IMAGE SIZE
    plot_grain_size_histogram(eqd, f"{output_dir}/11_grain_size_hist.png", img_shape)

    # Print summary
    print(f"Image size: {img_shape[0]}x{img_shape[1]} pixels")
    print(f"Estimated number of clusters: {n_clusters}")
    print(f"Number of grains: {num_grains}")
    print(f"Mean equivalent diameter: {np.mean(eqd):.2f} pixels")
    print(f"Median equivalent diameter: {np.median(eqd):.2f} pixels")
    print(f"Scale factor: {scale_factor} nm/pixel")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time: {total_time:.2f} seconds")

    # Save CSV
    csv_path = os.path.join(output_dir, "segmentation_statistics.csv")
    with open(csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Metric", "Value"])
        csv_writer.writerow(["Image dimensions (pixels)", f"{img_shape[0]}x{img_shape[1]}"])
        csv_writer.writerow(["Estimated number of clusters", n_clusters])
        csv_writer.writerow(["Number of grains", num_grains])
        csv_writer.writerow(["Mean equivalent diameter (pixels)", f"{np.mean(eqd):.2f}"])
        csv_writer.writerow(["Median equivalent diameter (pixels)", f"{np.median(eqd):.2f}"])
        csv_writer.writerow(["Scale factor (nm/pixel)", scale_factor])
        csv_writer.writerow(["Total time (seconds)", f"{total_time:.2f}"])

    
    print(f"Results saved to: {csv_path}")

# The ISODATA algorithm logic in this script is inspired by the implementation in the following repository:
# https://github.com/dfordandelia/ISODATA-Classifier/blob/main/ISODATA_Classifier/Classifier.py
# Credits to the original authors for their work.

# Example usage:
if __name__ == "__main__":
    print("Select the SEM image to process:")
    print("1. Au_3_BSD_50kx_A_woscale.tif")
    print("2. Au_3_BSD_50kx_B_woscale.tif")
    choice = input("Enter the number corresponding to your choice (1 or 2): ")

    if choice == "1":
        sem_path = "Au_3_BSD_50kx_A_woscale.tif"
    elif choice == "2":
        sem_path = "Au_3_BSD_50kx_B_woscale.tif"
    else:
        print("Invalid choice. Exiting.")
        exit(1)

    output_dir = "sem_segmentation_results"
    main_pipeline(sem_path, output_dir)