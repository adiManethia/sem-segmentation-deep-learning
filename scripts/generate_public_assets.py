from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from skimage.segmentation import find_boundaries

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def generate_voronoi_labels(size=256, n_seeds=70, seed=42):
    rng = np.random.default_rng(seed)
    seeds = rng.uniform(0, size, size=(n_seeds, 2))

    yy, xx = np.mgrid[0:size, 0:size]
    coords = np.column_stack([yy.ravel(), xx.ravel()])

    tree = cKDTree(seeds)
    _, idx = tree.query(coords)
    labels = idx.reshape(size, size)

    return labels


def labels_to_grayscale(labels, seed=42):
    rng = np.random.default_rng(seed)
    n_labels = labels.max() + 1
    gray_values = rng.integers(40, 220, size=n_labels)
    image = gray_values[labels].astype(np.float32)

    image = gaussian_filter(image, sigma=1.0)
    noise = rng.normal(0, 8, size=image.shape)
    image = np.clip(image + noise, 0, 255)

    return image.astype(np.uint8)


def save_synthetic_voronoi_demo():
    labels = generate_voronoi_labels()
    image = labels_to_grayscale(labels)
    boundaries = find_boundaries(labels, mode="thick")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Synthetic Voronoi Microstructure")
    axes[0].axis("off")

    axes[1].imshow(boundaries, cmap="gray")
    axes[1].set_title("Exact Ground-Truth Boundary Mask")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "synthetic_voronoi_demo.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return image, boundaries


def save_classical_segmentation_demo(image):
    pixels = image.reshape(-1, 1).astype(np.float32)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(pixels).reshape(image.shape)

    pred_boundaries = find_boundaries(cluster_ids, mode="thick")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Synthetic Input Image")
    axes[0].axis("off")

    axes[1].imshow(cluster_ids, cmap="nipy_spectral")
    axes[1].set_title("K-Means Clustering")
    axes[1].axis("off")

    axes[2].imshow(pred_boundaries, cmap="gray")
    axes[2].set_title("Derived Boundary Mask")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "classical_segmentation_demo.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_model_accuracy_plot():
    models = ["U-Net", "Attention U-Net"]
    iou = [0.8965, 0.8957]
    dice = [0.9454, 0.9450]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, iou, width, label="IoU")
    ax.bar(x + width / 2, dice, width, label="Dice")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Historical Synthetic-Test Segmentation Metrics")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_inference_comparison_plot():
    methods = ["ISODATA", "U-Net", "Attention U-Net"]
    times = [24.12, 6.59, 7.31]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(methods, times)
    ax.set_ylabel("Seconds per image")
    ax.set_title("Historical Inference Time Comparison")
    ax.set_ylim(0, max(times) + 5)

    for i, v in enumerate(times):
        ax.text(i, v + 0.4, f"{v:.2f}s", ha="center")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "inference_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    image, _ = save_synthetic_voronoi_demo()
    save_classical_segmentation_demo(image)
    save_model_accuracy_plot()
    save_inference_comparison_plot()

    print("Saved figures to:", FIG_DIR)
    for path in sorted(FIG_DIR.glob("*.png")):
        print(" -", path.name)


if __name__ == "__main__":
    main()
