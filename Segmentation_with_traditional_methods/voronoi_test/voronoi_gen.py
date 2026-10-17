import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from skimage.measure import label
import os




# 1) Seeds inside the base cell (reproducible)
def generate_seeds(num_seeds, image_size, seed=42):
    """
    Uniformly sample 'num_seeds' points in the base cell [0,H) x [0,W).
    Returns float coordinates shaped (num_seeds, 2) as (x, y).
    """
    np.random.seed(seed)
    H, W = image_size
    return np.random.rand(num_seeds, 2) * np.array([H, W], dtype=float)

# 2) Periodic Voronoi via 3x3 tiling of seeds (KD-tree on seeds, not on image)
def make_voronoi_label_map_periodic(seeds, image_size):
    """
    Assign each pixel to its nearest original seed under periodic (toroidal) geometry.
    Achieved by building a KD-tree over a 3x3 tiling of the seeds and mapping back.
    Returns integer labels 1..N of shape (H, W).
    """
    H, W = image_size
    tiled, tiled_idx = [], []

    for dx in (-H, 0, H):
        for dy in (-W, 0, W):
            tiled.append(seeds + np.array([dx, dy], dtype=float))
            tiled_idx.extend(range(len(seeds)))

    tiled = np.vstack(tiled)
    tiled_idx = np.array(tiled_idx, dtype=int)

    # All pixel coords (x, y) matching our seed coordinate convention
    y, x = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    coords = np.column_stack([x.ravel(), y.ravel()])

    tree = cKDTree(tiled)
    _, idxs = tree.query(coords, k=1)

    # Map to original seed labels [1..N]
    label_map = (tiled_idx[idxs] % len(seeds)) + 1
    return label_map.reshape((H, W))

# 3) Periodic connected components (8-connected), stitched across opposite borders per raw label
def periodic_connected_components(label_map):
    """
    Convert the per-seed ownership label_map into periodic grains:
    - For each raw seed label v, run 8-connected components on (label_map == v).
    - Stitch components that touch across opposite borders (left-right, top-bottom).
    - Assign global grain IDs 1..G across all v.
    Returns periodic_labels (int32) shaped (H, W).
    """
    H, W = label_map.shape
    periodic_labels = np.zeros_like(label_map, dtype=np.int32)
    next_id = 1

    unique_vals = np.unique(label_map)
    for v in unique_vals:
        # Binary mask for this seed label
        mask = (label_map == v)
        if not mask.any():  # fast skip if empty
            continue

        # 8-connectivity inside the cut rectangle
        comps = label(mask.astype(np.uint8), connectivity=2)
        K = int(comps.max())
        if K == 0:
            continue

        # Union-Find over component IDs of this v
        parent = {c: c for c in range(1, K + 1)}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # Stitch left-right borders: (i,0) with (i,W-1)
        left_col = comps[:, 0]
        right_col = comps[:, W - 1]
        for i in range(H):
            li, ri = int(left_col[i]), int(right_col[i])
            if li > 0 and ri > 0:
                union(li, ri)

        # Stitch top-bottom borders: (0,j) with (H-1,j)
        top_row = comps[0, :]
        bot_row = comps[H - 1, :]
        for j in range(W):
            ti, bi = int(top_row[j]), int(bot_row[j])
            if ti > 0 and bi > 0:
                union(ti, bi)

        # Map each local component to a root, then to a new global grain ID
        root = {c: find(c) for c in range(1, K + 1)}
        root_to_gid = {}
        for c in range(1, K + 1):
            r = root[c]
            if r not in root_to_gid:
                root_to_gid[r] = next_id
                next_id += 1

        # Write stitched IDs into output for this v
        coords_v = np.argwhere(mask)
        for (i, j) in coords_v:
            cid = int(comps[i, j])
            periodic_labels[i, j] = root_to_gid[root[cid]]

    return periodic_labels  # grain IDs 1..G

# 4) Map each periodic grain ID to a unique grayscale tone (uint8)
def grains_to_grayscale(periodic_labels):
    """
    Map integer grain IDs (1..G) to unique grayscale intensities in [1..255].
    Background (0) stays 0. Output is uint8 for saving/viewing.
    """
    ids = np.unique(periodic_labels)
    ids = ids[ids != 0]  # exclude background if any
    if len(ids) == 0:
        return np.zeros_like(periodic_labels, dtype=np.uint8)

    # Evenly spaced palette (avoid 0 to keep background distinct)
    palette = np.linspace(1, 255, len(ids), dtype=np.uint8)

    # Build LUT and map in one shot
    lut = np.zeros(int(periodic_labels.max()) + 1, dtype=np.uint8)
    lut[ids] = palette
    return lut[periodic_labels]

# 5) Save helpers
def save_arrays_and_image(gray_img, labels, out_dir, prefix):
    """
    Save the grayscale visualization (.png) and the integer labels (.npy).
    """
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(f"{out_dir}/{prefix}_grains_gray.png", gray_img)
    np.save(f"{out_dir}/{prefix}_labels.npy", labels)

def show(gray_img, title, out_path=None):
    """
    Quick matplotlib display; optionally save the figure.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(gray_img, cmap='gray', interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.show()

# 6) Main (periodic-only)
if __name__ == "__main__":
    # Ask user for input
    num_seeds = int(input("Enter the number of seeds: "))
    img_size = int(input("Enter the image size (width and height): "))

    image_size = (img_size, img_size)   # (H, W)
    output_dir = "voronoi_generation"
    prefix = "periodic"

    # Seeds in the base cell
    seeds = generate_seeds(num_seeds, image_size, seed=42)

    # Periodic Voronoi label map (nearest original seed under torus geometry)
    label_map = make_voronoi_label_map_periodic(seeds, image_size)

    # Periodic 8-connected grains (wrap across borders for the same seed)
    grains = periodic_connected_components(label_map)
    num_grains = int(grains.max())

    # Grayscale for visual inspection (each grain → unique tone)
    grains_gray = grains_to_grayscale(grains)

    # Save raw seed ownership and stitched grains
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/{prefix}_raw_label_map.npy", label_map)  # for reference
    save_arrays_and_image(grains_gray, grains, output_dir, prefix)

    # Display stitched grains
    show(
        grains_gray,
        f"{prefix} grains (periodic), count={num_grains}",
        out_path=f"{output_dir}/{prefix}_grains_view.png"
    )

    print(f"GRAINS (periodic): {num_grains} from {num_seeds} seeds")