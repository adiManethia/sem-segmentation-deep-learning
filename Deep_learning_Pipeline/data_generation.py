# data_generation.py
# It creates synthetic SEM images using Voronoi tessellation
# and refines them with a GAN to look more like real data. Think of it as building a virtual lab!
# We also chop up the real SEM image into patches to train the GAN.

import os
import glob
import random
import numpy as np
import cv2
from PIL import Image
from sklearn.neighbors import KDTree
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time

from config import PATCH_SIZE, STRIDE, REAL_PATCH_DIR, SYNTH_TRAIN_SAMPLES, VOR_SEED_RANGE, BOUNDARY_THICKNESS, DEVICE, SEM_PATH, BATCH_GAN, LR_G, LR_D, EPOCHS_GAN, USE_GAN, LAMBDA_E, LAMBDA_ID, GAN_SAMPLES_DIR, REFINER_CKPT, FINAL_PAIRS, IMG_SIZE, DATASET_IMG_DIR, DATASET_RENDER_DIR, DATASET_MSK_DIR, DATASET_LABEL_GRAY_DIR, BOUNDARY_JITTER_P

# Set a random seed for reproducibility—keeps our experiments consistent!
def set_seed(seed=42):
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed (int): The seed value to set (default: 42).
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Load and normalize the SEM image to 0-1 range
def load_sem(path):
    """Load a SEM image and normalize it to [0, 1] range.
    
    Args:
        path (str): Path to the SEM image file.
    
    Returns:
        np.ndarray: Normalized grayscale image.
    """
    img = np.array(Image.open(path))
    if img.ndim > 2:  # Convert to grayscale if it’s RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)  # Avoid division by zero

# Generate multiple augmented versions of an image
def augment6(p):
    """Create six augmented versions of an image (original, flips, rotations).
    
    Args:
        p (np.ndarray): Input image array.
    
    Returns:
        list: List of augmented images.
    """
    return [p, np.fliplr(p), np.flipud(p), np.rot90(p), np.rot90(p, 2), np.rot90(p, 3)]

# Save a normalized image as a grayscale PNG
def save_gray_img01(path, arr01):
    """Save a normalized [0, 1] array as a grayscale PNG image.
    
    Args:
        path (str): Output file path.
        arr01 (np.ndarray): Image array in [0, 1] range.
    """
    arr8 = (np.clip(arr01, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, arr8)

# Cut the real SEM into smaller patches for GAN training
def extract_real_patches(sem_img, patch=PATCH_SIZE, stride=STRIDE):
    """Extract overlapping patches from a real SEM image and save them.
    
    Args:
        sem_img (np.ndarray): Input SEM image.
        patch (int): Size of each patch (default from config).
        stride (int): Step size between patches (default from config).
    """
    os.makedirs(REAL_PATCH_DIR, exist_ok=True)  # Make sure the folder exists
    H, W = sem_img.shape
    idx = 0
    for y in range(0, max(1, H - patch + 1), stride):
        for x in range(0, max(1, W - patch + 1), stride):
            crop = sem_img[y:y+patch, x:x+patch]
            if crop.shape != (patch, patch):  # Skip if patch is incomplete
                continue
            for a in augment6(crop):  # Add some variety with flips and rotations
                save_gray_img01(os.path.join(REAL_PATCH_DIR, f"sem_{idx:05d}.png"), a)
                idx += 1
    print(f"[data_generator] Saved {idx} real SEM patches -> {REAL_PATCH_DIR}")

# Generate random seed points for Voronoi
def generate_seeds(num_seeds, H, W, seed=None):
    """Generate random seed points for Voronoi tessellation.
    
    Args:
        num_seeds (int): Number of seeds to generate.
        H (int): Height of the image.
        W (int): Width of the image.
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        np.ndarray: Array of (y, x) coordinates.
    """
    if seed is not None:
        np.random.seed(seed)
    s = np.random.rand(num_seeds, 2) * np.array([H, W], dtype=np.float32)
    return s

# Create a Voronoi label map with periodic boundaries
def voronoi_label_map_periodic(seeds_yx, H, W):
    """Generate a Voronoi label map with periodic tiling to avoid edge artifacts.
    
    Args:
        seeds_yx (np.ndarray): Array of (y, x) seed coordinates.
        H (int): Height of the image.
        W (int): Width of the image.
    
    Returns:
        np.ndarray: Label map with unique IDs for each region.
    """
    tiled = []
    tiled_idx = []
    for dy in (-H, 0, H):  # Tile in all directions
        for dx in (-W, 0, W):
            shifted = seeds_yx + np.array([dy, dx], dtype=np.float32)
            tiled.append(shifted)
            tiled_idx.extend(range(len(seeds_yx)))
    tiled = np.vstack(tiled).astype(np.float32)
    tiled_idx = np.asarray(tiled_idx, dtype=np.int32)

    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing='ij')
    coords = np.column_stack([yy.ravel(), xx.ravel()])
    tree = KDTree(tiled)
    _, idxs = tree.query(coords, k=1)
    seed_ids = tiled_idx[idxs]
    label_map = (seed_ids % len(seeds_yx)) + 1  # Shift to 1-based indexing
    return label_map.reshape(H, W).astype(np.int32)

# Extract boundaries from a label map
def boundary_from_labels_periodic(label_map, thickness=1):
    """Detect boundaries between Voronoi regions with periodic handling.
    
    Args:
        label_map (np.ndarray): Voronoi label map.
        thickness (int): Width of boundaries in pixels (default from config).
    
    Returns:
        np.ndarray: Binary boundary mask.
    """
    H, W = label_map.shape
    diffs = np.zeros((H, W), dtype=np.uint8)
    diffs |= (label_map != np.roll(label_map, -1, axis=0)).astype(np.uint8)  # Vertical diffs
    diffs |= (label_map != np.roll(label_map, -1, axis=1)).astype(np.uint8)  # Horizontal diffs
    if thickness > 1:  # Widen boundaries if needed
        k = np.ones((thickness, thickness), np.uint8)
        diffs = cv2.dilate(diffs, k, iterations=1)
    return diffs

# Render a synthetic SEM image from a boundary mask
def render_from_mask(mask01, grain_range=(0.35, 0.75), boundary_range=(0.18, 0.32),
                     blur_ksize=3, blur_sigma=0.8, noise_sigma=0.01, apply_blur=True, apply_noise=True):
    """Render a synthetic SEM image based on a boundary mask.
    
    Args:
        mask01 (np.ndarray): Binary mask (0 = grain, 1 = boundary).
        grain_range (tuple): Intensity range for grains.
        boundary_range (tuple): Intensity range for boundaries.
        blur_ksize (int): Kernel size for Gaussian blur.
        blur_sigma (float): Sigma for Gaussian blur.
        noise_sigma (float): Standard deviation for noise.
        apply_blur (bool): Whether to apply blur.
        apply_noise (bool): Whether to add noise.
    
    Returns:
        np.ndarray: Rendered synthetic image.
    """
    H, W = mask01.shape
    img = np.zeros((H, W), np.float32)
    grain_mask = (mask01 == 0).astype(np.uint8)  # Areas outside boundaries
    num, labels = cv2.connectedComponents(grain_mask)
    for g in range(1, num):  # Assign random intensities to each grain
        val = np.random.uniform(*grain_range)
        img[labels == g] = val
    img[mask01 == 1] = np.random.uniform(*boundary_range)  # Set boundary intensity
    if apply_blur and blur_ksize > 0:  # Smooth it out a bit
        k = int(blur_ksize) if int(blur_ksize) % 2 else int(blur_ksize) + 1
        img = cv2.GaussianBlur(img, (k, k), blur_sigma)
    if apply_noise and noise_sigma > 0:  # Add some realistic noise
        img = np.clip(img + np.random.normal(0, noise_sigma, img.shape), 0, 1)
    return img

# Convert labels to a grayscale image for visualization
def labels_to_grayscale_png(label_map: np.ndarray, boundary_mask: np.ndarray) -> np.ndarray:
    """Convert a label map to a grayscale image, highlighting boundaries.
    
    Args:
        label_map (np.ndarray): Voronoi label map.
        boundary_mask (np.ndarray): Binary boundary mask.
    
    Returns:
        np.ndarray: Grayscale image with boundaries set to 0.
    """
    lab = label_map.astype(np.int32)
    out = np.zeros_like(lab, dtype=np.uint8)
    max_id = int(lab.max())
    if max_id > 0:
        scaled = np.ceil(lab.astype(np.float32) * (255.0 / max_id)).astype(np.int32)
        scaled = np.clip(scaled, 1, 255).astype(np.uint8)
        out[lab > 0] = scaled[lab > 0]
    if boundary_mask is not None:
        out[boundary_mask.astype(bool)] = 0  # Boundaries are black
    return out

# GAN Models— image refiners
class ResBlock(nn.Module):
    def __init__(self, ch):
        """A residual block for the GAN generator—helps with deep networks!
        
        Args:
            ch (int): Number of input/output channels.
        """
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1)  # First convolution
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)  # Second convolution
        self.n1 = nn.InstanceNorm2d(ch, affine=True)  # Normalize after conv
        self.n2 = nn.InstanceNorm2d(ch, affine=True)

    def forward(self, x):
        h = F.relu(self.n1(self.c1(x)))  # Activate and normalize
        h = self.n2(self.c2(h))  # Another layer
        return F.relu(h + x)  # Add the input back (residual connection)

class RefinerG(nn.Module):
    def __init__(self, in_ch=1):
        """Generator network to refine synthetic images into SEM-like ones.
        
        Args:
            in_ch (int): Number of input channels (default: 1 for grayscale).
        """
        super().__init__()
        c = 64  # Base channel size
        self.c1 = nn.Conv2d(in_ch, c, 7, 1, 3)  # Initial conv
        self.d1 = nn.Conv2d(c, c*2, 3, 2, 1)  # Downsample
        self.d2 = nn.Conv2d(c*2, c*4, 3, 2, 1)  # Further downsample
        self.res = nn.Sequential(*[ResBlock(c*4) for _ in range(6)])  # Six residual blocks
        self.u1 = nn.ConvTranspose2d(c*4, c*2, 4, 2, 1)  # Upsample
        self.u2 = nn.ConvTranspose2d(c*2, c, 4, 2, 1)  # Final upsample
        self.out = nn.Conv2d(c, in_ch, 7, 1, 3)  # Output layer

    def forward(self, x):
        h = F.relu(self.c1(x))  # Start with a ReLU
        h = F.relu(self.d1(h))  # Downsample
        h = F.relu(self.d2(h))  # More downsampling
        h = self.res(h)  # Run through residual blocks
        h = F.relu(self.u1(h))  # Upsample back
        h = F.relu(self.u2(h))  # Final upsample
        y = torch.tanh(self.out(h))  # Tanh to [-1, 1], then shift to [0, 1]
        return (y + 1) / 2.0

class PatchD(nn.Module):
    def __init__(self, in_ch=1):
        """Discriminator network to judge real vs. fake SEM patches.
        
        Args:
            in_ch (int): Number of input channels (default: 1).
        """
        super().__init__()
        def blk(ci, co, norm=True):
            layers = [nn.Conv2d(ci, co, 4, 2, 1, bias=not norm)]
            if norm: layers.append(nn.InstanceNorm2d(co, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        c = 64
        self.m = nn.Sequential(
            blk(in_ch, c, norm=False),  # First conv without norm
            blk(c, c*2),               # Downsample
            blk(c*2, c*4),             # More downsampling
            nn.Conv2d(c*4, 1, 3, 1, 1),  # Final conv to score
        )

    def forward(self, x): return self.m(x)  # Pass through the network

# Helper function to compute edge maps
def sobel_edges(img):
    """Compute edge strength using Sobel filters.
    
    Args:
        img (torch.Tensor): Input image tensor.
    
    Returns:
        torch.Tensor: Edge strength map.
    """
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=img.device).view(1,1,3,3)/8.0
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=img.device).view(1,1,3,3)/8.0
    gx = F.conv2d(img, kx, padding=1); gy = F.conv2d(img, ky, padding=1)
    return torch.sqrt(gx*gx + gy*gy + 1e-8)  # Add epsilon to avoid division issues

# Datasets for GAN training
class RealPatchDataset(Dataset):
    def __init__(self, dir_path):
        """Dataset for real SEM patches.
        
        Args:
            dir_path (str): Directory containing patch images.
        """
        self.files = sorted(glob.glob(os.path.join(dir_path, "*.png")))  # List all PNG files

    def __len__(self):
        return len(self.files)  # Total number of patches

    def __getitem__(self, idx):
        g = cv2.imread(self.files[idx], cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
        return torch.from_numpy(g).unsqueeze(0)  # Return as [1, H, W] tensor

class SynthOnTheFly(Dataset):
    def __init__(self, n=SYNTH_TRAIN_SAMPLES, seed_range=VOR_SEED_RANGE, thickness=BOUNDARY_THICKNESS):
        """Dataset generating synthetic images on the fly.
        
        Args:
            n (int): Number of synthetic samples.
            seed_range (tuple): Range for number of Voronoi seeds.
            thickness (int): Boundary thickness.
        """
        self.n = n; self.seed_range = seed_range; self.thickness = int(thickness)

    def __len__(self): return self.n

    def __getitem__(self, idx):
        H = W = PATCH_SIZE
        n_seeds = np.random.randint(self.seed_range[0], self.seed_range[1]+1)
        seeds = generate_seeds(n_seeds, H, W)
        labels = voronoi_label_map_periodic(seeds, H, W)
        bmask = boundary_from_labels_periodic(labels, thickness=self.thickness)
        x = render_from_mask(bmask)
        return torch.from_numpy(x).unsqueeze(0).float()

# Generate a synthetic image-mask pair
def synth_pair(H, W):
    """Generate a single synthetic image and its boundary mask.
    
    Args:
        H (int): Height of the image.
        W (int): Width of the image.
    
    Returns:
        tuple: (synthetic_render, boundary_mask, label_map).
    """
    n_seeds = np.random.randint(VOR_SEED_RANGE[0], VOR_SEED_RANGE[1]+1)
    seeds = generate_seeds(n_seeds, H, W)
    seeds += np.random.normal(0, 5, seeds.shape)  # Add some noise for variety
    label_map = voronoi_label_map_periodic(seeds, H, W)
    bmask = boundary_from_labels_periodic(label_map, thickness=BOUNDARY_THICKNESS)
    if float(BOUNDARY_JITTER_P) > 0 and np.random.rand() < float(BOUNDARY_JITTER_P):
        k = np.ones((3,3), np.uint8)
        if np.random.rand() < 0.5: bmask = cv2.dilate(bmask, k, 1)
        else: bmask = cv2.erode(bmask, k, 1)
        bmask = (bmask > 0).astype(np.uint8)
    synthetic_render = render_from_mask(bmask)
    return synthetic_render, bmask, label_map

# Save a panel with synthetic render, labels, and boundary mask
def save_panel(stem, synthetic_render01, labels_gray, bmask01):
    """Save a side-by-side panel of synthetic render, labels, and boundary mask.
    
    Args:
        stem (str): Base filename.
        synthetic_render01 (np.ndarray): Synthetic image.
        labels_gray (np.ndarray): Grayscale label map.
        bmask01 (np.ndarray): Binary boundary mask.
    """
    import matplotlib.pyplot as plt
    sr = (np.clip(synthetic_render01, 0, 1) * 255).astype(np.uint8)
    lg = labels_gray
    bm = (bmask01 * 255).astype(np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(sr, cmap='gray')
    axes[0].set_title('Synthetic Render')
    axes[0].axis('off')
    axes[1].imshow(lg, cmap='gray')
    axes[1].set_title('Label Map (Grayscale)')
    axes[1].axis('off')
    axes[2].imshow(bm, cmap='gray')
    axes[2].set_title('Boundary Mask')
    axes[2].axis('off')
    plt.tight_layout()
    out_path = os.path.join(DATASET_RENDER_DIR, stem.replace(".png", "_panel.png"))
    plt.savefig(out_path)
    plt.close(fig)

# Main function to run the data generation process
def data_generation():
    """Generate synthetic data, extract real patches, and train the GAN.

    This function orchestrates the entire data preparation pipeline, from patch extraction
    to synthetic data creation and GAN refinement.
    """
    start_time = time.time()  # Start timing

    print(f"[data_generator] Device: {DEVICE}")

    # 1) Extract patches from the real SEM image
    sem = load_sem(SEM_PATH)
    extract_real_patches(sem, patch=PATCH_SIZE, stride=STRIDE)

    # 2) Set up datasets and loaders for GAN training
    real_ds  = RealPatchDataset(REAL_PATCH_DIR)
    synth_ds = SynthOnTheFly(SYNTH_TRAIN_SAMPLES)
    if len(real_ds) == 0:
        raise RuntimeError(f"No real patches found in {REAL_PATCH_DIR}. Ensure SEM exists at {SEM_PATH}.")
    real_loader  = DataLoader(real_ds,  batch_size=BATCH_GAN, shuffle=True, drop_last=True, num_workers=2)
    synth_loader = DataLoader(synth_ds, batch_size=BATCH_GAN, shuffle=True, drop_last=True, num_workers=2)

    real_iter = iter(real_loader)  # Infinite iterator for real data
    def next_real():
        nonlocal real_iter
        try: return next(real_iter)
        except StopIteration:
            real_iter = iter(real_loader)
            return next(real_iter)

    print(f"[data_generator] Real patches: {len(real_ds)} | Synth/epoch: {len(synth_ds)}")

    # 3) Initialize GAN models
    G = RefinerG().to(DEVICE)  # Generator
    D = PatchD().to(DEVICE)    # Discriminator
    optG = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5,0.999))
    optD = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5,0.999))

    # 4) Train the GAN (optional)
    if USE_GAN:
        G.train(); D.train()
        d_losses, g_losses = [], []
        d_real_outputs, d_fake_outputs = [], []
        for epoch in range(1, EPOCHS_GAN+1):
            pbar = tqdm(synth_loader, desc=f"GAN Epoch {epoch}/{EPOCHS_GAN}")
            for x_synth in pbar:
                x_synth = x_synth.to(DEVICE)
                y_real  = next_real().to(DEVICE)
                with torch.no_grad():
                    y_fake = G(x_synth)
                d_real = D(y_real)
                d_fake = D(y_fake)
                loss_d = torch.mean((d_real - 1)**2) + torch.mean((d_fake - 0)**2)  # LSGAN loss
                optD.zero_grad(); loss_d.backward(); optD.step()

                y_fake = G(x_synth)
                d_fake = D(y_fake)
                loss_g_gan = torch.mean((d_fake - 1)**2)
                e_x = sobel_edges(x_synth)
                e_y = sobel_edges(y_fake)
                loss_edge = torch.mean(torch.abs(e_x - e_y))  # Preserve edges
                y_id = G(y_real)
                loss_id = torch.mean(torch.abs(y_id - y_real))  # Identity preservation
                loss_g = loss_g_gan + LAMBDA_E*loss_edge + LAMBDA_ID*loss_id
                optG.zero_grad(); loss_g.backward(); optG.step()

                # Record losses and outputs
                d_losses.append(loss_d.item())
                g_losses.append(loss_g.item())
                d_real_outputs.append(d_real.mean().item())
                d_fake_outputs.append(d_fake.mean().item())

                pbar.set_postfix(d=float(loss_d.item()), g=float(loss_g.item()), edge=float(loss_edge.item()))

            # Save sample images
            with torch.no_grad():
                ys = (y_fake[:4].cpu().numpy()*255).astype(np.uint8)
                for i, s in enumerate(ys):
                    cv2.imwrite(os.path.join(GAN_SAMPLES_DIR, f"e{epoch:03d}_{i}.png"), s[0])
            torch.save(G.state_dict(), REFINER_CKPT)

        # Plot and save discriminator performance
        plt.figure(figsize=(10, 5))
        plt.plot(d_losses, label='Discriminator Loss', color='orange')
        plt.plot(g_losses, label='Generator Loss', color='blue')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss vs Iteration')
        plt.legend()
        plt.savefig(os.path.join(GAN_SAMPLES_DIR, 'loss_vs_iteration.png'))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(d_real_outputs, label='D(x) - Real', color='blue')
        plt.plot(d_fake_outputs, label='D(G(z)) - Fake', color='orange')
        plt.xlabel('Iteration')
        plt.ylabel('Discriminator Output')
        plt.title('Discriminator Output vs Iteration')
        plt.legend()
        plt.savefig(os.path.join(GAN_SAMPLES_DIR, 'discriminator_output_vs_iteration.png'))
        plt.close()

        print(f"[data_generator] GAN training done. Generator saved -> {REFINER_CKPT}")
    else:
        print("[data_generator] USE_GAN=False: Skipping GAN training.")

    # 5) Generate and save the final synthetic pairs
    G.eval()
    with torch.no_grad():
        for i in tqdm(range(FINAL_PAIRS), desc="Synthesizing pairs"):
            H = W = IMG_SIZE
            synthetic_render, bmask, label_map = synth_pair(H, W)
            inp = torch.from_numpy(synthetic_render)[None, None].float().to(DEVICE)
            sem_like = G(inp).cpu().numpy()[0, 0] if USE_GAN else synthetic_render
            stem = f"img_{i:04d}.png"
            cv2.imwrite(os.path.join(DATASET_IMG_DIR, stem), (np.clip(sem_like,0,1)*255).astype(np.uint8))
            cv2.imwrite(os.path.join(DATASET_RENDER_DIR, stem), (np.clip(synthetic_render,0,1)*255).astype(np.uint8))
            cv2.imwrite(os.path.join(DATASET_MSK_DIR, stem), (bmask*255).astype(np.uint8))
            labels_gray = labels_to_grayscale_png(label_map, bmask)
            cv2.imwrite(os.path.join(DATASET_LABEL_GRAY_DIR, stem), labels_gray)
            save_panel(stem, synthetic_render, labels_gray, bmask)

    print(f"[data_generator] Synthesized {FINAL_PAIRS} pairs.")
    print("  Images (train input)   :", DATASET_IMG_DIR)
    print("  Masks  (train target)  :", DATASET_MSK_DIR)
    print("  synthetic_render (ref) :", DATASET_RENDER_DIR)
    print("  Labels gray (viz)      :", DATASET_LABEL_GRAY_DIR)

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    print(f"[data_generator] Total time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    data_generation()