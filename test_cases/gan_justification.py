"""
Perfect Circle Identity cGAN — With BCE recon, warm-up, best checkpoint, and ablation
- Input: perfect circle image (binary 0/1), noise-free
- Target: same image
- Goals:
  1) Achieve near-exact copy (IoU~1, high edge sharpness)
  2) Provide simple ablation (recon-only vs GAN+BCE)
  3) Show edge sharpness advantage
Outputs:
- samples/samples_epoch_XXX.png
- plots/curves.png
- results/summary.json
- models/generator_best.pth (best by val IoU)
"""

import os, json
from datetime import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# -------------------- Config --------------------
OUT_DIR = "gan_justification_tests/perfect_circle_ablation"
IMG_SIZE = 128
RADIUS = 40
NUM = 100           # dataset size
EPOCHS = 80
WARMUP_EPOCHS = 10      
BATCH = 32
LR = 2e-4              # LR for optimizers
PIXEL_W = 50.0         # BCE recon weight (can go 50-200)
BINARIZE_EVAL = True   # evaluate metrics on thresholded output (0.5)
ABLATION_MODE = "gan_bce"  # "gan_bce" or "recon_only", training mode
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_ITERS = [1, 10, 20, 30, 40, 50, 60]

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------- Dirs ----------------------
os.makedirs(f"{OUT_DIR}/samples", exist_ok=True)
os.makedirs(f"{OUT_DIR}/results", exist_ok=True)
os.makedirs(f"{OUT_DIR}/models", exist_ok=True)
os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)

# -------------------- Data ----------------------
def perfect_circle(size=IMG_SIZE, radius=RADIUS):
    img = np.zeros((size, size), dtype=np.float32)
    cv2.circle(img, (size//2, size//2), int(radius), 1.0, -1)
    return img

X = np.stack([perfect_circle() for _ in range(NUM)], axis=0)
Y = X.copy()
n_train = int(0.7 * NUM)
n_val   = int(0.15 * NUM)
train_X, train_Y = X[:n_train], Y[:n_train]
val_X,   val_Y   = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
test_X,  test_Y  = X[n_train+n_val:], Y[n_train+n_val:]

class NPDs(torch.utils.data.Dataset):
    def __init__(self, X, Y): self.X, self.Y = X, Y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]).unsqueeze(0).float(), torch.from_numpy(self.Y[i]).unsqueeze(0).float()

train_loader = torch.utils.data.DataLoader(NPDs(train_X, train_Y), batch_size=BATCH, shuffle=True)
val_loader   = torch.utils.data.DataLoader(NPDs(val_X, val_Y), batch_size=BATCH, shuffle=False)

# -------------------- Models --------------------
class TinyGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.c2 = nn.Conv2d(32, 64, 4, 2, 1)   
        self.c3 = nn.Conv2d(64, 128, 4, 2, 1)  
        self.r  = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1)
        )
        self.u1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.u2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.out = nn.Conv2d(32, 1, 3, 1, 1)
    def forward(self, x):
        h1 = F.relu(self.c1(x))
        h2 = F.relu(self.c2(h1))
        h3 = F.relu(self.c3(h2))
        hr = F.relu(self.r(h3) + h3)
        u1 = F.relu(self.u1(hr))
        u2 = F.relu(self.u2(u1))
        y  = self.out(u2)  # logits 
        return y

class TinyDisc(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(ci, co): 
            return nn.Sequential(nn.Conv2d(ci, co, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.m = nn.Sequential(
            blk(1, 32),    # 64x64
            blk(32, 64),   # 32x32
            blk(64, 128),  # 16x16
            nn.Conv2d(128, 1, 3, 1, 1)  # 16x16 logits
        )
    def forward(self, x): return self.m(x)

G, D = TinyGen().to(DEVICE), TinyDisc().to(DEVICE)
optG = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
bce_logits = nn.BCEWithLogitsLoss()
bce_prob   = nn.BCELoss()

# -------------------- Metrics --------------------
def iou_bin(a, b, thr=0.5):
    A = (a > thr).astype(np.uint8); B = (b > thr).astype(np.uint8)
    inter = (A & B).sum(); union = (A | B).sum()
    return float(inter / (union + 1e-12))

def compare_pixel_counts(original, generated):
    """Compare white and black pixel counts between two images.

    Args:
        original (np.ndarray): Original binary image.
        generated (np.ndarray): Generated binary image.

    Returns:
        dict: Dictionary containing pixel count comparison results.
    """
    original_white = np.sum(original == 255)
    original_black = np.sum(original == 0)
    generated_white = np.sum(generated == 255)
    generated_black = np.sum(generated == 0)

    comparison = {
        "original_white": original_white,
        "original_black": original_black,
        "generated_white": generated_white,
        "generated_black": generated_black,
        "white_pixel_difference": abs(original_white - generated_white),
        "black_pixel_difference": abs(original_black - generated_black),
    }

    return comparison
 

def evaluate(model, Xn, Yn, binarize=BINARIZE_EVAL):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(Xn).unsqueeze(1).float().to(DEVICE)
        logits = model(x)
        y_hat = torch.sigmoid(logits).clamp(0, 1).cpu().numpy()
        y = torch.from_numpy(Yn).unsqueeze(1).float().cpu().numpy()
    if binarize:
        y_hat_eval = (y_hat > 0.5).astype(np.float32)
    else:
        y_hat_eval = y_hat
    ii = np.mean([iou_bin(y_hat_eval[i,0], y[i,0]) for i in range(y.shape[0])])
    return dict(iou=ii), y_hat
  

def save_samples(epoch, y_hat):
    x = test_X[:6]
    y = test_Y[:6]
    fig, axs = plt.subplots(3, 6, figsize=(12, 6))
    fig.suptitle(f"Epoch {epoch} — Input / Output / Target")
    for i in range(6):
        axs[0,i].imshow(x[i], cmap='gray', vmin=0, vmax=1); axs[0,i].axis('off'); axs[0,i].set_title("Input")
        axs[1,i].imshow(y_hat[i,0], cmap='gray', vmin=0, vmax=1); axs[1,i].axis('off'); axs[1,i].set_title("Output")
        axs[2,i].imshow(y[i], cmap='gray', vmin=0, vmax=1); axs[2,i].axis('off'); axs[2,i].set_title("Target")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/samples/samples_epoch_{epoch:03d}.png", dpi=180, bbox_inches='tight')
    plt.close()

# -------------------- Training --------------------
history = {"epoch": [], "g_loss": [], "d_loss": [], "iou": [], "edge_sharpness": []}
best_iou = -1e9
best_path = f"{OUT_DIR}/models/generator_best.pth"

print(f"Mode: {ABLATION_MODE} | Warm-up epochs: {WARMUP_EPOCHS} | BCE recon weight: {PIXEL_W} | Binarize eval: {BINARIZE_EVAL}")
for epoch in range(1, EPOCHS+1):
    G.train(); D.train()
    gls, dls = [], []

    for x, y in train_loader:
        x = x.to(DEVICE); y = y.to(DEVICE)

        # Forward
        logits_fake = G(x)                  # generator logits
        prob_fake  = torch.sigmoid(logits_fake)

        # Reconstruction loss (BCE on probabilities vs binary target)
        recon = bce_prob(prob_fake, y)

        if ABLATION_MODE == "recon_only" or epoch <= WARMUP_EPOCHS:
            # Train only G on recon
            loss_g = PIXEL_W * recon
            optG.zero_grad(); loss_g.backward(); optG.step()
            gls.append(float(loss_g.item()))
            # No D update
            dls.append(0.0)
        else:
            # Adversarial training
            # D step
            with torch.no_grad():
                logits_fake_detached = G(x)
                fake_detached = torch.sigmoid(logits_fake_detached)
            d_real = D(y)
            d_fake = D(fake_detached)
            loss_d = 0.5*(bce_logits(d_real, torch.ones_like(d_real)) + bce_logits(d_fake, torch.zeros_like(d_fake)))
            optD.zero_grad(); loss_d.backward(); optD.step()

            # G step
            d_fake = D(prob_fake)  # pass probabilities to D
            loss_g_adv = bce_logits(d_fake, torch.ones_like(d_fake))
            loss_g = loss_g_adv + PIXEL_W * recon
            optG.zero_grad(); loss_g.backward(); optG.step()

            gls.append(float(loss_g.item()))
            dls.append(float(loss_d.item()))

    # Validation
    val_metrics, val_yhat = evaluate(G, val_X, val_Y, binarize=BINARIZE_EVAL)
    history["epoch"].append(epoch)
    history["g_loss"].append(np.mean(gls))
    history["d_loss"].append(np.mean(dls))
    history.setdefault("iou", [])
    history["iou"].append(val_metrics["iou"])

    if epoch in SAMPLE_ITERS or epoch == EPOCHS:
      print(f"Epoch {epoch:03d} | G:{history['g_loss'][-1]:.4f} D:{history['d_loss'][-1]:.4f} | "
          f"IoU:{val_metrics['iou']:.4f}")
      save_samples(epoch, val_yhat)

    # Save best by IoU
    if val_metrics["iou"] > best_iou:
        best_iou = val_metrics["iou"]
        torch.save(G.state_dict(), best_path)

# -------------------- Test with best checkpoint --------------------
G_best = TinyGen().to(DEVICE)
G_best.load_state_dict(torch.load(best_path, map_location=DEVICE))
test_metrics, test_yhat = evaluate(G_best, test_X, test_Y, binarize=BINARIZE_EVAL)
save_samples(EPOCHS, test_yhat)  # final snapshot with best model too

# -------------------- Plots -----------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history["epoch"], history["g_loss"], label="G loss")
plt.plot(history["epoch"], history["d_loss"], label="D loss")
plt.title("Loss Curves"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True, alpha=0.3); plt.legend()

plt.subplot(1,2,2)
plt.plot(history["epoch"], history["iou"], label="IoU")
plt.title("Validation IoU"); plt.xlabel("Epoch"); plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plots/curves.png", dpi=200, bbox_inches='tight')
plt.close()

# -------------------- Summary ---------------------
summary = {
    "date": datetime.now().isoformat(),
    "config": {
        "mode": ABLATION_MODE,
        "img_size": IMG_SIZE,
        "radius": RADIUS,
        "epochs": EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "batch": BATCH,
        "lr": LR,
        "pixel_weight_bce": PIXEL_W,
        "binarize_eval": BINARIZE_EVAL
    },
    "validation_last": {
        "iou": history["iou"][-1]
    },
    "test_best_by_iou": test_metrics
}
with open(f"{OUT_DIR}/results/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Done. Outputs in:", OUT_DIR)
print("Best-checkpoint Test:", test_metrics)