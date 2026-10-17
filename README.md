# SEM Image Segmentation with Synthetic Data and Deep Learning

**Classical image processing · GAN-based synthetic data · U-Net · Attention U-Net · PyTorch**

A 23-week Master's Personal Programming Project completed at **TU Bergakademie Freiberg**. This project investigates automatic grain-boundary segmentation in scanning electron microscopy (SEM) images, comparing classical segmentation methods with deep learning trained on synthetically generated image–mask pairs.

> **Public release:** This repository contains source code, independently generated synthetic demonstrations, and historical experimental metrics. The original SEM images, restricted datasets, trained checkpoints, and full academic report are **not** distributed.

## Highlights

- Built classical segmentation baselines using **K-Means, ISODATA, and SLIC**, with filtering, contrast enhancement, morphology, and grain statistics.
- Generated labelled microstructure images with **periodic Voronoi tessellation** and refined synthetic appearance using a **GAN with edge-preservation objectives**.
- Implemented **U-Net and Attention U-Net** for pixel-wise grain-boundary prediction, including tiled inference for larger images.
- Compared segmentation quality and processing time, while documenting the limits of the original evaluation.

## Workflow

```text
Classical path
SEM image → preprocessing → K-Means / ISODATA / SLIC → grain statistics

Deep-learning path
Voronoi geometry → synthetic image + exact boundary mask
                  ↓
        GAN-based image refinement
                  ↓
       synthetic image–mask dataset
                  ↓
          U-Net / Attention U-Net
                  ↓
   tiled inference → boundary mask → grain analysis
```

The **standalone public demos below** do not use the private SEM images or reproduce the GAN-trained models. They are provided to illustrate the segmentation concepts safely.

## Visual demonstrations

### 1. Synthetic microstructure and ground truth

![Independently generated Voronoi microstructure and exact boundary mask](docs/figures/synthetic_voronoi_demo.png)

A new Voronoi microstructure and its mathematically defined grain-boundary mask. **Public synthetic demo; not an original experimental SEM image.**

### 2. Classical segmentation on synthetic data

![K-Means segmentation of a synthetic microstructure](docs/figures/classical_segmentation_demo.png)

An independent synthetic example showing K-Means clustering and a boundary mask derived from cluster labels. This visualization is illustrative rather than a reproduction of the original ISODATA experiment.

### 3. Historical synthetic-test performance

![Historical IoU and Dice comparison](docs/figures/model_accuracy.png)

| Model | IoU | Dice |
|---|---:|---:|
| U-Net | **0.8965** | **0.9454** |
| Attention U-Net | 0.8957 | 0.9450 |

These are **historical synthetic-test metrics** from the original academic project, **not** accuracy measured against independently annotated real SEM images. The public release has not rerun the original evaluation.

### 4. Historical processing-time comparison

![Historical segmentation processing times](docs/figures/inference_comparison.png)

| Method | Time per image |
|---|---:|
| ISODATA | 24.12 s |
| U-Net | 6.59 s |
| Attention U-Net | 7.31 s |

Reported for a **1387 × 2048** SEM image in the original project. In that comparison, U-Net was approximately **3.7× faster** than ISODATA. These measurements are historical and are not independently benchmarked in this release.

## Repository structure

```text
sem-segmentation-deep-learning/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── Deep_learning_Pipeline/
│   ├── config.py                          # Data paths and experiment settings
│   ├── data_generation.py                 # Voronoi + GAN data generation
│   ├── data_loader.py                     # Paired image/mask loaders and splits
│   ├── models.py                          # U-Net and Attention U-Net
│   ├── train.py                           # Training and synthetic-test evaluation
│   ├── inference.py                       # Tiled inference on SEM images
│   ├── evaluate_models_on_unseen_sem.py    # Qualitative unseen-image evaluation
│   └── segmentation_stats_generator.py    # Grain statistics and visualizations
│
├── Segmentation_with_traditional_methods/
│   ├── sem_segmentation.py                # Real-image classical pipeline
│   ├── scale_extraction.py                # Interactive SEM scale-bar measurement
│   └── voronoi_test/
│       ├── voronoi_gen.py                 # Periodic Voronoi test images and labels
│       └── segmentation.py                # K-Means / ISODATA / SLIC comparison
│
├── test_cases/
│   ├── gan_justification.py               # GAN identity/preservation experiment
│   └── test_metric.py                     # Metric sanity-check script
│
├── scripts/
│   └── generate_public_assets.py          # Private-data-free README figures
│
└── docs/
    ├── PROJECT_OVERVIEW.md
    └── figures/
        ├── synthetic_voronoi_demo.png
        ├── classical_segmentation_demo.png
        ├── model_accuracy.png
        └── inference_comparison.png
```

Private images, generated training sets, checkpoints, and experiment output directories are intentionally excluded from Git. Some paths used by the original scripts are created locally when they run.

## Getting started

### Requirements

- Python and `pip` (a modern Python 3 environment is recommended)
- The packages listed in [`requirements.txt`](requirements.txt)
- **No GPU, private SEM data, or trained checkpoints are required for the public demonstrations.**

```bash
git clone https://github.com/adiManethia/sem-segmentation-deep-learning.git
cd sem-segmentation-deep-learning

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> The original project dependencies are not version-locked. Install PyTorch using the appropriate platform-specific instructions if the default installation is unsuitable for your system.

### Option A — Recreate all public README figures

From the repository root:

```bash
python scripts/generate_public_assets.py
```

Outputs are written to `docs/figures/`. The script creates fresh Voronoi/K-Means illustrations and plots the **recorded historical** accuracy and timing numbers. It does **not** train a GAN, run a U-Net, or access any SEM dataset.

### Option B — Run the original classical Voronoi benchmark

The original scripts support a separate, synthetic-only experiment:

```bash
cd Segmentation_with_traditional_methods/voronoi_test
python voronoi_gen.py
# When prompted, for example enter 50 seeds and an image size of 256.
python segmentation.py
```

`voronoi_gen.py` creates a periodic synthetic image and its label map in `voronoi_generation/`. The second script evaluates K-Means, ISODATA, and SLIC on those outputs and writes comparison results there. This is a **new local run**, not a replication of the historical neural-network metrics.

### Option C — Explore the original deep-learning workflow (advanced)

The source files document the original sequence:

| Stage | Script | External inputs needed |
|---|---|---|
| Configuration | `Deep_learning_Pipeline/config.py` | Adjust local paths, image names, parameters |
| GAN refinement and synthetic-pair creation | `data_generation.py` | An SEM image you are authorized to use |
| U-Net / Attention U-Net training | `train.py` | Generated image–mask pairs |
| Model inference | `inference.py` | A valid checkpoint and authorized input image |
| Grain statistics | `segmentation_stats_generator.py` | Inference outputs |

The deep-learning scripts use relative paths beneath `grain_segmentation/` and are intended to be run **from inside `Deep_learning_Pipeline/`**. The default image names and configuration refer to the original local academic workflow. Users must supply **their own properly licensed data**, adjust `config.py`, and produce or supply compatible trained checkpoints.

**This is a research-code release, not a turnkey pretrained inference application.** Running `train.py` trains a model; this step is not necessary to view or reproduce the public demos.

## Methods and technologies

| Area | Methods / tools |
|---|---|
| Classical vision | Gaussian/bilateral filtering, CLAHE, K-Means, ISODATA, SLIC, morphology |
| Synthetic data | Voronoi tessellation, periodic boundaries, augmentation |
| Generative modelling | GAN refiner, adversarial and edge-preservation losses |
| Deep learning | PyTorch, U-Net, Attention U-Net, BCE–Dice loss |
| Evaluation | IoU, Dice, grain statistics, processing time |
| Libraries | OpenCV, NumPy, SciPy, scikit-image, scikit-learn, Matplotlib, Pandas |

## Evaluation notes and limitations

The public results should be interpreted with the following qualifications:

1. **Synthetic versus real images:** IoU and Dice were computed on generated image–mask data. The unseen real-SEM demonstrations were qualitative and did not establish pixel-level accuracy using independently annotated ground truth.
2. **Model comparison:** The original U-Net and Attention U-Net runs did not use a documented identical fixed random split, so the scores should not be interpreted as a controlled architectural ablation.
3. **Checkpoint selection:** Original test-time metrics were computed using the final model state, rather than explicitly reloading the best-validation checkpoint.
4. **Grain statistics:** The submitted report's median-diameter column contained values corresponding to standard deviations. A grain-count calculation was corrected in the public code copy; historical results were **not** recalculated.
5. **Reproducibility:** Private inputs, checkpoints, environment lockfiles, and exact historical run conditions are not public. Results are provided for documentation and were not independently reproduced for this release.

See [Project Overview](docs/PROJECT_OVERVIEW.md) for a concise technical summary.

## Data access and project report

The original SEM images were provided for academic use only. To respect those restrictions, **original images, derived private datasets, checkpoints, and the full report containing restricted images are not distributed publicly**. The demonstration images in this repository were generated independently and do not require access to the original SEM data.

## License

Source code released by the copyright holder is provided under the [MIT License](LICENSE). This license does **not** grant rights to third-party microscopy images, restricted datasets, or other material not owned by the author. Third-party credits in the source code remain applicable.

---

*Originally developed as a Master's Personal Programming Project; published as a source-code and documentation portfolio release without retraining the models.*
