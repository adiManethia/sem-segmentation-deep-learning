# Project Overview

This repository contains the implementation of a 23-week Master's Personal Programming Project completed at TU Bergakademie Freiberg.

## Objective

The project investigates automated segmentation of grain boundaries in Scanning Electron Microscopy (SEM) images using both classical image-processing methods and deep learning.

## Main Components

- Classical segmentation using K-Means, ISODATA, and SLIC
- Synthetic microstructure generation using Voronoi tessellation
- GAN-based refinement of synthetic images
- Deep-learning segmentation with U-Net and Attention U-Net
- Tiled inference for large images
- Grain statistics and segmentation evaluation

## Public Repository Notes

The original SEM images used in the academic project are not included in this repository because they were provided for academic use only.

To make the repository publicly shareable, this version includes:
- source code,
- historical metrics,
- public synthetic demonstration figures,
- documentation of the methodology and limitations.

## Important Evaluation Clarifications

- Reported IoU and Dice scores are historical results from synthetic test data.
- Real SEM evaluation in the original academic work focused on qualitative results and grain statistics, not manually annotated pixel-level real-image ground truth.
- No retraining was performed for this public release.
