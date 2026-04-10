# Cross-Disaster Building Damage Assessment from Satellite Imagery Using Two-Step and Bi-Temporal Models on xBD

This repository contains code and notebooks for building damage assessment on the **xBD** dataset using two complementary approaches:

- a **two-step pipeline** that first localizes buildings and then classifies damage at the building level
- a **bi-temporal end-to-end model** that directly predicts dense damage maps from paired pre- and post-disaster imagery

The project studies both **in-domain performance** and **cross-disaster generalization**, with a particular focus on held-out disaster settings.

## Project Overview

Post-disaster building damage assessment from satellite imagery is an important problem for emergency response and large-scale situational awareness. This project compares two modeling strategies on xBD:

### Model 1: Two-step pipeline
1. Binary building localization from the pre-disaster image
2. Building proposal extraction using thresholding and connected components
3. Contextual pre/post crop generation with binary object masks
4. Building-level damage classification with a Siamese EfficientNet-based classifier

### Model 2: Bi-temporal end-to-end model
1. Paired pre- and post-disaster imagery as input
2. Siamese U-Net-style damage segmentation
3. Dense 5-class damage prediction
4. Building-level assessment derived from the predicted masks

## Main Contents

### `notebooks/`
Main experimental notebooks:

- `01_model1_in_domain_two_step.ipynb`  
  Final in-domain two-step pipeline

- `02_model1_cross_disaster_two_step.ipynb`  
  Cross-disaster two-step pipeline with a held-out disaster type

- `03_model2_in_domain_bitemporal.ipynb`  
  Final in-domain bi-temporal segmentation model

- `04_model2_cross_disaster_bitemporal.ipynb`  
  Cross-disaster bi-temporal segmentation model

### `scripts/`
Utility scripts for dataset preparation:

- `build_metadata.py`  
  Builds metadata from xBD files

- `make_splits.py`  
  Creates train/validation/test split CSV files

## Experimental Focus

The repository is organized around four main experimental settings:

- **Model 1 / In-domain**
- **Model 1 / Cross-disaster**
- **Model 2 / In-domain**
- **Model 2 / Cross-disaster**

This setup makes it possible to compare not only the two model families, but also their behavior under disaster shift.

## Key Ideas Explored

- building-aware crop sampling for localization
- predicted-object crop generation for realistic two-step training
- Siamese pre/post building classification
- localization pretraining for the end-to-end model
- class-imbalance handling through loss design and crop selection
- tiled full-image inference and building-level assessment
- held-out disaster evaluation for robustness analysis

## Dataset

Experiments are based on the **xBD** dataset for building damage assessment from satellite imagery.

## License

This project is licensed under the [MIT License](LICENSE).
