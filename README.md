It Takes Two: A Dual-View Decoding Network for Road Extraction

Status: Manuscript under submission

This repository contains information and demo links for DVDNet, a dual-view decoding network for road extraction from high-resolution remote sensing images.

Demo Video 1 – Model Performance Comparison Visualization

Watch Demo1: https://youtu.be/_yUiXvgdzcE

Demo Video 2 – Using Existing Models on Randomly Captured Google Maps Images

Watch Demo2: https://youtu.be/JqB4Uaupq64


⚠️ Note: The associated manuscript is currently under submission. The code will be released after the paper is accepted.

⚠️ Note: In demo videos or early code, the network name DDMDNet may appear; this is the previous name before renaming to DVDNet.


Contact

Renbao Lian (Corresponding Author): luoshao@163.com


 DVDNet: A Dual-View Decoding Network for Road Extraction

Official PyTorch implementation of:

> **DVDNet: A Dual-View Decoding Network for Road Extraction**  
> Huayang Zhang, Shaohua Zheng, Renbao Lian*  
> * Corresponding author. Email: lrb@fjjxu.edu.cn

## Overview

DVDNet is a deep learning framework for road extraction from high-resolution remote sensing imagery. It is designed to improve the balance between local detail preservation and global contextual understanding through three key components:

- **Multi-Branch Feature Guidance Module (MFGM):** Extracts multi-scale, direction-sensitive, and context-enhanced features through three parallel branches.
- **Heterogeneous Dual-View Collaborative Decoder (HDCD):** Separates local geometric refinement and global contextual reasoning into two complementary decoding paths: the **Local Refinement Path (LRP)** and the **Global Coherence Path (GCP)**.
- **Path Importance Self-Evaluation Module (PISM):** Adaptively fuses the representations from the two decoding paths using sample-dependent weights.

## Project Structure

```text
DVDNet/
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── framework.py                # Training framework (ModelContainer)
├── loss.py                     # Dice + BCE loss
├── data.py                     # Dataset loaders
├── networks/
│   └── DVDNet.py               # DVDNet model definition
├── utils/                      # Learning-rate schedulers
│   ├── cosine_lr.py
│   ├── plateau_lr.py
│   ├── scheduler.py
│   ├── scheduler_factory.py
│   ├── step_lr.py
│   └── tanh_lr.py
├── dataset/                    # Dataset directories
│   ├── deepglobe/
│   └── roadtracer_mydata/
├── weights/                    # Trained model weights
├── logs/                       # Training logs
├── results/                     #output segmentation masks
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

- Python 3.10 or later
- PyTorch 2.8 or later
- CUDA 11.8 or later (recommended)
- NVIDIA GPU with at least 16 GB of memory

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Huayang122/DVDNet.git
cd DVDNet
pip install -r requirements.txt
```

## Dataset Preparation

### DeepGlobe Dataset

DVDNet supports the **DeepGlobe Road Extraction Challenge** dataset.

After downloading and preprocessing the dataset, organize the files as follows:

```text
dataset/deepglobe/
├── train/
│   ├── {id}_sat.jpg
│   └── {id}_mask.png
├── train.csv
├── val.csv
└── test.csv
```

The CSV files contain the image IDs used for the corresponding data splits.

### RoadTracer Dataset

For RoadTracer, we use the preprocessed version provided by Wang et al. in their work on the Local-Global Sparse Transformer.

Organize the dataset as follows:

```text
dataset/roadtracer_mydata/
├── 1024x1024/
│   ├── {id}.bmp
│   └── {id}_mask.bmp
├── train_list_1024.txt
├── val_list_1024.txt
└── test_list_1024.txt
```

### Dataset Split Files

For datasets requiring explicit split files, create text files containing the image IDs (without file extensions) for the training, validation, and test sets.

Please ensure that the dataset directory structure and split files are consistent with the paths expected by the corresponding dataset loader in `data.py`.

## Usage

### Training

To train DVDNet on the DeepGlobe dataset:

```bash
python train.py
```

To train DVDNet on the RoadTracer dataset, modify the dataset functions in `train.py`:

```python
train_dataset_method = get_roadtrace_trainset
val_dataset_method = get_roadtrace_valset
test_dataset_method = get_roadtrace_testset
```

Then run:

```bash
python train.py
```

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py
```

Modify the model weight path in `evaluate.py` as needed:

```python
metrics_eval(
    'DVDNet',
    get_deepglobe_testset,
    'your_model_weight.pt'
)
```

The evaluation script reports the quantitative performance metrics implemented in the repository.

## FLOPs and Parameter Count

The evaluation script also includes an automatic computational complexity analysis. To obtain the number of parameters and FLOPs, run:

```bash
python evaluate.py
```

The corresponding complexity evaluation is executed automatically through `param_gflops_eval`.


## Code and Data Availability

The source code will be publicly available at:

https://github.com/Huayang122/DVDNet

An archived version of this code will also be deposited in Zenodo for long-term preservation and DOI assignment.

The DeepGlobe and RoadTracer datasets are publicly available benchmark datasets and are not redistributed with this repository. Users should obtain the datasets from their respective official sources and comply with the applicable dataset licenses and terms of use.

This Zenodo deposit preserves a fixed snapshot of the source code used in the experiments reported in this paper.
Ongoing bug fixes and future developments will be maintained in the GitHub repository.


## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

The learning-rate scheduler implementation is based on [timm](https://github.com/huggingface/pytorch-image-models) by Ross Wightman and is used in accordance with its Apache 2.0 license.
