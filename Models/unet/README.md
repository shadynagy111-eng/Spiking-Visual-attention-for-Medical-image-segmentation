# UNet Architecture for Medical Image Segmentation

This folder contains the implementation of the UNet architecture for medical image segmentation.

## Overview

UNet is a convolutional neural network that was developed for biomedical image segmentation. The network architecture is illustrated below:

![UNet Architecture](./../../assets/unet_arch.png)

## Features

- **Encoder-Decoder Structure**: The network consists of a contracting path (encoder) and an expansive path (decoder) which gives it the u-shaped architecture.
- **Skip Connections**: Skip connections between the encoder and decoder help in retaining spatial information.

## Requirements

- Python 3.x
- PyTorch
- Albumentations
- tqdm

## Usage

### Training

To train the model, run the following command:

```bash
python train.py
```

### Testing

To test the model, run the following command:

```bash
python test.py
```

## Directory Structure

```
├── Models
│   ├── unet
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── README.md
│   └── ...
├── assets
│   ├── unet_arch.png
└── ...
```

## References

- [Original UNet Paper](https://arxiv.org/abs/1505.04597)
