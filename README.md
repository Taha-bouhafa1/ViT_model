# Vision Transformer (ViT) from Scratch
> Paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)

This repository contains a PyTorch implementation of the **Vision Transformer (ViT)** architecture, trained on the CIFAR-10 dataset. The project demonstrates how to build ViT from the ground up without using pretrained models or high-level libraries like `timm`.

![Vision Transformer Architecture](https://github.com/Taha-bouhafa1/ViT_model/blob/main/assets/Screenshot%202025-07-07%20212554.png)

![Vision Transformer Architecture](https://github.com/Taha-bouhafa1/ViT_model/blob/main/assets/The-Vision-Transformer-architecture-a-the-main-architecture-of-the-model-b-the.png)
  
---

##  Overview

The **Vision Transformer (ViT)** is a deep learning architecture introduced in the paper [*An Image is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929). Unlike traditional CNNs, ViT treats an image as a sequence of patches and applies the standard Transformer encoder, originally developed for NLP tasks, to process visual data.

---

##  Architecture Breakdown

The ViT model consists of the following main components:

| Component           | Description |
|---------------------|-------------|
| **Patch Embedding** | Splits the image into fixed-size patches and projects each patch into a vector (using Conv2D for efficient flattening + projection). |
| **CLS Token**       | A learnable token prepended to the sequence to represent the whole image. |
| **Position Embedding** | Learnable vectors added to the patch embeddings to encode positional information. |
| **Transformer Encoder** | A stack of standard Transformer blocks composed of multi-head self-attention and MLP layers. |
| **MLP Head**        | A final linear layer applied to the CLS token to classify the image. |

---

##  Dataset

| Property         | Value                     |
|------------------|---------------------------|
| Dataset          | CIFAR-10                  |
| Image Size       | 32 × 32                   |
| Number of Classes| 10                        |
| Training Samples | 50,000                    |
| Test Samples     | 10,000                    |

---

##  Model Configuration

| Hyperparameter   | Value     |
|------------------|-----------|
| Patch Size       | 4         |
| Embedding Dim    | 256       |
| Depth (Encoders) | 6         |
| Number of Heads  | 8         |
| MLP Dimension    | 512       |
| Dropout Rate     | 0.1       |
| Epochs           | 10        |
| Batch Size       | 128       |
| Learning Rate    | 3e-4      |
| Optimizer        | Adam      |

---

##  Results (After 10 Epochs)

| Metric           | Value     |
|------------------|-----------|
| Train Accuracy   | 75.83%    |
| Test Accuracy    | 63.81%    |
| Train Loss       | 0.6803    |

> Note: ViT models typically require **larger datasets** and longer training times to reach competitive accuracy. CIFAR-10 is a relatively small dataset for this type of architecture, which is why the test accuracy is lower than expected.

---

##  Future Improvements

- Train for more epochs (50+)
- Use learning rate warm-up and cosine decay
- Add data augmentation (e.g., random cropping, horizontal flip)
- Experiment with larger patch sizes and deeper models
- Evaluate on higher-resolution datasets like CIFAR-100 or ImageNet

---

##  Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Taha-bouhafa1/ViT_model.git
   cd ViT_model
   ```
2.Install dependencies:
 ```bash
   pip install -r requirements.txt
   ```
