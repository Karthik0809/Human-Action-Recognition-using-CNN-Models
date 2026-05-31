# Human Action Recognition using CNN Models

> Comparative study of VGG-16, ResNet-50, DenseNet-121, and GoogLeNet on the Stanford 40 Actions dataset, with a Streamlit deployment app.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-red)](https://streamlit.io/)
[![Research Paper](https://img.shields.io/badge/Paper-PDF-green)](Human_Action_Recognition_Research.pdf)

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models and Results](#models-and-results)
- [Hyperparameters](#hyperparameters)
- [Installation](#installation)
- [Usage](#usage)
- [Streamlit App](#streamlit-app)
- [Applications](#applications)
- [Author](#author)

---

## Overview

This project trains and evaluates four CNN architectures on 40 human action categories from still images. Models are either trained from scratch or fine-tuned from ImageNet pretrained weights. The best models (VGG-16 and DenseNet-121 pretrained) achieve approximately 77-78% test accuracy on the Stanford 40 dataset.

---

## Dataset

| Property | Value |
|---|---|
| Name | [Stanford 40 Action Dataset](http://vision.stanford.edu/Datasets/40actions.html) |
| Total Images | 9,532 |
| Classes | 40 human actions |
| Images per class | 180 - 300 |
| Split | 75% Train / 15% Val / 10% Test |

<details>
<summary>View all 40 action classes</summary>

`Applauding` `Blowing_Bubbles` `Brushing_Teeth` `Cleaning_The_Floor` `Climbing` `Cooking` `Cutting_Trees` `Cutting_Vegetables` `Drinking` `Feeding_a_horse` `Fishing` `Fixing_a_bike` `Fixing_a_Car` `Gardening` `Holding_an_Umbrella` `Jumping` `Looking_through_a_Microscope` `Looking_through_a_Telescope` `Phoning` `Playing_Guitar` `Playing_Violin` `Pouring_Liquid` `Pushing_a_Cart` `Reading` `Riding_a_Bike` `Riding_a_Horse` `Rowing_a_Boat` `Running` `Shooting_an_Arrow` `Smoking` `Taking_Photos` `Texting_Message` `Throwing_Frisby` `Using_a_Computer` `Walking_the_dog` `Washing_Dishes` `Watching_TV` `Waving_Hands` `Writing_on_a_Board` `Writing_on_a_Book`

</details>

**Preprocessing:**
- Resize to **224 x 224** pixels
- Normalize with ImageNet stats: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

---

## Project Structure

```
├── config.py               # All hyperparameters and class definitions
├── train.py                # CLI training entry point
├── app.py                  # Streamlit deployment app
├── requirements.txt        # Dependencies
├── main.ipynb              # Original Colab research notebook
├── Human_Action_Recognition_Research.pdf
├── Report.pdf
└── src/
    ├── dataset.py          # HARDataset class and DataLoader factory
    ├── train.py            # train_model() and evaluate_model()
    ├── utils.py            # Plotting utilities (loss, accuracy, confusion matrix)
    └── models/
        ├── vgg.py          # VGG-16 custom implementation + pretrained weights
        ├── resnet.py       # ResNet-50 custom implementation + pretrained weights
        ├── densenet.py     # DenseNet-121 from scratch and pretrained
        └── googlenet.py    # GoogLeNet from scratch and pretrained
```

---

## Models and Results

### Fine-Tuned from ImageNet Pretrained Weights

| Model | Test Accuracy | Test Loss |
|---|---|---|
| VGG-16 | 77.25% | 0.0874 |
| ResNet-50 | 75.68% | 0.0923 |
| GoogLeNet | 76.52% | — |
| DenseNet-121 | 77.99% | — |

### Trained from Scratch

| Model | Test Accuracy | Test Loss |
|---|---|---|
| GoogLeNet | 19.60% | 0.2978 |
| DenseNet-121 | 31.76% | 0.2473 |

### Pretrained Model Metrics (Weighted)

| Model | Precision | Recall | F1 Score |
|---|---|---|---|
| GoogLeNet | 78.14% | 76.52% | 76.80% |
| DenseNet-121 | 79.34% | 77.99% | 77.90% |

---

## Hyperparameters

| Model | Mode | Learning Rate | Weight Decay | Epochs |
|---|---|---|---|---|
| VGG-16 | Pretrained | 1e-5 | 1e-4 | 10 |
| ResNet-50 | Pretrained | 1e-4 | 1e-4 | 10 |
| DenseNet-121 | From Scratch | 1e-3 | 1e-5 | 15 |
| GoogLeNet | From Scratch | 1e-3 | 1e-5 | 15 |
| GoogLeNet | Pretrained | 1e-4 | — | 10 |
| DenseNet-121 | Pretrained | 1e-4 | — | 10 |

All models use:
- **Optimizer:** Adam
- **Loss:** CrossEntropyLoss
- **Batch Size:** 32
- **Dropout (DenseNet scratch):** 0.5

---

## Installation

```bash
git clone https://github.com/Karthik0809/Human-Action-Recognition-using-CNN-Models.git
cd Human-Action-Recognition-using-CNN-Models
pip install -r requirements.txt
```

Download the [Stanford 40 Actions dataset](http://vision.stanford.edu/Datasets/40actions.html) and extract it so the folder structure looks like:

```
JPEGImages/
├── Applauding/
├── Blowing_Bubbles/
├── ...
└── Writing_on_a_Book/
```

---

## Usage

Train any model from the command line:

```bash
# VGG-16 pretrained (lr=1e-5, 10 epochs)
python train.py --model vgg --data_dir JPEGImages

# ResNet-50 pretrained (lr=1e-4, 10 epochs)
python train.py --model resnet --data_dir JPEGImages

# DenseNet-121 from scratch (lr=1e-3, 15 epochs)
python train.py --model densenet --data_dir JPEGImages

# GoogLeNet from scratch (lr=1e-3, 15 epochs)
python train.py --model googlenet --data_dir JPEGImages

# GoogLeNet pretrained fine-tuned (lr=1e-4, 10 epochs)
python train.py --model pretrained_googlenet --data_dir JPEGImages

# DenseNet-121 pretrained fine-tuned (lr=1e-4, 10 epochs)
python train.py --model pretrained_densenet --data_dir JPEGImages
```

Each run prints per-epoch metrics, saves a loss/accuracy plot, a confusion matrix, and the model weights as `<model>_har.pth`.

---

## Streamlit App

An interactive web app that trains ResNet-50 and lets you predict random images:

```bash
streamlit run app.py
```

Features:
- Set your dataset directory from the UI
- Live training with loss and accuracy plots
- Predict random images with true vs. predicted labels

---

## Applications

| Domain | Use Case |
|---|---|
| Surveillance and Security | Detecting suspicious activities |
| Healthcare | Monitoring patient behavior |
| Sports and Fitness | Analyzing exercise routines |
| Human-Computer Interaction | Gesture recognition |
| Robotics | Human-robot interaction |
| Smart Homes | Detecting unusual behavior |
| Education | Monitoring student engagement |

---

## Author

**Karthik Mulugu**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/karthikmulugu/)
