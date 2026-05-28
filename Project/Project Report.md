# Project Report: Plant Disease Image Classification

## Problem Statement

This project addresses a supervised computer vision task: classifying plant leaf images into three health conditions: `Healthy`, `Powdery`, and `Rust`. The motivation is agricultural decision support: visible symptoms such as color changes, texture patterns, spots, and rust-like marks can provide early evidence of plant disease, and an automated classifier can help speed up diagnosis.

The dataset used is `Plant_disease_recognition_dataset`, organized with separate `Train`, `Validation`, and `Test` folders. It contains 1,532 RGB `.jpg` images: 1,322 for training, 60 for validation, and 150 for testing. The test set is balanced with 50 images per class. The project follows the proposal plan: first establish a simple CNN baseline, then compare it with transfer learning using ResNet50 and EfficientNet-B0.

## Model Architecture

Two model families were implemented and compared.

**Baseline CNN.** The baseline is a small convolutional neural network trained from scratch. Images are resized to `128x128` and passed through a single convolutional block: `Conv2d(3, 16, kernel_size=3, padding=1)`, `ReLU`, `MaxPool2d`, `Flatten`, and a final linear classifier with 3 output classes. This model has 197,059 trainable parameters and serves as the minimum reference performance.

**ResNet50 transfer learning.** The improved model uses ResNet50 pretrained on ImageNet. Inputs are resized/cropped to `224x224` and normalized with ImageNet mean and standard deviation. Two transfer learning settings were tested: a frozen version where only the final `fc` layer is trained, and a partial fine-tuning version where `layer3`, `layer4`, and `fc` are trainable. The partial fine-tuning stage starts from the best frozen ResNet50 model.

**EfficientNet-B0 transfer learning.** EfficientNet-B0 pretrained on ImageNet was loaded via HuggingFace Transformers. Inputs are preprocessed to `224x224` with ImageNet normalization using EfficientNetImageProcessor. A single partial fine-tuning configuration was tested: the first `6 encoder blocks` were frozen, and only the last `2 MBConv blocks` and the classification `cl` head were trained, adapting the output from 1,000 ImageNet classes to 3.

## Training Procedure

| Model | Optimizer | Learning rate | Epochs | Batch size | Loss |
|---|---:|---:|---:|---:|---|
| Baseline CNN | Adam | `1e-3` | 8 | 32 | CrossEntropyLoss |
| ResNet50 frozen | AdamW | `1e-3` | 8 | 32 | CrossEntropyLoss |
| ResNet50 partial fine tuning | AdamW | `1e-5` for `layer3/layer4`, `1e-4` for `fc` | 8 | 32 | CrossEntropyLoss |
| EfficientNet-B0 partial fine tuning | AdamW | `1e-5` | 10 | 16 | CrossEntropyLoss |

Training was run with CUDA available. The frozen ResNet50 stage took approximately 31.4 minutes, and the partial fine-tuning stage took approximately 33.3 minutes, also EfficientNet-B0 took approximately 18.2 minutes. Data augmentation for ResNet50 included random resized crop and horizontal flip on the training set; validation and test used deterministic resize and center crop.

## Results

| Model | Trainable layers | Test loss | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---|---:|---:|---:|---:|---:|
| Baseline CNN | Full CNN | 0.8657 | 0.6800 | 0.7974 | 0.6800 | 0.6689 |
| ResNet50 frozen | `fc` | 0.2495 | 0.9533 | 0.9538 | 0.9533 | 0.9533 |
| ResNet50 partial fine tuning | `layer3 + layer4 + fc` | 0.1893 | 0.9600 | 0.9629 | 0.9600 | 0.9601 |
| EfficientNet-B0 partial fine tuning | `last 2 MBConv blocks + cl` | - | 0.9200 | 0.9200 | 0.9200 | 0.9200 |

The baseline CNN improved quickly on training data but generalized less well: final training accuracy reached 0.9818 while validation accuracy ended at 0.7500, indicating overfitting. Its test performance was weakest on `Rust`, with recall of 0.3600.

ResNet50 produced a large improvement. With the backbone frozen, validation accuracy reached 0.9833 and test accuracy reached 0.9533. Partial fine tuning gave the best overall result, reaching 1.0000 validation accuracy throughout the fine-tuning stage and 0.9600 test accuracy. Class-level performance was also strong: `Healthy` recall was 1.0000, `Powdery` precision was 1.0000, and `Rust` F1-score was 0.9697.

EfficientNet-B0 also produced a large improvement validation accuracy reached 0.9833 and test accuracy reached 0.9200. Taking in consideration that this model take less time to retraining against ResNet50.

Overall, transfer learning clearly outperformed the CNN trained from scratch. The small gain from frozen ResNet50 to partial fine tuning and EfficientNet-B0 suggests that ImageNet features already transfer well to this dataset, while adapting deeper residual blocks adds a modest but measurable improvement. The high validation score should still be interpreted carefully because the validation split contains only 60 images.
