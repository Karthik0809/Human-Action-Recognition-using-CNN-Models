# Human Action Recognition in Images Using Convolutional Neural Networks

## Overview
This project focuses on recognizing human actions in images using deep learning models, specifically Convolutional Neural Networks (CNNs). The dataset used for this task is the **Stanford 40 Action Dataset**, which consists of images depicting 40 different human actions.

## Dataset
- **Dataset Name**: Stanford 40 Action Dataset
- **Total Images**: 9532
- **Images per Action Class**: 180-300
- **Labels**: 40 different human actions
- **Dataset Link**: [Stanford 40 Actions](http://vision.stanford.edu/Datasets/40actions.html)

## Methodology
We employed various deep learning architectures for action recognition:

### Preprocessing Steps
- **Image Resizing**: All images are resized to **224x224 pixels**.
- **Normalization**: Mean values ([0.485, 0.456, 0.406]) and standard deviations ([0.229, 0.224, 0.225]) are used for normalization.
- **Custom Dataset Loader**: The dataset is divided into 40 classes and loaded via a custom dataset class.

## Models Used
### 1. VGG Model
- **Architecture**: 5 convolutional blocks with max-pooling and fully connected layers.
- **Test Accuracy**: 77.25%
- **Loss**: 0.0874

### 2. ResNet Model
- **Architecture**: Residual learning with skip connections to avoid vanishing gradients.
- **Test Accuracy**: 75.68%
- **Loss**: 0.0923

### 3. GoogleNet Model
- **Architecture**: Uses **Inception Modules** to capture multi-scale features.
- **Test Accuracy**: 19.60%
- **Loss**: 0.2978

### 4. DenseNet Model
- **Architecture**: Utilizes **dense connectivity** to improve gradient flow.
- **Test Accuracy**: 31.76%
- **Loss**: 0.2473

## Pretrained Models
We also tested pretrained versions of GoogleNet and DenseNet for better performance:

| Model          | Test Accuracy | Precision | Recall | F1 Score |
|---------------|--------------|-----------|--------|----------|
| GoogleNet (Pretrained) | 76.52% | 78.14% | 76.52% | 76.80% |
| DenseNet (Pretrained)  | 77.99% | 79.34% | 77.99% | 77.90% |

## Deployment
The model is deployed using **Streamlit** for an interactive web application. Features include:
- Displaying real-time training progress (loss & accuracy).
- Classifying randomly selected images from the dataset.

## Real-World Applications
- **Surveillance & Security**: Detecting suspicious activities.
- **Healthcare**: Monitoring patients for abnormal behavior.
- **Sports & Fitness**: Analyzing exercise routines.
- **Human-Computer Interaction**: Gesture recognition.
- **Robotics**: Assisting in human-robot interaction.
- **Smart Homes**: Detecting unusual behavior.
- **Education**: Monitoring student engagement.

## References
- [Understanding ResNet](https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/)
- [GoogleNet Model](https://www.geeksforgeeks.org/understanding-googlenet-model-cnn-architecture/)
- [VGG Implementation](https://medium.com/@ilaslanduzgun/create-vgg-from-scratch-in-pytorch-aa194c269b55)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## How to Run the Project
1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd <project-directory>
   ```
2. Run the Streamlit app:
   ```sh
   streamlit run pro.py
   ```

