"""
Streamlit deployment app for Human Action Recognition.
Trains ResNet-50 on the Stanford 40 dataset and provides an interactive prediction UI.

Run:
    streamlit run app.py
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import streamlit as st
from torchvision import transforms
from PIL import Image

from config import CLASSES, NUM_CLASSES, DATA_DIR, IMG_SIZE, MEAN, STD, BATCH_SIZE
from src.dataset import HARDataset, load_data, build_transform, get_dataloaders
from src.models import make_resnet50
from src.train import train_model

st.title('Human Action Recognition using CNN Models')
st.write('Stanford 40 Actions — ResNet-50 Fine-Tuned')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = st.text_input('Dataset directory', value=DATA_DIR)

if not os.path.isdir(data_dir):
    st.error(f'Directory not found: {data_dir}')
    st.stop()

train_loader, val_loader, test_loader, dataset = get_dataloaders(data_dir, batch_size=BATCH_SIZE)

model = make_resnet50(NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

num_epochs = 10

if st.button('Train Model'):
    model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        CLASSES, device, num_epochs=num_epochs,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()

    st.pyplot(fig)
    st.session_state['model_trained'] = True
    st.session_state['model'] = model


def predict_random(model, dataset, classes, device):
    model.eval()
    idx = random.randint(0, len(dataset) - 1)
    image_tensor, true_label = dataset[idx]
    image_pil = transforms.ToPILImage()(image_tensor)
    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0).to(device))
        _, predicted = torch.max(outputs, 1)
    return image_pil, classes[predicted.item()], true_label


if st.session_state.get('model_trained'):
    model = st.session_state['model']
    image_pil, pred_class, true_label = predict_random(model, dataset, CLASSES, device)
    st.image(image_pil,
             caption=f'True: {true_label}  |  Predicted: {pred_class}',
             use_column_width=True)
    if st.button('Predict Another'):
        image_pil, pred_class, true_label = predict_random(model, dataset, CLASSES, device)
        st.image(image_pil,
                 caption=f'True: {true_label}  |  Predicted: {pred_class}',
                 use_column_width=True)
