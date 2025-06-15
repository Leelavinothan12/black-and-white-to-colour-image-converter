# 🎨 Image Colorization App

This is a simple **Image Colorization Web App** built using **PyTorch** and **Streamlit**. It allows users to upload grayscale images and returns a colorized version generated using a custom convolutional neural network model.

---

## 🚀 Features

- Upload grayscale images (`.jpg`, `.jpeg`, `.png`)
- Automatically colorizes the image using a deep learning model
- Real-time preview of original and colorized images
- Simple and intuitive web interface with Streamlit

---

## 🧠 Model Architecture

The model used is a **custom CNN** (`ColorizationNet`) with the following layers:

- Convolutional layers with dilation
- ReLU activations
- Final sigmoid activation to output 3-channel RGB image

```python
Input (1 channel) → Conv2d → ReLU → Conv2d → ReLU → Conv2d → ReLU → Conv2d → Sigmoid → Output (3 channels)
