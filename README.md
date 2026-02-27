# 🧠 MNIST Digit Recognition Web App

A **high-accuracy MNIST digit recognition** web app built with **Python**, **Streamlit**, and **scikit-learn**.  

Draw a digit (0–9) on the canvas, and the app predicts it in real-time using a **multi-layer perceptron (MLP) classifier**. The app includes robust preprocessing to ensure **hand-drawn digits are accurately recognized** (~98–99% accuracy).

---

## ⚡ Features

- Draw digits on a **Streamlit canvas**  
- High-accuracy predictions with a **3-layer MLP classifier**  
- Digit preprocessing:
  - Cropping & centering using **center of mass**
  - Resizing while preserving aspect ratio  
  - Gaussian blur for smoothing strokes  
  - Normalization to MNIST scale  
- Lightweight and **no TensorFlow required**  
- Saves trained model for fast loading on subsequent runs  
- Browser tab title and footer customization

