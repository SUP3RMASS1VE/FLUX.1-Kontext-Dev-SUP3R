# 🌌 FLUX.1 Kontext - Contextual Image Editing Interface
![Screenshot 2025-07-01 210529](https://github.com/user-attachments/assets/97360677-3a7a-45b9-8d72-f8b729d7d81d)

A Gradio-powered web interface for **contextual image editing** using the [FLUX.1-Kontext](https://huggingface.co/fuliucansheng/FLUX.1-Kontext-dev-diffusers) model and `DFloat11` enhancements. Just upload an image, type what you want changed (e.g., *“add sunglasses”*), and let the model transform your image accordingly.

---

## ✨ Features

* 🧠 Text-guided image editing with powerful transformer diffusion.
* 🎲 Seed control (with optional randomization) for reproducibility.
* ⚙️ Adjustable inference settings (steps, guidance scale).
* 🎨 Custom UI with dark mode support and responsive animations.

---

## 🚀 Installation

1. **Clone this repository** and navigate to it:

```bash
git clone https://github.com/SUP3RMASS1VE/FLUX.1-Kontext-Dev-SUP3R
cd FLUX.1-Kontext-Dev-SUP3R

# Create a virtual environment named 'env'
python -m venv env

# Activate it (Windows)
env\Scripts\activate
```
---

### 2. **Install Dependencies**

```bash
# Install PyTorch with CUDA 12.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install other required packages
pip install -r requirements.txt
```

### 3. **Run the Application**

```bash
python app.py
```
---

## 🧠 Model Info

* **Diffusion Model**: `fuliucansheng/FLUX.1-Kontext-dev-diffusers`
* **DFloat11 Enhancement**: `DFloat11/FLUX.1-Kontext-dev-DF11`

These models are automatically downloaded at runtime using Hugging Face’s `from_pretrained` methods.

---

## 💡 Usage

Launch the app using Gradio:

```bash
python app.py
```

Then open the local URL (usually `http://127.0.0.1:7860/`) in your browser.

---

## 🎨 Custom Styling

All UI components are styled using a CSS theme inspired by modern glassmorphism, gradients, and dark/light mode responsiveness. Font: **Poppins**.

---

## 📜 License

This project uses model weights from Hugging Face. Please consult the respective model licenses before commercial use.

---

## 🤝 Acknowledgments

* **black-forest-labs** for releasing the FLUX.1-Kontext model.
* **Hugging Face** for `diffusers`.
* **Gradio** for a clean and fast UI interface.
* **DFloat11** contributors for float-optimized transformers.

---



