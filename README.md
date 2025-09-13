# Diff-EvINR: Event-to-Video Reconstruction with Diffusion Priors and Implicit Neural Representations

This repository contains the implementation of **Diff-EvINR**, a method for reconstructing high-quality videos from event data by leveraging diffusion priors and implicit neural representations (INRs).  
Our approach bridges **physics-based event alignment** with **data-driven generative priors**, leading to robust and detailed reconstructions.

---

## 🚀 Features
- Physics-informed reconstruction using event streams.
- Plug-and-play framework combining INR with pre-trained diffusion models.
- Data preprocessing pipeline for converting raw event text files into `.npy`.
- Supports single-event sequences and benchmark datasets (e.g., ECD).
- Implementation based on **PyTorch**.

---

## 📦 Installation

Run the following commands in **Google Colab** (or local environment):

```bash
# 1. Clone repository
!git clone https://github.com/111495zjy/Diff-EvINR.git
%cd /content/Diff-EvINR

# 2. Install dependencies
!pip install -r requirements.txt
!pip install lpips
!pip install scikit-image

# 3. Download pretrained models and additional resources
!bash download.sh
