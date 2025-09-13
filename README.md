# Diff-EvINR: event-to-video reconstruction using diffusion models and implicit neural representations

This repository contains the implementation of **Diff-EvINR**, a method for reconstructing high-quality videos from event data by leveraging diffusion models and implicit neural representations (INRs).  
Our approach bridges **physics-based event alignment** with **data-driven generative priors**, leading to robust and detailed reconstructions.

## 📦 Implementation

Run the following commands:

```bash
# 1. Clone repository
git clone https://github.com/111495zjy/Diff-EvINR.git
cd ~/Diff-EvINR

# 2. Install dependencies
pip install -r requirements.txt
pip install lpips
pip install scikit-image

# 3. Download pretrained models
bash download.sh
# 4. Download dataset
unzip ~/ECD.zip -d ~/Diff-EvINR/
cd ~/Diff-EvINR/EvINR_towards_fastevent
python txt_npy.py
# 5. Run model
%cd /content/Diff-EvINR
python main_ddpir_deblur.py
