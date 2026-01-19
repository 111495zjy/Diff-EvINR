# Diff-EvINR: event-to-video reconstruction using diffusion models and implicit neural representations

This repository contains the implementation of **Diff-EvINR**, a method for reconstructing high-quality videos from event data by leveraging diffusion models and implicit neural representations (INRs).  
Our approach bridges **physics-based event alignment** with **data-driven generative priors**, leading to robust and detailed reconstructions.


## ▶️ Video examples

| ![Sequence1](https://github.com/111495zjy/Video_examples_of_Diff-EvINR/raw/main/sequence1.gif) | ![Sequence2](https://github.com/111495zjy/Video_examples_of_Diff-EvINR/raw/main/sequence2.gif) |
|:--:|:--:|
|Sequence 1|Sequence 2|

| ![Sequence3](https://github.com/111495zjy/Video_examples_of_Diff-EvINR/raw/main/sequence3.gif) | ![Sequence4](https://github.com/111495zjy/Video_examples_of_Diff-EvINR/raw/main/sequence4.gif) |
|:--:|:--:|
|Sequence 3|Sequence 4|

## 📦 Implementation

1. Clone repository:

```bash
git clone https://github.com/111495zjy/Diff-EvINR.git
cd ~/Diff-EvINR
```
2. Install dependencies
 ```bash
pip install -r requirements.txt
pip install lpips
pip install scikit-image
 ```

3. Download pretrained models
 ```bash
bash download.sh
 ```
4. Download dataset. Please download the ECD.zip in the following link first: https://drive.google.com/file/d/1CgW9PEtQ2M3CtrH0E0aRR8CJe7HeWHsI/view?usp=drive_link
 ```bash   
unzip ~/ECD.zip -d ~/Diff-EvINR/
cd ~/Diff-EvINR/EvINR_towards_fastevent
python txt_npy.py # you need to modify its content about the dataset path
 ```
You can also use this link for the dataset:https://drive.google.com/drive/folders/1BQCoPXlr65XpYwAW7i2SpcnFNi9Od2ij?usp=drive_link. Make sure the datasets are included in the file Diff-EvINR. Then, run the model:
 ```bash
%cd /content/Diff-EvINR
python main.py
 ```
