# MRNet: Effective Music Genre Classification

This repository contains the official implementation of **MRNet**, a convolutional architecture for music genre classification.  
It includes the dataset splitting code (K-fold) and training code for experiments (e.g., FMA Small).

---

## 📊 Dataset Splits

- **GTZAN / Melon**: 10-fold cross validation following prior research  
- **FMA**: Official test protocols  

We provide both:

1. **K-fold splitting scripts** (to reproduce splits yourself) 

2. **Pre-generated K-fold split files** (for convenience)  

---

## 🚀 Running Experiments

1. **Set the scripts/argument.py** in your command.  

2. Run training:

```bash
python scripts/main.py
