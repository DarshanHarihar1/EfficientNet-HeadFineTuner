# EfficientNet Head-Only FineTuner for Indian-Food Classification

Fine-tune EfficientNet-B0’s final layer on a 15-class Indian-food dataset. Fine tuning this as I will be using the resulting checkpoint in one of the projects I am building in similar field.

---

## Overview

- **Backbone**: Pretrained EfficientNet-B0, freeze all feature layers, train only the final linear head (15 outputs).  
- **Dataset**: “bharat-raghunathan/indian-foods-dataset” (15 classes, ~4 770 images).  
- **Compute**: CPU-only (Mac) with head-only training → fast convergence (~8-10 epochs).  
- **Final Test Accuracy**: ~87–90% top-1.

---
