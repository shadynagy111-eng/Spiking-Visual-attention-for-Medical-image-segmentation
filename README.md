# Spiking Neural Network for Visual Attention in Medical Imaging

![Project Banner](./assets/banner.png)  

## 📌 Overview
Brain tumor detection and saliency-based tumor localization are critical for effective diagnosis and treatment. Traditional deep learning models like CNNs are computationally expensive and require significant power resources. This project explores the use of **Spiking Neural Networks (SNNs)** with a spike-based **visual attention mechanism** to enhance efficiency and accuracy in **medical image analysis**.

## 🏥 Dataset
We utilize the **Brain Tumor Segmentation dataset**, which includes MRI scans with **tumor**. The dataset provides:
- MRI brain images with tumor
- Ground truth segmentation masks

📌 **Dataset Link**: [Kaggle: Brain Tumor Segmentation](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation)  

**Sample MRI Scan with Ground Truth Segmentation:**  
![Dataset Sample](./assets/dataset_sample.png)

**Sample GIF of Brain with Tumor:**  
![Brain Tumor GIF](./assets/patient_244.gif)

## 🏗️ Proposed Model Architecture
The model integrates a **Convolutional Spiking Neural Network (CSNN)** for feature extraction and a **spike-based spatial attention mechanism** to focus on the most salient regions in the image.

### 🔷 Block Diagram of the Proposed Model:
![CSNN Block Diagram](./assets/csnn_diagram.png)

### 🔹 SNN Attention Mechanism:
![SNN Attention Model](./assets/snn_attention.png)

## 🔬 Methodology
1. **Dataset Preprocessing** - Normalization, augmentation, and conversion of MRI images into a format suitable for SNN processing.
2. **Model Development** - Implementing a hybrid **CNN-SNN fusion model** with spike-based attention.
3. **Training & Evaluation** - Comparing performance with traditional CNNs in terms of accuracy and power efficiency.
4. **Optimization** - Fine-tuning hyperparameters to enhance SNN learning dynamics.

## ⚡ Key Features
✔️ **Bio-Inspired Computing** - Mimics human brain activity for power efficiency.  
✔️ **Saliency-Based Attention** - Focuses on **important tumor regions** dynamically.  
✔️ **Efficient Computation** - Reduces hardware constraints compared to CNNs.  
✔️ **Scalability** - Can be extended for real-time applications.  

## 🚀 Installation & Setup
1. Clone this repository:
   ```sh
   git https://github.com/shadynagy111-eng/Spiking-Visual-attention-for-Medical-image-segmentation.git
   cd Spiking-Visual-attention-for-Medical-image-segmentation
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run training:
   ```sh
   python train.py --dataset path/to/dataset --epochs 50
   ```

## 📊 Results & Comparisons
| Model | Accuracy | Computation Time | Power Efficiency |
|--------|-----------|------------------|------------------|
| CNN | _____ | _____ | _____ |
| SNN | _____ | _____ | _____ |
| Hybrid CNN-SNN | _____ | _____ | _____ |

📌 **Key Takeaway:** _____

## 📌 Future Work
🔹 Extending the model for **real-time applications** on edge devices.  
🔹 Exploring **neuromorphic hardware** for further efficiency gains.  
🔹 Improving **spike-based learning rules** for better accuracy.  

## 📜 Citation
If you use this work in your research, please cite:
```
@article{spiking2025,
  title={Spiking Neural Network for Visual Attention in Medical Imaging},
  author={Mostafa Ahmed, Mohammed Abdel-Megeed, Shady Ahmed},
  journal={German University in Cairo},
  year={2025}
}
```

## 📩 Contact
For inquiries or collaborations, reach out via:
- 📧 Email: mostafaahmed96320@gmail.com
- 🔗 LinkedIn: [Mostafa Ahmed](https://www.linkedin.com/in/mostafaahmedgalal/)

---
_This project is part of our research on **bio-inspired neural networks for medical imaging**._ 🎯
