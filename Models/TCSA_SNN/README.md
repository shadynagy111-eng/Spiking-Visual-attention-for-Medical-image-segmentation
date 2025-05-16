# TCSA_SNN Model

This folder contains the implementation of the **Temporal Convolutional Spiking Attention SNN (TCSA_SNN)** model. The TCSA_SNN model extends spiking neural networks with temporal and spatial attention mechanisms for advanced medical image segmentation.

## Overview

The TCSA_SNN model is optimized for temporal dynamics in spiking neural networks, enabling it to process sequential data effectively while maintaining high segmentation accuracy.

![TCSA_SNN Model Diagram](./../../assets/TCSA_SNN_model.png)

## Results

### Quantitative Results

| Metric            | Value           |
| ----------------- | --------------- |
| Accuracy          | 99.12 %         |
| Dice Coefficient  | 0.773           |
| Parameters        | 22,982,497      |
| MACs              | 763.78 MMac     |
| Power             | 0.2505 kWh      |
| CO2 emission      | 0.1179 kg       |
| Training Duration | 9881.24 seconds |

![TCSA_SNN Training Loss](./../../assets/TCSA_SNN_Training_Loss.png)

![TCSA_SNN Validation Accuarcy](./../../assets/TCSA_SNN_Validation_Accuarcy.png)

![TCSA_SNN Dice Score](./../../assets/TCSA_SNN_Dice_Score.png)

![TCSA_SNN NASAR](./../../assets/TCSA_SNN_NASAR.png)

### Qualitative Results

![TCSA_SNN Masks Output](./../../assets/TCSA_SNN_Masks.png)

![TCSA_SNN Grad-CAM Output](./../../assets/TCSA_SNN_Grad_CAM.png)

## Environmental Impact

![TCSA_SNN Eco2AI Summary](./eco2ai/TCSA_SNN_eco2ai_summary_plot.png)
