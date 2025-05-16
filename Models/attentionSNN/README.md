# AttentionSNN Model

This folder contains the implementation of the **Attention Spiking Neural Network (AttentionSNN)** model. The AttentionSNN model combines spiking neural networks with attention mechanisms and Spiking Upsampling to enhance segmentation performance in medical imaging.

## Overview

The AttentionSNN model focuses on improving segmentation accuracy by dynamically attending to important regions in the input images.

## Results

### Quantitative Results

| Metric            | Value           |
| ----------------- | --------------- |
| Accuracy          | 97.94 %         |
| Dice Coefficient  | 0.503           |
| Parameters        | 23,483,986      |
| MACs              | 16.82 MMac      |
| Power             | 0.0485 kWh      |
| CO2 emission      | 0.0228 kg       |
| Training Duration | 1939.52 seconds |

### Qualitative Results

![AttentionSNN Training Loss](./../../assets/AttentionSNN_Training_Loss.png)

![AttentionSNN Validation Accuarcy](./../../assets/AttentionSNN_Validation_Accuarcy.png)

![AttentionSNN Dice Score](./../../assets/AttentionSNN_Dice_Score.png)

![AttentionSNN NASAR](./../../assets/AttentionSNN_NASAR.png)

### Qualitative Results

![AttentionSNN Masks Output](./../../assets/AttentionSNN_Masks.png)

![AttentionSNN Grad-CAM Output](./../../assets/AttentionSNN_Grad_CAM.png)

## Environmental Impact

![AttentionSNN Eco2AI Summary](./eco2ai/attentionSNN_eco2ai_summary_plot.png)
