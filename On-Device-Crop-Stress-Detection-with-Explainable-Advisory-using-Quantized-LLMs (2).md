<!-- Slide number: 1 -->

![preencoded.png](Image0.jpg)
On-Device Crop Stress Detection with Explainable Advisory using Quantized LLMs
A precision agriculture system combining deep learning, edge AI, and natural language processing for real-time crop disease diagnosis and treatment recommendations

### Notes:

<!-- Slide number: 2 -->

![preencoded.png](Image0.jpg)
System Overview
A comprehensive pipeline for precision agriculture, integrating advanced AI models from data acquisition to actionable insights, designed for efficient on-device execution.

![preencoded.png](Image1.jpg)
CNN Detection
92-97% accuracy

![preencoded.png](Image2.jpg)
GradCAM Explainability
visual heatmaps

![preencoded.png](Image3.jpg)
Quantized LLM
local advisory

![preencoded.png](Image4.jpg)
Edge Deployment
mobile/Raspberry Pi

### Notes:

<!-- Slide number: 3 -->

![preencoded.png](Image0.jpg)
Technical Implementation
Model Architecture
Quantization Strategy
EfficientNetB0 with transfer learning, 224px input, 10 epochs.
90MB → 15MB (6x reduction) using TFLite for mobile deployment.
90MB
15MB
6x
Original Model Size
Quantized Model Size
Size Reduction

### Notes:

<!-- Slide number: 4 -->

![preencoded.png](Image0.jpg)
Key Features

Explainable AI
Local Intelligence
GradCAM heatmaps highlight infected regions for transparency
Quantized LLM generates treatment advice without internet

Farmer-Friendly
Multi-language support (English, Hindi, regional languages)

### Notes:

<!-- Slide number: 5 -->
Data & Training
Dataset Strategy
Performance Metrics
PlantVillage (54K lab images) + PlantDoc (real field images) for balanced training.
92-97% accuracy, 224px input size, trained on combined datasets.

![preencoded.png](Image0.jpg)
PlantVillage
54K lab images
Combined Training Data
Diverse dataset
PlantDoc
Real field images
Balanced training set

### Notes:

<!-- Slide number: 6 -->

![preencoded.png](Image0.jpg)
Deployment & Impact

FastAPI Backend
RESTful API for image upload and inference.

Web Interface
React/Flask frontend for farmer-friendly results display.

Real-World Impact
Offline capability, cost-effective solution for smallholder farmers.

The offline capability is a key advantage, ensuring farmers in remote areas can access crucial information without internet connectivity.

![preencoded.png](Image1.jpg)

### Notes: