<!-- Slide number: 1 -->

# ![placeholder](./On-Device-Crop-Stress-Detection-with-Explainable-Advisory-using-Quantized-LLMs (2)_images\image_001.png)
On-Device Crop Stress Detection with Explainable Advisory using Quantized LLMs
A precision agriculture system combining deep learning, edge AI, and natural language processing for real-time crop disease diagnosis and treatment recommendations

### Notes:

<!-- Slide number: 2 -->

![placeholder](./On-Device-Crop-Stress-Detection-with-Explainable-Advisory-using-Quantized-LLMs (2)_images\image_002.png)
System Overview
A comprehensive pipeline for precision agriculture, integrating advanced AI models from data acquisition to actionable insights, designed for efficient on-device execution.

![placeholder](./On-Device-Crop-Stress-Detection-with-Explainable-Advisory-using-Quantized-LLMs (2)_images\image_003.png)
CNN Detection
92-97% accuracy

![placeholder](./On-Device-Crop-Stress-Detection-with-Explainable-Advisory-using-Quantized-LLMs (2)_images\image_004.png)
GradCAM Explainability
visual heatmaps

![placeholder](./On-Device-Crop-Stress-Detection-with-Explainable-Advisory-using-Quantized-LLMs (2)_images\image_005.png)
Quantized LLM
local advisory

![placeholder](./On-Device-Crop-Stress-Detection-with-Explainable-Advisory-using-Quantized-LLMs (2)_images\image_006.png)
Edge Deployment
mobile/Raspberry Pi

### Notes:

<!-- Slide number: 3 -->

![placeholder](./On-Device-Crop-Stress-Detection-with-Explainable-Advisory-using-Quantized-LLMs (2)_images\image_007.png)
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

![placeholder](./On-Device-Crop-Stress-Detection-with-Explainable-Advisory-using-Quantized-LLMs (2)_images\image_008.png)
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

![placeholder](./On-Device-Crop-Stress-Detection-with-Explainable-Advisory-using-Quantized-LLMs (2)_images\image_009.png)
PlantVillage
54K lab images
Combined Training Data
Diverse dataset
PlantDoc
Real field images
Balanced training set

### Notes:

<!-- Slide number: 6 -->

![placeholder](./On-Device-Crop-Stress-Detection-with-Explainable-Advisory-using-Quantized-LLMs (2)_images\image_010.png)
Deployment & Impact

FastAPI Backend
RESTful API for image upload and inference.

Web Interface
React/Flask frontend for farmer-friendly results display.

Real-World Impact
Offline capability, cost-effective solution for smallholder farmers.

The offline capability is a key advantage, ensuring farmers in remote areas can access crucial information without internet connectivity.

![placeholder](./On-Device-Crop-Stress-Detection-with-Explainable-Advisory-using-Quantized-LLMs (2)_images\image_011.png)

### Notes: