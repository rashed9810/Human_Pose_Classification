# Human Pose Classification with Vision Transformer

This project implements a human pose classification system using the Vision Transformer (ViT) architecture. Using the power of self-attention mechanisms, the model accurately categorizes human activities from images into five distinct classes. This solution is built on top of the Hugging Face Transformers library, ensuring state-of-the-art performance and ease of use.

## Key Features

-   **Advanced Architecture**: Utilizes `google/vit-base-patch16-224-in21k`, a pre-trained Vision Transformer model known for its superior image classification capabilities compared to traditional CNNs.
-   **Multi-Class Classification**: Capable of distinguishing between 5 complex human activities:
    -   `sitting`
    -   `listening_to_music`
    -   `texting`
    -   `calling`
    -   `using_laptop`
-   **Production Ready**: Includes an inference pipeline ready for deployment on real-world images.

## Prerequisites

To run this project, you will need Python installed along with the following libraries:

```bash
pip install transformers torch pillow
```

## Getting Started

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd Human_Pose_Classification
    ```

2.  **Run the Jupyter Notebook**:
    Launch `Image Classification.ipynb` to explore the training process, evaluation metrics, and inference examples.

## Model Architecture

The core of this project is the Vision Transformer (ViT). Unlike convolutional neural networks that process pixels, ViT splits an image into fixed-size patches, linearly embeds each of them, add position embeddings, and feeds the resulting sequence of vectors to a standard Transformer encoder. This allows the model to capture global dependencies within the image effectively.

## Usage

Here is a quick example of how to use the trained model for inference:

```python
from transformers import pipeline, AutoImageProcessor

# Load model and processor
model_ckpt = "google/vit-base-patch16-224-in21k" # Or path to your fine-tuned model
image_processor = AutoImageProcessor.from_pretrained(model_ckpt, use_fast=True)
pipe = pipeline('image-classification', model='vit-human-pose-classification', image_processor=image_processor)

# Predict
url = "https://images.pexels.com/photos/1755385/pexels-photo-1755385.jpeg"
output = pipe(url)

print(output)
# Output format:
# [{'label': 'sitting', 'score': 0.92}, {'label': 'listening_to_music', 'score': 0.68}, ...]
```

## Performance

The model's performance has been evaluated using a confusion matrix on a validation set, demonstrating high accuracy across all classes. Detailed performance metrics and visualizations can be found in the accompanying Jupyter Notebook.
