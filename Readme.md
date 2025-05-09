# AI Pharma App

An open-source application for pharmaceutical information and medical image analysis, built entirely with free and open-source models.

## Features

- **Drug Information Queries**: Ask questions about medications, treatments, and medical conditions using Phi-3-mini
- **Medical Image Analysis**: Analyze X-rays, CT scans, MRIs and other medical images using MedSAM
- **Dual Inference Engines**: Options for both transformers and llama.cpp backends for optimal performance
- **Enhanced UI**: Beautiful Gradio interface with medical theme
- **Zero-Cost Deployment**: Designed to run on free tiers of cloud platforms like Hugging Face Spaces

## Models Used

- **Text Generation**: Microsoft Phi-3-mini-4k-instruct
- **Medical Image Segmentation**: MedSAM (Medical Segment Anything Model)

## Setup & Installation

### Local Setup

1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. (Optional) For llama.cpp support:
   - Run `python download_model.py` to download and convert the model to GGUF format
4. Run the app: `python app.py`

### Hugging Face Spaces Deployment

1. Create a new Space on Hugging Face
2. Select Gradio as the SDK or Docker as Container
3. Upload the repository files
4. The app will automatically deploy

## Docker Deployment

```bash
# Build Docker image
docker build -t ai-pharma-app .

# Run container
docker run -p 7860:7860 ai-pharma-app