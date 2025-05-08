# AI Pharma App

An open-source application for pharmaceutical information and medical image analysis, built entirely with free and open-source models.

## Features

- **Drug Information Queries**: Ask questions about medications, treatments, and medical conditions using Phi-4
- **Medical Image Analysis**: Analyze X-rays, CT scans, MRIs and other medical images using MedSAM
- **Enhanced UI**: Beautiful Streamlit interface with medical theme
- **Zero-Cost Deployment**: Designed to run on free tiers of cloud platforms like Hugging Face Spaces

## Models Used

- **Text Generation**: Microsoft Phi-4-mini models (Instruct and Reasoning variants)
- **Medical Image Segmentation**: MedSAM (Medical Segment Anything Model)

## Deployment

### Local Setup

1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

### Hugging Face Spaces Deployment

1. Create a new Space on Hugging Face
2. Select Streamlit as the SDK
3. Upload the repository files
4. The app will automatically deploy on the free tier (2 vCPUs, 16GB RAM)

## Memory Optimization

- Models are loaded with 4-bit quantization to fit within 16GB RAM limits
- Memory cache can be cleared via the sidebar button
- Efficient streaming generation to minimize memory usage during inference

## Limitations

- Inference speed is limited on CPU-only environments
- First query will be slow as models need to load
- Image segmentation may require manual prompting for best results

## License

This project is available under the MIT License.