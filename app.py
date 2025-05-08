import streamlit as st
import torch
import os
import time
import threading
import numpy as np
from PIL import Image
import base64
from datetime import datetime

# Import utility modules
from utils.text_model import load_text_model, format_prompt, generate_text
from utils.vision_model import load_vision_model, process_medical_image

# Set page config
st.set_page_config(
    page_title="AI Pharma App",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define models - simplified to one each
TEXT_MODEL = "microsoft/Phi-4-mini-instruct"
VISION_MODEL = "flaviagiammarino/medsam-vit-base"

# Define custom CSS
def get_custom_css():
    return """
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #0f52ba;
            text-align: center;
            margin-bottom: 1rem;
        }
        .subheader {
            font-size: 1.8rem;
            color: #1e88e5;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .card {
            padding: 1.5rem;
            border-radius: 0.7rem;
            background-color: #f8f9fa;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
            color: #212529;
        }
        .highlight-text {
            color: #0f52ba;
            font-weight: bold;
        }
        .info-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #e3f2fd;
            border-left: 4px solid #1e88e5;
            margin-bottom: 1rem;
            color: #0d47a1;
        }
        .success-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #e8f5e9;
            border-left: 4px solid #4caf50;
            margin-bottom: 1rem;
            color: #2e7d32;
        }
        .warning-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #fff8e1;
            border-left: 4px solid #ffc107;
            margin-bottom: 1rem;
            color: #ff6f00;
        }
        .stButton>button {
            background-color: #1e88e5;
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            font-weight: bold;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #0d47a1;
            color: white;
        }
        /* Custom loader */
        .loader-wrapper {
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 1rem 0;
        }
        .loader {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #1e88e5;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin-right: 1rem;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        /* Footer styling */
        .footer {
            text-align: center;
            padding: 1rem 0;
            font-size: 0.9rem;
            color: #6c757d;
            border-top: 1px solid #e9ecef;
            margin-top: 2rem;
        }
        /* Pharma colors for medical theme */
        .label-blue {
            color: #0d47a1;
            font-weight: bold;
        }
        .label-red {
            color: #d32f2f;
            font-weight: bold;
        }
        .stTextArea textarea {
            border: 1px solid #1e88e5;
            border-radius: 0.5rem;
            background-color: white;
            color: #212529;
        }
        .stTextArea textarea:focus {
            border-color: #0d47a1;
            box-shadow: 0 0 0 0.2rem rgba(14, 71, 161, 0.25);
        }
        /* Fix for dark mode text color issues */
        div.stMarkdown p {
            color: #f8f9fa;  
        }
        div.card p, div.info-box p, div.success-box p, div.warning-box p {
            color: inherit !important;
        }
        /* Chat history styling fixes */
        div[style*="background-color: #e3f2fd"] {
            color: #0d47a1 !important;
            background-color: #e3f2fd !important;
        }
        div[style*="background-color: #f1f3f4"] {
            color: #212529 !important;
            background-color: #f1f3f4 !important;
        }
        /* Tab styling */
        .stTabs button {
            color: #1e88e5 !important;
        }
        .stTabs button[aria-selected="true"] {
            color: #0d47a1 !important;
            border-bottom-color: #0d47a1 !important;
        }
        /* File uploader styling */
        .css-1offfwp {
            color: #212529 !important;
        }
        .uploadedFile {
            color: #212529 !important;
        }
        /* Make standard text more visible */
        .stTextInput>div>div>input {
            color: #212529 !important;
            background-color: white;
        }
        .stNumberInput>div>div>input {
            color: #212529 !important;
            background-color: white;
        }
    </style>
    """

# Function to create a custom box
def custom_box(content, box_type="info"):
    if box_type == "info":
        st.markdown(f'<div class="info-box">{content}</div>', unsafe_allow_html=True)
    elif box_type == "success":
        st.markdown(f'<div class="success-box">{content}</div>', unsafe_allow_html=True)
    elif box_type == "warning":
        st.markdown(f'<div class="warning-box">{content}</div>', unsafe_allow_html=True)

# Initialize session state for cancellation
if "cancel_generation" not in st.session_state:
    st.session_state.cancel_generation = threading.Event()

# Add chat history to session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Cache to track if models are loaded
if "text_model" not in st.session_state:
    st.session_state.text_model = None

if "vision_model" not in st.session_state:
    st.session_state.vision_model = None

# Function to load text model
def load_text_model_if_needed():
    if st.session_state.text_model is None:
        with st.spinner("Loading text model..."):
            try:
                # Don't use quantization on CPU-only systems
                use_quantization = torch.cuda.is_available()
                model, tokenizer = load_text_model(TEXT_MODEL, quantize=use_quantization)
                st.session_state.text_model = (model, tokenizer)
            except Exception as e:
                st.error(f"Error loading text model: {e}")
                return None
    return st.session_state.text_model

# Function to load vision model
def load_vision_model_if_needed():
    if st.session_state.vision_model is None:
        with st.spinner("Loading vision model..."):
            try:
                model, processor = load_vision_model(VISION_MODEL)
                st.session_state.vision_model = (model, processor)
            except Exception as e:
                st.error(f"Error loading vision model: {e}")
                return None
    return st.session_state.vision_model

# UI Components - Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# App header with logo and title
st.markdown('<div class="main-header">💊 AI Pharma Assistant</div>', unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="info-box">
This AI-powered application helps you access pharmaceutical information and analyze medical images. 
Using advanced language and vision models, it provides accurate information about medications and 
can identify structures in medical scans.
</div>
""", unsafe_allow_html=True)

# Sidebar - simplified with no model selection
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/pharmacy-logo-concept_23-2148452251.jpg", width=100)
    st.header("⚙️ Model Settings")
    
    # Only model parameters remain
    st.markdown("**Using Phi-4-mini-Instruct for text**")
    st.markdown("**Using MedSAM for vision**")
    
    st.markdown("---")
    st.header("🔧 Parameters")
    
    # Model parameters
    max_tokens = st.slider(
        "Response Length",
        128, 4096, 1024,
        help="Maximum number of tokens in the AI's response. Higher values produce longer answers."
    )
    
    temperature = st.slider(
        "Creativity",
        0.1, 1.5, 0.7, 0.1,
        help="Higher values make output more creative but potentially less accurate"
    )
    
    # Memory management
    st.markdown("---")
    st.header("🧹 Maintenance")
    
    if st.button("Clear Cache & History", help="Free up memory by clearing model cache and conversation history"):
        st.session_state.text_model = None
        st.session_state.vision_model = None
        st.session_state.chat_history = []
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        st.success("Memory cache and history cleared!")
    
    # System information
    st.markdown("---")
    st.markdown("### 💻 System Info")
    st.markdown(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
    st.markdown(f"**Device:** {'GPU' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        st.markdown(f"**GPU:** {torch.cuda.get_device_name(0)}")

# Create tabs for different functionality
tab1, tab2 = st.tabs(["💬 Drug Information", "🔬 Medical Image Analysis"])

# Tab 1: Drug Information
with tab1:
    st.markdown('<div class="subheader">📋 Drug Information Assistant</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    Ask questions about medications, treatments, side effects, drug interactions, or any other pharmaceutical information.
    The AI will provide accurate and relevant information based on its training data.
    </div>
    """, unsafe_allow_html=True)
    
    # Chat interface
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.markdown(f'<div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px;"><span class="label-blue">You:</span> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background-color: #f1f3f4; padding: 10px; border-radius: 10px; margin-bottom: 10px;"><span style="color: #d32f2f; font-weight: bold;">AI:</span> <span style="color: #d32f2f;">{message["content"]}</span></div>', unsafe_allow_html=True)
    
    # User input
    query = st.text_area(
        "Type your question about medications here:",
        height=100,
        placeholder="Example: What are the common side effects of Lisinopril? How does Metformin work?"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        submit_button = st.button("Ask AI", key="submit_drug_query")
    with col2:
        cancel_button = st.button("Cancel", key="cancel_drug_query", help="Stop the AI response generation")
    
    if cancel_button:
        st.session_state.cancel_generation.set()
        st.warning("Generation cancelled.")
    
    if submit_button and query:
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Reset cancellation event
        st.session_state.cancel_generation.clear()
        
        # Load the text model
        model_result = load_text_model_if_needed()
        
        if model_result:
            model, tokenizer = model_result
            
            # Format the prompt
            prompt = format_prompt(tokenizer, query)
            
            # Response container
            response_container = st.empty()
            
            # Function to update the streamed response
            def update_response(text):
                response_container.markdown(f'<div style="background-color: #f1f3f4; padding: 10px; border-radius: 10px;"><span style="color: #d32f2f; font-weight: bold;">AI:</span> <span style="color: #d32f2f;">{text}</span></div>', unsafe_allow_html=True)
                # Update chat history (last message)
                if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
                    st.session_state.chat_history[-1]["content"] = text
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": text})
            
            # Generate response with loader animation
            st.markdown('<div class="loader-wrapper"><div class="loader"></div><span>AI is thinking...</span></div>', unsafe_allow_html=True)
            
            response = generate_text(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                stream_handler=update_response,
                cancel_event=st.session_state.cancel_generation
            )
            
            # Add final response to chat history if not already added
            if not st.session_state.chat_history or st.session_state.chat_history[-1]["role"] != "assistant":
                st.session_state.chat_history.append({"role": "assistant", "content": response})
# Tab 2: Medical Image Analysis
with tab2:
    st.markdown('<div class="subheader">🔍 Medical Image Analyzer</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    Upload medical images such as X-rays, CT scans, MRIs, ultrasound images, or other diagnostic visuals.
    The AI will automatically identify and analyze important structures in the image and provide insights.
    </div>
    """, unsafe_allow_html=True)
    
    # Help box
    custom_box("""
    <strong>How to use:</strong>
    <ol>
        <li>Upload your medical image (X-ray, CT scan, MRI, etc.)</li>
        <li>Click "Analyze Image" to process</li>
        <li>Review the AI's assessment and segmentation</li>
    </ol>
    <strong>Note:</strong> This tool works best with X-ray images. The analysis is automated and should be reviewed by a healthcare professional.
    """, box_type="info")
    
    # File uploader with custom styling
    st.markdown('<div style="padding: 20px; border: 2px dashed #1e88e5; border-radius: 10px; text-align: center;">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a medical image file",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Medical Image", use_column_width=True)
        
        # Process the image
        if st.button("Analyze Image", key="analyze_img_btn"):
            # Ensure vision model is loaded
            model_result = load_vision_model_if_needed()
            
            if model_result:
                with st.spinner("Analyzing medical image..."):
                    try:
                        result_image, metadata, analysis_text = process_medical_image(
                            image,
                            model_name=VISION_MODEL
                        )
                        
                        # Display results
                        st.markdown('<div class="subheader">Analysis Results</div>', unsafe_allow_html=True)
                        
                        # Create two columns for image and data
                        col1, col2 = st.columns([3, 2])
                        
                        with col1:
                            st.image(result_image, caption="Segmentation Result", use_column_width=True)
                        
                        with col2:
                            st.markdown(analysis_text)
                        
                        # Show technical details in expander
                        with st.expander("View Technical Details"):
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.markdown('<span class="highlight-text">Technical Metrics</span>', unsafe_allow_html=True)
                            st.markdown(f"**Coverage:** {metadata['mask_percentage']:.2f}% of image")
                            st.markdown(f"**Confidence:** {metadata['score']:.4f} (0-1 scale)")
                            st.markdown(f"**Region Size:** {metadata['size']['width']}×{metadata['size']['height']} pixels")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Option to generate clinical explanation
                        if st.checkbox("Generate Clinical Interpretation"):
                            # Load text model for explanation
                            text_model_result = load_text_model_if_needed()
                            
                            if text_model_result:
                                model, tokenizer = text_model_result
                                
                                # Create a detailed prompt based on the analysis
                                region = analysis_text.split("Region:")[1].split("\n")[0].strip()
                                laterality = analysis_text.split("Laterality:")[1].split("\n")[0].strip()
                                structures = analysis_text.split("Structures potentially visible:")[1].split("\n")[0].strip()
                                image_type = analysis_text.split("Image Type:")[1].split("\n")[0].strip()
                                
                                prompt = f"""
                                Provide a detailed clinical interpretation of a {image_type} with the following characteristics:
                                - Segmented region: {region}
                                - Laterality: {laterality}
                                - Structures visible: {structures}
                                - The segmentation covers {metadata['mask_percentage']:.2f}% of the image
                                
                                Describe potential clinical findings, differential diagnoses, and recommendations. Include educational information about normal anatomy in this region and common pathologies that might be found. Be thorough but concise.
                                """
                                
                                # Container for the explanation
                                st.markdown('<div class="subheader">Clinical Interpretation</div>', unsafe_allow_html=True)
                                explanation_container = st.empty()
                                
                                # Stream handler
                                def update_explanation(text):
                                    explanation_container.markdown(f'<div class="card">{text}</div>', unsafe_allow_html=True)
                                
                                # Generate explanation
                                generate_text(
                                    model,
                                    tokenizer,
                                    format_prompt(tokenizer, prompt),
                                    max_new_tokens=max_tokens,
                                    temperature=temperature,
                                    stream_handler=update_explanation,
                                    cancel_event=st.session_state.cancel_generation
                                )
                    
                    except Exception as e:
                        st.error(f"Error analyzing image: {e}")
                        st.info("Try uploading a different medical image, preferably an X-ray.")
# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>AI Pharma Assistant — Built with open-source models</p>
    <p>Powered by Phi-4 and MedSAM | Deployed on Hugging Face Spaces</p>
</div>
""", unsafe_allow_html=True)