import os
import gc
import threading
import time
import torch
import numpy as np
from PIL import Image
from datetime import datetime
import gradio as gr

# Import utility modules
from utils.text_model import generate_text_with_transformers, generate_text_with_llamacpp, check_llamacpp_available
from utils.vision_model import process_medical_image, load_vision_model

# Define models
TEXT_MODEL = "microsoft/Phi-3-mini-4k-instruct"  # Smaller model for better speed
VISION_MODEL = "flaviagiammarino/medsam-vit-base"

# Initialize threading event for cancellation
cancel_event = threading.Event()

# Cache for models
vision_model_cache = {"model": None, "processor": None}
transformers_model_cache = {"model": None, "tokenizer": None}

# Check if llamacpp is available
llamacpp_available = check_llamacpp_available()

# Function for text generation
def generate_pharmaceutical_response(query, use_llamacpp=False, progress=gr.Progress()):
    cancel_event.clear()
    
    if not query.strip():
        return "Please enter a question about medications or medical conditions."
    
    # Choose generation method based on user preference and availability
    if use_llamacpp and llamacpp_available:
        progress(0.1, desc="Loading llama.cpp model...")
        try:
            result = generate_text_with_llamacpp(
                query=query,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                cancel_event=cancel_event,
                progress_callback=lambda p, d: progress(0.1 + 0.8 * p, desc=d)
            )
            progress(1.0, desc="Done!")
            return result
        except Exception as e:
            return f"Error generating response with llama.cpp: {e}"
    else:
        # Fallback to transformers if llamacpp is not available or not chosen
        progress(0.1, desc="Loading transformers model...")
        try:
            # Get or load model and tokenizer
            if transformers_model_cache["model"] is None or transformers_model_cache["tokenizer"] is None:
                from utils.text_model import load_text_model
                model, tokenizer = load_text_model(TEXT_MODEL, quantize=torch.cuda.is_available())
                transformers_model_cache["model"] = model
                transformers_model_cache["tokenizer"] = tokenizer
            else:
                model = transformers_model_cache["model"]
                tokenizer = transformers_model_cache["tokenizer"]
                
            # Prepare the input with the correct format for Phi-3
            inputs = tokenizer(query, return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                model = model.cuda()
            
            # Generate with updated parameters and progress tracking
            progress(0.2, desc="Generating response...")
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,
                callback=lambda x: progress(0.2 + 0.7 * (x / 512), desc="Generating response...")
            )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            progress(1.0, desc="Done!")
            return result
        except Exception as e:
            return f"Error generating response with transformers: {e}"

# Function for image analysis
def analyze_medical_image_gradio(image, progress=gr.Progress()):
    if image is None:
        return None, "Please upload a medical image to analyze."
    
    progress(0.1, desc="Loading vision model...")
    try:
        # Get or load vision model
        if vision_model_cache["model"] is None or vision_model_cache["processor"] is None:
            model, processor = load_vision_model(VISION_MODEL)
            vision_model_cache["model"] = model
            vision_model_cache["processor"] = processor
        else:
            model = vision_model_cache["model"]
            processor = vision_model_cache["processor"]
    except Exception as e:
        return None, f"Error loading vision model: {e}"
    
    progress(0.3, desc="Processing image...")
    try:
        # FIX: Handle the return values correctly
        result_image, metadata, analysis_text = process_medical_image(
            image,
            model=model,
            processor=processor
        )
        
        progress(0.9, desc="Finalizing results...")
        
        # Format the analysis text for Gradio
        analysis_html = analysis_text.replace("##", "<h3>").replace("#", "</h3>")
        analysis_html = analysis_html.replace("**", "<b>").replace("**", "</b>")
        analysis_html = analysis_html.replace("\n\n", "<br><br>").replace("\n- ", "<br>• ")
        
        progress(1.0, desc="Done!")
        return result_image, analysis_html
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error analyzing image: {e}\n{error_details}")
        return None, f"Error analyzing image: {e}"

# Function for clinical interpretation
def generate_clinical_interpretation(image, analysis_html, use_llamacpp=False, progress=gr.Progress()):
    if image is None or not analysis_html:
        return "Please analyze an image first."
    
    # Extract information from the analysis HTML
    import re
    
    image_type = re.search(r"<b>Image Type</b>: ([^<]+)", analysis_html)
    region = re.search(r"<b>Region</b>: ([^<]+)", analysis_html)
    laterality = re.search(r"<b>Laterality</b>: ([^<]+)", analysis_html)
    findings = re.search(r"<b>Findings</b>:[^•]*• ([^<]+)", analysis_html)
    
    image_type = image_type.group(1) if image_type else "Medical X-ray"
    region = region.group(1) if region else "Unknown region"
    laterality = laterality.group(1) if laterality else "Unknown laterality"
    findings = findings.group(1) if findings else "No findings detected"
    
    # Create a detailed prompt
    prompt = f"""
    Provide a detailed clinical interpretation of a {image_type} with the following characteristics:
    - Segmented region: {region}
    - Laterality: {laterality}
    - Findings: {findings}
    
    Describe potential clinical significance, differential diagnoses, and recommendations. Include educational information about normal anatomy in this region and common pathologies that might be found. Be thorough but concise.
    """
    
    # Choose generation method based on user preference and availability
    if use_llamacpp and llamacpp_available:
        progress(0.1, desc="Loading llama.cpp model...")
        try:
            result = generate_text_with_llamacpp(
                query=prompt,
                max_tokens=768,
                temperature=0.7,
                top_p=0.9,
                cancel_event=cancel_event,
                progress_callback=lambda p, d: progress(0.1 + 0.8 * p, desc=d)
            )
            progress(1.0, desc="Done!")
            return result
        except Exception as e:
            return f"Error generating interpretation with llama.cpp: {e}"
    else:
        # Fallback to transformers if llamacpp is not available or not chosen
        progress(0.1, desc="Loading transformers model...")
        try:
            # Get or load model and tokenizer
            if transformers_model_cache["model"] is None or transformers_model_cache["tokenizer"] is None:
                from utils.text_model import load_text_model
                model, tokenizer = load_text_model(TEXT_MODEL, quantize=torch.cuda.is_available())
                transformers_model_cache["model"] = model
                transformers_model_cache["tokenizer"] = tokenizer
            else:
                model = transformers_model_cache["model"]
                tokenizer = transformers_model_cache["tokenizer"]
                
            result = generate_text_with_transformers(
                model=model,
                tokenizer=tokenizer,
                query=prompt,
                max_tokens=768,
                temperature=0.7,
                cancel_event=cancel_event,
                progress_callback=lambda p, d: progress(0.1 + 0.8 * p, desc=d)
            )
            progress(1.0, desc="Done!")
            return result
        except Exception as e:
            return f"Error generating interpretation with transformers: {e}"

# Function to cancel generation
def cancel_generation():
    cancel_event.set()
    return "Generation cancelled."

# Create Gradio Interface
with gr.Blocks(title="AI Pharma App", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💊 AI Pharma Assistant")
    gr.Markdown("This AI-powered application helps you access pharmaceutical information and analyze medical images.")
    
    # Set up model choice
    with gr.Accordion("Model Settings", open=False):
        use_llamacpp = gr.Checkbox(
            label="Use Llama.cpp for faster inference", 
            value=llamacpp_available,
            interactive=llamacpp_available
        )
        if not llamacpp_available:
            gr.Markdown("⚠️ Llama.cpp is not available. Using transformers backend.")
            gr.Markdown("To enable llama.cpp, run: pip install llama-cpp-python --no-cache-dir")
            gr.Markdown("Then run: python download_model.py")
    
    with gr.Tab("Drug Information"):
        gr.Markdown("### 📋 Drug Information Assistant")
        gr.Markdown("Ask questions about medications, treatments, side effects, drug interactions, or any other pharmaceutical information.")
        
        with gr.Row():
            with gr.Column():
                drug_query = gr.Textbox(
                    label="Type your question about medications here:",
                    placeholder="Example: What are the common side effects of Lisinopril? How does Metformin work?",
                    lines=4
                )
                
                with gr.Row():
                    drug_submit = gr.Button("Ask AI", variant="primary")
                    drug_cancel = gr.Button("Cancel", variant="secondary")
                
            drug_response = gr.Markdown(label="AI Response")
        
        drug_submit.click(
            generate_pharmaceutical_response,
            inputs=[drug_query, use_llamacpp],
            outputs=drug_response
        )
        drug_cancel.click(
            cancel_generation,
            inputs=[],
            outputs=drug_response
        )
    
    with gr.Tab("Medical Image Analysis"):
        gr.Markdown("### 🔍 Medical Image Analyzer")
        gr.Markdown("Upload medical images such as X-rays, CT scans, MRIs, ultrasound images, or other diagnostic visuals. The AI will automatically analyze important structures.")
        
        with gr.Row():
            with gr.Column(scale=2):
                image_input = gr.Image(
                    label="Upload a medical image",
                    type="pil"
                )
                image_submit = gr.Button("Analyze Image", variant="primary")
                
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.TabItem("Visualization"):
                        image_output = gr.Image(label="Segmentation Result")
                    with gr.TabItem("Analysis"):
                        analysis_output = gr.HTML(label="Analysis Results")
                
                with gr.Accordion("Clinical Interpretation", open=False):
                    interpret_button = gr.Button("Generate Clinical Interpretation")
                    interpretation_output = gr.Markdown()
                    
        image_submit.click(
            analyze_medical_image_gradio,
            inputs=[image_input],
            outputs=[image_output, analysis_output]
        )
        
        interpret_button.click(
            generate_clinical_interpretation,
            inputs=[image_input, analysis_output, use_llamacpp],
            outputs=interpretation_output
        )
    
    with gr.Accordion("System Information", open=False):
        gr.Markdown(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
        gr.Markdown(f"**Device:** {'GPU' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            gr.Markdown(f"**GPU:** {torch.cuda.get_device_name(0)}")
        gr.Markdown(f"**Text Model:** {TEXT_MODEL}")
        gr.Markdown(f"**Vision Model:** {VISION_MODEL}")
        gr.Markdown("**Llama.cpp:** " + ("Available" if llamacpp_available else "Not available"))
        gr.Markdown("""
        **Note:** This application is for educational purposes only and should not be used for medical diagnosis.
        The analysis provided is automated and should be reviewed by a qualified healthcare professional.
        """)

    gr.Markdown("""
    <div style="text-align: center; margin-top: 20px; padding: 10px; border-top: 1px solid #ddd;">
        <p>AI Pharma Assistant — Built with open-source models</p>
        <p>Powered by Phi-3 and MedSAM | © 2025</p>
    </div>
    """, elem_id="footer")

# Launch the app with customization for Hugging Face Spaces
if __name__ == "__main__":
    # Check if running on HF Spaces
    if os.environ.get('SPACE_ID'):
        demo.launch(server_name="0.0.0.0", share=False)
    else:
        # Local development
        demo.launch(share=False)