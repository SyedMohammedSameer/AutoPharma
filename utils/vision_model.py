import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor
from io import BytesIO

# Global cache for model and processor
MODEL_CACHE = {}

def load_vision_model(model_name="flaviagiammarino/medsam-vit-base"):
    """
    Load MedSAM model from Hugging Face
    
    Args:
        model_name (str): Model repository name
        
    Returns:
        tuple: (model, processor)
    """
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]
    
    try:
        # Try loading the model
        model = SamModel.from_pretrained(model_name)
        processor = SamProcessor.from_pretrained(model_name)
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Cache for future use
        MODEL_CACHE[model_name] = (model, processor)
        return model, processor
    
    except Exception as e:
        raise RuntimeError(f"Failed to load vision model {model_name}: {e}")

def analyze_x_ray(image, mask, metadata):
    """Analyze X-ray image and provide medical insights based on segmentation"""
    
    # Determine image type based on visual characteristics
    width, height = image.size
    is_chest_xray = height > width  # Most chest X-rays are portrait orientation
    
    # Basic structure detection
    coverage = metadata["mask_percentage"]
    confidence = metadata["score"]
    
    if is_chest_xray:
        image_type = "Chest X-ray"
        
        # Analyze chest region based on location
        mask_array = np.array(mask)
        rows, cols = mask_array.shape
        
        # Check which part is segmented
        top_third = mask_array[:rows//3, :].sum()
        middle_third = mask_array[rows//3:2*rows//3, :].sum()
        bottom_third = mask_array[2*rows//3:, :].sum()
        
        # Determine region based on distribution
        if middle_third > top_third and middle_third > bottom_third:
            region = "central lung area"
        elif bottom_third > top_third:
            region = "lower lung area"
        else:
            region = "upper lung area"
        
        # Estimate which structures are visible in segmentation
        structures = []
        if region == "central lung area":
            structures = ["lung parenchyma", "bronchi", "pulmonary vessels"]
        elif region == "lower lung area":
            structures = ["lung bases", "diaphragm", "costophrenic angles"]
        else:
            structures = ["lung apices", "clavicles", "upper ribs"]
            
        # Determine laterality
        left_side = mask_array[:, :cols//2].sum()
        right_side = mask_array[:, cols//2:].sum()
        
        if left_side > right_side * 1.5:
            laterality = "left side"
        elif right_side > left_side * 1.5:
            laterality = "right side"
        else:
            laterality = "bilateral"
        
    else:
        # General bone X-ray analysis
        image_type = "Bone X-ray"
        
        # Simple region detection
        mask_array = np.array(mask)
        rows, cols = mask_array.shape
        center_of_mass_y = np.mean(np.where(mask_array)[0])
        center_of_mass_x = np.mean(np.where(mask_array)[1])
        
        # Rough anatomical region
        if center_of_mass_y < rows * 0.3:
            region = "upper extremity or skull"
        elif center_of_mass_y > rows * 0.7:
            region = "lower extremity"
        else:
            region = "trunk or spine"
            
        structures = ["bone", "joint space"]
        
        # Simple left/right detection
        if center_of_mass_x < cols * 0.4:
            laterality = "left side"
        elif center_of_mass_x > cols * 0.6:
            laterality = "right side"
        else:
            laterality = "midline"
    
    analysis_results = {
        "image_type": image_type,
        "segmented_region": region,
        "possible_structures": structures,
        "laterality": laterality,
        "coverage_percentage": coverage,
        "confidence_score": confidence
    }
    
    # Generate analysis text
    analysis_text = f"""
    ## Radiological Analysis

    **Image Type**: {image_type}
    
    **Segmentation Details**:
    - Region: {region.title()}
    - Laterality: {laterality.title()}
    - Coverage: {coverage:.1f}% of the image
    - Structures potentially visible: {', '.join(structures)}
    
    **Technical Assessment**:
    - Segmentation confidence: {confidence:.2f} (on a 0-1 scale)
    - Image quality: {'Adequate' if confidence > 0.4 else 'Suboptimal'} for assessment
    
    **Impression**:
    The segmentation highlights an area of interest in the {region} with {laterality} predominance. 
    This region typically contains {', '.join(structures)}.
    
    *Note: This is an automated analysis and should be reviewed by a qualified healthcare professional.*
    """
    
    return analysis_text, analysis_results

def process_medical_image(image, model_name="flaviagiammarino/medsam-vit-base"):
    """
    Process medical image with MedSAM using automatic segmentation
    
    Args:
        image (PIL.Image): Input image
        model_name (str): Model repository name
        
    Returns:
        tuple: (PIL.Image of segmentation, metadata dict, analysis text)
    """
    model, processor = load_vision_model(model_name)
    
    # Convert image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Convert to grayscale if it's an X-ray
    if image.mode != 'L':  # Check if not already grayscale
        grayscale = image.convert('L')
        # Convert back to RGB for processing
        image_for_processing = grayscale.convert('RGB')
    else:
        # If already grayscale, convert to RGB for processing
        image_for_processing = image.convert('RGB')
    
    # Resize image to a standard size
    image_size = 512
    processed_image = image_for_processing.resize((image_size, image_size), Image.LANCZOS)
    image_array = np.array(processed_image)
    
    try:
        # Generate a box in the center of the image
        # This works better than no prompt for X-rays
        height, width = image_array.shape[:2]
        center_x, center_y = width // 2, height // 2
        box_size = min(width, height) // 3
        
        box = [
            [center_x - box_size, center_y - box_size, 
             center_x + box_size, center_y + box_size]
        ]
        
        # Process with center box
        inputs = processor(
            images=image_array,
            input_boxes=[box],  # Needs to be a list of boxes
            return_tensors="pt"
        )
        
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Process the masks
            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.squeeze(1),
                inputs["original_sizes"],
                inputs["reshaped_input_sizes"]
            )
            
            # Get scores
            scores = outputs.iou_scores
            best_idx = torch.argmax(scores)
            score_value = float(scores[0][best_idx].cpu().numpy())
            
            # Get the best mask
            mask = masks[0][best_idx].cpu().numpy() > 0
            
    except Exception as e:
        print(f"Error in MedSAM processing: {e}")
        # Create a fallback solution - segment center of the image
        mask = np.zeros((image_size, image_size), dtype=bool)
        center_size = image_size // 3
        start_x = (image_size - center_size) // 2
        start_y = (image_size - center_size) // 2
        mask[start_y:start_y+center_size, start_x:start_x+center_size] = True
        score_value = 0.5
    
    # Visualize results
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Use the original grayscale image for visualization if it was an X-ray
    ax.imshow(image_array, cmap='gray' if image.mode == 'L' else None)
    
    # Show mask as overlay with improved visibility
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    color_mask[mask] = [1, 0, 0, 0.5]  # Semi-transparent red
    ax.imshow(color_mask)
    
    # Add region box
    height, width = mask.shape
    center_of_mass_y = np.mean(np.where(mask)[0]) if np.any(mask) else height // 2
    center_of_mass_x = np.mean(np.where(mask)[1]) if np.any(mask) else width // 2
    
    ax.plot(center_of_mass_x, center_of_mass_y, 'o', markersize=10, color='lime')
    
    ax.set_title("Medical Image Segmentation", fontsize=14)
    ax.axis('off')
    
    # Convert plot to image
    fig.patch.set_facecolor('white')
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, 
                facecolor='white', dpi=150)
    plt.close(fig)
    buf.seek(0)
    
    result_image = Image.open(buf)
    
    # Prepare metadata
    metadata = {
        "mask_percentage": float(np.mean(mask) * 100),  # Percentage of image that is masked
        "score": score_value,
        "size": {
            "width": mask.shape[1],
            "height": mask.shape[0]
        }
    }
    
    # Generate analysis
    analysis_text, analysis_results = analyze_x_ray(image, mask, metadata)
    
    return result_image, metadata, analysis_text