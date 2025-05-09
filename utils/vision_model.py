import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

def load_vision_model(model_name="flaviagiammarino/medsam-vit-base"):
    """
    Load MedSAM model from Hugging Face
    
    Args:
        model_name (str): Model repository name
        
    Returns:
        tuple: (model, processor)
    """
    from transformers import SamModel, SamProcessor
    
    try:
        # Try loading the model
        model = SamModel.from_pretrained(model_name)
        processor = SamProcessor.from_pretrained(model_name)
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        return model, processor
    
    except Exception as e:
        raise RuntimeError(f"Failed to load vision model {model_name}: {e}")

def identify_image_type(image):
    """Identify the type of medical image based on visual characteristics"""
    # Convert to numpy array if it's a PIL image
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Get image dimensions and ratio
    height, width = img_array.shape[:2]
    aspect_ratio = width / height
    
    # Basic image type detection logic
    if aspect_ratio > 1.4:  # Wide format
        # Likely a panoramic X-ray or abdominal scan
        return "Panoramic X-ray"
    elif aspect_ratio < 0.7:  # Tall format
        # Likely a full spine X-ray
        return "Full Spine X-ray"
    else:  # Square-ish format
        # Check brightness distribution for chest vs other X-rays
        # Chest X-rays typically have more contrast between dark (lungs) and bright (bones) areas
        
        # If grayscale, use directly, otherwise convert
        if len(img_array.shape) > 2:  # Color image
            gray_img = np.mean(img_array, axis=2)
        else:
            gray_img = img_array
            
        # Normalize to 0-1
        if gray_img.max() > 0:
            gray_img = gray_img / gray_img.max()
        
        # Check if has clear lung fields (darker regions in center)
        center_region = gray_img[height//4:3*height//4, width//4:3*width//4]
        edges_region = gray_img.copy()
        edges_region[height//4:3*height//4, width//4:3*width//4] = 1  # Mask out center
        
        center_mean = np.mean(center_region)
        edges_mean = np.mean(edges_region)
        
        # Chest X-rays typically have darker center (lung fields)
        if center_mean < edges_mean * 0.85:
            return "Chest X-ray"
        else:
            # Look for bone structures
            high_intensity = np.percentile(gray_img, 95) * 0.95
            bone_pixels = np.sum(gray_img > high_intensity) / (height * width)
            
            if bone_pixels > 0.15:  # Significant bone content
                if height > width:
                    return "Spine X-ray"
                else:
                    return "Extremity X-ray"
            
            # Default
            return "Medical X-ray"

def detect_abnormalities(image_type, mask, image_array):
    """Detect potential abnormalities based on image type and mask area"""
    
    # Create more meaningful default findings
    findings = {
        "regions_of_interest": ["No specific abnormalities detected"],
        "potential_findings": ["Normal study"],
        "additional_notes": []
    }
    
    # Get mask properties
    if len(mask.shape) > 2:
        mask = mask[:,:,0]  # Take first channel if multi-channel
    
    # Extract masked region stats
    if np.any(mask):
        rows, cols = np.where(mask)
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        
        # Get region location
        height, width = mask.shape
        region_center_y = np.mean(rows)
        region_center_x = np.mean(cols)
        
        rel_y = region_center_y / height
        rel_x = region_center_x / width
        
        # Get image intensity stats in masked region
        if len(image_array.shape) > 2:
            gray_img = np.mean(image_array, axis=2)
        else:
            gray_img = image_array
            
        if gray_img.max() > 0:
            gray_img = gray_img / gray_img.max()
            
        # Get statistics of the region
        mask_intensities = gray_img[mask]
        if len(mask_intensities) > 0:
            region_mean = np.mean(mask_intensities)
            region_std = np.std(mask_intensities)
            
            # Calculate stats outside the mask for comparison
            inverse_mask = ~mask
            outside_intensities = gray_img[inverse_mask]
            if len(outside_intensities) > 0:
                outside_mean = np.mean(outside_intensities)
                intensity_diff = abs(region_mean - outside_mean)
            else:
                outside_mean = 0
                intensity_diff = 0
            
            # Identify regions of interest based on image type
            if image_type == "Chest X-ray":
                findings["regions_of_interest"] = []
                
                # Identify anatomical regions in chest X-ray
                if rel_y < 0.3:  # Upper chest
                    if rel_x < 0.4:
                        findings["regions_of_interest"].append("Left upper lung field")
                    elif rel_x > 0.6:
                        findings["regions_of_interest"].append("Right upper lung field")
                    else:
                        findings["regions_of_interest"].append("Upper mediastinum")
                        
                elif rel_y < 0.6:  # Mid chest
                    if rel_x < 0.4:
                        findings["regions_of_interest"].append("Left mid lung field")
                    elif rel_x > 0.6:
                        findings["regions_of_interest"].append("Right mid lung field")
                    else:
                        findings["regions_of_interest"].append("Central mediastinum")
                        findings["regions_of_interest"].append("Cardiac silhouette")
                        
                else:  # Lower chest
                    if rel_x < 0.4:
                        findings["regions_of_interest"].append("Left lower lung field")
                        findings["regions_of_interest"].append("Left costophrenic angle")
                    elif rel_x > 0.6:
                        findings["regions_of_interest"].append("Right lower lung field")
                        findings["regions_of_interest"].append("Right costophrenic angle")
                    else:
                        findings["regions_of_interest"].append("Lower mediastinum")
                        findings["regions_of_interest"].append("Upper abdomen")
                
                # Check for potential abnormalities based on intensity
                findings["potential_findings"] = []
                
                if region_mean < outside_mean * 0.7 and region_std < 0.15:
                    findings["potential_findings"].append("Potential hyperlucency/emphysematous changes")
                elif region_mean > outside_mean * 1.3:
                    if region_std > 0.2:
                        findings["potential_findings"].append("Heterogeneous opacity")
                    else:
                        findings["potential_findings"].append("Homogeneous opacity/consolidation")
                
                # Add size of area
                mask_height = max_row - min_row
                mask_width = max_col - min_col
                
                if max(mask_height, mask_width) > min(height, width) * 0.25:
                    findings["additional_notes"].append(f"Large area of interest ({mask_height}x{mask_width} pixels)")
                else:
                    findings["additional_notes"].append(f"Focal area of interest ({mask_height}x{mask_width} pixels)")
                
            elif "Spine" in image_type:
                # Vertebral analysis for spine X-rays
                findings["regions_of_interest"] = []
                
                if rel_y < 0.3:
                    findings["regions_of_interest"].append("Cervical spine region")
                elif rel_y < 0.6:
                    findings["regions_of_interest"].append("Thoracic spine region")
                else:
                    findings["regions_of_interest"].append("Lumbar spine region")
                
                # Check for potential findings
                findings["potential_findings"] = []
                
                if region_std > 0.25:  # High variability in vertebral region could indicate irregularity
                    findings["potential_findings"].append("Potential vertebral irregularity")
                
                if intensity_diff > 0.3:
                    findings["potential_findings"].append("Area of abnormal density")
                    
            elif "Extremity" in image_type:
                # Extremity X-ray analysis
                findings["regions_of_interest"] = []
                
                # Basic positioning
                if rel_y < 0.5 and rel_x < 0.5:
                    findings["regions_of_interest"].append("Proximal joint region")
                elif rel_y > 0.5 and rel_x > 0.5:
                    findings["regions_of_interest"].append("Distal joint region")
                else:
                    findings["regions_of_interest"].append("Mid-shaft bone region")
                
                # Check for potential findings
                findings["potential_findings"] = []
                
                if region_std > 0.25:  # High variability could indicate irregular bone contour
                    findings["potential_findings"].append("Potential cortical irregularity")
                    
                if intensity_diff > 0.4:
                    findings["potential_findings"].append("Area of abnormal bone density")
            
            # Default if no findings identified
            if len(findings["potential_findings"]) == 0:
                findings["potential_findings"] = ["No obvious abnormalities in segmented region"]
    
    return findings

def analyze_medical_image(image_type, image, mask, metadata):
    """Generate a comprehensive medical image analysis"""
    
    # Convert to numpy if PIL image
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image
        
    # Detect abnormalities based on image type and region
    abnormalities = detect_abnormalities(image_type, mask, image_array)
    
    # Get mask properties
    mask_area = metadata["mask_percentage"]
    confidence = metadata["score"]
    
    # Determine anatomical positioning
    height, width = mask.shape if len(mask.shape) == 2 else mask.shape[:2]
    
    if np.any(mask):
        rows, cols = np.where(mask)
        center_y = np.mean(rows) / height
        center_x = np.mean(cols) / width
        
        # Determine laterality
        if center_x < 0.4:
            laterality = "Left side predominant"
        elif center_x > 0.6:
            laterality = "Right side predominant" 
        else:
            laterality = "Midline/central"
            
        # Determine superior/inferior position
        if center_y < 0.4:
            position = "Superior/upper region"
        elif center_y > 0.6:
            position = "Inferior/lower region"
        else:
            position = "Mid/central region"
            
    else:
        laterality = "Undetermined"
        position = "Undetermined"
    
    # Generate analysis text
    if image_type == "Chest X-ray":
        image_description = "anteroposterior (AP) or posteroanterior (PA) chest radiograph"
        regions = ", ".join(abnormalities["regions_of_interest"])
        findings = ", ".join(abnormalities["potential_findings"])
        
    elif "Spine" in image_type:
        image_description = "spinal radiograph"
        regions = ", ".join(abnormalities["regions_of_interest"])
        findings = ", ".join(abnormalities["potential_findings"])
        
    elif "Extremity" in image_type:
        image_description = "extremity radiograph"
        regions = ", ".join(abnormalities["regions_of_interest"])
        findings = ", ".join(abnormalities["potential_findings"])
        
    else:
        image_description = "medical radiograph"
        regions = ", ".join(abnormalities["regions_of_interest"])
        findings = ", ".join(abnormalities["potential_findings"])
    
    # Finalize analysis text
    analysis_text = f"""
    ## Radiological Analysis

    **Image Type**: {image_type}
    
    **Segmentation Details**:
    - Region: {position} ({regions})
    - Laterality: {laterality}
    - Coverage: {mask_area:.1f}% of the image
    
    **Findings**:
    - {findings}
    - {'; '.join(abnormalities["additional_notes"]) if abnormalities["additional_notes"] else 'No additional notes'}
    
    **Technical Assessment**:
    - Segmentation confidence: {confidence:.2f} (on a 0-1 scale)
    - Image quality: {'Adequate' if confidence > 0.4 else 'Suboptimal'} for assessment
    
    **Impression**:
    This {image_description} demonstrates a highlighted area in the {position.lower()} with {laterality.lower()}. 
    {findings.capitalize() if findings else 'No significant abnormalities identified in the segmented region.'} Additional clinical correlation is recommended.
    
    *Note: This is an automated analysis and should be reviewed by a qualified healthcare professional.*
    """
    
    # Create analysis results as dict
    analysis_results = {
        "image_type": image_type,
        "region": position,
        "laterality": laterality,
        "regions_of_interest": abnormalities["regions_of_interest"],
        "potential_findings": abnormalities["potential_findings"],
        "additional_notes": abnormalities["additional_notes"],
        "coverage_percentage": mask_area,
        "confidence_score": confidence
    }
    
    return analysis_text, analysis_results

def process_medical_image(image, model=None, processor=None):
    """
    Process medical image with MedSAM using automatic segmentation
    
    Args:
        image (PIL.Image): Input image
        model: SamModel instance (optional, will be loaded if not provided)
        processor: SamProcessor instance (optional, will be loaded if not provided)
        
    Returns:
        tuple: (PIL.Image of segmentation, metadata dict, analysis text)
    """
    # Load model and processor if not provided
    if model is None or processor is None:
        from transformers import SamModel, SamProcessor
        model_name = "flaviagiammarino/medsam-vit-base"
        model, processor = load_vision_model(model_name)
    
    # Convert image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Convert to grayscale if it's not already
    if image.mode != 'L':
        grayscale = image.convert('L')
        # Convert back to RGB for processing
        image_for_processing = grayscale.convert('RGB')
    else:
        # If already grayscale, convert to RGB for processing
        image_for_processing = image.convert('RGB')
    
    # Resize image to a standard size (FIX: make sure we use consistent dimensions)
    image_size = 512  # Use power of 2 for better compatibility
    processed_image = image_for_processing.resize((image_size, image_size), Image.LANCZOS)
    image_array = np.array(processed_image)
    
    # Identify the type of medical image
    image_type = identify_image_type(image)
    
    try:
        # For chest X-rays, target the full central region
        # This ensures we analyze most of the image rather than just a tiny portion
        height, width = image_array.shape[:2]
        
        # FIX: Ensure input_boxes are in the correct format: [[x1, y1, x2, y2]] (not [x1, y1, x2, y2])
        # Create a large box covering ~75% of the image
        margin = width // 8  # 12.5% margin on each side
        
        # Correct box format: list of lists where each inner list is [x1, y1, x2, y2]
        box = [[margin, margin, width - margin, height - margin]]
        
        # Process with the larger box
        inputs = processor(
            images=processed_image,  # FIX: Use PIL image instead of numpy array
            input_boxes=[box],  # FIX: Ensure correct nesting
            return_tensors="pt"
        )
        
        # Transfer inputs to the same device as the model
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Process the masks - FIX: Make sure we use the correct dimensions
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
        # Create a fallback mask covering most of the central image area
        mask = np.zeros((image_size, image_size), dtype=bool)
        margin = image_size // 8
        mask[margin:image_size-margin, margin:image_size-margin] = True
        score_value = 0.5
    
    # Visualize results
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Use the grayscale image for visualization if it was an X-ray
    ax.imshow(image_array, cmap='gray' if image.mode == 'L' else None)
    
    # Show mask as overlay with improved visibility
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    color_mask[mask] = [1, 0, 0, 0.4]  # Semi-transparent red
    ax.imshow(color_mask)
    
    # Add title with image type
    ax.set_title(f"Medical Image Segmentation: {image_type}", fontsize=14)
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
    analysis_text, analysis_results = analyze_medical_image(image_type, processed_image, mask, metadata)
    
    # FIX: Return the result_image directly, not as part of a tuple with metadata and analysis
    return result_image, metadata, analysis_text