import os
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer

# Global cache for model and tokenizer
MODEL_CACHE = {}

def load_text_model(model_name, quantize=False):
    """
    Load text model with appropriate configuration for CPU or GPU
    
    Args:
        model_name (str): Hugging Face model ID
        quantize (bool): Whether to use 4-bit quantization (only works with GPU)
        
    Returns:
        tuple: (model, tokenizer)
    """
    # Check cache first
    cache_key = f"{model_name}_{quantize}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    
    # Only try quantization if CUDA is available
    if quantize and cuda_available:
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        except Exception as e:
            print(f"Quantization config creation failed: {e}")
            quantization_config = None
            quantize = False
    else:
        quantization_config = None
        quantize = False
    
    # Try loading the model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Try with quantization first if requested and available
        if quantize and quantization_config:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"Failed to load with quantization: {e}")
                quantize = False
        
        # If quantization is not used or failed, try standard loading
        if not quantize:
            # For CPU, just load without specifing dtype
            if not cuda_available:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # Try different dtypes for GPU
                for dtype in (torch.bfloat16, torch.float16, torch.float32):
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=dtype,
                            device_map="auto",
                            trust_remote_code=True
                        )
                        break
                    except Exception as e:
                        if dtype == torch.float32:
                            # Last resort: try without specifying dtype
                            model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                device_map="auto",
                                trust_remote_code=True
                            )
        
        # Cache the loaded model and tokenizer
        MODEL_CACHE[cache_key] = (model, tokenizer)
        return model, tokenizer
    
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")

def format_prompt(tokenizer, query):
    """
    Format prompt according to model's requirements
    
    Args:
        tokenizer: The model tokenizer
        query (str): User query
        
    Returns:
        str: Formatted prompt
    """
    enhanced_query = f"Please answer this question about pharmaceuticals or medical topics.\n\nQuestion: {query}"
    
    # Use chat template if available
    if hasattr(tokenizer, "apply_chat_template") and callable(getattr(tokenizer, "apply_chat_template")):
        messages = [{"role": "user", "content": enhanced_query}]
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return formatted
        except:
            # Fallback if chat template fails
            pass
    
    # Simple formatting fallback
    return f"User: {enhanced_query}\nAssistant:"

def generate_text(model, tokenizer, prompt, max_new_tokens=1024, temperature=0.7, 
                  top_p=0.9, repetition_penalty=1.1, stream_handler=None, 
                  cancel_event=None):
    """
    Generate text from the model with streaming support
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt (str): Formatted prompt
        max_new_tokens (int): Maximum tokens to generate
        temperature (float): Temperature for sampling
        top_p (float): Top-p sampling parameter
        repetition_penalty (float): Penalty for repetition
        stream_handler (callable): Function to handle streamed tokens
        cancel_event (threading.Event): Event to signal cancellation
        
    Returns:
        str: Generated response
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Setup streamer if streaming is requested
    streamer = None
    if stream_handler:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = {
        "input_ids": inputs.input_ids,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "do_sample": temperature > 0.1,
    }
    
    if streamer:
        generation_kwargs["streamer"] = streamer
    
    # Generate in a thread if streaming
    if streamer:
        thread = threading.Thread(
            target=model.generate, 
            kwargs=generation_kwargs
        )
        thread.start()
        
        # Stream the tokens
        response_text = ""
        for new_text in streamer:
            if cancel_event and cancel_event.is_set():
                break
            response_text += new_text
            if stream_handler:
                stream_handler(response_text)
        
        return response_text
    else:
        # Generate without streaming
        output = model.generate(**generation_kwargs)
        return tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)