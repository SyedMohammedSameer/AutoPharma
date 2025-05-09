import os
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from transformers.generation.utils import DynamicCache
DynamicCache.get_max_length = DynamicCache.get_max_cache_shape


# Check if llama-cpp-python is available
def check_llamacpp_available():
    try:
        import llama_cpp
        return True
    except ImportError:
        return False

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
        
        # Fix for attention mask warning
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
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
                for dtype in (torch.float16, torch.float32):
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

def generate_text_with_transformers(model, tokenizer, query, max_tokens=512, temperature=0.7, 
                       top_p=0.9, repetition_penalty=1.1, cancel_event=None, 
                       progress_callback=None):
    """
    Generate text using the transformers pipeline
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        query (str): User query
        max_tokens (int): Maximum tokens to generate
        temperature (float): Temperature for sampling
        top_p (float): Top-p sampling parameter
        repetition_penalty (float): Penalty for repetition
        cancel_event (threading.Event): Event to signal cancellation
        progress_callback (callable): Function to report progress
        
    Returns:
        str: Generated response
    """
    # Format the prompt
    prompt = format_prompt(tokenizer, query)
    
    # Prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Update progress
    if progress_callback:
        progress_callback(0.2, "Starting generation...")
    
    try:
        from transformers import TextIteratorStreamer
        
        # Set up streamer for token-by-token generation
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Prepare generation parameters
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,  # Explicitly provide attention mask
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": temperature > 0.1,
            "streamer": streamer
        }
        
        # Start generation in a separate thread
        generation_thread = threading.Thread(
            target=model.generate, 
            kwargs=generation_kwargs
        )
        generation_thread.start()
        
        # Collect tokens as they're generated
        response_text = ""
        
        for i, new_text in enumerate(streamer):
            if cancel_event and cancel_event.is_set():
                break
            
            response_text += new_text
            
            # Update progress periodically
            if progress_callback and i % 5 == 0:
                progress_callback(0.3 + min(0.6, len(response_text) / 500), "Generating response...")
        
        return response_text
    
    except Exception as e:
        print(f"Streaming generation failed, falling back to standard generation: {e}")
        # Fallback to standard generation
        try:
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0.1,
            )
            
            # Decode and remove prompt
            prompt_length = inputs.input_ids.shape[1]
            response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
            
            return response
        except Exception as e2:
            return f"Error in text generation: {e2}"

# Global llamacpp model cache
LLAMA_MODEL = None

def load_llamacpp_model(model_path=None):
    """Load the llama.cpp model"""
    global LLAMA_MODEL
    
    # Return cached model if available
    if LLAMA_MODEL is not None:
        return LLAMA_MODEL
    
    try:
        from llama_cpp import Llama
        
        # Use provided path or check for model in predefined locations
        if model_path is None:
            # Try to find model in standard locations
            possible_paths = [
                "models/Phi-3-mini-4k-instruct.Q4_K_M.gguf",  # Local models dir
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "models/Phi-3-mini-4k-instruct.Q4_K_M.gguf"),  # Project root
                "/models/Phi-3-mini-4k-instruct.Q4_K_M.gguf",  # Docker container
                os.path.expanduser("~/.cache/huggingface/hub/models/Phi-3-mini-4k-instruct.Q4_K_M.gguf")  # HF cache
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError("Could not find GGUF model file. Please provide the path explicitly.")
        
        # Load the model
        LLAMA_MODEL = Llama(
            model_path=model_path,
            n_ctx=4096,  # Context window size
            n_batch=512,  # Batch size for prompt processing
            n_threads=4,  # CPU threads to use
            n_gpu_layers=0  # Set higher if you have GPU
        )
        
        return LLAMA_MODEL
        
    except ImportError:
        raise ImportError("llama-cpp-python is not installed. Please install it to use this functionality.")
    except Exception as e:
        raise RuntimeError(f"Failed to load llama.cpp model: {e}")

def generate_text_with_llamacpp(query, max_tokens=512, temperature=0.7, top_p=0.9, 
                   stop=None, cancel_event=None, progress_callback=None, model_path=None):
    """
    Generate text using llama.cpp
    
    Args:
        query (str): User query
        max_tokens (int): Maximum tokens to generate
        temperature (float): Temperature for sampling
        top_p (float): Top-p sampling parameter
        stop (list): List of stop sequences
        cancel_event (threading.Event): Event to signal cancellation
        progress_callback (callable): Function to report progress
        model_path (str): Path to GGUF model file (optional)
        
    Returns:
        str: Generated response
    """
    if progress_callback:
        progress_callback(0.1, "Loading llama.cpp model...")
    
    # Load model
    try:
        model = load_llamacpp_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load llama.cpp model: {e}")
    
    if progress_callback:
        progress_callback(0.3, "Starting generation...")
    
    # Format prompt
    prompt = f"You are a helpful pharmaceutical assistant. Please answer this question about medications or medical topics.\n\nQuestion: {query}\n\nAnswer:"
    
    # Define stop sequences if not provided
    if stop is None:
        stop = ["Question:", "\n\n"]
    
    try:
        # Check if create_completion method exists (newer versions)
        if hasattr(model, "create_completion"):
            # Stream response
            response_text = ""
            
            # Generate completion with streaming
            stream = model.create_completion(
                prompt,
                max_tokens=1024,
                temperature=temperature,
                top_p=top_p,
                top_k=40,
                stop=None,
                stream=True
            )
            
            # Process stream
            for i, chunk in enumerate(stream):
                if cancel_event and cancel_event.is_set():
                    break
                
                text_chunk = chunk["choices"][0]["text"]
                response_text += text_chunk
                
                # Update progress periodically
                if progress_callback and i % 5 == 0:
                    progress_callback(0.4 + min(0.5, len(response_text) / 500), "Generating response...")
            
            return response_text.strip()
        else:
            # Fallback to older call method
            result = model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=40,
                stop=stop,
                echo=False
            )
            
            if progress_callback:
                progress_callback(0.9, "Finalizing...")
                
            return result["choices"][0]["text"].strip()
            
    except Exception as e:
        raise RuntimeError(f"Error in llama.cpp generation: {e}")