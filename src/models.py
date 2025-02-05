import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import move_cache

# Global caches
loaded_models = {}
loaded_tokenizers = {}

# Quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

def get_gpu_memory():
    """Return current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0

def unload_model(model_choice: str):
    """Unload a model from GPU and clear from cache."""
    from .models import loaded_models, loaded_tokenizers
    if model_choice in loaded_models:
        print(f"Unloading {model_choice} model from GPU...", flush=True)
        if hasattr(loaded_models[model_choice], 'cpu'):
            loaded_models[model_choice].cpu()
        del loaded_models[model_choice]
        if model_choice in loaded_tokenizers:
            del loaded_tokenizers[model_choice]
        torch.cuda.empty_cache()
        print(f"{model_choice} model unloaded successfully!", flush=True)


def get_llasa_model(model_choice: str, hf_api_key: str = None):
    """
    Load and cache the specified model (1B, 3B, or 8B).
    If an API key is provided, it is used to authenticate with Hugging Face.
    """
    from .models import loaded_models, loaded_tokenizers, quantization_config

    # Determine repo name
    if model_choice == "1B":
        repo = "HKUSTAudio/Llasa-1B"
    elif model_choice == "3B":
        repo = "srinivasbilla/llasa-3b"
    else:
        repo = "HKUSTAudio/Llasa-8B"

    # Unload any other loaded model
    for existing_model in list(loaded_models.keys()):
        if existing_model != model_choice:
            unload_model(existing_model)

    if model_choice not in loaded_models:
        print(f"Preparing to load {repo}...", flush=True)
        print(f"Current GPU memory usage: {get_gpu_memory():.2f}GB", flush=True)
        
        hub_path = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "huggingface",
            "hub",
            "models--" + repo.replace("/", "--")
        )
        
        if os.path.exists(hub_path):
            print(f"Loading {repo} from local cache...", flush=True)
        else:
            print(f"Model {repo} not found in cache. Starting download...", flush=True)
        
        print("Loading tokenizer...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(repo, use_auth_token=hf_api_key)
        print("Tokenizer loaded successfully!", flush=True)
        
        print(f"Loading {model_choice} model into memory...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            trust_remote_code=True,
            device_map='cuda',
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            use_auth_token=hf_api_key,
            torch_dtype=torch.float16
        )
        torch.cuda.empty_cache()
        print(f"{model_choice} model loaded successfully! (GPU memory: {get_gpu_memory():.2f}GB)", flush=True)
        loaded_tokenizers[model_choice] = tokenizer
        loaded_models[model_choice] = model

    return loaded_tokenizers[model_choice], loaded_models[model_choice]


def check_model_in_cache(model_choice: str) -> bool:
    """
    Check if the given model repo is already present in the local Hugging Face cache.
    """
    if model_choice == "1B":
        repo = "HKUSTAudio/Llasa-1B"
    elif model_choice == "3B":
        repo = "srinivasbilla/llasa-3b"
    else:
        repo = "HKUSTAudio/Llasa-8B"

    hub_path = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "huggingface",
        "hub",
        "models--" + repo.replace("/", "--")
    )
    return os.path.exists(hub_path)
