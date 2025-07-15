import streamlit as st
from transformers import pipeline
from transformers import GemmaTokenizer, BitsAndBytesConfig
from PIL import Image
import torch
import os

# Base models directory
MODELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

# Streamlit app configuration
st.set_page_config(
    page_title="Gemma OCR Assistant",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and tokenizer from a specified subdirectory
@st.cache_resource
def load_model(model_name: str):
    model_path = os.path.join(MODELS_PATH, model_name)
    if not os.path.exists(model_path):
        raise ValueError(f"Model not found at {model_path}. Please ensure the model is downloaded.")

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for processing.")

    # Load the image-text-to-text pipeline with 4-bit quantization
    pipe = pipeline(
        "image-text-to-text",
        model="unsloth/gemma-3-4b-it-bnb-4bit",
        device=device,
        quantization_config=bnb_config
    )

    return pipe

# Rest of the code remains the same...
