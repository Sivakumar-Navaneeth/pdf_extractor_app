import streamlit as st
from transformers import pipeline
from transformers import GemmaTokenizer
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

    tokenizer = GemmaTokenizer.from_pretrained(model_path)

    pipe = pipeline(
        "image-text-to-text",
        model=model_path,
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.bfloat16,
        tokenizer=tokenizer
    )

    return pipe

# Generate response from image and prompt
def generate_response(image: Image.Image, prompt: str, model_name="google/gemma-3-4b-it"):
    pipe = load_model(model_name)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    output = pipe(text=messages, max_new_tokens=512, do_sample=False)
    print(f"Generated output: {output}")
    return output[0]["generated_text"][-1]["content"]

# UI
st.markdown("Extract clean, structured Markdown from images using the locally downloaded Gemma-3 model.")
st.markdown("---")

# Sidebar input controls
with st.sidebar:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    default_prompt = (
        "Extract all readable text from the image and format it in Markdown.\n"
        "Use appropriate headings, bullet points, or code blocks if needed."
    )
    user_prompt = st.text_area("Custom Prompt", default_prompt, height=150)

    extract_btn = st.button("🔍 Extract Text")

# Clear output
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("🗑️ Clear"):
        st.session_state.pop("ocr_output", None)
        st.rerun()

# Process image and generate text
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if extract_btn:
        with st.spinner("Extracting text from image..."):
            try:
                result = generate_response(image, user_prompt)
                st.session_state["ocr_output"] = result
            except Exception as e:
                st.error(f"⚠️ Error: {e}")

# Show extracted text
st.markdown("---")
st.subheader("📄 Extracted Markdown Output")
if "ocr_output" in st.session_state:
    st.markdown(st.session_state["ocr_output"])
else:
    st.info("Upload an image and click 'Extract Text' to view results.")
