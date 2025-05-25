import streamlit as st
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from io import BytesIO
import os
import time

# Enable automatic reloading
st.set_page_config(layout="wide")

st.title("📄 PDF Data Extraction Tool (Qwen-3 Powered)")

@st.cache_resource
def load_model():
    try:
        # Get the parent directory of pdf_extractor_app
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        model_path = os.path.join(parent_dir, "models", "Qwen", "Qwen3-1.7B")
        
        if not os.path.exists(model_path):
            st.error(f"Model directory not found at: {model_path}")
            return None
        
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto"
        )
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

generator = load_model()

def extract_text_from_pdf(file: BytesIO):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text_per_page = [page.get_text("text") for page in doc]
    return text_per_page

if "pdf_file" not in st.session_state:
    st.session_state.pdf_file = None
if "extracted" not in st.session_state:
    st.session_state.extracted = []

if generator is None:
    st.error("⚠️ Model failed to load. Please check the model directory and try again.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", key="pdf_uploader")
    if uploaded_file:
        st.session_state.pdf_file = uploaded_file
        pages = extract_text_from_pdf(uploaded_file)
        for i, text in enumerate(pages):
            st.markdown(f"### Page {i + 1}")
            st.text_area(label="", value=text, height=300, key=f"page_{i}_text")

with col2:
    if st.session_state.pdf_file:
        pages = extract_text_from_pdf(st.session_state.pdf_file)
        st.subheader("🧠 Extracted Content (via Qwen3)")
        st.session_state.extracted = []
        for i, text in enumerate(pages):
            prompt = f"Extract structured elements from this PDF page:\n{text}"
            output = generator(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]
            formatted = output.replace(prompt, "")
            st.markdown(f"### Page {i + 1}")
            st.text_area(label="", value=formatted.strip(), height=300, key=f"page_{i}_extracted")

st.divider()
if st.button("🧹 Clear"):
    st.session_state.pdf_file = None
    st.session_state.extracted = []
    st.rerun()