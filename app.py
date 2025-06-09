import streamlit as st
from transformers import pipeline
from torchvision import transforms
# from transformers import BitsAndBytesConfig
from PIL import Image
import torch
import os
import io

# Base models directory
MODELS_PATH = os.path.join(os.path.expanduser("~"), "LegendsLair", "models")

# Streamlit app configuration
st.set_page_config(
    page_title="Gemma OCR Assistant",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and processor from a specified subdirectory inside MODELS_PATH
# @st.cache_resource
def load_model(model_name: str):
    model_path = os.path.join(MODELS_PATH, model_name)
    if not os.path.exists(model_path):
        raise ValueError(f"Model not found at {model_path}. Please ensure the model is downloaded.")
    
    from transformers import GemmaTokenizer
    tokenizer = GemmaTokenizer.from_pretrained(model_path)
    
    pipe = pipeline(
        "image-text-to-text",
        model=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
        tokenizer=tokenizer
    )

    return pipe

# Generate text from image using model
def generate_response(image_bytes, prompt, model_name="google/gemma-3-4b-it"):
    pipe = load_model(model_name)

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        raise ValueError("Uploaded file is not a valid image.") from e

    # # Convert image to bytes for model
    # image_bytes = io.BytesIO()
    # image.save(image_bytes, format="JPEG")
    # image_bytes = image_bytes.getvalue()
    # image_bytes = torch.tensor(image_bytes).unsqueeze(0)

    # Assuming `image` is a PIL.Image
    transform = transforms.ToTensor()  # Converts to float tensor [0,1]
    # image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

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

    return output[0]["generated_text"][-1]["content"]

    # inputs = processor.apply_chat_template(
    #     messages,
    #     add_generation_prompt=True,
    #     tokenize=True,
    #     return_tensors="pt"
    # ).to(model.device, dtype=torch.bfloat16)

    # input_len = inputs["input_ids"].shape[-1]

    # with torch.inference_mode():
    #     output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    #     output = output[0][input_len:]

    # return processor.decode(output, skip_special_tokens=True)

# # App title with logo
# with open("assets/gemma3.png", "rb") as f:
#     encoded_logo = base64.b64encode(f.read()).decode()

# st.markdown(f"""
#     <h1 style="display: flex; align-items: center;">
#         <img src="data:image/png;base64,{encoded_logo}" width="50" style="margin-right: 10px;">
#         Gemma OCR Assistant
#     </h1>
# """, unsafe_allow_html=True)

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

# Display image and process OCR
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if extract_btn:
        with st.spinner("Extracting text from image..."):
            try:
                image_bytes = uploaded_file.getvalue()
                result = generate_response(image_bytes, user_prompt)
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

# # Footer
# st.markdown("---")
# st.markdown("Made with ❤️ using [Gemma-3 Vision](https://huggingface.co/google/gemma-3-4b-it) | [GitHub](https://github.com/patchy631/ai-engineering-hub)")