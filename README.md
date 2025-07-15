# PDF Data Extraction Tool (Gemma-3B Powered)

A Streamlit-based application that extracts and analyzes content from PDF documents using the Gemma-3B language model.

## Features

- 📤 PDF Upload: Support for uploading PDF documents
- 📄 Text Extraction: Extracts raw text from PDF files
- 🧠 Smart Analysis: Uses Gemma-3B model to structure and analyze PDF content
- 📱 Responsive UI: Clean, wide-layout interface for better visualization
- 💾 Session Management: Maintains state across interactions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sivakumar-Navaneeth/pdf_extractor_app.git
cd pdf_extractor_app
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the quantized Gemma model:
```bash
huggingface-cli download unsloth/gemma-3-4b-it-bnb-4bit --local-dir models/gemma
```
Note: Using the pre-quantized model from unsloth saves bandwidth and storage by directly downloading the 4-bit version instead of the full model.

5. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Launch the application using the command above
2. Use the file uploader to select a PDF file
3. The left panel will show the raw extracted text from each page
4. The right panel will display the structured content extracted by Gemma-3B
5. Use the "Clear" button to reset the application state

## Requirements

- Python 3.8+
- See requirements.txt for complete list of dependencies

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 