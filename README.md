# PDF Data Extraction Tool (Qwen-3 Powered)

A Streamlit-based application that extracts and analyzes content from PDF documents using the Qwen-3 language model.

## Features

- 📤 PDF Upload: Support for uploading PDF documents
- 📄 Text Extraction: Extracts raw text from PDF files
- 🧠 Smart Analysis: Uses Qwen-3 model to structure and analyze PDF content
- 📱 Responsive UI: Clean, wide-layout interface for better visualization
- 💾 Session Management: Maintains state across interactions

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
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

4. Run the application:
```bash
streamlit run main.py
```

## Usage

1. Launch the application using the command above
2. Use the file uploader to select a PDF file
3. The left panel will show the raw extracted text from each page
4. The right panel will display the structured content extracted by Qwen-3
5. Use the "Clear" button to reset the application state

## Requirements

- Python 3.8+
- See requirements.txt for complete list of dependencies

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 