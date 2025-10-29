# 📘 Document Chatbot with Cerebras AI

A chatbot that allows you to upload documents (PDF, Word, Excel) and ask questions about their content using Cerebras AI's powerful language models.

## Features

- 📄 **Multi-format Support**: Upload PDF, DOCX, or XLSX files
- 🤖 **Powered by Cerebras AI**: Uses the Qwen-3-235B model for intelligent responses
- 🔍 **Semantic Search**: Uses FAISS vector database for efficient document retrieval
- 💬 **Interactive Chat**: Ask questions and get answers based on your document content
- 🎯 **Source Citations**: See which pages the answers come from

## Prerequisites

- Python 3.8 or higher
- Cerebras API key ([Get one here](https://cloud.cerebras.ai/))

## Installation

1. **Clone or download this repository**

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows (PowerShell):
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - Windows (Command Prompt):
     ```cmd
     .\venv\Scripts\activate.bat
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up your API key**
   
   Create a `.env` file in the project root and add your Cerebras API key:
   ```
   CEREBRAS_API_KEY=your_api_key_here
   ```

## Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```
   Or using the venv Python directly:
   ```bash
   .\venv\Scripts\python.exe -m streamlit run app.py
   ```

2. **Open your browser**
   
   The app will automatically open at `http://localhost:8501`

3. **Upload a document**
   
   Click "Browse files" and select a PDF, DOCX, or XLSX file

4. **Ask questions**
   
   Type your question in the chat input and get AI-powered answers based on your document

## Project Structure

```
chatbot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API key)
├── README.md             # This file
└── venv/                 # Virtual environment (created after setup)
```

## Technologies Used

- **Streamlit**: Web interface
- **LangChain**: LLM orchestration framework
- **Cerebras AI**: Language model API
- **FAISS**: Vector database for semantic search
- **Sentence Transformers**: Text embeddings
- **PDFPlumber**: PDF text extraction
- **python-docx**: Word document processing
- **Pandas**: Excel file handling

## Configuration

### Supported Models

The app currently uses `qwen-3-235b-a22b-instruct-2507`. You can change this in `app.py`:

```python
llm = ChatCerebras(
    model="qwen-3-235b-a22b-instruct-2507",  # Change model here
    temperature=0,
    max_tokens=600,
)
```

### Embedding Model

Default: `sentence-transformers/all-MiniLM-L6-v2`

You can change this in the `create_vectorstore()` function.

## Troubleshooting

### ModuleNotFoundError

Make sure you're using the virtual environment:
```bash
.\venv\Scripts\python.exe -m streamlit run app.py
```

### API Key Error

- Verify your API key is correct in the `.env` file
- Ensure the `.env` file is in the same directory as `app.py`
- Restart the Streamlit app after changing the `.env` file

### Import Errors

Reinstall dependencies in the virtual environment:
```bash
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Security Notes

- **Never commit your `.env` file** to version control
- Keep your API key confidential
- Add `.env` to your `.gitignore` file

## License

This project is open source and available for personal and educational use.

## Contributing

Feel free to fork this project and submit pull requests for improvements!

## Support

For issues with:
- **Cerebras API**: Visit [Cerebras Documentation](https://inference-docs.cerebras.ai/)
- **Streamlit**: Check [Streamlit Documentation](https://docs.streamlit.io/)
- **LangChain**: See [LangChain Documentation](https://python.langchain.com/)
