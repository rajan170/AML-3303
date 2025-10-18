# RAG Application

A modern Retrieval-Augmented Generation (RAG) application with a beautiful web UI for uploading and querying PDF and DOCX documents.

## Features

- 📤 **Document Upload**: Support for PDF and DOCX files
- 🤖 **AI-Powered Q&A**: Ask questions about your documents
- 🎨 **Modern UI**: Beautiful, responsive interface with drag-and-drop support
- 🔍 **Semantic Search**: Uses vector embeddings for accurate document retrieval
- 💾 **Persistent Storage**: Documents are stored in ChromaDB
- 🚀 **Fast & Efficient**: Built with FastAPI and modern Python tools

## Tech Stack

- **Backend**: FastAPI, LangChain, ChromaDB
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM**: OpenAI GPT-3.5-turbo (optional)
- **Document Processing**: pypdf, python-docx
- **Package Manager**: uv

## Prerequisites

- Python 3.10 or higher
- OpenAI API key (optional, for enhanced answers)
- uv package manager

## Installation

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone or navigate to the project directory**:
```bash
cd /Users/rajan/Projects/rag
```

3. **Create a virtual environment and install dependencies using uv**:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

4. **Set up environment variables** (optional for better quality):

The app **will generate answers even without any API keys** using a local model! However, you can get better quality answers by using:

**Option A - OpenAI (Best Quality):**
```bash
echo "OPENAI_API_KEY=your_openai_key_here" > .env
```

**Option B - HuggingFace (Free Tier):**
```bash
echo "HUGGINGFACE_API_KEY=your_hf_key_here" > .env
```
Get a free key at: https://huggingface.co/settings/tokens

**Option C - Local Model (No API Key Needed):**
Just run without any `.env` file - the app will automatically download and use a local FLAN-T5 model (takes a moment on first run).

## Usage

1. **Start the application**:
```bash
python app.py
```

Or with uvicorn:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

2. **Open your browser** and navigate to:
```
http://localhost:8000
```

3. **Upload documents**:
   - Click "Choose Files" or drag and drop PDF/DOCX files
   - Click "Upload Documents" to process them

4. **Ask questions**:
   - Type your question in the text area
   - Press Enter or click "Search & Answer"
   - View the AI-generated answer with source citations

## Project Structure

```
rag/
├── app.py              # FastAPI application with embedded UI
├── rag_engine.py       # Core RAG functionality
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Project configuration
├── .gitignore          # Git ignore rules
├── README.md           # This file
├── uploads/            # Uploaded documents (created automatically)
└── chroma_db/          # Vector database (created automatically)
```

## API Endpoints

- `GET /` - Main UI
- `POST /upload` - Upload documents (multipart/form-data)
- `POST /query` - Query documents (JSON body: `{"query": "your question"}`)
- `GET /health` - Health check

## How It Works

1. **Document Upload**: Documents are uploaded and text is extracted
2. **Text Chunking**: Text is split into manageable chunks with overlap
3. **Embedding**: Each chunk is converted to a vector embedding
4. **Storage**: Embeddings are stored in ChromaDB for fast retrieval
5. **Query**: User questions are embedded and matched against stored chunks
6. **Answer Generation**: Retrieved chunks are sent to an LLM to generate a comprehensive answer

## Answer Generation Options

The app supports **three modes** for generating answers:

1. **OpenAI GPT-3.5** (Best quality, requires paid API key)
   - Most accurate and natural answers
   - Costs ~$0.002 per 1000 tokens

2. **HuggingFace Mistral-7B** (Good quality, free tier available)
   - High quality open-source model
   - Free tier: 30 requests/hour

3. **Local FLAN-T5** (No API key needed, completely free)
   - Runs on your machine (CPU)
   - No API costs or rate limits
   - Downloads automatically on first use (~900MB)

The app automatically selects the best available option based on your environment variables.

## Configuration

You can customize the RAG engine by modifying parameters in `rag_engine.py`:

- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `top_k`: Number of relevant chunks to retrieve (default: 3)
- `model_name`: Embedding model (default: all-MiniLM-L6-v2)

## Troubleshooting

### No OpenAI API Key
If you see "LLM not available" in the answers, add your OpenAI API key to the `.env` file.

### Document Processing Errors
- Ensure your PDF files are text-based (not scanned images)
- Check that DOCX files are not corrupted
- Large files may take longer to process

### Port Already in Use
If port 8000 is busy, change it in `app.py` or when running uvicorn:
```bash
uvicorn app:app --port 8001
```

## Development

To add more file types:
1. Add extraction method in `rag_engine.py`
2. Update file type validation in `app.py`
3. Update UI file accept attribute

## License

MIT License - feel free to use this project for your own purposes!

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

