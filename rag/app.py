import os
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from rag_engine import RAGEngine

# Load environment variables
load_dotenv()

app = FastAPI(title="RAG Application")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine
rag_engine = RAGEngine()

# Ensure upload directory exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Application</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            
            h1 {
                color: white;
                text-align: center;
                margin-bottom: 40px;
                font-size: 2.5rem;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            }
            
            .card {
                background: white;
                border-radius: 16px;
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }
            
            h2 {
                color: #333;
                margin-bottom: 20px;
                font-size: 1.5rem;
            }
            
            .upload-section {
                border: 3px dashed #667eea;
                border-radius: 12px;
                padding: 40px;
                text-align: center;
                background: #f8f9ff;
                transition: all 0.3s ease;
                cursor: pointer;
            }
            
            .upload-section:hover {
                border-color: #764ba2;
                background: #f0f2ff;
            }
            
            .upload-section.drag-over {
                border-color: #764ba2;
                background: #e8ebff;
                transform: scale(1.02);
            }
            
            input[type="file"] {
                display: none;
            }
            
            .file-input-label {
                display: inline-block;
                padding: 12px 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                transition: transform 0.2s ease;
            }
            
            .file-input-label:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            }
            
            .file-name {
                margin-top: 15px;
                color: #666;
                font-size: 0.9rem;
            }
            
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 1rem;
                font-weight: 600;
                transition: all 0.2s ease;
                margin-top: 15px;
            }
            
            button:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            }
            
            button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            textarea {
                width: 100%;
                padding: 15px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 1rem;
                font-family: inherit;
                resize: vertical;
                min-height: 100px;
                transition: border-color 0.3s ease;
            }
            
            textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            
            .answer-box {
                background: #f8f9ff;
                border-left: 4px solid #667eea;
                padding: 20px;
                border-radius: 8px;
                margin-top: 20px;
            }
            
            .answer-box h3 {
                color: #667eea;
                margin-bottom: 10px;
            }
            
            .answer-text {
                color: #333;
                line-height: 1.6;
                margin-bottom: 15px;
            }
            
            .sources {
                background: white;
                padding: 15px;
                border-radius: 6px;
                margin-top: 10px;
            }
            
            .sources h4 {
                color: #764ba2;
                font-size: 0.9rem;
                margin-bottom: 8px;
            }
            
            .source-item {
                color: #666;
                font-size: 0.85rem;
                padding: 8px;
                background: #f8f9ff;
                border-radius: 4px;
                margin-bottom: 6px;
            }
            
            .status {
                padding: 10px;
                border-radius: 6px;
                margin-top: 15px;
                font-size: 0.9rem;
            }
            
            .status.success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            
            .status.error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            
            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-left: 10px;
                vertical-align: middle;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .stats {
                display: flex;
                gap: 20px;
                margin-top: 20px;
            }
            
            .stat-card {
                flex: 1;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 12px;
                text-align: center;
            }
            
            .stat-number {
                font-size: 2rem;
                font-weight: bold;
                margin-bottom: 5px;
            }
            
            .stat-label {
                font-size: 0.9rem;
                opacity: 0.9;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📚 RAG Application</h1>
            
            <div class="card">
                <h2>📤 Upload Documents</h2>
                <div class="upload-section" id="uploadSection">
                    <p style="color: #667eea; font-size: 3rem; margin-bottom: 10px;">📄</p>
                    <label for="fileInput" class="file-input-label">Choose Files</label>
                    <input type="file" id="fileInput" accept=".pdf,.docx,.doc" multiple>
                    <p style="margin-top: 15px; color: #666;">or drag and drop files here</p>
                    <p style="margin-top: 10px; color: #999; font-size: 0.85rem;">Supported formats: PDF, DOCX, DOC</p>
                    <div class="file-name" id="fileName"></div>
                </div>
                <button id="uploadBtn" disabled>Upload Documents</button>
                <div id="uploadStatus"></div>
            </div>
            
            <div class="card">
                <h2>💬 Ask Questions</h2>
                <textarea id="queryInput" placeholder="Ask a question about your uploaded documents..."></textarea>
                <button id="queryBtn">Search & Answer</button>
                <div id="answerSection"></div>
            </div>
            
            <div class="card">
                <h2>📊 Statistics</h2>
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-number" id="docCount">0</div>
                        <div class="stat-label">Documents Uploaded</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="queryCount">0</div>
                        <div class="stat-label">Queries Made</div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let selectedFiles = [];
            let docCount = 0;
            let queryCount = 0;
            
            // Load initial status
            async function loadStatus() {
                try {
                    const response = await fetch('/status');
                    const status = await response.json();
                    if (status.documents_in_database !== undefined) {
                        docCount = status.documents_in_database;
                        document.getElementById('docCount').textContent = docCount;
                    }
                    console.log('System status:', status);
                } catch (error) {
                    console.error('Error loading status:', error);
                }
            }
            
            // Load status on page load
            loadStatus();
            
            const uploadSection = document.getElementById('uploadSection');
            const fileInput = document.getElementById('fileInput');
            const fileName = document.getElementById('fileName');
            const uploadBtn = document.getElementById('uploadBtn');
            const uploadStatus = document.getElementById('uploadStatus');
            const queryInput = document.getElementById('queryInput');
            const queryBtn = document.getElementById('queryBtn');
            const answerSection = document.getElementById('answerSection');
            
            // File input handling
            fileInput.addEventListener('change', (e) => {
                selectedFiles = Array.from(e.target.files);
                updateFileDisplay();
            });
            
            // Drag and drop
            uploadSection.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadSection.classList.add('drag-over');
            });
            
            uploadSection.addEventListener('dragleave', () => {
                uploadSection.classList.remove('drag-over');
            });
            
            uploadSection.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadSection.classList.remove('drag-over');
                selectedFiles = Array.from(e.dataTransfer.files).filter(file => 
                    file.name.endsWith('.pdf') || file.name.endsWith('.docx') || file.name.endsWith('.doc')
                );
                updateFileDisplay();
            });
            
            function updateFileDisplay() {
                if (selectedFiles.length > 0) {
                    fileName.textContent = `Selected: ${selectedFiles.map(f => f.name).join(', ')}`;
                    uploadBtn.disabled = false;
                } else {
                    fileName.textContent = '';
                    uploadBtn.disabled = true;
                }
            }
            
            // Upload documents
            uploadBtn.addEventListener('click', async () => {
                if (selectedFiles.length === 0) return;
                
                uploadBtn.disabled = true;
                uploadStatus.innerHTML = '<div class="status">Uploading and processing documents... <span class="loading"></span></div>';
                
                const formData = new FormData();
                selectedFiles.forEach(file => formData.append('files', file));
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        let message = result.message;
                        if (result.warnings && result.warnings.length > 0) {
                            message += '<br><small>Warnings: ' + result.warnings.join(', ') + '</small>';
                        }
                        uploadStatus.innerHTML = `<div class="status success">${message}</div>`;
                        
                        // Reload status to get accurate count
                        await loadStatus();
                        
                        selectedFiles = [];
                        fileInput.value = '';
                        fileName.textContent = '';
                    } else {
                        uploadStatus.innerHTML = `<div class="status error">${result.detail || 'Upload failed'}</div>`;
                        uploadBtn.disabled = false;
                    }
                } catch (error) {
                    uploadStatus.innerHTML = `<div class="status error">Error: ${error.message}</div>`;
                    uploadBtn.disabled = false;
                }
            });
            
            // Query documents
            queryBtn.addEventListener('click', async () => {
                const query = queryInput.value.trim();
                if (!query) return;
                
                queryBtn.disabled = true;
                answerSection.innerHTML = '<div class="status">Searching and generating answer... <span class="loading"></span></div>';
                
                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ query })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        queryCount++;
                        document.getElementById('queryCount').textContent = queryCount;
                        
                        let sourcesHtml = '';
                        if (result.sources && result.sources.length > 0) {
                            sourcesHtml = '<div class="sources"><h4>📑 Sources:</h4>';
                            result.sources.forEach((source, idx) => {
                                sourcesHtml += `<div class="source-item">${idx + 1}. ${source}</div>`;
                            });
                            sourcesHtml += '</div>';
                        }
                        
                        answerSection.innerHTML = `
                            <div class="answer-box">
                                <h3>Answer:</h3>
                                <div class="answer-text">${result.answer}</div>
                                ${sourcesHtml}
                            </div>
                        `;
                    } else {
                        answerSection.innerHTML = `<div class="status error">${result.detail || 'Query failed'}</div>`;
                    }
                } catch (error) {
                    answerSection.innerHTML = `<div class="status error">Error: ${error.message}</div>`;
                } finally {
                    queryBtn.disabled = false;
                }
            });
            
            // Enter key to submit query
            queryInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    queryBtn.click();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    processed_files = []
    errors = []
    
    for file in files:
        if not file.filename:
            continue
            
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.pdf', '.docx', '.doc']:
            errors.append(f"{file.filename}: Unsupported file type")
            continue
        
        # Save file
        file_path = UPLOAD_DIR / file.filename
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            print(f"\n{'='*60}")
            print(f"Processing: {file.filename}")
            print(f"{'='*60}")
            
            # Process document and add to vector store
            rag_engine.add_document(str(file_path))
            processed_files.append(file.filename)
            print(f"Successfully processed: {file.filename}\n")
            
        except Exception as e:
            error_msg = f"{file.filename}: {str(e)}"
            errors.append(error_msg)
            print(f"Error processing {file.filename}: {str(e)}\n")
    
    if not processed_files and errors:
        raise HTTPException(status_code=500, detail=f"Failed to process files: {'; '.join(errors)}")
    
    response = {
        "message": f"Successfully processed {len(processed_files)} document(s)",
        "files": processed_files
    }
    
    if errors:
        response["warnings"] = errors
    
    return JSONResponse(response)


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system"""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = rag_engine.query(request.query, top_k=request.top_k)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/status")
async def get_status():
    """Get RAG engine status"""
    try:
        doc_count = rag_engine.collection.count()
        return {
            "status": "ready",
            "documents_in_database": doc_count,
            "llm_type": getattr(rag_engine, 'llm_type', 'none')
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

