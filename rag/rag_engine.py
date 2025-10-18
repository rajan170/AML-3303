import os
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from pypdf import PdfReader
from docx import Document


class RAGEngine:
    """RAG Engine for document processing and querying"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize the RAG engine"""
        self.persist_directory = persist_directory
        
        # Initialize embeddings model (using local model, no API key needed)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize LLM with multiple fallback options
        openai_api_key = os.getenv("OPENAI_API_KEY")
        huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        
        if openai_api_key:
            # Option 1: Use OpenAI (best quality)
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                api_key=openai_api_key
            )
            self.llm_type = "openai"
            print("Using OpenAI GPT-3.5-turbo for answer generation")
        elif huggingface_api_key:
            # Option 2: Use HuggingFace API 
            self.llm = HuggingFaceHub(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                huggingfacehub_api_token=huggingface_api_key,
                model_kwargs={"temperature": 0.7, "max_length": 512}
            )
            self.llm_type = "huggingface"
            print("Using HuggingFace Mistral-7B for answer generation")
        else:
            # Option 3: Use local model (no API key needed)
            try:
                from transformers import pipeline
                print("Loading local model for answer generation (this may take a moment)...")
                self.llm = pipeline(
                    "text-generation",
                    model="google/flan-t5-base",
                    max_length=512,
                    device=-1  # CPU
                )
                self.llm_type = "local"
                print("Using local FLAN-T5 model for answer generation")
            except Exception as e:
                print(f"Warning: Could not load local model: {e}")
                print("Falling back to retrieval-only mode")
                self.llm = None
                self.llm_type = None
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from supported file types"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            return self.extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    def add_document(self, file_path: str) -> None:
        """Process and add a document to the vector store"""
        print(f"Processing document: {file_path}")
        
        # Extract text
        text = self.extract_text(file_path)
        
        if not text.strip():
            raise ValueError(f"No text could be extracted from {file_path}")
        
        print(f"Extracted {len(text)} characters from {file_path}")
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        print(f"Split into {len(chunks)} chunks")
        
        # Get filename for metadata
        filename = Path(file_path).name
        
        # Generate embeddings and add to collection
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = self.embeddings.embed_query(chunk)
            
            # Create unique ID
            doc_id = f"{filename}_{i}"
            
            # Add to collection
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "source": filename,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }]
            )
        
        print(f"Successfully added {len(chunks)} chunks to vector store")
        print(f"Total documents in collection: {self.collection.count()}")
    
    def query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Query the vector store and generate an answer"""
        # Check if collection has documents
        doc_count = self.collection.count()
        print(f"Query: '{query}'")
        print(f"Collection has {doc_count} documents")
        
        if doc_count == 0:
            return {
                "answer": "No documents have been uploaded yet. Please upload some documents first.",
                "sources": []
            }
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search in vector store
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, doc_count)
            )
            print(f"Search returned {len(results['documents'][0]) if results['documents'] else 0} results")
        except Exception as e:
            print(f"Error during search: {e}")
            return {
                "answer": f"Error searching documents: {str(e)}",
                "sources": []
            }
        
        if not results['documents'] or not results['documents'][0]:
            return {
                "answer": "I couldn't find any relevant information in the uploaded documents. Try rephrasing your question or upload more relevant documents.",
                "sources": []
            }
        
        # Get retrieved documents
        retrieved_docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        # Extract unique sources
        sources = list(set([meta['source'] for meta in metadatas]))
        
        # Generate answer using LLM
        if self.llm:
            # Create context from retrieved documents
            context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs)])
            
            # Create prompt
            prompt = f"""Based on the following context from uploaded documents, answer the question.
If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {query}

Answer:"""
            
            # Generate response based on LLM type
            try:
                if self.llm_type == "openai":
                    response = self.llm.invoke(prompt)
                    answer = response.content
                elif self.llm_type == "huggingface":
                    answer = self.llm(prompt)
                elif self.llm_type == "local":
                    # For local transformer model
                    response = self.llm(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
                    answer = response[0]['generated_text'].replace(prompt, '').strip()
                    # If the model just repeats or gives poor output, clean it up
                    if not answer or len(answer) < 20:
                        answer = self._create_extractive_answer(retrieved_docs, query)
                else:
                    answer = self._create_extractive_answer(retrieved_docs, query)
            except Exception as e:
                print(f"Error generating answer with LLM: {e}")
                answer = self._create_extractive_answer(retrieved_docs, query)
        else:
            # Create an extractive answer from the chunks
            answer = self._create_extractive_answer(retrieved_docs, query)
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def _create_extractive_answer(self, docs: List[str], query: str) -> str:
        """Create an extractive answer by intelligently combining relevant chunks"""
        # Combine the most relevant passages
        answer = "Based on the uploaded documents:\n\n"
        
        # Add the most relevant chunk(s)
        if len(docs) > 0:
            answer += docs[0]
        
        # If we have multiple relevant chunks, add them
        if len(docs) > 1:
            answer += f"\n\nAdditionally:\n{docs[1]}"
        
        return answer
    
    def reset(self) -> None:
        """Clear all documents from the vector store"""
        # Delete and recreate collection
        self.client.delete_collection(name="documents")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

