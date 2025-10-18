#!/usr/bin/env python3
"""
Add test content directly to the RAG database for testing
"""
from rag_engine import RAGEngine

def add_test_content():
    print("Adding test content to RAG database...")
    print("=" * 60)
    
    # Initialize RAG engine
    rag = RAGEngine()
    
    # Create a temporary test file
    test_content = """Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on developing algorithms and statistical models that enable computers to improve their performance on a specific task through experience.

Types of Machine Learning:

1. Supervised Learning
Supervised learning involves training a model on labeled data. The algorithm learns to map inputs to outputs based on example input-output pairs. Common applications include image classification, spam detection, and price prediction.

2. Unsupervised Learning
Unsupervised learning works with unlabeled data. The algorithm tries to find patterns and structures in the data without explicit guidance. Examples include customer segmentation, anomaly detection, and data compression.

3. Reinforcement Learning
Reinforcement learning is about training agents to make decisions by rewarding desired behaviors and punishing undesired ones. Applications include game playing (like AlphaGo), robotics, and autonomous vehicles.

Key Concepts:

Training Data: The dataset used to train the machine learning model.
Features: The input variables used to make predictions.
Labels: The output or target variable in supervised learning.
Model: The mathematical representation learned from the data.
Overfitting: When a model performs well on training data but poorly on new data.

Applications:
Machine learning has revolutionized many industries including healthcare (disease diagnosis, drug discovery), finance (fraud detection, algorithmic trading), transportation (autonomous vehicles, traffic prediction), and entertainment (recommendation systems, content generation).

Deep Learning:
Deep learning is a subset of machine learning that uses neural networks with multiple layers. It has achieved remarkable success in areas like computer vision, natural language processing, and speech recognition. Popular frameworks include TensorFlow, PyTorch, and Keras.

Conclusion:
As data continues to grow exponentially, machine learning will play an increasingly important role in extracting insights and making intelligent decisions across all sectors of society.
"""
    
    # Save to temporary file
    import tempfile
    import os
    
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    
    test_file = "uploads/test_ml_document.txt"
    with open(test_file, "w") as f:
        f.write(test_content)
    
    print(f"Created temporary file: {test_file}")
    
    # Manually add chunks
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_text(test_content)
    print(f"Split content into {len(chunks)} chunks")
    
    # Add to vector store
    for i, chunk in enumerate(chunks):
        embedding = rag.embeddings.embed_query(chunk)
        doc_id = f"test_ml_document_{i}"
        
        rag.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{
                "source": "test_ml_document.txt",
                "chunk_id": i,
                "total_chunks": len(chunks)
            }]
        )
    
    print(f"Successfully added {len(chunks)} chunks to database")
    print(f"Total documents in collection: {rag.collection.count()}")
    print("\n" + "=" * 60)
    print("Test content added successfully!")
    print("\nYou can now query the system. Try asking:")
    print("  - 'What is machine learning?'")
    print("  - 'What are the types of machine learning?'")
    print("  - 'What is deep learning?'")
    print("\nRefresh your browser and try a query!")

if __name__ == "__main__":
    try:
        add_test_content()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

