#!/usr/bin/env python3
"""
Quick test script to verify RAG engine functionality
"""
import sys
from rag_engine import RAGEngine

def test_rag():
    print("Testing RAG Engine...")
    print("=" * 60)
    
    try:
        # Initialize engine
        print("\n1. Initializing RAG Engine...")
        rag = RAGEngine()
        print(f"   ✓ RAG Engine initialized")
        print(f"   LLM Type: {getattr(rag, 'llm_type', 'none')}")
        
        # Check document count
        print("\n2. Checking document count...")
        count = rag.collection.count()
        print(f"   Documents in database: {count}")
        
        if count == 0:
            print("\nNo documents in database!")
            print("   Please upload documents through the web UI first.")
            return
        
        # Test query
        print("\n3. Testing query...")
        test_query = "What is this document about?"
        result = rag.query(test_query)
        
        print(f"\n   Query: {test_query}")
        print(f"   Answer: {result['answer'][:200]}...")
        print(f"   Sources: {result['sources']}")
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_rag()

