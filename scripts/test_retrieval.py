from pathlib import Path
import sys
import os

# Add scripts to path
sys.path.append(str(Path(os.getcwd()) / "scripts"))

try:
    from vector_store import SqliteRagIndex
    import numpy as np
    
    db_path = Path("test_index.sqlite")
    if db_path.exists(): db_path.unlink()
    
    # Use a small model for testing
    index = SqliteRagIndex(db_path, model_name="all-MiniLM-L6-v2", rerank_model_name=None)
    
    ids = ["1", "2"]
    texts = ["The quick brown fox jumps over the lazy dog.", "Artificial intelligence is transforming the world."]
    metadatas = [{"source": "test1"}, {"source": "test2"}]
    
    print("Upserting...")
    index.upsert_batch(ids, texts, metadatas)
    
    print("Querying (Vector)...")
    results = index.query("fox dog", k=2)
    for r in results:
        print(f"ID: {r['id']}, Score: {r['score']:.4f}, Text: {r['document']}")
        
    print("Querying (Hybrid)...")
    results = index.query("Artificial intelligence", k=2)
    for r in results:
        print(f"ID: {r['id']}, Score: {r['score']:.4f}, Text: {r['document']}")
        
    index.close()
    if db_path.exists(): db_path.unlink()
    print("Test passed!")
    
except Exception as e:
    print(f"Test failed: {e}")
    import traceback
    traceback.print_exc()
