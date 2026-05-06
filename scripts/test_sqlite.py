import sqlite3
import json
from pathlib import Path

def test_sqlite():
    db_path = Path("test_sqlite.sqlite")
    if db_path.exists(): db_path.unlink()
    
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, text TEXT);")
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks');")
    
    conn.execute("INSERT INTO chunks (id, text) VALUES ('1', 'The quick brown fox');")
    conn.execute("INSERT INTO chunks_fts (rowid, text) VALUES (1, 'The quick brown fox');")
    
    # Test query
    try:
        cur = conn.execute(
            """
            SELECT c.id FROM chunks c
            JOIN chunks_fts f ON c.rowid = f.rowid
            WHERE chunks_fts MATCH ?
            """,
            ('fox',)
        )
        print(f"Result: {cur.fetchall()}")
    except Exception as e:
        print(f"Query failed: {e}")
        
    conn.close()
    if db_path.exists(): db_path.unlink()

if __name__ == "__main__":
    test_sqlite()
