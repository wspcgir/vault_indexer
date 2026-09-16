import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lancedb
import yaml
from rank_bm25 import BM25Okapi


# --- Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "mxbai-embed-large"  # or nomic-embed-text
DB_DIR = "./.vault_index"
CHUNK_SIZE = 800  # character limit per chunk
CHUNK_OVERLAP = 150


# --- Frontmatter & Document Processing ---
def parse_obsidian_note(file_path: Path) -> Tuple[Dict[str, Any], str]:
    """Extracts YAML frontmatter metadata and remaining Markdown body."""
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    
    # Check for YAML frontmatter block starting/ending with ---
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if frontmatter_match:
        yaml_text, body = frontmatter_match.groups()
        try:
            metadata = yaml.safe_load(yaml_text) or {}
            if not isinstance(metadata, dict):
                metadata = {}
        except yaml.YAMLError:
            metadata = {}
    else:
        metadata = {}
        body = content

    # Add default file properties
    metadata["title"] = metadata.get("title", file_path.stem)
    metadata["file_path"] = str(file_path.resolve())
    metadata["filename"] = file_path.name
    
    # Normalize list/tags/kind values to lower-case strings for easy filtering
    if "tags" in metadata:
        if isinstance(metadata["tags"], str):
            metadata["tags"] = [t.strip().lower() for t in metadata["tags"].split(",")]
        elif isinstance(metadata["tags"], list):
            metadata["tags"] = [str(t).lower() for t in metadata["tags"]]
            
    if "kind" in metadata:
        metadata["kind"] = str(metadata["kind"]).lower()
    elif "type" in metadata:
        metadata["kind"] = str(metadata["type"]).lower()
    else:
        metadata["kind"] = "note"

    return metadata, body


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Splits body text into overlapping text chunks."""
    text = text.strip()
    if not text:
        return []
        
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks


# --- Ollama API Embeddings ---
import requests

def get_ollama_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """Generates embeddings using Ollama's local embeddings endpoint."""
    res = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=30,
    )
    res.raise_for_status()
    return res.json()["embedding"]


# --- Index Management ---
def build_or_update_index(vault_path: Path, db_path: Path = Path(DB_DIR)):
    """Recursively scans vault, generates embeddings, and updates LanceDB table."""
    print(f"Scanning vault at: {vault_path.resolve()}...")
    
    db = lancedb.connect(db_path)
    
    records = []
    chunk_counter = 0

    for root, _, files in os.walk(vault_path):
        for file in files:
            if file.endswith(".md"):
                file_path = Path(root) / file
                metadata, body = parse_obsidian_note(file_path)
                chunks = chunk_text(body)
                
                for idx, chunk_str in enumerate(chunks):
                    # Fetch embedding vector from local Ollama
                    vector = get_ollama_embedding(chunk_str)
                    
                    records.append({
                        "id": f"{file_path.stem}_{idx}",
                        "vector": vector,
                        "text": chunk_str,
                        "title": str(metadata.get("title", "")),
                        "filename": metadata.get("filename", ""),
                        "file_path": metadata.get("file_path", ""),
                        "kind": str(metadata.get("kind", "note")),
                        "tags": json.dumps(metadata.get("tags", [])),
                        "chunk_index": idx,
                    })
                    chunk_counter += 1
                    if chunk_counter % 20 == 0:
                        print(f"Indexed {chunk_counter} chunks...")

    if not records:
        print("No markdown files found to index.")
        return

    # Create/Overwrite Table
    table = db.create_table("obsidian_notes", data=records, mode="overwrite")
    print(f"\u2705 Successfully indexed {len(records)} text chunks into LanceDB index.")


# --- Hybrid Search Execution ---
def search_index(
    query: str,
    kind_filter: Optional[str] = None,
    tag_filter: Optional[str] = None,
    limit: int = 5,
    db_path: Path = Path(DB_DIR),
):
    """Executes a hybrid search: Dense vector similarity + BM25 keyword matching + frontmatter filtering."""
    db = lancedb.connect(db_path)
    if "obsidian_notes" not in db.table_names():
        print("Index table not found. Please run '--update' first.")
        return

    table = db.open_table("obsidian_notes")
    df = table.to_pandas()

    # Apply Metadata Filters (Frontmatter)
    if kind_filter:
        df = df[df["kind"] == kind_filter.lower()]
    if tag_filter:
        df = df[df["tags"].str.contains(tag_filter.lower(), na=False)]

    if df.empty:
        print("No documents match the specified frontmatter metadata filter criteria.")
        return

    # 1. Vector Search Score (Cosine Similarity via Ollama Query Vector)
    query_vector = get_ollama_embedding(query)
    
    # Calculate vector distance manually over filtered frame
    import numpy as np
    
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    df["vector_score"] = df["vector"].apply(lambda vec: cosine_similarity(query_vector, vec))

    # 2. BM25 Keyword Search Score
    tokenized_corpus = [doc.split(" ") for doc in df["text"].tolist()]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    raw_bm25_scores = bm25.get_scores(tokenized_query)
    
    # Normalize BM25 Scores to 0-1 range
    max_bm25 = max(raw_bm25_scores) if max(raw_bm25_scores) > 0 else 1.0
    df["bm25_score"] = [s / max_bm25 for s in raw_bm25_scores]

    # 3. Reciprocal/Weighted Hybrid Fusion Score (50% Semantic + 50% Exact Keyword)
    df["hybrid_score"] = (0.5 * df["vector_score"]) + (0.5 * df["bm25_score"])
    
    # Sort and Display Results
    results = df.sort_values(by="hybrid_score", ascending=False).head(limit)

    print(f"\n--- Search Results for: '{query}' ---")
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        tags = json.loads(row['tags'])
        print(f"\n[{rank}] {row['title']} (Kind: {row['kind']}) | Score: {row['hybrid_score']:.4f}")
        print(f"    Path: {row['file_path']}")
        print(f"    Tags: {', '.join(tags)}")
        print(f"    Snippet: {row['text'][:200]}...")


# --- CLI Interface ---
def main():
    parser = argparse.ArgumentParser(description="Obsidian Hybrid Semantic & Frontmatter Search Engine")
    parser.add_argument("--vault", type=str, help="Path to your Obsidian Vault directory")
    parser.add_argument("--update", action="store_true", help="Re-scan vault and re-build the embedding index")
    parser.add_argument("--query", type=str, help="Search query string")
    parser.add_argument("--kind", type=str, help="Filter by frontmatter 'kind' or 'type' field (e.g. procedure, concept)")
    parser.add_argument("--tag", type=str, help="Filter by specific tag")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    if args.update:
        if not args.vault:
            print("Error: --vault path is required when updating index.")
            return
        build_or_update_index(Path(args.vault))
    elif args.query:
        search_index(
            query=args.query,
            kind_filter=args.kind,
            tag_filter=args.tag,
            limit=args.limit
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()