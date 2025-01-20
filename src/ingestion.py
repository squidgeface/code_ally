"""
ingestion.py

This module handles parsing a repository to extract
functions, classes, and their relevant metadata for embedding.
"""

import os
import ast
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.errors import UniqueConstraintError
from rapidfuzz import fuzz, process
from pygments.lexers import guess_lexer_for_filename
from pygments.util import ClassNotFound

def load_exclusions(base_path: str) -> Dict[str, set]:
    """
    Load exclusions from .gitignore or similar files.

    Args:
        base_path (str): Path to the root directory.

    Returns:
        Dict[str, set]: Directories and files to exclude.
    """
    excluded_dirs = set()
    excluded_files = set()
    gitignore_path = os.path.join(base_path, ".gitignore")

    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as gitignore:
            for line in gitignore:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.endswith("/"):
                    excluded_dirs.add(line.rstrip("/"))
                else:
                    excluded_files.add(line)

    return {"dirs": excluded_dirs, "files": excluded_files}

def detect_language(file_path: str) -> str:
    """
    Detects the programming language of a given file using Pygments.

    Args:
        file_path: Path to the file to be analyzed.

    Returns:
        Detected language name or 'Unknown'.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        lexer = guess_lexer_for_filename(file_path, content)
        return lexer.name
    except (ClassNotFound, FileNotFoundError, UnicodeDecodeError):
        return "Unknown"
    
def store_embeddings(entities: List[Dict[str, any]], db_path: str = "data/vector_db") -> None:
    """
    Store embeddings and their metadata in a Chroma vector database.

    Args:
        entities (List[Dict[str, any]]): List of entities with metadata and embeddings.
        db_path (str): Path to the Chroma database directory.
    """
    print(f"Initializing Chroma database at: {db_path}")
    client = PersistentClient(path=db_path)

    # Access or create the collection
    collection_name = "code_entities"
    try:
        collection = client.create_collection(collection_name)
        print(f"Created new collection: {collection_name}")
    except UniqueConstraintError:
        collection = client.get_collection(collection_name)
        print(f"Using existing collection: {collection_name}")

    print("Storing embeddings for updated or new files...")

    # Fetch all existing IDs upfront
    existing_ids = collection.get(ids=[f"{entity['name']}_{entity['file']}_{entity['line']}" for entity in entities])["ids"]
    print(f"Existing IDs in the collection: {existing_ids}")

    for entity in entities:
        entity_id = f"{entity['name']}_{entity['file']}_{entity['line']}"  # Unique ID based on the function or class name, file and line
        if entity_id in existing_ids:  # Update existing embedding
            print(f"Updating existing embedding for ID: {entity_id}")
            collection.update(
                documents=[f"{entity['type']}: {entity['name']}"],
                metadatas=[{
                    "file": entity["file"],
                    "line": entity["line"],
                    "docstring": entity["docstring"]
                }],
                ids=[entity_id]
            )
        else:  # Add new embedding
            print(f"Adding new embedding for ID: {entity_id}")
            collection.add(
                documents=[f"{entity['type']}: {entity['name']}"],
                metadatas=[{
                    "file": entity["file"],
                    "line": entity["line"],
                    "docstring": entity["docstring"]
                }],
                ids=[entity_id]
            )

    print("Embeddings updated successfully.")


def generate_embeddings(entities: List[Dict[str, str]], model_name: str = "all-MiniLM-L6-v2") -> List[Dict[str, any]]:
    """
    Generate embeddings for the parsed code entities.

    Args:
        entities (List[Dict[str, str]]): List of code entities with metadata.
        model_name (str): Name of the pre-trained Sentence Transformer model.

    Returns:
        List[Dict[str, any]]: List of entities with their embeddings added.
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    entity_texts = [
        f"{entity['type']}: {entity['name']}\nDocstring: {entity['docstring']}\nFile: {entity['file']}"
        for entity in entities
    ]

    print("Entities to embed:")
    for text in entity_texts:
        print(text)

    print("Generating embeddings...")
    embeddings = model.encode(entity_texts, show_progress_bar=True)

    for i, entity in enumerate(entities):
        entity["embedding"] = embeddings[i]

    print("Embeddings generated successfully.")
    return entities


def parse_file(file_path: str) -> List[Dict[str, str]]:
    """
    Parse a file to extract meaningful entities, depending on the language.

    Args:
        file_path (str): Path to the file.

    Returns:
        List[Dict[str, str]]: A list of parsed entities.
    """
    language = detect_language(file_path)

    if language.lower() == "python":
        try:
            normalized_path = os.path.normpath(file_path)
            with open(normalized_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=normalized_path)

            parsed_entities = []

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    parsed_entities.append({
                        "name": node.name,
                        "type": "function" if isinstance(node, ast.FunctionDef) else "class",
                        "file": normalized_path,
                        "line": node.lineno,
                        "docstring": ast.get_docstring(node) or "No docstring available"
                    })

            return parsed_entities

        except Exception as e:
            print(f"Error parsing Python file {file_path}: {e}")
            return []

    # Add fallback for other languages if needed
    print(f"Language {language} is not currently supported for detailed parsing.")
    return []


def ingest_code(base_path: str) -> List[Dict[str, str]]:
    """
    Ingest all files in a given directory, excluding directories and files based on rules.

    Args:
        base_path (str): The root directory containing files.

    Returns:
        List[Dict[str, str]]: A list of parsed entities from all supported files.
    """
    exclusions = load_exclusions(base_path)
    all_entities = []

    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if d not in exclusions["dirs"]]
        for file in files:
            file_path = os.path.join(root, file)
            if file in exclusions["files"]:
                print(f"Skipping excluded file: {file_path}")
                continue

            language = detect_language(file_path)

            if language == "Unknown":
                print(f"Skipping {file_path}: Unable to detect language.")
                continue

            print(f"Processing {file_path} (Language: {language})")
            all_entities.extend(parse_file(file_path))

    print(f"Total entities parsed: {len(all_entities)}")
    return all_entities


def retrieve_entities(query: str, db_path: str = "data/vector_db", top_k: int = 5, distance_threshold: float = 2.0, fuzz_threshold: int = 75) -> List[Dict[str, any]]:
    """
    Retrieve the most relevant code entities from the Chroma vector database with fuzzy matching.

    Args:
        query (str): The query string to search for.
        db_path (str): Path to the Chroma database directory.
        top_k (int): Number of top results to return.
        distance_threshold (float): Maximum distance to consider a result relevant.
        fuzz_threshold (int): Minimum fuzzy match score to include a document.

    Returns:
        List[Dict[str, any]]: A list of the most relevant entities with metadata.
    """
    print(f"Connecting to Chroma database at: {db_path}")
    client = PersistentClient(path=db_path)

    collection_name = "code_entities"
    collection = client.get_collection(collection_name)

    all_documents = collection.query(
        query_texts=[""],
        n_results=100,
        include=["documents"]
    ).get("documents", [[]])[0]

    matched_documents = process.extract(
        query, all_documents, scorer=fuzz.partial_ratio, limit=top_k
    )

    best_matches = [doc for doc, score, _ in matched_documents if score >= fuzz_threshold]
    print(f"Fuzzy matches: {best_matches}")

    if not best_matches:
        print("No fuzzy matches found.")


    print(f"Searching for: {query}")
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["distances", "documents", "metadatas"]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved_entities = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        if doc in best_matches and dist <= distance_threshold:
            retrieved_entities.append({**meta, "document": doc, "distance": dist})

    print(f"{len(retrieved_entities)} results retrieved.")
    return retrieved_entities


def remove_embeddings(file_paths: List[str], db_path: str = "data/vector_db"):
    """
    Remove embeddings from the Chroma database associated with the given file paths.

    Args:
        file_paths (List[str]): A list of file paths whose embeddings should be removed.
        db_path (str): Path to the Chroma database directory.
    """
    print(f"Connecting to Chroma database at: {db_path}")
    client = PersistentClient(path=db_path)

    collection_name = "code_entities"
    collection = client.get_collection(collection_name)

    print(f"Removing embeddings for files: {file_paths}")
    for file_path in file_paths:
        try:
            collection.delete(where={"file": file_path})
            print(f"Successfully removed embeddings for: {file_path}")
        except Exception as e:
            print(f"Error removing embeddings for {file_path}: {e}")
