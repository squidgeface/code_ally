import os
import hashlib
import json
from typing import Dict, List

from chromadb import PersistentClient

from ingestion import generate_embeddings, ingest_code, remove_embeddings, store_embeddings

STATE_FILE = "file_state.json"

def compute_file_hash(file_path: str) -> str:
    """
    Compute the SHA256 hash of a file's content.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: SHA256 hash of the file content.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def load_file_state() -> Dict[str, Dict[str, str]]:
    """
    Load the saved file state from the JSON file.

    Returns:
        Dict[str, Dict[str, str]]: A dictionary with file paths as keys and their hashes/timestamps as values.
    """
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_file_state(state: Dict[str, Dict[str, str]]) -> None:
    """
    Save the current file state to the JSON file.

    Args:
        state (Dict[str, Dict[str, str]]): The current state of files.
    """
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def detect_changes(files: List[str], excluded_files: List[str] = None) -> Dict[str, List[str]]:
    """
    Detect changes in the given files by comparing their current state with the saved state.

    Args:
        files (List[str]): List of file paths to check.
        excluded_files (List[str], optional): List of file names to exclude. Defaults to None.

    Returns:
        Dict[str, List[str]]: A dictionary with added, modified, removed, and moved files.
    """
    if excluded_files is None:
        excluded_files = []

    # Filter out excluded files
    files = [file for file in files if os.path.basename(file) not in excluded_files]

    current_state = load_file_state()
    new_state = {}
    added, modified, moved, removed = [], [], {}, []

    for file in files:
        if not os.path.exists(file):
            removed.append(file)
            continue
        file_hash = compute_file_hash(file)
        new_state[file] = {"hash": file_hash, "mtime": os.path.getmtime(file)}

        if file not in current_state:
            added.append(file)
        elif current_state[file]["hash"] != file_hash:
            modified.append(file)
            
    # Detect removed files
    for old_file in current_state:
        if old_file not in new_state:
            removed.append(old_file)

    # Detect moved files
    for old_file, old_data in current_state.items():
        if old_file in removed:
            for new_file, new_data in new_state.items():
                if old_data["hash"] == new_data["hash"]:
                    moved[old_file] = new_file
                    removed.remove(old_file)  # No longer considered removed
                    break

    # Save the new state
    save_file_state(new_state)

    return {"added": added, "modified": modified, "removed": removed, "moved": moved}


def handle_updates(updates: Dict[str, List[str]]):
    """
    Process updates in the repository, including re-embedding for a new database.

    Args:
        updates (dict): A dictionary containing lists of added, modified, removed, and moved files.
    """
    client = PersistentClient(path="data/vector_db")
    collection_name = "code_entities"

    # Check if collection exists
    try:
        collection = client.get_collection(collection_name)
        print(f"Using existing collection: {collection_name}")
    except Exception:
        print(f"Collection {collection_name} does not exist. Treating all files as new.")
        updates["added"] = updates["added"] + updates["modified"]
        updates["modified"] = []

    # Handle added and modified files
    changed_files = updates["added"] + updates["modified"]
    if changed_files:
        print(f"Processing added/modified files: {changed_files}")

        # Normalize changed files
        normalized_changed_files = [os.path.normpath(file) for file in changed_files]
        print(f"Normalized changed files: {normalized_changed_files}")

        # Ingest code
        entities = ingest_code(".")

        # Filter entities by changed files
        entities = [
            entity for entity in entities
            if os.path.normpath(entity["file"]) in normalized_changed_files
        ]

        if entities:
            print(f"Filtered entities: {len(entities)}")
            entities_with_embeddings = generate_embeddings(entities)
            store_embeddings(entities_with_embeddings)
        else:
            print("No valid entities found in changed files.")
    else:
        print("No changes detected.")


def validate_embedding_metadata(metadata: List[Dict[str, any]], valid_files: List[str]) -> List[Dict[str, any]]:
    """
    Validate and clean up embedding metadata by ensuring all file references are valid.

    Args:
        metadata (List[Dict[str, any]]): List of embedding metadata to validate.
        valid_files (List[str]): List of valid file paths currently in the repository.

    Returns:
        List[Dict[str, any]]: Cleaned metadata with only valid file references.
    """
    cleaned_metadata = []
    for entry in metadata:
        file_path = entry.get("file")
        if file_path in valid_files:
            cleaned_metadata.append(entry)
        else:
            print(f"Invalid metadata entry removed: {entry}")
    return cleaned_metadata

def update_moved_function_metadata(
    metadata: List[Dict[str, any]], moved_files: Dict[str, str], db_path: str = "data/vector_db"
) -> None:
    """
    Update embedding metadata for functions moved to new files and save to the database.

    Args:
        metadata (List[Dict[str, any]]): List of embedding metadata to update.
        moved_files (Dict[str, str]): Mapping of old file paths to new file paths.
        db_path (str): Path to the Chroma database directory.
    """
    print("Updating metadata for moved functions...")
    client = PersistentClient(path=db_path)
    collection = client.get_collection("code_entities")

    updated_entries = []
    for entry in metadata:
        file_path = entry.get("file")
        if file_path in moved_files:
            new_file_path = moved_files[file_path]
            entry["file"] = new_file_path
            print(f"Updated file reference for: {entry['name']} -> {new_file_path}")
            updated_entries.append(entry)

    # Save updated metadata to the database
    if updated_entries:
        for entry in updated_entries:
            try:
                collection.update(
                    documents=[f"{entry['type']}: {entry['name']}"],
                    metadatas=[{
                        "file": entry["file"],
                        "line": entry["line"],
                        "docstring": entry["docstring"]
                    }],
                    ids=[entry["name"]]
                )
                print(f"Updated database entry for: {entry['name']}")
            except Exception as e:
                print(f"Error updating metadata for {entry['name']}: {e}")
