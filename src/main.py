"""
main.py

The main entry point for the Code Ally application.
"""
import os

from chromadb import PersistentClient
from file_state_tracker import detect_changes, handle_updates
from ingestion import retrieve_entities
from query_utils import display_results, display_suggestions

def main():
    """
    Main function to run the Code Ally pipeline.
    """
    # List of Python files to track
    tracked_files = [
        os.path.join(root, file)
        for root, _, files in os.walk("src/")
        for file in files if file.endswith(".py")
    ]

    excluded_files = ["__init__.py"]
    updates = detect_changes(tracked_files, excluded_files=excluded_files)
    print(f"Updates detected: {updates}")

    # Handle updates
    handle_updates(updates)

    while True:
        query = input("\nEnter a query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        results = retrieve_entities(query)
        if results:
            display_results(results)
        else:
            display_suggestions(query)


if __name__ == "__main__":
    main()
