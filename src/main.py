"""
main.py

The main entry point for the Code Ally application.
"""
import os

from chromadb import PersistentClient
from file_state_tracker import detect_changes, handle_updates
from ingestion import retrieve_entities
from query_utils import process_user_query
from groq_integration import TTTAIModel

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
    
    ttt_ai = TTTAIModel()
    ttt_ai.initialize_model()
    
    while True:
        query = input("\nEnter a query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        context = process_user_query(query)
        if context:
            output = ttt_ai.call_ai_model(context=context, prompt=query)
            print(output)

if __name__ == "__main__":
    main()
