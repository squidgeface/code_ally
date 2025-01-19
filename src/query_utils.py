
from typing import Dict, List
from ingestion import retrieve_entities
from pygments.lexers import guess_lexer_for_filename
from pygments.util import ClassNotFound


def display_results(results: List[Dict[str, any]]):
    """
    Display the retrieved results to the user.

    Args:
        results (List[Dict[str, any]]): List of retrieved entities with metadata.
    """
    for i, result in enumerate(results, 1):
        docstring_snippet = result['docstring'].splitlines()[:3]  # Show first 3 lines
        relevance = 100 - result['distance'] * 100  # Convert distance to relevance
        print(f"\nResult {i} (Relevance: {relevance:.1f}%):")
        print(f"Document: {result['document']}")
        print(f"File: {result['file']} (Line {result['line']})")
        print(f"Docstring (Snippet):\n{' '.join(docstring_snippet)}\n")

def display_suggestions(query: str):
    """
    Display suggestions for the given query if no exact results are found.

    Args:
        query (str): The user query.
    """
    print("No results found for your query.")
    print("Suggestions:")
    all_suggestions = retrieve_entities(query, fuzz_threshold=50)  # Lower threshold for suggestions
    if all_suggestions:
        for suggestion in all_suggestions:
            print(f"- {suggestion['document']}")
    else:
        print("No suggestions available.")
