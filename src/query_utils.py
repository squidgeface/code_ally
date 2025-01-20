
from typing import Dict, List
from ingestion import retrieve_entities
from typing import Dict
import spacy
from keybert import KeyBERT

kw_model = KeyBERT()

def extract_keywords_with_keybert(prompt: str) -> List[str]:
    """
    Extract keywords from a user prompt using KeyBERT.

    Args:
        prompt (str): The user's question.

    Returns:
        List[str]: A list of ranked keywords.
    """
    keywords = [kw[0] for kw in kw_model.extract_keywords(prompt, top_n=5)]
    unique_strings = list(set(keywords))
    return " ".join(unique_strings)

nlp = spacy.load("en_core_web_sm")

def parse_user_prompt(prompt: str) -> List[str]:
    """
    Parse the user prompt to extract intent and key phrases, prioritizing relevant terms.

    Args:
        prompt (str): The user's question.

    Returns:
        Dict[str, List[str]]: Parsed information including intent and cleaned key phrases.
    """
    doc = nlp(prompt)

    # Extract relevant tokens (nouns, proper nouns, and adjectives)
    keywords = [token.text.lower() for token in doc if token.pos_ in {"NOUN", "PROPN", "ADJ"}]

    # Combine noun chunks for broader context
    noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks]

    # Merge keywords and noun chunks, remove duplicates, and prioritize relevance
    key_phrases = list(set(keywords + noun_chunks))

    # Filter out redundant phrases
    cleaned_key_phrases = [phrase for phrase in key_phrases if phrase not in {"me", "the", "to"}]
    unique_strings = list(set(cleaned_key_phrases))
    return " ".join(unique_strings)

def construct_llm_context(embeddings: List[Dict[str, any]]) -> str:
    """
    Construct a prompt for the LLM using the user query and embeddings.

    Args:
        user_prompt (str): The original user question.
        embeddings (List[Dict[str, any]]): Retrieved embeddings and their metadata.

    Returns:
        str: A formatted prompt for the LLM.
    """
    context = "\n\n".join(
        f"File: {embedding['file']} (Line {embedding['line']})\n"
        f"Entity: {embedding['document']}\n"
        f"Docstring: {embedding['docstring']}"
        for embedding in embeddings
    )

    return f"You are a helpful Coding assistant. Answer questuons based on this code: \n{context}."

def process_user_query(user_prompt: str) -> str:
    """
    Process the user query to retrieve relevant embeddings and construct an LLM prompt.

    Args:
        user_prompt (str): The user's question.

    Returns:
        str: The constructed prompt for the LLM.
    """
    # Step 1: Parse the user prompt
    parsed_prompt = parse_user_prompt(user_prompt)

    # Step 2: Retrieve relevant embeddings
    embeddings = retrieve_entities(query=parsed_prompt, fuzz_threshold=50)

    if not embeddings:
        return None
    # Step 3: Construct the LLM prompt
    llm_prompt = construct_llm_context(embeddings)

    return llm_prompt

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
