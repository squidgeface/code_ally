import os
from groq import Groq

class TTTAIModel:

    def __init__(self):
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
        if not self.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        self._is_initialized = False  # Private initializer variable

    @property
    def is_initialized(self):
        """Getter function for the initializer variable."""
        return self._is_initialized

    @is_initialized.setter
    def is_initialized(self, value):
        """Setter function for the initializer variable."""
        self._is_initialized = value
    
    def initialize_model(self):
        try:
            self.client = Groq(api_key=self.GROQ_API_KEY)
            self.is_initialized = True
        except Exception as e:
            print(f"Error initializing Groq client: {e}")
            self.is_initialized = False

    def call_ai_model(self, context: str, prompt: str, temperature = 0.7) -> str:
        """
        Generic method to send a prompt to the AI model and return the response.
        """
        if not self.is_initialized:
            raise RuntimeError("The model is not initialized. Call 'initialize_model' first.")
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"{context}"},
                    {"role": "user", "content": f"{prompt}"},
                ],
                model="llama-3.3-70b-versatile",
                temperature=temperature,
            )
            return chat_completion.choices[0].message.content # type: ignore
        except Exception as e:
            print(f"Error calling AI model: {e}")
            return "An error occurred while processing your request."
        