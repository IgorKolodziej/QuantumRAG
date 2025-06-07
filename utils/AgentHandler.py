import os
from typing import List, Optional, Dict, Any, Tuple
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()
class AgentHandler:
    """
    Handles loading and inference for multiple LLM APIs using huggingface_hub.InferenceClient.
    Uses a single Hugging Face API token for all supported endpoints.
    """

    # Updated model dictionary with (provider, model_id) format
    SUPPORTED_MODELS = {
        # Format: "model_key": (provider, model_id)
        # All these models can be accessed with a single HF API token
        "llama-3-8b": ("hf-inference", "meta-llama/Llama-3.3-70B-Instruct"),
        "mixtral-8x7b": ("hf-inference", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
        # "mistral-7b": ("hf-inference", "mistralai/Mistral-7B-Instruct-v0.2"),
        # "phi-3": ("hf-inference", "microsoft/phi-3-medium-4k-instruct"),
        "phi-3.5": ("hf-inference", "microsoft/Phi-3.5-mini-instruct"),
    }

    def __init__(
        self,
        model_keys: Optional[List[str]] = None,
        api_key: Optional[str] = None,
    ):
        """
        Args:
            model_keys: List of model keys to use (from SUPPORTED_MODELS). If None, use all.
            api_key: Hugging Face API token. If None, will use HF_API_KEY env variable.
        """
        if model_keys is None:
            model_keys = list(self.SUPPORTED_MODELS.keys())
        self.model_keys = model_keys

        # Set up API key, using environment variable as fallback
        self.api_key = api_key or os.getenv("HF_API_KEY")
        if not self.api_key:
            raise ValueError("Hugging Face API token not found. Set HF_API_KEY env variable or pass api_key.")

        # Create a single client for the Hugging Face endpoint
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=self.api_key
        )

    def _format_prompt(self, question: str, context: Optional[List[Any]]) -> str:
        """Format the prompt with or without context"""
        if context:
            # Handle context properly whether it's a dict with 'document' key or plain text
            context_text = "\n\n".join(
                str(c["document"]) if isinstance(c, dict) and "document" in c else str(c)
                for c in context
            )
            return (
                f"Answer the question based only on the following context.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {question}\n\nAnswer:"
            )
        else:
            return (
                f"Answer the following question to the best of your ability.\n\n"
                f"Question: {question}\n\nAnswer:"
            )

    def generate(
        self,
        question: str,
        context: Optional[List[Any]] = None,
        model_key: str = "llama-3-8b",
    ) -> str:
        """
        Generate an answer using the specified API model, with or without context.
        
        Args:
            question: The question to answer
            context: Optional list of context passages (can be dicts with 'document' key or strings)
            model_key: Which model to use (must be a key in SUPPORTED_MODELS)
            
        Returns:
            The generated answer as a string
        """
        if model_key not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_key}' not supported. Available models: {list(self.SUPPORTED_MODELS.keys())}")
            
        _, model_id = self.SUPPORTED_MODELS[model_key]
        prompt = self._format_prompt(question, context)
        
        try:
            completion = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.0,
            )
            
            # Extract the response text
            if hasattr(completion.choices[0], "message"):
                return completion.choices[0].message.content
            else:
                return str(completion.choices[0])
                
        except Exception as e:
            print(f"Error generating response with {model_key} ({model_id}): {str(e)}")
            return f"Error: Failed to generate response with {model_key}. {str(e)}"

    def compare_all(self, question: str, context: List[Any]) -> Dict[str, Dict[str, str]]:
        """
        Generate answers with and without context for all selected API models.
        
        Args:
            question: The question to answer
            context: List of context passages
            
        Returns:
            Dictionary with results for each model:
            {
                model_key: {
                    "with_context": "...",
                    "without_context": "..."
                },
                ...
            }
        """
        results = {}
        for key in self.model_keys:
            if key not in self.SUPPORTED_MODELS:
                print(f"Skipping unsupported model: {key}")
                continue
                
            print(f"Generating answers with model: {key}")
            try:
                with_ctx = self.generate(question, context, model_key=key)
                without_ctx = self.generate(question, None, model_key=key)
                results[key] = {
                    "with_context": with_ctx,
                    "without_context": without_ctx
                }
            except Exception as e:
                print(f"Error comparing results for {key}: {str(e)}")
                results[key] = {
                    "with_context": f"Error: {str(e)}",
                    "without_context": f"Error: {str(e)}"
                }
        return results