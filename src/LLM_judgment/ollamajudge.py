from typing import Optional
from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_community.llms import Ollama

class LocalOllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="llama3.1:8b"):
        self.model_name = model_name
        # Low temperature keeps the scoring metric assignments stable
        self.ollama = Ollama(model=model_name, temperature=0.1)

    def load_model(self):
        return self.ollama

    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> str:
        """Standard generation handling JSON schemas via direct string passing."""
        if schema:
            # Tell standard models exactly how to format the keys in plain text
            schema_fields = ", ".join([f'"{k}": "your_{k}_here"' for k in schema.model_fields.keys()])
            text_injection = (
                f"\n\nReturn your output ONLY as a valid JSON object matching this schema blueprint:\n"
                f"{{\n  {schema_fields}\n}}\n"
                f"Do not include conversational text or code blocks. Output pure JSON format strings."
            )
            prompt = f"{prompt}{text_injection}"
            
        return self.ollama.invoke(prompt).strip()

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> str:
        """Asynchronous execution path handled directly by DeepEval execution loops."""
        if schema:
            schema_fields = ", ".join([f'"{k}": "your_{k}_here"' for k in schema.model_fields.keys()])
            text_injection = (
                f"\n\nReturn your output ONLY as a valid JSON object matching this schema blueprint:\n"
                f"{{\n  {schema_fields}\n}}\n"
                f"Do not include conversational text or code blocks. Output pure JSON format strings."
            )
            prompt = f"{prompt}{text_injection}"
            
        return self.ollama.invoke(prompt).strip()

    def get_model_name(self) -> str:
        return self.model_name