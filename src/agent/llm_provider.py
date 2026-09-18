import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List

# Local LLM default (M1 Pro 16GB, personal, offline-safe): qwen2.5:3b via
# Ollama at http://localhost:11434. Cloud (Groq) is optional fallback only.
# Retrieval stays BM25 keyword — no heavy embeddings added (YAGNI).
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_LOCAL_MODEL = "qwen2.5:3b"
MAX_PROMPT_WORDS = 150  # keep local synthesis prompt short for 3b models

class RCALLMProvider(ABC):
    @abstractmethod
    def synthesize_rca(self, system_prompt: str, user_content: str) -> str:
        pass

class CloudGroqProvider(RCALLMProvider):
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name

    def synthesize_rca(self, system_prompt: str, user_content: str) -> str:
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage
        
        llm = ChatGroq(
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1,
        )
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        return response.content

class LocalOllamaProvider(RCALLMProvider):
    """Local Ollama synthesis: qwen2.5:3b, temp 0.1, num_predict 256, timeout 30s."""

    def __init__(self, model_name: str = "", base_url: str = ""):
        self.model_name = (
            (model_name or os.getenv("LOCAL_MODEL", "") or DEFAULT_LOCAL_MODEL).strip()
            or DEFAULT_LOCAL_MODEL
        )
        self.base_url = (
            (base_url or os.getenv("OLLAMA_BASE_URL", "") or OLLAMA_BASE_URL).strip()
            or OLLAMA_BASE_URL
        )
        self.temperature = 0.1
        self.num_predict = 256
        self.timeout = 30

    def _cap_words(self, text: str, limit: int = MAX_PROMPT_WORDS) -> str:
        """Cap prompt to <=150 words so small local models stay grounded."""
        words = (text or "").split()
        if len(words) <= limit:
            return text
        return " ".join(words[:limit])

    def synthesize_rca(self, system_prompt: str, user_content: str) -> str:
        from langchain_community.chat_models import ChatOllama
        from langchain_core.messages import SystemMessage, HumanMessage

        # System prompt already requires [SOP-SMT-xxx] citations; cap only the
        # variable user content so the citation instruction is never truncated.
        capped_content = self._cap_words(user_content)
        kwargs = dict(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,  # 0.1 deterministic
            num_predict=self.num_predict,  # 256 tokens max
            timeout=self.timeout,  # 30s
        )
        try:
            llm = ChatOllama(**kwargs)
        except TypeError:
            # Older langchain_community without timeout/num_predict kwargs.
            kwargs.pop("timeout", None)
            kwargs.pop("num_predict", None)
            llm = ChatOllama(**kwargs)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=capped_content),
        ])
        return response.content

class DeterministicProvider(RCALLMProvider):
    def synthesize_rca(self, system_prompt: str, user_content: str) -> str:
        raise NotImplementedError("Deterministic fallback is handled directly by the RCA Agent logic")

