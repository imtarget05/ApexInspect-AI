import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List

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
    def __init__(self, model_name: str):
        self.model_name = model_name or "llama3.2"

    def synthesize_rca(self, system_prompt: str, user_content: str) -> str:
        from langchain_community.chat_models import ChatOllama
        from langchain_core.messages import SystemMessage, HumanMessage
        
        llm = ChatOllama(
            model=self.model_name,
            temperature=0.1,
        )
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        return response.content

class DeterministicProvider(RCALLMProvider):
    def synthesize_rca(self, system_prompt: str, user_content: str) -> str:
        raise NotImplementedError("Deterministic fallback is handled directly by the RCA Agent logic")

