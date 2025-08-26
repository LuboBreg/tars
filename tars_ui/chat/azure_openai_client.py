import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load .env file
load_dotenv()

class ChatClient:
    def __init__(self, endpoint: str = None, key: str = None, deployment: str = None, api_version: str = None):
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.key = key or os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        
        if not (self.endpoint and self.key and self.deployment):
            raise RuntimeError("Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT")
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint, 
            api_key=self.key,  # Fixed: use string directly
            api_version=self.api_version
        )
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        # messages: list of {role, content}
        resp = self.client.chat.completions.create(  # Fixed: correct method name
            model=self.deployment,
            messages=messages,
            temperature=0.3
        )
        return resp.choices[0].message.content